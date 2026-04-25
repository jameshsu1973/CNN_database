"""
Zilliz Cloud (Milvus) based Vector Database for CNN Weight Storage and Retrieval
Cloud-based alternative to ChromaDB for multi-user collaboration
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import uuid
import json
import re
from collections import defaultdict

# Suppress Milvus RPC error logs
os.environ['GRPC_VERBOSITY'] = 'ERROR'
logging.getLogger('pymilvus').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)

try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("Warning: pymilvus not installed. Install with: pip install pymilvus")

from .config import get_config


class MilvusWeightVault:
    """
    Vector database for storing and retrieving CNN weights using Zilliz Cloud (Milvus).
    Uses cosine similarity for matching and supports cloud-based storage.
    
    This enables multiple users to share the same vault without local storage.
    """

    def __init__(self,
                 uri: Optional[str] = None,
                 token: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 top_k_ratio: Optional[float] = None):
        """
        Initialize the Zilliz/Milvus Weight Vault.
        
        Args:
            uri: Zilliz Cloud cluster URI (e.g., https://xxx.zillizcloud.com:19530)
            token: Zilliz Cloud API key or username:password
            collection_name: Base name for collections
            similarity_threshold: Minimum cosine similarity for matching
            top_k_ratio: Ratio of top weights to keep (for masking)
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed. "
                "Install with: pip install pymilvus"
            )

        # Load config
        config = get_config()

        # Get Milvus config from environment or config file
        self.uri = uri or config.get('vector_db.milvus.uri') or config.get('vector_db.milvus.endpoint')
        self.token = token or config.get('vector_db.milvus.token') or config.get('vector_db.milvus.api_key')
        
        if not self.uri or not self.token:
            raise ValueError(
                "Zilliz Cloud URI and token are required. "
                "Set them via parameters or config (vector_db.milvus.uri, vector_db.milvus.token)"
            )

        self.collection_name = collection_name or config.get('vector_db.milvus.collection_name', 'cnn_weights')
        self.similarity_threshold = similarity_threshold or config.similarity_threshold
        self.top_k_ratio = top_k_ratio or config.top_k_ratio

        # Initialize Milvus client
        import sys
        print(f"[Milvus] Connecting to {self.uri}...", flush=True)
        self.client = MilvusClient(
            uri=self.uri,
            token=self.token,
            timeout=30  # 30 seconds timeout
        )
        print(f"[Milvus] Connected successfully!", flush=True)

        # Collections cache (name -> dimension mapping)
        self.collection_dims: Dict[str, int] = {}

        # Statistics
        self.total_entries = 0
        self.object_categories = set()

    def _flatten_kernel(self, weight_tensor: torch.Tensor) -> np.ndarray:
        """Flatten a weight tensor to 1D vector."""
        return weight_tensor.detach().cpu().numpy().flatten().astype(np.float32)

    def _get_layer_key(self, layer: nn.Module, layer_name: str = "") -> str:
        """Generate unique key based on layer dimensions."""
        if isinstance(layer, nn.Conv2d):
            key = f"conv_{layer.in_channels}_{layer.out_channels}_{layer.kernel_size[0]}_{layer.kernel_size[1]}"
        elif isinstance(layer, nn.Linear):
            key = f"linear_{layer.in_features}_{layer.out_features}"
        else:
            key = f"other_{type(layer).__name__}"

        # Detection-specific prefixes
        if layer_name:
            if "head" in layer_name or "bbox" in layer_name or "class" in layer_name:
                key = f"det_{key}"
            elif "backbone" in layer_name:
                key = f"backbone_{key}"

        return key

    def _get_collection_name(self, layer_key: str) -> str:
        """Get collection name for a specific layer key.
        
        Each unique layer_key (e.g., 'conv_3_32_3_3') gets its own collection
        to ensure all vectors in the collection have the same dimension.
        """
        # Sanitize layer_key for collection name
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', layer_key)[:50]
        return f"{self.collection_name}_{safe_key}"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _generate_topology_query(self, shape: tuple) -> np.ndarray:
        """Generate a deterministic query vector based on layer topology."""
        np.random.seed(42)
        query = np.random.randn(*shape).astype(np.float32)
        query = query / np.linalg.norm(query)
        return query.flatten()

    def _apply_top_k_mask(self, weight_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-k masking to weight tensor."""
        flat_weights = weight_tensor.abs().flatten()
        k = int(self.top_k_ratio * flat_weights.numel())
        k = max(k, 1)  # At least keep 1 weight

        top_k_values, top_k_indices = torch.topk(flat_weights, k)
        mask = torch.zeros_like(flat_weights, dtype=torch.bool)
        mask[top_k_indices] = True

        masked_flat = weight_tensor.flatten() * mask.float()
        return masked_flat.view_as(weight_tensor), mask.view_as(weight_tensor)

    def _compute_fingerprint(self, weight_tensor: torch.Tensor, target_dim: int = 16384) -> np.ndarray:
        """
        Compute a fixed-dimension fingerprint vector from weight tensor.
        Uses memory-efficient vectorized Count Sketch.
        """
        flat = weight_tensor.cpu().float().numpy().flatten()
        original_dim = len(flat)
        
        if original_dim <= target_dim:
            result = np.zeros(target_dim, dtype=np.float32)
            result[:original_dim] = flat
            return result
        
        # Vectorized Count Sketch (fast!)
        np.random.seed(42)
        n_positions = 50  # Few positions per output for speed
        
        # Generate random indices and signs
        indices = np.random.randint(0, original_dim, size=(target_dim, n_positions))
        signs = np.random.choice([-1, 1], size=(target_dim, n_positions))
        
        # Vectorized dot product
        fingerprint = np.sum(signs * flat[indices], axis=1) / np.sqrt(n_positions)
        
        return fingerprint

    def _compute_fingerprint_with_seed(self, weight_tensor: torch.Tensor, target_dim: int, seed: int) -> np.ndarray:
        """Compute fingerprint with specific seed for consistent retrieval."""
        flat = weight_tensor.cpu().float().numpy().flatten()
        original_dim = len(flat)
        
        if original_dim <= target_dim:
            result = np.zeros(target_dim, dtype=np.float32)
            result[:original_dim] = flat
            return result
        
        # Vectorized Count Sketch
        np.random.seed(seed)
        n_positions = 50
        
        indices = np.random.randint(0, original_dim, size=(target_dim, n_positions))
        signs = np.random.choice([-1, 1], size=(target_dim, n_positions))
        
        fingerprint = np.sum(signs * flat[indices], axis=1) / np.sqrt(n_positions)
        
        return fingerprint

    def _ensure_collection_exists(self, collection_name: str, dimension: int):
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists by trying to describe it
            try:
                self.client.describe_collection(collection_name=collection_name)
                # Collection exists, just update dimension cache
                self.collection_dims[collection_name] = dimension
                return
            except Exception:
                # Collection doesn't exist, create it
                pass
            
            # Create collection with schema
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                primary_field_name="id",
                vector_field_name="vector",
                id_type="int",
                metric_type="COSINE",
                auto_id=True
            )
            print(f"[MilvusVault] Created collection: {collection_name} (dim={dimension})")
            
            self.collection_dims[collection_name] = dimension
        except Exception as e:
            # Check if it's a "already exists" error (race condition)
            if "already exists" in str(e).lower():
                pass  # Already created by another process
            else:
                print(f"[MilvusVault] Warning creating collection: {e}")

    def store_weights(self,
                     layer: nn.Module,
                     layer_name: str,
                     model_name: str,
                     epoch: int,
                     accuracy: Optional[float] = None,
                     object_category: Optional[str] = None,
                     num_objects: int = 0):
        """
        Store weights from a trained layer into the vector database.

        Args:
            layer: The nn.Module (Conv2d or Linear) to store
            layer_name: Name of the layer in the model
            model_name: Identifier for the model
            epoch: Training epoch
            accuracy: Model accuracy (optional metadata)
            object_category: Type of object being detected
            num_objects: Number of objects in training images
        """
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return

        layer_key = self._get_layer_key(layer, layer_name)
        collection_name = self._get_collection_name(layer_key)

        # Apply top-k masking
        weight_tensor = layer.weight.data
        masked_weights, mask = self._apply_top_k_mask(weight_tensor)

        # Flatten the masked weights
        flattened = self._flatten_kernel(masked_weights)
        dimension = len(flattened)
        
        # Fingerprint compression for large layers (exceeds Zilliz limit)
        MAX_DIMENSION = 32768  # Zilliz Cloud free tier limit
        FINGERPRINT_DIM = 16384  # Use larger fingerprint for less information loss
        use_fingerprint = dimension > MAX_DIMENSION
        
        if use_fingerprint:
            # Compress to fingerprint
            flattened = self._compute_fingerprint(masked_weights, target_dim=FINGERPRINT_DIM)
            dimension = FINGERPRINT_DIM
            print(f"[MilvusVault] Using fingerprint for {layer_key}: {dimension}")

        # Ensure collection exists
        self._ensure_collection_exists(collection_name, dimension)

        # Prepare metadata
        metadata = {
            'layer_name': layer_name,
            'model_name': model_name,
            'epoch': epoch,
            'accuracy': float(accuracy) if accuracy is not None else 0.0,
            'shape': json.dumps(list(weight_tensor.shape)),
            'layer_key': layer_key,
            'object_category': object_category or '',
            'num_objects': num_objects,
            'use_fingerprint': str(use_fingerprint),  # Store as string for Milvus
            'original_dimension': dimension if use_fingerprint else 0
        }
        # Note: Not storing 'mask' as JSON - it exceeds Zilliz Cloud's 64KB limit for dynamic fields

        # Add to Milvus
        try:
            self.client.insert(
                collection_name=collection_name,
                data=[{
                    'vector': flattened.tolist(),
                    **metadata
                }]
            )
            self.total_entries += 1
            
            # Track categories
            if object_category:
                self.object_categories.add(object_category)
                
        except Exception as e:
            print(f"[MilvusVault] Error storing weights for {layer_key}: {e}")
            # Don't re-raise - continue storing other layers

    def query_similar_weights(self,
                              layer: nn.Module,
                              layer_name: str = "",
                              object_category: Optional[str] = None,
                              k: int = 3) -> Optional[List[Dict]]:
        """
        Query the vault for similar weights to initialize a layer.

        Args:
            layer: The layer to find initialization for
            layer_name: Name of the layer
            object_category: Preferred object category
            k: Number of candidates to retrieve

        Returns:
            List of candidate entries with similarity scores, or None if no match
        """
        layer_key = self._get_layer_key(layer, layer_name)
        collection_name = self._get_collection_name(layer_key)

        # Check if collection exists
        try:
            collection_info = self.client.describe_collection(collection_name=collection_name)
            collection_dim = collection_info.get('dimension', 0)
        except Exception:
            return None  # Collection doesn't exist

        # Determine if this is a fingerprint collection based on dimension
        # Fingerprint collections have dimension == FINGERPRINT_DIM (16384)
        # Normal collections have dimension matching actual layer dimension
        FINGERPRINT_DIM = 16384
        
        # Generate query vector
        if isinstance(layer, nn.Conv2d):
            query_shape = (layer.out_channels, layer.in_channels,
                          layer.kernel_size[0], layer.kernel_size[1])
        else:
            query_shape = (layer.out_features, layer.in_features)
        
        # Check if this is a fingerprint layer (dimension exceeds MAX_DIMENSION)
        actual_dim = query_shape[0] * query_shape[1] * query_shape[2] * query_shape[3]
        MAX_DIMENSION = 32768
        
        # If collection dim matches FINGERPRINT_DIM, use fingerprint query
        # OR if actual dimension exceeds limit, use fingerprint
        use_fingerprint = (collection_dim == FINGERPRINT_DIM) or (actual_dim > MAX_DIMENSION)
        
        if use_fingerprint:
            # Generate a dummy weight tensor with the right shape for fingerprint
            dummy_tensor = torch.randn(query_shape)
            query_vector = self._compute_fingerprint(dummy_tensor, target_dim=FINGERPRINT_DIM)
        else:
            query_vector = self._generate_topology_query(query_shape)

        # Query Milvus
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector.tolist()],
                limit=min(k, 100),
                output_fields=["id", "vector", "layer_name", "model_name", "epoch", 
                              "accuracy", "shape", "layer_key", "object_category", 
                              "num_objects", "use_fingerprint", "original_dimension"],
                search_params={"metric_type": "COSINE"}
            )
        except Exception as e:
            print(f"[MilvusVault] Query error: {e}")
            return None

        # Process results
        candidates = []
        if results and len(results) > 0 and len(results[0]) > 0:
            for hit in results[0]:
                distance = hit.get('distance', 0)
                # Milvus returns distance, convert to similarity
                # COSINE distance = 1 - similarity
                similarity = 1.0 - distance

                # Boost similarity if object category matches
                entry_category = hit.get('object_category', '')
                if object_category and entry_category == object_category:
                    similarity *= 1.2  # 20% boost

                candidates.append({
                    'entry': {
                        'vector': np.array(hit.get('vector', []), dtype=np.float32),
                        'metadata': hit
                    },
                    'distance': distance,
                    'similarity': similarity
                })

        # Filter by threshold
        candidates = [c for c in candidates
                       if c['similarity'] >= self.similarity_threshold]

        return candidates if candidates else None

    def get_initialization_weights(self,
                                   layer: nn.Module,
                                   layer_name: str = "",
                                   object_category: Optional[str] = None,
                                   force: bool = False) -> Optional[torch.Tensor]:
        """
        Get initialization weights for a layer from the vault.

        Args:
            layer: The layer to initialize
            layer_name: Name of the layer
            object_category: Preferred object category
            force: If True, ignore similarity threshold

        Returns:
            Weight tensor or None
        """
        if force:
            return self._get_latest_weights(layer, layer_name, object_category)

        candidates = self.query_similar_weights(layer, layer_name, object_category, k=1)

        if not candidates:
            return None

        best_match = candidates[0]['entry']
        vector = best_match['vector']
        metadata = best_match['metadata']
        
        # Check if this is a fingerprint (compressed) entry
        use_fingerprint = metadata.get('use_fingerprint', 'False') == 'True'
        if use_fingerprint:
            # Fingerprint cannot reconstruct exact weights
            # Return a reshaped version as initialization hint
            print(f"[MilvusVault] Using fingerprint hint for {layer_name}")
            try:
                # Reshape fingerprint to layer's expected shape (approximation)
                if isinstance(layer, nn.Conv2d):
                    shape = (layer.out_channels, layer.in_channels,
                            layer.kernel_size[0], layer.kernel_size[1])
                else:
                    shape = (layer.out_features, layer.in_features)
                
                # Reshape fingerprint to match layer dimensions
                # Use the fingerprint values as a rough initialization
                flat = np.array(vector, dtype=np.float32)
                # Tile or truncate to match total size
                total_params = np.prod(shape)
                if len(flat) < total_params:
                    # Tile the fingerprint
                    repeats = (total_params // len(flat)) + 1
                    flat = np.tile(flat, repeats)[:total_params]
                else:
                    flat = flat[:total_params]
                
                weights = torch.from_numpy(flat.reshape(shape))
                return weights
            except Exception as e:
                print(f"[MilvusVault] Fingerprint reconstruction failed: {e}")
                return None

        # Reconstruct shape for normal (non-fingerprint) weights
        try:
            shape = tuple(json.loads(metadata.get('shape', '[]')))
            if not shape:
                # Fallback: calculate shape from vector and layer
                if isinstance(layer, nn.Conv2d):
                    shape = (layer.out_channels, layer.in_channels, 
                            layer.kernel_size[0], layer.kernel_size[1])
                else:
                    shape = (layer.out_features, layer.in_features)
            
            weights = torch.from_numpy(vector).reshape(shape)
            return weights
        except Exception as e:
            print(f"[MilvusVault] Error reconstructing weights: {e}")
            return None

    def _get_latest_weights(self,
                           layer: nn.Module,
                           layer_name: str = "",
                           object_category: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get the latest (most recent) weights for a layer."""
        layer_key = self._get_layer_key(layer, layer_name)
        collection_name = self._get_collection_name(layer_key)

        try:
            collections = self.client.collections
            if collection_name not in collections:
                return None
        except Exception:
            return None

        # Query for latest entry (sorted by epoch descending)
        try:
            results = self.client.query(
                collection_name=collection_name,
                filter=f"object_category == '{object_category}'" if object_category else "",
                output_fields=["vector", "shape", "epoch", "model_name", "object_category"],
                limit=1
            )
            
            if not results or len(results) == 0:
                # Try without filter
                results = self.client.query(
                    collection_name=collection_name,
                    output_fields=["vector", "shape", "epoch", "model_name", "object_category"],
                    limit=1
                )

            if not results or len(results) == 0:
                return None

            best_entry = results[0]
            vector = np.array(best_entry.get('vector', []), dtype=np.float32)
            metadata = best_entry
            
            # Check if this is a fingerprint entry
            use_fingerprint = metadata.get('use_fingerprint', 'False') == 'True'
            
            if use_fingerprint:
                # Cannot reconstruct exact weights from fingerprint
                # Return reshaped fingerprint as initialization hint
                try:
                    if isinstance(layer, nn.Conv2d):
                        shape = (layer.out_channels, layer.in_channels,
                                layer.kernel_size[0], layer.kernel_size[1])
                    else:
                        shape = (layer.out_features, layer.in_features)
                    
                    total_params = np.prod(shape)
                    flat = vector
                    if len(flat) < total_params:
                        repeats = (total_params // len(flat)) + 1
                        flat = np.tile(flat, repeats)[:total_params]
                    else:
                        flat = flat[:total_params]
                    
                    weights = torch.from_numpy(flat.reshape(shape))
                    return weights
                except Exception as e:
                    print(f"[MilvusVault] Fingerprint reconstruction failed: {e}")
                    return None
            
            shape = tuple(json.loads(metadata.get('shape', '[]')))

            if not shape:
                if isinstance(layer, nn.Conv2d):
                    shape = (layer.out_channels, layer.in_channels,
                            layer.kernel_size[0], layer.kernel_size[1])
                else:
                    shape = (layer.out_features, layer.in_features)

            weights = torch.from_numpy(vector).reshape(shape)
            return weights
        except Exception as e:
            print(f"[MilvusVault] Error getting latest weights: {e}")
            return None

    def has_weights_for_layer(self, layer: nn.Module, layer_name: str = "") -> bool:
        """Check if vault has weights for a specific layer."""
        layer_key = self._get_layer_key(layer, layer_name)
        collection_name = self._get_collection_name(layer_key)

        try:
            self.client.describe_collection(collection_name)
            return True
        except Exception:
            return False

    def save_vault(self):
        """Persist the vault (Milvus auto-persists to cloud)."""
        print(f"Zilliz Cloud vault synchronized")
        print(f"Total entries: {self.total_entries}")
        print(f"Object categories: {self.object_categories}")

    def load_vault(self) -> bool:
        """Load the vault (Milvus loads from cloud automatically)."""
        try:
            # List collections and get stats
            all_collections = []
            try:
                all_collections = self.client.collections
            except Exception:
                pass
            
            total = 0
            for coll in all_collections:
                if coll.startswith(self.collection_name):
                    try:
                        stats = self.client.get_collection_stats(coll)
                        count = stats.get('row_count', 0)
                        total += count
                    except Exception:
                        pass
            
            print(f"Zilliz Cloud vault loaded from {self.uri}")
            print(f"Total entries: {total}")
            return total > 0
        except Exception as e:
            print(f"[MilvusVault] Error loading vault: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get vault statistics."""
        try:
            all_collections = []
            try:
                all_collections = self.client.collections
            except Exception:
                pass
            
            entries_per_collection = {}
            total = 0
            
            for coll in all_collections:
                if coll.startswith(self.collection_name):
                    try:
                        stats = self.client.get_collection_stats(coll)
                        count = stats.get('row_count', 0)
                        entries_per_collection[coll] = count
                        total += count
                    except Exception:
                        entries_per_collection[coll] = 0

            stats = {
                'total_entries': total,
                'collections': len(entries_per_collection),
                'object_categories': list(self.object_categories),
                'entries_per_collection': entries_per_collection,
                'uri': self.uri
            }
            return stats
        except Exception as e:
            print(f"[MilvusVault] Error getting stats: {e}")
            return {'total_entries': 0, 'collections': 0, 'object_categories': []}

    def delete_collection(self, layer_key: str) -> bool:
        """Delete a collection for a specific layer key."""
        collection_name = self._get_collection_name(layer_key)
        
        try:
            self.client.drop_collection(collection_name=collection_name)
            print(f"[MilvusVault] Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"[MilvusVault] Error deleting collection: {e}")
            return False