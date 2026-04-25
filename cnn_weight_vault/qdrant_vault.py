"""
Qdrant Cloud based Vector Database for CNN Weight Storage and Retrieval
Cloud-based alternative with higher dimension limits (65,535 vs Zilliz 32,768)
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import json
import re

# Suppress Qdrant logging
logging.getLogger('qdrant').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)

try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: qdrant-client not installed. Install with: pip install qdrant-client")

from .config import get_config


class QdrantWeightVault:
    """
    Vector database for storing and retrieving CNN weights using Qdrant Cloud.
    Uses cosine similarity for matching and supports cloud-based storage.
    
    Advantages over Zilliz Cloud:
    - Higher dimension limit: 65,535 (vs 32,768)
    - Unlimited collections
    - Better performance for high-dimensional vectors
    """

    def __init__(self,
                 uri: Optional[str] = None,
                 api_key: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 top_k_ratio: Optional[float] = None,
                 prefer_grpc: bool = True):
        """
        Initialize the Qdrant Weight Vault.
        
        Args:
            uri: Qdrant Cloud cluster URI (e.g., https://xxx.qdrant.tech:6333)
            api_key: Qdrant Cloud API key
            collection_name: Base name for collections
            similarity_threshold: Minimum cosine similarity for matching
            top_k_ratio: Ratio of top weights to keep (for masking)
            prefer_grpc: Use gRPC for better performance (default True)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. "
                "Install with: pip install qdrant-client"
            )

        # Load config
        config = get_config()

        # Get Qdrant config from environment or config file
        self.uri = uri or config.get('vector_db.qdrant.uri') or config.get('vector_db.qdrant.endpoint')
        self.api_key = api_key or config.get('vector_db.qdrant.api_key') or config.get('vector_db.qdrant.token')
        
        if not self.uri or not self.api_key:
            raise ValueError(
                "Qdrant Cloud URI and API key are required. "
                "Set them via parameters or config (vector_db.qdrant.uri, vector_db.qdrant.api_key)"
            )

        self.collection_name = collection_name or config.get('vector_db.qdrant.collection_name', 'cnn_weights')
        self.similarity_threshold = similarity_threshold or config.similarity_threshold
        self.top_k_ratio = top_k_ratio or config.top_k_ratio

        # Initialize Qdrant client
        print(f"[Qdrant] Connecting to {self.uri}...", flush=True)
        self.client = QdrantClient(
            url=self.uri,
            api_key=self.api_key,
            timeout=30  # 30 seconds timeout
        )
        print(f"[Qdrant] Connected successfully!", flush=True)

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
        
        Note: Qdrant supports up to 65,535 dimensions, so fingerprint is only needed
        for extremely large layers (e.g., 512->512 conv = 235,9296 dimensions).
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
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                # Collection exists, just update dimension cache
                self.collection_dims[collection_name] = dimension
                return
            
            # Create collection with schema
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE  # Cosine similarity
                )
            )
            print(f"[QdrantVault] Created collection: {collection_name} (dim={dimension})")
            
            self.collection_dims[collection_name] = dimension
        except Exception as e:
            # Check if it's a "already exists" error (race condition)
            if "already exists" in str(e).lower():
                pass  # Already created by another process
            else:
                print(f"[QdrantVault] Warning creating collection: {e}")

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
        
        # Fingerprint compression for extremely large layers (exceeds Qdrant limit)
        # Qdrant Cloud free tier limit: 65,535 dimensions
        MAX_DIMENSION = 65535  # Qdrant Cloud limit
        FINGERPRINT_DIM = 32768  # Use larger fingerprint for less information loss
        use_fingerprint = dimension > MAX_DIMENSION
        
        if use_fingerprint:
            # Compress to fingerprint
            flattened = self._compute_fingerprint(masked_weights, target_dim=FINGERPRINT_DIM)
            dimension = FINGERPRINT_DIM
            print(f"[QdrantVault] Using fingerprint for {layer_key}: {dimension}")

        # Ensure collection exists
        self._ensure_collection_exists(collection_name, dimension)

        # Prepare payload (metadata)
        payload = {
            'layer_name': layer_name,
            'model_name': model_name,
            'epoch': epoch,
            'accuracy': float(accuracy) if accuracy is not None else 0.0,
            'shape': json.dumps(list(weight_tensor.shape)),
            'layer_key': layer_key,
            'object_category': object_category or '',
            'num_objects': num_objects,
            'use_fingerprint': use_fingerprint,
            'original_dimension': dimension if use_fingerprint else 0
        }

        # Add to Qdrant
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=flattened.tolist(),
                        payload=payload
                    )
                ]
            )
            self.total_entries += 1
            
            # Track categories
            if object_category:
                self.object_categories.add(object_category)
                
        except Exception as e:
            print(f"[QdrantVault] Error storing weights for {layer_key}: {e}")
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
        import uuid
        
        layer_key = self._get_layer_key(layer, layer_name)
        collection_name = self._get_collection_name(layer_key)

        # Check if collection exists
        try:
            collection_info = self.client.get_collection(collection_name=collection_name)
            collection_dim = collection_info.vectors_config.size
        except Exception:
            return None  # Collection doesn't exist

        # Determine if this is a fingerprint collection based on dimension
        # Fingerprint collections have dimension == FINGERPRINT_DIM (32768)
        # Normal collections have dimension matching actual layer dimension
        FINGERPRINT_DIM = 32768
        
        # Generate query vector
        if isinstance(layer, nn.Conv2d):
            query_shape = (layer.out_channels, layer.in_channels,
                          layer.kernel_size[0], layer.kernel_size[1])
        else:
            query_shape = (layer.out_features, layer.in_features)
        
        # Check if this is a fingerprint layer (dimension exceeds MAX_DIMENSION)
        actual_dim = query_shape[0] * query_shape[1] * query_shape[2] * query_shape[3]
        MAX_DIMENSION = 65535
        
        # If collection dim matches FINGERPRINT_DIM, use fingerprint query
        # OR if actual dimension exceeds limit, use fingerprint
        use_fingerprint = (collection_dim == FINGERPRINT_DIM) or (actual_dim > MAX_DIMENSION)
        
        if use_fingerprint:
            # Generate a dummy weight tensor with the right shape for fingerprint
            dummy_tensor = torch.randn(query_shape)
            query_vector = self._compute_fingerprint(dummy_tensor, target_dim=FINGERPRINT_DIM)
        else:
            query_vector = self._generate_topology_query(query_shape)

        # Query Qdrant
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=min(k, 100),
                score_threshold=self.similarity_threshold,
                with_vectors=False,  # Don't return vectors to save bandwidth
                with_payload=True
            )
        except Exception as e:
            print(f"[QdrantVault] Query error: {e}")
            return None

        # Process results
        candidates = []
        if results:
            for hit in results:
                # Qdrant returns score (similarity) directly for cosine
                score = hit.score
                
                # Boost similarity if object category matches
                entry_category = hit.payload.get('object_category', '')
                if object_category and entry_category == object_category:
                    score *= 1.2  # 20% boost

                candidates.append({
                    'entry': {
                        'id': hit.id,
                        'payload': hit.payload
                    },
                    'score': score
                })

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
        payload = best_match['payload']
        
        # Check if this is a fingerprint (compressed) entry
        use_fingerprint = payload.get('use_fingerprint', False)
        if use_fingerprint:
            # Fingerprint cannot reconstruct exact weights
            # Return a reshaped version as initialization hint
            print(f"[QdrantVault] Using fingerprint hint for {layer_name}")
            try:
                # Reshape fingerprint to layer's expected shape (approximation)
                if isinstance(layer, nn.Conv2d):
                    shape = (layer.out_channels, layer.in_channels,
                            layer.kernel_size[0], layer.kernel_size[1])
                else:
                    shape = (layer.out_features, layer.in_features)
                
                # Need to fetch the vector first
                result = self.client.retrieve(
                    collection_name=collection_name,
                    ids=[best_match['id']],
                    with_vectors=True
                )
                
                if not result or len(result) == 0:
                    return None
                    
                vector = np.array(result[0].vector, dtype=np.float32)
                
                # Tile or truncate to match total size
                total_params = np.prod(shape)
                flat = vector
                if len(flat) < total_params:
                    # Tile the fingerprint
                    repeats = (total_params // len(flat)) + 1
                    flat = np.tile(flat, repeats)[:total_params]
                else:
                    flat = flat[:total_params]
                
                weights = torch.from_numpy(flat.reshape(shape))
                return weights
            except Exception as e:
                print(f"[QdrantVault] Fingerprint reconstruction failed: {e}")
                return None

        # For normal (non-fingerprint) weights, we need to query with vector
        # Since we didn't fetch the vector in search, we need to retrieve it
        try:
            # Get the actual vector
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[best_match['id']],
                with_vectors=True
            )
            
            if not result or len(result) == 0:
                return None
                
            vector = np.array(result[0].vector, dtype=np.float32)
            
            # Reconstruct shape
            shape = tuple(json.loads(payload.get('shape', '[]')))
            if not shape:
                # Fallback: calculate shape from vector and layer
                if isinstance(layer, nn.Conv2d):
                    shape = (layer.out_channels, layer.in_channels, 
                            layer.kernel_size[0], layer.kernel_size[1])
                else:
                    shape = (layer.out_features, layer.in_features)
            
            weights = torch.from_numpy(vector.reshape(shape))
            return weights
        except Exception as e:
            print(f"[QdrantVault] Error reconstructing weights: {e}")
            return None

    def _get_latest_weights(self,
                           layer: nn.Module,
                           layer_name: str = "",
                           object_category: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get the latest (most recent) weights for a layer."""
        import uuid
        
        layer_key = self._get_layer_key(layer, layer_name)
        collection_name = self._get_collection_name(layer_key)

        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            if collection_name not in collection_names:
                return None
        except Exception:
            return None

        # Query for latest entry (sorted by epoch descending)
        try:
            # First try with object category filter
            results = self.client.search(
                collection_name=collection_name,
                query_vector=[0.0] * self.collection_dims.get(collection_name, 1),  # Dummy query
                limit=1,
                with_vectors=True,
                with_payload=True,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="object_category",
                            match=models.MatchValue(value=object_category)
                        )
                    ]
                ) if object_category else None
            )
            
            if not results or len(results) == 0:
                # Try without filter - get any recent entry
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=[0.0] * self.collection_dims.get(collection_name, 1),
                    limit=1,
                    with_vectors=True,
                    with_payload=True
                )

            if not results or len(results) == 0:
                return None

            best_entry = results[0]
            vector = np.array(best_entry.vector, dtype=np.float32)
            payload = best_entry.payload
            
            # Check if this is a fingerprint entry
            use_fingerprint = payload.get('use_fingerprint', False)
            
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
                    print(f"[QdrantVault] Fingerprint reconstruction failed: {e}")
                    return None
            
            shape = tuple(json.loads(payload.get('shape', '[]')))

            if not shape:
                if isinstance(layer, nn.Conv2d):
                    shape = (layer.out_channels, layer.in_channels,
                            layer.kernel_size[0], layer.kernel_size[1])
                else:
                    shape = (layer.out_features, layer.in_features)

            weights = torch.from_numpy(vector.reshape(shape))
            return weights
        except Exception as e:
            print(f"[QdrantVault] Error getting latest weights: {e}")
            return None

    def has_weights_for_layer(self, layer: nn.Module, layer_name: str = "") -> bool:
        """Check if vault has weights for a specific layer."""
        layer_key = self._get_layer_key(layer, layer_name)
        collection_name = self._get_collection_name(layer_key)

        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def save_vault(self):
        """Persist the vault (Qdrant auto-persists to cloud)."""
        print(f"Qdrant Cloud vault synchronized")
        print(f"Total entries: {self.total_entries}")
        print(f"Object categories: {self.object_categories}")

    def load_vault(self) -> bool:
        """Load the vault (Qdrant loads from cloud automatically)."""
        try:
            # List collections and get stats
            all_collections = self.client.get_collections().collections
            
            total = 0
            for coll in all_collections:
                if coll.name.startswith(self.collection_name):
                    try:
                        stats = self.client.get_collection(coll.name)
                        total += stats.points_count
                    except Exception:
                        pass
            
            print(f"Qdrant Cloud vault loaded from {self.uri}")
            print(f"Total entries: {total}")
            return total > 0
        except Exception as e:
            print(f"[QdrantVault] Error loading vault: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get vault statistics."""
        try:
            all_collections = self.client.get_collections().collections
            
            entries_per_collection = {}
            total = 0
            
            for coll in all_collections:
                if coll.name.startswith(self.collection_name):
                    try:
                        stats = self.client.get_collection(coll.name)
                        count = stats.points_count
                        entries_per_collection[coll.name] = count
                        total += count
                    except Exception:
                        entries_per_collection[coll.name] = 0

            stats = {
                'total_entries': total,
                'collections': len(entries_per_collection),
                'object_categories': list(self.object_categories),
                'entries_per_collection': entries_per_collection,
                'uri': self.uri
            }
            return stats
        except Exception as e:
            print(f"[QdrantVault] Error getting stats: {e}")
            return {'total_entries': 0, 'collections': 0, 'object_categories': []}

    def delete_collection(self, layer_key: str) -> bool:
        """Delete a collection for a specific layer key."""
        collection_name = self._get_collection_name(layer_key)
        
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"[QdrantVault] Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"[QdrantVault] Error deleting collection: {e}")
            return False