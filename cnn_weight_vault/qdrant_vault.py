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
import uuid

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
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 top_k_ratio: Optional[float] = None,
                 prefer_grpc: bool = True):
        """
        Initialize the Qdrant Weight Vault.
        
        Args:
            url: Qdrant Cloud cluster URL (e.g., https://xxx.qdrant.tech:6333)
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
        self.url = url or config.get('vector_db.qdrant.url') or config.get('vector_db.qdrant.uri')
        self.api_key = api_key or config.get('vector_db.qdrant.api_key') or config.get('vector_db.qdrant.token')
        
        if not self.url or not self.api_key:
            raise ValueError(
                "Qdrant Cloud URL and API key are required. "
                "Set them via parameters or config (vector_db.qdrant.url, vector_db.qdrant.api_key)"
            )

        self.collection_name = collection_name or config.get('vector_db.qdrant.collection_name', 'cnn_weights')
        self.similarity_threshold = similarity_threshold or config.similarity_threshold
        self.top_k_ratio = top_k_ratio or config.top_k_ratio

        # Initialize Qdrant client
        print(f"[Qdrant] Connecting to {self.url}...", flush=True)
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30,  # 30 seconds timeout
            check_compatibility=False  # Skip version check
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

    # ===== SIMPLIFIED: One collection per category =====
    
    # ===== Image-based category retrieval =====
    
    def store_category_weights(self,
                               model_state: Dict[str, torch.Tensor],
                               model_name: str,
                               category: str,
                               epoch: int,
                               accuracy: Optional[float] = None,
                               dataset_sample: Optional[List[torch.Tensor]] = None):
        """
        Store model weights for a category with optional image features.
        """
        collection_name = f"{self.collection_name}"
        
        # Ensure collection exists with HNSW
        self._ensure_hnsw_collection(collection_name)
        
        # Extract image features (or use model features as fallback)
        if dataset_sample:
            feature_vector = self._extract_image_features(dataset_sample)
        else:
            feature_vector = self._extract_features(model_state)
        
        # Convert all weights to JSON for payload
        weights_json = {}
        for name, tensor in model_state.items():
            if isinstance(tensor, torch.Tensor):
                weights_json[name] = tensor.detach().cpu().numpy().tolist()
        
        # Get image stats if available
        image_stats = self._get_image_stats(dataset_sample) if dataset_sample else {}
        
        payload = {
            "model_name": model_name,
            "category": category,
            "epoch": epoch,
            "accuracy": accuracy,
            "weights": weights_json,
            "image_stats": image_stats
        }
        
        import uuid
        
        # Delete existing and upsert
        self._delete_category(collection_name, category)
        
        self.client.upsert(
            collection_name=collection_name,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=feature_vector,
                payload=payload
            )]
        )
        
        print(f"[QdrantVault] Stored '{category}' with image features (acc={accuracy:.1f}%)")
    
    def _extract_image_features(self, images: List[torch.Tensor]) -> List[float]:
        """Extract statistical features from images."""
        features = []
        
        for img in images[:50]:
            if not isinstance(img, torch.Tensor):
                continue
            
            img_flat = img.flatten()
            
            # Basic statistics
            features.append(float(img_flat.mean()))
            features.append(float(img_flat.std()))
            features.append(float(img_flat.min()))
            features.append(float(img_flat.max()))
            
            # Histogram
            hist = torch.histc(img_flat, bins=8)
            for h in hist:
                features.append(float(h) / img_flat.numel())
        
        # Pad to fixed size
        MAX_FEATURES = 512
        if len(features) < MAX_FEATURES:
            features.extend([0.0] * (MAX_FEATURES - len(features)))
        else:
            features = features[:MAX_FEATURES]
        
        return features
    
    def _get_image_stats(self, images: List[torch.Tensor]) -> Dict:
        """Get simple statistics from dataset images."""
        if not images:
            return {}
        
        all_means = []
        all_stds = []
        
        for img in images[:100]:
            if isinstance(img, torch.Tensor):
                img_flat = img.flatten()
                all_means.append(img_flat.mean().item())
                all_stds.append(img_flat.std().item())
        
        return {
            "mean": float(np.mean(all_means)) if all_means else 0.0,
            "std": float(np.mean(all_stds)) if all_stds else 0.0,
            "num_samples": len(images)
        }
    
    def _extract_features(self, model_state: Dict[str, torch.Tensor]) -> List[float]:
        """Fallback: extract features from model weights."""
        features = []
        
        for name in sorted(model_state.keys()):
            tensor = model_state[name]
            if not isinstance(tensor, torch.Tensor):
                continue
            
            arr = tensor.detach().cpu().numpy().flatten()
            features.append(float(np.mean(arr)))
            features.append(float(np.std(arr)))
            features.append(float(np.min(arr)))
            features.append(float(np.max(arr)))
        
        MAX_FEATURES = 512
        if len(features) < MAX_FEATURES:
            features.extend([0.0] * (MAX_FEATURES - len(features)))
        else:
            features = features[:MAX_FEATURES]
        
        return features
    
    def _ensure_hnsw_collection(self, collection_name: str):
        """Create collection with HNSW index if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                return
            
            FEATURE_DIM = 512
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=FEATURE_DIM,
                    distance=models.Distance.COSINE
                )
            )
            
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="category",
                    field_schema=models.KeywordIndexParams(type="keyword")
                )
            except Exception as e:
                print(f"[QdrantVault] Warning creating index: {e}")
            
            print(f"[QdrantVault] Created collection: {collection_name}")
            
        except Exception as e:
            print(f"[QdrantVault] Error creating collection: {e}")
    
    def _delete_category(self, collection_name: str, category: str):
        """Delete existing entry for a category."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category",
                            match=models.MatchValue(value=category)
                        )
                    ]
                )
            )
        except:
            pass
    
    def find_similar_category_by_images(self, dataset_sample: List[torch.Tensor]) -> Tuple[Optional[str], float]:
        """Find similar category by image features."""
        collection_name = f"{self.collection_name}"
        
        query_features = self._extract_image_features(dataset_sample)
        
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_features,
                limit=3,
                with_payload=True
            )
            
            # Handle different response formats
            if hasattr(results, 'result'):
                results = results.result
            if hasattr(results, 'points'):
                results = results.points
            
            if not results:
                print("[QdrantVault] No categories found in vault")
                return None, 0.0
            
            for r in results:
                cat = r.payload.get('category', 'unknown')
                stats = r.payload.get('image_stats', {})
                print(f"[QdrantVault]   {cat}: score={r.score:.4f} (mean={stats.get('mean', 0):.3f})")
            
            best = results[0]
            return best.payload.get('category'), best.score
            
        except Exception as e:
            print(f"[QdrantVault] Image search error: {e}")
            return None, 0.0
    
    def get_category_weights(self, category: str) -> Optional[Dict[str, np.ndarray]]:
        """Retrieve all weights for a specific category."""
        collection_name = f"{self.collection_name}"
        
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=[0.0] * 512,
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category",
                            match=models.MatchValue(value=category)
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )
            
            # Handle different response formats
            if hasattr(results, 'result'):
                results = results.result
            if hasattr(results, 'points'):
                results = results.points
            
            if not results:
                print(f"[QdrantVault] No weights found for category '{category}'")
                return None
            
            payload = results[0].payload
            weights_json = payload.get('weights', {})
            
            weights = {}
            for name, data in weights_json.items():
                weights[name] = np.array(data, dtype=np.float32)
            
            print(f"[QdrantVault] Loaded weights for '{category}' (epoch={payload.get('epoch')})")
            return weights
            
        except Exception as e:
            print(f"[QdrantVault] Error loading weights: {e}")
            return None
    
    # ===== END =====
    
    def _extract_image_features(self, images: List[torch.Tensor]) -> List[float]:
        """
        Extract features from image tensors using statistical features.
        Fast and simple - no deep learning needed.
        """
        features = []
        
        for img in images[:50]:  # Use up to 50 images
            if not isinstance(img, torch.Tensor):
                continue
            
            # Handle different image formats
            if img.dim() == 3:
                # (C, H, W) -> (H, W, C) or (C, H, W)
                img_flat = img.flatten()
            else:
                img_flat = img.flatten()
            
            # Basic statistics
            features.append(float(img_flat.mean()))
            features.append(float(img_flat.std()))
            features.append(float(img_flat.min()))
            features.append(float(img_flat.max()))
            
            # Histogram (simplified)
            hist = torch.histc(img_flat, bins=8)
            for h in hist:
                features.append(float(h) / img_flat.numel())
        
        # Pad to fixed size
        MAX_FEATURES = 512
        if len(features) < MAX_FEATURES:
            features.extend([0.0] * (MAX_FEATURES - len(features)))
        else:
            features = features[:MAX_FEATURES]
        
        return features
    
    def _get_image_stats(self, images: List[torch.Tensor]) -> Dict:
        """Get simple statistics from dataset images."""
        if not images:
            return {}
        
        # Calculate overall stats
        all_means = []
        all_stds = []
        
        for img in images[:100]:  # Sample 100 images
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    for c in img:
                        all_means.append(c.mean().item())
                        all_stds.append(c.std().item())
                else:
                    all_means.append(img.mean().item())
                    all_stds.append(img.std().item())
        
        return {
            "mean": float(np.mean(all_means)) if all_means else 0.0,
            "std": float(np.mean(all_stds)) if all_stds else 0.0,
            "num_samples": len(images)
        }
    
    def find_similar_category_by_images(self, dataset_sample: List[torch.Tensor]) -> Tuple[Optional[str], float]:
        """
        Find the most similar category by comparing image features.
        Use this for new datasets before training.
        
        Args:
            dataset_sample: List of image tensors from new dataset
            
        Returns:
            (best_category, similarity_score)
        """
        collection_name = f"{self.collection_name}"
        
        # Extract image features from new dataset
        query_features = self._extract_image_features(dataset_sample)
        
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_features,
                limit=3,
                with_payload=True
            )
            
            # Handle different response formats
            if hasattr(results, 'result'):
                results = results.result
            if hasattr(results, 'points'):
                results = results.points
            
            if not results:
                print("[QdrantVault] No categories found in vault")
                return None, 0.0
            
            for r in results:
                cat = r.payload.get('category', 'unknown')
                stats = r.payload.get('image_stats', {})
                print(f"[QdrantVault]   {cat}: score={r.score:.4f} (mean={stats.get('mean', 0):.3f})")
            
            best = results[0]
            return best.payload.get('category'), best.score
            
        except Exception as e:
            print(f"[QdrantVault] Image search error: {e}")
            return None, 0.0
    
    # ===== END IMAGE SEARCH =====
    
    def _ensure_category_collection(self, collection_name: str):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                return
            
            # Create with fixed dimension for fingerprint
            FINGERPRINT_DIM = 1024
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=FINGERPRINT_DIM,
                    distance=models.Distance.COSINE
                )
            )
            
            # Create index on category for filtering
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="category",
                    field_schema=models.KeywordIndexParams(type="keyword")
                )
            except Exception as e:
                print(f"[QdrantVault] Warning creating index: {e}")
            
            print(f"[QdrantVault] Created collection: {collection_name}")
            
        except Exception as e:
            print(f"[QdrantVault] Error creating collection: {e}")
    
    def get_category_weights(self, category: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve all weights for a category.
        
        Returns:
            dict of layer_name -> numpy array, or None if not found
        """
        collection_name = f"{self.collection_name}_{category}"
        
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=[0.0] * 1024,  # Dummy query
                limit=1,
                with_vectors=False,
                with_payload=True
            )
            
            if not results or not results.points:
                print(f"[QdrantVault] No weights found for category '{category}'")
                return None
            
            payload = results.points[0].payload
            weights_json = payload.get('weights', {})
            
            # Convert JSON back to numpy
            weights = {}
            for name, data in weights_json.items():
                weights[name] = np.array(data, dtype=np.float32)
            
            print(f"[QdrantVault] Loaded weights for category '{category}' (epoch={payload.get('epoch')})")
            return weights
            
        except Exception as e:
            print(f"[QdrantVault] Error loading weights for '{category}': {e}")
            return None
    
    def find_similar_category(self, model_state: Dict[str, torch.Tensor]) -> Tuple[Optional[str], float]:
        """
        Find the most similar category by comparing full weights.
        
        Args:
            model_state: dict of layer_name -> weight tensor for the new model
            
        Returns:
            (best_category, similarity_score) or (None, 0.0) if no match
        """
        # Flatten all weights from the new model
        new_weights_flat = []
        for name, tensor in model_state.items():
            if isinstance(tensor, torch.Tensor):
                new_weights_flat.append(tensor.detach().cpu().numpy().flatten())
        new_concat = np.concatenate(new_weights_flat)
        new_norm = np.linalg.norm(new_concat)
        
        if new_norm == 0:
            return None, 0.0
        
        # Get all available categories
        all_categories = self._list_categories()
        if not all_categories:
            print("[QdrantVault] No categories found in vault")
            return None, 0.0
        
        print(f"[QdrantVault] Searching for similar category among: {all_categories}")
        
        best_category = None
        best_similarity = -1.0
        
        for category in all_categories:
            # Load weights for this category
            weights = self.get_category_weights(category)
            if weights is None:
                continue
            
            # Flatten and compute similarity
            cat_weights_flat = []
            for name, arr in weights.items():
                cat_weights_flat.append(arr.flatten())
            cat_concat = np.concatenate(cat_weights_flat)
            
            # Cosine similarity
            similarity = np.dot(new_concat, cat_concat) / (new_norm * np.linalg.norm(cat_concat))
            
            print(f"[QdrantVault]   vs {category}: similarity = {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category
        
        if best_category:
            print(f"[QdrantVault] Best match: '{best_category}' (similarity={best_similarity:.4f})")
        else:
            print(f"[QdrantVault] No similar category found")
        
        return best_category, best_similarity
    
    def _list_categories(self) -> List[str]:
        """List all available categories in the vault."""
        try:
            collections = self.client.get_collections().collections
            prefix = f"{self.collection_name}_"
            categories = []
            
            for c in collections:
                if c.name.startswith(prefix):
                    category = c.name[len(prefix):]
                    categories.append(category)
            
            return categories
        except Exception as e:
            print(f"[QdrantVault] Error listing categories: {e}")
            return []
    
    # ===== END SIMPLIFIED =====

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
            
            # Create index on object_category for filtering
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="object_category",
                    field_schema=models.KeywordIndexParams(type="keyword")
                )
            except Exception as e:
                print(f"[QdrantVault] Warning creating index: {e}")
            
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
        
        # Qdrant Cloud free tier limit: 65,535 dimensions
        MAX_DIMENSION = 65535  # Qdrant Cloud limit

        # Check if chunking is needed (for large layers like fc1)
        # For Linear layers with out_features > in_features, chunk by output
        use_chunking = isinstance(layer, nn.Linear) and layer.out_features > MAX_DIMENSION // layer.in_features
        
        if use_chunking:
            # Chunk the weights by output neurons
            chunk_size = MAX_DIMENSION // layer.in_features  # Max chunk that fits
            n_chunks = (layer.out_features + chunk_size - 1) // chunk_size
            print(f"[QdrantVault] Chunking {layer_key} into {n_chunks} parts (each {chunk_size}x{layer.in_features})")
            
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, layer.out_features)
                chunk = masked_weights[start_idx:end_idx, :]
                flattened_chunk = self._flatten_kernel(chunk)
                dim_chunk = len(flattened_chunk)
                
                chunk_key = f"{layer_key}_chunk_{chunk_idx}"
                chunk_collection = self._get_collection_name(chunk_key)
                
                self._ensure_collection_exists(chunk_collection, dim_chunk)
                
                payload = {
                    'model_name': model_name,
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'object_category': object_category,
                    'num_objects': num_objects,
                    'layer_key': layer_key,
                    'chunk_idx': chunk_idx,
                    'n_chunks': n_chunks,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'original_shape': list(layer.weight.shape)
                }
                
                self.client.upsert(
                    collection_name=chunk_collection,
                    points=[models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=flattened_chunk.tolist(),
                        payload=payload
                    )]
                )
            
            print(f"[QdrantVault] Stored {n_chunks} chunks for {layer_key}")
            return  # Skip fingerprint storage
        
        # Fingerprint compression for extremely large layers (exceeds Qdrant limit)
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
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
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
        if results and results.points:
            for hit in results.points:
                # Qdrant returns score (similarity) directly for cosine
                score = hit.score
                
                # Boost similarity if object category matches
                entry_category = hit.payload.get('object_category', '')
                if object_category and entry_category == object_category:
                    score *= 1.2  # 20% boost

                candidates.append({
                    'entry': {
                        'id': str(hit.id),
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
            results = None
            if object_category:
                try:
                    results = self.client.query_points(
                        collection_name=collection_name,
                        query=[0.0] * self.collection_dims.get(collection_name, 1),
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
                        )
                    )
                except Exception as e:
                    print(f"[QdrantVault] Filter failed (no index?), trying without: {e}")
            
            if not results or not results.points or len(results.points) == 0:
                # Try without filter - get any recent entry
                results = self.client.query_points(
                    collection_name=collection_name,
                    query=[0.0] * self.collection_dims.get(collection_name, 1),
                    limit=1,
                    with_vectors=True,
                    with_payload=True
                )

            if not results or not results.points or len(results.points) == 0:
                return None

            best_entry = results.points[0]
            vector = np.array(best_entry.vector, dtype=np.float32)
            payload = best_entry.payload
            
            # Check if this is a chunked entry
            if 'chunk_idx' in payload:
                # This is a chunk, need to retrieve all chunks
                layer_key = payload.get('layer_key', layer_name)
                n_chunks = payload.get('n_chunks', 1)
                
                # Retrieve all chunks
                all_chunks = []
                for ci in range(n_chunks):
                    chunk_key = f"{layer_key}_chunk_{ci}"
                    chunk_collection = self._get_collection_name(chunk_key)
                    
                    # Get the actual dimension from collection info
                    chunk_dim = self.collection_dims.get(chunk_collection, None)
                    if chunk_dim is None:
                        try:
                            info = self.client.get_collection(chunk_collection)
                            # Get vector config for dimension
                            chunk_dim = info.config.vectors.size
                            self.collection_dims[chunk_collection] = chunk_dim
                            print(f"[QdrantVault] Retrieved chunk {ci} dim: {chunk_dim}")
                        except Exception as e:
                            print(f"[QdrantVault] Cannot get collection {chunk_collection}: {e}")
                            continue
                    
                    try:
                        chunk_results = self.client.query_points(
                            collection_name=chunk_collection,
                            query=[0.0] * chunk_dim,
                            limit=1,
                            with_vectors=True,
                            with_payload=True
                        )
                        if chunk_results and chunk_results.points:
                            all_chunks.append((chunk_results.points[0].payload.get('start_idx', 0), 
                                             np.array(chunk_results.points[0].vector, dtype=np.float32)))
                    except Exception as e:
                        print(f"[QdrantVault] Error fetching chunk {ci}: {e}")
                        pass
                
                if all_chunks:
                    # Sort by start_idx and concatenate
                    all_chunks.sort(key=lambda x: x[0])
                    concatenated = np.concatenate([c[1] for c in all_chunks])
                    original_shape = tuple(json.loads(payload.get('original_shape', '[]')))
                    if original_shape:
                        return torch.from_numpy(concatenated.reshape(original_shape))
                    return torch.from_numpy(concatenated)
                return None
            
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
            
            print(f"Qdrant Cloud vault loaded from {self.url}")
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
                'url': self.url
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