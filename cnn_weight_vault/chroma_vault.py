"""
ChromaDB-based Vector Database for CNN Weight Storage and Retrieval
Replaces pickle-based storage with proper vector database
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import uuid
import json
import os
import re
from collections import defaultdict

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. Falling back to pickle storage.")

from .config import get_config


class ChromaWeightVault:
    """
    Vector database for storing and retrieving CNN weights using ChromaDB.
    Uses cosine similarity for matching and supports HNSW indexing.
    """

    def __init__(self,
                 collection_name: Optional[str] = None,
                 persist_directory: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 top_k_ratio: Optional[float] = None):
        """
        Initialize the ChromaDB Weight Vault.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            similarity_threshold: Minimum cosine similarity for matching
            top_k_ratio: Ratio of top weights to keep (for masking)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. "
                "Install with: pip install chromadb>=0.4.22"
            )

        # Load config
        config = get_config()

        self.collection_name = collection_name or config.chroma_collection_name
        self.persist_directory = persist_directory or config.chroma_persist_dir
        self.similarity_threshold = similarity_threshold or config.similarity_threshold
        self.top_k_ratio = top_k_ratio or config.top_k_ratio

        # Initialize ChromaDB PersistentClient (new API)
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # Collections will be created dynamically per layer_key
        self.collections = {}

        # Statistics
        self.total_entries = 0
        self.object_categories = set()

    def _get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one."""
        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            return self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

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

    def _get_collection_for_layer(self, layer_key: str):
        """Get or create collection for a specific layer key.

        Each unique layer_key (e.g., 'conv_3_32_3_3') gets its own collection
        to ensure all vectors in the collection have the same dimension.
        """
        # Sanitize layer_key for collection name (ChromaDB requires alphanumeric, underscores, hyphens only)
        # Remove any special characters
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', layer_key)[:50]
        collection_name = f"{self.collection_name}_{safe_key}"

        # Check if collection already exists in cache
        if collection_name in self.collections:
            return self.collections[collection_name]

        # Get or create collection
        collection = self._get_or_create_collection(collection_name)
        self.collections[collection_name] = collection
        return collection

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

        # Apply top-k masking
        weight_tensor = layer.weight.data
        masked_weights, mask = self._apply_top_k_mask(weight_tensor)

        # Flatten the masked weights
        flattened = self._flatten_kernel(masked_weights)

        # Generate unique ID
        entry_id = str(uuid.uuid4())

        # Prepare metadata
        metadata = {
            'layer_name': layer_name,
            'model_name': model_name,
            'epoch': epoch,
            'accuracy': accuracy if accuracy is not None else 0.0,
            'shape': json.dumps(list(weight_tensor.shape)),
            'layer_key': layer_key,
            'object_category': object_category or '',
            'num_objects': num_objects,
            'mask': json.dumps(mask.cpu().numpy().tolist())
        }

        # Get appropriate collection
        collection = self._get_collection_for_layer(layer_key)

        # Add to ChromaDB
        collection.add(
            embeddings=[flattened.tolist()],
            ids=[entry_id],
            metadatas=[metadata]
        )

        # Track categories
        if object_category:
            self.object_categories.add(object_category)

        self.total_entries += 1

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

        # Get appropriate collection
        collection = self._get_collection_for_layer(layer_key)

        # Check if collection has data
        if collection.count() == 0:
            return None

        # Generate query based on layer shape
        if isinstance(layer, nn.Conv2d):
            query_shape = (layer.out_channels, layer.in_channels,
                          layer.kernel_size[0], layer.kernel_size[1])
        else:
            query_shape = (layer.out_features, layer.in_features)

        query_vector = self._generate_topology_query(query_shape)

        # Query ChromaDB
        try:
            results = collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=min(k, collection.count()),
                include=['metadatas', 'distances', 'embeddings']
            )
        except Exception as e:
            print(f"Query error: {e}")
            return None

        # Process results
        candidates = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, entry_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                embedding = results['embeddings'][0][i]

                # Convert distance to similarity (ChromaDB cosine distance = 1 - cosine similarity)
                similarity = 1.0 - distance

                # Boost similarity if object category matches
                entry_category = metadata.get('object_category', '')
                if object_category and entry_category == object_category:
                    similarity *= 1.2  # 20% boost

                candidates.append({
                    'entry': {
                        'vector': np.array(embedding, dtype=np.float32),
                        'metadata': metadata
                    },
                    'distance': 1.0 - similarity,
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

        shape = tuple(json.loads(metadata['shape']))
        weights = torch.from_numpy(vector).reshape(shape)

        return weights

    def _get_latest_weights(self,
                           layer: nn.Module,
                           layer_name: str = "",
                           object_category: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get the latest stored weights for a layer (for force mode)."""
        layer_key = self._get_layer_key(layer, layer_name)
        collection = self._get_collection_for_layer(layer_key)

        try:
            if collection.count() == 0:
                return None

            # Get all entries
            results = collection.get(include=['metadatas', 'embeddings'])

            if not results or not results.get('ids'):
                return None
        except Exception as e:
            print(f"Warning: Error reading from collection: {e}")
            return None

        # Find best entry (prefer same category, otherwise latest epoch)
        best_entry = None
        best_score = -1

        for i, metadata in enumerate(results['metadatas']):
            entry_category = metadata.get('object_category', '')
            epoch = metadata.get('epoch', 0)

            score = epoch
            if object_category and entry_category == object_category:
                score += 10000  # Boost for matching category

            if score > best_score:
                best_score = score
                best_entry = {
                    'vector': np.array(results['embeddings'][i], dtype=np.float32),
                    'metadata': metadata
                }

        if best_entry is None:
            return None

        vector = best_entry['vector']
        metadata = best_entry['metadata']
        shape = tuple(json.loads(metadata['shape']))

        try:
            weights = torch.from_numpy(vector).reshape(shape)
            return weights
        except Exception:
            return None

    def has_weights_for_layer(self, layer: nn.Module, layer_name: str = "") -> bool:
        """Check if vault has weights for a specific layer."""
        layer_key = self._get_layer_key(layer, layer_name)
        collection = self._get_collection_for_layer(layer_key)
        return collection.count() > 0

    def save_vault(self):
        """Persist the vault to disk."""
        # ChromaDB auto-persists, but we can force it
        print(f"ChromaDB vault persisted to {self.persist_directory}")
        print(f"Total entries: {self.total_entries}")
        print(f"Object categories: {self.object_categories}")

    def load_vault(self) -> bool:
        """Load the vault (ChromaDB loads automatically)."""
        # ChromaDB loads from persist_directory automatically
        total = sum(c.count() for c in self.collections.values())
        print(f"ChromaDB vault loaded from {self.persist_directory}")
        print(f"Total entries: {total}")
        return total > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get vault statistics."""
        stats = {
            'total_entries': sum(c.count() for c in self.collections.values()),
            'collections': len(self.collections),
            'object_categories': list(self.object_categories),
            'entries_per_collection': {
                name: collection.count()
                for name, collection in self.collections.items()
            }
        }
        return stats

    def migrate_from_pickle(self, pickle_vault_path: str) -> int:
        """
        Migrate data from a pickle-based vault to ChromaDB.

        Args:
            pickle_vault_path: Path to the pickle file

        Returns:
            Number of entries migrated
        """
        import pickle

        if not os.path.exists(pickle_vault_path):
            print(f"Pickle vault not found: {pickle_vault_path}")
            return 0

        with open(pickle_vault_path, 'rb') as f:
            data = pickle.load(f)

        database = data.get('database', {})
        migrated = 0

        for layer_key, entries in database.items():
            for entry in entries:
                try:
                    vector = entry['vector']
                    metadata = entry['metadata']

                    # Convert to ChromaDB format
                    entry_id = str(uuid.uuid4())

                    # Handle metadata serialization
                    chroma_metadata = {
                        'layer_name': metadata.get('layer_name', ''),
                        'model_name': metadata.get('model_name', ''),
                        'epoch': metadata.get('epoch', 0),
                        'accuracy': metadata.get('accuracy') or 0.0,
                        'shape': json.dumps(metadata.get('shape', [])),
                        'layer_key': layer_key,
                        'object_category': metadata.get('object_category', ''),
                        'num_objects': metadata.get('num_objects', 0),
                        'mask': json.dumps(metadata.get('mask', []).tolist())
                    }

                    # Get collection
                    collection = self._get_collection_for_layer(layer_key)

                    # Add to ChromaDB
                    collection.add(
                        embeddings=[vector.tolist()],
                        ids=[entry_id],
                        metadatas=[chroma_metadata]
                    )

                    migrated += 1
                except Exception as e:
                    print(f"Error migrating entry: {e}")
                    continue

        self.total_entries = migrated
        print(f"Migrated {migrated} entries from {pickle_vault_path}")
        return migrated
