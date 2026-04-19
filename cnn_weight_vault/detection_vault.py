"""
Object Detection Weight Vault - Vector Database for Detection Model Weights
Supports storing/retrieving weights for object detection tasks (bounding box + classification)
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pickle
import os


class DetectionWeightVault:
    """
    Vector database for storing and retrieving object detection model weights.
    Extends WeightVault to support detection-specific features (bbox regression + classification).
    """

    def __init__(self,
                 similarity_threshold: float = 0.5,
                 top_k_ratio: float = 0.3,
                 vault_path: str = "./detection_vault"):
        """
        Initialize the Detection Weight Vault.

        Args:
            similarity_threshold: Minimum cosine similarity for matching
            top_k_ratio: Ratio of top weights to keep (for masking)
            vault_path: Path to save/load the vault
        """
        self.similarity_threshold = similarity_threshold
        self.top_k_ratio = top_k_ratio
        self.vault_path = vault_path

        # Database structure: layer_type -> list of entries
        self.database = defaultdict(list)
        self.graphs = {}
        self.total_entries = 0

        # Detection-specific: store metadata about object categories
        self.object_categories = set()  # Track which object types are in vault

        os.makedirs(vault_path, exist_ok=True)

    def _flatten_kernel(self, weight_tensor: torch.Tensor) -> np.ndarray:
        """Flatten a Conv2d weight tensor to 1D vector."""
        return weight_tensor.detach().cpu().numpy().flatten()

    def _get_layer_key(self, layer: nn.Module, layer_name: str = "") -> str:
        """Generate unique key based on layer dimensions and name."""
        if isinstance(layer, nn.Conv2d):
            key = f"conv_{layer.in_channels}_{layer.out_channels}_{layer.kernel_size[0]}_{layer.kernel_size[1]}"
        elif isinstance(layer, nn.Linear):
            key = f"linear_{layer.in_features}_{layer.out_features}"
        else:
            key = f"other_{type(layer).__name__}"

        # Add layer name hint for detection-specific layers (backbone vs head)
        if "head" in layer_name or "bbox" in layer_name or "class" in layer_name:
            key = f"det_{key}"
        elif "backbone" in layer_name or "conv" in layer_name:
            key = f"backbone_{key}"

        return key

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _distance(self, similarity: float) -> float:
        """Convert similarity to distance (HNSW uses distance metric)."""
        return 1.0 - similarity

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
        Store weights from a trained layer into the vault.

        Args:
            layer: The nn.Module (Conv2d or Linear) to store
            layer_name: Name of the layer in the model
            model_name: Identifier for the model
            epoch: Training epoch
            accuracy: Model mAP accuracy (optional)
            object_category: Type of object being detected (e.g., "cat", "dog")
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

        # Create metadata with detection-specific info
        metadata = {
            'layer_name': layer_name,
            'model_name': model_name,
            'epoch': epoch,
            'accuracy': accuracy,
            'shape': list(weight_tensor.shape),
            'mask': mask.cpu().numpy(),
            'object_category': object_category,  # e.g., "cat", "dog"
            'num_objects': num_objects,
        }

        # Track object categories
        if object_category:
            self.object_categories.add(object_category)

        # Store in database
        entry = {
            'vector': flattened,
            'metadata': metadata
        }
        self.database[layer_key].append(entry)
        self.total_entries += 1

        # Update HNSW graph
        self._update_graph(layer_key, flattened)

    def _update_graph(self, layer_key: str, new_vector: np.ndarray):
        """Update HNSW graph for a layer type with new entry."""
        if layer_key not in self.graphs:
            self.graphs[layer_key] = {'nodes': [], 'edges': []}

        graph = self.graphs[layer_key]
        node_id = len(graph['nodes'])
        graph['nodes'].append(new_vector)

        # Simple HNSW-inspired connection
        if len(graph['nodes']) > 1:
            distances = [(i, self._distance(self._cosine_similarity(new_vector, n)))
                        for i, n in enumerate(graph['nodes'][:-1])]
            distances.sort(key=lambda x: x[1])

            M = min(16, len(distances))
            for i in range(M):
                neighbor_id = distances[i][0]
                graph['edges'].append((node_id, neighbor_id))
                graph['edges'].append((neighbor_id, node_id))

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
            object_category: Preferred object category (e.g., "cat")
            k: Number of candidates to retrieve
        """
        layer_key = self._get_layer_key(layer, layer_name)

        if layer_key not in self.database or len(self.database[layer_key]) == 0:
            return None

        # Generate query based on layer shape
        if isinstance(layer, nn.Conv2d):
            query_shape = (layer.out_channels, layer.in_channels,
                          layer.kernel_size[0], layer.kernel_size[1])
        else:
            query_shape = (layer.out_features, layer.in_features)

        query_vector = self._generate_topology_query(query_shape)

        # Greedy graph search
        candidates = self._greedy_search(layer_key, query_vector, k, object_category)

        if not candidates or candidates[0]['similarity'] < self.similarity_threshold:
            return None

        return candidates

    def _greedy_search(self,
                       layer_key: str,
                       query: np.ndarray,
                       k: int,
                       object_category: Optional[str] = None) -> List[Dict]:
        """
        Greedy graph search on HNSW structure with object category preference.
        """
        nodes = self.database[layer_key]

        if not nodes:
            return []

        scored = []
        for i, entry in enumerate(nodes):
            sim = self._cosine_similarity(query, entry['vector'])

            # Boost similarity if object category matches
            entry_category = entry['metadata'].get('object_category')
            if object_category and entry_category == object_category:
                sim *= 1.2  # 20% boost for same category

            dist = self._distance(sim)
            scored.append((entry, dist, sim))

        # Sort by distance (lower is better)
        scored.sort(key=lambda x: x[1])

        return [{'entry': s[0], 'distance': s[1], 'similarity': s[2]}
                for s in scored[:k]]

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
            force: If True, ignore similarity threshold and use best match

        Returns:
            Weight tensor or None
        """
        if force:
            # Force mode: get latest weights without similarity check
            return self._get_latest_weights(layer, layer_name, object_category)

        candidates = self.query_similar_weights(layer, layer_name, object_category, k=1)

        if not candidates:
            return None

        best_match = candidates[0]['entry']
        vector = best_match['vector']
        metadata = best_match['metadata']

        shape = tuple(metadata['shape'])
        weights = torch.from_numpy(vector).reshape(shape)

        return weights

    def _get_latest_weights(self,
                           layer: nn.Module,
                           layer_name: str = "",
                           object_category: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get the latest stored weights for a layer (for force mode)."""
        layer_key = self._get_layer_key(layer, layer_name)

        if layer_key not in self.database or len(self.database[layer_key]) == 0:
            return None

        entries = self.database[layer_key]

        # Prefer same category, otherwise use latest
        best_entry = None
        for entry in reversed(entries):  # Start from latest
            entry_category = entry['metadata'].get('object_category')
            if object_category and entry_category == object_category:
                best_entry = entry
                break

        if best_entry is None:
            best_entry = entries[-1]  # Use latest

        vector = best_entry['vector']
        metadata = best_entry['metadata']
        shape = tuple(metadata['shape'])

        try:
            weights = torch.from_numpy(vector).reshape(shape)
            return weights
        except:
            return None

    def has_weights_for_layer(self, layer: nn.Module, layer_name: str = "") -> bool:
        """Check if vault has weights for a specific layer."""
        layer_key = self._get_layer_key(layer, layer_name)
        return layer_key in self.database and len(self.database[layer_key]) > 0

    def save_vault(self, filename: str = "detection_vault.pkl"):
        """Save the vault to disk."""
        filepath = os.path.join(self.vault_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'database': dict(self.database),
                'graphs': self.graphs,
                'total_entries': self.total_entries,
                'similarity_threshold': self.similarity_threshold,
                'top_k_ratio': self.top_k_ratio,
                'object_categories': list(self.object_categories)
            }, f)
        print(f"Detection Vault saved to {filepath} ({self.total_entries} entries)")

    def load_vault(self, filename: str = "detection_vault.pkl") -> bool:
        """Load the vault from disk."""
        filepath = os.path.join(self.vault_path, filename)
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.database = defaultdict(list, data['database'])
            self.graphs = data['graphs']
            self.total_entries = data['total_entries']
            self.similarity_threshold = data['similarity_threshold']
            self.top_k_ratio = data['top_k_ratio']
            self.object_categories = set(data.get('object_categories', []))

        print(f"Detection Vault loaded from {filepath} ({self.total_entries} entries)")
        print(f"Object categories: {self.object_categories}")
        return True

    def get_stats(self) -> Dict:
        """Get vault statistics."""
        return {
            'total_entries': self.total_entries,
            'layer_types': len(self.database),
            'object_categories': list(self.object_categories),
            'entries_per_type': {k: len(v) for k, v in self.database.items()}
        }
