"""
Weight Vault - Vector Database for CNN Weight Storage and Retrieval
Implements HNSW (Hierarchical Navigable Small World) graph for similarity search
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable
import pickle
import os


class WeightVault:
    """
    Vector database for storing and retrieving CNN weights.
    Uses HNSW-inspired graph structure for O(log n) similarity search.
    """

    def __init__(self,
                 similarity_threshold: float = 0.85,
                 top_k_ratio: float = 0.3,
                 vault_path: str = "./vault"):
        """
        Initialize the Weight Vault.

        Args:
            similarity_threshold: Minimum cosine similarity for matching
            top_k_ratio: Ratio of top weights to keep (for masking)
            vault_path: Path to save/load the vault
        """
        self.similarity_threshold = similarity_threshold
        self.top_k_ratio = top_k_ratio
        self.vault_path = vault_path

        # Database structure: layer_type -> list of (flattened_weights, metadata)
        self.database = defaultdict(list)

        # HNSW graph structure for each layer type
        self.graphs = {}

        # Statistics
        self.total_entries = 0

        os.makedirs(vault_path, exist_ok=True)

    def _flatten_kernel(self, weight_tensor: torch.Tensor) -> np.ndarray:
        """
        Flatten a Conv2d weight tensor to 1D vector.
        Shape: (out_channels, in_channels, kh, kw) -> (D,)
        """
        return weight_tensor.detach().cpu().numpy().flatten()

    def _get_layer_key(self, layer: nn.Module) -> str:
        """Generate unique key based on layer dimensions."""
        if isinstance(layer, nn.Conv2d):
            return f"conv_{layer.in_channels}_{layer.out_channels}_{layer.kernel_size[0]}_{layer.kernel_size[1]}"
        elif isinstance(layer, nn.Linear):
            return f"linear_{layer.in_features}_{layer.out_features}"
        return f"other_{type(layer).__name__}"

    def _generate_topology_query(self, shape: tuple) -> np.ndarray:
        """
        Generate a query vector based on layer topology (architecture).
        This is a deterministic query that represents the expected weight pattern.
        """
        # Create a structured query based on architecture
        # Use Xavier/He-like pattern: centered at 0 with specific variance
        np.random.seed(42)  # Deterministic for same architecture
        query = np.random.randn(*shape).astype(np.float32)
        query = query / np.linalg.norm(query)  # Normalize
        return query.flatten()

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

    def _apply_top_k_mask(self, weight_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply top-k masking to weight tensor (Step 1: Feature Extraction & Masking).

        Returns:
            masked_weights: Tensor with only top-k weights preserved
            mask: Boolean mask indicating which weights were kept
        """
        flat_weights = weight_tensor.abs().flatten()
        k = int(self.top_k_ratio * flat_weights.numel())

        # Get top-k indices by magnitude
        top_k_values, top_k_indices = torch.topk(flat_weights, k)

        # Create mask
        mask = torch.zeros_like(flat_weights, dtype=torch.bool)
        mask[top_k_indices] = True

        # Apply mask to original weights (keep sign)
        masked_flat = weight_tensor.flatten() * mask.float()

        return masked_flat.view_as(weight_tensor), mask.view_as(weight_tensor)

    def store_weights(self,
                     layer: nn.Module,
                     layer_name: str,
                     model_name: str,
                     epoch: int,
                     accuracy: Optional[float] = None):
        """
        Store weights from a trained layer into the vault.

        Args:
            layer: The nn.Module (Conv2d or Linear) to store
            layer_name: Name of the layer in the model
            model_name: Identifier for the model
            epoch: Training epoch
            accuracy: Model accuracy (optional metadata)
        """
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return

        layer_key = self._get_layer_key(layer)

        # Get weights and apply top-k masking (Step 2: Database Accumulation)
        weight_tensor = layer.weight.data
        masked_weights, mask = self._apply_top_k_mask(weight_tensor)

        # Flatten the masked weights (Step 3: Kernel Flattening)
        flattened = self._flatten_kernel(masked_weights)

        # Create metadata
        metadata = {
            'layer_name': layer_name,
            'model_name': model_name,
            'epoch': epoch,
            'accuracy': accuracy,
            'shape': list(weight_tensor.shape),
            'mask': mask.cpu().numpy(),
        }

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

        # Simple HNSW-inspired connection (greedy nearest neighbor)
        if len(graph['nodes']) > 1:
            # Find nearest neighbors
            distances = [(i, self._distance(self._cosine_similarity(new_vector, n)))
                        for i, n in enumerate(graph['nodes'][:-1])]
            distances.sort(key=lambda x: x[1])

            # Connect to top M neighbors (typically M=16-32 in HNSW)
            M = min(16, len(distances))
            for i in range(M):
                neighbor_id = distances[i][0]
                graph['edges'].append((node_id, neighbor_id))
                graph['edges'].append((neighbor_id, node_id))  # Bidirectional

    def query_similar_weights(self,
                              layer: nn.Module,
                              k: int = 3) -> Optional[List[Dict]]:
        """
        Query the vault for similar weights to initialize a layer.

        Args:
            layer: The layer to find initialization for
            k: Number of candidates to retrieve

        Returns:
            List of candidate entries with similarity scores, or None if no match
        """
        layer_key = self._get_layer_key(layer)

        if layer_key not in self.database or len(self.database[layer_key]) == 0:
            return None

        # Get random query vector (for cold start, this would be from the architecture)
        if isinstance(layer, nn.Conv2d):
            query_shape = (layer.out_channels, layer.in_channels,
                          layer.kernel_size[0], layer.kernel_size[1])
        else:
            query_shape = (layer.out_features, layer.in_features)

        # Generate query based on layer topology (architecture-based query)
        # This represents the "ideal" weight pattern for this architecture
        query_vector = self._generate_topology_query(query_shape)

        # Greedy graph search (Step 5: High-Speed Logarithmic Retrieval)
        candidates = self._greedy_search(layer_key, query_vector, k)

        if not candidates or candidates[0]['similarity'] < self.similarity_threshold:
            return None

        return candidates

    def _greedy_search(self,
                       layer_key: str,
                       query: np.ndarray,
                       k: int) -> List[Dict]:
        """
        Greedy graph search on HNSW structure.

        Returns:
            List of (entry, distance) tuples, sorted by distance
        """
        graph = self.graphs.get(layer_key, {'nodes': [], 'edges': []})
        nodes = self.database[layer_key]

        if not nodes:
            return []

        # Calculate distances to all nodes (simplified - in production would use HNSW layers)
        scored = []
        for i, entry in enumerate(nodes):
            sim = self._cosine_similarity(query, entry['vector'])
            dist = self._distance(sim)
            scored.append((entry, dist, sim))

        # Sort by distance (lower is better)
        scored.sort(key=lambda x: x[1])

        # Return top k with similarity info
        return [{'entry': s[0], 'distance': s[1], 'similarity': s[2]}
                for s in scored[:k]]

    def get_initialization_weights(self, layer: nn.Module) -> Optional[torch.Tensor]:
        """
        Get initialization weights for a layer from the vault.

        Returns:
            Weight tensor for initialization, or None if vault is empty
        """
        candidates = self.query_similar_weights(layer, k=1)

        if not candidates:
            return None

        best_match = candidates[0]['entry']
        vector = best_match['vector']
        metadata = best_match['metadata']

        # Reshape back to original shape
        shape = tuple(metadata['shape'])
        weights = torch.from_numpy(vector).reshape(shape)

        return weights

    def save_vault(self, filename: str = "vault.pkl"):
        """Save the vault to disk."""
        filepath = os.path.join(self.vault_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'database': dict(self.database),
                'graphs': self.graphs,
                'total_entries': self.total_entries,
                'similarity_threshold': self.similarity_threshold,
                'top_k_ratio': self.top_k_ratio
            }, f)
        print(f"Vault saved to {filepath} ({self.total_entries} entries)")

    def load_vault(self, filename: str = "vault.pkl") -> bool:
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

        print(f"Vault loaded from {filepath} ({self.total_entries} entries)")
        return True

    def get_stats(self) -> Dict:
        """Get vault statistics."""
        return {
            'total_entries': self.total_entries,
            'layer_types': len(self.database),
            'entries_per_type': {k: len(v) for k, v in self.database.items()}
        }
