#!/usr/bin/env python3
"""
Demo script showing ChromaDB vault usage
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.config import get_config


class SimpleNet(nn.Module):
    """Simple CNN for demo."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    print("ChromaDB Vault Demo")
    print("=" * 50)

    # Show config
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  Persist directory: {config.chroma_persist_dir}")
    print(f"  Collection name: {config.chroma_collection_name}")
    print(f"  Similarity threshold: {config.similarity_threshold}")
    print(f"  Top-k ratio: {config.top_k_ratio}")

    # Create vault
    print("\n" + "=" * 50)
    print("Creating ChromaDB vault...")
    vault = ChromaWeightVault()

    # Create a model
    print("\nCreating model...")
    model = SimpleNet()

    # Store weights
    print("\n" + "=" * 50)
    print("Storing weights from model...")

    epoch = 10
    accuracy = 85.5
    model_name = "demo_model"

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            vault.store_weights(
                layer=layer,
                layer_name=name,
                model_name=model_name,
                epoch=epoch,
                accuracy=accuracy
            )
            print(f"  Stored: {name} ({type(layer).__name__})")

    # Get stats
    print("\n" + "=" * 50)
    stats = vault.get_stats()
    print(f"Vault stats:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Collections: {stats['collections']}")
    print(f"  Entries per collection:")
    for name, count in stats['entries_per_collection'].items():
        print(f"    {name}: {count}")

    # Create new model and try to initialize from vault
    print("\n" + "=" * 50)
    print("Creating new model and querying vault for initialization...")

    new_model = SimpleNet()

    for name, layer in new_model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            weights = vault.get_initialization_weights(layer, layer_name=name)
            if weights is not None:
                print(f"  [OK] {name}: Found matching weights in vault")
                layer.weight.data = weights
            else:
                print(f"  [--] {name}: No match, using default init")

    # Save vault
    print("\n" + "=" * 50)
    print("Saving vault...")
    vault.save_vault()

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
