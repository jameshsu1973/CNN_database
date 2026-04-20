#!/usr/bin/env python3
"""
Database-Driven Object Detection Demo using ChromaDB

This demo is identical to demo_cat_detection_vault.py but uses ChromaDB
instead of pickle-based storage.

Shows:
1. First Training: Cat detection with ChromaDB vault
2. Store weights to ChromaDB after training
3. Second Training: Initialize from ChromaDB vault weights
4. Compare convergence speed
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Use ChromaDB vault instead of DetectionWeightVault
from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.detection_model import (
    create_detection_model, DBConv2dDetect, DBLinearDetect
)
from cnn_weight_vault.config import get_config


class CatDetectionDataset(Dataset):
    """Synthetic dataset for cat detection (single-class)."""

    def __init__(self, num_samples=200, image_size=112, grid_size=7, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.grid_size = grid_size
        self.transform = transform
        self.cell_size = image_size / grid_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.new('RGB', (self.image_size, self.image_size), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)

        bg_color = tuple(np.random.randint(100, 200, 3).tolist())
        draw.rectangle([(0, 0), (self.image_size, self.image_size)], fill=bg_color)

        targets = []
        cat_w = np.random.randint(20, 40)
        cat_h = np.random.randint(15, 30)
        x = np.random.randint(cat_w // 2, self.image_size - cat_w // 2)
        y = np.random.randint(cat_h // 2, self.image_size - cat_h // 2)

        color = (255, 165, 0)
        draw.ellipse([(x - cat_w//2, y - cat_h//2),
                     (x + cat_w//2, y + cat_h//2)], fill=color)

        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        rel_x = (x % int(self.cell_size)) / self.cell_size
        rel_y = (y % int(self.cell_size)) / self.cell_size
        rel_w = cat_w / self.image_size
        rel_h = cat_h / self.image_size

        targets.append({
            'grid_x': min(grid_x, self.grid_size - 1),
            'grid_y': min(grid_y, self.grid_size - 1),
            'x': rel_x, 'y': rel_y, 'w': rel_w, 'h': rel_h, 'confidence': 1.0
        })

        if self.transform:
            img = self.transform(img)

        target_tensor = torch.zeros(self.grid_size, self.grid_size, 5)
        for t in targets:
            gx, gy = t['grid_x'], t['grid_y']
            if target_tensor[gy, gx, 4] == 0:
                target_tensor[gy, gx] = torch.tensor([
                    t['x'], t['y'], t['w'], t['h'], t['confidence']
                ])

        return img, target_tensor


def detection_loss(pred, target):
    """YOLO-style detection loss (simplified)."""
    obj_mask = target[..., 4] > 0.5
    noobj_mask = ~obj_mask

    coord_loss = torch.sum(
        (pred[..., :4][obj_mask] - target[..., :4][obj_mask]) ** 2
    ) if obj_mask.sum() > 0 else 0

    obj_conf_loss = torch.sum((pred[..., 4][obj_mask] - target[..., 4][obj_mask]) ** 2) if obj_mask.sum() > 0 else 0
    noobj_conf_loss = 0.3 * torch.sum((pred[..., 4][noobj_mask] - target[..., 4][noobj_mask]) ** 2)

    total_loss = coord_loss + obj_conf_loss + noobj_conf_loss
    return total_loss / (pred.size(0) + 1e-6)


def calculate_map(pred, target, iou_threshold=0.5):
    """Calculate simplified mAP."""
    obj_mask = target[..., 4] > 0.5
    if obj_mask.sum() == 0:
        return 0.0

    pred_boxes = pred[..., :4][obj_mask]
    target_boxes = target[..., :4][obj_mask]

    ious = []
    for p, t in zip(pred_boxes, target_boxes):
        iou = 1.0 - torch.sum((p - t) ** 2).item()
        ious.append(max(0, iou))

    return sum(ious) / len(ious) * 100 if ious else 0.0


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_map = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = detection_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_map += calculate_map(output, target)

    avg_loss = total_loss / len(train_loader)
    avg_map = total_map / len(train_loader)
    return avg_loss, avg_map


def evaluate(model, test_loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_map = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = detection_loss(output, target)
            total_loss += loss.item()
            total_map += calculate_map(output, target)

    avg_loss = total_loss / len(test_loader)
    avg_map = total_map / len(test_loader)
    return avg_loss, avg_map


def first_training_run(vault, category="cat", num_epochs=5):
    """First training: Cat detection with ChromaDB vault."""
    print("\n" + "="*70)
    print(f"FIRST TRAINING: {category.upper()} Detection (ChromaDB - Cold Start)")
    print("="*70)

    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[1] Creating detection model with ChromaDB vault (device: {device})...")

    model = create_detection_model(
        vault=vault,
        object_category=category,
        num_classes=1,
        force_load=False
    ).to(device)

    print(f"[2] Model created using ChromaDB vault")
    print(f"[3] ChromaDB persist directory: {vault.persist_directory}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CatDetectionDataset(num_samples=100, transform=transform)
    test_dataset = CatDetectionDataset(num_samples=50, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"[4] Dataset: 100 train, 50 test images (112x112)")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"\n[5] Training for {num_epochs} epochs...")
    print("-" * 50)

    results = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_map = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_map = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.3f}, Train mAP={train_map:.1f}%, "
              f"Test mAP={test_map:.1f}%")
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_map': train_map,
            'test_map': test_map
        })

    print(f"\n[6] Storing weights to ChromaDB vault...")
    final_map = results[-1]['test_map']

    stored_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (DBConv2dDetect, DBLinearDetect)):
            module.store_weights(num_epochs, final_map)
            stored_count += 1

    vault.save_vault()
    print(f"[Vault] Stored {stored_count} layers to ChromaDB (mAP: {final_map:.1f}%)")

    return results


def second_training_run(vault, category="cat", num_epochs=5):
    """Second training: Initialize from ChromaDB vault weights."""
    print("\n" + "="*70)
    print(f"SECOND TRAINING: {category.upper()} Detection (ChromaDB - Vault-Init)")
    print("="*70)
    print("MODE: FORCE LOAD - Initialize from stored ChromaDB vault weights\n")

    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[1] Creating detection model with ChromaDB vault weights (device: {device})...")

    model = create_detection_model(
        vault=vault,
        object_category=category,
        num_classes=1,
        force_load=True
    ).to(device)

    print(f"[2] Model created (initialized from ChromaDB vault)")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CatDetectionDataset(num_samples=100, transform=transform)
    test_dataset = CatDetectionDataset(num_samples=50, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"[3] Dataset: 100 train, 50 test images")

    learning_rate = 0.0003
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"\n[4] Training for {num_epochs} epochs (lr={learning_rate}, fine-tuning mode)...")
    print("-" * 50)

    results = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_map = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_map = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.3f}, Train mAP={train_map:.1f}%, "
              f"Test mAP={test_map:.1f}%")
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_map': train_map,
            'test_map': test_map
        })

    return results


def compare_results(cold_results, vault_results):
    """Compare cold start vs vault initialization."""
    print("\n" + "="*70)
    print("COMPARISON: Cold Start vs ChromaDB Vault-Init")
    print("="*70)

    for epoch_idx in range(len(cold_results)):
        epoch_num = epoch_idx + 1
        print(f"\nEpoch {epoch_num}:")
        print(f"  Cold Start:    Train Loss={cold_results[epoch_idx]['train_loss']:.3f}, "
              f"Test mAP={cold_results[epoch_idx]['test_map']:.1f}%")
        print(f"  Vault-Init:    Train Loss={vault_results[epoch_idx]['train_loss']:.3f}, "
              f"Test mAP={vault_results[epoch_idx]['test_map']:.1f}%")

    map_improvement = vault_results[0]['test_map'] - cold_results[0]['test_map']
    loss_reduction = cold_results[0]['train_loss'] - vault_results[0]['train_loss']

    print("\n" + "-"*70)
    print(f"Epoch 1 Improvement from ChromaDB Vault-Init:")
    print(f"  Test mAP:      +{map_improvement:.1f}%")
    print(f"  Train Loss:    -{loss_reduction:.3f}")
    print("-"*70)


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("DATABASE-DRIVEN CNN INITIALIZATION - CHROMADB VERSION")
    print("="*70)
    print("\nThis demo uses ChromaDB vector database:")
    print("  - HNSW indexing for O(log n) similarity search")
    print("  - Cosine similarity matching")
    print("  - Automatic persistence (DuckDB + Parquet)")
    print("  - Scalable to millions of vectors")

    # Show config
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  Persist directory: {config.chroma_persist_dir}")
    print(f"  Collection name: {config.chroma_collection_name}")
    print(f"  Similarity threshold: {config.similarity_threshold}")
    print(f"  Top-k ratio: {config.top_k_ratio}")

    # Create ChromaDB vault
    print("\n" + "="*70)
    print("Creating ChromaDB vault...")
    print("="*70)

    vault = ChromaWeightVault(
        collection_name=config.chroma_collection_name,
        persist_directory="./chroma_db_cat_detection",
        similarity_threshold=0.3,
        top_k_ratio=0.3
    )

    num_epochs = 5

    # First training
    cold_results = first_training_run(vault, category="cat", num_epochs=num_epochs)

    # Show vault stats
    print("\n" + "="*70)
    print("CHROMADB VAULT STATUS AFTER FIRST TRAINING")
    print("="*70)
    stats = vault.get_stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Collections: {stats['collections']}")
    print(f"  Entries per collection:")
    for name, count in stats['entries_per_collection'].items():
        if count > 0:
            print(f"    {name}: {count}")

    # Second training
    vault_results = second_training_run(vault, category="cat", num_epochs=num_epochs)

    # Compare
    compare_results(cold_results, vault_results)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Takeaways:
1. ChromaDB Integration:
   - Uses HNSW graph structure for fast similarity search
   - Automatic persistence to disk
   - Supports metadata filtering (e.g., by object category)

2. Same Functionality:
   - Works identically to pickle-based DetectionWeightVault
   - DBConv2dDetect/DBLinearDetect compatible with both vault types

3. Performance Benefits:
   - O(log n) search vs O(n) linear scan
   - Better scalability for large datasets
   - ACID transactions via DuckDB backend

Files created:
  - ./chroma_db_cat_detection/: ChromaDB persistent storage
    - chroma.sqlite3 (metadata)
    - *.parquet (vector data)
    - HNSW indexes

Migration:
  - Existing pickle vaults can be migrated using:
    python scripts/migrate_to_chroma.py
    """)


if __name__ == "__main__":
    main()
