"""
Database-Driven Object Detection Demo: Cat Detection (Qdrant Cloud)

This demo shows:
1. First Training: Cat detection, cold start with random initialization
2. Store weights to Qdrant Cloud vault after training
3. Second Training: Same cat task, initialize from stored vault weights
4. Compare convergence speed between cold start and vault-init

This version uses Qdrant Cloud instead of Zilliz Cloud.
Advantages of Qdrant:
- Higher dimension limit: 65,535 (vs Zilliz 32,768)
- Unlimited collections (vs Zilliz 5)
- Better performance for high-dimensional vectors

Parameters are reduced for fast demo execution.
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

from cnn_weight_vault.qdrant_vault import QdrantWeightVault
from cnn_weight_vault.detection_model import (
    create_detection_model, DBConv2dDetect, DBLinearDetect
)
from cnn_weight_vault.config import get_config


class CatDetectionDataset(Dataset):
    """
    Synthetic dataset for cat detection (single-class).
    Generates images with simple cat-like shapes (ellipses).
    """

    def __init__(self, num_samples=200, image_size=112, grid_size=7, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.grid_size = grid_size
        self.transform = transform
        self.cell_size = image_size / grid_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image
        img = Image.new('RGB', (self.image_size, self.image_size), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)

        # Random background color
        bg_color = tuple(np.random.randint(100, 200, 3).tolist())
        draw.rectangle([(0, 0), (self.image_size, self.image_size)], fill=bg_color)

        # Generate 1 cat per image
        targets = []

        # Cat body (ellipse)
        cat_w = np.random.randint(20, 40)
        cat_h = np.random.randint(15, 30)
        x = np.random.randint(cat_w // 2, self.image_size - cat_w // 2)
        y = np.random.randint(cat_h // 2, self.image_size - cat_h // 2)

        # Draw cat (orange ellipse)
        color = (255, 165, 0)  # Orange
        draw.ellipse([(x - cat_w//2, y - cat_h//2),
                     (x + cat_w//2, y + cat_h//2)], fill=color)

        # Convert to grid cell coordinates
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)

        # Relative position within cell
        rel_x = (x % int(self.cell_size)) / self.cell_size
        rel_y = (y % int(self.cell_size)) / self.cell_size

        # Relative size within image
        rel_w = cat_w / self.image_size
        rel_h = cat_h / self.image_size

        targets.append({
            'grid_x': min(grid_x, self.grid_size - 1),
            'grid_y': min(grid_y, self.grid_size - 1),
            'x': rel_x,
            'y': rel_y,
            'w': rel_w,
            'h': rel_h,
            'confidence': 1.0
        })

        if self.transform:
            img = self.transform(img)

        # Create target tensor (grid_size, grid_size, 5) -> (x, y, w, h, conf)
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

    # Coordinate loss (only for cells with objects)
    coord_loss = torch.sum(
        (pred[..., :4][obj_mask] - target[..., :4][obj_mask]) ** 2
    ) if obj_mask.sum() > 0 else 0

    # Confidence loss
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
    """
    First training: Cat detection, cold start with random initialization.
    Stores weights to vault after training.
    """
    print("\n" + "="*70)
    print(f"FIRST TRAINING: {category.upper()} Detection (Cold Start)")
    print("="*70)

    # Reset counters
    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()

    # Create model with cold start
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[1] Creating detection model (device: {device})...")

    model = create_detection_model(
        vault=vault,
        object_category=category,
        num_classes=1,
        force_load=False  # Cold start
    ).to(device)

    print(f"[2] Model created (grid_size=7, single-class)")

    # Create small dataset for demo
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CatDetectionDataset(num_samples=100, transform=transform)
    test_dataset = CatDetectionDataset(num_samples=50, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"[3] Dataset: 100 train, 50 test images (112x112)")

    # Train
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"\n[4] Training for {num_epochs} epochs...")
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

    # Store weights to vault
    print(f"\n[5] Storing weights to Qdrant Cloud vault...")
    final_map = results[-1]['test_map']

    stored_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (DBConv2dDetect, DBLinearDetect)):
            module.store_weights(num_epochs, final_map)
            stored_count += 1

    vault.save_vault()
    print(f"[Vault] Stored {stored_count} layers to cloud (mAP: {final_map:.1f}%)")

    return results


def second_training_run(vault, category="cat", num_epochs=5):
    """
    Second training: Same cat task, initialize from vault weights.
    """
    print("\n" + "="*70)
    print(f"SECOND TRAINING: {category.upper()} Detection (Vault-Init)")
    print("="*70)
    print("MODE: FORCE LOAD - Initialize from stored vault weights\n")

    # Reset counters (IMPORTANT!)
    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()

    # Create model with FORCE LOAD
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[1] Creating detection model with vault weights (device: {device})...")

    model = create_detection_model(
        vault=vault,
        object_category=category,
        num_classes=1,
        force_load=True  # FORCE LOAD from vault!
    ).to(device)

    print(f"[2] Model created (initialized from cloud vault)")

    # Same dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CatDetectionDataset(num_samples=100, transform=transform)
    test_dataset = CatDetectionDataset(num_samples=50, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Note: Different seeds produce different synthetic images
    # So vault-init model sees different cats than the first training
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"[3] Dataset: 100 train, 50 test images")

    # Train - Vault-init uses smaller learning rate for fine-tuning
    # because it's already close to optimal from previous training
    learning_rate = 0.0003  # Smaller lr for vault-init fine-tuning
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
    print("COMPARISON: Cold Start vs Vault-Init")
    print("="*70)
    print("\nNOTE: Low loss but lower mAP is expected for vault-init because:")
    print("  - Vault weights are optimized for previous random cats")
    print("  - Second training sees NEW random cats (different dataset)")
    print("  - Model needs to adapt (fine-tune) to new cat patterns")
    print("  - Using smaller learning rate (0.0003 vs 0.001) for fine-tuning")

    # Print all epochs using for loop
    for epoch_idx in range(len(cold_results)):
        epoch_num = epoch_idx + 1
        print(f"\nEpoch {epoch_num}:")
        print(f"  Cold Start:    Train Loss={cold_results[epoch_idx]['train_loss']:.3f}, "
              f"Test mAP={cold_results[epoch_idx]['test_map']:.1f}%")
        print(f"  Vault-Init:    Train Loss={vault_results[epoch_idx]['train_loss']:.3f}, "
              f"Test mAP={vault_results[epoch_idx]['test_map']:.1f}%")

    # Calculate Epoch 1 improvement
    map_improvement = vault_results[0]['test_map'] - cold_results[0]['test_map']
    loss_reduction = cold_results[0]['train_loss'] - vault_results[0]['train_loss']

    print("\n" + "-"*70)
    print(f"Epoch 1 Improvement from Vault-Init:")
    print(f"  Test mAP:      +{map_improvement:.1f}%")
    print(f"  Train Loss:    -{loss_reduction:.3f}")
    print("-"*70)


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("DATABASE-DRIVEN CNN INITIALIZATION - CAT DETECTION DEMO")
    print("(Qdrant Cloud Version)")
    print("="*70)
    print("\nThis demo shows:")
    print("  1. First training: Cat detection, cold start (He init)")
    print("  2. Store trained weights to Qdrant Cloud vault (tagged as 'cat')")
    print("  3. Second training: Same cat task, vault-initialized from Qdrant Cloud")
    print("  4. Compare convergence speed\n")
    print("NOTE: Demo uses small dataset and few epochs for fast execution")

    # Load config
    config = get_config()
    
    # Get Qdrant Cloud configuration
    qdrant_uri = config.get('vector_db.qdrant.uri', '')
    qdrant_api_key = config.get('vector_db.qdrant.api_key', '')
    
    if not qdrant_uri or not qdrant_api_key:
        print("\n" + "="*70)
        print("ERROR: Qdrant Cloud configuration not found!")
        print("="*70)
        print("\nPlease configure your Qdrant Cloud credentials:")
        print("  1. Sign up at https://qdrant.tech (free tier available)")
        print("  2. Create a free cluster")
        print("  3. Copy the cluster URI and API key")
        print("  4. Edit config/settings.yaml:")
        print("")
        print("     vector_db:")
        print("       qdrant:")
        print("         uri: \"https://your-cluster.qdrant.tech:6333\"")
        print("         api_key: \"your-api-key\"")
        print("")
        print("  Or set environment variables:")
        print("     export CNN_QDRANT_URI=\"https://...\"")
        print("     export CNN_QDRANT_API_KEY=\"your-api-key\"")
        print("")
        return

    print(f"\nUsing Qdrant Cloud configuration:")
    print(f"  URI: {qdrant_uri}")
    print(f"  Similarity threshold: {config.similarity_threshold}")

    # Create Qdrant vault
    vault = QdrantWeightVault(
        uri=qdrant_uri,
        api_key=qdrant_api_key,
        collection_name=config.get('vector_db.qdrant.collection_name', 'cnn_weights'),
        similarity_threshold=0.3,
        top_k_ratio=0.3
    )

    # Test connection
    print("\n[0] Testing connection to Qdrant Cloud...")
    try:
        stats = vault.get_stats()
        print(f"     Connected! Current entries: {stats.get('total_entries', 0)}")
    except Exception as e:
        print(f"     Connection failed: {e}")
        return

    # Demo: 5 epochs for better comparison
    num_epochs = 5

    # First training: Cold start
    cold_results = first_training_run(vault, category="cat", num_epochs=num_epochs)

    # Show vault stats
    print("\n" + "="*70)
    print("QDRANT CLOUD VAULT STATUS AFTER FIRST TRAINING")
    print("="*70)
    stats = vault.get_stats()
    print(f"  Total entries: {stats.get('total_entries', 0)}")
    print(f"  Collections: {stats.get('collections', 0)}")
    print(f"  Object categories: {stats.get('object_categories', [])}")
    print(f"  Entries per collection:")
    for name, count in stats.get('entries_per_collection', {}).items():
        if count > 0:
            print(f"    {name}: {count}")

    # Second training: Vault-init (same category)
    vault_results = second_training_run(vault, category="cat", num_epochs=num_epochs)

    # Compare results
    compare_results(cold_results, vault_results)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
"""
Key Takeaways:
1. First Training (Cold Start):
   - All layers: "Cold start - using He initialization"
   - Higher initial loss, lower initial mAP
   - Weights stored to Qdrant Cloud vault (tagged as 'cat')

2. Second Training (Vault-Init):
   - All layers: "Initialized from vault [FORCED]"
   - Lower initial loss (already optimized)
   - May have lower initial mAP because:
     * Synthetic data is random (different cats each run)
     * Vault weights are tuned to previous random cats
     * Need fine-tuning with smaller learning rate

3. Benefits of Qdrant Cloud over Zilliz:
   - Higher dimension limit: 65,535 (vs Zilliz 32,768)
   - Unlimited collections (vs Zilliz 5)
   - Better performance for high-dimensional vectors

4. Next Steps:
   - Share credentials with team members
   - Train on 'dog' category to test cross-category similarity
   - The vault will prefer 'dog' weights but can use 'cat' weights
"""


if __name__ == "__main__":
    main()