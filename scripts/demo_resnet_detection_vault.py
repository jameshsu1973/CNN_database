"""
Database-Driven Object Detection Demo: ResNet-18 Backbone

This demo shows:
1. First Training: Use real ResNet-18 backbone + simple detection head
2. Store ResNet weights to vault after training
3. Second Training: Same task, initialize ResNet from stored vault weights
4. Compare convergence

Uses ResNet-18 (not ResNet-50) for faster training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cnn_weight_vault.detection_vault import DetectionWeightVault
from cnn_weight_vault.detection_model import (
    DBConv2dDetect, DBLinearDetect, convert_to_db_layers
)


class SimpleDetectionHead(nn.Module):
    """
    Simple YOLO-like detection head.
    Uses standard nn.Conv2d (NOT DB-aware) - detection head is task-specific
    and should be retrained for each new task.
    """
    def __init__(self, in_channels=512, grid_size=7):
        super().__init__()
        self.grid_size = grid_size

        # Standard Conv2d - NOT converted to DB-aware
        # Detection head is task-specific, retrain from scratch each time
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 5, kernel_size=1)

    def forward(self, x):
        # x: (batch, channels, H, W)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        # Resize to grid_size x grid_size
        x = nn.functional.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))

        # Permute to (batch, grid, grid, 5)
        x = x.permute(0, 2, 3, 1)

        # Sigmoid for all outputs
        x = torch.sigmoid(x)

        return x


class ResNetDetectionModel(nn.Module):
    """
    ResNet-18 backbone + Simple detection head.
    """
    def __init__(self, grid_size=7, pretrained=True):
        super().__init__()

        # Load ResNet-18 backbone (without final FC layer)
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Use layers up to layer4 (before avgpool)
        self.backbone = nn.Sequential(
            resnet.conv1,      # 224x224 -> 112x112
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # 112x112 -> 56x56
            resnet.layer1,     # 56x56
            resnet.layer2,     # 56x56 -> 28x28
            resnet.layer3,     # 28x28 -> 14x14
            resnet.layer4,     # 14x14 -> 7x7
        )

        # Detection head
        self.detection_head = SimpleDetectionHead(in_channels=512, grid_size=grid_size)

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)  # (batch, 512, 7, 7)

        # Detection head
        output = self.detection_head(features)

        return output


class SimpleObjectDataset(Dataset):
    """Synthetic dataset for object detection."""

    def __init__(self, num_samples=200, image_size=224, grid_size=7, transform=None, seed=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.grid_size = grid_size
        self.transform = transform
        self.cell_size = image_size / grid_size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use seed for reproducibility
        img_seed = self.rng.randint(0, 100000)
        np.random.seed(img_seed)

        # Create image
        img = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)

        # Random background
        bg_color = tuple(np.random.randint(50, 200, 3).tolist())
        draw.rectangle([(0, 0), (self.image_size, self.image_size)], fill=bg_color)

        # Draw object (red rectangle)
        obj_w = np.random.randint(40, 80)
        obj_h = np.random.randint(40, 80)
        x = np.random.randint(obj_w // 2, self.image_size - obj_w // 2)
        y = np.random.randint(obj_h // 2, self.image_size - obj_h // 2)

        color = (255, 0, 0)
        draw.rectangle([(x - obj_w//2, y - obj_h//2),
                       (x + obj_w//2, y + obj_h//2)], fill=color)

        # Grid coordinates
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)

        rel_x = (x % int(self.cell_size)) / self.cell_size
        rel_y = (y % int(self.cell_size)) / self.cell_size
        rel_w = obj_w / self.image_size
        rel_h = obj_h / self.image_size

        if self.transform:
            img = self.transform(img)

        # Target tensor
        target = torch.zeros(self.grid_size, self.grid_size, 5)
        target[min(grid_y, self.grid_size-1), min(grid_x, self.grid_size-1)] = \
            torch.tensor([rel_x, rel_y, rel_w, rel_h, 1.0])

        return img, target


def detection_loss(pred, target):
    """YOLO-style loss."""
    obj_mask = target[..., 4] > 0.5
    noobj_mask = ~obj_mask

    coord_loss = torch.sum((pred[..., :4][obj_mask] - target[..., :4][obj_mask]) ** 2) if obj_mask.sum() > 0 else 0
    obj_conf_loss = torch.sum((pred[..., 4][obj_mask] - target[..., 4][obj_mask]) ** 2) if obj_mask.sum() > 0 else 0
    noobj_conf_loss = 0.5 * torch.sum((pred[..., 4][noobj_mask] - target[..., 4][noobj_mask]) ** 2)

    return (coord_loss + obj_conf_loss + noobj_conf_loss) / (pred.size(0) + 1e-6)


def calculate_map(pred, target):
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

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = detection_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_map += calculate_map(output, target)

    return total_loss / len(train_loader), total_map / len(train_loader)


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

    return total_loss / len(test_loader), total_map / len(test_loader)


def first_training_run(vault, category="object", num_epochs=3):
    """First training: Cold start with ResNet-18 (no pretrained weights)."""
    print("\n" + "="*70)
    print(f"FIRST TRAINING: ResNet-18 Detection (Cold Start)")
    print("Using random initialization (pretrained=False)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[1] Creating ResNet-18 model (device: {device})...")

    # Setup vault BEFORE creating model (so detection head also uses DB-aware layers)
    DBConv2dDetect.reset_layer_counter()
    DBConv2dDetect.set_vault(vault)
    DBConv2dDetect.set_object_category(category)
    DBConv2dDetect.set_force_load(False)

    # Create model WITHOUT pretrained weights for fair comparison
    model = ResNetDetectionModel(grid_size=7, pretrained=False)

    print(f"[2] Converting backbone Conv2d layers to DB-aware...")
    # Only convert backbone, NOT detection head
    model.backbone = convert_to_db_layers(model.backbone, vault=vault, object_category=category, force_load=False)

    model = model.to(device)

    # Count DB layers
    db_conv = sum(1 for m in model.modules() if isinstance(m, DBConv2dDetect))
    db_linear = sum(1 for m in model.modules() if isinstance(m, DBLinearDetect))
    print(f"[3] Model has {db_conv} DBConv2d + {db_linear} DBLinear layers")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SimpleObjectDataset(num_samples=150, transform=transform, seed=42)
    test_dataset = SimpleObjectDataset(num_samples=50, transform=transform, seed=999)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"[4] Dataset: 150 train, 50 test images (224x224)")

    # Train
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"\n[5] Training for {num_epochs} epochs...")
    print("-" * 50)

    results = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_map = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_map = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: Loss={train_loss:.3f}, Train mAP={train_map:.1f}%, Test mAP={test_map:.1f}%")
        results.append({'epoch': epoch, 'train_loss': train_loss, 'test_map': test_map})

    # Store weights - only store backbone, NOT detection head
    print(f"\n[6] Storing backbone weights to vault...")
    final_map = results[-1]['test_map']
    stored_count = 0
    for name, module in model.backbone.named_modules():
        if isinstance(module, (DBConv2dDetect, DBLinearDetect)):
            module.store_weights(num_epochs, final_map)
            stored_count += 1

    vault.save_vault()
    print(f"[Vault] Stored {stored_count} backbone layers (final mAP: {final_map:.1f}%)")

    return results


def second_training_run(vault, category="object", num_epochs=3):
    """Second training: Vault-initialized ResNet-18."""
    print("\n" + "="*70)
    print(f"SECOND TRAINING: ResNet-18 Detection (Vault-Init)")
    print("="*70)
    print("MODE: FORCE LOAD from vault\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[1] Creating ResNet-18 model with vault weights...")

    # Setup vault BEFORE creating model (so detection head also loads from vault)
    DBConv2dDetect.reset_layer_counter()
    DBConv2dDetect.set_vault(vault)
    DBConv2dDetect.set_object_category(category)
    DBConv2dDetect.set_force_load(True)

    # Create model - will load only BACKBONE weights from vault
    model = ResNetDetectionModel(grid_size=7, pretrained=False)
    # Only convert backbone, NOT detection head (head is task-specific)
    model.backbone = convert_to_db_layers(model.backbone, vault=vault, object_category=category, force_load=True)
    model = model.to(device)

    print(f"[2] Model created (backbone from vault, head re-initialized)")

    # Same dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SimpleObjectDataset(num_samples=150, transform=transform, seed=42)
    test_dataset = SimpleObjectDataset(num_samples=50, transform=transform, seed=999)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"[3] Dataset: 150 train, 50 test images")

    # Use smaller lr for vault-init: backbone needs fine-tuning, head needs training
    # Backbone is already optimized, so use smaller lr (0.0001)
    # Head is new, so use normal lr (0.001)
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 0.0001},  # Fine-tune backbone
        {'params': model.detection_head.parameters(), 'lr': 0.001}  # Train head
    ])
    print(f"\n[4] Training for {num_epochs} epochs (backbone lr=0.0001, head lr=0.001)...")
    print("-" * 50)

    results = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_map = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_map = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: Loss={train_loss:.3f}, Train mAP={train_map:.1f}%, Test mAP={test_map:.1f}%")
        results.append({'epoch': epoch, 'train_loss': train_loss, 'test_map': test_map})

    return results


def compare_results(cold_results, vault_results):
    """Compare results."""
    print("\n" + "="*70)
    print("COMPARISON: Cold Start vs Vault-Init (ResNet-18)")
    print("="*70)

    for i in range(len(cold_results)):
        epoch = i + 1
        print(f"\nEpoch {epoch}:")
        print(f"  Cold Start:  Loss={cold_results[i]['train_loss']:.3f}, mAP={cold_results[i]['test_map']:.1f}%")
        print(f"  Vault-Init:  Loss={vault_results[i]['train_loss']:.3f}, mAP={vault_results[i]['test_map']:.1f}%")

    # Final improvement
    final_cold = cold_results[-1]['test_map']
    final_vault = vault_results[-1]['test_map']
    improvement = final_vault - final_cold

    print("\n" + "-"*70)
    print(f"Final Epoch Improvement:")
    print(f"  Test mAP: {final_vault:.1f}% vs {final_cold:.1f}% ({improvement:+.1f}%)")
    print("-"*70)


def main():
    """Main demo."""
    print("\n" + "="*70)
    print("DATABASE-DRIVEN CNN - RESNET-18 DETECTION DEMO")
    print("="*70)
    print("\nThis demo uses REAL ResNet-18 backbone:")
    print("  1. First: Train ResNet-18 + detection head (cold start)")
    print("  2. Store ResNet Conv/Linear weights to vault")
    print("  3. Second: New ResNet-18, init backbone from vault")
    print("  4. Compare convergence")

    # Create vault
    vault = DetectionWeightVault(
        similarity_threshold=0.3,
        top_k_ratio=0.3,
        vault_path="../resnet_detection_vault"
    )

    num_epochs = 3  # Quick demo

    # First training
    cold_results = first_training_run(vault, category="object", num_epochs=num_epochs)

    # Show vault status
    print("\n" + "="*70)
    print("VAULT STATUS")
    print("="*70)
    stats = vault.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Second training
    vault_results = second_training_run(vault, category="object", num_epochs=num_epochs)

    # Compare
    compare_results(cold_results, vault_results)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
"""
Key Points:
1. NO pretrained weights - both start from random initialization
2. Different learning rates for vault-init (backbone: 0.0001, head: 0.001)
3. Only BACKBONE stored/loaded from vault (detection head is re-initialized)
4. First training: Backbone "Cold start", Head "Cold start"
5. Second training: Backbone "Vault-Init" (fine-tune), Head "Cold start" (train)

Why different learning rates?
  - Vault-init backbone is already optimized → needs fine-tuning (small lr)
  - New detection head needs training from scratch (normal lr)

Why only store backbone?
  - Backbone learns generic features (edges, textures, shapes)
  - Detection head is task-specific → re-initialize each time

This is the correct transfer learning design!

Files created:
  - ../resnet_detection_vault/: Contains ResNet-18 backbone weights only
"""


if __name__ == "__main__":
    main()
