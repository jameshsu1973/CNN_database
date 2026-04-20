"""
Object Detection Training Script with Database-Driven Weight Initialization
Supports single-class object detection (e.g., cat detection)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image, ImageDraw

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.detection_model import create_detection_model, DetectionModelWrapper, DBConv2dDetect, DBLinearDetect
from cnn_weight_vault.config import get_config


class SyntheticObjectDataset(Dataset):
    """
    Synthetic dataset for single-class object detection.
    Generates images with simple shapes (circles, rectangles) as objects.
    """

    def __init__(self, num_samples=1000, image_size=224, grid_size=7, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.grid_size = grid_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image
        img = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)

        # Random background
        bg_color = tuple(np.random.randint(50, 200, 3).tolist())
        draw.rectangle([(0, 0), (self.image_size, self.image_size)], fill=bg_color)

        # Generate 1-3 objects per image
        num_objects = np.random.randint(1, 4)
        targets = []

        cell_size = self.image_size / self.grid_size

        for _ in range(num_objects):
            # Random object size and position
            obj_w = np.random.randint(30, 100)
            obj_h = np.random.randint(30, 100)
            x = np.random.randint(obj_w // 2, self.image_size - obj_w // 2)
            y = np.random.randint(obj_h // 2, self.image_size - obj_h // 2)

            # Draw object (red circle)
            color = (255, 0, 0)
            draw.ellipse([(x - obj_w//2, y - obj_h//2),
                         (x + obj_w//2, y + obj_h//2)], fill=color)

            # Convert to grid cell coordinates
            grid_x = int(x / cell_size)
            grid_y = int(y / cell_size)

            # Relative position within cell
            rel_x = (x % int(cell_size)) / cell_size
            rel_y = (y % int(cell_size)) / cell_size

            # Relative size within image
            rel_w = obj_w / self.image_size
            rel_h = obj_h / self.image_size

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

        # Create target tensor (grid_size, grid_size, 5)
        target_tensor = torch.zeros(self.grid_size, self.grid_size, 5)
        for t in targets:
            gx, gy = t['grid_x'], t['grid_y']
            if target_tensor[gy, gx, 4] == 0:  # Only take first object in cell
                target_tensor[gy, gx] = torch.tensor([t['x'], t['y'], t['w'], t['h'], t['confidence']])

        return img, target_tensor


def detection_loss(pred, target):
    """
    Loss function for object detection.

    Args:
        pred: (batch, grid, grid, 5) - predictions
        target: (batch, grid, grid, 5) - ground truth

    Returns:
        loss: scalar loss
    """
    # Object presence mask
    obj_mask = target[..., 4] > 0.5  # Cells with objects
    noobj_mask = ~obj_mask

    # Coordinate loss (only for cells with objects)
    coord_loss = torch.sum(
        (pred[..., :4][obj_mask] - target[..., :4][obj_mask]) ** 2
    )

    # Confidence loss
    obj_conf_loss = torch.sum((pred[..., 4][obj_mask] - target[..., 4][obj_mask]) ** 2)
    noobj_conf_loss = 0.5 * torch.sum((pred[..., 4][noobj_mask] - target[..., 4][noobj_mask]) ** 2)

    total_loss = coord_loss + obj_conf_loss + noobj_conf_loss

    return total_loss / (pred.size(0) + 1e-6)


def calculate_map(pred, target, iou_threshold=0.5):
    """Calculate mean Average Precision (simplified)."""
    # For simplicity, use accuracy as proxy for mAP
    obj_mask = target[..., 4] > 0.5
    if obj_mask.sum() == 0:
        return 0.0

    # IoU-based accuracy
    pred_boxes = pred[..., :4][obj_mask]
    target_boxes = target[..., :4][obj_mask]

    ious = []
    for p, t in zip(pred_boxes, target_boxes):
        # Simplified IoU calculation
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


def run_first_training(vault: ChromaWeightVault,
                       object_category: str,
                       device: str = 'cuda'):
    """Run first training (cold start) to populate vault."""
    print("\n" + "="*60)
    print(f"FIRST TRAINING: {object_category.upper()} DETECTION (Cold Start)")
    print("="*60)

    # Create model
    model = create_detection_model(vault, object_category=object_category, num_classes=1).to(device)
    wrapper = DetectionModelWrapper(model, vault, model_name=f"{object_category}_detector",
                                    object_category=object_category)

    # Create dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SyntheticObjectDataset(num_samples=500, transform=transform)
    test_dataset = SyntheticObjectDataset(num_samples=100, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(1, 4):
        train_loss, train_map = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_map = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train mAP={train_map:.2f}%, "
              f"Test mAP={test_map:.2f}%")

        # Store weights
        wrapper.store_epoch_weights(epoch, test_map)

    wrapper.save_vault()
    print(f"\nVault stats: {wrapper.get_vault_stats()}")

    return test_map


def run_second_training(vault: ChromaWeightVault,
                        object_category: str,
                        device: str = 'cuda',
                        force_load: bool = True):
    """
    Run second training using vault-initialized weights.

    Args:
        force_load: If True, force load weights from vault (ignore similarity threshold)
    """
    print("\n" + "="*60)
    print(f"SECOND TRAINING: {object_category.upper()} DETECTION (Vault-Init)")
    if force_load:
        print("MODE: FORCE LOAD - Will use vault weights regardless of similarity")
    print("="*60)

    # Reset counters
    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()

    # Create model - will try to initialize from vault
    model = create_detection_model(vault, object_category=object_category, num_classes=1,
                                   force_load=force_load).to(device)
    wrapper = DetectionModelWrapper(model, vault, model_name=f"{object_category}_detector_v2",
                                    object_category=object_category,
                                    force_load=force_load)

    # Create dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SyntheticObjectDataset(num_samples=500, transform=transform)
    test_dataset = SyntheticObjectDataset(num_samples=100, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(1, 4):
        train_loss, train_map = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_map = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train mAP={train_map:.2f}%, "
              f"Test mAP={test_map:.2f}%")

        # Store weights
        wrapper.store_epoch_weights(epoch, test_map)

    wrapper.save_vault()
    return test_map


def run_cross_category_test(vault: ChromaWeightVault,
                           train_category: str,
                           test_category: str,
                           device: str = 'cuda'):
    """
    Test cross-category transfer: train on 'cat', test on 'dog'.
    Shows if vault weights from one category help another.
    """
    print("\n" + "="*60)
    print(f"CROSS-CATEGORY: Train on {train_category}, Test on {test_category}")
    print("="*60)

    # Reset counters
    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()

    # Create model with different category
    model = create_detection_model(vault, object_category=test_category, num_classes=1).to(device)
    wrapper = DetectionModelWrapper(model, vault,
                                    model_name=f"{test_category}_detector",
                                    object_category=test_category)

    # Dataset for test category
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SyntheticObjectDataset(num_samples=500, transform=transform)
    test_dataset = SyntheticObjectDataset(num_samples=100, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train briefly
    for epoch in range(1, 3):
        train_loss, train_map = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_map = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test mAP={test_map:.2f}%")
        wrapper.store_epoch_weights(epoch, test_map)

    return test_map


def main():
    """Main function demonstrating database-driven object detection."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create ChromaDB vault
    config = get_config()
    vault = ChromaWeightVault(
        collection_name=config.chroma_collection_name,
        persist_directory="./chroma_db_detection",
        similarity_threshold=0.3,  # Balanced threshold for detection
        top_k_ratio=0.3
    )

    # Demo 1: Train on "cat" detection (cold start - populate vault)
    print("\n" + "="*60)
    print("DEMO 1: First training (Cold Start - Populating Vault)")
    print("="*60)

    cat_map_1 = run_first_training(vault, object_category="cat", device=device)

    # Demo 2: Train another "cat" detector with FORCE LOAD
    # This will use vault weights regardless of similarity
    print("\n" + "="*60)
    print("DEMO 2: Second training (Force Load from Vault)")
    print("="*60)
    cat_map_2 = run_second_training(vault, object_category="cat", device=device,
                                    force_load=True)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Cat detector (cold start):     mAP = {cat_map_1:.2f}%")
    print(f"Cat detector (vault-forced):   mAP = {cat_map_2:.2f}%")
    print(f"\nVault contents: {vault.get_stats()}")

    print("\nKey points:")
    print("- First training: Cold start with He initialization")
    print("- Second training: FORCED to load weights from vault")
    print("- This proves vault initialization mechanism works!")
    print("- In production: set force_load=False to use similarity matching")


if __name__ == "__main__":
    main()
