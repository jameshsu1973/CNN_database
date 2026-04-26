"""
Database-Driven CNN Initialization Demo: Binary Classification (Qdrant Cloud)

Binary classification: cat (1) vs not-cat (0)

This demo shows:
1. First Training: Binary classification, cold start with He initialization
2. Store weights to Qdrant Cloud vault after training
3. Second Training: Same task, initialize from stored vault weights
4. Compare convergence speed between cold start and vault-init
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cnn_weight_vault.qdrant_vault import QdrantWeightVault
from cnn_weight_vault.db_initialization import (
    create_db_cnn, DBConv2d, DBLinear
)
from cnn_weight_vault.config import get_config


def create_db_cnn_from_vault(vault_weights: dict, num_classes: int = 1) -> nn.Module:
    """
    Create CNN model directly from vault weights (no He init).
    Uses standard nn.Module layers.
    """
    class SimpleVaultCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleVaultCNN(num_classes)
    
    # Load vault weights directly (no He init)
    device = 'cpu'
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in vault_weights:
                param.data = torch.from_numpy(vault_weights[name]).to(device)
                print(f"  [VaultInit] Loaded: {name}")
    
    return model


class BinaryCatDataset(Dataset):
    """CIFAR-100 converted to binary: caterpillar (class 10) = 1, worm (class 99) = 0"""
    
    def __init__(self, cifar_dataset, balanced=False):
        self.dataset = cifar_dataset
        self.balanced = balanced
        
        if balanced:
            # Find all caterpillar indices and sample equal non-caterpillar
            self.caterpillar_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label == 10]
            # Sample same number of non-caterpillar
            non_cat_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label != 10]
            np.random.seed(42)
            sampled_non_cat = np.random.choice(non_cat_indices, size=len(self.caterpillar_indices), replace=False)
            self.selected_indices = self.caterpillar_indices + list(sampled_non_cat)
            np.random.shuffle(self.selected_indices)
    
    def __len__(self):
        if self.balanced:
            return len(self.selected_indices)
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.balanced:
            actual_idx = self.selected_indices[idx]
            img, label = self.dataset[actual_idx]
        else:
            img, label = self.dataset[idx]
        
        # caterpillar vs others
        binary_label = 1 if label == 10 else 0
        return img, torch.tensor(binary_label, dtype=torch.float32)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch (binary classification)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (torch.sigmoid(output) > 0.5).float()
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model (binary classification)."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            loss = criterion(output, target)

            total_loss += loss.item()
            predicted = (torch.sigmoid(output) > 0.5).float()
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def first_training_run(vault, num_epochs=3):
    """First training: Binary classification (cat=1, other=0), cold start."""
    print("\n" + "="*70)
    print("FIRST TRAINING: Binary Cat Classification (Cold Start)")
    print("="*70)

    DBConv2d.reset_layer_counter()
    DBLinear.reset_layer_counter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[1] Creating binary classification model (device: {device})...")

    object_category = "caterpillar"
    
    # Binary classification: caterpillar vs others - with object category for vault
    model = create_db_cnn(vault, num_classes=1, object_category=object_category).to(device)

    # Data augmentation for training (increases difficulty)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Use CIFAR-100 (100 classes) for harder task
    full_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Use only 10% of training data (10000 samples)
    train_size = int(0.10 * len(full_train))
    train_subset = torch.utils.data.Subset(full_train, range(train_size))
    
    train_dataset = BinaryCatDataset(train_subset, balanced=True)  # Balance training
    test_dataset = BinaryCatDataset(full_test, balanced=True)      # Balance testing!

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Very small batch
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"[2] Train: {len(train_dataset)} (balanced 50/50), Test: {len(test_dataset)} (balanced 50/50)")

    # BCE loss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n[3] Training for {num_epochs} epochs...")
    print("-" * 50)

    results = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.3f}, Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%")
        results.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': test_acc})

    print(f"\n[4] Storing weights to Qdrant Cloud vault...")
    final_acc = results[-1]['test_acc']

    object_category = "caterpillar"
    
    # Get model state dict
    model_state = {}
    for name, param in model.named_parameters():
        model_state[name] = param.data
    
    # Sample images for image-based search
    image_sample = []
    for i in range(min(50, len(train_dataset))):
        img, _ = train_dataset[i]
        image_sample.append(img)
    
    # Store with image features
    vault.store_category_weights(
        model_state=model_state,
        model_name="SimpleCNN",
        category=object_category,
        epoch=num_epochs,
        accuracy=final_acc,
        dataset_sample=image_sample
    )
    
    vault.save_vault()
    print(f"[Vault] Stored weights + images (category: {object_category}, Acc: {final_acc:.1f}%)")

    return results


def second_training_run(vault, num_epochs=3):
    """Second training: Same task, initialize from vault weights."""
    print("\n" + "="*70)
    print("SECOND TRAINING: Binary Cat Classification (Vault-Init)")
    print("="*70)
    print("MODE: IMAGE SEARCH\n")

    DBConv2d.reset_layer_counter()
    DBLinear.reset_layer_counter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-100
    full_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_size = int(0.10 * len(full_train))
    train_subset = torch.utils.data.Subset(full_train, range(train_size))
    
    train_dataset = BinaryCatDataset(train_subset, balanced=True)
    test_dataset = BinaryCatDataset(full_test, balanced=True)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Step 1: 用图片搜索 vault
    print(f"[0] Searching vault by image features...")
    image_sample = []
    for i in range(min(50, len(train_dataset))):
        img, _ = train_dataset[i]
        image_sample.append(img)
    
    best_category, similarity = vault.find_similar_category_by_images(image_sample)
    
    # Step 2: 如果找到相似类别，直接从 vault 加载权重（不经过 He init）
    vault_weights = None
    if best_category and similarity > 0.3:
        print(f"[1] Found category: '{best_category}' (score={similarity:.4f})")
        vault_weights = vault.get_category_weights(best_category)
        if vault_weights is not None:
            print(f"[VaultInit] Loading weights from vault (no He init)...")
    
    # Step 3: 创建模型
    print(f"[2] Creating model...")
    
    if vault_weights is not None:
        # 直接从 vault 权重创建模型（不经过 He init）
        model = create_db_cnn_from_vault(vault_weights, num_classes=1).to(device)
    else:
        # 没有 vault 权重，使用 He init
        DBConv2d.set_force_load(False)
        DBConv2d.set_object_category(None)
        model = create_db_cnn(vault, num_classes=1).to(device)
        print(f"[DBInit] No similar category found - using cold start")
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Use CIFAR-100 (same as first run)
    full_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Same 10% training data
    train_size = int(0.10 * len(full_train))
    train_subset = torch.utils.data.Subset(full_train, range(train_size))
    
    train_dataset = BinaryCatDataset(train_subset, balanced=True)
    test_dataset = BinaryCatDataset(full_test, balanced=True)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"[2] Train: {len(train_dataset)} (balanced), Test: {len(test_dataset)} (balanced)")

    # BCE loss for binary
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    print(f"\n[3] Training for {num_epochs} epochs (lr=0.0003, fine-tuning)...")
    print("-" * 50)

    results = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.3f}, Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%")
        results.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': test_acc})

    return results


def compare_results(cold_results, vault_results):
    """Compare cold start vs vault initialization."""
    print("\n" + "="*70)
    print("COMPARISON: Cold Start vs Vault-Init")
    print("="*70)

    for i in range(len(cold_results)):
        print(f"\nEpoch {i+1}:")
        print(f"  Cold Start:  Loss={cold_results[i]['train_loss']:.3f}, Acc={cold_results[i]['test_acc']:.1f}%")
        print(f"  Vault-Init:  Loss={vault_results[i]['train_loss']:.3f}, Acc={vault_results[i]['test_acc']:.1f}%")

    acc_diff = vault_results[0]['test_acc'] - cold_results[0]['test_acc']
    print(f"\nEpoch 1 Improvement: Test Acc +{acc_diff:.1f}%")


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("DATABASE-DRIVEN CNN - IMAGE RECOGNITION (Qdrant Cloud)")
    print("Task: caterpillar (1) vs not-caterpillar (0) on CIFAR-100")
    print("="*70)

    config = get_config()
    qdrant_url = config.get('vector_db.qdrant.url')
    qdrant_api_key = config.get('vector_db.qdrant.api_key')

    if not qdrant_url or not qdrant_api_key:
        print("ERROR: Qdrant Cloud config not found in settings.yaml")
        return

    print(f"\n[0] Connecting to Qdrant Cloud: {qdrant_url}")

    vault = QdrantWeightVault(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=config.get('vector_db.qdrant.collection_name', 'cnn_weights'),
        similarity_threshold=0.3,
        top_k_ratio=0.3
    )

    try:
        stats = vault.get_stats()
        print(f"    Connected! Current entries: {stats.get('total_entries', 0)}")
    except Exception as e:
        print(f"    Connection failed: {e}")
        return

    num_epochs = 3  # Reduced for faster demo

    cold_results = first_training_run(vault, num_epochs=num_epochs)

    print("\n" + "="*70)
    print("VAULT STATUS")
    print("="*70)
    stats = vault.get_stats()
    print(f"  Total entries: {stats.get('total_entries', 0)}")
    print(f"  Collections: {stats.get('collections', 0)}")

    vault_results = second_training_run(vault, num_epochs=num_epochs)

    compare_results(cold_results, vault_results)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()