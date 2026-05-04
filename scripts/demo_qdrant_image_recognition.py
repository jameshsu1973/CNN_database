"""
Database-Driven CNN Initialization Demo: Binary Classification (Qdrant Cloud)

Binary classification: cat (1) vs not-cat (0)

This demo shows:
1. First Training: Binary classification, cold start with random initialization
2. Store weights to Qdrant Cloud vault after training
3. Second Training: Same task, initialize from stored vault weights
4. Compare convergence speed between cold start and vault-init

Uses the new generic weight management (wrap.py) approach.
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

from cnn_weight_vault import QdrantWeightVault, extract_weights, load_weights
from cnn_weight_vault.config import get_config


class SimpleCNN(nn.Module):
    """Simple CNN for binary classification."""
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BinaryCatDataset(Dataset):
    """CIFAR-10 converted to binary: cat (3) = 1, others = 0"""

    def __init__(self, cifar_dataset, balanced=False):
        self.dataset = cifar_dataset
        self.balanced = balanced

        if balanced:
            self.cat_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label == 3]
            non_cat_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label != 3]
            np.random.seed(42)
            sampled_non_cat = np.random.choice(non_cat_indices, size=len(self.cat_indices), replace=False)
            self.selected_indices = self.cat_indices + list(sampled_non_cat)
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

        binary_label = 1 if label == 3 else 0
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


def first_training_run(vault, object_category="cat", num_epochs=3):
    """First training: Binary classification (cat=1, other=0), cold start."""
    print("\n" + "=" * 70)
    print("FIRST TRAINING: Binary Cat Classification (Cold Start)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[1] Creating binary classification model (device: {device})...")

    # Create fresh model (random initialization)
    model = SimpleCNN(num_classes=1).to(device)

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

    # Use CIFAR-100
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Use 10% of training data
    train_size = int(0.10 * len(full_train))
    train_subset = torch.utils.data.Subset(full_train, range(train_size))

    train_dataset = BinaryCatDataset(train_subset, balanced=True)
    test_dataset = BinaryCatDataset(full_test, balanced=True)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"[2] Train: {len(train_dataset)} (balanced), Test: {len(test_dataset)}")

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

    # 使用新的 extract_weights 函數存儲權重
    extract_weights(
        model=model,
        vault=vault,
        model_name="SimpleCNN",
        epoch=num_epochs,
        accuracy=final_acc,
        object_category=object_category
    )

    vault.save_vault()
    print(f"[Vault] Stored weights (category: {object_category}, Acc: {final_acc:.1f}%)")

    return results


def second_training_run(vault, object_category="cat", num_epochs=3):
    """Second training: Same task, initialize from vault weights."""
    print("\n" + "=" * 70)
    print("SECOND TRAINING: Binary Cat Classification (Vault-Init)")
    print("=" * 70)
    print("MODE: force=True (always load from vault)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create new model (same architecture)
    model = SimpleCNN(num_classes=1).to(device)

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
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_size = int(0.10 * len(full_train))
    train_subset = torch.utils.data.Subset(full_train, range(train_size))

    train_dataset = BinaryCatDataset(train_subset, balanced=True)
    test_dataset = BinaryCatDataset(full_test, balanced=True)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"[1] Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # 使用新的 load_weights 函數載入權重，force=True 確保必定載入
    print(f"[2] Loading weights from vault (force=True)...")
    results = load_weights(
        model=model,
        vault=vault,
        object_category=object_category,
        force=True  # 強制載入，忽略相似度閾值
    )

    loaded = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"[3] Loaded {loaded}/{total} layers from vault")

    # BCE loss for binary
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)  # Lower LR for fine-tuning

    print(f"\n[4] Training for {num_epochs} epochs (lr=0.0003, fine-tuning)...")
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
    print("\n" + "=" * 70)
    print("COMPARISON: Cold Start vs Vault-Init")
    print("=" * 70)

    for i in range(len(cold_results)):
        print(f"\nEpoch {i+1}:")
        print(f"  Cold Start:  Loss={cold_results[i]['train_loss']:.3f}, Acc={cold_results[i]['test_acc']:.1f}%")
        print(f"  Vault-Init:  Loss={vault_results[i]['train_loss']:.3f}, Acc={vault_results[i]['test_acc']:.1f}%")

    if vault_results[0]['test_acc'] > cold_results[0]['test_acc']:
        acc_diff = vault_results[0]['test_acc'] - cold_results[0]['test_acc']
        print(f"\nEpoch 1 Improvement: Test Acc +{acc_diff:.1f}%")
    else:
        acc_diff = cold_results[0]['test_acc'] - vault_results[0]['test_acc']
        print(f"\nEpoch 1 Difference: Test Acc -{acc_diff:.1f}%")


def main():
    """Main demo function."""
    print("\n" + "=" * 70)
    print("DATABASE-DRIVEN CNN - IMAGE RECOGNITION (Qdrant Cloud)")
    print("Task: cat (1) vs not-cat (0) on CIFAR-10")
    print("Using generic weight management (wrap.py)")
    print("=" * 70)

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
        collection_name="demo_image_recognition",
        similarity_threshold=0.3,
        top_k_ratio=0.3
    )

    try:
        stats = vault.get_stats()
        print(f"    Connected! Current entries: {stats.get('total_entries', 0)}")
    except Exception as e:
        print(f"    Connection failed: {e}")
        return

    # First training: cold start
    cold_results = first_training_run(vault, object_category="cat", num_epochs=3)

    # Second training: vault init
    vault_results = second_training_run(vault, object_category="cat", num_epochs=3)

    # Compare results
    compare_results(cold_results, vault_results)

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()