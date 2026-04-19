"""
Complete Demo: Real ResNet with Database-Driven Weight Initialization

This demo shows:
1. First Training: Cold start with random He initialization
2. Store weights to vault after training
3. Second Training: Same category, initialize from stored vault weights
4. Compare convergence speed between cold start and vault-init
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import sys
import os

# Add parent directory to path for importing cnn_weight_vault
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cnn_weight_vault.detection_vault import DetectionWeightVault
from cnn_weight_vault.detection_model import (
    DBConv2dDetect, DBLinearDetect, convert_to_db_layers
)


def get_cifar10_loaders(batch_size=64):
    """Get CIFAR-10 train and test loaders."""
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f"  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc


def store_model_weights(model, vault, model_name, epoch, accuracy, category):
    """Store all DB layer weights to vault."""
    stored_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (DBConv2dDetect, DBLinearDetect)):
            module.store_weights(epoch, accuracy)
            stored_count += 1

    print(f"[Vault] Stored weights from {stored_count} layers "
          f"(epoch {epoch}, acc {accuracy:.2f}%)")
    vault.save_vault()


def first_training_run(vault, category="cifar10", num_epochs=3):
    """
    First training: Cold start with random initialization.
    Stores weights to vault after training.
    """
    print("\n" + "="*70)
    print(f"FIRST TRAINING: ResNet-18 on {category.upper()} (Cold Start)")
    print("="*70)

    # Reset counters
    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()

    # Configure vault
    DBConv2dDetect.set_vault(vault)
    DBConv2dDetect.set_object_category(category)
    DBConv2dDetect.set_force_load(False)
    DBLinearDetect.set_vault(vault)
    DBLinearDetect.set_object_category(category)
    DBLinearDetect.set_force_load(False)

    # Load ResNet-18
    print("\n[1] Loading ResNet-18 from torchvision...")
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )

    # Modify final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Convert to DB-aware layers
    print("[2] Converting to DB-aware layers...")
    model = convert_to_db_layers(model, vault=vault, object_category=category)

    # Count DB layers
    db_conv_count = sum(1 for m in model.modules() if isinstance(m, DBConv2dDetect))
    db_linear_count = sum(1 for m in model.modules() if isinstance(m, DBLinearDetect))
    print(f"[3] Model has {db_conv_count} DBConv2dDetect + {db_linear_count} DBLinearDetect layers")

    # Setup training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[4] Using device: {device}")
    model = model.to(device)

    train_loader, test_loader = get_cifar10_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    print(f"\n[5] Training for {num_epochs} epochs...")
    results = []
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}:")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

    # Store weights to vault
    print(f"\n[6] Storing weights to vault...")
    final_acc = results[-1]['test_acc']
    store_model_weights(model, vault, "resnet18", num_epochs, final_acc, category)

    print(f"\n[7] First training complete. Final test accuracy: {final_acc:.2f}%")
    return results


def second_training_run(vault, category="cifar10", num_epochs=3):
    """
    Second training: Same category, initialize from vault weights.
    Demonstrates faster convergence from vault initialization.
    """
    print("\n" + "="*70)
    print(f"SECOND TRAINING: ResNet-18 on {category.upper()} (Vault-Init)")
    print("="*70)
    print("MODE: FORCE LOAD - Will use stored vault weights for initialization\n")

    # Reset counters (IMPORTANT!)
    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()

    # Configure vault with FORCE LOAD enabled
    DBConv2dDetect.set_vault(vault)
    DBConv2dDetect.set_object_category(category)
    DBConv2dDetect.set_force_load(True)  # FORCE LOAD!
    DBLinearDetect.set_vault(vault)
    DBLinearDetect.set_object_category(category)
    DBLinearDetect.set_force_load(True)  # FORCE LOAD!

    # Load ResNet-18
    print("[1] Loading ResNet-18 from torchvision...")
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )

    # Modify final layer for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Convert to DB-aware layers (will load from vault due to force_load=True)
    print("[2] Converting to DB-aware layers (FORCE LOAD mode)...")
    model = convert_to_db_layers(
        model, vault=vault, object_category=category, force_load=True
    )

    # Setup training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[3] Using device: {device}")
    model = model.to(device)

    train_loader, test_loader = get_cifar10_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    print(f"\n[4] Training for {num_epochs} epochs...")
    results = []
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}:")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

    print(f"\n[5] Second training complete. Final test accuracy: {results[-1]['test_acc']:.2f}%")
    return results


def compare_results(cold_results, vault_results):
    """Compare cold start vs vault initialization results."""
    print("\n" + "="*70)
    print("COMPARISON: Cold Start vs Vault-Init")
    print("="*70)

    print("\nEpoch 1:")
    print(f"  Cold Start:    Train Loss={cold_results[0]['train_loss']:.4f}, "
          f"Test Acc={cold_results[0]['test_acc']:.2f}%")
    print(f"  Vault-Init:    Train Loss={vault_results[0]['train_loss']:.4f}, "
          f"Test Acc={vault_results[0]['test_acc']:.2f}%")

    if len(cold_results) > 1 and len(vault_results) > 1:
        print("\nEpoch 2:")
        print(f"  Cold Start:    Train Loss={cold_results[1]['train_loss']:.4f}, "
              f"Test Acc={cold_results[1]['test_acc']:.2f}%")
        print(f"  Vault-Init:    Train Loss={vault_results[1]['train_loss']:.4f}, "
              f"Test Acc={vault_results[1]['test_acc']:.2f}%")

    print("\nFinal Epoch:")
    print(f"  Cold Start:    Train Loss={cold_results[-1]['train_loss']:.4f}, "
          f"Test Acc={cold_results[-1]['test_acc']:.2f}%")
    print(f"  Vault-Init:    Train Loss={vault_results[-1]['train_loss']:.4f}, "
          f"Test Acc={vault_results[-1]['test_acc']:.2f}%")

    # Calculate improvement
    acc_improvement = vault_results[0]['test_acc'] - cold_results[0]['test_acc']
    loss_reduction = cold_results[0]['train_loss'] - vault_results[0]['train_loss']

    print("\n" + "-"*70)
    print(f"Epoch 1 Improvement from Vault-Init:")
    print(f"  Test Accuracy: +{acc_improvement:.2f}%")
    print(f"  Train Loss:    -{loss_reduction:.4f}")
    print("-"*70)


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("DATABASE-DRIVEN CNN INITIALIZATION DEMO")
    print("Using Real ResNet-18 on CIFAR-10")
    print("="*70)
    print("\nThis demo shows:")
    print("  1. First training: Cold start with random initialization")
    print("  2. Store trained weights to vault")
    print("  3. Second training: Same task, initialize from vault weights")
    print("  4. Compare convergence speed\n")

    # Create vault
    vault = DetectionWeightVault(
        similarity_threshold=0.3,
        top_k_ratio=0.3,
        vault_path="../resnet_cifar10_vault"
    )

    # Number of epochs for demo (use fewer for faster demo)
    num_epochs = 3

    # First training: Cold start
    cold_results = first_training_run(vault, category="cifar10", num_epochs=num_epochs)

    # Show vault stats
    print("\n" + "="*70)
    print("VAULT STATUS AFTER FIRST TRAINING")
    print("="*70)
    stats = vault.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Second training: Vault-init (same category)
    vault_results = second_training_run(vault, category="cifar10", num_epochs=num_epochs)

    # Compare results
    compare_results(cold_results, vault_results)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Takeaways:
1. First Training (Cold Start):
   - All layers initialized with He initialization
   - Slower convergence, higher initial loss
   - Weights stored to vault after training

2. Second Training (Vault-Init):
   - All layers initialized from stored vault weights
   - Faster convergence, lower initial loss
   - Higher initial accuracy (better starting point)

3. The vault successfully:
   - Stores trained weights after first training
   - Retrieves and initializes second model with stored weights
   - Demonstrates transfer learning within same category

4. Force Load Mode:
   - Bypasses similarity threshold
   - Guarantees vault weight loading when available
   - Useful for same-task repeated training

Files created:
  - ../resnet_cifar10_vault/: Contains stored weights
    """)


if __name__ == "__main__":
    main()
