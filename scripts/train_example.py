"""
Example training script demonstrating Database-Driven CNN Initialization
This shows the workflow: train -> store weights in vault -> use vault for next training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.db_initialization import (
    create_db_cnn, DBModelWrapper, DBConv2d, DBLinear
)
from cnn_weight_vault.config import get_config


def get_mnist_loaders(batch_size=64):
    """Get MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    accuracy = 100. * correct / total
    return total_loss / len(train_loader), accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return test_loss / len(test_loader), accuracy


def run_first_training(vault: ChromaWeightVault, device: str = 'cuda'):
    """
    Run the first training session (cold start).
    This will populate the vault with initial weights.
    """
    print("\n" + "="*60)
    print("FIRST TRAINING RUN (Cold Start - Populating Vault)")
    print("="*60)

    # Create model with DB-aware layers
    model = create_db_cnn(vault, num_classes=10).to(device)
    wrapper = DBModelWrapper(model, vault, model_name="mnist_cnn_v1")

    train_loader, test_loader = get_mnist_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for a few epochs
    for epoch in range(1, 4):  # Just 3 epochs for demo
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Test Acc={test_acc:.2f}%")

        # Store weights to vault after each epoch
        wrapper.store_epoch_weights(epoch, test_acc)

    # Save vault
    wrapper.save_vault()

    print(f"\nVault stats: {wrapper.get_vault_stats()}")

    return test_acc


def run_second_training(vault: ChromaWeightVault, device: str = 'cuda'):
    """
    Run the second training session using vault-initialized weights.
    This demonstrates the speedup from database-driven initialization.
    """
    print("\n" + "="*60)
    print("SECOND TRAINING RUN (Vault-Initialized)")
    print("="*60)

    # Reset layer counters for new model
    DBConv2d.reset_layer_counter()
    DBLinear.reset_layer_counter()

    # Create new model - will be initialized from vault!
    model = create_db_cnn(vault, num_classes=10).to(device)
    wrapper = DBModelWrapper(model, vault, model_name="mnist_cnn_v2")

    train_loader, test_loader = get_mnist_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining with vault-initialized weights...")

    # Train
    for epoch in range(1, 4):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Test Acc={test_acc:.2f}%")

        # Store weights (these will be added to vault too)
        wrapper.store_epoch_weights(epoch, test_acc)

    wrapper.save_vault()

    return test_acc


def compare_convergence():
    """
    Compare convergence between cold start and vault-initialized training.
    """
    print("\n" + "="*60)
    print("CONVERGENCE COMPARISON")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create ChromaDB vault
    config = get_config()
    vault = ChromaWeightVault(
        collection_name=config.chroma_collection_name,
        persist_directory="./chroma_db_mnist",
        similarity_threshold=0.0,  # Force vault initialization for demo
        top_k_ratio=0.3
    )

    # Run first training (cold start)
    acc1 = run_first_training(vault, device)

    # Run second training (vault initialized)
    acc2 = run_second_training(vault, device)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Cold start final accuracy: {acc1:.2f}%")
    print(f"Vault-init final accuracy: {acc2:.2f}%")
    print(f"\nVault contents: {vault.get_stats()}")


def demonstrate_steps():
    """
    Demonstrate the 5 steps of the database-driven initialization system.
    """
    print("\n" + "="*60)
    print("DEMONSTRATING THE 5 STEPS")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 1: Feature Extraction & Masking
    print("\n[Step 1] Feature Extraction & Masking")
    print("- After training, extract top-k most influential features by magnitude")
    print("- Formula: I(t) = arg top_k(|W_final(t)|)")

    # Step 2: Database Accumulation
    print("\n[Step 2] Database Accumulation")
    print("- Update global database D(t)")
    print("- Formula: D(t) = D(t-1) + (W_final(t) ⊙ 1_I(t))")
    print("- Only top-k indices receive 1 in indicator vector")

    # Step 3: Kernel Flattening
    print("\n[Step 3] Kernel Flattening for Storage")
    print("- Flatten kernels into 1D array V = [w_1, w_2, ..., w_D]")
    print("- D = output_channels × input_channels × height × width")

    # Step 4: Similarity-to-Distance Translation
    print("\n[Step 4] Similarity-to-Distance Translation")
    print("- Cosine Similarity: S(A,B) = (A·B) / (||A||·||B||)")
    print("- HNSW Distance: Dist(A,B) = 1 - S(A,B)")
    print("- Identical vectors have distance 0")

    # Step 5: High-Speed Logarithmic Retrieval
    print("\n[Step 5] High-Speed Logarithmic Retrieval")
    print("- Use Greedy Graph Search on HNSW structure")
    print("- Retrieve optimal kernels in O(log n) time")
    print("- Equation 12 allows anti-similarity routing to avoid dead zones")

    # Run the actual training
    compare_convergence()


if __name__ == "__main__":
    demonstrate_steps()
