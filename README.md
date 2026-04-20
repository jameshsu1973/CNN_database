# Database-Driven CNN Initialization

A PyTorch implementation of Database-Driven CNN Weight Initialization using **ChromaDB** vector database for efficient weight retrieval.

## Overview

This project replaces traditional random weight initialization (He/Xavier) with a queryable, cumulative database of optimized historical weights using **ChromaDB** as the vector database backend. The system supports:

- **Image Classification** (MNIST, CIFAR-10, etc.)
- **Object Detection** (single-class and multi-class)
- **Transfer Learning** (cross-category weight sharing)
- **Existing Model Wrapping** (ResNet, VGG, etc. without modification)
- **ChromaDB Integration** (HNSW indexing, cosine similarity, persistent storage)

## Key Features

1. **ChromaDB Vector Database**: Native vector storage with HNSW indexing for O(log n) similarity search
2. **Automatic Weight Storage**: After training epochs, weights are automatically stored to ChromaDB
3. **Smart Initialization**: New models initialize from similar weights in the database using cosine similarity
4. **Force Load Mode**: Bypass similarity checks to guarantee vault weight loading
5. **Object Category Tagging**: Label weights by object type (e.g., "cat", "dog") for transfer learning
6. **Cold Start Fallback**: Falls back to He initialization when database is empty
7. **Top-K Masking**: Only most influential weights (top 30%) are stored
8. **Persistent Storage**: Automatic persistence via DuckDB + Parquet backend

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+
- NumPy
- torchvision
- **ChromaDB >= 0.4.22** (vector database)
- PyYAML (configuration)

## Quick Start

### 1. Basic Image Classification (using ChromaDB)

```python
from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.db_initialization import create_db_cnn, DBModelWrapper

# Create ChromaDB vault
vault = ChromaWeightVault(
    collection_name="cnn_weights",
    persist_directory="./chroma_db",
    similarity_threshold=0.85,
    top_k_ratio=0.3
)

# Create model with DB-aware layers
model = create_db_cnn(vault, num_classes=10)
wrapper = DBModelWrapper(model, vault, model_name="my_cnn")

# Train...
for epoch in range(num_epochs):
    train_loss = train_epoch(model, ...)
    test_acc = test_epoch(model, ...)

    # Store weights to ChromaDB vault after each epoch
    wrapper.store_epoch_weights(epoch, test_acc)

# Save vault (ChromaDB auto-persists)
vault.save_vault()
```

### 2. Object Detection (using ChromaDB)

```python
from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.detection_model import create_detection_model, DetectionModelWrapper

# Create ChromaDB vault
vault = ChromaWeightVault(
    collection_name="detection_weights",
    persist_directory="./chroma_db_detection",
    similarity_threshold=0.3
)

# Create object detection model (e.g., cat detector)
model = create_detection_model(
    vault,
    object_category="cat",
    num_classes=1
)
wrapper = DetectionModelWrapper(
    model,
    vault,
    model_name="cat_detector",
    object_category="cat"
)

# Train...
for epoch in range(num_epochs):
    train_loss, train_map = train_epoch(model, ...)

    # Store weights to ChromaDB
    wrapper.store_epoch_weights(epoch, train_map)

wrapper.save_vault()
```

### 3. Force Load Mode

Force load mode ensures weights are always loaded from vault (if available):

```python
from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.detection_model import DBConv2dDetect, DBLinearDetect

# Load existing ChromaDB vault
vault = ChromaWeightVault(persist_directory="./chroma_db")

# Enable force load mode
DBConv2dDetect.set_force_load(True)
DBLinearDetect.set_force_load(True)

# Create model - will FORCE load from vault
model = create_detection_model(
    vault,
    object_category="cat",
    force_load=True  # Force load parameter
)
```

### 4. Wrap Existing Models (ResNet, VGG, etc.)

Convert ANY PyTorch model to use database-driven initialization:

```python
import torchvision.models as models
from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.detection_model import convert_to_db_layers

# Load pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Create ChromaDB vault
vault = ChromaWeightVault(persist_directory="./chroma_db_resnet")

# Wrap model with DB-aware layers
model_db = convert_to_db_layers(
    model,
    vault=vault,
    object_category="animal",
    force_load=False
)

# Forward pass works normally
output = model_db(torch.randn(1, 3, 224, 224))

# Now train and weights will be stored to ChromaDB
```

## Running Examples

### Object Detection Demo

```bash
python train_detection.py
```

This demonstrates:
- First training run (cold start with He initialization)
- Second training run (force load from vault)
- Comparison of convergence speed

**Expected Output**:
```
============================================================
FIRST TRAINING: CAT DETECTION (Cold Start)
============================================================
[DBInit] Layer conv_0: Cold start - using He initialization
Epoch 1: Train Loss=7.31, Test mAP=31.23%
Epoch 2: Train Loss=5.12, Test mAP=52.34%
Epoch 3: Train Loss=3.89, Test mAP=68.91%

============================================================
SECOND TRAINING: CAT DETECTION (Vault-Init)
MODE: FORCE LOAD - Will use vault weights regardless of similarity
============================================================
[DBInit] Layer conv_0: Initialized from vault [FORCED]
Epoch 1: Train Loss=2.70, Test mAP=61.45%  # 2.7x faster start!
Epoch 2: Train Loss=2.15, Test mAP=72.34%
Epoch 3: Train Loss=1.89, Test mAP=78.92%
```

### Wrap Existing Model Demo

```bash
python wrap_existing_cnn_demo.py
```

This demonstrates:
- Wrapping torchvision ResNet-18 (20 Conv2d → 20 DBConv2dDetect)
- Wrapping torchvision VGG-16 (13 Conv2d → 13 DBConv2dDetect)
- Forward pass verification

### Classification Demo

```bash
python train_example.py
```

MNIST classification example with vault initialization.

## Usage Patterns

### Pattern 1: First Training (Cold Start)

```python
from cnn_weight_vault.chroma_vault import ChromaWeightVault

# Create new ChromaDB vault
vault = ChromaWeightVault(persist_directory="./chroma_db")

# Create model (will use He initialization)
model = create_detection_model(vault, object_category="cat")
wrapper = DetectionModelWrapper(model, vault, model_name="cat_detector")

# Train and store weights
for epoch in range(num_epochs):
    train(model, ...)
    wrapper.store_epoch_weights(epoch, accuracy)

vault.save_vault()
```

### Pattern 2: Subsequent Training (Vault-Init with Similarity)

```python
# ChromaDB vault auto-loads from persist_directory
vault = ChromaWeightVault(persist_directory="./chroma_db")

# Create model (will auto-load if similarity > threshold)
model = create_detection_model(vault, object_category="cat")

# Train...
```

### Pattern 3: Force Load Mode (Guaranteed Vault-Init)

```python
# ChromaDB vault auto-loads from persist_directory
vault = ChromaWeightVault(persist_directory="./chroma_db")

# Create model with force_load=True
model = create_detection_model(
    vault,
    object_category="cat",
    force_load=True  # Always use vault weights if available
)
```

### Pattern 4: Cross-Category Transfer

```python
# Train on "cat"
vault = ChromaWeightVault(persist_directory="./chroma_db")
model = create_detection_model(vault, object_category="cat")
# ... train and save ...

# Train on "dog" - will try to use "cat" weights if no "dog" weights
model = create_detection_model(vault, object_category="dog")
# System gives 20% similarity boost to same-category weights
```

## API Reference

### ChromaWeightVault

```python
vault = ChromaWeightVault(
    collection_name="cnn_weights",      # Collection name prefix
    persist_directory="./chroma_db",    # ChromaDB storage path
    similarity_threshold=0.3,           # Cosine similarity threshold
    top_k_ratio=0.3                     # Retain top 30% weights
)

# Methods
vault.store_weights(layer, layer_name, model_name, epoch, accuracy, category)
vault.get_initialization_weights(layer, layer_name, category, force=False)
vault.has_weights_for_layer(layer, layer_name)
vault.save_vault()
vault.migrate_from_pickle(pickle_path)  # Migrate from legacy pickle vault
```
vault.get_stats()
```

### DBConv2dDetect / DBLinearDetect

```python
# Class methods for configuration
DBConv2dDetect.set_vault(vault)
DBConv2dDetect.set_model_name("my_model")
DBConv2dDetect.set_object_category("cat")
DBConv2dDetect.set_force_load(True)  # Enable force load
DBConv2dDetect.reset_layer_counter()  # Reset before creating model
```

### DetectionModelWrapper

```python
wrapper = DetectionModelWrapper(
    model,
    vault,
    model_name="detector",
    object_category="cat",
    force_load=False
)

# Methods
wrapper.store_epoch_weights(epoch, accuracy)
wrapper.save_vault()
wrapper.get_vault_stats()
```

### convert_to_db_layers

```python
from detection_model import convert_to_db_layers

wrapped_model = convert_to_db_layers(
    existing_model,      # Any nn.Module
    vault=vault,         # DetectionWeightVault instance
    object_category="cat",
    force_load=False     # Whether to force load from vault
)
```

## How It Works

### Weight Storage Flow

```
Training Complete
       ↓
Top-K Masking (retain top 30% by |weight|)
       ↓
Flatten Kernel → 1D Vector
       ↓
Store in Vault with Metadata (model, epoch, accuracy, category)
       ↓
Update HNSW Graph Structure
```

### Weight Initialization Flow

```
Create Layer
     ↓
Check Vault for Similar Weights
     ↓
├─ Force Mode ON ─→ Load from Vault (if exists)
├─ Similarity > Threshold ─→ Load from Vault
└─ Otherwise ─→ He Initialization (Cold Start)
```

### Similarity Search

- **Cosine Similarity**: `S(A,B) = (A·B) / (||A||·||B||)`
- **HNSW Distance**: `Dist(A,B) = 1 - S(A,B)`
- **Search Complexity**: O(log n) using greedy graph traversal

## File Structure

| File | Description |
|------|-------------|
| `detection_vault.py` | Vector database with HNSW-inspired indexing |
| `detection_model.py` | DB-aware PyTorch layers and model wrapper |
| `train_detection.py` | Object detection training demo |
| `wrap_existing_cnn_demo.py` | ResNet/VGG wrapping demo |
| `example_wrap_existing_model.py` | Usage examples |
| `train_example.py` | Classification training demo |

## Technical Details

### No PyTorch Modifications

The system uses inheritance, not modification:

```python
# Standard PyTorch
nn.Conv2d(in_channels, out_channels, kernel_size)

# Database-driven version
class DBConv2dDetect(nn.Conv2d):  # Inherit
    def __init__(self, ...):
        super().__init__(...)  # Call parent
        self._initialize_from_vault()  # Add DB initialization
```

### Object Category Support

Weights are tagged by object category for transfer learning:

```python
# Training on cats
model = create_detection_model(vault, object_category="cat")

# Training on dogs - system prefers "dog" weights,
# but will use "cat" weights if no "dog" weights available
model = create_detection_model(vault, object_category="dog")
```

### Force Load Mode

Normal mode checks similarity before loading:
- Similarity > threshold → Load from vault
- Similarity ≤ threshold → He initialization

Force load mode:
- If weights exist in vault → Load from vault
- No weights in vault → He initialization

## Troubleshooting

### "All layers showing Cold Start"

- Vault is empty (first training) - this is expected
- Set `similarity_threshold` lower (e.g., 0.0 to 0.3)
- Use `force_load=True` to guarantee vault loading

### Shape Mismatch Errors

- Ensure layer dimensions match between stored and new weights
- The system automatically falls back to He initialization on shape mismatch

### "No weights being stored"

- Call `wrapper.store_epoch_weights(epoch, accuracy)` after each epoch
- Call `wrapper.save_vault()` at the end of training

## Contributing

This is a research implementation. Key areas for improvement:

1. Full HNSW multi-layer graph structure
2. Kernel size tolerance (5×5 vs 3×3 matching)
3. Disk-based persistent indexing
4. More layer types (BatchNorm, etc.)
5. Real dataset integration (COCO, ImageNet)

## License

MIT License - Research Use

## References

- HNSW: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2016)
- He Initialization: He et al., "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" (2015)
