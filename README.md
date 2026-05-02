# CNN Weight Vault - 資料庫驅動的 CNN 權重初始化系統

## 專案目標

建立一個 CNN 權重資料庫，讓不同任務的模型可以互相分享學習到的權重，實現**遷移學習**。

### 核心功能
- 儲存訓練好的 CNN 權重到向量資料庫
- 根據**圖片特徵**進行相似度搜尋，找到相似的已訓練類別
- 新任務可以從資料庫中取得相似類別的權重作為初始化

---

## Overview

A PyTorch implementation of **CNN Weight Vault** for transfer learning via vector database. Instead of random initialization (He/Xavier), new models query the vault to find similar categories and initialize with proven weights.

**Key Innovation**: Image-based similarity search - match tasks by analyzing training image statistics, not just model architecture.

---

## Key Features

1. **Qdrant Cloud Vector Database**: Cloud-based HNSW indexing for O(log n) similarity search
2. **Image-Based Similarity**: Match new datasets to similar categories using image statistics (mean, std, histogram)
3. **Complete Weight Storage**: Full model weights stored as JSON in payload (not just layer vectors)
4. **Direct Model Creation**: `create_db_cnn_from_vault()` loads weights directly, bypassing He init
5. **Cold Start Fallback**: Falls back to He initialization when no similar category found

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+
- NumPy, torchvision
- **qdrant-client** (Qdrant Cloud client)
- PyYAML

---

## Quick Start

### 1. First Training (Cold Start + Store)

```python
from cnn_weight_vault.qdrant_vault import QdrantWeightVault
from cnn_weight_vault.db_initialization import create_db_cnn

# Connect to Qdrant Cloud
vault = QdrantWeightVault(
    url="https://aee6f6e3-503a-4368-9950-ca7599a12bdf.us-west-1-0.aws.cloud.qdrant.io:6333",
    api_key="your_api_key",
    collection_name="cnn_weights"
)

# Create model (uses He initialization)
model = create_db_cnn(vault, num_classes=1)

# Train model...
train_model(model)

# Store to vault with image features
model_state = {name: param.data for name, param in model.named_parameters()}
image_sample = [train_dataset[i][0] for i in range(50)]

vault.store_category_weights(
    model_state=model_state,
    model_name="SimpleCNN",
    category="caterpillar",
    epoch=3,
    accuracy=59.5,
    dataset_sample=image_sample
)
```

### 2. Second Training (Vault Init via Image Similarity)

```python
# Query vault with new dataset images
vault = QdrantWeightVault(url=..., api_key=..., collection_name="cnn_weights")

# Find similar category using image features
image_sample = [new_dataset[i][0] for i in range(50)]
best_category, similarity = vault.find_similar_category_by_images(image_sample)

# Load weights from similar category
if best_category and similarity > 0.3:
    vault_weights = vault.get_category_weights(best_category)
    model = create_db_cnn_from_vault(vault_weights, num_classes=1)
else:
    model = create_db_cnn(vault, num_classes=1)  # Cold start

train_model(model)
```

---

## Database Schema (Qdrant Cloud)

### Collection: `cnn_weights`

**Single collection**, each point represents one **category**.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique point ID |
| `vector` | float[512] | Image features for HNSW search |

### Payload Structure

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | str | Model identifier (e.g., "SimpleCNN") |
| `category` | str | Object category (e.g., "caterpillar") |
| `epoch` | int | Training epoch |
| `accuracy` | float | Model accuracy (%) |
| `weights` | dict | Complete model weights (JSON) |
| `image_stats` | dict | Image statistics (mean, std, num_samples) |

### Visual Schema

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Collection: cnn_weights (vectors: 512-dim, COSINE, HNSW index)         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Point 1 (UUID-a1b2...)              Point 2 (UUID-c3d4...)           │
│  ┌──────────────────────────┐          ┌──────────────────────────┐     │
│  │ vector: [0.45, 0.22,     │          │ vector: [0.52, 0.18,     │     │
│  │        0.12, ...]        │          │        0.33, ...]        │     │
│  │                          │          │                          │     │
│  │ payload:                 │          │ payload:                 │     │
│  │   category: "caterpillar"│          │   category: "cat"        │     │
│  │   model_name: "SimpleCNN"│          │   model_name: "SimpleCNN"│    │
│  │   epoch: 3               │          │   epoch: 5              │     │
│  │   accuracy: 59.5         │          │   accuracy: 72.3         │     │
│  │   weights: {              │          │   weights: {             │     │
│  │     conv1.weight: [...], │          │     conv1.weight: [...],│     │
│  │     fc1.weight: [...],   │          │     fc1.weight: [...],  │     │
│  │     ...                  │          │     ...                 │     │
│  │   }                      │          │   }                     │     │
│  │   image_stats: {mean, std}│          │   image_stats: {mean, std}│   │
│  └──────────────────────────┘          └──────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Vector: Image Features (max 512 dimensions)

Image statistics are used as the feature vector for similarity search:

| Feature Type | Per Image | × 50 Images | Dimensions |
|-------------|-----------|-------------|------------|
| Basic stats (mean, std, min, max) | 4 | 4 × 50 | 200 |
| Histogram (8-bin normalized) | 8 | 8 × 50 | 400 |
| **Total** | **12** | | **600** |

**Note**: Actual dimension is capped at `MAX_FEATURES = 512` — features beyond 512 are truncated.


**Histogram calculation** (`torch.histc`):
```python
hist = torch.histc(img_flat, bins=8)      # 8-bin histogram
hist = hist / img_flat.numel()           # Normalize by total pixels
# Result: 8 values representing pixel intensity distribution
```

### Search Flow

```
┌─────────────┐     Extract     ┌─────────────┐     HNSW      ┌──────────────────┐
│ New Dataset │ ──────────────→ │ Image Features│ ───────────→ │ cnn_weights      │
│ (50 imgs)   │   (512-dim)    │ (512-dim)     │   Search    │ collection       │
└─────────────┘                └─────────────┘               └────────┬────────┘
                                                                          │
                    ←── Most Similar Category + Similarity Score ──────────┘
                              │
                              ↓
              ┌───────────────────────────────┐
              │  similarity > threshold?       │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              ↓ (yes)                        ↓ (no)
    ┌─────────────────────┐        ┌─────────────────┐
    │ get_category_weights │        │ create_db_cnn()  │
    │ → create_db_cnn_    │        │ (He init)       │
    │   from_vault()      │        └─────────────────┘
    └─────────────────────┘
```

### Weight Storage Flow

```
Training Complete
       ↓
Extract image features (512-dim)
       ↓
Upsert Point to Qdrant
├── vector = image features
└── payload = {
      model_name,
      category,
      epoch,
      accuracy,
      weights (full JSON),
      image_stats
    }
```

---

## API Reference

### QdrantWeightVault

```python
vault = QdrantWeightVault(
    url="https://aee6f6e3-503a-4368-9950-ca7599a12bdf.us-west-1-0.aws.cloud.qdrant.io:6333",
    api_key="your_api_key",
    collection_name="cnn_weights"
)
```

| Method | Description |
|--------|-------------|
| `store_category_weights()` | Store model weights + image features |
| `get_category_weights(category)` | Retrieve weights for a specific category |
| `find_similar_category_by_images(images)` | Search by image features |
| `find_similar_category(model_state)` | Search by model features |
| `get_stats()` | Get vault statistics |

### create_db_cnn_from_vault

```python
def create_db_cnn_from_vault(vault_weights: dict, num_classes: int = 1) -> nn.Module
```

Creates a model with weights from vault, **bypassing He initialization**.

---

## File Structure

```
CNN_database/
├── cnn_weight_vault/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── qdrant_vault.py        # Qdrant Cloud (main)
│   ├── chroma_vault.py        # ChromaDB (legacy)
│   ├── milvus_vault.py        # Milvus (legacy)
│   ├── db_initialization.py   # DB-driven initialization
│   └── detection_model.py     # Detection model wrapper
├── config/
│   └── settings.yaml          # Configuration file
├── scripts/
│   └── migrate_to_chroma.py   # Migration script
└── logs/
    └── 0427.md                # Development log
```

---

## Configuration

Edit `config/settings.yaml`:

```yaml
vector_db:
  type: qdrant
  
  qdrant:
    url: "https://aee6f6e3-503a-4368-9950-ca7599a12bdf.us-west-1-0.aws.cloud.qdrant.io:6333"
    api_key: "your_api_key"
    collection_name: "cnn_weights"

search:
  similarity_threshold: 0.85
  top_k_ratio: 0.3
```

---

## Qdrant Cloud Dashboard

Access your collections at:
```
https://aee6f6e3-503a-4368-9950-ca7599a12bdf.us-west-1-0.aws.cloud.qdrant.io:6333/dashboard#/collections
```

---

## Future Improvements

1. **PCA/dimensionality reduction** for better feature representation
2. **Pretrained model features** (e.g., ResNet embeddings) for deeper similarity
3. **Cross-architecture transfer** - weight sharing between different model architectures
4. **Incremental learning** - continuously update vault with new training

---

## License

MIT License - Research Use

## References

- [Qdrant Cloud](https://qdrant.tech/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- HNSW: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2016)
- He Initialization: He et al., "Delving Deep into Rectifiers" (2015)
