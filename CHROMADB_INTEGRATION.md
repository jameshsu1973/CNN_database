# ChromaDB Vector Database Integration

This document describes the ChromaDB-based vector database implementation for CNN Weight Vault.

## Overview

The system now supports **ChromaDB** as the primary vector database backend, replacing the previous pickle-based storage. ChromaDB provides:

- **Native vector storage** with HNSW indexing
- **Cosine similarity search** (1 - cosine_distance)
- **Automatic persistence** (DuckDB + Parquet backend)
- **Metadata filtering** for object categories
- **Scalable architecture** (supports millions of vectors)

## Architecture

### Collection Structure

```
chroma_db/
├── chroma.sqlite3          # DuckDB metadata storage
├── *.parquet              # Vector data files
└── index/                 # HNSW indexes
```

Each layer type gets its own collection:
- `cnn_weights_conv` - Conv2d layers
- `cnn_weights_linear` - Linear layers
- `cnn_weights_det_conv` - Detection Conv2d layers
- `cnn_weights_det_linear` - Detection Linear layers
- `cnn_weights_backbone_conv` - Backbone Conv2d layers

### Data Flow

```
Store Weights:
    Layer → Top-K Mask → Flatten → ChromaDB Collection

Query Weights:
    Query Vector → ChromaDB Search → Cosine Similarity → Return Top-K
```

## Configuration

### Configuration File: `config/settings.yaml`

```yaml
vector_db:
  type: chromadb
  chromadb:
    persist_directory: "./chroma_db"
    collection_name: "cnn_weights"
    distance_metric: "cosine"
    embedding_dimension: 512

search:
  default_top_k: 3
  similarity_threshold: 0.85
  top_k_ratio: 0.3
```

### Environment Variables

- `CNN_VAULT_CONFIG` - Path to custom config file

## Usage

### Basic Usage

```python
from cnn_weight_vault.chroma_vault import ChromaWeightVault
from cnn_weight_vault.config import get_config

# Create vault with config
config = get_config()
vault = ChromaWeightVault(
    collection_name=config.chroma_collection_name,
    persist_directory=config.chroma_persist_dir
)

# Store weights
vault.store_weights(
    layer=conv_layer,
    layer_name="conv1",
    model_name="my_model",
    epoch=10,
    accuracy=92.5
)

# Query weights
weights = vault.get_initialization_weights(
    layer=new_conv_layer,
    layer_name="conv1"
)

if weights is not None:
    new_conv_layer.weight.data = weights
```

### With Object Categories (Detection)

```python
vault.store_weights(
    layer=conv_layer,
    layer_name="head_conv",
    model_name="cat_detector",
    epoch=5,
    accuracy=88.3,
    object_category="cat",  # Tag with object category
    num_objects=42
)

# Query with category preference
weights = vault.get_initialization_weights(
    layer=new_layer,
    layer_name="head_conv",
    object_category="cat"  # Prefer "cat" weights
)
```

## Migration from Pickle

### Automatic Migration Script

```bash
# Migrate all default vaults
python scripts/migrate_to_chroma.py

# Migrate specific vault
python scripts/migrate_to_chroma.py ./vault/vault.pkl ./chroma_db
```

### Programmatic Migration

```python
from cnn_weight_vault.chroma_vault import ChromaWeightVault

vault = ChromaWeightVault()
count = vault.migrate_from_pickle("./vault/vault.pkl")
print(f"Migrated {count} entries")
```

## API Reference

### ChromaWeightVault

```python
class ChromaWeightVault:
    def __init__(self,
                 collection_name: Optional[str] = None,
                 persist_directory: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 top_k_ratio: Optional[float] = None)

    def store_weights(self,
                     layer: nn.Module,
                     layer_name: str,
                     model_name: str,
                     epoch: int,
                     accuracy: Optional[float] = None,
                     object_category: Optional[str] = None,
                     num_objects: int = 0)

    def query_similar_weights(self,
                              layer: nn.Module,
                              layer_name: str = "",
                              object_category: Optional[str] = None,
                              k: int = 3) -> Optional[List[Dict]]

    def get_initialization_weights(self,
                                   layer: nn.Module,
                                   layer_name: str = "",
                                   object_category: Optional[str] = None,
                                   force: bool = False) -> Optional[torch.Tensor]

    def has_weights_for_layer(self, layer: nn.Module, layer_name: str = "") -> bool

    def save_vault(self)

    def load_vault(self) -> bool

    def get_stats(self) -> Dict[str, Any]

    def migrate_from_pickle(self, pickle_vault_path: str) -> int
```

### Configuration API

```python
from cnn_weight_vault.config import get_config, reload_config

config = get_config()

# Access values with dot notation
persist_dir = config.get('vector_db.chromadb.persist_directory')
threshold = config.similarity_threshold

# Reload config from file
reload_config()
```

## Comparison: Pickle vs ChromaDB

| Feature | Pickle | ChromaDB |
|---------|--------|----------|
| Storage | Single file | Multiple files (DB) |
| Search | O(n) linear scan | O(log n) HNSW |
| Persistence | Manual pickle | Auto (DuckDB+Parquet) |
| Scalability | Limited by RAM | Disk-backed, scalable |
| Metadata filtering | Manual | Built-in query API |
| Distance metrics | Cosine only | Cosine, L2, IP |
| ACID | No | Yes (via DuckDB) |

## Troubleshooting

### ChromaDB Not Found

```bash
pip install chromadb>=0.4.22
```

### Permission Errors on Windows

Run the migration script with administrator privileges, or change persist_directory to a user-writable location:

```yaml
vector_db:
  chromadb:
    persist_directory: "C:/Users/<username>/chroma_db"
```

### Collection Already Exists

ChromaDB handles this automatically with `get_or_create_collection()`.

## Performance Notes

- **HNSW Index**: Built automatically by ChromaDB
- **Memory Usage**: Vectors are memory-mapped, not all loaded
- **Query Speed**: Typically <10ms for 100K+ vectors
- **Write Speed**: ~100-1000 vectors/second (batching recommended)
