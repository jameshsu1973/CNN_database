"""
CNN Weight Vault - Database-Driven CNN Weight Initialization

A PyTorch extension that replaces random weight initialization with
a queryable, cumulative database of optimized historical weights.

Uses ChromaDB vector database for efficient similarity search.
"""

__version__ = "2.0.0"

# ChromaDB-based vault (local vector database)
from .chroma_vault import ChromaWeightVault

# Milvus/Zilliz Cloud-based vault (cloud vector database)
from .milvus_vault import MilvusWeightVault

# Qdrant Cloud-based vault (cloud vector database with higher dimension limit)
from .qdrant_vault import QdrantWeightVault

# Configuration
from .config import get_config, reload_config, Config

# Qdrant vault (cloud)
from .qdrant_vault import QdrantWeightVault

# Configuration
from .config import get_config, reload_config, Config

# Generic weight management (works with any PyTorch model)
from .wrap import (
    extract_weights,
    load_weights,
    save_model_to_vault,
    load_model_from_vault,
    find_similar_model,
    TrainingHook
)

__all__ = [
    # Qdrant vault (cloud) - only supported now
    'QdrantWeightVault',

    # Generic weight management
    'extract_weights',
    'load_weights',
    'save_model_to_vault',
    'load_model_from_vault',
    'find_similar_model',
    'TrainingHook',

    # Configuration
    'get_config',
    'reload_config',
    'Config',
]