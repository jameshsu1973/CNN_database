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

# Database-aware layers (unified for classification & binary classification)
from .db_initialization import DBConv2d, DBLinear, DBModelWrapper, create_db_cnn

__all__ = [
    # ChromaDB vault (local)
    'ChromaWeightVault',
    
    # Milvus vault (cloud)
    'MilvusWeightVault',
    
    # Qdrant vault (cloud)
    'QdrantWeightVault',

    # Layers
    'DBConv2d',
    'DBLinear',
    'DBModelWrapper',
    'create_db_cnn',

    # Configuration
    'get_config',
    'reload_config',
    'Config',
]