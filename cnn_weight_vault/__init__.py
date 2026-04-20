"""
CNN Weight Vault - Database-Driven CNN Weight Initialization

A PyTorch extension that replaces random weight initialization with
a queryable, cumulative database of optimized historical weights.

Uses ChromaDB vector database for efficient similarity search.
"""

__version__ = "2.0.0"

# ChromaDB-based vault (unified vector database)
from .chroma_vault import ChromaWeightVault

# Configuration
from .config import get_config, reload_config, Config

# Database-aware layers for classification
from .db_initialization import DBConv2d, DBLinear, DBModelWrapper, create_db_cnn

# Database-aware layers for object detection
from .detection_model import (
    DBConv2dDetect, DBLinearDetect,
    SimpleDetectionNet, DetectionModelWrapper,
    create_detection_model, convert_to_db_layers
)

__all__ = [
    # ChromaDB vault (unified)
    'ChromaWeightVault',

    # Classification layers
    'DBConv2d',
    'DBLinear',
    'DBModelWrapper',
    'create_db_cnn',

    # Detection layers
    'DBConv2dDetect',
    'DBLinearDetect',
    'SimpleDetectionNet',
    'DetectionModelWrapper',
    'create_detection_model',
    'convert_to_db_layers',

    # Configuration
    'get_config',
    'reload_config',
    'Config',
]