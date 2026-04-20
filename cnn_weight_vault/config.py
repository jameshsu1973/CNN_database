"""
Configuration management for CNN Weight Vault
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration singleton for CNN Weight Vault."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from YAML file."""
        # Default config path
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

        # Check for environment variable override
        env_config = os.environ.get("CNN_VAULT_CONFIG")
        if env_config:
            config_path = Path(env_config)

        # Load YAML config
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        else:
            # Fallback to default config
            self._config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if file not found."""
        return {
            'vector_db': {
                'type': 'chromadb',
                'chromadb': {
                    'persist_directory': './chroma_db',
                    'collection_name': 'cnn_weights',
                    'distance_metric': 'cosine',
                    'embedding_dimension': 512
                }
            },
            'search': {
                'default_top_k': 3,
                'similarity_threshold': 0.85,
                'top_k_ratio': 0.3
            },
            'model': {
                'default_model_name': 'default_model'
            },
            'vault': {
                'default_path': './vault',
                'detection_path': './detection_vault'
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        Example: config.get('vector_db.chromadb.persist_directory')
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        return self._config.get('vector_db', {})

    def get_chromadb_config(self) -> Dict[str, Any]:
        """Get ChromaDB specific configuration."""
        vector_db = self.get_vector_db_config()
        return vector_db.get('chromadb', {})

    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration."""
        return self._config.get('search', {})

    def get_vault_path(self, vault_type: str = 'default') -> str:
        """Get vault storage path."""
        vault_config = self._config.get('vault', {})
        if vault_type == 'detection':
            return vault_config.get('detection_path', './detection_vault')
        return vault_config.get('default_path', './vault')

    @property
    def similarity_threshold(self) -> float:
        return self.get('search.similarity_threshold', 0.85)

    @property
    def top_k_ratio(self) -> float:
        return self.get('search.top_k_ratio', 0.3)

    @property
    def default_top_k(self) -> int:
        return self.get('search.default_top_k', 3)

    @property
    def chroma_persist_dir(self) -> str:
        return self.get('vector_db.chromadb.persist_directory', './chroma_db')

    @property
    def chroma_collection_name(self) -> str:
        return self.get('vector_db.chromadb.collection_name', 'cnn_weights')

    @property
    def distance_metric(self) -> str:
        return self.get('vector_db.chromadb.distance_metric', 'cosine')


# Global config instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config():
    """Reload configuration from file."""
    global _config_instance
    _config_instance = Config()
