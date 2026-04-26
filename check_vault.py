from cnn_weight_vault.qdrant_vault import QdrantWeightVault
from cnn_weight_vault.config import get_config

config = get_config()
vault = QdrantWeightVault(
    url=config.get('vector_db.qdrant.url'),
    api_key=config.get('vector_db.qdrant.api_key')
)

stats = vault.get_stats()
print("Collections:")
for name, dim in vault.collection_dims.items():
    print(f"  {name}: dim={dim}")