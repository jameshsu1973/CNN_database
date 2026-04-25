from cnn_weight_vault.config import get_config
from cnn_weight_vault.milvus_vault import MilvusWeightVault

config = get_config()
milvus_uri = config.get('vector_db.milvus.uri')
milvus_token = config.get('vector_db.milvus.token')

vault = MilvusWeightVault(uri=milvus_uri, token=milvus_token, collection_name='cnn_weights')
print('Collections:')
try:
    for coll in vault.client.list_collections():
        print(f'  - {coll}')
except Exception as e:
    print(f'Error: {e}')