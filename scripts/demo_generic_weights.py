"""
Generic Weight Management Demo - 使用任意 PyTorch 模型

展示如何使用通用權重管理系統：
1. 訓練後：從任意 PyTorch 模型提取權重，存儲到 Qdrant
2. 初始化時：從 Qdrant 檢索權重，載入到相同架構的模型

特點：
- 支持任意模型（ResNet, VGG, MobileNet, 自定義等）
- 無需修改模型定義
- 基於 state_dict 操作
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cnn_weight_vault import QdrantWeightVault, extract_weights, load_weights


def demo_simple():
    """簡單演示：ResNet18 權重的存儲和載入"""
    print("=" * 60)
    print("Demo: Generic Weight Management with ResNet18")
    print("=" * 60)

    # 1. 創建 vault
    print("\n[1] Creating Qdrant vault...")
    vault = QdrantWeightVault(
        collection_name="demo_resnet18",
        similarity_threshold=0.3
    )

    # 2. 創建一個"訓練好"的 ResNet18
    print("\n[2] Creating a 'trained' ResNet18 model...")
    trained_model = models.resnet18(weights=None)

    # 模擬訓練後的權重（隨機初始化作為示例）
    # 實際使用時，這裡是你訓練好的權重
    for name, param in trained_model.named_parameters():
        if len(param.shape) >= 2:
            # 只有 2 维及以上的参数才能用 xavier_uniform_
            nn.init.xavier_uniform_(param)
        else:
            # BatchNorm 的 weight/bias 用 default 初始化
            nn.init.zeros_(param)

    # 3. 存儲權重到 vault
    print("\n[3] Storing weights to vault...")
    extract_weights(
        model=trained_model,
        vault=vault,
        model_name="resnet18_trained",
        epoch=10,
        accuracy=95.5,
        object_category="cat"
    )

    # 4. 創建新的 ResNet18（相同架構，未訓練）
    print("\n[4] Creating new ResNet18 model (same architecture)...")
    new_model = models.resnet18(weights=None)

    # 5. 從 vault 載入權重
    print("\n[5] Loading weights from vault...")
    results = load_weights(
        model=new_model,
        vault=vault,
        object_category="cat",
        force=False  # False: 相似度閾值內才載入
    )

    # 6. 統計結果
    loaded = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n[Result] Loaded {loaded}/{total} layers from vault")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def demo_compare_layers():
    """演示：展示不同模型的層結構"""
    print("\n" + "=" * 60)
    print("Demo: Compare Layer Structures")
    print("=" * 60)

    test_models = [
        ('resnet18', models.resnet18(weights=None)),
        ('resnet50', models.resnet50(weights=None)),
        ('vgg16', models.vgg16(weights=None)),
        ('mobilenet_v2', models.mobilenet_v2(weights=None)),
    ]

    for name, model in test_models:
        state_dict = model.state_dict()
        print(f"\n{name}:")
        print(f"  Total layers: {len(state_dict)}")

        # 統計不同類型的層
        layer_types = {}
        for key in state_dict.keys():
            shape = state_dict[key].shape
            layer_type = _get_layer_type(shape)
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

        for lt, count in sorted(layer_types.items()):
            print(f"    {lt}: {count}")


def _get_layer_type(shape):
    """根據形狀判斷層類型"""
    if len(shape) == 4:
        return f"conv_{shape[0]}_{shape[1]}_{shape[2]}x{shape[3]}"
    elif len(shape) == 2:
        return f"linear_{shape[0]}x{shape[1]}"
    elif len(shape) == 1:
        return f"1d_{shape[0]}"
    else:
        return f"other_{len(shape)}d"


def demo_custom_model():
    """演示：自定義模型的權重管理"""
    print("\n" + "=" * 60)
    print("Demo: Custom Model")
    print("=" * 60)

    # 定義一個簡單的自定義模型
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 1. 創建 vault
    vault = QdrantWeightVault(collection_name="demo_custom")

    # 2. 訓練（模擬）
    trained_model = SimpleCNN(num_classes=10)
    for name, param in trained_model.named_parameters():
        if len(param.shape) >= 2:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)

    # 3. 存儲
    extract_weights(
        model=trained_model,
        vault=vault,
        model_name="simple_cnn",
        epoch=5,
        accuracy=85.0,
        object_category="bird"
    )

    # 4. 載入
    new_model = SimpleCNN(num_classes=10)
    results = load_weights(
        model=new_model,
        vault=vault,
        object_category="bird",
        force=False
    )

    loaded = sum(1 for v in results.values() if v)
    print(f"\n[Result] Loaded {loaded}/{len(results)} layers")


def demo_force_load():
    """演示：強制載入模式"""
    print("\n" + "=" * 60)
    print("Demo: Force Load Mode")
    print("=" * 60)

    vault = QdrantWeightVault(collection_name="demo_force")

    # 創建和存儲
    model1 = models.resnet18(weights=None)
    for name, param in model1.named_parameters():
        if len(param.shape) >= 2:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)

    extract_weights(model1, vault, "resnet18", 10, 90.0, "cat")

    # 創建新模型並強制載入（忽略相似度閾值）
    model2 = models.resnet18(weights=None)
    results = load_weights(model2, vault, object_category="cat", force=True)

    loaded = sum(1 for v in results.values() if v)
    print(f"[Force] Loaded {loaded}/{len(results)} layers (force=True)")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║     Generic Weight Management Demo - Qdrant Cloud        ║
╠════════════════════════════════════════════════════════════╣
║  1. extract_weights(model, vault, ...)  -> 存儲權重        ║
║  2. load_weights(model, vault, ...)    -> 載入權重        ║
╚════════════════════════════════════════════════════════════╝
    """)

    # 檢查 Qdrant 配置
    from cnn_weight_vault.config import get_config
    config = get_config()

    if not config.get('vector_db.qdrant.url') or not config.get('vector_db.qdrant.api_key'):
        print("ERROR: Please configure Qdrant in config/settings.yaml")
        print("""
配置示例 (config/settings.yaml):
```yaml
vector_db:
  qdrant:
    url: "https://xxx.qdrant.cloud:6333"
    api_key: "your_api_key"
```
        """)
        sys.exit(1)

    # 執行演示
    demo_compare_layers()  # 先看看不同模型的層結構
    demo_simple()           # 簡單演示
    demo_custom_model()     # 自定義模型
    demo_force_load()       # 強制載入模式

    print("\nAll demos completed!")