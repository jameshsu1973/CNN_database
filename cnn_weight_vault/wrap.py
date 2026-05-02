"""
通用向量資料庫權重管理系統

功能：
1. 訓練後：從任意 PyTorch 模型提取權重（state_dict），存儲到向量資料庫
2. 初始化時：從向量資料庫檢索權重，載入到相同架構的模型

特點：
- 泛化能力強：支持任意 PyTorch 模型（ResNet, VGG, MobileNet, 自定義等）
- 層類型無關：不管裡面是 Conv2d, Linear, BatchNorm, Embedding 等都能處理
- 基於 state_dict：直接操作模型的權重字典，無需修改模型定義
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import json

from .config import get_config


def extract_weights(model: nn.Module,
                   vault,
                   model_name: str,
                   epoch: int,
                   accuracy: float,
                   object_category: Optional[str] = None,
                   top_k_ratio: float = 0.3) -> int:
    """
    從訓練好的模型中提取所有權重並存儲到向量資料庫。

    Args:
        model: PyTorch 模型（任意架構）
        vault: 向量資料庫實例
        model_name: 模型名稱（用於標識）
        epoch: 當前 epoch
        accuracy: 準確率
        object_category: 對象類別（用於分類/檢索）
        top_k_ratio: Top-k 遮罩比例

    Returns:
        存儲的權重數量

    Example:
        >>> import torchvision.models as models
        >>> from cnn_weight_vault import QdrantWeightVault
        >>>
        >>> # 訓練完成後
        >>> vault = QdrantWeightVault(collection_name="my_vault")
        >>> model = models.resnet18(weights=None)
        >>> model.load_state_dict(trained_weights)
        >>>
        >>> extract_weights(
        ...     model=model,
        ...     vault=vault,
        ...     model_name="resnet18_cat",
        ...     epoch=10,
        ...     accuracy=95.5,
        ...     object_category="cat"
        ... )
    """
    state_dict = model.state_dict()
    stored_count = 0

    for layer_name, weight_tensor in state_dict.items():
        if not isinstance(weight_tensor, torch.Tensor):
            continue

        # 生成層鍵值（基於形狀）
        layer_key = _generate_layer_key(weight_tensor.shape, layer_name)

        # 應用 top-k 遮罩
        masked_weights, mask = _apply_top_k_mask(weight_tensor, top_k_ratio)

        # 轉換為 numpy
        flattened = masked_weights.detach().cpu().numpy().flatten().astype(np.float32)

        # 準備元數據
        metadata = {
            'layer_name': layer_name,
            'model_name': model_name,
            'epoch': epoch,
            'accuracy': accuracy,
            'shape': list(weight_tensor.shape),
            'layer_key': layer_key,
            'object_category': object_category or '',
            'dtype': str(weight_tensor.dtype),
            'mask': mask.cpu().numpy().tolist() if mask is not None else []
        }

        # 存儲到 vault（調用 vault 的原始存儲方法）
        try:
            vault._store_raw(
                vector=flattened,
                layer_key=layer_key,
                metadata=metadata
            )
            stored_count += 1
        except Exception as e:
            print(f"[WeightStore] Warning: Failed to store {layer_name}: {e}")

    print(f"[WeightStore] Stored {stored_count} weight tensors to vault (model={model_name})")
    return stored_count


def load_weights(model: nn.Module,
                vault,
                object_category: Optional[str] = None,
                force: bool = False) -> Dict[str, bool]:
    """
    從向量資料庫載入權重到模型。

    Args:
        model: PyTorch 模型（需要初始化的模型，架構要與存儲時相同）
        vault: 向量資料庫實例
        object_category: 對象類別（用於匹配）
        force: 是否強制載入（True = 忽略相似度閾值）

    Returns:
        Dict[str, bool]: 每層的載入狀態 {layer_name: success}

    Example:
        >>> # 初始化新模型
        >>> vault = QdrantWeightVault(collection_name="my_vault")
        >>> new_model = models.resnet18(weights=None)
        >>>
        >>> results = load_weights(
        ...     model=new_model,
        ...     vault=vault,
        ...     object_category="cat",
        ...     force=False  # False: 只有相似度 > 閾值才載入
        ... )
        >>> # results = {'conv1.weight': True, 'layer1.0.conv1.weight': True, ...}
    """
    state_dict = model.state_dict()
    results = {}

    for layer_name, weight_tensor in state_dict.items():
        if not isinstance(weight_tensor, torch.Tensor):
            continue

        # 生成層鍵值
        layer_key = _generate_layer_key(weight_tensor.shape, layer_name)

        # 從 vault 獲取權重
        weights = vault._get_weights_by_key(
            layer_key=layer_key,
            shape=weight_tensor.shape,
            object_category=object_category,
            force=force
        )

        if weights is not None and weights.shape == weight_tensor.shape:
            with torch.no_grad():
                model.state_dict()[layer_name].copy_(weights)
            results[layer_name] = True
            print(f"[WeightLoader] Layer {layer_name}: Loaded from vault")
        else:
            results[layer_name] = False

    loaded = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"[WeightLoader] Loaded {loaded}/{total} layers from vault")

    return results


def _generate_layer_key(shape: Tuple[int, ...], layer_name: str) -> str:
    """
    根據權重形狀生成唯一的層鍵值。

    這個鍵值用於在 vault 中識別相同類型的層。
    例如：conv_3_64_7_7, linear_512_10, bn_64, embedding_10000_512
    """
    if len(shape) == 4:
        # Conv2d: (out_ch, in_ch, kh, kw)
        return f"conv_{shape[0]}_{shape[1]}_{shape[2]}_{shape[3]}"
    elif len(shape) == 2:
        # Linear: (out_features, in_features)
        return f"linear_{shape[0]}_{shape[1]}"
    elif len(shape) == 1:
        # Bias or 1D weight (Embedding, LayerNorm, etc.)
        if 'bias' in layer_name:
            return f"bias_{shape[0]}"
        else:
            return f"1d_{shape[0]}"
    elif len(shape) == 3:
        # 3D weight (e.g., some LSTM, Transformer)
        return f"3d_{shape[0]}_{shape[1]}_{shape[2]}"
    else:
        # 其他情況
        return f"other_{'_'.join(map(str, shape))}"


def _apply_top_k_mask(weight_tensor: torch.Tensor, top_k_ratio: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    應用 top-k 遮罩，保留最重要的權重。

    Args:
        weight_tensor: 原始權重張量
        top_k_ratio: 保留比例 (0.0-1.0)

    Returns:
        (masked_weights, mask)
    """
    flat_weights = weight_tensor.abs().flatten()
    k = int(top_k_ratio * flat_weights.numel())
    k = max(k, 1)

    _, top_k_indices = torch.topk(flat_weights, k)
    mask = torch.zeros_like(flat_weights, dtype=torch.bool)
    mask[top_k_indices] = True

    masked_weights = weight_tensor.flatten() * mask.float()
    return masked_weights.view_as(weight_tensor), mask.view_as(weight_tensor)


# ============================================================================
# 簡化版：直接存儲和檢索整個模型
# ============================================================================

def save_model_to_vault(model: nn.Module,
                        vault,
                        model_name: str,
                        object_category: str,
                        epoch: int = 0,
                        accuracy: float = 0.0):
    """
    將整個模型的權重作為單一記錄存儲（適用於完整模型檢索）。

    與 extract_weights 的區別：
    - extract_weights: 每層單獨存儲，可單獨檢索
    - save_model_to_vault: 整個模型一起存儲，方便整體檢索

    Args:
        model: PyTorch 模型
        vault: 向量資料庫實例
        model_name: 模型名稱
        object_category: 對象類別
        epoch: 訓練 epoch
        accuracy: 準確率
    """
    if hasattr(vault, 'store_category_weights'):
        # 使用 Qdrant 的 category 方式
        model_state = {k: v for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        vault.store_category_weights(
            model_state=model_state,
            model_name=model_name,
            category=object_category,
            epoch=epoch,
            accuracy=accuracy
        )
    else:
        # 使用通用方式
        extract_weights(model, vault, model_name, epoch, accuracy, object_category)


def load_model_from_vault(model: nn.Module,
                         vault,
                         object_category: str) -> bool:
    """
    從 vault 載入整個模型的權重。

    Args:
        model: PyTorch 模型（需要初始化的模型）
        vault: 向量資料庫實例
        object_category: 對象類別

    Returns:
        是否成功載入
    """
    if hasattr(vault, 'get_category_weights'):
        # 使用 Qdrant 的 category 方式
        weights = vault.get_category_weights(object_category)
        if weights is None:
            print(f"[WeightLoader] No weights found for category '{object_category}'")
            return False

        # 載入權重
        state_dict = model.state_dict()
        loaded_count = 0

        for name, arr in weights.items():
            if name in state_dict:
                tensor = torch.from_numpy(arr)
                if tensor.shape == state_dict[name].shape:
                    with torch.no_grad():
                        state_dict[name].copy_(tensor)
                    loaded_count += 1

        print(f"[WeightLoader] Loaded {loaded_count}/{len(weights)} layers from vault")
        return loaded_count > 0
    else:
        # 使用通用方式
        results = load_weights(model, vault, object_category, force=True)
        return sum(1 for v in results.values() if v) > 0


def find_similar_model(model: nn.Module,
                      vault) -> Tuple[Optional[str], float]:
    """
    從 vault 中找到與當前模型最相似的已存儲模型。

    Args:
        model: PyTorch 模型
        vault: 向量資料庫實例

    Returns:
        (最相似的類別, 相似度分數)
    """
    if hasattr(vault, 'find_similar_category'):
        model_state = {k: v for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        return vault.find_similar_category(model_state)
    else:
        print("[WeightLoader] Vault does not support find_similar_category")
        return None, 0.0


# ============================================================================
# 培訓鉤子 - 自動存儲權重
# ============================================================================

class TrainingHook:
    """
    訓練鉤子 - 在訓練過程中自動存儲權重。

    使用方式：
        hook = TrainingHook(vault, model_name="resnet18", save_interval=5)
        hook.register(model)

        for epoch in range(num_epochs):
            train(model)
            hook.on_epoch_end(model, epoch, accuracy)
    """

    def __init__(self,
                 vault,
                 model_name: str,
                 object_category: Optional[str] = None,
                 save_interval: int = 1,
                 top_k_ratio: float = 0.3):
        self.vault = vault
        self.model_name = model_name
        self.object_category = object_category
        self.save_interval = save_interval
        self.top_k_ratio = top_k_ratio

    def register(self, model: nn.Module):
        """註冊鉤子到模型。"""
        self.model = model
        print(f"[TrainingHook] Registered for model: {self.model_name}")

    def on_epoch_end(self, model: nn.Module, epoch: int, accuracy: float):
        """每個 epoch 結束時調用。"""
        if epoch % self.save_interval == 0:
            extract_weights(
                model,
                self.vault,
                self.model_name,
                epoch,
                accuracy,
                self.object_category,
                self.top_k_ratio
            )

    def save(self):
        """保存 vault 到磁盤。"""
        if hasattr(self.vault, 'save_vault'):
            self.vault.save_vault()


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """
    展示如何使用通用權重管理系統。
    """
    print("""
# ==================== 示例用法 ====================

# 1. 創建 vault
from cnn_weight_vault import QdrantWeightVault, extract_weights, load_weights

vault = QdrantWeightVault(
    collection_name="my_weights",
    url="https://xxx.qdrant.cloud",
    api_key="your_api_key"
)

# ==================== 存儲模式 ====================

# 假設你已經訓練好一個模型
import torchvision.models as models
trained_model = models.resnet18(weights=None)
trained_model.load_state_dict(trained_weights)  # 加載訓練好的權重

# 存儲所有層的權重到 vault
extract_weights(
    model=trained_model,
    vault=vault,
    model_name="resnet18",
    epoch=10,
    accuracy=95.5,
    object_category="cat"
)

vault.save_vault()

# ==================== 載入模式 ====================

# 創建一個新的 ResNet18 模型（相同架構）
new_model = models.resnet18(weights=None)

# 從 vault 載入權重
results = load_weights(
    model=new_model,
    vault=vault,
    object_category="cat",
    force=False  # False: 只有相似度 > 閾值才載入
)

# 結果: {'conv1.weight': True, 'layer1.0.conv1.weight': True, ...}

# ==================== 整體模式 ====================

# 存儲整個模型（方便整體檢索）
save_model_to_vault(
    model=trained_model,
    vault=vault,
    model_name="resnet18_cat",
    object_category="cat",
    epoch=10,
    accuracy=95.5
)

# 載入整個模型
new_model = models.resnet18(weights=None)
success = load_model_from_vault(
    model=new_model,
    vault=vault,
    object_category="cat"
)

# ==================== 查找相似模型 ====================

# 根據已有模型找到 vault 中最相似的類別
new_model = models.resnet18(weights=None)
category, similarity = find_similar_model(new_model, vault)
# category: "cat", similarity: 0.85
    """)