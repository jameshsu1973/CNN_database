"""
Database-Driven Weight Initialization for PyTorch
Integrates Weight Vault with PyTorch layers for automatic weight storage/retrieval

Uses ChromaDB vector database for weight storage and retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
from .chroma_vault import ChromaWeightVault


class DBConv2d(nn.Conv2d):
    """
    Conv2d layer with database-driven initialization.
    Automatically stores weights after training and retrieves from vault for initialization.

    Uses ChromaDB vector database for weight storage and retrieval.
    """

    _vault: Optional[ChromaWeightVault] = None
    _model_name: str = "default_model"
    _layer_counter: int = 0
    _force_load: bool = False  # Force loading from vault, skip He initialization
    _object_category: Optional[str] = None  # Object category for vault matching

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):

        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias,
            padding_mode, device, dtype
        )

        self.layer_id = DBConv2d._layer_counter
        DBConv2d._layer_counter += 1
        self.layer_name = f"conv_{self.layer_id}"

        # Try to initialize from vault
        self._initialize_from_vault()

    def _initialize_from_vault(self):
        """Initialize weights from vault if available, otherwise use He init."""
        if DBConv2d._vault is None:
            # No vault connected - use standard He initialization
            self._he_initialization()
            return

        # If force_load is True, MUST get weights from vault
        if DBConv2d._force_load:
            # Query vault for weights - MUST succeed (pass force=True)
            vault_weights = DBConv2d._vault.get_initialization_weights(
                self, 
                force=True,
                object_category=DBConv2d._object_category
            )
            
            if vault_weights is not None and vault_weights.shape == self.weight.shape:
                self.weight.data = vault_weights
                print(f"[DBInit] Layer {self.layer_name}: Loaded from vault (cat={DBConv2d._object_category}, shape={vault_weights.shape}) [FORCED]")
            else:
                # Force load failed - still use He but warn
                self._he_initialization()
                if vault_weights is None:
                    print(f"[DBInit] Layer {self.layer_name}: Vault empty (cat={DBConv2d._object_category}) - using He init")
                else:
                    print(f"[DBInit] Layer {self.layer_name}: Shape mismatch (vault={vault_weights.shape if vault_weights else 'None'}, layer={self.weight.shape}) - using He")
            return

        # Normal mode: try vault first, fall back to He
        vault_weights = DBConv2d._vault.get_initialization_weights(self, force=False)

        if vault_weights is not None:
            # Found matching weights in vault - use them!
            if vault_weights.shape == self.weight.shape:
                self.weight.data = vault_weights
                print(f"[DBInit] Layer {self.layer_name}: Initialized from vault")
            else:
                # Shape mismatch - fall back to He
                self._he_initialization()
        else:
            # No match in vault - use He initialization (Cold Start fallback)
            self._he_initialization()
            print(f"[DBInit] Layer {self.layer_name}: Cold start - using He initialization")

    def _he_initialization(self):
        """Standard He initialization for cold start."""
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    @classmethod
    def set_vault(cls, vault: ChromaWeightVault):
        """Set the global ChromaDB vault for all DBConv2d layers."""
        cls._vault = vault

    @classmethod
    def set_model_name(cls, name: str):
        """Set the model name for vault entries."""
        cls._model_name = name

    @classmethod
    def set_force_load(cls, force: bool):
        """Set force load mode - must get weights from vault."""
        cls._force_load = force

    @classmethod
    def set_object_category(cls, category: str):
        """Set object category for vault matching."""
        cls._object_category = category

    @classmethod
    def reset_layer_counter(cls):
        """Reset layer counter (call before creating a new model)."""
        cls._layer_counter = 0

    def store_weights(self, epoch: int, accuracy: Optional[float] = None, object_category: str = None, num_objects: int = 0):
        """Store current weights to the vault."""
        if DBConv2d._vault is not None:
            DBConv2d._vault.store_weights(
                self, self.layer_name, DBConv2d._model_name, epoch, accuracy,
                object_category=object_category, num_objects=num_objects
            )


class DBLinear(nn.Linear):
    """
    Linear layer with database-driven initialization.

    Uses ChromaDB vector database for weight storage and retrieval.
    """

    _vault: Optional[ChromaWeightVault] = None
    _model_name: str = "default_model"
    _layer_counter: int = 0
    _force_load: bool = False
    _object_category: Optional[str] = None

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None):

        super().__init__(in_features, out_features, bias, device, dtype)

        self.layer_id = DBLinear._layer_counter
        DBLinear._layer_counter += 1
        self.layer_name = f"linear_{self.layer_id}"

        self._initialize_from_vault()

    def _initialize_from_vault(self):
        """Initialize weights from vault if available."""
        if DBLinear._vault is None:
            self._he_initialization()
            return

        # If force_load is True, MUST get weights from vault
        if DBLinear._force_load:
            vault_weights = DBLinear._vault.get_initialization_weights(
                self, 
                force=True,
                object_category=DBLinear._object_category
            )
            
            if vault_weights is not None and vault_weights.shape == self.weight.shape:
                self.weight.data = vault_weights
                print(f"[DBInit] Layer {self.layer_name}: Loaded from vault (cat={DBLinear._object_category}, shape={vault_weights.shape}) [FORCED]")
            else:
                self._he_initialization()
                if vault_weights is None:
                    print(f"[DBInit] Layer {self.layer_name}: Vault empty (cat={DBLinear._object_category}) - using He init")
                else:
                    print(f"[DBInit] Layer {self.layer_name}: Shape mismatch (vault={vault_weights.shape if vault_weights else 'None'}, layer={self.weight.shape}) - using He")
            return

        # Normal mode: try vault first, fall back to He
        vault_weights = DBLinear._vault.get_initialization_weights(self, force=False)

        if vault_weights is not None and vault_weights.shape == self.weight.shape:
            self.weight.data = vault_weights
            print(f"[DBInit] Layer {self.layer_name}: Initialized from vault")
        else:
            self._he_initialization()
            if vault_weights is None:
                print(f"[DBInit] Layer {self.layer_name}: Cold start - using He initialization")

    def _he_initialization(self):
        """Standard He initialization."""
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    @classmethod
    def set_vault(cls, vault: ChromaWeightVault):
        """Set the global vault for all DBLinear layers (WeightVault or ChromaWeightVault)."""
        cls._vault = vault

    @classmethod
    def set_model_name(cls, name: str):
        """Set the model name for vault entries."""
        cls._model_name = name

    @classmethod
    def set_force_load(cls, force: bool):
        """Set force load mode - must get weights from vault."""
        cls._force_load = force

    @classmethod
    def set_object_category(cls, category: str):
        """Set object category for vault matching."""
        cls._object_category = category

    @classmethod
    def reset_layer_counter(cls):
        """Reset layer counter."""
        cls._layer_counter = 0

    def store_weights(self, epoch: int, accuracy: Optional[float] = None, object_category: str = None, num_objects: int = 0):
        """Store current weights to the vault."""
        if DBLinear._vault is not None:
            DBLinear._vault.store_weights(
                self, self.layer_name, DBLinear._model_name, epoch, accuracy,
                object_category=object_category, num_objects=num_objects
            )


class DBModelWrapper:
    """
    Wrapper for PyTorch models to enable database-driven training.
    Hooks into training loop to automatically store weights after epochs.

    Uses ChromaDB vector database for weight storage and retrieval.
    """

    def __init__(self, model: nn.Module, vault: ChromaWeightVault,
                 model_name: str = "model"):
        self.model = model
        self.vault = vault
        self.model_name = model_name
        self.epoch = 0
        self.accuracy = 0.0

        # Set up vault for all DB layers
        DBConv2d.set_vault(vault)
        DBConv2d.set_model_name(model_name)
        DBLinear.set_vault(vault)
        DBLinear.set_model_name(model_name)

    def prepare_model(self) -> nn.Module:
        """
        Prepare the model by replacing standard layers with DB-aware layers.
        This creates a new model with DBConv2d/DBLinear instead of Conv2d/Linear.
        """
        # Note: In a full implementation, this would traverse the model
        # and replace layers. For now, the user should define models
        # using DBConv2d and DBLinear directly.
        return self.model

    def store_epoch_weights(self, epoch: int, accuracy: float = 0.0):
        """
        Store weights from all DB-aware layers after an epoch.
        Call this after each training epoch.
        """
        self.epoch = epoch
        self.accuracy = accuracy

        stored_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (DBConv2d, DBLinear)):
                module.store_weights(epoch, accuracy)
                stored_count += 1

        if stored_count > 0:
            print(f"[Vault] Stored weights from {stored_count} layers (epoch {epoch}, acc {accuracy:.2f}%)")

    def save_vault(self):
        """Save the vault to disk."""
        self.vault.save_vault()

    def get_vault_stats(self) -> Dict[str, Any]:
        """Get vault statistics."""
        return self.vault.get_stats()


def create_db_cnn(vault: ChromaWeightVault, num_classes: int = 10, force_load: bool = False, object_category: str = None) -> nn.Module:
    """
    Create a simple CNN using database-driven layers.

    Args:
        vault: The WeightVault instance
        num_classes: Number of output classes
        force_load: If True, must load weights from vault (skip He init)
        object_category: The object category being trained (for vault matching)

    Returns:
        A CNN model with DB-aware layers
    """
    DBConv2d.reset_layer_counter()
    DBLinear.reset_layer_counter()
    DBConv2d.set_vault(vault)
    DBLinear.set_vault(vault)
    DBConv2d.set_force_load(force_load)
    DBLinear.set_force_load(force_load)
    DBConv2d.set_object_category(object_category)
    DBLinear.set_object_category(object_category)

    class DBCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            # CIFAR-10: 3 input channels
            self.conv1 = DBConv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = DBConv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)

            # Calculate flattened size for CIFAR-10 (32x32 -> 8x8 after 2 pools)
            self.fc1 = DBLinear(64 * 8 * 8, 128)
            self.fc2 = DBLinear(128, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return x

    return DBCNN(num_classes)


def he_initialization_fallback(layer: nn.Module):
    """
    Mathematical Safety Net - fallback to He initialization.
    Equations from the paper:
        W_init(t) ~ N(0, 2/n_in)
        sigma_l = sqrt(2 / n_in)
        w_ij <- z_ij * sigma_l
    """
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        # Calculate fan_in
        if isinstance(layer, nn.Conv2d):
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            fan_in = layer.in_features

        # He initialization (Eq 5-7)
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(layer.weight, mean=0.0, std=std)

        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
