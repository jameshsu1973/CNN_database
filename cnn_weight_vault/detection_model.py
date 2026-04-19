"""
Object Detection Model with Database-Driven Weight Initialization
Simple YOLO-like architecture for single-class object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple
from .detection_vault import DetectionWeightVault


class DBConv2dDetect(nn.Conv2d):
    """
    Conv2d layer for object detection with database-driven initialization.
    Supports backbone and detection head layers.
    """

    _vault: Optional[DetectionWeightVault] = None
    _model_name: str = "detection_model"
    _object_category: Optional[str] = None
    _layer_counter: int = 0
    _force_load: bool = False  # If True, always use vault weights if available

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

        self.layer_id = DBConv2dDetect._layer_counter
        DBConv2dDetect._layer_counter += 1
        self.layer_name = f"conv_{self.layer_id}"

        # Initialize from vault
        self._initialize_from_vault()

    def _initialize_from_vault(self):
        """Initialize weights from vault if available."""
        if DBConv2dDetect._vault is None:
            self._he_initialization()
            return

        # Check if vault has weights for this layer
        has_weights = DBConv2dDetect._vault.has_weights_for_layer(self, self.layer_name)

        if has_weights and DBConv2dDetect._force_load:
            # Force mode: always load from vault
            vault_weights = DBConv2dDetect._vault.get_initialization_weights(
                self, self.layer_name, DBConv2dDetect._object_category, force=True
            )
            if vault_weights is not None and vault_weights.shape == self.weight.shape:
                self.weight.data = vault_weights
                print(f"[DBInit] Layer {self.layer_name}: Initialized from vault [FORCED] "
                      f"(category: {DBConv2dDetect._object_category or 'any'})")
                return

        # Normal mode: check similarity
        vault_weights = DBConv2dDetect._vault.get_initialization_weights(
            self, self.layer_name, DBConv2dDetect._object_category, force=False
        )

        if vault_weights is not None:
            if vault_weights.shape == self.weight.shape:
                self.weight.data = vault_weights
                print(f"[DBInit] Layer {self.layer_name}: Initialized from vault "
                      f"(category: {DBConv2dDetect._object_category or 'any'})")
            else:
                self._he_initialization()
        else:
            self._he_initialization()
            cat_str = f" for {DBConv2dDetect._object_category}" if DBConv2dDetect._object_category else ""
            print(f"[DBInit] Layer {self.layer_name}: Cold start{cat_str} - using He initialization")

    def _he_initialization(self):
        """Standard He initialization for cold start."""
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    @classmethod
    def set_vault(cls, vault: DetectionWeightVault):
        """Set the global vault."""
        cls._vault = vault

    @classmethod
    def set_model_name(cls, name: str):
        """Set the model name."""
        cls._model_name = name

    @classmethod
    def set_object_category(cls, category: Optional[str]):
        """Set the object category (e.g., 'cat', 'dog')."""
        cls._object_category = category

    @classmethod
    def set_force_load(cls, force: bool):
        """Set force load mode. If True, always use vault weights when available."""
        cls._force_load = force

    @classmethod
    def reset_layer_counter(cls):
        """Reset layer counter."""
        cls._layer_counter = 0

    def store_weights(self, epoch: int, accuracy: Optional[float] = None):
        """Store current weights to the vault."""
        if DBConv2dDetect._vault is not None:
            DBConv2dDetect._vault.store_weights(
                self, self.layer_name, DBConv2dDetect._model_name,
                epoch, accuracy, DBConv2dDetect._object_category
            )


class DBLinearDetect(nn.Linear):
    """Linear layer for object detection with database-driven initialization."""

    _vault: Optional[DetectionWeightVault] = None
    _model_name: str = "detection_model"
    _object_category: Optional[str] = None
    _layer_counter: int = 0
    _force_load: bool = False

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None):

        super().__init__(in_features, out_features, bias, device, dtype)

        self.layer_id = DBLinearDetect._layer_counter
        DBLinearDetect._layer_counter += 1
        self.layer_name = f"linear_{self.layer_id}"

        self._initialize_from_vault()

    def _initialize_from_vault(self):
        """Initialize weights from vault."""
        if DBLinearDetect._vault is None:
            self._he_initialization()
            return

        # Check if vault has weights for this layer
        has_weights = DBLinearDetect._vault.has_weights_for_layer(self, self.layer_name)

        if has_weights and DBLinearDetect._force_load:
            # Force mode: always load from vault
            vault_weights = DBLinearDetect._vault.get_initialization_weights(
                self, self.layer_name, DBLinearDetect._object_category, force=True
            )
            if vault_weights is not None and vault_weights.shape == self.weight.shape:
                self.weight.data = vault_weights
                print(f"[DBInit] Layer {self.layer_name}: Initialized from vault [FORCED] "
                      f"(category: {DBLinearDetect._object_category or 'any'})")
                return

        # Normal mode: check similarity
        vault_weights = DBLinearDetect._vault.get_initialization_weights(
            self, self.layer_name, DBLinearDetect._object_category, force=False
        )

        if vault_weights is not None and vault_weights.shape == self.weight.shape:
            self.weight.data = vault_weights
            print(f"[DBInit] Layer {self.layer_name}: Initialized from vault "
                  f"(category: {DBLinearDetect._object_category or 'any'})")
        else:
            self._he_initialization()
            cat_str = f" for {DBLinearDetect._object_category}" if DBLinearDetect._object_category else ""
            print(f"[DBInit] Layer {self.layer_name}: Cold start{cat_str} - using He initialization")

    def _he_initialization(self):
        """Standard He initialization."""
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    @classmethod
    def set_vault(cls, vault: DetectionWeightVault):
        cls._vault = vault

    @classmethod
    def set_model_name(cls, name: str):
        cls._model_name = name

    @classmethod
    def set_object_category(cls, category: Optional[str]):
        cls._object_category = category

    @classmethod
    def set_force_load(cls, force: bool):
        """Set force load mode. If True, always use vault weights when available."""
        cls._force_load = force

    @classmethod
    def reset_layer_counter(cls):
        cls._layer_counter = 0

    def store_weights(self, epoch: int, accuracy: Optional[float] = None):
        """Store current weights to the vault."""
        if DBLinearDetect._vault is not None:
            DBLinearDetect._vault.store_weights(
                self, self.layer_name, DBLinearDetect._model_name,
                epoch, accuracy, DBLinearDetect._object_category
            )


class SimpleDetectionNet(nn.Module):
    """
    Simple object detection network for single-class detection.
    Backbone: Conv layers for feature extraction
    Head: Predicts bounding box (x, y, w, h) + confidence
    """

    def __init__(self, num_classes: int = 1, grid_size: int = 7):
        """
        Args:
            num_classes: Number of object classes (1 for single-class detection)
            grid_size: Divide image into grid_size x grid_size cells
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes

        # Backbone: Feature extraction
        self.conv1 = DBConv2dDetect(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = DBConv2dDetect(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = DBConv2dDetect(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = DBConv2dDetect(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # Detection Head
        # For 224x224 input with 4 poolings: 224 -> 112 -> 56 -> 28 -> 14
        self.det_conv = DBConv2dDetect(256, 512, kernel_size=3, padding=1)
        self.det_bn = nn.BatchNorm2d(512)

        # Final detection layer: predicts (x, y, w, h, confidence) for each grid cell
        # Output shape: (batch, 5, grid_size, grid_size)
        self.det_head = DBConv2dDetect(512, 5, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images (batch, 3, 224, 224)

        Returns:
            predictions: (batch, grid_size, grid_size, 5)
                - [:, :, :, 0:2]: bounding box center (x, y) relative to cell
                - [:, :, :, 2:4]: bounding box size (w, h) relative to image
                - [:, :, :, 4]: object confidence
        """
        # Backbone
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 112x112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 56x56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 28x28
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 14x14

        # Detection head
        x = F.relu(self.det_bn(self.det_conv(x)))  # 14x14, 512 channels

        # Global average pooling to grid_size x grid_size
        x = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))  # (batch, 512, 7, 7)

        # Predict for each grid cell: (x, y, w, h, confidence)
        # Use 1x1 conv to reduce channels to 5
        x = self.det_head(x)  # (batch, 5, 7, 7)

        # Permute to (batch, grid_size, grid_size, 5)
        x = x.permute(0, 2, 3, 1)

        # Apply sigmoid to get values in [0, 1]
        x = torch.sigmoid(x)

        return x


class DetectionModelWrapper:
    """Wrapper for object detection models to enable database-driven training."""

    def __init__(self, model: nn.Module, vault: DetectionWeightVault,
                 model_name: str = "detection_model",
                 object_category: Optional[str] = None,
                 force_load: bool = False):
        self.model = model
        self.vault = vault
        self.model_name = model_name
        self.object_category = object_category
        self.force_load = force_load
        self.epoch = 0
        self.accuracy = 0.0

        # Set up vault for all DB layers
        DBConv2dDetect.set_vault(vault)
        DBConv2dDetect.set_model_name(model_name)
        DBConv2dDetect.set_object_category(object_category)
        DBConv2dDetect.set_force_load(force_load)
        DBLinearDetect.set_vault(vault)
        DBLinearDetect.set_model_name(model_name)
        DBLinearDetect.set_object_category(object_category)
        DBLinearDetect.set_force_load(force_load)

    def store_epoch_weights(self, epoch: int, accuracy: float = 0.0):
        """Store weights from all DB-aware layers after an epoch."""
        self.epoch = epoch
        self.accuracy = accuracy

        stored_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (DBConv2dDetect, DBLinearDetect)):
                module.store_weights(epoch, accuracy)
                stored_count += 1

        if stored_count > 0:
            print(f"[Vault] Stored weights from {stored_count} layers "
                  f"(epoch {epoch}, mAP {accuracy:.2f}%)")

    def save_vault(self):
        """Save the vault to disk."""
        self.vault.save_vault()

    def get_vault_stats(self):
        """Get vault statistics."""
        return self.vault.get_stats()


def create_detection_model(vault: DetectionWeightVault,
                          object_category: str = "object",
                          num_classes: int = 1,
                          force_load: bool = False) -> SimpleDetectionNet:
    """
    Create an object detection model with database-driven initialization.

    Args:
        vault: The DetectionWeightVault instance
        object_category: Type of object (e.g., "cat", "dog", "car")
        num_classes: Number of classes (usually 1 for single-class detection)
        force_load: If True, always use vault weights when available

    Returns:
        A SimpleDetectionNet model with DB-aware layers
    """
    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()
    DBConv2dDetect.set_vault(vault)
    DBConv2dDetect.set_object_category(object_category)
    DBConv2dDetect.set_force_load(force_load)
    DBLinearDetect.set_vault(vault)
    DBLinearDetect.set_object_category(object_category)
    DBLinearDetect.set_force_load(force_load)

    model = SimpleDetectionNet(num_classes=num_classes, grid_size=7)
    return model


def convert_to_db_layers(model: nn.Module, vault: DetectionWeightVault = None,
                         object_category: str = None, force_load: bool = False):
    """
    Convert all Conv2d and Linear layers in a model to DB-aware versions.

    This is the KEY function that wraps existing models!

    Args:
        model: The PyTorch model to wrap
        vault: The DetectionWeightVault instance
        object_category: Category tag for weights (e.g., "cat", "dog")
        force_load: If True, always load from vault when weights exist

    Returns:
        The model with DB-aware layers
    """
    # Set vault settings BEFORE creating layers
    if vault is not None:
        DBConv2dDetect.set_vault(vault)
        DBConv2dDetect.set_object_category(object_category)
        DBConv2dDetect.set_force_load(force_load)
        DBLinearDetect.set_vault(vault)
        DBLinearDetect.set_object_category(object_category)
        DBLinearDetect.set_force_load(force_load)

    # Counter for naming
    DBConv2dDetect.reset_layer_counter()
    DBLinearDetect.reset_layer_counter()

    # Replace all Conv2d layers
    def replace_module(parent_module, child_name, child_module):
        """Replace a child module with DB-aware version"""

        if isinstance(child_module, nn.Conv2d):
            # Create DB-aware Conv2d with same parameters
            db_conv = DBConv2dDetect(
                in_channels=child_module.in_channels,
                out_channels=child_module.out_channels,
                kernel_size=child_module.kernel_size,
                stride=child_module.stride,
                padding=child_module.padding,
                dilation=child_module.dilation,
                groups=child_module.groups,
                bias=child_module.bias is not None
            )

            # Copy original weights (will be overwritten if vault has weights)
            with torch.no_grad():
                db_conv.weight.copy_(child_module.weight.data)
                if child_module.bias is not None:
                    db_conv.bias.copy_(child_module.bias.data)

            setattr(parent_module, child_name, db_conv)
            return True

        elif isinstance(child_module, nn.Linear):
            # Create DB-aware Linear with same parameters
            db_linear = DBLinearDetect(
                in_features=child_module.in_features,
                out_features=child_module.out_features,
                bias=child_module.bias is not None
            )

            # Copy original weights
            with torch.no_grad():
                db_linear.weight.copy_(child_module.weight.data)
                if child_module.bias is not None:
                    db_linear.bias.copy_(child_module.bias.data)

            setattr(parent_module, child_name, db_linear)
            return True

        return False

    # Recursively traverse and replace
    def recursive_replace(module, prefix=''):
        """Recursively replace all Conv2d and Linear layers"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Try to replace this child
            replaced = replace_module(module, name, child)

            if not replaced:
                # Recurse into children
                recursive_replace(child, full_name)

    recursive_replace(model)

    return model
