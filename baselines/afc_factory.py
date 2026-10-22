"""Factory entry point for the locally implemented AFC-inspired baseline."""
from .afc_unet import AFC_UNet


def build_model(num_classes: int = 2, in_channels: int = 1, base_channels: int = 32, num_heads: int = 4, **kwargs):
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported AFC-inspired model arguments: {unknown}")
    return AFC_UNet(in_channels=in_channels, num_classes=num_classes, base_c=base_channels, num_heads=num_heads)
