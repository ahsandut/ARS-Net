import torch
from baselines.afc_factory import build_model
from baselines.adapter import segmentation_logits


def test_afc_factory_contract():
    model = build_model(in_channels=1, num_classes=2, base_channels=8, num_heads=4)
    model.eval()
    with torch.no_grad():
        output = segmentation_logits(model(torch.randn(1, 1, 64, 64)))
    assert output.shape == (1, 2, 64, 64)
