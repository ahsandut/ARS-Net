"""Fast shape smoke test for the reconstructed ARS-Net architecture."""
import torch
from models import ARSNet


def test_ars_net_shapes():
    model = ARSNet(input_size=320)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 320, 320))

    assert output["segmentation_logits"].shape == (1, 2, 320, 320)
    assert output["acl_region_logits"].shape == (1, 1, 40, 40)
    assert output["acl_centerline_logits"].shape == (1, 1, 40, 40)
    assert output["acl_distance"].shape == (1, 1, 40, 40)
    assert output["pcl_region_logits"].shape == (1, 1, 40, 40)
    assert output["pcl_centerline_logits"].shape == (1, 1, 40, 40)
    assert output["pcl_distance"].shape == (1, 1, 40, 40)


def test_architecture_ablation_variants_keep_output_shapes():
    for variant in ("no_arrm", "no_msfm", "no_arrm_no_msfm"):
        model = ARSNet(input_size=320, architecture_variant=variant)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 1, 320, 320))
        assert output["segmentation_logits"].shape == (1, 2, 320, 320)
        assert output["acl_centerline_logits"].shape == (1, 1, 40, 40)
        assert output["pcl_distance"].shape == (1, 1, 40, 40)
