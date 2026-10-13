from __future__ import annotations

import torch

from slicerel.models import SliceRelUNet


def main() -> None:
    model = SliceRelUNet(
        in_channels=1,
        num_classes=4,
        slice_count=3,
        backbone_profile="legacy_bn_bilinear_32",
        variant="slicerel_unet",
        graph_hidden_channels=512,
        attention_dim=None,
        attention_dim_rule="quarter_channels",
        adjacency_mlp_hidden=9,
        spatial_reduction=4,
    )
    inputs = torch.randn(1, 3, 1, 32, 32)
    outputs = model.forward_grouped(inputs)
    assert outputs.shape == (1, 3, 4, 32, 32)
    outputs.mean().backward()
    print("SliceRel-U-Net smoke test passed:", tuple(outputs.shape))


if __name__ == "__main__":
    main()

