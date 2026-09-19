"""AF Unet with timm-efficientnet-b4 encoder (segmentation-models-pytorch).

First conv inflated 3 -> in_channels by mean-init: the ImageNet-pretrained
RGB stem weights are averaged over the channel dim and replicated, so the
expected activation scale is preserved at init. Full finetune: all params
trainable.
"""

from __future__ import annotations

import torch.nn as nn


def build_af_unet(
    in_channels: int = 12,
    encoder: str = "timm-efficientnet-b4",
    encoder_weights: str = "imagenet",
    classes: int = 1,
) -> nn.Module:
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,  # load pristine RGB stem first
        classes=classes,
    )
    if in_channels != 3:
        stem = model.encoder.conv_stem  # Conv2d(3, C, k, s, p, bias=...)
        w3 = stem.weight.data  # (C, 3, kH, kW)
        w_new = w3.mean(dim=1, keepdim=True).expand(-1, in_channels, -1, -1).contiguous()
        new_stem = nn.Conv2d(
            in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            bias=stem.bias is not None,
        )
        new_stem.weight.data.copy_(w_new)
        if stem.bias is not None:
            new_stem.bias.data.copy_(stem.bias.data)
        model.encoder.conv_stem = new_stem
    for p in model.parameters():
        p.requires_grad = True  # full finetune
    return model
