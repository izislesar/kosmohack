"""Fallback lane: smp.Unet timm-efficientnet-b4, in_channels=N=27, classes=4.

Started in parallel (same task, sequential on 1 GPU — fallback FIRST so a
deliverable always exists). Trigger: main lane shows no val IoU_burn by ~hour 2
-> finish this lane as the deliverable.
"""
import torch


def build_fallback_bs(cfg):
    import segmentation_models_pytorch as smp
    m = cfg["model"]
    n = int(cfg["data"]["in_channels"])
    nc = int(cfg["data"]["num_classes"])
    try:
        model = smp.Unet(
            encoder_name=m.get("encoder_name", "timm-efficientnet-b4"),
            encoder_weights=m.get("encoder_weights", "imagenet"),
            in_channels=n,
            classes=nc,
            activation=m.get("activation", None),
        )
        weights_note = f"encoder_weights={m.get('encoder_weights', 'imagenet')}"
    except Exception as e:
        model = smp.Unet(
            encoder_name=m.get("encoder_name", "timm-efficientnet-b4"),
            encoder_weights=None,
            in_channels=n,
            classes=nc,
            activation=m.get("activation", None),
        )
        weights_note = f"imagenet-download-failed({e}) -> random-init"
    model.train()
    with torch.no_grad():
        out = model(torch.zeros(1, n, 128, 128))
        assert tuple(out.shape) == (1, nc, 128, 128), f"unexpected shape {tuple(out.shape)}"
    return model, {"arch": "Unet", "weights": weights_note}
