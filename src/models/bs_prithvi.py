"""Main lane: Prithvi-EO-2.0-300M BurnScars via terratorch — patch-embed inflate pretrained->N, head 2->4.

Build: terratorch EncoderDecoderFactory, backbone=prithvi_eo_v2_300, pretrained,
backbone_bands = 6 HLS names + 21 custom BS_* names (N=27, see configs/bs.yaml).
terratorch maps pretrained patch-embed weights for known bands and random-inits
unknown ones; inflate_patch_embed_to_N() then overwrites the 21 custom slices with
the mean of the 6 pretrained slices (scaled by 6/21 to preserve activation scale) —
exact "inflate pretrained->N" semantics, deterministic under the run seed.
Head: fresh 4-class segmentation head (the HF BurnScars head is 2-class; we need 4).
Uses terratorch's own pretrained backbone weights (same Prithvi-EO-2.0-300M base as
ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars) — no separate HF weight surgery.

Gradient checkpointing: best-effort — tries backbone hooks, logs what engaged.
"""
import torch
import torch.nn as nn

N_PRETRAINED = 6  # first 6 channels carry HLS names / pretrained weights


def _shim_smp_use_batchnorm():
    """terratorch 1.0 UNetDecoder vs smp 0.5.0 incompatibilities, translated BEFORE
    terratorch is imported. Contained here; no shared files.
      init: use_batchnorm= -> use_norm= ; center= -> add_center_block=
      forward: terratorch calls decoder(*feats) (old varargs API) but smp 0.5
               takes a single features list -> accept both."""
    try:
        import segmentation_models_pytorch.decoders.unet.decoder as _m
        orig_init = _m.UnetDecoder.__init__

        def patched_init(self, *a, use_batchnorm=None, center=None, **k):
            if use_batchnorm is not None and "use_norm" not in k:
                k["use_norm"] = "batchnorm" if use_batchnorm else "identity"
            if center is not None and "add_center_block" not in k:
                k["add_center_block"] = bool(center)
            return orig_init(self, *a, **k)

        _m.UnetDecoder.__init__ = patched_init
        orig_fwd = _m.UnetDecoder.forward

        def patched_fwd(self, features=None, *args):
            if args:
                features = [features, *args] if features is not None else list(args)
            return orig_fwd(self, features)

        _m.UnetDecoder.forward = patched_fwd
        return "smp-compat-shim: init+forward applied"
    except Exception as e:
        return f"smp-compat-shim: SKIPPED ({e})"


_SMP_SHIM_NOTE = _shim_smp_use_batchnorm()


def find_patch_embed(model, n_channels=27):
    # Prithvi-EO-V2 terratorch: backbone.patch_embed.proj is Conv3d(out, C, T=1, 16, 16)
    for name, m in model.named_modules():
        if "patch_embed" in name and isinstance(m, (nn.Conv2d, nn.Conv3d)):
            return m
    for m in model.modules():
        if isinstance(m, nn.Conv3d) and m.in_channels == n_channels:
            return m
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == n_channels and tuple(m.kernel_size) == (16, 16):
            return m
    raise RuntimeError("patch-embed conv not found")


def inflate_patch_embed_to_N(model, n_channels=27, seed=19):
    """Overwrite random-init extra patch-embed slices with scaled pretrained mean. Returns report dict."""
    pe = find_patch_embed(model, n_channels)
    assert pe.in_channels == n_channels, f"patch-embed in={pe.in_channels} != {n_channels}"
    with torch.no_grad():
        w = pe.weight.data  # Conv2d (O,N,kh,kw) or Conv3d (O,N,T,kh,kw)
        pre = w[:, :N_PRETRAINED].clone()
        mean = pre.mean(dim=1, keepdim=True) * (N_PRETRAINED / (n_channels - N_PRETRAINED))
        w[:, N_PRETRAINED:] = mean.expand(w.shape[0], n_channels - N_PRETRAINED, *w.shape[2:])
        # note: terratorch random-init of extras is fully overwritten -> deterministic
    return {"patch_embed": f"{type(pe).__name__}{tuple(pe.weight.shape)}", "in_channels": pe.in_channels,
            "inflated_from": N_PRETRAINED, "scale": N_PRETRAINED / (n_channels - N_PRETRAINED)}


def try_grad_checkpoint(model):
    """Best-effort gradient checkpointing. Returns human-readable status string."""
    ok = []
    for name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing",
                 "set_grad_checkpointing"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                fn()
                ok.append(name)
            except Exception as e:
                ok.append(f"{name}:FAILED({e})")
    # walk backbones/encoders for the same hooks
    for sub in list(model.children()):
        for name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
            fn = getattr(sub, name, None)
            if callable(fn):
                try:
                    fn()
                    ok.append(f"child:{name}")
                except Exception:
                    pass
    # torch>=2.x checkpoint on ViT blocks is skipped (surgery risk); AMP+batch1-2+accum carry memory instead
    return "grad-checkpoint: " + (", ".join(ok) if ok else "none-engaged (AMP+batch1-2+accum used)")


def build_prithvi_bs(cfg):
    from terratorch.models import EncoderDecoderFactory
    m = cfg["model"]
    factory = EncoderDecoderFactory()
    model = factory.build_model(
        task="segmentation",
        backbone=m["backbone"],
        backbone_pretrained=bool(m.get("backbone_pretrained", True)),
        backbone_bands=list(m["backbone_bands"]),
        backbone_img_size=int(m.get("backbone_img_size", 512)),
        backbone_num_frames=int(m.get("backbone_num_frames", 1)),
        necks=list(m.get("necks", [])),
        decoder=m.get("decoder", "UNetDecoder"),
        decoder_channels=list(m.get("decoder_channels", [512, 256, 128, 64])),
        head_dropout=float(m.get("head_dropout", 0.1)),
        num_classes=int(cfg["data"]["num_classes"]),
    )
    rep = inflate_patch_embed_to_N(model, n_channels=int(cfg["data"]["in_channels"]),
                                   seed=int(cfg.get("seed", 19)))
    if bool(m.get("freeze_backbone", False)):
        for n, p in model.named_parameters():
            if not any(k in n for k in ("decoder", "head", "neck")):
                p.requires_grad = False
        rep["frozen"] = True
    else:
        for p in model.parameters():
            p.requires_grad = True
        rep["frozen"] = False
    rep["trainable_M"] = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    rep["grad_ckpt"] = try_grad_checkpoint(model) if bool(m.get("grad_checkpoint", True)) else "disabled-by-config"
    rep["smp_shim"] = _SMP_SHIM_NOTE
    # sanity: forward shape on CPU with zeros (cheap, catches channel mismatches early)
    model.eval()
    with torch.no_grad():
        n = int(cfg["data"]["in_channels"])
        out = model(torch.zeros(1, n, 64, 64))
        out = out.output if hasattr(out, "output") else out
        assert tuple(out.shape) == (1, 4, 64, 64), f"unexpected head shape {tuple(out.shape)}"
    model.train()
    return model, rep
