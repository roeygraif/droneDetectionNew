"""Hook the frozen detector and read per-stage, per-tube features for probing.

The Q1 instrument: push oracle tubes through the frozen `TrackletRecurrentUNet`,
capture the activation at each stage, and pool it to one feature vector per tube
using the SAME recipe as the input probe in Exp 01 (per-channel mean, max, and
center-region temporal sum). A Fisher-LDA probe on those features then estimates
the best *linear* detectability surviving to that stage, which we compare to the
whitening optimum to localize where detectability collapses.

Stages, in forward order (folded to (B,T,Cf,h,w) before pooling):
    input → enc1 → enc2 → enc3 → convlstm → dec2 → dec1 → logit
`enc3` is the bottleneck just before temporal integration; `convlstm` is just
after it — the pair brackets the temporal-integration step, a prime suspect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.recurrent_unet import TrackletRecurrentUNet, TrackletRecurrentUNetConfig  # noqa: E402


def set_pooling(model, kind: str = "max"):
    """Swap the encoder downsampling op (the Exp 06b cliff is at pool1).

    The pools carry no parameters, so this is safe before or after loading a
    state_dict. "avg" replaces max-pooling (which inflates the noise floor at low
    SNR) with averaging (matched-filter-like integration) — the C3 repair.
    """
    if kind == "max":
        return model
    if kind == "avg":
        model.pool1 = nn.AvgPool2d(2)
        model.pool2 = nn.AvgPool2d(2)
    else:
        raise ValueError(f"unknown pool kind {kind!r}")
    return model

STAGES = ["input", "enc1", "enc2", "enc3", "convlstm", "dec2", "dec1", "logit"]
_HOOKED = {"enc1": "enc1", "enc2": "enc2", "enc3": "enc3",
           "convlstm": "temporal", "dec2": "dec2", "dec1": "dec1"}


def load_detector(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    m = TrackletRecurrentUNet(TrackletRecurrentUNetConfig(
        in_channels=ck["in_channels"], base_channels=ck["base_channels"],
        bottleneck_channels=ck["bottleneck_channels"], use_convlstm=True,
        crop_size=ck["crop_size"]))
    m.load_state_dict(ck["model_state"])
    set_pooling(m, ck.get("pool", "max"))  # match the repaired architecture, if any
    m.to(device).eval()
    return m, ck


def _fold(act: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """Bring a stage activation to (B,T,Cf,h,w) whether it came out per-frame
    (B*T,Cf,h,w) or already temporal (B,T,Cf,h,w)."""
    if act.ndim == 5:
        return act
    if act.ndim == 4:
        _, Cf, h, w = act.shape
        return act.reshape(B, T, Cf, h, w)
    raise ValueError(f"unexpected stage activation ndim={act.ndim}")


def _pool_per_tube(act: torch.Tensor) -> torch.Tensor:
    """(B,T,Cf,h,w) -> (B, 3*Cf): per-channel [mean, max, center-region time-sum]."""
    B, T, Cf, h, w = act.shape
    c = h // 2
    f_mean = act.mean(dim=(1, 3, 4))                 # (B,Cf) over time+space
    f_max = act.amax(dim=(1, 3, 4))                  # (B,Cf)
    if h >= 3 and w >= 3:
        center = act[:, :, :, c - 1:c + 2, c - 1:c + 2].mean(dim=(3, 4)).sum(dim=1)
    else:
        center = act[:, :, :, c, c].sum(dim=1)       # (B,Cf)
    return torch.cat([f_mean, f_max, center], dim=1)  # (B, 3Cf)


@torch.no_grad()
def collect_stage_features(model, X_np, device, batch: int = 32):
    """Return (features, logits): features[stage] is (N, F) per-tube, logits is (N,)."""
    N, T = X_np.shape[0], X_np.shape[1]
    buf: dict[str, torch.Tensor] = {}
    handles = []
    for name, attr in _HOOKED.items():
        def mk(nm):
            def hook(_m, _i, out):
                buf[nm] = out[0] if isinstance(out, tuple) else out
            return hook
        handles.append(getattr(model, attr).register_forward_hook(mk(name)))

    acc = {s: [] for s in STAGES}
    logits = []
    try:
        for i in range(0, N, batch):
            xb = torch.from_numpy(X_np[i:i + batch]).float().to(device)
            B = xb.shape[0]
            out = model(xb)
            acc["input"].append(_pool_per_tube(xb).cpu())
            for name in _HOOKED:
                acc[name].append(_pool_per_tube(_fold(buf[name], B, T)).cpu())
            lg = out["track_logit"].reshape(B, 1).cpu()
            acc["logit"].append(lg)
            logits.append(lg.reshape(B))
    finally:
        for h in handles:
            h.remove()

    feats = {s: np.concatenate([t.numpy() for t in acc[s]]).astype(np.float64) for s in STAGES}
    return feats, np.concatenate([t.numpy() for t in logits])


__all__ = ["STAGES", "load_detector", "collect_stage_features"]
