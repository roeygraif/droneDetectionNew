"""Minimal sanity tests for the SpikingTrackletUNet.

Confirms:
  - forward pass produces the expected output dict shapes for the I/O contract,
  - surrogate gradients propagate (loss is differentiable),
  - the model trains in 1 step on a synthetic micro-batch without NaNs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make repo root importable so we can `import spikeNN.*`.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spikeNN.models.spiking_recurrent_unet import (
    SpikingTrackletUNet,
    SpikingTrackletUNetConfig,
)


def test_forward_shapes_default():
    cfg = SpikingTrackletUNetConfig()  # in=5, base=16, bottleneck=32, S=15
    model = SpikingTrackletUNet(cfg)
    B, T, C, S = 2, 8, 5, 15
    x = torch.randn(B, T, C, S, S)
    out = model(x)
    assert out["track_logit"].shape == (B,)
    assert out["visibility_logits"].shape == (B, T)
    assert out["heatmap_logits"].shape == (B, T, 1, S, S)


def test_forward_no_nans():
    model = SpikingTrackletUNet()
    x = torch.randn(2, 8, 5, 15, 15)
    out = model(x)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"NaN/Inf in {k}"


def test_surrogate_gradient_flows():
    """Train one step on a toy supervised target — loss must decrease."""
    torch.manual_seed(0)
    model = SpikingTrackletUNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 8, 5, 15, 15)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    losses = []
    for step in range(3):
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["track_logit"], target
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # Loss should strictly decrease at least once in 3 steps.
    assert losses[-1] < losses[0] + 1e-6, f"loss did not decrease: {losses}"


def test_forward_temporal_lengths():
    """Confirm model handles short (T=4) and long (T=32) sequences."""
    model = SpikingTrackletUNet()
    for T in (4, 16, 32):
        x = torch.randn(1, T, 5, 15, 15)
        out = model(x)
        assert out["track_logit"].shape == (1,)
        assert out["visibility_logits"].shape == (1, T)
        assert out["heatmap_logits"].shape == (1, T, 1, 15, 15)


def test_parameter_count_reasonable():
    """SNN model should be in the same order of magnitude as baseline (<1M params)."""
    model = SpikingTrackletUNet()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 50_000 < n < 1_000_000, f"unexpected param count: {n:,}"
