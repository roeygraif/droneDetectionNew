"""A small ConvLSTM2D cell + per-sequence wrapper.

Implementation follows the standard formulation (Shi et al., 2015):

    i = sigmoid(Wxi * x + Whi * h + bi)
    f = sigmoid(Wxf * x + Whf * h + bf)
    o = sigmoid(Wxo * x + Who * h + bo)
    g = tanh   (Wxg * x + Whg * h + bg)
    c' = f * c + i * g
    h' = o * tanh(c')

We fuse the four gate convolutions into one (4*hidden) conv for efficiency.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def init_state(self, batch_size: int, spatial: tuple[int, int], device, dtype):
        H, W = spatial
        zeros = torch.zeros(batch_size, self.hidden_channels, H, W, device=device, dtype=dtype)
        return zeros, zeros.clone()

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class ConvLSTM(nn.Module):
    """Single-layer ConvLSTM over a (B,T,C,H,W) sequence.

    Returns (outputs, (h, c)) where ``outputs`` is (B,T,hidden,H,W).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=kernel_size)
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"ConvLSTM expects (B,T,C,H,W); got {tuple(x.shape)}")
        B, T, C, H, W = x.shape
        state = self.cell.init_state(B, (H, W), device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h, state = self.cell(x[:, t], state)
            outs.append(h)
        outputs = torch.stack(outs, dim=1)
        return outputs, state


__all__ = ["ConvLSTM", "ConvLSTMCell"]
