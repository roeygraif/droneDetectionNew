"""LIAF (Leaky Integrate and Analog Fire) building blocks.

Based on the LIAF-Net paper (Wu et al., "LIAF-Net: Leaky Integrate and Analog
Fire Network for Lightweight and Efficient Spatiotemporal Information
Processing", arXiv:2011.06176, 2020).

The key idea:
  - **Temporal**: same as LIF — membrane potential decays over time and
    integrates incoming current. u_t = beta * u_{t-1} + (BN ∘ Conv)(x_t).
  - **Spatial**: output is a CONTINUOUS analog activation of the membrane,
    not a binary spike. We use sigmoid((u - threshold) * slope), bounded in
    (0, 1), which keeps amplitude information flowing between layers.

This addresses the matched-filter information-loss diagnosed in V1/V2:
binary spike outputs threshold-out the marginal-amplitude signal that's the
binding constraint at −6 dB. LIAF preserves that information through the
hidden layers, at the cost of giving up the spiking-sparsity / energy-
efficiency claim.

The temporal integration (beta-weighted membrane) is preserved, so LIAF
still has the leaky-integrator-as-matched-filter property that's the
theoretical motivation for the SNN direction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvLIAFBlock(nn.Module):
    """Conv2d → BatchNorm2d → LIAF (leaky integrate + analog fire).

    Output is a continuous analog activation in (0, 1) — NOT a binary spike.
    Per LIAF-Net: y_t = sigmoid((u_t − threshold) × slope), where u_t is the
    leaky-integrated membrane potential.

    Stateful: caller manages the membrane and passes it in/out each frame.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 4.0,        # sigmoid sharpness (lower = smoother)
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.beta = beta
        self.threshold = threshold
        self.slope = slope
        self.out_channels = out_channels

    def init_mem(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        return torch.zeros(
            B, self.out_channels, H, W, device=x.device, dtype=x.dtype
        )

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One timestep.

        Args:
            x:   (B, C_in, H, W) input — either raw pixels or upstream analog
                 outputs.
            mem: (B, C_out, H, W) membrane potential from previous step.

        Returns:
            y:   (B, C_out, H, W) continuous analog activation.
            mem: (B, C_out, H, W) updated membrane (leaky-integrated).
        """
        cur = self.bn(self.conv(x))
        mem_new = self.beta * mem + cur
        y = torch.sigmoid((mem_new - self.threshold) * self.slope)
        return y, mem_new


class AnalogRecurrentBlock(nn.Module):
    """LIAF analog of the spiking recurrent bottleneck.

    Same structure as SpikingRecurrentBlock but with continuous analog output
    that's also fed back through the recurrent Conv2d. Membrane potential
    persists across all T frames so the bottleneck has explicit temporal
    memory.
    """

    def __init__(
        self,
        channels: int,
        beta: float = 0.95,        # longer time constant than per-frame blocks
        threshold: float = 1.0,
        slope: float = 4.0,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.feedforward = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.recurrent = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(channels)
        self.beta = nn.Parameter(torch.tensor(beta))  # learnable time constant
        self.threshold = threshold
        self.slope = slope
        self.channels = channels

    def init_state(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        zeros = torch.zeros(B, self.channels, H, W, device=x.device, dtype=x.dtype)
        return zeros.clone(), zeros.clone()

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        prev_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One timestep.

        Args:
            x:      (B, C, H, W) encoder bottleneck analog output for this frame.
            mem:    (B, C, H, W) membrane potential from previous frame.
            prev_y: (B, C, H, W) recurrent block's own analog output from previous frame.

        Returns:
            y:   (B, C, H, W) analog output for this frame.
            mem: (B, C, H, W) updated membrane.
        """
        cur = self.bn(self.feedforward(x) + self.recurrent(prev_y))
        beta = torch.sigmoid(self.beta)  # keep in (0, 1) — learnable but bounded
        mem_new = beta * mem + cur
        y = torch.sigmoid((mem_new - self.threshold) * self.slope)
        return y, mem_new


__all__ = ["ConvLIAFBlock", "AnalogRecurrentBlock"]
