"""Building blocks for the spiking detector.

Two reusable modules:
  - ConvLIFBlock: Conv2d + BatchNorm2d + snn.Leaky. Stateful per-frame; its
    membrane potential is provided by the caller and updated in place. The
    caller decides whether to carry membrane across frames (for temporal
    integration) or to reset it (for per-frame feature extraction).
  - SpikingRecurrentBlock: a Conv-based recurrent LIF — membrane integrates
    both the feed-forward input from the encoder bottleneck and a recurrent
    Conv2d feedback from its own previous spike output. This is the SNN
    analog of the ConvLSTM bottleneck in the baseline.

We use init_hidden=False (the snntorch default-explicit pattern) so the caller
threads membrane state through the temporal loop. This is the safest pattern
when membranes have to coexist across multiple parallel layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


def _spike_grad(slope: float = 25.0):
    return surrogate.fast_sigmoid(slope=slope)


class ConvLIFBlock(nn.Module):
    """Conv2d → BatchNorm2d → LIF.

    Stateful: the caller manages the membrane potential ``mem`` and passes it
    in / out each forward call. ``init_mem(x)`` returns a zero-initialised
    membrane shaped to match the output spatial shape.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        kernel_size: int = 3,
        padding: int = 1,
        slope: float = 25.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=_spike_grad(slope),
            init_hidden=False,
            learn_beta=False,
            reset_mechanism="subtract",
        )
        self.out_channels = out_channels

    def init_mem(self, x: torch.Tensor) -> torch.Tensor:
        """Make a zero-initialised membrane potential for the OUTPUT shape.

        Given an input ``x`` of shape (B, C_in, H, W), returns (B, C_out, H, W)
        zeros.
        """
        B, _, H, W = x.shape
        return torch.zeros(
            B, self.out_channels, H, W, device=x.device, dtype=x.dtype
        )

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep.

        Args:
            x:   (B, C_in, H, W) input — could be raw pixels or upstream spikes.
            mem: (B, C_out, H, W) membrane potential from the previous step (or
                 ``init_mem(x)`` zeros at t=0).

        Returns:
            spk: (B, C_out, H, W) binary spikes.
            mem: (B, C_out, H, W) updated membrane.
        """
        cur = self.bn(self.conv(x))
        spk, mem = self.lif(cur, mem)
        return spk, mem


class SpikingRecurrentBlock(nn.Module):
    """Spiking recurrent block — the SNN analog of a ConvLSTM cell.

    Membrane integrates two inputs at every timestep:
      1. feed-forward Conv2d on the encoder spike output.
      2. recurrent Conv2d on its OWN previous spike output.

    Both go through a single Leaky neuron whose membrane potential persists
    across all T frames. This gives the bottleneck explicit temporal memory
    without needing the four LSTM gates.
    """

    def __init__(
        self,
        channels: int,
        beta: float = 0.95,         # longer time constant than per-frame blocks
        threshold: float = 1.0,
        kernel_size: int = 3,
        padding: int = 1,
        slope: float = 25.0,
    ):
        super().__init__()
        self.feedforward = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.recurrent = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        # No BN on the recurrent path — its statistics shift each timestep.
        self.bn = nn.BatchNorm2d(channels)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=_spike_grad(slope),
            init_hidden=False,
            learn_beta=True,        # let the time constant be learnable here
            reset_mechanism="subtract",
        )
        self.channels = channels

    def init_state(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero membrane potential AND zero previous-spike output."""
        B, _, H, W = x.shape
        zeros = torch.zeros(B, self.channels, H, W, device=x.device, dtype=x.dtype)
        return zeros.clone(), zeros.clone()

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        prev_spk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One timestep.

        Args:
            x:        (B, C, H, W) encoder bottleneck spikes for this frame.
            mem:      (B, C, H, W) membrane potential from previous frame.
            prev_spk: (B, C, H, W) recurrent block's own spike from previous frame.

        Returns:
            spk: (B, C, H, W) spike output for this frame.
            mem: (B, C, H, W) updated membrane.
        """
        cur = self.bn(self.feedforward(x) + self.recurrent(prev_spk))
        spk, mem = self.lif(cur, mem)
        return spk, mem


__all__ = ["ConvLIFBlock", "SpikingRecurrentBlock"]
