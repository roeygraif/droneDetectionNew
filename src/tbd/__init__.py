"""Track-before-detect (TBD) pipeline modules.

The pipeline turns a raw frame stack into a small set of scored tracklet
crop tubes that a downstream model can classify as real UAV / false alarm:

    frames (T,H,W)
      -> evidence maps (multi-channel per-frame)
      -> candidates per frame (top-K local maxima)
      -> tracklets (motion-consistent linked candidates)
      -> crop tubes (T,C,S,S) centered on each tracklet
"""

from tbd.accumulator import (
    AccumulatorConfig,
    accumulate_tracks,
    build_velocity_grid,
    extract_seed_tracklets,
)
from tbd.evidence import EvidenceConfig, compute_evidence_maps
from tbd.candidates import Candidate, CandidateConfig, extract_candidates
from tbd.tracklets import Tracklet, TrackletConfig, build_tracklets
from tbd.crop_tubes import CropTubeConfig, extract_crop_tube

__all__ = [
    "AccumulatorConfig",
    "Candidate",
    "CandidateConfig",
    "CropTubeConfig",
    "EvidenceConfig",
    "Tracklet",
    "TrackletConfig",
    "accumulate_tracks",
    "build_tracklets",
    "build_velocity_grid",
    "compute_evidence_maps",
    "extract_candidates",
    "extract_crop_tube",
    "extract_seed_tracklets",
]
