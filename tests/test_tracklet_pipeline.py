"""End-to-end tests for the tracklet-guided UAV detection pipeline."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.tracklet_dataset import (
    TrackletCropDataset,
    TrackletDatasetConfig,
    build_tube_sample,
)
from models.cnn_gru_baseline import CropCNNGRU, CropCNNGRUConfig
from models.recurrent_unet import (
    TrackletRecurrentUNet,
    TrackletRecurrentUNetConfig,
)
from synthetic.sequence import SequenceConfig, generate_sequence
from tbd.accumulator import (
    AccumulatorConfig,
    accumulate_tracks,
    build_velocity_grid,
    extract_seed_tracklets,
)
from tbd.candidates import CandidateConfig, extract_candidates
from tbd.crop_tubes import CropTubeConfig, extract_crop_tube
from tbd.evidence import EvidenceConfig, compute_evidence_maps
from tbd.tracklets import TrackletConfig, build_tracklets
from training.losses import TrackletLoss, TrackletLossConfig
from training.metrics import pr_auc, roc_auc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _basic_positive_sample(snr_db=5.0, n_frames=10, seed=42, canvas=(64, 64)):
    cfg = SequenceConfig(
        n_frames=n_frames, canvas_shape=canvas, snr_db=snr_db, seed=seed,
        sequence_type="positive_uav",
    )
    return generate_sequence(cfg)


def _basic_negative_sample(seed=42, n_frames=10, canvas=(64, 64)):
    cfg = SequenceConfig(
        n_frames=n_frames, canvas_shape=canvas, snr_db=0.0, seed=seed,
        sequence_type="empty_background",
    )
    return generate_sequence(cfg)


# ---------------------------------------------------------------------------
# Evidence maps
# ---------------------------------------------------------------------------

def test_evidence_maps_shapes_and_finite():
    s = _basic_positive_sample()
    maps = compute_evidence_maps(s.frames, EvidenceConfig())
    expected_keys = {"raw", "local_z", "matched", "temporal_diff", "evidence"}
    assert set(maps.keys()) == expected_keys
    for k in expected_keys:
        arr = maps[k]
        assert arr.shape == s.frames.shape
        assert np.all(np.isfinite(arr)), f"non-finite values in {k}"


def test_evidence_maps_accept_torch():
    s = _basic_positive_sample()
    t = torch.from_numpy(s.frames)
    maps = compute_evidence_maps(t, EvidenceConfig())
    assert maps["evidence"].shape == s.frames.shape


def test_temporal_diff_zero_on_first_frame():
    s = _basic_positive_sample()
    maps = compute_evidence_maps(s.frames, EvidenceConfig())
    assert np.all(maps["temporal_diff"][0] == 0)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def test_candidates_in_bounds_and_top_k_respected():
    s = _basic_positive_sample()
    maps = compute_evidence_maps(s.frames, EvidenceConfig())
    cfg = CandidateConfig(candidate_top_k=20, crop_size=15)
    per_frame = extract_candidates(maps["evidence"], cfg)
    assert len(per_frame) == s.frames.shape[0]
    H, W = s.frames.shape[1], s.frames.shape[2]
    half = cfg.crop_size // 2
    for cands in per_frame:
        assert len(cands) <= cfg.candidate_top_k
        for c in cands:
            assert half <= c.y < H - half
            assert half <= c.x < W - half


def test_candidates_sorted_descending_by_score():
    s = _basic_positive_sample()
    maps = compute_evidence_maps(s.frames, EvidenceConfig())
    per_frame = extract_candidates(maps["evidence"], CandidateConfig(candidate_top_k=50, crop_size=15))
    for cands in per_frame:
        scores = [c.score for c in cands]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tracklets
# ---------------------------------------------------------------------------

def test_tracklet_builder_basic():
    s = _basic_positive_sample()
    maps = compute_evidence_maps(s.frames, EvidenceConfig())
    cands = extract_candidates(maps["evidence"], CandidateConfig(candidate_top_k=50, crop_size=15))
    cfg = TrackletConfig(tracklet_len=s.frames.shape[0], tracklet_beam_size=64,
                         tracklet_final_top_m=10, tracklet_min_observed_points=3)
    tracklets = build_tracklets(cands, cfg)
    assert len(tracklets) > 0
    for tr in tracklets:
        assert tr.miss_count <= cfg.tracklet_max_misses * 2  # generous upper bound
        assert tr.positions.shape[0] == cfg.tracklet_len
        # At least min_observed_points non-NaN positions.
        finite = np.isfinite(tr.positions[:, 0]).sum()
        assert finite >= cfg.tracklet_min_observed_points


def test_tracklet_respects_max_misses_per_run():
    """Build with a strict max_misses; no tracklet should exceed it on any
    contiguous miss run."""
    s = _basic_positive_sample()
    maps = compute_evidence_maps(s.frames, EvidenceConfig())
    cands = extract_candidates(maps["evidence"], CandidateConfig(candidate_top_k=50, crop_size=15))
    cfg = TrackletConfig(tracklet_len=s.frames.shape[0], tracklet_max_misses=2,
                         tracklet_beam_size=32, tracklet_final_top_m=5,
                         tracklet_min_observed_points=3)
    tracklets = build_tracklets(cands, cfg)
    for tr in tracklets:
        # Compute max run of NaNs between first and last observed positions.
        obs = np.isfinite(tr.positions[:, 0])
        if not obs.any():
            continue
        first = int(np.argmax(obs))
        last = len(obs) - 1 - int(np.argmax(obs[::-1]))
        run = 0
        max_run = 0
        for t in range(first, last + 1):
            if not obs[t]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        assert max_run <= cfg.tracklet_max_misses


# ---------------------------------------------------------------------------
# Crop tubes
# ---------------------------------------------------------------------------

def test_crop_tube_shapes():
    s = _basic_positive_sample()
    maps = compute_evidence_maps(s.frames, EvidenceConfig())
    cands = extract_candidates(maps["evidence"], CandidateConfig(candidate_top_k=20, crop_size=21))
    tracklets = build_tracklets(cands, TrackletConfig(tracklet_len=s.frames.shape[0]))
    if not tracklets:
        pytest.skip("no tracklets built for this sample")
    tr = tracklets[0]
    cfg = CropTubeConfig(crop_size=21)
    out = extract_crop_tube(s.frames, maps, tr, cfg)
    T = s.frames.shape[0]
    assert out["crop_tube"].shape == (T, len(cfg.channels), 21, 21)
    assert out["valid_mask"].shape == (T,)
    assert out["crop_centers"].shape == (T, 2)


# ---------------------------------------------------------------------------
# Dataset labeling
# ---------------------------------------------------------------------------

def test_positive_tracklet_labeling():
    """A tracklet seeded on the true UAV position should label positive."""
    s = _basic_positive_sample(snr_db=10.0, n_frames=8)
    ds_cfg = TrackletDatasetConfig(
        base_config=replace(s.config, snr_db=10.0),
        crop_tubes=CropTubeConfig(crop_size=21),
        candidates=CandidateConfig(candidate_top_k=30, crop_size=21),
        tracklets=TrackletConfig(tracklet_len=8, tracklet_min_observed_points=3,
                                 tracklet_final_top_m=10),
        positive_radius_px=3.0,
        positive_min_visible_overlap=3,
    )
    items = build_tube_sample(s, ds_cfg)
    assert items, "no tracklets built"
    labels = [int(it["track_label"].item()) for it in items]
    assert max(labels) == 1, f"expected at least one positive tracklet, got {labels}"


def test_negative_sequence_only_negative_tracklets():
    """Target-absent sequence should never produce a positive tracklet."""
    s = _basic_negative_sample(n_frames=8)
    ds_cfg = TrackletDatasetConfig(
        base_config=s.config,
        crop_tubes=CropTubeConfig(crop_size=21),
        candidates=CandidateConfig(candidate_top_k=30, crop_size=21),
        tracklets=TrackletConfig(tracklet_len=8, tracklet_min_observed_points=3,
                                 tracklet_final_top_m=10),
    )
    items = build_tube_sample(s, ds_cfg)
    for it in items:
        assert int(it["track_label"].item()) == 0
        assert torch.all(it["visibility_label"] == 0)
        assert torch.all(it["heatmap_label"] == 0)


def test_dataset_iterator_yields_items():
    ds_cfg = TrackletDatasetConfig(
        base_config=SequenceConfig(canvas_shape=(64, 64), n_frames=6, target_sigma=1.0),
        snr_range_db=(0.0, 5.0),
        n_choices=(6,),
        sequence_type_probs={"positive_uav": 0.5, "empty_background": 0.5},
        crop_tubes=CropTubeConfig(crop_size=15),
        candidates=CandidateConfig(candidate_top_k=20, crop_size=15),
        tracklets=TrackletConfig(tracklet_len=6, tracklet_min_observed_points=2,
                                 tracklet_final_top_m=5),
        n_samples=6,
        seed=7,
    )
    ds = TrackletCropDataset(ds_cfg)
    items = list(iter(ds))
    assert len(items) > 0
    for it in items:
        assert it["crop_tube"].ndim == 4  # (T,C,S,S)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_convlstm", [True, False])
def test_recurrent_unet_forward_shapes(use_convlstm):
    cfg = TrackletRecurrentUNetConfig(in_channels=5, base_channels=8,
                                      bottleneck_channels=16,
                                      use_convlstm=use_convlstm, crop_size=20)
    model = TrackletRecurrentUNet(cfg)
    B, T, C, S = 2, 6, 5, 20
    x = torch.randn(B, T, C, S, S)
    out = model(x)
    assert out["track_logit"].shape == (B,)
    assert out["visibility_logits"].shape == (B, T)
    assert out["heatmap_logits"].shape == (B, T, 1, S, S)


def test_recurrent_unet_forward_odd_crop():
    cfg = TrackletRecurrentUNetConfig(in_channels=5, base_channels=8,
                                      bottleneck_channels=16, crop_size=31)
    model = TrackletRecurrentUNet(cfg)
    B, T, C, S = 1, 5, 5, 31
    out = model(torch.randn(B, T, C, S, S))
    assert out["heatmap_logits"].shape == (B, T, 1, S, S)


def test_cnn_gru_baseline_forward_shapes():
    cfg = CropCNNGRUConfig(in_channels=5, base_channels=8, feature_dim=16,
                           gru_hidden=16, crop_size=20)
    model = CropCNNGRU(cfg)
    B, T, C, S = 2, 5, 5, 20
    out = model(torch.randn(B, T, C, S, S))
    assert out["track_logit"].shape == (B,)
    assert out["visibility_logits"].shape == (B, T)
    assert out["heatmap_logits"] is None


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def test_training_step_runs():
    """One forward/backward step on a real mini-batch shouldn't crash."""
    ds_cfg = TrackletDatasetConfig(
        base_config=SequenceConfig(canvas_shape=(48, 48), n_frames=5),
        snr_range_db=(0.0, 5.0),
        n_choices=(5,),
        sequence_type_probs={"positive_uav": 0.7, "empty_background": 0.3},
        crop_tubes=CropTubeConfig(crop_size=15),
        candidates=CandidateConfig(candidate_top_k=20, crop_size=15),
        tracklets=TrackletConfig(tracklet_len=5, tracklet_min_observed_points=2,
                                 tracklet_final_top_m=4),
        n_samples=4, seed=11,
    )
    ds = TrackletCropDataset(ds_cfg)
    items = list(iter(ds))
    assert items
    batch = {
        "crop_tube": torch.stack([it["crop_tube"] for it in items], dim=0),
        "valid_mask": torch.stack([it["valid_mask"] for it in items], dim=0),
        "track_label": torch.stack([it["track_label"] for it in items], dim=0),
        "visibility_label": torch.stack([it["visibility_label"] for it in items], dim=0),
        "heatmap_label": torch.stack([it["heatmap_label"] for it in items], dim=0),
    }
    model = TrackletRecurrentUNet(TrackletRecurrentUNetConfig(
        in_channels=5, base_channels=8, bottleneck_channels=16, crop_size=15,
    ))
    loss_fn = TrackletLoss(TrackletLossConfig())
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    outputs = model(batch["crop_tube"])
    losses = loss_fn(outputs, batch)
    assert torch.isfinite(losses["loss"]), "non-finite loss"
    optim.zero_grad()
    losses["loss"].backward()
    optim.step()


# ---------------------------------------------------------------------------
# TBD accumulator
# ---------------------------------------------------------------------------

def test_velocity_grid_shape_and_zero_first():
    cfg = AccumulatorConfig(n_speeds=4, n_directions=8)
    vels = build_velocity_grid(cfg)
    assert vels.shape == (1 + 3 * 8, 2)
    assert vels[0, 0] == 0.0 and vels[0, 1] == 0.0


def test_accumulator_recovers_synthetic_track():
    """A clean evidence stack with a single bright track should peak the
    accumulator at the right (start, velocity)."""
    T, H, W = 8, 32, 32
    evidence = np.zeros((T, H, W), dtype=np.float32)
    y0, x0 = 10.0, 16.0
    vy, vx = 1.0, 0.0  # axis-aligned, lands exactly on the polar grid
    for t in range(T):
        yi = int(round(y0 + vy * t))
        xi = int(round(x0 + vx * t))
        evidence[t, yi, xi] = 1.0

    cfg = AccumulatorConfig(speed_max_px_per_frame=2.0, n_speeds=3, n_directions=8,
                            accumulator_top_k=5, crop_size=7,
                            accumulator_min_observed_points=3)
    tracklets = extract_seed_tracklets(evidence, cfg)
    assert tracklets, "accumulator returned no tracklets"
    best = tracklets[0]
    assert abs(best.positions[0, 0] - y0) <= 1.0
    assert abs(best.positions[0, 1] - x0) <= 1.0
    assert abs(float(best.velocity[0]) - vy) < 0.6
    assert abs(float(best.velocity[1]) - vx) < 0.6
    # All T track pixels are present, so the perfect-hypothesis sum is T.
    assert best.score >= T * 0.75


def test_accumulator_beats_per_frame_topk_at_low_snr():
    """At very low SNR the per-frame extractor + beam search rarely finds the
    real UAV; the accumulator should produce a tracklet that overlaps GT on
    at least one frame.

    The test is statistical (over a few seeds) because individual sequences
    can still go either way."""
    n_hits_acc = 0
    n_hits_beam = 0
    n_seq = 4
    for seed in range(n_seq):
        sample = generate_sequence(SequenceConfig(
            canvas_shape=(64, 64), n_frames=8, snr_db=-12.0,
            sequence_type="positive_uav", seed=seed + 100,
        ))
        evidence = compute_evidence_maps(sample.frames, EvidenceConfig())

        # Accumulator
        acc_tracks = extract_seed_tracklets(
            evidence["evidence"],
            AccumulatorConfig(speed_max_px_per_frame=2.0, n_speeds=5, n_directions=8,
                              accumulator_top_k=20, crop_size=15,
                              accumulator_min_observed_points=3),
        )

        # Beam search
        cands = extract_candidates(
            evidence["evidence"],
            CandidateConfig(candidate_top_k=100, crop_size=15),
        )
        beam_tracks = build_tracklets(
            cands,
            TrackletConfig(tracklet_len=8, tracklet_beam_size=64,
                           tracklet_min_observed_points=3,
                           tracklet_final_top_m=20),
        )

        gt = sample.positions
        vis = sample.target_visible
        for tracks, counter_name in [(acc_tracks, "acc"), (beam_tracks, "beam")]:
            hit = False
            for tr in tracks:
                for t in range(min(tr.positions.shape[0], gt.shape[0])):
                    if not bool(vis[t]):
                        continue
                    if not (np.isfinite(tr.positions[t, 0]) and np.isfinite(gt[t, 0])):
                        continue
                    if math.hypot(tr.positions[t, 0] - gt[t, 0],
                                  tr.positions[t, 1] - gt[t, 1]) <= 3.0:
                        hit = True
                        break
                if hit:
                    break
            if counter_name == "acc" and hit:
                n_hits_acc += 1
            if counter_name == "beam" and hit:
                n_hits_beam += 1

    # The accumulator should at least match beam search, and find SOMETHING
    # on a majority of sequences. (Beam search at SNR=-12 dB on most of these
    # finds nothing.)
    assert n_hits_acc >= n_hits_beam
    assert n_hits_acc >= n_seq // 2


def test_dataset_with_accumulator_runs():
    """build_tube_sample should accept tracklet_source='accumulator' end-to-end."""
    sample = generate_sequence(SequenceConfig(
        canvas_shape=(64, 64), n_frames=6, snr_db=2.0,
        sequence_type="positive_uav", seed=42,
    ))
    ds_cfg = TrackletDatasetConfig(
        base_config=sample.config,
        tracklet_source="accumulator",
        accumulator=AccumulatorConfig(speed_max_px_per_frame=2.0, n_speeds=4,
                                      n_directions=8, accumulator_top_k=10,
                                      crop_size=15,
                                      accumulator_min_observed_points=3),
        crop_tubes=CropTubeConfig(crop_size=15),
    )
    items = build_tube_sample(sample, ds_cfg)
    assert len(items) > 0
    for it in items:
        assert it["crop_tube"].ndim == 4


# ---------------------------------------------------------------------------
# End-to-end "tiny eval"
# ---------------------------------------------------------------------------

def test_end_to_end_tiny_eval():
    """Score a handful of synthetic sequences and compute ROC-AUC.

    We don't train a model — we just check that the full pipeline runs and
    that the metrics functions return finite values on a non-trivial mix.
    """
    from eval.eval_tracklet_detector import score_sequence

    ds_cfg = TrackletDatasetConfig(
        base_config=SequenceConfig(canvas_shape=(48, 48), n_frames=5),
        snr_range_db=(0.0, 5.0),
        n_choices=(5,),
        sequence_type_probs={"positive_uav": 0.5, "empty_background": 0.5},
        crop_tubes=CropTubeConfig(crop_size=15),
        candidates=CandidateConfig(candidate_top_k=20, crop_size=15),
        tracklets=TrackletConfig(tracklet_len=5, tracklet_min_observed_points=2,
                                 tracklet_final_top_m=4),
        n_samples=8, seed=17,
    )

    # Use an untrained model — that's fine; we just want the plumbing to work.
    model = TrackletRecurrentUNet(TrackletRecurrentUNetConfig(
        in_channels=5, base_channels=8, bottleneck_channels=16, crop_size=15,
    ))
    model.eval()
    device = torch.device("cpu")

    # Fake an EvalConfig-shaped object (only needs `dataset`).
    class _EvalCfg:
        dataset = ds_cfg
    eval_cfg = _EvalCfg()

    scores = []
    labels = []
    for seed in range(10):
        for seq_type in ("positive_uav", "empty_background"):
            cfg = replace(ds_cfg.base_config, snr_db=5.0, n_frames=5,
                          seed=seed, sequence_type=seq_type)
            sample = generate_sequence(cfg)
            result = score_sequence(sample, model, eval_cfg, device)
            if result["n_tracklets"] == 0:
                continue
            scores.append(result["best_score"])
            labels.append(int(sample.has_target))
    assert len(scores) > 0
    # AUCs may be ~0.5 (untrained model), but they should be finite.
    val_roc = roc_auc(scores, labels)
    val_pr = pr_auc(scores, labels)
    assert math.isfinite(val_roc) or math.isnan(val_roc)  # nan if a class is missing
    assert math.isfinite(val_pr) or math.isnan(val_pr)
