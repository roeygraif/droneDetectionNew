"""Evaluation tools for the tracklet-guided UAV detector."""

from eval.eval_tracklet_detector import (
    EvalConfig,
    SequenceResult,
    evaluate,
    score_sequence,
)

__all__ = [
    "EvalConfig",
    "SequenceResult",
    "evaluate",
    "score_sequence",
]
