"""Timing contract for FK windows.

The counterpart of ``pointflow_window.PointFlowTiming``, and a separate type on
purpose: FK runs in a worktree that contains no PointFlow module at all (v1 is
"FK only"), so importing the PointFlow one would drag the whole point-cloud data
path in for a four-field dataclass.  The fields and their validation are the same
because the two modalities must land on the same 15 Hz / 32-step / 4-step-per-token
lattice as the video -- that agreement is checked in ``from_cosmos``, which reads
the same dataset and tokenizer config the video path does.

If PointFlow and FK are ever loaded together (v2), compare the *fields*, not the
types: these are two distinct classes and ``==`` between them is always False.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FKTiming:
    """Physical timing copied from the resolved Cosmos dataset/tokenizer configuration."""

    fps: float = 15.0
    steps: int = 32
    steps_per_token: int = 4

    def __post_init__(self):
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if type(self.steps) is not int or type(self.steps_per_token) is not int:
            raise ValueError("steps and steps_per_token must be integers")
        if self.steps < 1 or self.steps_per_token < 1 or self.steps % self.steps_per_token:
            raise ValueError("Future steps must be positive and divisible by the tokenizer temporal compression")

    @property
    def blocks(self) -> int:
        """Number of motion blocks: one noisy token per keypoint per block."""
        return self.steps // self.steps_per_token

    @classmethod
    def from_cosmos(cls, dataset_config, tokenizer_config):
        timing = cls(
            fps=float(dataset_config["fps"]),
            steps=dataset_config["chunk_length"],
            steps_per_token=tokenizer_config["temporal_compression_factor"],
        )
        # The dataset's window length and the tokenizer's expected clip length have
        # to be the same number: if they disagree the video and the FK labels are
        # built for different durations, and the mRoPE time axis would drift by a
        # whole block without any shape error to catch it.
        durations = tokenizer_config.get("encode_exact_durations")
        if durations is not None and timing.steps + 1 not in durations:
            raise ValueError("FK states do not match Cosmos encode_exact_durations")
        return timing
