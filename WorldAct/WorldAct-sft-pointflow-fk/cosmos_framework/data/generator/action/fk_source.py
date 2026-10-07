"""Per-episode FK-21 keypoint labels, aligned to the action dataset's exact window.

The FK counterpart of ``pointflow_source.py``, and deliberately much smaller: the
keypoints already carry their own identity (21 named joints, fixed order), so
there is no voxelisation, no point budget, no tracker, no image, and -- in v1 --
no uv.  What is left is: read 33 rows of the stored annotation, rotate them out of
the robot base frame, and difference them against the anchor.

Two contract details this module exists to enforce:

* the window is the *dataset's* ``frame_ids``, never a recomputed one.  The video
  and the labels must describe the same source frames or the mRoPE time axis
  silently misaligns -- and nothing downstream notices.
* the annotations are used as shipped.  ``tools/verify_fk21_allowlist.py``
  reproduces them bit-exactly from qpos through the wuji-mjlab pipeline, so a
  mismatch here would mean the *dataset* is wrong, not the labels.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np

from cosmos_framework.data.fk_camera_extrinsic import base_to_camera
from cosmos_framework.data.fk_window import FKTiming

# ``wuji_fk21.npz`` stores both hands; this dataset only ever observes the right.
HAND_INDEX = {"left": 0, "right": 1}
ANNOTATION = Path("annotations") / "wuji_fk21.npz"
EXPECTED_FRAME = "Link_Base"
EXPECTED_UNITS = "metre"
KEYPOINTS = 21


class FKSource:
    """Stored FK annotations for the episodes the action dataset is training on.

    ``root`` is the directory holding one subdirectory per episode
    (``raw_data/sandwich_fk21``), not the raw-data root: the annotations were
    exported to their own tree.
    """

    def __init__(self, root, *, timing: FKTiming, hand: str = "right", cache_size: int = 2):
        if hand not in HAND_INDEX:
            raise ValueError(f"hand must be one of {sorted(HAND_INDEX)}, got {hand!r}")
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"FK annotation root is not a directory: {self.root}")
        self.timing, self.hand, self.side = timing, hand, HAND_INDEX[hand]
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def path_for(self, episode: str) -> Path:
        return self.root / episode / ANNOTATION

    def positions(self, episode: str) -> np.ndarray:
        """``[T, 21, 3]`` for the configured hand, in the base frame, metres."""
        cached = self._cache.get(episode)
        if cached is not None:
            self._cache.move_to_end(episode)
            return cached
        path = self.path_for(episode)
        if not path.is_file():
            raise FileNotFoundError(f"missing FK annotation for {episode}: {path}")
        with np.load(path, allow_pickle=True) as store:
            missing = {"positions", "coordinate_frame", "units", "side_is_observed"} - set(store.files)
            if missing:
                raise ValueError(f"{episode}: annotation lacks {sorted(missing)}")
            frame = str(store["coordinate_frame"])
            units = str(store["units"])
            observed = np.asarray(store["side_is_observed"], dtype=bool)
            positions = np.asarray(store["positions"])
            clipped = bool(store["qpos_was_clipped"]) if "qpos_was_clipped" in store.files else False
        if frame != EXPECTED_FRAME:
            raise ValueError(f"{episode}: annotation frame is {frame!r}, expected {EXPECTED_FRAME!r}")
        if units != EXPECTED_UNITS:
            raise ValueError(f"{episode}: annotation units are {units!r}, expected {EXPECTED_UNITS!r}")
        # A clipped qpos means the stored keypoints describe a pose the robot never
        # held, so the labels would be quietly wrong rather than missing.
        if clipped:
            raise ValueError(f"{episode}: qpos was clipped to joint limits, FK labels are not trustworthy")
        if observed.shape != (2,) or not observed[self.side]:
            raise ValueError(f"{episode}: the {self.hand} hand is not marked observed ({observed})")
        if positions.ndim != 4 or positions.shape[1:3] != (2, KEYPOINTS) or positions.shape[-1] != 3:
            raise ValueError(f"{episode}: expected positions [T,2,{KEYPOINTS},3], got {positions.shape}")
        if not np.isfinite(positions).all():
            raise ValueError(f"{episode}: annotation contains non-finite keypoints")
        hand = np.ascontiguousarray(positions[:, self.side])
        self._cache[episode] = hand
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return hand

    def load(self, episode: str, frame_ids):
        """Build one window: anchor geometry plus the future displacement labels.

        ``frame_ids`` are the dataset's source frame numbers for this window,
        ``[1 + timing.steps]`` of them, strictly increasing.  Returns the same
        ``inputs`` / ``targets`` / ``metadata`` shape the PointFlow samples use so
        the batch builders and the packer treat both modalities alike.
        """
        frame_ids = np.asarray(frame_ids)
        expected = self.timing.steps + 1
        if frame_ids.shape != (expected,) or not np.issubdtype(frame_ids.dtype, np.integer):
            raise ValueError(f"{episode}: expected {expected} integer frame IDs, got {frame_ids.shape}")
        if np.any(np.diff(frame_ids) <= 0):
            raise ValueError(f"{episode}: frame IDs must strictly increase, got {frame_ids[:5]}...")
        positions = self.positions(episode)
        if frame_ids[-1] >= len(positions):
            raise ValueError(
                f"{episode}: window ends at frame {int(frame_ids[-1])} but only {len(positions)} are annotated"
            )
        # Rotate once for the whole window: the transform is a single constant, and
        # differencing in the camera frame is the same as differencing in base.
        camera = base_to_camera(positions[frame_ids]).astype(np.float64)  # [steps+1, 21, 3]
        anchor = camera[0]
        displacement = camera[1:] - anchor[None]
        valid = np.ones(displacement.shape[:2], dtype=bool)  # every annotated frame is usable
        return {
            "inputs": {
                "anchor_xyz": anchor.astype(np.float32),
                # Fixed anatomical identity, not a tracker ID: keypoint i is always
                # the same joint, which is what lets the encoder use per-index
                # embeddings instead of a permutation-invariant backbone.
                "point_ids": np.arange(KEYPOINTS, dtype=np.int64),
            },
            "targets": {
                "displacement": displacement.astype(np.float32),
                "valid": valid,
            },
            "metadata": {
                "episode": episode,
                "start_frame": int(frame_ids[0]),
                "raw_frame_ids": frame_ids.copy(),
                "timing": self.timing,
                "hand": self.hand,
                "coordinate": "camera_d435_real",
                "point_block_seconds": np.arange(
                    self.timing.steps_per_token, self.timing.steps + 1, self.timing.steps_per_token
                )
                / self.timing.fps,
            },
        }


def attach_fk_projection(fk, pointflow):
    """Adapt the existing MANO FK projector to the tracker canvas; no refitting."""
    from tools.verify_fk_camera_projection import IMG_H, IMG_W, project

    if pointflow is None:
        raise ValueError("Projected FK requires PointFlow canvas metadata")
    if not np.array_equal(fk["metadata"]["raw_frame_ids"], pointflow["metadata"]["raw_frame_ids"]):
        raise ValueError("FK and PointFlow must use identical source frames")
    uv, front = project(fk["inputs"]["anchor_xyz"])
    if not front.all() or not np.isfinite(uv).all():
        raise ValueError("FK anchors must be finite and in front of the camera")
    # Inverse of the tracker's pixel-center resize back to the raw head view.
    tracker_wh = np.asarray(pointflow["inputs"]["image_size_wh"])
    fk["inputs"]["anchor_uv"] = ((uv + 0.5) * tracker_wh / [IMG_W, IMG_H] - 0.5).astype(np.float32)
