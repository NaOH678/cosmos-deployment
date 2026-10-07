"""Optional per-episode dense labels, aligned to the action dataset's exact window."""

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np

from cosmos_framework.data.pointflow_dataset import pointflow_sample
from cosmos_framework.data.pointflow_window import PointFlowTiming, prepare_window
from cosmos_framework.data.pointflow_window_cache import SELECTION_KEYS, read_cached_window, validate_manifest

logger = logging.getLogger(__name__)


def window_seed(seed, episode, start_frame):
    """Per-window deterministic seed; the cache builder must reproduce this exactly."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{episode}:{int(start_frame)}".encode()).digest()[:8], "little")


class PointFlowSource:
    """Manifest entries explicitly declare labeled and unlabeled episodes.

    Paths resolve relative to the manifest, never the process working directory.
    uv_to_video maps tracker pixel centers into the dataset's returned video.
    """

    def __init__(
        self,
        manifest,
        *,
        timing: PointFlowTiming,
        max_points=8192,
        voxel_size=0.02,
        seed=0,
        select_motion_fraction=0.0,
        select_top_n=0,
        min_voxel_members=0,
        supervise_cluster_n=0,
        select_regions=(),
        select_region_quotas=(),
        select_min_valid_steps=0,
        select_phantom_guard=False,
        window_cache_root=None,
        phantom_guard_disp_mm=30.0,
        phantom_guard_uv_px=2.0,
    ):
        path = Path(manifest).resolve()
        document = json.loads(path.read_text())
        if document.get("schema_version") != 1:
            raise ValueError("Expected PointFlow manifest schema_version=1")
        self.entries = {}
        for row in document["episodes"]:
            name = row["name"]
            if name in self.entries:
                raise ValueError(f"Duplicate PointFlow episode: {name}")
            source = row["pointflow_source"]
            if source is not None:
                source = dict(source)
                source["path"] = (path.parent / source["path"]).resolve()
                affine = np.asarray(source["uv_to_video"], dtype=np.float32)
                size = np.asarray(source["video_size_wh"])
                if affine.shape != (2, 3) or not np.isfinite(affine).all():
                    raise ValueError(f"{name}: expected finite uv_to_video[2,3]")
                if size.shape != (2,) or not np.all(size > 0):
                    raise ValueError(f"{name}: invalid video_size_wh")
                source["uv_to_video"] = affine
                offset = float(source.get("timestamp_offset_sec", 0.0))
                if not np.isfinite(offset):
                    raise ValueError(f"{name}: invalid timestamp_offset_sec")
                source["timestamp_offset_sec"] = offset
            self.entries[name] = source
        self.timing, self.max_points, self.voxel_size, self.seed = timing, max_points, voxel_size, seed
        # `${oc.env:...}` in the recipe always yields a STRING, so coerce before
        # validating -- np.isfinite("0.05") is a TypeError, not a range error.
        try:
            fraction = float(select_motion_fraction)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"select_motion_fraction must be a number in [0, 1), got {select_motion_fraction!r}"
            ) from error
        if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError(f"select_motion_fraction must be in [0, 1), got {fraction}")
        self.select_motion_fraction = fraction

        # Same string coercion: these also arrive through `${oc.env:...}`.
        def non_negative_int(value, name):
            try:
                number = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}") from error
            if number < 0:
                raise ValueError(f"{name} must be >= 0, got {number}")
            return number

        self.select_top_n = non_negative_int(select_top_n, "select_top_n")
        self.min_voxel_members = non_negative_int(min_voxel_members, "min_voxel_members")
        self.supervise_cluster_n = non_negative_int(supervise_cluster_n, "supervise_cluster_n")
        self.select_min_valid_steps = non_negative_int(select_min_valid_steps, "select_min_valid_steps")
        # `${oc.env:POINTFLOW_SELECT_REGIONS,}` arrives as a comma-separated string.
        if isinstance(select_regions, str):
            select_regions = tuple(part.strip() for part in select_regions.split(",") if part.strip())
        try:
            regions = tuple(int(region) for region in select_regions)
        except (TypeError, ValueError) as error:
            raise ValueError(f"select_regions must be comma-separated integers, got {select_regions!r}") from error
        if any(region < 1 or region > 255 for region in regions):
            raise ValueError(f"select_regions labels must be in [1, 255], got {regions}")
        self.select_regions = regions
        # `${oc.env:POINTFLOW_SELECT_REGION_QUOTAS,}` arrives as "2:0.40,3:0.45,4:0.15".
        if isinstance(select_region_quotas, str):
            parts = [part.strip() for part in select_region_quotas.split(",") if part.strip()]
            try:
                select_region_quotas = tuple(
                    (int(label), float(fraction)) for label, fraction in (part.split(":") for part in parts)
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"select_region_quotas must be comma-separated label:fraction pairs, got {select_region_quotas!r}"
                ) from error
        self.select_region_quotas = tuple(select_region_quotas)
        # Bool arriving as a string through `${oc.env:...}`.
        if isinstance(select_phantom_guard, str):
            select_phantom_guard = select_phantom_guard.strip().lower() in {"1", "true", "yes"}
        self.select_phantom_guard = bool(select_phantom_guard)
        self.phantom_guard_disp_mm = float(phantom_guard_disp_mm)
        self.phantom_guard_uv_px = float(phantom_guard_uv_px)
        # Per-window selection cache (cosmos_framework/data/pointflow_window_cache.py).
        # The cache is keyed by (episode, start_frame); a miss replays the online
        # path with an identical result.  Handles are LRU-capped: one open npz holds
        # one fd, and hundreds of dataloader workers otherwise exhaust ulimit.
        self.window_cache_root = Path(window_cache_root) if window_cache_root else None
        self._archives: OrderedDict[str, tuple[np.lib.npyio.NpzFile, int]] = OrderedDict()
        self._archive_cache_size = 4
        self.cache_misses = 0
        if self.window_cache_root is not None:
            validate_manifest(self.window_cache_root, self.selection_config(), SELECTION_KEYS)

    def selection_config(self):
        """The selection knobs recorded in the window cache manifest."""
        return {
            "max_points": self.max_points,
            "voxel_size": self.voxel_size,
            "seed": self.seed,
            "select_motion_fraction": self.select_motion_fraction,
            "select_top_n": self.select_top_n,
            "min_voxel_members": self.min_voxel_members,
            "supervise_cluster_n": self.supervise_cluster_n,
            "select_regions": self.select_regions,
            "select_region_quotas": self.select_region_quotas,
            "select_min_valid_steps": self.select_min_valid_steps,
            "select_phantom_guard": self.select_phantom_guard,
            "phantom_guard_disp_mm": self.phantom_guard_disp_mm,
            "phantom_guard_uv_px": self.phantom_guard_uv_px,
        }

    def _cached_window(self, episode, start):
        archive_path = self.window_cache_root / f"{episode}.npz"
        if not archive_path.is_file():
            return None
        cached = self._archives.get(episode)
        if cached is None:
            archive = np.load(archive_path)
            self._archives[episode] = (archive, 0)
            while len(self._archives) > self._archive_cache_size:
                _, (evicted, _) = self._archives.popitem(last=False)
                evicted.close()
            cached = self._archives[episode]
        else:
            self._archives.move_to_end(episode)
        return read_cached_window(cached[0], start, self.timing)

    def load(self, episode, frame_ids, source_fps, video_size_wh):
        # A missing entry is a configuration error, not a silently missing label.
        source = self.entries[episode]
        if source is None:
            return None
        # The source canvas and Cosmos output canvas are intentionally different;
        # resize_pointflow_metadata composes the affine after video preprocessing.
        seed = window_seed(self.seed, episode, frame_ids[0])
        # ``uv_to_video`` is authored for the tracker canvas recorded in the
        # manifest, and ``resize_pointflow_metadata`` reads ``video_size_wh`` as
        # exactly that: the canvas the affine currently maps into, which it rebases
        # onto whatever tensor it is about to transform.  Overwriting it with the
        # canvas the dataset happens to return destroys that information -- the
        # align step then sees "recorded == actual", concludes the affine is
        # already in the returned canvas, and skips the very rescale it exists to
        # perform.  Keep the manifest's value; the caller's canvas is recorded
        # separately so a mismatch stays visible.
        recorded = np.asarray(source["video_size_wh"], dtype=np.int64)
        reported = np.asarray(video_size_wh, dtype=np.int64)
        if reported.shape != (2,):
            raise ValueError(f"{episode}: returned video canvas must be (width, height), got {reported.shape}")
        window = self._cached_window(episode, int(frame_ids[0])) if self.window_cache_root else None
        if window is None:
            if self.window_cache_root is not None:
                self.cache_misses += 1
                if self.cache_misses <= 3 or self.cache_misses % 1000 == 0:
                    logger.warning(
                        "PointFlow window cache miss #%d: %s@%s (falling back to online preparation)",
                        self.cache_misses,
                        episode,
                        int(frame_ids[0]),
                    )
            window = prepare_window(
                source["path"],
                int(frame_ids[0]),
                self.max_points,
                self.voxel_size,
                seed,
                timing=self.timing,
                allow_empty=True,
                select_motion_fraction=self.select_motion_fraction,
                select_top_n=self.select_top_n,
                min_voxel_members=self.min_voxel_members,
                supervise_cluster_n=self.supervise_cluster_n,
                select_regions=self.select_regions,
                select_region_quotas=self.select_region_quotas,
                select_min_valid_steps=self.select_min_valid_steps,
                select_phantom_guard=self.select_phantom_guard,
                phantom_guard_disp_mm=self.phantom_guard_disp_mm,
                phantom_guard_uv_px=self.phantom_guard_uv_px,
            )
        if not np.array_equal(window["raw_frame_ids"], frame_ids):
            raise ValueError(f"{episode}: video and PointFlow frame IDs differ")
        expected_time = np.asarray(frame_ids) / source_fps + source["timestamp_offset_sec"]
        if not np.allclose(window["timestamps_sec"], expected_time, atol=1e-4, rtol=0):
            raise ValueError(f"{episode}: video and PointFlow timestamps differ")
        result = pointflow_sample(window, episode, int(frame_ids[0]), seed, self.timing)
        result["metadata"]["source_path"] = str(source["path"])
        result["metadata"]["uv_to_video"] = source["uv_to_video"].copy()
        result["metadata"]["video_size_wh"] = recorded
        result["metadata"]["returned_video_size_wh"] = reported
        return result


def resize_pointflow_metadata(pointflow, original_wh, resized_wh, canvas_wh):
    """Compose align_corners=False pixel-center resize; padding is bottom/right.

    ``video_size_wh`` always names the canvas the stored affine maps into, so the
    caller must pass the size of the tensor it is about to transform.  Seeing that
    canvas already equal ``canvas_wh`` while ``original_wh`` differs means a
    previous call already composed the full chain and this one would rescale an
    affine that is no longer in ``original_wh`` space.
    """
    if pointflow is None:
        return
    metadata = pointflow["metadata"]
    if tuple(np.asarray(metadata["video_size_wh"]).tolist()) == tuple(canvas_wh) and tuple(original_wh) != tuple(
        canvas_wh
    ):
        raise ValueError(
            f"PointFlow metadata is stale: the affine already maps into the {tuple(canvas_wh)} canvas, "
            f"but the caller reports a {tuple(original_wh)} source"
        )
    if tuple(metadata["video_size_wh"]) != tuple(original_wh):
        recorded = np.asarray(metadata["video_size_wh"], dtype=np.float32)
        actual = np.asarray(original_wh, dtype=np.float32)
        scale0 = actual / recorded
        align = np.eye(3, dtype=np.float32)
        align[0, 0], align[1, 1] = scale0
        metadata["uv_to_video"] = (align @ np.vstack([metadata["uv_to_video"], [0, 0, 1]]))[:2]
        metadata["video_size_wh"] = actual.astype(np.int64)
    scale = np.asarray(resized_wh, dtype=np.float32) / np.asarray(original_wh)
    resize = np.eye(3, dtype=np.float32)
    resize[0, 0], resize[1, 1] = scale
    resize[:2, 2] = (scale - 1) / 2
    previous = np.eye(3, dtype=np.float32)
    previous[:2] = metadata["uv_to_video"]
    metadata["uv_to_video"] = (resize @ previous)[:2]
    metadata["video_size_wh"] = np.asarray(canvas_wh, dtype=np.int64)
