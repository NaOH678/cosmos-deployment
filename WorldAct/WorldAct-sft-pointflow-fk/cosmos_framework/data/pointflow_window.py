"""Read fixed-camera Track4World windows without memory-mapping large NPY files."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PointFlowTiming:
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

    @classmethod
    def from_cosmos(cls, dataset_config, tokenizer_config):
        timing = cls(
            fps=float(dataset_config["fps"]),
            steps=dataset_config["chunk_length"],
            steps_per_token=tokenizer_config["temporal_compression_factor"],
        )
        durations = tokenizer_config.get("encode_exact_durations")
        if durations is not None and timing.steps + 1 not in durations:
            raise ValueError("PointFlow states do not match Cosmos encode_exact_durations")
        return timing


def _episode_metadata(episode: Path):
    """Return (video, width, height) for both delivery formats.

    Dense exports carry COMPLETE.json; the labeled delivery (pf_out README) is a
    quality-filtered subset of the same metric camera_depthanythingv3 export and
    carries report.json instead -- no intrinsics, and the native tracker grid is
    fixed by the export pipeline.
    """
    complete = episode / "COMPLETE.json"
    if complete.is_file():
        metadata = json.loads(complete.read_text())
        if metadata.get("coordinate") != "camera_depthanythingv3" or not metadata.get("metric_scale"):
            raise ValueError("Expected metric camera-coordinate Track4World data")
        return metadata["video"], int(metadata["inference_width"]), int(metadata["inference_height"])
    report = episode / "report.json"
    if report.is_file():
        metadata = json.loads(report.read_text())
        width, height = 640, 448
        queries = metadata.get("native_pixel_queries")
        if queries is None:
            # Older labeled deliveries predate the native_pixel_queries field.
            # Verify the fixed tracker grid against the stored uv range instead.
            uv = read_frame(episode / "uv_px.npy", 0)
            valid = read_frame(episode / "valid.npy", 0)
            sel = uv[valid]
            if sel.size == 0 or sel.min() < 0 or sel[:, 0].max() > width - 1 or sel[:, 1].max() > height - 1:
                raise ValueError(f"{episode}: labeled uv outside the {width}x{height} tracker grid")
        elif int(queries) != width * height:
            raise ValueError(f"{episode}: unexpected labeled query grid")
        return metadata["video"], width, height
    raise ValueError(f"{episode}: expected COMPLETE.json (dense export) or report.json (labeled delivery)")


def read_frame(path: Path, index: int) -> np.ndarray:
    """Read one C-order frame, including on filesystems which reject mmap."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY version {version}: {path}")
        if fortran or dtype.hasobject or not 0 <= index < shape[0]:
            raise ValueError(f"Invalid frame/layout: {path}, index={index}")
        size = int(np.prod(shape[1:])) * dtype.itemsize
        stream.seek(index * size, 1)
        content = stream.read(size)
        if len(content) != size:
            raise ValueError(f"Truncated NPY: {path}")
        return np.frombuffer(content, dtype=dtype).reshape(shape[1:]).copy()


def window_rows(frame_ids, timestamps, start_frame, fps=15.0, steps=32):
    """Match the target timeline exactly; reject short or irregular windows."""
    if (
        not np.isfinite(fps)
        or fps <= 0
        or steps < 1
        or frame_ids.ndim != 1
        or timestamps.ndim != 1
        or len(frame_ids) != len(timestamps)
        or len(timestamps) < 2
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
        or np.any(np.diff(frame_ids) <= 0)
    ):
        raise ValueError("Invalid timeline")
    starts = np.flatnonzero(frame_ids == start_frame)
    if len(starts) != 1:
        raise ValueError(f"Start frame {start_frame} not uniquely present")
    target = timestamps[starts[0]] + np.arange(steps + 1) / fps
    right = np.searchsorted(timestamps, target).clip(0, len(timestamps) - 1)
    left = (right - 1).clip(0)
    rows = np.where(abs(timestamps[left] - target) < abs(timestamps[right] - target), left, right)
    if np.max(abs(timestamps[rows] - target)) > 1e-4 or len(np.unique(rows)) != steps + 1:
        raise ValueError("Window does not cover the requested Cosmos timeline exactly")
    return rows


def _stratified_top_n(rank_score, labels, budget, quotas):
    """Split the top-N budget across semantic regions, rank within each region.

    ``quotas`` is a tuple of ``(label, fraction)``; each region takes its
    ``round(budget * fraction)`` best-ranked points.  Any shortfall (a region
    with too few candidates) is redistributed globally by rank score, so the
    total stays at ``budget`` whenever enough points exist.  Deterministic.
    """
    order = np.argsort(-rank_score, kind="stable")
    keep = np.zeros(len(rank_score), dtype=bool)
    for label, fraction in quotas:
        quota = int(round(budget * fraction))
        if quota <= 0:
            continue
        region_idx = np.flatnonzero(labels == label)
        region_sorted = region_idx[np.argsort(-rank_score[region_idx], kind="stable")]
        keep[region_sorted[:quota]] = True
    missing = budget - int(keep.sum())
    if missing > 0:
        keep[order[~keep[order]][:missing]] = True
    return keep


def prepare_window(
    episode: Path,
    start_frame=0,
    max_points=8192,
    voxel_size=0.02,
    seed=0,
    timing: PointFlowTiming | None = None,
    allow_empty=False,
    select_motion_fraction=0.0,
    select_top_n=0,
    min_voxel_members=0,
    supervise_cluster_n=0,
    select_regions=(),
    select_region_quotas=(),
    select_min_valid_steps=0,
    select_phantom_guard=False,
    phantom_guard_disp_mm=30.0,
    phantom_guard_uv_px=2.0,
    frame_provider=None,
    anchor_frame_bgr=None,
):
    """Prepare a Cosmos-timed window, selecting input IDs using only the anchor.

    Defaults retain the existing smoke-test contract. Training callers supply timing
    from the resolved Cosmos config and allow_empty=True to retain empty windows.

    ``select_motion_fraction`` keeps only the fastest-moving fraction of the anchor
    points, ranked by ground-truth displacement magnitude.  ``0`` keeps the full cloud.
    This is a **diagnostic**: the ranking reads the future labels, so a model trained
    this way cannot be conditioned the same way at inference, where no future exists.
    It exists to check whether the PointFlow branch can fit motion at all, on a handful
    of clusters instead of all 8192 points.

    ``select_top_n`` keeps exactly N points by the same ranking, and unlike the fraction
    it has **no floor** -- it exists to drive the cloud down to one point, where the
    question "one cluster token has to describe many points" disappears entirely and the
    remaining pipeline can be tested on its own.  ``0`` disables it.  When both are set,
    ``select_top_n`` wins.

    ``min_voxel_members`` guards every motion ranking: a point whose voxel holds fewer
    than this many points is ranked last, because a lost track is always alone.  ``0``
    disables the guard.

    ``supervise_cluster_n`` changes **only the loss mask, never the cloud**: it keeps the
    N points nearest the centroid of the most-moving voxel that satisfies
    ``min_voxel_members``, and marks every other point invalid.  The geometry encoder
    still sees the whole surface, so a failure here is the decode path's, not the
    encoder's.  ``0`` supervises everything.

    ``select_regions`` restricts the anchor candidates to the given semantic region
    labels (1=fingertip, 2=hand, 3+=objects in the labeled delivery).  It reads only
    anchor-time information, so unlike the motion ranking it stays a deployable
    conditioning rule.  It requires region_labels.npy; ``()`` keeps every region.

    ``select_region_quotas`` stratifies the ``select_top_n`` budget across semantic
    regions: a tuple of ``(label, fraction)``, each region keeps its quota of
    best-ranked points (same guards, same motion ranking), and any shortfall is
    redistributed globally so the total stays at N.  Requires ``select_top_n > 0``
    and region_labels.npy; ``()`` falls back to the flat global ranking.

    ``select_min_valid_steps`` demotes points whose future label is valid for fewer
    than this many steps in motion-ranked selection -- a track that dies mid-window
    cannot be the anchor of a full-window prediction.  Like the motion ranking it
    reads the labels, so it is a training-time selection rule.  ``0`` disables it.

    ``select_phantom_guard`` handles phantom-drift points: the tracker's uv stays
    glued (< ``phantom_guard_uv_px`` over the window) while the 3D label drifts
    (> ``phantom_guard_disp_mm`` at the last valid step) -- a depth-estimation
    failure, not real motion.  With a selection budget active the phantom points
    are demoted to the bottom of the motion ranking; without a budget (no
    ``select_top_n``/``select_motion_fraction``) ranking is unused, so they are
    dropped from the cloud outright.  It reads the labels, so like the motion
    ranking it is a training-time rule; measurement lives in
    tools/scan_pointflow_phantom_drift.py (same thresholds).
    ``frame_provider`` / ``anchor_frame_bgr`` let batch builders serve frames from
    memory: the provider is called as ``frame_provider(name, labeled_row)`` with
    name in {"position", "uv_px", "valid"} in place of the per-frame seek-read,
    and a pre-decoded anchor frame skips the per-window video open/seek entirely.
    Both default to the standalone on-disk behavior.
    """
    timing = timing or PointFlowTiming()
    episode = Path(episode)
    if max_points < 3 or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("max_points must be >=3 and voxel_size positive")
    select_regions = tuple(int(region) for region in select_regions)
    quotas = tuple((int(label), float(fraction)) for label, fraction in select_region_quotas)
    if quotas:
        if select_top_n <= 0:
            raise ValueError("select_region_quotas requires select_top_n > 0 (the total budget)")
        if any(fraction <= 0 or not np.isfinite(fraction) for _, fraction in quotas):
            raise ValueError(f"select_region_quotas fractions must be positive: {quotas}")
        total = sum(fraction for _, fraction in quotas)
        if total > 1.0 + 1e-6:
            raise ValueError(f"select_region_quotas fractions sum to {total} > 1: {quotas}")
        labels_seen = [label for label, _ in quotas]
        if len(set(labels_seen)) != len(labels_seen):
            raise ValueError(f"select_region_quotas has duplicate labels: {quotas}")
    if type(select_min_valid_steps) is not int or select_min_valid_steps < 0:
        raise ValueError("select_min_valid_steps must be an integer >= 0")
    if select_phantom_guard:
        if not np.isfinite(phantom_guard_disp_mm) or phantom_guard_disp_mm <= 0:
            raise ValueError("phantom_guard_disp_mm must be finite and positive")
        if not np.isfinite(phantom_guard_uv_px) or phantom_guard_uv_px < 0:
            raise ValueError("phantom_guard_uv_px must be finite and >= 0")
    video, width, height = _episode_metadata(episode)
    frame_ids = np.load(episode / "frame_indices.npy", allow_pickle=False)
    timestamps = np.load(episode / "timestamps_sec.npy", allow_pickle=False)
    rows = window_rows(frame_ids, timestamps, start_frame, timing.fps, timing.steps)

    def _frame(name, index):
        if frame_provider is not None:
            return frame_provider(name, int(index))
        return read_frame(episode / f"{name}.npy", int(index))

    xyz = _frame("position", rows[0]).reshape(-1, 3)
    uv = _frame("uv_px", rows[0]).reshape(-1, 2)
    valid = _frame("valid", rows[0]).reshape(-1)
    good = valid & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0) & np.isfinite(uv).all(1)
    good &= (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    region_labels = None
    if select_regions or quotas:
        labels_path = episode / "region_labels.npy"
        if not labels_path.is_file():
            raise ValueError(f"{episode}: region selection requires region_labels.npy")
        region_labels = np.load(labels_path, allow_pickle=False).reshape(-1)
        if region_labels.shape != (xyz.shape[0],):
            raise ValueError(f"{episode}: region_labels does not match the point count")
        if select_regions:
            good &= np.isin(region_labels, np.asarray(select_regions))
    candidates = np.flatnonzero(good)
    if len(candidates) < 3 and not allow_empty:
        raise ValueError("Fewer than 3 valid observed anchor points")
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(candidates, min(max_points, len(candidates)), replace=False))
    # Estimate normals in current 3D space, never in the original query-ID image grid.
    if len(candidates) >= 3:
        tree = cKDTree(xyz[candidates])
        _, neighbours = tree.query(xyz[ids], k=min(16, len(candidates)), workers=1)
        local = xyz[candidates][neighbours].astype(np.float64)
        local -= local.mean(1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", local, local) / local.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        reliable = eigenvalues[:, 1] > 1e-12
        ids = ids[reliable]
        normals = eigenvectors[reliable, :, 0].astype(np.float32)
        normals[np.sum(normals * xyz[ids], axis=1) > 0] *= -1  # face the observing camera
    else:
        ids = np.empty(0, dtype=np.int64)
        normals = np.empty((0, 3), dtype=np.float32)
    if len(ids) < 3 and not allow_empty:
        raise ValueError("Degenerate anchor neighbourhoods: cannot estimate normals")

    if anchor_frame_bgr is None:
        capture = cv2.VideoCapture(video)
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open head video: {video}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_ids[rows[0]]))
            ok, bgr = capture.read()
            if not ok:
                raise ValueError("Cannot decode anchor head frame")
        finally:
            capture.release()
    else:
        bgr = anchor_frame_bgr
    # The exporter directly resizes head RGB to the tracker inference resolution.
    rgb = cv2.cvtColor(cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    anchor_xyz, anchor_uv = xyz[ids], uv[ids]
    colors = (
        cv2.remap(rgb, anchor_uv[:, 0].reshape(-1, 1), anchor_uv[:, 1].reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1, 3)
        if len(ids)
        else np.empty((0, 3), dtype=np.uint8)
    )

    def voxelize(sel_xyz, sel_colors, sel_normals):
        """Anchor-only voxelization; deterministic representative per voxel."""
        shift = np.zeros(3, dtype=np.float32)
        if len(sel_xyz):
            shift = (sel_xyz.min(0) + sel_xyz.max(0)) / 2
            shift[2] = sel_xyz[:, 2].min()
        centered = sel_xyz - shift
        grid = np.floor(centered / voxel_size).astype(np.int64)
        if len(sel_xyz):
            grid -= grid.min(0)
        # Deterministic representative per voxel; inverse preserves every selected original ID.
        _, representatives, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        feats = np.concatenate((centered, sel_colors.astype(np.float32) / 255, sel_normals), axis=1)
        return shift, centered, grid, representatives, inverse, feats

    shift, centered, grid, representatives, inverse, features = voxelize(anchor_xyz, colors, normals)
    target = np.zeros((timing.steps, len(ids), 3), dtype=np.float32)
    target_valid = np.zeros((timing.steps, len(ids)), dtype=bool)
    for k, row in enumerate(rows[1:]):
        future = _frame("position", row).reshape(-1, 3)[ids]
        mask = _frame("valid", row).reshape(-1)[ids]
        mask &= np.isfinite(future).all(1) & (future[:, 2] > 0)
        target_valid[k] = mask
        target[k, mask] = future[mask] - anchor_xyz[mask]

    # Per-point motion drives both the selection and the supervision mask, and is cheap
    # enough to always compute.
    magnitude = np.linalg.norm(target, axis=-1)  # [steps, N]
    counts = target_valid.sum(0)  # [N]
    per_point = np.where(counts > 0, (magnitude * target_valid).sum(0) / np.maximum(counts, 1), 0.0)

    # A real surface patch leaves several points in one 2 cm cell; a tracker that lost a
    # point leaves it alone.  Measured on the sandwich data: the ten fastest points of a
    # window all sit in single-member voxels and "move" up to 1.6 m per point, while the
    # fastest point in a voxel of three or more moves 0.15 m.  Ranking only among the
    # latter is what keeps a motion-ranked selection physical.
    if min_voxel_members > 0 and len(ids):
        members = np.bincount(inverse, minlength=len(representatives))
        rank_score = np.where(members[inverse] >= min_voxel_members, per_point, -np.inf)
    else:
        rank_score = per_point
    if select_min_valid_steps > 0 and len(ids):
        rank_score = np.where(counts >= select_min_valid_steps, rank_score, -np.inf)
    if select_phantom_guard and len(ids):
        uv_last = _frame("uv_px", rows[-1]).reshape(-1, 2)[ids]
        uv_move = np.linalg.norm(uv_last - anchor_uv, axis=1)
        last = np.where(counts > 0, timing.steps - 1 - np.argmax(target_valid[::-1], axis=0), -1)
        final = np.array([magnitude[last[i], i] if last[i] >= 0 else 0.0 for i in range(len(ids))])
        phantom = (final * 1000 > phantom_guard_disp_mm) & (uv_move < phantom_guard_uv_px)
        if select_top_n <= 0 and select_motion_fraction <= 0:
            # No selection budget: ranking is unused, so demotion is a no-op --
            # drop the phantom points outright instead.
            if phantom.any() and (~phantom).any():
                keep = ~phantom
                ids = ids[keep]
                anchor_xyz, anchor_uv = anchor_xyz[keep], anchor_uv[keep]
                normals, colors = normals[keep], colors[keep]
                target, target_valid = target[:, keep], target_valid[:, keep]
                per_point = per_point[keep]
                shift, centered, grid, representatives, inverse, features = voxelize(anchor_xyz, colors, normals)
        else:
            rank_score = np.where(phantom, -np.inf, rank_score)

    if select_top_n > 0:
        keep_count, keep_floor = min(int(select_top_n), len(ids)), 1
    elif select_motion_fraction > 0:
        keep_count, keep_floor = max(3, int(round(select_motion_fraction * len(ids)))), 3
    else:
        keep_count, keep_floor = 0, 0
    if keep_count and len(ids):
        # Keep the fastest-moving FRACTION OF POINTS.  Two measured reasons this beats
        # keeping the top-K voxels outright:
        #   * the single fastest voxels are tracking outliers -- one window's top voxel
        #     "moved" 1516 mm and had an empty 15 cm neighbourhood -- so an extreme tail
        #     selects artefacts, not a moving part;
        #   * dropping the other members of a voxel leaves a set so sparse PointTransformerV3
        #     has nothing to pool.  Measured across 10 episodes: top-K voxels (K=4..32)
        #     left 4-33 points at 77-302 mm spacing, while the top 5% of points leaves
        #     409 points at 4-10 mm -- the same density as the full cloud.
        # Motion is spatially correlated (the arm moves as one piece), so a fraction
        # stays dense where an extreme tail does not.
        #
        # ``select_top_n`` deliberately uses a floor of 1 instead of 3: it exists to go
        # below the fraction's floor, down to a single point.
        if quotas:
            # Stratified: per-region quota of best-ranked points, shortfall refilled
            # globally, so region coverage survives windows where the hand dominates
            # the global motion ranking.
            keep_point = _stratified_top_n(rank_score, region_labels[ids], keep_count, quotas)
        else:
            order = np.argsort(-rank_score, kind="stable")
            keep_point = np.zeros(len(ids), dtype=bool)
            keep_point[order[:keep_count]] = True
        if keep_point.sum() >= keep_floor and (~keep_point).any():
            # Subset the points, then re-derive the voxels on what is left: cheaper
            # and less error-prone than renumbering the existing mapping.
            ids = ids[keep_point]
            anchor_xyz, anchor_uv = anchor_xyz[keep_point], anchor_uv[keep_point]
            normals, colors = normals[keep_point], colors[keep_point]
            target, target_valid = target[:, keep_point], target_valid[:, keep_point]
            per_point = per_point[keep_point]
            shift, centered, grid, representatives, inverse, features = voxelize(anchor_xyz, colors, normals)

    # Supervise a handful of points from the single most-moving *real* cluster and mask
    # everything else out of the loss.  The cloud itself is untouched, so the geometry
    # encoder still sees a full surface: this separates "the decode path cannot emit one
    # point's motion" from "the geometry input was degenerate", which a cloud of one
    # point could not.
    if supervise_cluster_n > 0 and len(representatives):
        members = np.bincount(inverse, minlength=len(representatives))
        sums = np.bincount(inverse, weights=per_point, minlength=len(representatives))
        floor = max(int(min_voxel_members), 1)
        voxel_motion = np.where(members >= floor, sums / np.maximum(members, 1), -np.inf)
        if np.isfinite(voxel_motion).any():
            inside = np.flatnonzero(inverse == int(np.argmax(voxel_motion)))
            # Nearest to the voxel centroid, not the fastest member: the fastest points of
            # a patch sit on its noisy edge, the centroid is what the patch as a whole does.
            centroid = centered[inside].mean(0)
            offsets = np.linalg.norm(centered[inside] - centroid, axis=-1)
            supervised = inside[np.argsort(offsets, kind="stable")[: int(supervise_cluster_n)]]
            keep_supervised = np.zeros(len(ids), dtype=bool)
            keep_supervised[supervised] = True
            target_valid &= keep_supervised[None, :]

    intrinsics_path = episode / "intrinsics.npy"
    if intrinsics_path.is_file():
        intrinsics = read_frame(intrinsics_path, int(rows[0]))
        pixel_k = np.diag([width, height, 1]) @ intrinsics
        projected = anchor_xyz @ pixel_k.T
        projected = projected[:, :2] / projected[:, 2:]
        projection_error = np.linalg.norm(projected - anchor_uv, axis=1)
    else:
        # Labeled deliveries ship no intrinsics: skip the projection guard and mark
        # both outputs NaN so an accidental consumer fails loudly instead of
        # silently trusting an identity matrix.
        intrinsics = np.full((3, 3), np.nan, dtype=np.float32)
        projection_error = np.full(len(ids), np.nan, dtype=np.float32)
    return {
        "point_ids": ids,
        "anchor_xyz": anchor_xyz,
        "anchor_uv": anchor_uv,
        "normal": normals,
        "color": colors,
        "coord_shift": shift,
        "coord": centered[representatives],
        "feat": features[representatives],
        "grid_coord": grid[representatives],
        "original_to_voxel": inverse,
        "voxel_representatives": representatives,
        "target_displacement": target,
        "target_valid": target_valid,
        "raw_frame_ids": frame_ids[rows],
        "timestamps_sec": timestamps[rows],
        "projection_error_px": projection_error,
        "anchor_rgb": rgb,
        "intrinsics_normalized": intrinsics,
        "image_size_wh": np.array([width, height], dtype=np.int64),
        "target_seconds": timestamps[rows] - timestamps[rows[0]],
        "point_block_seconds": np.arange(timing.steps_per_token, timing.steps + 1, timing.steps_per_token) / timing.fps,
    }
