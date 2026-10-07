#!/usr/bin/env python3
"""Build a PointFlow manifest mixing the labeled delivery with a legacy dense export.

Every episode under the raw dataset root gets an entry (the action dataset refuses
episodes missing from the manifest).  Episodes with a labeled delivery
(report.json + flat arrays + region_labels) point at it; episodes only present in
the legacy dense export keep their legacy entry; the rest are null (unlabeled).
The eval episode must stay labeled or fixed eval cases stop loading -- it lives in
the legacy export today.
"""

import argparse
import json
from pathlib import Path

REQUIRED_LABELED = (
    "report.json",
    "position.npy",
    "uv_px.npy",
    "valid.npy",
    "region_labels.npy",
    "frame_indices.npy",
    "timestamps_sec.npy",
)
REQUIRED_LEGACY = (
    "COMPLETE.json",
    "position.npy",
    "uv_px.npy",
    "valid.npy",
    "intrinsics.npy",
    "frame_indices.npy",
    "timestamps_sec.npy",
)
# Tracker canvas (640x448) -> dataset head video canvas, measured on the original
# sandwich export and unchanged since (fixed camera, same videos).
VIDEO_SIZE_WH = [640, 842]
UV_TO_VIDEO = [[1.0, 0, 0.0], [0, 1.0714285714285714, 362.0357142857143]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True, help="raw dataset root (defines the episode list)")
    parser.add_argument("--labeled-root", type=Path, required=True, help="pf_out labeled/ directory, preferred source")
    parser.add_argument(
        "--legacy-root", type=Path, default=None, help="legacy dense outputs/ directory, fallback source"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = sorted(p.name for p in args.raw_root.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not episodes:
        raise ValueError(f"no episodes under {args.raw_root}")

    rows, n_labeled, n_legacy = [], 0, 0
    for name in episodes:
        source = None
        labeled = args.labeled_root / name
        legacy = (args.legacy_root / name) if args.legacy_root else None
        if labeled.is_dir():
            missing = [f for f in REQUIRED_LABELED if not (labeled / f).is_file()]
            if missing:
                raise ValueError(f"{name}: labeled delivery incomplete: {missing}")
            source = {"path": str(labeled.resolve())}
            n_labeled += 1
        elif legacy is not None and legacy.is_dir():
            missing = [f for f in REQUIRED_LEGACY if not (legacy / f).is_file()]
            if missing:
                raise ValueError(f"{name}: legacy export incomplete: {missing}")
            source = {"path": str(legacy.resolve())}
            n_legacy += 1
        if source is not None:
            source.update(video_size_wh=VIDEO_SIZE_WH, uv_to_video=UV_TO_VIDEO)
        rows.append({"name": name, "pointflow_source": source})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "episodes": rows}, indent=2) + "\n")
    print(f"{len(rows)} episodes: {n_labeled} labeled, {n_legacy} legacy, {len(rows) - n_labeled - n_legacy} unlabeled")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
