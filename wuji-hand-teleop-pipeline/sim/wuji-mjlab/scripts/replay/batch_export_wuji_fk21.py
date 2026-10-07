#!/usr/bin/env python3
"""Export FK-21 coordinates for every finalized episode below a directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import export_wuji_fk21


DEFAULT_URDF = export_wuji_fk21.DEFAULT_URDF


def find_episodes(root_dir: Path) -> list[Path]:
    """Return sorted episode directories that contain an LMDB database."""
    root_dir = root_dir.expanduser().resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"directory does not exist: {root_dir}")

    candidates = [root_dir] if root_dir.name.startswith("episode_") else []
    candidates.extend(root_dir.rglob("episode_*"))
    episodes = {
        path.resolve()
        for path in candidates
        if path.is_dir()
        and (path / "lmdb" / "data.mdb").is_file()
    }
    return sorted(episodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        type=Path,
        help="task/dataset directory containing episode_* directories",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "separate output directory; default is <root_dir_name>_fk21 "
            "next to root_dir"
        ),
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite episodes that already have both FK-21 output files",
    )
    parser.add_argument(
        "--clip-limits",
        action="store_true",
        help="clip measured qpos to URDF limits before FK",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list episodes and outputs without exporting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.root_dir.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else source_root.with_name(f"{source_root.name}_fk21")
    )
    if output_root == source_root:
        raise ValueError("--output-root must differ from the source root")

    episodes = find_episodes(source_root)
    if not episodes:
        raise FileNotFoundError(
            f"no episode_* directory containing lmdb/data.mdb under "
            f"{source_root}"
        )

    print(f"Found {len(episodes)} episode(s).")
    print(f"Output root: {output_root}")
    exported = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []

    for index, episode_dir in enumerate(episodes, start=1):
        relative_episode = (
            Path(episode_dir.name)
            if episode_dir == source_root
            else episode_dir.relative_to(source_root)
        )
        output = (
            output_root
            / relative_episode
            / "annotations"
            / "wuji_fk21.npz"
        )
        summary_path = output.with_suffix(".json")
        prefix = f"[{index}/{len(episodes)}] {episode_dir.name}"

        if args.dry_run:
            print(f"{prefix}: would export -> {output}")
            continue
        if not args.force and output.is_file() and summary_path.is_file():
            print(f"{prefix}: skipped (already exists)")
            skipped += 1
            continue

        try:
            summary = export_wuji_fk21.export(
                argparse.Namespace(
                    episode_dir=episode_dir,
                    urdf=args.urdf,
                    output=output,
                    clip_limits=args.clip_limits,
                )
            )
        except Exception as exc:  # Continue so one bad episode does not stop the batch.
            message = f"{type(exc).__name__}: {exc}"
            print(f"{prefix}: FAILED: {message}", file=sys.stderr)
            failures.append((episode_dir, message))
            continue

        observed = [
            side
            for side, valid in zip(
                summary["sides"], summary["side_is_observed"]
            )
            if valid
        ]
        print(
            f"{prefix}: exported {summary['shape'][0]} frames; "
            f"observed={observed} -> {output}"
        )
        exported += 1

    if args.dry_run:
        print(f"Dry run complete: {len(episodes)} episode(s).")
        return 0

    print(
        f"Complete: exported={exported}, skipped={skipped}, "
        f"failed={len(failures)}"
    )
    if failures:
        print("Failed episodes:", file=sys.stderr)
        for episode_dir, message in failures:
            print(f"  {episode_dir}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
