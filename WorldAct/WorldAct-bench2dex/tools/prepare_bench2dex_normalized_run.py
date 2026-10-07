"""Snapshot and pin normalization stats before launching a distinct training run."""

import argparse
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from cosmos_framework.utils.bench2dex_normalization import METHOD, load_bench2dex_normalizer


def prepare(stats_path, output_root):
    _, stats, digest = load_bench2dex_normalizer(stats_path)
    root = Path(output_root).resolve()
    config = root / "cosmos3_action/action_sft/action_policy_bench2dex_edge/config.yaml"
    if config.exists():
        old = OmegaConf.select(
            OmegaConf.load(config), "dataloader_train.dataloader.datasets.bench2dex.dataset.action_stats_sha256"
        )
        if old != digest:
            raise ValueError("Use a new OUTPUT_ROOT: existing run has a different/raw action space")
    elif stats["method"] != METHOD:
        raise ValueError("New runs require action-only q01/q99 v2 stats; legacy stats are for existing runs only")
    snapshot = root / "action_stats.json"
    if snapshot.exists():
        load_bench2dex_normalizer(snapshot, digest)
    else:
        if any(root.glob("**/checkpoints/**/.metadata")):
            raise ValueError("Existing checkpoints without a stats snapshot: use a new OUTPUT_ROOT")
        root.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids silently replacing a concurrent run's stats.
        with snapshot.open("xb") as dst, Path(stats_path).open("rb") as src:
            shutil.copyfileobj(src, dst)
        load_bench2dex_normalizer(snapshot, digest)
    return snapshot, digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    snapshot, digest = prepare(args.stats, args.output_root)
    print(snapshot)
    print(digest)
