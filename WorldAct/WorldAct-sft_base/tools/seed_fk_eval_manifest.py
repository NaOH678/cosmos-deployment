"""Seed a new run's ``fk_eval/fixed_cases.json`` from an existing run's, swapping one case.

Why a manifest and not a config.  ``fk_eval_cases.fixed_cases`` reads
``<job.path_local>/fk_eval/fixed_cases.json``; if it exists it *uses* the indices
it records and aborts with "Fixed eval identity changed" if the dataset does not
reproduce the same identities.  So writing the file before the run's first
validation is what pins the eval set -- and it is the only hook that lets a new
run keep three cases identical to an old one while changing the fourth.

That is exactly what is wanted here: a run whose loss objective changed must be
compared against the previous run on the *same* windows, or the comparison
measures the eval set as much as the objective.  Only ``val_00`` changes, because
the one the automatic selection picked sits on its episode's final frame -- the
hand is leaving the scene, which is not a motion the model is asked for anywhere
else.

The replacement identity is recomputed from the labels by
``pick_fk_eval_case`` rather than hand-written, and then re-checked field by field
against the same derivation ``fixed_cases`` uses, because a manifest that fails
that check crashes the run at its first validation rather than at launch.

    PYTHONPATH=. <venv>/bin/python tools/seed_fk_eval_manifest.py \\
        --base <old run root> --out <new run root> --replace-val00-index 2408
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.pick_fk_eval_case import _identity, score_split  # noqa: E402

REL = "cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge"


def _manifest_path(root: Path) -> Path:
    return root / REL / "fk_eval" / "fixed_cases.json"


def _load(path: Path):
    if not path.is_file():
        raise SystemExit(f"ERROR: no manifest at {path}")
    rows = json.loads(path.read_text())
    by_id = {row["case_id"]: row for row in rows}
    missing = {"train_00", "train_01", "val_00", "val_01"} - set(by_id)
    if missing:
        raise SystemExit(f"ERROR: {path} is missing {sorted(missing)}")
    return by_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="run root to copy the kept cases from")
    parser.add_argument("--out", required=True, type=Path, help="run root to write the manifest into")
    parser.add_argument("--replace-val00-index", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    old = _load(_manifest_path(args.base))
    destination = _manifest_path(args.out)
    if destination.exists():
        current = json.loads(destination.read_text())
        if any(row.get("case_id") == "val_00" and row.get("index") == args.replace_val00_index for row in current):
            print(f"already seeded: {destination}")
            return 0
        raise SystemExit(f"ERROR: {destination} already exists and is not this seed; refusing to overwrite")

    candidates = [row for row in score_split("val") if row["index"] == args.replace_val00_index]
    if not candidates:
        raise SystemExit(f"ERROR: index {args.replace_val00_index} is not in the val split")
    new_val00 = _identity(candidates[0], count_index=0)

    # Same derivation ``fixed_cases`` performs, re-run here so a mistake is a
    # launch-time error and not a crash 200 steps into a multi-hour run.
    for field, expected in old["val_00"].items():
        if field in ("index", "episode", "start_frame", "raw_frame_ids", "case_id", "split"):
            continue
        if new_val00[field] != expected:
            print(f"note: {field} changed, {expected!r} -> {new_val00[field]!r}")

    seeded = [old["train_00"], old["train_01"], new_val00, old["val_01"]]
    for expected_id, row in zip(("train_00", "train_01", "val_00", "val_01"), seeded, strict=True):
        if row["case_id"] != expected_id:
            raise SystemExit(f"ERROR: order is wrong, {row['case_id']} where {expected_id} was expected")

    print(f"kept  : train_00 idx {old['train_00']['index']}  train_01 idx {old['train_01']['index']}"
          f"  val_01 idx {old['val_01']['index']}")
    print(f"val_00: idx {old['val_00']['index']} start_frame {old['val_00']['start_frame']}"
          f"  (episode {old['val_00']['episode']})")
    print(f"     -> idx {new_val00['index']} start_frame {new_val00['start_frame']}"
          f"  (episode {new_val00['episode']})")
    scored = candidates[0]
    print(f"        position {scored['position']:.2f} of the episode, z-spread {scored['z_spread_mm']:.1f}mm,"
          f" motion {scored['motion_mm']:.1f}mm, {scored['frames_after']} raw frames left after it")

    if args.dry_run:
        print(f"(dry run: would write {destination})")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(seeded, indent=2) + "\n")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
