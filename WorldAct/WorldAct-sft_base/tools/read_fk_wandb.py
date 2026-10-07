#!/usr/bin/env python3
"""Read a wandb offline run without a wandb server.

``wandb_mode = "offline"`` writes one binary ``run-XXXXXXXX.wandb`` that nothing
in the repo can open, so the metrics that only exist there -- the per-sigma loss
bins, the per-layer weight and gradient norms, the eval rat ios of every case --
are effectively invisible. This reads that file directly.

    # what is in it
    python tools/read_fk_wandb.py --run <wandb/offline-run-.../> --list

    # watch the FK numbers over training
    python tools/read_fk_wandb.py --run <dir> --keys train/fk_loss train/fk_ade_mm fk/val_00/ratio_to_zero

    # the per-sigma bins, which is where a sampler problem shows first
    python tools/read_fk_wandb.py --run <dir> --prefix "train@2_detail/fk_loss_sigma"

    # everything under a prefix, as CSV
    python tools/read_fk_wandb.py --run <dir> --prefix stats/grad_norm --csv > grad.csv

``--every N`` thins the output; the raw history is one row per logging step and
there are thousands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_run(path: Path) -> Path:
    """Accept the wandb/ dir, an offline-run dir, or the .wandb file itself."""
    if path.is_file() and path.suffix == ".wandb":
        return path
    if path.is_dir():
        direct = sorted(path.glob("*.wandb"))
        if direct:
            return direct[-1]
        nested = sorted(path.glob("offline-run-*/*.wandb"))
        if nested:
            return nested[-1]
    raise SystemExit(f"no .wandb file under {path}")


def read_history(run_file: Path) -> list[dict]:
    """Every history record, in order. Values are decoded from their JSON strings."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    store = datastore.DataStore()
    store.open_for_scan(str(run_file))
    rows = []
    while True:
        try:
            data = store.scan_data()
        except Exception:
            break
        if data is None:
            break
        record = pb.Record()
        try:
            record.ParseFromString(data)
        except Exception:
            continue
        if record.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in record.history.item:
            key = item.nested_key
            if not isinstance(key, str):  # a repeated field on some records
                key = "|".join(key)
            try:
                row[key] = json.loads(item.value_json)
            except (json.JSONDecodeError, ValueError):
                row[key] = item.value_json
        if row:
            rows.append(row)
    rows.sort(key=lambda r: r.get("_step", 0))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="wandb/ dir, offline-run dir, or .wandb file")
    parser.add_argument("--list", action="store_true", help="print every metric key and exit")
    parser.add_argument("--keys", nargs="*", default=[], help="exact keys to tabulate")
    parser.add_argument("--prefix", default=None, help="every key starting with this")
    parser.add_argument("--every", type=int, default=1, help="print one row in N (default 1)")
    parser.add_argument("--csv", action="store_true", help="emit CSV instead of a table")
    args = parser.parse_args()

    run_file = find_run(args.run)
    rows = read_history(run_file)
    if not rows:
        raise SystemExit(f"{run_file} has no history records")
    all_keys = {k for row in rows for k in row}
    print(f"# {run_file}\n# {len(rows)} history rows, {len(all_keys)} keys, "
          f"steps {rows[0].get('_step')}..{rows[-1].get('_step')}", file=sys.stderr)

    if args.list:
        for key in sorted(all_keys):
            print(key)
        return

    keys = list(args.keys)
    if args.prefix:
        keys += sorted(k for k in all_keys if k.startswith(args.prefix))
    # '_step' plus whatever was asked for, in the order asked, minus duplicates.
    columns = ["_step"] + [k for i, k in enumerate(keys) if k not in keys[:i]]
    missing = [k for k in columns if k not in all_keys]
    if missing:
        raise SystemExit(f"not in this run: {missing}\n(use --list to see what is)")

    # Two loggers write into this one file at different intervals (the main one at
    # logging_iter, the detail one at twice that), so most rows carry only some of
    # the requested keys. Dropping a row only when *every* requested column is
    # absent is what makes the table readable: thinning by row index instead lands
    # on the detail rows and shows blanks for everything asked for.
    rows = [row for row in rows if any(column in row for column in columns[1:])]

    separator = "," if args.csv else "  "
    print(separator.join(columns))
    for index, row in enumerate(rows):
        if index % args.every:
            continue
        cells = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            elif value is None:
                cells.append("")
            else:
                cells.append(str(value))
        print(separator.join(cells))


if __name__ == "__main__":
    main()
