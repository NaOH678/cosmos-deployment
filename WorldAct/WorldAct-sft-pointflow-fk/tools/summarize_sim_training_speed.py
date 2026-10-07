"""Summarize a fixed training step range from rank-zero logs and offline W&B."""

import argparse
import json
import re
import statistics
from pathlib import Path

from wandb.proto.wandb_internal_pb2 import Record
from wandb.sdk.internal.datastore import DataStore


def stats(values):
    return (
        dict(
            n=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            minimum=min(values),
            maximum=max(values),
        )
        if values
        else None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first", type=int, default=6)
    parser.add_argument("--last", type=int, default=26)
    args = parser.parse_args()
    log = (args.run / "logs/action_policy_sim_pointfk_edge_sft.log").read_text()
    steps = {int(i): float(t) for i, t in re.findall(r"\[RANK 0\] Iteration (\d+):[^\n]*Time: ([\d.]+)s", log)}
    history = {}
    for path in args.run.rglob("*.wandb"):
        store = DataStore()
        store._fname = str(path)
        store._fp = path.open("rb")
        store._index = 0
        store._size_bytes = path.stat().st_size
        store._opened_for_scan = True
        store._read_header()
        try:
            while (raw := store.scan_data()) is not None:
                record = Record()
                record.ParseFromString(raw)
                if not record.HasField("history"):
                    continue
                row = {
                    item.key or "/".join(item.nested_key): json.loads(item.value_json) for item in record.history.item
                }
                if "iteration" in row:
                    history.setdefault(int(row["iteration"]), {}).update(row)
        finally:
            store._fp.close()
    selected = range(args.first, args.last + 1)
    result = dict(
        run=str(args.run),
        step_range=[args.first, args.last],
        wall_seconds=stats([steps[i] for i in selected if i in steps]),
        timers={
            key: stats([history[i][key] for i in selected if key in history.get(i, {})])
            for key in ("timer/dataloader_train", "timer/forward", "timer/backward", "timer/optimizer_step")
        },
        steps={
            i: dict(
                wall_seconds=steps.get(i), **{k: v for k, v in history.get(i, {}).items() if k.startswith("timer/")}
            )
            for i in selected
        },
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
