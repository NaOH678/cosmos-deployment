"""Attribute scalar readbacks and long GPU-idle gaps to Python call sites."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("trace", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
with gzip.open(a.trace, "rt") as f:
    events = json.load(f)["traceEvents"]
host = sorted(
    (e for e in events if e.get("ph") == "X" and e.get("cat") in ("python_function", "cpu_op", "cuda_runtime")),
    key=lambda e: (e["tid"], e["ts"], -e.get("dur", 0)),
)
stack, tid = [], None
groups = defaultdict(lambda: [0, 0.0])
for e in host:
    if e["tid"] != tid:
        stack, tid = [], e["tid"]
    end = e["ts"] + e.get("dur", 0)
    while stack and stack[-1]["ts"] + stack[-1].get("dur", 0) < end:
        stack.pop()
    if e["name"] == "aten::item":
        repo = [x["name"] for x in stack if "cosmos_framework/" in x["name"]]
        key = repo[-1] if repo else "unattributed"
        groups[key][0] += 1
        groups[key][1] += e["dur"] / 1000
    stack.append(e)
result = [dict(scope=k, count=v[0], inclusive_ms=v[1]) for k, v in sorted(groups.items(), key=lambda x: -x[1][1])]
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result[:25], indent=2))
