"""Attribute short CUDA kernels to Python scopes and autograd forward sites."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def analyze(path):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    host = [e for e in events if e.get("ph") == "X" and e.get("cat") in {"python_function", "cpu_op", "cuda_runtime"}]
    host.sort(key=lambda e: (e["tid"], e["ts"], -e.get("dur", 0)))
    stack, thread = [], None
    scopes, forward, launches = {}, {}, {}
    for e in host:
        if e["tid"] != thread:
            stack, thread = [], e["tid"]
        end = e["ts"] + e.get("dur", 0)
        while stack and stack[-1]["ts"] + stack[-1].get("dur", 0) < end:
            stack.pop()
        args = e.get("args", {})
        if e["cat"] == "cpu_op":
            repo = [p["name"] for p in stack if "cosmos_framework/" in p["name"]]
            scope = repo[-1] if repo else "unattributed"
            bw = [p for p in [*stack, e] if p["name"].startswith("autograd::engine::evaluate_function:")]
            seq = bw[-1].get("args", {}).get("Sequence number") if bw else None
            scopes[args.get("External id")] = (scope, seq, e["name"])
            if not bw and "Sequence number" in args and repo:
                forward.setdefault(args["Sequence number"], scope)
        if e["cat"] == "cuda_runtime" and "LaunchKernel" in e["name"]:
            launches[args.get("correlation")] = e.get("dur", 0)
        stack.append(e)
    groups = defaultdict(lambda: dict(kernels=0, gpu_ms=0.0, small_kernels=0, small_gpu_ms=0.0, launch_cpu_ms=0.0))
    details = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for e in events:
        if e.get("cat") != "kernel":
            continue
        args = e.get("args", {})
        scope, seq, op = scopes.get(args.get("External id"), ("unattributed", None, "unknown"))
        if seq is not None:
            scope = "backward: " + forward.get(seq, scope)
        key = scope
        g = groups[key]
        g["kernels"] += 1
        g["gpu_ms"] += e["dur"] / 1000
        g["launch_cpu_ms"] += launches.get(args.get("correlation"), 0) / 1000
        if e["dur"] <= 10:
            g["small_kernels"] += 1
            g["small_gpu_ms"] += e["dur"] / 1000
            d = details[key][op + " | " + e["name"]]
            d[0] += 1
            d[1] += e["dur"] / 1000
    return dict(
        trace=str(path),
        threshold_us=10,
        note="Kernel External id links CPU op; backward Sequence number links forward site. Inclusive launch CPU time is not pure overhead.",
        groups=[
            dict(scope=k, **v, top_small_kernels=sorted(details[k].items(), key=lambda x: x[1][0], reverse=True)[:6])
            for k, v in sorted(groups.items(), key=lambda x: x[1]["small_kernels"], reverse=True)
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.trace)
    args.output.write_text(json.dumps(result, indent=2))
    for g in result["groups"][:25]:
        print(json.dumps({k: v for k, v in g.items() if k != "top_small_kernels"}))
