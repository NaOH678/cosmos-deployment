"""Summarize a PyTorch Chrome trace without requiring a browser."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def summarize(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    events = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]
    gpu = [e for e in events if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}]
    spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    if not spans:
        raise ValueError("No CUDA activities in trace")
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = sorted(
        [(a[1], b[0]) for a, b in zip(merged, merged[1:]) if b[0] - a[1] >= 1000],
        key=lambda x: x[1] - x[0],
        reverse=True,
    )
    tables = {}
    for category in ("kernel", "cuda_runtime", "cpu_op", "gpu_memcpy"):
        totals = defaultdict(lambda: [0, 0.0])
        for e in events:
            if e.get("cat") == category:
                totals[e["name"]][0] += 1
                totals[e["name"]][1] += e["dur"] / 1000
        tables[category] = [
            {"name": name, "count": value[0], "inclusive_ms": value[1]}
            for name, value in sorted(totals.items(), key=lambda x: x[1][1], reverse=True)[:30]
        ]
    gap_details = []
    for start, end in gaps[:20]:
        overlap = []
        for e in events:
            if e.get("cat") not in {"cpu_op", "cuda_runtime", "user_annotation"}:
                continue
            duration = min(end, e["ts"] + e["dur"]) - max(start, e["ts"])
            if duration > 0:
                overlap.append((duration, e))
        gap_details.append(
            dict(
                start_us=start,
                duration_ms=(end - start) / 1000,
                overlapping_cpu=[
                    dict(name=e["name"], cat=e["cat"], tid=e["tid"], overlap_ms=d / 1000)
                    for d, e in sorted(overlap, key=lambda x: x[0], reverse=True)[:12]
                ],
            )
        )
    total = merged[-1][1] - merged[0][0]
    busy = sum(b - a for a, b in merged)
    return dict(
        trace=str(path),
        gpu_span_ms=total / 1000,
        gpu_activity_union_ms=busy / 1000,
        gpu_activity_fraction=busy / total,
        gaps_over_1ms=len(gaps),
        gaps_over_1ms_total_ms=sum(b - a for a, b in gaps) / 1000,
        tables=tables,
        largest_gaps=gap_details,
        note="CPU durations are inclusive; GPU activity includes NCCL wait kernels, not just useful compute.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in {"tables", "largest_gaps"}}, indent=2))
    for category, rows in result["tables"].items():
        print(category, json.dumps(rows[:8]))
