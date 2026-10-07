"""Summarize full GEN graph timing, exact outputs, and actual launch evidence."""

import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
base = root / "results/baseline_corpus"
graph = root / "results/graph_corpus"
b = json.loads((base / "report.json").read_text())
g = json.loads((graph / "report.json").read_text())
arrays = {
    name: [np.load(directory / f"actions_{i:03d}.npy") for i in range(25)]
    for name, directory in [("baseline", base), ("graph", graph)]
}
trace = next((graph / "traces").glob("*/*.json.gz"))
events = json.load(gzip.open(trace))["traceEvents"]
launches = sum(e.get("name") in ("cudaGraphLaunch", "cuGraphLaunch") for e in events)
audit = json.loads((graph / "full_gen_audit.json").read_text())
summary = {
    "scope": "Prepared five-real-observation corpus -> CPU raw actions; excludes HTTP and production recording; 5 warmups +20 measured each, extra2 graph profile requests excluded",
    "conditions": {
        "checkpoint": "4w EMA",
        "quantization": "FP8 linear, BF16 cuDNN attention",
        "sampler": "4 UniPC steps CFG3 shift5",
        "compile": "unchanged default dynamic regional; inner compiler CUDA Graphs disabled",
        "production": "native port18005 resident idle; not stopped; no robot controls",
    },
    "baseline": {
        k: b[k]
        for k in (
            "p50_ms",
            "p95_ms",
            "max_memory_allocated_bytes",
            "max_memory_reserved_bytes",
        )
    },
    "full_gen_graph": {
        k: g[k]
        for k in (
            "p50_ms",
            "p95_ms",
            "max_memory_allocated_bytes",
            "max_memory_reserved_bytes",
        )
    },
    "p50_reduction_ms": b["p50_ms"] - g["p50_ms"],
    "p50_reduction_percent": 100 * (b["p50_ms"] - g["p50_ms"]) / b["p50_ms"],
    "cold_first_request_ms": {
        "baseline": b["samples"][0]["end_to_end_ms"],
        "graph": g["samples"][0]["end_to_end_ms"],
    },
    "capture_cost_including_three_warmups_ms": {
        str(x["text_length"]): x["warmup_capture_ms"] for x in audit["graphs"]
    },
    "actual_profile": {
        "trace": str(trace),
        "requests": 2,
        "cudaGraphLaunch_calls": launches,
        "launches_per_request": launches / 2,
        "expected": 8,
        "inner_inductor_cudagraphs": g["inductor_inner_cudagraphs"],
    },
    "output_validation": {
        "all_25_pairs_bitwise_equal": all(
            np.array_equal(a, c)
            for a, c in zip(arrays["baseline"], arrays["graph"], strict=True)
        ),
        "all_actions_max_abs_diff": max(
            float(abs(a - c).max())
            for a, c in zip(arrays["baseline"], arrays["graph"], strict=True)
        ),
        "graph_same_input_after_four_other_inputs_max_abs": max(
            float(abs(arrays["graph"][i] - arrays["graph"][i % 5]).max())
            for i in range(5, 25)
        ),
        "graph_different_input_0_vs_1_max_abs": float(
            abs(arrays["graph"][0] - arrays["graph"][1]).max()
        ),
    },
    "capture_design": "Two stable complete 28-layer GEN+norm graphs, lengths140/19; each call updates hidden, every KV tensor and RoPE; original UND, preprocess, postprocess, CFG and sampler unchanged",
    "limitations": [
        "One serial then graph experiment order; modest speed benefit needs matched HTTP confirmation.",
        "Only fixed WAM shapes and five observations tested; exact equality on these inputs is not a general floating-point equivalence proof.",
        "The fixed old-fixture smoke was418-422ms; do not quote that instead of new-corpus formal432.48ms.",
        "Cold capture reported includes3 GPU warmups and graph construction, not only cudaStreamBeginCapture interval.",
        "Prototype graph cache has no eviction and excludes concurrent/model-weight hot-swapping; not yet production-ready.",
        "Memory maxima include initialization/warmup and are not standalone graph private-pool sizes.",
    ],
    "artifacts": {
        "baseline_report": str(base / "report.json"),
        "graph_report": str(graph / "report.json"),
        "audit": str(graph / "full_gen_audit.json"),
    },
    "source_hashes": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            root / "full_gen_graph_impl.py",
            root / "benchmark_full_gen.py",
            root / "source/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py",
        ]
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(
    json.dumps(
        {
            k: summary[k]
            for k in (
                "baseline",
                "full_gen_graph",
                "p50_reduction_ms",
                "actual_profile",
                "output_validation",
            )
        },
        indent=2,
    )
)
