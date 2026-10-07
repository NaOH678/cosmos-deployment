import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
names = ["serial_corpus", "split_corpus", "masked_clean"]
summary = {
    "conditions": {
        "model": "4w EMA",
        "quantization": "vLLM CUTLASS FP8 weights; activation per-token",
        "attention": "strict CUDNN_ATTN BF16",
        "sampler": "4 UniPC steps, CFG3, shift5",
        "input": "five real observation packets rotating;5 warmup20 measured",
        "scope": "packet->CPU raw actions, no HTTP or production recording",
        "production_service": "idle native port18005 retained",
        "official_reference": "https://raw.githubusercontent.com/NVIDIA/cosmos-framework/main/cosmos_framework/model/generator/omni_mot_model.py",
    },
    "runs": {},
    "output_comparison": {},
}
ref = np.stack(
    [np.load(root / f"results/serial_corpus/actions_{i:03d}.npy") for i in range(5, 25)]
).astype(np.float64)
for name in names:
    path = root / "results" / name
    report = json.loads((path / "report.json").read_text())
    d = {k: report[k] for k in ["p50_ms", "p95_ms"]}
    d["report"] = str(path / "report.json")
    d["peak_memory_allocated_bytes"] = report.get("max_memory_allocated_bytes")
    d["peak_memory_reserved_bytes"] = report.get("max_memory_reserved_bytes")
    if (path / "batch_audit.json").exists():
        d["batch_evidence"] = json.loads((path / "batch_audit.json").read_text())
    arrays = [np.load(path / f"actions_{i:03d}.npy") for i in range(25)]
    d["same_input_repeat_after_four_other_inputs_max_abs"] = max(
        float(np.abs(arrays[i] - arrays[i % 5]).max()) for i in range(5, 25)
    )
    d["different_input_first_vs_second_max_abs"] = float(
        np.abs(arrays[0] - arrays[1]).max()
    )
    summary["runs"][name] = d
    if name == "serial_corpus":
        continue
    arr = np.stack(arrays[5:25]).astype(np.float64)
    diff = arr - ref
    pos = np.linalg.norm(diff[:, :, :3], axis=-1)
    rq = ref[:, :, 3:7]
    cq = arr[:, :, 3:7]
    cosine = abs(
        np.sum(rq * cq, axis=-1)
        / (np.linalg.norm(rq, axis=-1) * np.linalg.norm(cq, axis=-1))
    )
    angle = np.degrees(2 * np.arccos(np.clip(cosine, 0, 1)))
    hand = abs(diff[:, :, 7:])
    out = {
        "position_m_mean": float(pos.mean()),
        "position_m_max": float(pos.max()),
        "orientation_deg_mean": float(angle.mean()),
        "orientation_deg_max": float(angle.max()),
        "hand_native_units_mae": float(hand.mean()),
        "hand_native_units_max": float(hand.max()),
        "all_action_max_abs": float(abs(diff).max()),
        "branch_predictions": {},
    }
    for step in (1, 4):
        a = np.load(root / f"results/serial_corpus/branches/step_{step}.npz")
        b = np.load(path / f"branches/step_{step}.npz")
        out["branch_predictions"][str(step)] = {
            k: {
                "max_abs": float(abs(a[k] - b[k]).max()),
                "mean_abs": float(abs(a[k] - b[k]).mean()),
                "reference_shape": list(a[k].shape),
            }
            for k in a.files
        }
    summary["output_comparison"][name] = out
summary["padding_gpu_test"] = json.loads((root / "padding_test.json").read_text())
summary["conclusion"] = (
    "No meaningful acceleration: split-attention median saves ~2.5ms but P95 worsens; full masked B2 is slower. Keep current production serial CFG."
)
summary["limitations"] = [
    "Numerical differences already appear at first denoising step; root cause not fully localized. Do not automatically attribute to FP8 scale/rounding.",
    "GPU padding sentinel proves padding exclusion for tested kernel case, not whole-model equivalence.",
    "All output differences are offline prediction differences, not actual task error or measured success rates.",
    "Serial peak allocator statistics were not captured in this run; compare batch candidates absolute peak only.",
    "Standalone experiment supports WAM B1 per branch, TP/SP/CFG parallel1 and no cache-dit; not production-ready.",
    "masked_corpus older run had per-request audit I/O; final comparison exclusively uses masked_clean with audit only first warmup.",
]
summary["source_hashes"] = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [
        root / "batched_cfg_impl.py",
        root / "benchmark_cfg.py",
        root / "source/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py",
        root / "source/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py",
    ]
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ["runs", "padding_gpu_test"]}, indent=2))
