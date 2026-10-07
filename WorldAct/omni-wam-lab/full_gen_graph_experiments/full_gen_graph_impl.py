"""One CUDA Graph per complete GEN stack and text-length specialization."""

import json
import os
from pathlib import Path
import time
import torch


def _write_audit(model):
    path = os.environ.get("WAM_FULL_GEN_AUDIT")
    if not path:
        return
    records = []
    for key, state in model._full_gen_graphs.items():
        records.append(
            {
                "key": repr(key),
                "hidden_shape": list(state["hidden"].shape),
                "text_length": state["kv"][0][0].shape[1],
                "gen_layers": len(state["kv"]),
                "warmup_capture_ms": state["capture_ms"],
                "replays_at_capture_audit_snapshot": state["replays"],
                "output_shape": list(state["output"].shape),
            }
        )
    Path(path).write_text(
        json.dumps(
            {
                "graphs": records,
                "memory_allocated_bytes": torch.cuda.memory_allocated(),
                "memory_reserved_bytes": torch.cuda.memory_reserved(),
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            indent=2,
        )
    )


def run_full_gen_graph(model, prep, original):
    if (
        prep.ulysses_size != 1
        or prep.has_control
        or prep.has_sound
        or prep.hidden_gen.shape[0] != 1
    ):
        raise RuntimeError(
            "Full GEN graph experiment requires single-GPU B1 video/action"
        )
    if model.cached_kv is None or model.cached_freqs_gen is None:
        raise RuntimeError(
            "UND KV and per-branch RoPE must be prepared before GEN graph"
        )
    incoming_kv = model.cached_kv
    incoming_rope = model.cached_freqs_gen
    key = (
        tuple(prep.hidden_gen.shape),
        str(prep.hidden_gen.dtype),
        tuple(k.shape for k, v in incoming_kv),
        tuple(x.shape for x in incoming_rope),
    )
    states = getattr(model, "_full_gen_graphs", None)
    if states is None:
        states = model._full_gen_graphs = {}
    state = states.get(key)
    if state is None:
        started = time.perf_counter()
        state = {
            "hidden": prep.hidden_gen.clone(),
            "kv": [(k.clone(), v.clone()) for k, v in incoming_kv],
            "rope": tuple(x.clone() for x in incoming_rope),
            "replays": 0,
        }
        static_prep = prep._replace(hidden_gen=state["hidden"])
        state["prep"] = static_prep
        state["graph"] = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        model.cached_kv, model.cached_freqs_gen = state["kv"], state["rope"]
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    scratch = original(static_prep)
                del scratch
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            with torch.cuda.graph(state["graph"], stream=stream):
                state["output"] = original(static_prep)
        finally:
            model.cached_kv, model.cached_freqs_gen = incoming_kv, incoming_rope
        torch.cuda.synchronize()
        state["capture_ms"] = (time.perf_counter() - started) * 1000
        states[key] = state
        print(
            "WAM_FULL_GEN_CAPTURE",
            state["kv"][0][0].shape[1],
            state["capture_ms"],
            flush=True,
        )
        _write_audit(model)
    else:
        state["hidden"].copy_(prep.hidden_gen)
        for (target_k, target_v), (source_k, source_v) in zip(
            state["kv"], incoming_kv, strict=True
        ):
            target_k.copy_(source_k)
            target_v.copy_(source_v)
        for target, source in zip(state["rope"], incoming_rope, strict=True):
            target.copy_(source)
    state["graph"].replay()
    state["replays"] += 1
    # Graph-owned output is consumed immediately by original postprocess, which
    # allocates independent video/action predictions before this graph is replayed.
    return state["output"]
