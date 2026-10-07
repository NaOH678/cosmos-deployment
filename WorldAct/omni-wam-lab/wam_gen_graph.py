"""Opt-in WAM complete GEN graphs; bounded cache and unchanged generic fallback."""

import json
import logging
import os
from pathlib import Path
import time
import torch


MAX_GRAPH_VARIANTS = 4


def graph_audit(model):
    """Return an in-memory snapshot; never write on the request hot path."""
    states = getattr(model, "_wam_gen_graph_states", {})
    return {
        "max_graph_variants": MAX_GRAPH_VARIANTS,
        "graph_count": len(states),
        "total_replays": sum(s["replays"] for s in states.values()),
        "graphs": [
            {
                "hidden_shape": list(s["hidden"].shape),
                "text_length": s["kv"][0][0].shape[1],
                "capture_ms": s["capture_ms"],
                "replays": s["replays"],
            }
            for s in states.values()
        ],
    }


def release_graphs(model):
    """Release graph-owned buffers before weight replacement or sleep."""
    states = getattr(model, "_wam_gen_graph_states", {})
    if states:
        torch.cuda.synchronize()
        for state in states.values():
            state["graph"].reset()
        states.clear()


def _write_audit(model):
    path = os.environ.get("WAM_FULL_GEN_AUDIT")
    if not path:
        return
    records = []
    for key, state in model._wam_gen_graph_states.items():
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
    try:
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
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not save WAM cold graph audit: %s", exc)



def run_full_gen_graph(model, prep, original):
    if (
        prep.ulysses_size != 1
        or prep.has_control
        or prep.has_sound
        or prep.hidden_gen.shape[0] != 1
    ):
        return original(prep)
    if (
        getattr(model, "_model_cpu_offload_enabled", False)
        or getattr(model, "mixed_precision_runtime", None) is not None
        or (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() != 1
        )
    ):
        return original(prep)
    if model.cached_kv is None or model.cached_freqs_gen is None:
        raise RuntimeError(
            "UND KV and per-branch RoPE must be prepared before GEN graph"
        )
    incoming_kv = model.cached_kv
    incoming_rope = model.cached_freqs_gen
    key = (
        tuple(prep.hidden_gen.shape),
        str(prep.hidden_gen.dtype),
        str(prep.hidden_gen.device),
        (
            prep.s_video,
            prep.s_action,
            prep.has_action,
            prep.use_multi_control_attention,
        ),
        tuple(k.shape for k, v in incoming_kv),
        tuple(x.shape for x in incoming_rope),
    )
    states = getattr(model, "_wam_gen_graph_states", None)
    if states is None:
        states = model._wam_gen_graph_states = {}
    state = states.get(key)
    if state is None and len(states) >= MAX_GRAPH_VARIANTS:
        # Never retain unbounded graph-private pools for arbitrary new shapes.
        return original(prep)
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
