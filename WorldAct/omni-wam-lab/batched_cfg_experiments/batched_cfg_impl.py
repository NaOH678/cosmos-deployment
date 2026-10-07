"""Isolated GEN batching; independent UND caches and per-branch mRoPE."""

import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F


def predict_pair(pipe, common, cond_ids, cond_mask, uncond_ids, uncond_mask, cache):
    model = pipe.transformer
    kwargs = dict(common)
    for key in (
        "hidden_states",
        "action_latents",
        "sound_latents",
        "control_latents",
        "noisy_frame_mask",
        "action_noisy_mask",
    ):
        if isinstance(kwargs.get(key), torch.Tensor):
            kwargs[key] = kwargs[key].to(pipe.dtype)
    prepared = []
    for index, (ids, mask) in enumerate(
        ((cond_ids, cond_mask), (uncond_ids, uncond_mask))
    ):
        model.cached_kv, model.cached_freqs_gen = cache.get(index, (None, None))
        prep = model._gen_preprocess(text_ids=ids, text_mask=mask, **kwargs)
        if prep.ulysses_size != 1 or prep.has_control or prep.has_sound:
            raise RuntimeError(
                "Only single-GPU video/action WAM is supported by this experiment"
            )
        cache[index] = (model.cached_kv, model.cached_freqs_gen)
        prepared.append(prep)
    if "batched_kv" not in cache:
        lengths = tuple(c[0][0][0].shape[1] for c in (cache[0], cache[1]))
        maximum = max(lengths)
        assert len(lengths) == 2 and all(0 < n <= maximum for n in lengths)
        cache["lengths"] = lengths
        cache["batched_kv"] = []
        for left, right in zip(cache[0][0], cache[1][0], strict=True):
            pair = []
            for field in (0, 1):
                pair.append(
                    torch.cat(
                        [
                            F.pad(
                                x[field], (0, 0, 0, 0, 0, maximum - x[field].shape[1])
                            )
                            for x in (left, right)
                        ],
                        dim=0,
                    )
                )
            cache["batched_kv"].append(tuple(pair))
        cache["rope"] = tuple(
            torch.cat([cache[0][1][i], cache[1][1][i]], dim=0) for i in (0, 1)
        )
        mask = torch.ones(
            (2, maximum + prepared[0].hidden_gen.shape[1]),
            dtype=torch.bool,
            device=prepared[0].hidden_gen.device,
        )
        for i, n in enumerate(lengths):
            mask[i, n:maximum] = False
        for layer in model.gen_layers:
            layer.cross_attention._cfg_text_lengths = lengths
            layer.cross_attention._cfg_attention_mode = os.environ.get(
                "WAM_CFG_ATTENTION", "split"
            )
            layer.cross_attention._cfg_attention_mask = mask
    hidden = torch.cat([p.hidden_gen for p in prepared], dim=0)
    cosine, sine = cache["rope"]
    for layer, (key, value) in zip(model.gen_layers, cache["batched_kv"], strict=True):
        hidden = layer(hidden, k_und=key, v_und=value, freqs_cos=cosine, freqs_sin=sine)
    hidden = model.norm_moe_gen(hidden)
    result = []
    for index, prep in enumerate(prepared):
        value = model._gen_postprocess(hidden[index : index + 1], prep)
        result.append(
            tuple(x.to(pipe.sampling_dtype) for x in value)
            if isinstance(value, tuple)
            else value.to(pipe.sampling_dtype)
        )
    cache["calls"] = cache.get("calls", 0) + 1
    if cache["calls"] == 4 and not getattr(pipe, "_batch_audit_written", False):
        pipe._batch_audit_written = True
        audit = {
            "gen_batch": hidden.shape[0],
            "gen_shape": list(hidden.shape),
            "gen_layers": len(model.gen_layers),
            "gen_step_calls": cache["calls"],
            "text_lengths": cache["lengths"],
            "rope_batch": cosine.shape[0],
            "attention_mode": os.environ.get("WAM_CFG_ATTENTION", "split"),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "padding_handling": "slice_to_real_length"
            if os.environ.get("WAM_CFG_ATTENTION", "split") == "split"
            else "explicit_boolean_key_mask",
        }
        Path(os.environ["WAM_BATCH_AUDIT"]).write_text(json.dumps(audit, indent=2))
        print("WAM_BATCH_CFG_AUDIT", json.dumps(audit), flush=True)
    dump_branches(result[0], result[1], cache["calls"])
    return tuple(result)


def dump_branches(cond, uncond, step):
    location = os.environ.get("WAM_SAVE_BRANCHES")
    if not location or step not in (1, 4):
        return
    import numpy as np

    target = Path(location)
    target.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for branch, value in (("cond", cond), ("uncond", uncond)):
        values = value if isinstance(value, tuple) else (value,)
        for i, tensor in enumerate(values):
            arrays[f"{branch}_{i}"] = tensor.detach().float().cpu().numpy()
    np.savez(target / f"step_{step}.npz", **arrays)
