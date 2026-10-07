import json
import os
from pathlib import Path

lab = Path(__file__).resolve().parents[1]
source = Path(json.loads((lab / "upstream.json").read_text())["source"])
out = Path(__file__).resolve().parent / "source"
for root, dirs, files in os.walk(source):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
    dest = out / Path(root).relative_to(source)
    dest.mkdir(parents=True, exist_ok=True)
    for file in files:
        target = dest / file
        if not target.exists():
            target.symlink_to(Path(root) / file)
rel = Path("vllm_omni/diffusion/models/cosmos3")
pipeline = (source / rel / "pipeline_cosmos3.py").read_text()
new = """            elif do_cfg and os.environ.get("WAM_BATCH_CFG") == "1":
                from batched_cfg_impl import predict_pair
                if kv_state is not None or self._cache_requires_paired_cfg():
                    raise RuntimeError("Batched CFG experiment excludes session/cache-dit")
                pair_cache = {}
                for step_index, t in enumerate(self.progress_bar(timesteps)):
                    self._set_denoise_step_metadata(step_index, timesteps, step_scheduler)
                    self._set_mixed_precision_step(step_index, len(timesteps))
                    common = dict(hidden_states=latents, timestep=t.unsqueeze(0),
                                  action_latents=action_latents, sound_latents=sound_latents,
                                  **shared_kwargs)
                    noise_cond, noise_uncond = predict_pair(
                        self, common, cond_ids, cond_mask, uncond_ids, uncond_mask, pair_cache)
                    noise_pred = self.combine_cfg_noise(
                        noise_cond, noise_uncond,
                        guidance_scale if _cfg_active_at(t) else 1.0, cfg_normalize=False)
                    _assign_step_out(_step(noise_pred, t, latents, action_latents, sound_latents))

"""
assert pipeline.count("            elif do_cfg:\n") == 1
pipeline = pipeline.replace(
    "            elif do_cfg:\n", new + "            elif do_cfg:\n"
)
pipeline = pipeline.replace(
    "                        step_scale = guidance_scale if cfg_active else 1.0",
    "                        from batched_cfg_impl import dump_branches\n                        dump_branches(noise_cond, noise_uncond, step_index+1)\n                        step_scale = guidance_scale if cfg_active else 1.0",
)
if "\nimport os\n" not in pipeline:
    pipeline = pipeline.replace(
        "from __future__ import annotations",
        "from __future__ import annotations\nimport os",
    )
transform = (source / rel / "transformer_cosmos3.py").read_text()
old = """        B, S_gen = q.shape[:2]
        k_all = torch.cat([k_und, k], dim=1)"""
new = """        B, S_gen = q.shape[:2]
        lengths = getattr(self, "_cfg_text_lengths", None)
        if B == 2 and lengths is not None:
            if getattr(self, "_cfg_attention_mode", "split") == "split":
                outputs = []
                for i, length in enumerate(lengths):
                    ki = torch.cat([k_und[i:i+1, :length], k[i:i+1]], dim=1)
                    vi = torch.cat([v_und[i:i+1, :length], v[i:i+1]], dim=1)
                    outputs.append(self.attn(q[i:i+1], ki, vi))
                return torch.cat(outputs, dim=0).reshape(B, S_gen, -1)
            k_all = torch.cat([k_und, k], dim=1)
            v_all = torch.cat([v_und, v], dim=1)
            return self.attn(q, k_all, v_all, AttentionMetadata(attn_mask=self._cfg_attention_mask)).reshape(B,S_gen,-1)
        k_all = torch.cat([k_und, k], dim=1)"""
assert transform.count(old) == 1
transform = transform.replace(old, new)
for name, content in [
    ("pipeline_cosmos3.py", pipeline),
    ("transformer_cosmos3.py", transform),
]:
    p = out / rel / name
    if p.is_symlink():
        p.unlink()
    p.write_text(content)
print(out)
