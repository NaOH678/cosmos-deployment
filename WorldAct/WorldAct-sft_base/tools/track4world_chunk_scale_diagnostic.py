"""Recover the final Track4World geometry chunk's Metric calibration ratio.

Original infer_pair normalizes each chunk by its point norm and restores the last
chunk's norm to all frames. Keep that legacy behavior and frozen point support;
change only Metric focal conversion and final XYZ backprojection. This is not a
new tracking run. The cached output focal is validated against the rerun.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict
from depth_anything_3.cfg import create_object
from omegaconf import OmegaConf
from safetensors.torch import load_file

EP = "episode_0013_20260731_133649"
ROOT = Path("/data/shichaojian")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
torch.set_num_threads(8)
torch.cuda.set_per_process_memory_fraction(0.22 if "--low-memory" in sys.argv else 0.80)
ckpt = ROOT / "checkpoints/DA3NESTED-GIANT-LARGE-1.1"
cfg = json.loads((ckpt / "config.json").read_text())["config"]
cfg["anyview"].pop("gs_head", None)
cfg["anyview"].pop("gs_adapter", None)
print("Loading nested pretrained model + Track4World AnyView weights", flush=True)
model = create_object(OmegaConf.create(cfg))
state = {
    k.removeprefix("model."): v
    for k, v in load_file(str(ckpt / "model.safetensors")).items()
    if not k.startswith(("model.da3.gs_head.", "model.da3.gs_adapter."))
}
# Restore safetensors aliases for the shared auxiliary LayerNorm.
aliases = {}
for name, p in model.named_parameters(remove_duplicate=False):
    aliases.setdefault(id(p), []).append(name)
for names in aliases.values():
    src = next((n for n in names if n in state), None)
    if src:
        for n in names:
            state.setdefault(n, state[src])
model.load_state_dict(state, strict=True)
del state
state = torch.load(ROOT / "checkpoints/track4world_da3.pth", map_location="cpu", weights_only=True)
remapped = {k.replace("backbone.model.", "da3.", 1): v for k, v in state.items() if k.startswith("backbone.model.")}
missing, unexpected = model.load_state_dict(remapped, strict=False)
assert all(k.startswith(("da3.gs_head.", "da3.gs_adapter.")) for k in unexpected), unexpected
assert all(k.startswith("da3_metric.") for k in missing), missing
print("Track4World AnyView loaded with every parameter accounted for", flush=True)
del state, remapped
model.eval().cuda()
if "--low-memory" in sys.argv:
    from track4world_memory import offload_retained_features

    offload_retained_features(model.da3.backbone.pretrained)
    model.da3_metric.cpu()
    # Keep block weights on CPU between calls; tensors/attention context remain
    # on the same GPU. This changes residency, not the model or sequence length.
    def load_block(module, args):
        module.cuda()

    def unload_block(module, args, output):
        module.cpu()

    for block in model.da3.backbone.pretrained.blocks:
        block.cpu()
        block.register_forward_pre_hook(load_block)
        block.register_forward_hook(unload_block)
    # MLPs are tokenwise at eval: tile only their linear/activation operations,
    # retaining the complete 128-frame attention context unchanged.
    for module in model.modules():
        if module.__class__.__name__ in ("SwiGLUFFN", "SwiGLUFFNFused", "Mlp"):
            original_forward = module.forward

            def tiled_forward(x, forward=original_forward):
                flat = x.reshape(-1, x.shape[-1])
                output = torch.empty_like(flat)
                for offset in range(0, len(flat), 2048):
                    output[offset : offset + 2048] = forward(flat[offset : offset + 2048])
                return output.reshape_as(x)

            module.forward = tiled_forward
from track4world_scale_fix import ChunkScaleLedger

raw = ROOT / "raw_data/singlerighthand_sandwich_100" / EP
ci = json.loads((raw / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"]["head"]["streams"][
    "color"
]["intrinsics"]
K = torch.tensor(
    [
        [ci["fx"] * 630 / 640, 0, (ci["ppx"] + 0.5) * 630 / 640 - 0.5],
        [0, ci["fy"] * 448 / 480, (ci["ppy"] + 0.5) * 448 / 480 - 0.5],
        [0, 0, 1],
    ],
    device="cuda",
)
cap = cv2.VideoCapture(str(raw / "videos/head.mp4"))
T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = list(range(T)) if "--all-chunks" in sys.argv else [0, 300, 650, 1000]
starts = sorted(set([f // 128 * 128 for f in frames] + [(T - 1) // 128 * 128]))
results = {}
for start in starts:
    end = min(start + 128, T)
    target = OUT / f"chunk_{start:04d}.json"
    if target.exists():
        results[start] = json.loads(target.read_text())
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    images = []
    for f in range(start, end):
        ok, bgr = cap.read()
        assert ok
        images.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 448)))
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().cuda() / 255
    x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor([0.229, 0.224, 0.225])[
        None, :, None, None
    ]
    x = F.interpolate(x, (448, 630), mode="bilinear", align_corners=False, antialias=True)[None]
    print(f"Running original chunk {start}:{end}", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model.da3(
            x, export_feat_layers=[], infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced"
        )
        pred = Dict({k: output[k].float().clone() for k in ["depth", "depth_conf", "intrinsics", "extrinsics"]})
        del output
        if "--low-memory" in sys.argv:
            model.da3_metric.cuda()
        # Metric branch is monocular; process frames in smaller batches to save memory.
        mets = []
        for offset in range(0, end - start, 4):
            m = model.da3_metric(x[:, offset : offset + 4])
            mets.append({k: m[k].float() for k in ["depth", "sky"]})
            del m
        metric = Dict({k: torch.cat([m[k] for m in mets], dim=1) for k in ["depth", "sky"]})
        del mets
        if "--low-memory" in sys.argv:
            model.da3_metric.cpu()
    del x
    scales = {}
    norms = {}
    roundtrip = {}
    for variant in ["estimated", "B"]:
        p = Dict({k: v.clone() for k, v in pred.items()})
        m = Dict({k: v.clone() for k, v in metric.items()})
        original_K = p.intrinsics
        if variant == "B":
            p.intrinsics = K[None, None].expand_as(original_K)
        torch.manual_seed(42)
        with torch.inference_mode():
            model._apply_metric_scaling(p, m)
            p.intrinsics = original_K
            model._apply_depth_alignment(p, m)
            model._handle_sky_regions(p, m)
            # Identical rays and norm axes to Track4World process_geometry.
            yy, xx = torch.meshgrid(torch.arange(448, device="cuda"), torch.arange(630, device="cuda"), indexing="ij")
            z = p.depth[0]
            k = p.intrinsics[0]
            xyz = torch.stack(
                [
                    (xx[None] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None] * z,
                    (yy[None] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None] * z,
                    z,
                ],
                dim=1,
            )
            ledger = ChunkScaleLedger()
            normalized = ledger.normalize(xyz)
            restored = ledger.restore(normalized)
            roundtrip[variant] = float((restored - xyz).abs().max())
            norms[variant] = float(ledger.divisors[0]) - 1e-6
            scales[variant] = p.scale_factor
        del p, m, xyz, normalized, restored, z, k, ledger
    result = dict(
        start=start,
        end=end,
        norms=norms,
        metric_alignment_scales=scales,
        metric_ratio=scales["B"] / scales["estimated"],
        roundtrip_max_abs_m=roundtrip,
        mean_fx_patch=float(pred.intrinsics[..., 0, 0].mean()),
    )
    target.write_text(json.dumps(result, indent=2) + "\n")
    results[start] = result
    print(json.dumps(result), flush=True)
    del pred, metric
    torch.cuda.empty_cache()
cap.release()
(OUT / ("chunks_all.json" if "--all-chunks" in sys.argv else "chunks.json")).write_text(
    json.dumps(dict(episode=EP, frames=frames, total_frames=T, chunks=results), indent=2) + "\n"
)
