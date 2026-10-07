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
torch.cuda.set_per_process_memory_fraction(0.22)
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
raw = ROOT / "raw_data/singlerighthand_sandwich_100" / EP
cap = cv2.VideoCapture(str(raw / "videos/head.mp4"))
T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start = (T - 1) // 128 * 128
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
images = []
for f in range(start, T):
    ok, bgr = cap.read()
    assert ok
    images.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 448)))
cap.release()
x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().cuda() / 255
x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor([0.229, 0.224, 0.225])[
    None, :, None, None
]
x = F.interpolate(x, (448, 630), mode="bilinear", align_corners=False, antialias=True)[None]
print(f"Running original final chunk {start}:{T}", flush=True)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    raw_pred = model.da3(
        x, export_feat_layers=[], infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced"
    )
    pred = Dict({k: raw_pred[k].float().clone() for k in ["depth", "depth_conf", "intrinsics", "extrinsics"]})
    del raw_pred
    raw_metric = model.da3_metric(x)
    metric = Dict({k: raw_metric[k].float().clone() for k in ["depth", "sky"]})
    del raw_metric
cached = np.load(ROOT / "pf_out/9.24/sandwich/efep_seg_v2" / EP / "intrinsics.npy")
# infer_pair uses normalized fx = mean(fx_patch)/sqrt(Wpatch^2+Hpatch^2) * sqrt(1+aspect^2)/aspect.
fx = float(pred.intrinsics[..., 0, 0].mean())
expected = fx / np.hypot(630, 448) * np.hypot(640, 448) / 640
observed = float(cached[-1, 0, 0])
rel = abs(expected / observed - 1)
print("Cached focal validation", expected, observed, rel, flush=True)
assert rel < 0.002, "Original baseline focal did not reproduce; do not render B"
ci = json.loads((raw / "auxiliary_camera/metadata.json").read_text())["capture_metadata"]["cameras"]["head"]["streams"][
    "color"
]["intrinsics"]
assert np.allclose(ci["coeffs"], 0)
K = np.array(
    [
        [ci["fx"] * 630 / 640, 0, (ci["ppx"] + 0.5) * 630 / 640 - 0.5],
        [0, ci["fy"] * 448 / 480, (ci["ppy"] + 0.5) * 448 / 480 - 0.5],
        [0, 0, 1],
    ],
    np.float32,
)
scales = {}
for label in ["estimated", "B"]:
    p = Dict({k: v.clone() for k, v in pred.items()})
    m = Dict({k: v.clone() for k, v in metric.items()})
    # Override K only for Metric scaling; geometry stays on the original predicted
    # rays until the final export backprojection, avoiding an extra normalization change.
    original_K = p.intrinsics
    if label == "B":
        p.intrinsics = torch.tensor(K, device="cuda")[None, None].expand_as(p.intrinsics)
    torch.manual_seed(42)
    with torch.inference_mode():
        model._apply_metric_scaling(p, m)
        p.intrinsics = original_K
        model._apply_depth_alignment(p, m)
        model._handle_sky_regions(p, m)
    scales[label] = p.scale_factor
ratio = scales["B"] / scales["estimated"]
report = dict(
    episode=EP,
    frames_total=T,
    geometry_chunk=[start, T],
    seed=42,
    scales=scales,
    ratio_B_over_original=ratio,
    cached_normalized_fx=observed,
    rerun_normalized_fx=expected,
    focal_relative_error=rel,
    K_true_patch=K.tolist(),
    rgb_intrinsics=ci,
    method="Frozen original selected observations; original Track4World AnyView weights and final 128-frame geometry chunk; replace Metric focal calibration then final RGB unprojection; preserve legacy last-chunk metric scale restoration.",
    limitation="Not a retracking run; freezes original validity/unique masks. Per-chunk norm epsilon 1e-6 and sky clipping can break ideal scale cancellation; baseline focal match alone does not prove bitwise full-export reproduction.",
)
(OUT / "scale_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
