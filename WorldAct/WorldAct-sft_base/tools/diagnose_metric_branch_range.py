"""Isolate DA3 Metric branch depth variation on same-ray D435 hand samples."""

import json
from pathlib import Path

import cv2
import numpy as np
import render_fk_vs_pointflow as core
import torch
import torch.nn.functional as F
from depth_anything_3.cfg import create_object
from omegaconf import OmegaConf
from safetensors import safe_open
from scipy.spatial import cKDTree

EP = "episode_0013_20260731_133649"
OUT = Path("outputs/da3_depth_range_diagnosis")
ROOT = Path("/data/shichaojian")
ck = ROOT / "checkpoints/DA3NESTED-GIANT-LARGE-1.1"
torch.set_num_threads(8)
torch.manual_seed(42)
torch.cuda.set_per_process_memory_fraction(0.12)
cfg = json.loads((ck / "config.json").read_text())["config"]["metric"]
model = create_object(OmegaConf.create(cfg))
with safe_open(str(ck / "model.safetensors"), framework="pt") as checkpoint:
    state = {
        k.removeprefix("model.da3_metric."): checkpoint.get_tensor(k)
        for k in checkpoint.keys()
        if k.startswith("model.da3_metric.")
    }
model.load_state_dict(state, strict=True)
del state
model.eval().cuda()
ci = core.colour_intrinsics(EP)
K = core.colour_K(ci)
focal = (ci["fx"] * 630 / 640 + ci["fy"] * 448 / 480) / 2
obs = core.load_pointflow(EP)
meta = core.depth_metadata(EP)
depthdir = Path("/tmp/fk_vs_pointflow_depth") / EP / "depth.lmdb"
frames = core.depth_keys(depthdir)
cap = cv2.VideoCapture(str(core.RAW_ROOT / EP / "videos/head.mp4"))
rows = []
for offset in range(0, len(frames), 4):
    ids = frames[offset : offset + 4]
    images = []
    for f in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, bgr = cap.read()
        assert ok
        images.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 448)))
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().cuda() / 255
    x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor([0.229, 0.224, 0.225])[
        None, :, None, None
    ]
    x = F.interpolate(x, (448, 630), mode="bilinear", align_corners=False, antialias=True)[None]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model(x)
    depth = output.depth[0].float() * focal / 300
    depth = F.interpolate(depth[:, None], (448, 640), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    for i, f in enumerate(ids):
        p, uv = core.hand_observations(obs, f)
        cloud, _ = core.d435_hand_points(EP, f, uv, K, depthdir, meta, gate="front")
        hom = cloud @ K.T
        qu, qv = core.canvas_uv_to_colour(uv)
        dist, idx = cKDTree(hom[:, :2] / hom[:, 2:]).query(np.column_stack([qu, qv]))
        keep = dist <= 1.5
        d435 = cloud[idx[keep], 2]
        pixel = uv[keep].astype(int)
        pred = depth[i, pixel[:, 1], pixel[:, 0]]
        rows.append(
            [f, len(pred), np.median(d435), np.median(pred), np.median(pred - d435), np.median(abs(pred - d435))]
        )
    if offset % 40 == 0:
        print(f"Processed {offset + len(ids)}/{len(frames)}", flush=True)
cap.release()
a = np.array(rows)
np.save(OUT / "metric_only_series.npy", a)
x, y = a[:, 2], a[:, 3]
slope, intercept = np.polyfit(x, y, 1)
q = np.quantile(x, [0.25, 0.75])
bins = {}
for name, m in [("near_quartile", x <= q[0]), ("far_quartile", x >= q[1])]:
    bins[name] = dict(
        n=int(m.sum()),
        sensor_mean_m=float(x[m].mean()),
        metric_mean_m=float(y[m].mean()),
        mean_bias_mm=float(np.mean(y[m] - x[m]) * 1000),
    )
report = dict(
    episode=EP,
    frames=len(a),
    metric_only=True,
    real_RGB_focal_patch=focal,
    slope=float(slope),
    intercept_m=float(intercept),
    correlation=float(np.corrcoef(x, y)[0, 1]),
    median_frame_paired_abs_error_mm=float(np.median(a[:, 5]) * 1000),
    bins=bins,
    columns=["frame", "pixels", "sensor_median_Z", "metric_median_Z", "median_paired_bias", "median_paired_abs_error"],
    note="Independent monocular Metric branch with real focal scaling only; no AnyView alignment, no chunk normalization, no FK fitting. Same paired RGB-ray support as front-gated sensor diagnostic. D435 is a sensor reference, not assumed perfect truth.",
)
(OUT / "metric_only_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
