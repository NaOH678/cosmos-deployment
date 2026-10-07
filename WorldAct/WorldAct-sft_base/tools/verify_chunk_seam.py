#!/usr/bin/env python3
"""Test whether the DualDPT head's ``chunk_size=8`` leaves a depth seam every 8 frames.

WHY THIS EXISTS
    Measured on the exported point clouds (``datasets/sandwich_dense_fullseq_10_0298_20260908``),
    the frame-to-frame depth jump of the dense 3D trajectory is not uniform in time:
    every jump that exceeds the episode's own p99 lands on a frame index congruent to
    7 modulo 8 -- i.e. exactly when the transition crosses an 8-frame boundary.  All
    ten episodes show it; eight of them have 100% of their outlier frames in that one
    bucket out of eight, so it is not chance.

    Eight is ``DualDPT.forward``'s default chunk size::

        depth_anything_3/model/dualdpt.py:162   def forward(..., chunk_size: int = 8)
        depth_anything_3/model/dualdpt.py:189   for s0 in range(0, B*S, chunk_size): ...

    and the only call site does not pass it::

        depth_anything_3/model/da3.py:241       return self.head(feats, H, W, patch_start_idx=0)

    The head's fusion is frame-wise convolution, so chunking *looks* like a pure memory
    optimisation that cannot change the result.  The measurement says otherwise.  This
    script settles it by running the same frames through the same loaded model twice,
    changing only that one argument, and comparing the temporal structure of the output.

WHAT IT MEASURES
    ``geometry['points']`` is the per-frame point map produced straight from the depth
    head -- no tracking, so any seam in it is the head's own.  ``motion['flow_3d']`` is
    the tracked trajectory that the exporter actually saves.  Reporting both separates
    a seam in the depth head from one in the flow head (which chunks at 12, not 8).

    The test is not "does B still have a seam" but "does the seam's *period* follow the
    chunk size".  Bucketing |dz| by (frame index mod p) and comparing the tallest bucket
    against the rest only ties p with 2p and 3p -- every multiple shows a tall bucket of
    the same height, since a seam every p frames is also a seam every 2p.  Counting how
    many buckets are tall breaks the tie, and the period is the LARGEST p with exactly
    one tall bucket.  The phase (which bucket) is found, not assumed: a jump between
    frame k and k+1 lands in bucket k, and assuming period-1 would read an off-by-one as
    "no seam".

    A period needs MIN_SAMPLES_PER_BUCKET frames per bucket to be considered at all, so
    --frames and the chunk size must be chosen together: 120 frames reach period 20,
    which covers the 8-vs-16 comparison.  Every number is computed on the in-memory
    float32 output and the cache is float32 too, so ``--analyse-only`` reproduces what
    the run printed -- float16 resolves only ~0.5 mm at z~1 m, the same order as the
    baseline noise and a fifth of the seam.

USAGE
    # one episode prefix, both settings; A (chunk 8, the production value) took 32 s for
    # 120 frames on an A800, and the whole run including model load finished in ~2 min
    python tools/verify_chunk_seam.py --episode episode_0013_20260731_133649 --frames 120

    # read back a previous run without re-running the model
    python tools/verify_chunk_seam.py --analyse-only --out /data/shichaojian/renders/chunk_seam_verify

NOTE ON THE INTERPRETER
    This must run under Track4World's own venv (python 3.10), not the cosmos venv
    (3.13): the model code imports ``depth_anything_3`` and ``utils3d`` from there.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# --------------------------------------------------------------------------- paths
# New-cluster defaults.  Every one is overridable; nothing here is inferred.
DATA_ROOT = Path('/data/shichaojian')
REPO_DEFAULT = DATA_ROOT / 'Track4World_portable/Track4World'
DA3_DEFAULT = DATA_ROOT / 'checkpoints/DA3NESTED-GIANT-LARGE-1.1'
SOURCE_DEFAULT = DATA_ROOT / 'raw_data/singlerighthand_sandwich_100'
OUT_DEFAULT = DATA_ROOT / 'renders/chunk_seam_verify'

# The resize the exporter uses (run_dense_export.read_video): fit to 640 on the long
# side, then floor to a multiple of 64.  640x480 -> 448x640, which is the canvas the
# saved intrinsics and point clouds are expressed in.
SIDE = 640
MULTIPLE = 64

CHUNK_A = 8      # DualDPT.forward's default: what the current export actually uses
CHUNK_B = None   # the whole sequence in one _forward_impl


# ------------------------------------------------------------------------ helpers
def read_video(path: Path, frames: int) -> tuple[np.ndarray, int, int]:
    """First ``frames`` frames as (T, H, W, 3) uint8, resized exactly as the exporter."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'cannot open {path}')
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = min(SIDE / height, SIDE / width)
    out_h = int(height * scale) // MULTIPLE * MULTIPLE
    out_w = int(width * scale) // MULTIPLE * MULTIPLE
    buffer = np.empty((frames, out_h, out_w, 3), np.uint8)
    for i in range(frames):
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f'video ended at frame {i}; ask for fewer --frames')
        buffer[i] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (out_w, out_h),
                               interpolation=cv2.INTER_LINEAR)
    capture.release()
    return buffer, out_h, out_w


def load_model(repo: Path, da3_model: Path, ckpt: Path):
    """Load Track4World exactly as run_dense_export.py does, so the comparison is on
    the production path and not on a lookalike."""
    os.environ.update(
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TORCH_HOME=str(DATA_ROOT / 'checkpoints/torch-hub'),
        TRACK4WORLD_DA3_MODEL=str(da3_model),
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
    )
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    torch.set_num_threads(8)
    import demo
    config = json.loads((repo / 'track4world/config/eval/v1.json').read_text())
    args = SimpleNamespace(coordinate='camera_depthanythingv3', ckpt_init=str(ckpt),
                           use_original_backbone=False, metric_scale=True)
    return demo.load_model(args, config), torch


def depth_head_chunk(model, chunk):
    """Force ``chunk_size=chunk`` on both DA3 branches' depth head.

    Patched on the *instance*, not the class: ``depth_anything_3`` is importable both
    as a top-level package and under ``track4world.nets.external``, so there can be two
    module objects for the same file and a class patch would silently miss the copy the
    model actually holds.  Binding the method onto each net avoids that entirely.
    """
    import types

    def patched(self, feats, H, W):
        return self.head(feats, H, W, patch_start_idx=0, chunk_size=chunk)

    nested = model.backbone.model          # NestedDepthAnything3Net
    for name in ('da3', 'da3_metric'):
        # Assert rather than getattr: if the nesting changes, an AttributeError here
        # is the difference between "the patch did nothing and the two runs were
        # identical" and a loud failure.  A silent no-op would read as a clean result.
        net = getattr(nested, name)
        assert hasattr(net, '_process_depth_head'), f'{name} has no _process_depth_head'
        assert hasattr(net.head, 'forward'), f'{name}.head is not a module'
        net._process_depth_head = types.MethodType(patched, net)
    return nested


def keep_final_iteration_only(model):
    """Drop the intermediate refinement iterations the flow head keeps on the GPU.

    Copied from run_efep_export.py, where it is what makes a 2000-frame episode fit.
    It changes GPU memory only; the returned tensors are the same ones infer_pair
    would have used anyway.
    """
    if getattr(model, '_final_iteration_patched', False):
        return                                    # run_once calls this once per run
    model._final_iteration_patched = True
    original = model.forward_window_unified

    def final_only(*args, **kwargs):
        flow, flow3d, vis, *rest = original(*args, **kwargs)

        def last(x):
            return None if x is None else [x[-1]]
        return (last(flow), last(flow3d), last(vis), *rest)

    model.forward_window_unified = final_only


def run_once(model, torch, images, chunk, mem_patch: bool, label: str, min_frames: int = 24):
    """One full model.infer() with the depth head chunked at ``chunk``.

    ``chunk_size=None`` holds every frame's feature pyramid at once, so it can OOM where
    the production chunking does not -- measured: 120 frames OOM at None on an 80 GB
    card while the same 120 frames run in 32 s at 8.  Rather than make the caller guess
    a frame count that fits, halve until it does and say how far it got.  The seam is a
    local property of each chunk boundary, so a shorter prefix still shows it; the
    caller trims both sides to the common length before comparing.

    ``--mem-patch`` does not help here: the peak is in the depth head (``get_fmaps``),
    which runs before the flow head that patch targets.
    """
    depth_head_chunk(model, chunk)
    if mem_patch:
        keep_final_iteration_only(model)
    n = int(images.shape[1])
    while True:
        print(f'[{label}] chunk_size={chunk}  frames={n}', flush=True)
        start = time.monotonic()
        try:
            with torch.inference_mode():
                result, cache = model.infer(images[:, :n], iters=4, sw=None,
                                            is_training=False, tracking3d=True)
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if n <= min_frames:
                raise SystemExit(
                    f'\nOOM on chunk_size={chunk} even at {n} frames.  chunk_size=None\n'
                    f'needs the whole sequence resident at once; give it fewer --frames\n'
                    f'or compare against a finite --chunk-b (e.g. 32) instead of None.')
            n //= 2
            print(f'[{label}] OOM; retrying at {n} frames', flush=True)
    torch.cuda.synchronize()
    geometry, motion = result
    points = geometry['points'][0].float().cpu().numpy()          # (T,H,W,3) no tracking
    flow3d = motion['flow_3d'][0].float().cpu().numpy()           # (T,H,W,3) tracked
    del result, cache, geometry, motion
    gc.collect()
    torch.cuda.empty_cache()
    print(f'[{label}] {time.monotonic() - start:.1f}s  points{points.shape}', flush=True)
    return points[..., 2], flow3d[..., 2]


# ------------------------------------------------------------------------ analysis
def frame_profile(z: np.ndarray, stride: int = 8):
    """Per-frame median |dz| over a spatial subsample.  Returns (T-1,) in millimetres."""
    sample = z[:, ::stride, ::stride]
    dz = np.abs(np.diff(sample, axis=0))
    with np.errstate(invalid='ignore'):
        return np.nanmedian(dz, axis=(1, 2)) * 1000.0


def mod_profile(profile: np.ndarray, period: int = 8):
    """Median |dz| per (frame index mod period) -- the headline number.

    A seam at every ``period`` frames puts all of the height into bucket period-1.
    """
    idx = np.arange(len(profile))
    return np.array([np.nanmedian(profile[idx % period == r]) for r in range(period)])


# Candidate periods for the sweep.  Multiples of 4 up to 64 covers every plausible
# chunk size plus the flow head's 12; keeping it to a dozen keeps the table readable.
SWEEP_PERIODS = (4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)

# A bucket must hold this many frames before its median is trusted.  It caps the
# detectable period at len(profile)/MIN_SAMPLES_PER_BUCKET, so the frame count and the
# chunk size under test have to be chosen together: 120 frames reach period 20, and
# testing a 32-frame chunk needs >= 192 frames.
MIN_SAMPLES_PER_BUCKET = 6


def period_sweep(profile: np.ndarray, periods=SWEEP_PERIODS) -> dict:
    """For each candidate period p: the tallest bucket's ratio, and how many are tall.

    The *pair* is what identifies the period, and the height alone cannot.  A seam every
    P frames puts one bucket above the rest at p=P, but two at p=2P (those frames fall
    into two residue classes mod 2P), three at p=3P, and so on -- every multiple of P
    shows a tall bucket with the same height, because the ratio compares the tall bucket
    against a baseline that is unchanged.  Reading the height only would tie 8 against
    16 and 24 with no way to choose.

    Counting buckets breaks the tie, and the answer is the LARGEST p that still shows
    exactly one tall bucket: at p=P exactly one residue class is all-seam; at p=2P two
    are; and any divisor of P shows one too, so taking the largest lands on P itself.
    (A divisor d of P also yields one tall bucket -- every seam frame is congruent mod d
    -- which is why the search must go upward, not downward.)

    A period is only considered once every bucket holds at least MIN_PER_BUCKET samples.
    This bound is what keeps the search honest at large p: with three samples per bucket
    a median is noise, and a noise spike reads as "exactly one tall bucket", which is
    precisely the signature the search is looking for.  Measured on synthetic data, a
    3-sample bound reported a period of 64 for a series that had no seam at all.

    ``n_tall`` is forced to 0 when nothing stands out, so a quiet run cannot be read as a
    one-bucket period by threshold noise.
    """
    out = {}
    for p in periods:
        if len(profile) < MIN_SAMPLES_PER_BUCKET * p:
            continue
        m = mod_profile(profile, p)
        base = float(np.median(m))
        tall = float(np.max(m))
        ratio = tall / base if base else float('nan')
        n_tall = int((m > 1.5 * base).sum()) if ratio >= 1.5 else 0
        out[p] = dict(ratio=ratio, n_tall=n_tall)
    return out


def seam_period(sweep: dict):
    """The largest period with exactly one tall bucket, or None if there is no seam."""
    single = [p for p, v in sweep.items() if v['n_tall'] == 1]
    return max(single) if single else None


def report(name: str, profile: np.ndarray, period: int = 8) -> dict:
    """Two independent readings of the same seam.

    ``ratio`` is bucket (period-1) against the median of the other buckets -- how much
    taller the seam frame is than a normal one.  ``hot_frac`` is the share of the p99
    outliers that land in that bucket -- where the seam *is*, which stays meaningful
    even when the two heights are close, because the outliers are counted rather than
    averaged.  Reading both is what keeps a noisy baseline (float rounding, a genuinely
    jumpy scene) from being mistaken for a seam, and a real seam from being averaged
    away.
    """
    # The tallest bucket is *found*, not assumed to be period-1.  Which residue class
    # holds the jump depends on where the chunk boundary falls between frames -- a jump
    # between frame k and k+1 lands in bucket k -- and an off-by-one would otherwise be
    # read as "no seam" while the seam sits in the next bucket over.
    mod = mod_profile(profile, period)
    tall = int(np.argmax(mod))
    base = float(np.median(np.delete(mod, tall)))
    seam = float(mod[tall])
    thr = float(np.nanpercentile(profile, 99))
    hot = np.nonzero(profile > thr)[0]
    hist = np.bincount(hot % period, minlength=period).tolist()
    ratio = seam / base if base else float('nan')
    hot_frac = hist[tall] / max(1, sum(hist))
    sweep = period_sweep(profile)
    best = seam_period(sweep)
    phase = int(np.argmax(mod_profile(profile, best))) if best else None
    print(f'\n--- {name} ---')
    print(f'  |dz| 中位 (mm)           : {np.nanmedian(profile):.4f}')
    print(f'  (帧号 mod {period}) 的中位 |dz| : {np.array2string(mod, precision=3)}')
    print(f'     → 最高桶 {tall} {seam:.3f} vs 其余中位 {base:.3f}   比值 {ratio:.2f}x')
    print(f'  超 p99({thr:.3f}mm) 的帧数 : {len(hot)}   mod{period} 分布 {hist}'
          f'   → 桶 {tall} 占 {hot_frac * 100:.0f}%')
    print('  周期扫描 (周期: 最高桶比值 / 高桶个数):')
    print('    ' + '  '.join(f'{p}:{v["ratio"]:.2f}x/{v["n_tall"]}'
                             for p, v in sorted(sweep.items())))
    print(f'     → 接缝周期 = {best}'
          + (f'  相位(最高桶) = {phase}  ({sweep[best]["ratio"]:.2f}x, 单桶)'
             if best else '  (无单桶周期)'))
    return dict(median_mm=float(np.nanmedian(profile)), mod_mm=mod.tolist(),
                seam_mm=seam, base_mm=base, tall_bucket=tall, ratio=ratio,
                hot_frac=hot_frac, p99_mm=thr, n_hot=len(hot), hot_mod=hist,
                sweep=sweep, seam_period=best, phase=phase)


def has_seam(r: dict, period: int = 8) -> bool:
    """Whether one reading shows a seam at the chunk boundary.

    Two independent signals, either one sufficient.  ``ratio`` is a height comparison
    and needs no sample size.  ``hot_frac`` says *where* the outliers are, which
    survives a noisy baseline -- but only once there are enough outliers for the
    location to mean anything: with 2 outliers over 8 buckets, one landing in bucket
    7 is 50% by chance, so requiring ``n_hot >= period`` is what stops that from
    reading as a seam.
    """
    return r['ratio'] >= 1.5 or (r['n_hot'] >= period and r['hot_frac'] >= 0.5)


def compare(a: dict, b: dict, period: int = 8):
    """Print the verdict.  Returns (period_moved, period_pinned).

    The test is not "does B still have a seam" but "does the seam's *period* follow the
    chunk size".  A seam at period 8 that stays at period 8 when the chunk size becomes
    32 is not the chunking; one that moves to 32 is, and that is a causal statement
    rather than a correlation with one setting.

    Returns (period_moved, period_pinned); both False means neither side reproduced a
    seam at all, which is an empty result and must not be reported as a pass.
    """
    print('\n' + '=' * 82)
    print(f'{"":24}{"A: chunk=" + str(CHUNK_A):>22}{"B: chunk=" + str(CHUNK_B):>22}')
    print('-' * 82)
    for key, fmt in (('median_mm', '{:.4f}'), ('seam_mm', '{:.4f}'), ('base_mm', '{:.4f}'),
                     ('ratio', '{:.2f}x'), ('seam_period', '{:}'), ('phase', '{:}')):
        print(f'  {key:<22}{fmt.format(a[key]):>22}{fmt.format(b[key]):>22}')
    print(f'  {"hot_mod":<22}{str(a["hot_mod"]):>22}{str(b["hot_mod"]):>22}')

    pa, pb = a['seam_period'], b['seam_period']
    strong_a = pa is not None
    strong_b = pb is not None
    print('-' * 82)
    print(f'  接缝周期: A = {pa} (相位 {a["phase"]})    B = {pb} (相位 {b["phase"]})')
    print(f'  chunk_size: A = {CHUNK_A}    B = {CHUNK_B}')

    moved = pinned = False
    if not (strong_a or strong_b):
        msg = ('两侧都没有明显接缝 —— 这条帧段上复现不出该现象，结论为空。加 --frames、'
               '换一条 episode，或改用 geometry["points"] 之外的量')
    elif strong_a and strong_b and pa == pb:
        pinned = True
        msg = (f'两侧峰值都在周期 {pa} —— 接缝周期没有跟着 chunk_size 变，'
               f'所以 DualDPT 的分块不是原因；下一个嫌疑是 flow 头的 chunk_size=12'
               f'（model.py:1324）')
    elif strong_a and strong_b and pa != pb:
        moved = True
        msg = (f'接缝周期从 {pa} 移到了 {pb}，正好跟着 chunk_size 从 {CHUNK_A} 变到 '
               f'{CHUNK_B} —— 因果关系成立，DualDPT 的 chunk_size 就是接缝的来源')
    elif strong_a and not strong_b:
        moved = True
        msg = (f'A 在周期 {pa} 有接缝，B（chunk_size={CHUNK_B}）没有了 —— '
               f'接缝随分块消失，DualDPT 的 chunk_size 就是来源')
    elif strong_b and not strong_a:
        msg = '异常：只有 B 有接缝 —— 先别下结论，重跑一次确认'
    else:
        msg = f'峰值周期 A={pa} B={pb} 与 chunk_size 不对应，人工看上面的扫描表'
    print('  ' + msg)
    return moved, pinned


# ---------------------------------------------------------------------------- main
def main() -> int:
    global CHUNK_A, CHUNK_B
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episode', default='episode_0013_20260731_133649')
    ap.add_argument('--frames', type=int, default=120,
                    help='frames from the start; 120 gives 15 seams at period 8')
    ap.add_argument('--repo', type=Path, default=REPO_DEFAULT)
    ap.add_argument('--da3-model', type=Path, default=DA3_DEFAULT)
    ap.add_argument('--source-root', type=Path, default=SOURCE_DEFAULT)
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--cache', type=Path, default=None,
                    help='explicit npz for --analyse-only (default: the only one in --out)')
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--chunk-a', type=int, default=CHUNK_A,
                    help='A side; 8 is DualDPT.forward\'s default and what the export uses')
    ap.add_argument('--chunk-b', type=int, default=16,
                    help='B side.  Default 16, not -1/None: None holds every frame at '
                         'once and OOM\'d at 120 frames on an 80 GB card, and the halving '
                         'that follows leaves too few frames per bucket for the period '
                         'search.  16 is memory-safe and is a multiple of 8, so B must '
                         'show its seam at 16 -- not at 8 like A -- if chunking is the '
                         'cause.  -1 selects None if it fits.')
    ap.add_argument('--period', type=int, default=8,
                    help='bucket frames by (index mod period) -- the chunk size under test')
    ap.add_argument('--mem-patch', action='store_true',
                    help='drop intermediate flow iterations (use if it OOMs)')
    ap.add_argument('--analyse-only', action='store_true',
                    help='read the cached npz and skip the model entirely')
    args = ap.parse_args()

    # Absolute before anything else: load_model() chdir's into the Track4World repo (the
    # model code imports `depth_anything_3` and `utils3d` relative to it), so a relative
    # --out would be created here and then written into the repo instead.
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    norm = lambda c: None if c is None or c < 0 else c       # noqa: E731
    if not args.analyse_only:
        CHUNK_A, CHUNK_B = norm(args.chunk_a), norm(args.chunk_b)
    period = args.period
    cache_name = f'{args.episode}_chunk{CHUNK_A}_vs_{CHUNK_B}.npz'
    cache_path = args.cache or (args.out / cache_name)

    if args.analyse_only:
        # The chunk sizes live *inside* the npz, so the file name cannot be derived from
        # them before opening it.  Accept an explicit path, else take the only npz in
        # --out; more than one there and the choice is the caller's to make.
        if not args.cache:
            found = sorted(args.out.glob('*.npz'))
            if len(found) == 1:
                cache_path = found[0]
            elif not found:
                print(f'ERROR: no cached run (*.npz) in {args.out}', file=sys.stderr)
                return 2
            else:
                print(f'ERROR: {len(found)} cached runs in {args.out}; name one with '
                      f'--cache', file=sys.stderr)
                return 2
        if not cache_path.is_file():
            print(f'ERROR: no cached run at {cache_path}', file=sys.stderr)
            return 2
        with np.load(cache_path) as d:
            za, zf = d['points_a'], d['flow_a']
            zb, zg = d['points_b'], d['flow_b']
            ca, cb = int(d['chunk_a']), int(d['chunk_b'])
            CHUNK_A = None if ca < 0 else ca
            CHUNK_B = None if cb < 0 else cb
    else:
        ckpt = args.repo / 'checkpoints/track4world_da3.pth'
        for path in (args.repo, args.da3_model, ckpt,
                     args.source_root / args.episode / 'videos' / 'head.mp4'):
            if not path.exists():
                print(f'ERROR: missing {path}', file=sys.stderr)
                return 2

        model, torch = load_model(args.repo, args.da3_model, ckpt)
        raw, H, W = read_video(args.source_root / args.episode / 'videos' / 'head.mp4',
                               args.start_frame + args.frames)
        raw = raw[args.start_frame:]
        print(f'video {raw.shape}  (exporter canvas {H}x{W})')
        images = torch.from_numpy(raw).permute(0, 3, 1, 2).unsqueeze(0).to(
            device='cuda', dtype=torch.float32)
        del raw

        za, zf = run_once(model, torch, images, CHUNK_A, args.mem_patch, 'A')
        zb, zg = run_once(model, torch, images, CHUNK_B, args.mem_patch, 'B')
        del images
        gc.collect()
        torch.cuda.empty_cache()

        # float32, not float16: at z ~ 1 m float16 resolves ~0.5 mm, the same order as
        # the baseline noise, and re-analysis would then report a seam the first run
        # had not.  The seam itself is only ~5 mm, so the storage must not round it.
        # -1 encodes None; a 0-d object array would need allow_pickle to read back.
        np.savez_compressed(cache_path, points_a=za, flow_a=zf,
                            points_b=zb, flow_b=zg,
                            chunk_a=np.int64(-1 if CHUNK_A is None else CHUNK_A),
                            chunk_b=np.int64(-1 if CHUNK_B is None else CHUNK_B))
        print(f'wrote {cache_path}')

    # The two runs may not have covered the same number of frames: chunk_size=None
    # halves on OOM.  Compare on the common prefix -- and refuse if that leaves too
    # little for the period sweep to have any force, rather than quietly reporting a
    # statistic computed from three buckets.
    ta, tb = int(za.shape[0]), int(zb.shape[0])
    T = min(ta, tb)
    if ta != tb:
        print(f'\n注意: A 跑了 {ta} 帧，B 跑了 {tb} 帧（chunk_size=None 显存不够，减半重试）'
              f' → 只在共同的前 {T} 帧上比较')
    if T < 3 * max(SWEEP_PERIODS[0], CHUNK_A or 8):
        print(f'ERROR: 共同帧数只有 {T}，不足以判断周期。给 --frames 更多，'
              f'或把 --chunk-b 设成一个有限值（如 32）而不是 None。', file=sys.stderr)
        return 3
    za, zb, zf, zg = za[:T], zb[:T], zf[:T], zg[:T]

    # ---- compare.  points first: a seam there is the depth head's own, since nothing
    # has tracked anything yet.  flow_3d is the tracked trajectory the exporter saves,
    # so it carries the depth head's seam plus whatever the flow head (chunk_size=12)
    # adds; if only flow_3d shows one, the depth head is exonerated.
    print('\n' + '#' * 82)
    print('# A: geometry["points"]   (逐帧点图，未跟踪 —— 接缝若在此，就是深度头自己的)')
    print('#' * 82)
    ra = report(f'points  chunk_size={CHUNK_A}', frame_profile(za), period)
    rb = report(f'points  chunk_size={CHUNK_B}', frame_profile(zb), period)
    moved_points, pinned_points = compare(ra, rb, period)

    print('\n' + '#' * 82)
    print('# B: motion["flow_3d"]   (逐帧链接的轨迹 —— 导出实际保存的那一份)')
    print('#' * 82)
    fa = report(f'flow_3d chunk_size={CHUNK_A}', frame_profile(zf), period)
    fb = report(f'flow_3d chunk_size={CHUNK_B}', frame_profile(zg), period)
    moved_flow, pinned_flow = compare(fa, fb, period)

    summary = dict(episode=args.episode, frames_a=ta, frames_b=tb, frames_compared=T,
                   period=period, chunk_a=CHUNK_A, chunk_b=CHUNK_B,
                   points=dict(a=ra, b=rb, period_moved=moved_points,
                               period_pinned=pinned_points),
                   flow=dict(a=fa, b=fb, period_moved=moved_flow,
                             period_pinned=pinned_flow))
    report_path = args.out / f'{args.episode}_chunk{CHUNK_A}_vs_{CHUNK_B}.json'
    report_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'\nwrote {report_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
