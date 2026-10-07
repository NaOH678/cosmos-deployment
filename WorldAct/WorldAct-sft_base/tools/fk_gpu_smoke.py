"""FK branch smoke test on a CUDA device -- the half the dev node cannot cover.

What this covers that the CPU unit tests do not: the training path runs the
branch under autocast bf16 on CUDA, so the places dtype or device can go wrong
are exactly the ones a float32/CPU run never exercises --

* ``xyz_scale`` is a float64 buffer, and dividing bf16 coordinates by it is a
  silent upcast unless the result is cast back;
* ``sigma_frequencies`` multiplies a float32 sigma, and the sigma MLP's input
  dtype has to follow the branch's parameters;
* ``FKBranch.to(dtype=...)`` is called by ``install_fk`` *after* ``.to(device)``,
  and an ``nn.Embedding`` that misses that cast fails only at the first forward.

What it does NOT cover: the ``Cosmos3VFMNetwork`` integration.  Installing FK
into a real network and running one forward needs the training config and a
checkpoint, which is the launch script from design stage 5.  Until that exists,
a green run here means "the branch is sound on GPU", not "training starts".

Run from this worktree's root.  The venv lives in the sibling worktree, so give
its absolute path -- this one has none, and falling back to the ambient ``python``
would pick up a Conda prefix with no torch:

    cd /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft_mano
    PYTHONPATH=. /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python \
        tools/fk_gpu_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmos_framework.data.fk_batch import build_fk_batch  # noqa: E402
from cosmos_framework.data.fk_window import FKTiming  # noqa: E402
from cosmos_framework.data.generator.action.fk_source import FKSource  # noqa: E402
from cosmos_framework.model.generator.fk_branch import FKBranch  # noqa: E402
from cosmos_framework.model.generator.fk_training import fk_add_noise, fk_loss  # noqa: E402

ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian")
FK_ROOT = ROOT / "raw_data" / "sandwich_fk21"
KEYPOINTS = 21
HIDDEN = 256


def main() -> int:
    if not torch.cuda.is_available():
        print("❌ 没有可见的 CUDA 设备；这个脚本必须在 GPU 节点上跑")
        return 1
    device = torch.device("cuda")
    print(f"✅ CUDA: {torch.cuda.get_device_name(0)}  ({torch.cuda.device_count()} 张可见)")

    timing = FKTiming()
    source = FKSource(FK_ROOT, timing=timing)
    frame_ids = np.arange(0, 2 * (timing.steps + 1), 2)
    batch = build_fk_batch(
        [source.load("episode_0013_20260731_133649", frame_ids),
         source.load("episode_0014_20260731_133743", frame_ids)],
        batch_size=2,
    )
    print(f"✅ 批: anchor {tuple(batch.inputs['anchor_xyz'].shape)}  displacement {tuple(batch.displacement.shape)}")

    # install_fk does .to(device) then .to(dtype); reproduce that order exactly.
    branch = FKBranch(timing=timing, hidden_dim=HIDDEN, index_dim=64, content_dim=256, decoder_dim=256)
    branch = branch.to(device=device).to(dtype=torch.bfloat16)
    print(f"✅ 分支已 .to(cuda).to(bfloat16) —— 参数 dtype = {next(branch.parameters()).dtype}")

    batch = batch.to(device)
    sigma = torch.tensor([0.3, 0.7], device=device)
    # NOT zeros: a zero input gives every Linear a zero weight gradient
    # (dL/dW = dL/dout . input^T), so step 4 would report "no gradient" for the
    # motion encoder on a branch that is perfectly healthy.  Real xt is
    # noise-plus-signal, so this is also the more faithful fixture.
    noisy = torch.randn(timing.steps, 2 * KEYPOINTS, 3, device=device) * 0.02

    print("\n① encode（bf16）")
    encoded = branch.encode(batch, noisy, sigma)
    anchor_tokens, noisy_tokens = encoded["anchor_tokens"], encoded["noisy_tokens"]
    print(f"   anchor_tokens {tuple(anchor_tokens.shape)} {anchor_tokens.dtype}")
    print(f"   noisy_tokens  {tuple(noisy_tokens.shape)} {noisy_tokens.dtype}")
    assert noisy_tokens.shape == (timing.blocks, 2 * KEYPOINTS, HIDDEN)
    assert torch.isfinite(anchor_tokens).all() and torch.isfinite(noisy_tokens).all(), "bf16 溢出成 inf/nan"

    print("② decode（bf16）")
    # Feed decode the ENCODER's own output rather than a fresh random tensor. In
    # training the backbone sits between the two, but it is the encoder's tokens
    # that reach it -- so this is the closest stand-in, and it is the version that
    # actually exercises the whole branch graph. A detached `torch.randn` here
    # would leave the encoder out of the graph entirely and step ④ would report
    # "no gradient" for a reason that has nothing to do with the branch.
    hidden = noisy_tokens
    velocity = branch.decode(hidden, noisy, sigma, batch.inputs["point_batch"], 2)
    print(f"   velocity {tuple(velocity.shape)} {velocity.dtype}")
    assert velocity.shape == (timing.steps, 2 * KEYPOINTS, 3)
    assert torch.isfinite(velocity).all()

    print("③ 加噪 + loss + ADE（loss 内部转 float32）")
    state = fk_add_noise(batch, torch.tensor([0.5, 0.5], device=device), scale=1.0)
    loss, metrics = fk_loss(velocity, state, batch, scale=1.0)
    print(f"   loss = {loss.item():.4f}   （未训练分支应≈1.0：单位方差噪声主导）")
    print(f"   ade  = {metrics['fk_ade_mm'].item():.1f} mm   zero_ade = {metrics['fk_zero_ade_mm'].item():.1f} mm")
    assert torch.isfinite(loss)

    print("④ 反传（确认梯度能流到 index_embedding 和 xyz_encoder）")
    # Only three parameters are asserted, and anchor_projection is deliberately not
    # one of them: the anchor tokens are *conditioning*, and without a backbone they
    # reach the loss by no path at all, so their projection is expected to have no
    # gradient here.  In training they sit in the sequence and the noisy tokens'
    # hidden states depend on them through attention, so the gradient does flow.
    loss.backward()
    for name in ("encoder.index_embedding.weight", "encoder.xyz_encoder.0.weight", "decoder.0.weight"):
        param = dict(branch.named_parameters())[name]
        grad = param.grad
        ok = grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        print(f"   {'✅' if ok else '❌'} {name}  grad_sum = {float(grad.abs().sum()) if grad is not None else 'None'}")
        assert ok, f"{name} 没拿到有效梯度"

    print("\n✅ GPU 上的 FK 分支自检通过")
    print("   ⚠️ 这不等于训练能起来 —— Cosmos3VFMNetwork 的集成要等阶段 5 的启动脚本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
