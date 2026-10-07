# 任务 12：PointFlow 训练接线恢复、优化器冻结修复与可读指标

本文记录 2026-09-10 对 PointFlow 训练链路的一次排查与修复。任务 8 描述的加噪/loss 数学、任务 11 描述的单卡入口均未改变；本次修的是**参数有没有真的被优化**，以及**指标能不能读出信息**。

## 现象

单卡训练日志里 PointFlow Loss 恒在 `1.00x`，只有小数点后第三位抖动：

```
Iteration 338: Total Loss: 2.9562 | Video Loss: 0.1748 | Action Loss: 0.0207 | PointFlow Loss: 1.0011
Iteration 379: Total Loss: 2.7440 | Video Loss: 0.1624 | Action Loss: 0.0119 | PointFlow Loss: 1.0014
```

与之对比，同一次运行的 `Video Loss` 从 0.17 稳定下降，说明训练本身是正常的。

## 根因一：PointFlow 分支整体被优化器冻结

`cosmos_framework/utils/generator/optimizer.py:173` 里，`keys_to_select` 是一份**子串白名单**，不匹配的参数会被强制 `requires_grad = False` 并排除出优化器：

```python
for pn, p in param_dict.items():
    if len(keys_to_select) > 0 and not any(key in pn for key in keys_to_select):
        p.requires_grad = False
        continue
```

Edge-DROID recipe 的白名单只有 `moe_gen / time_embedder / vae2llm / llm2vae / action2llm / llm2action / action_modality_embed / k_norm_und_for_gen`，而 PointFlow 分支的参数名是 `net.pointflow_branch.geometry.*` 和 `net.pointflow_branch.codec.*`，**一个都不匹配** → 整个分支（Sonata + Codec，约 42.0M 参数、300 个 tensor）只前向、不更新。

证据（同一条命令在修复前后的输出）：

| | Total tensors | selected tensors | 优化器元素数 |
| --- | --- | --- | --- |
| 修复前 | 849 | **322** | 1,414,924,992（与无 PointFlow 的 baseline 完全一致） |
| 修复后 | 849 | **622** | 1,456,963,692（**+42,038,700**） |

多出来的 300 个 tensor / 42.0M 参数正好是 `pointflow_branch`。

## 根因二：`omni_mot_model.py` / `utils/callback.py` 的接线曾被回退

排查过程中一度出现"PointFlow loss 没有进总 loss"的结论，那是误判：这两个文件在当天 15:33:40 被一条 `git checkout -- cosmos_framework/model/generator/omni_mot_model.py cosmos_framework/utils/callback.py` 还原成 HEAD，而训练进程 15:09 就已启动，继续执行内存中的旧模块，因此日志仍有 `PointFlow Loss` 字段 —— 磁盘状态与运行状态不一致。

接线已按原样恢复（从当时的会话记录取回 diff，`git apply` 应用，应用前校验过 blob hash `8dfd9b3` / `295abda` 与工作区一致）。恢复的内容包括：

- `build_net`：`POINTFLOW_SONATA_CHECKPOINT` 存在时挂载 Sonata + Codec（在 FSDP/优化器构建**之前**）；
- `get_data_and_condition`：`build_pointflow_batch` 构造 PointFlowBatch；
- `_add_noise_to_input`：`pointflow_add_noise` 生成 PointFlowNoised，写入 PackedSequence；
- `denoise`：把 `pointflow_displacement` / `pointflow_sigma` 传给网络；
- `_compute_losses`：`total_loss += fm_point * pointflow_loss_weight`；
- `validation_step`（复用 training_step）与 `sample_pointflow`（供 `PointFlowEvalCallback` 使用）。

判断接线是否在位的快速方法：总损失必须等于各项加权和，例如 `10×0.1748 + 10×0.0207 + 1×1.0011 = 2.9561 ≈ 2.9562`。

## 为什么是 1.00x

`pointflow_loss` 是 `(pred − v_target)²` 在 XYZ 上求均值，`v_target = ε − x_clean`，`ε ~ N(0, 1)` 每个分量方差为 1，三分量平均后仍为 1。用任务 5 manifest 的实测数据（8192 点、32 步、`E|x0|²/3 = 1.53e-4`，位移平均 12.58 mm、p95 41.39 mm）算出的理论下限：

| 模型输出 | loss |
| --- | --- |
| 完全不学（pred = 0） | `1 + E\|x0\|²/3` = **1.000153** |
| 只学平凡关系 `v ≈ x_t/σ`（σ=0.8） | 2.4e-4 |
| 完美预测 | `E\|x0\|²/3` = **1.53e-4** |

观测到的 1.0009~1.0015 就落在"pred = 0"的下限上（略高，因为输出有小的随机偏移）。**裸 MSE 被单位方差的噪声项淹没，小数点后第 4 位才有信息**——这是下面要加 ADE 指标的原因。

另外，本 recipe 的视频 sigma 调度是 `waver, shift=5`，实测分布 `mean = 0.788, p5 = 0.389, p95 = 0.977`，大部分步都高度加噪，进一步压低了 loss 的可读性。

## 改动清单

| 文件 | 改动 |
| --- | --- |
| `configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py:52` | `keys_to_select` 追加 `pointflow_branch`，让整个分支参与训练 |
| 同上 `:62` | `lr_multipliers["pointflow_branch.codec"] = 25.0` |
| `model/generator/omni_mot_model.py` | 恢复 PointFlow 接线（+183/−27），并在 `pointflow_loss` 调用处传入 `scale` |
| `model/generator/pointflow_training.py:29,50` | `pointflow_loss` 新增 `scale` 参数；新增 `pointflow_ade()`（`@torch.no_grad`）返回 `pointflow_ade_mm` / `pointflow_zero_ade_mm` |
| `utils/callback.py:471,496` | 两个新指标进 W&B，并在训练日志里打印 ADE |
| `model/generator/pointflow_training_test.py` | 新增 3 个测试（完美预测 ADE=0、零位移基线、scale 往返、无有效 valid 分支） |

### 为什么给 codec 25×

- Codec 是**从零初始化**的（Sonata 来自 `sonata_small.pth` 预训练，action head 来自 base checkpoint）；其他可训练张量都有预训练权重。
- Adam 每步对每个权重的移动量 ≈ `lr`。目标速度 `v` 的量级是 `√3 ≈ 1.7`，而初始输出 RMS ≈ 0.069，最后一层权重需要从 ~0.06 涨到 ~1；在 base `lr = 2e-5` 下约需 5 万步（9.5 s/iter ≈ 5 天）。
- 同系列 Nano recipe 给 action head 用的是 base `2e-4` × 5 = `1e-3`；本 recipe base 降到 `2e-5`，故 codec 取 25× → `5e-4`，仍在"从零训练的 head"常见区间 `1e-4 ~ 1e-3` 内，且低于 Nano 给 action head 的 `1e-3`。

调参：`PointFlow Loss` 若长时间不降可提到 `50.0`；`Video Loss` 若恶化或抖动则降到 `10.0`（PointFlow 的梯度也会流回共享主干，`clip_norm = 1.0` 一直处于激活状态）。

注意：**调大 `pointflow_loss_weight`（1.0 → 10.0）不会加速**。FusedAdam 对 loss 的常数缩放近似不变（`g → c·g` 时 `m → c·m`、`v → c²·v`，`m̂/√v̂` 不变），而 codec/geometry 的梯度只来自 PointFlow loss。它只会让日志变成 `10.00x`，并让共享主干的梯度方向更偏向 PointFlow。

## 新增指标

`pointflow_ade_mm`：一步去噪估计与真值的毫米级 L2。

```
x̂0 = (x_t − σ · v̂) · scale
pointflow_ade_mm      = mean_valid ‖x̂0 − x0‖ · 1000
pointflow_zero_ade_mm = mean_valid ‖x0‖ · 1000        # "预测完全不动"的基线
```

日志形如：

```
Iteration 300: Total Loss: 12.31 | Video Loss: 0.19 | Action Loss: 0.98
             | PointFlow Loss: 0.8321 | PointFlow ADE: 412.35mm (zero 12.64mm)
```

读法（用真实数据 + 真实 sigma 调度估算）：

| | 量级 |
| --- | --- |
| 零位移基线 `zero` | ~12.6 mm（全点集） |
| 启动时（预测≈随机） | ~1,244 mm |

启动时看到四位数属于正常（`σ·ε` 主导）。判断标准：`ADE` 跌破 `zero` 线才算超过"预测不动"，之后应向 0 收敛。

注意 `v̂ = 0` **不等于** zero 基线：它对应 `x̂0 = x_t`，仍被噪声主导，ADE 远大于基线。zero 基线指的是位移恒为 0 的静止轨迹。

## 验证

- 修复后启动日志：`selected tensors: 322 → 622`，优化器元素 `+42,038,700`；
- `pointflow_training_test.py` + `callback_test.py` 合计 8 passed；
- 用真实的 `_build_params_with_metadata` 冻结逻辑验证：`pointflow_branch.codec` 全部参数被选中且 `requires_grad=True`，codec 有效 lr = 25 × base，Sonata 保持 1×；
- `pointflow_branch_test.py` 在本节点 2 例失败，原因是 `ValueError: Could not find a compatible Attention backend`（走 flash-attn，需要 GPU），与该测试和本次改动无关；
- 节点 pytest 缺少 `pytest-xdist`，root `conftest.py` 无法加载，测试需带 `-o addopts="" --noconftest` 运行。

## 分相位计时（定位 step 时间去向）

`pointflow_profiling.py` 提供 opt-in 的同步计时。**默认关闭**，设 `POINTFLOW_PROFILE=1` 开启；开启后每 20 步在 rank 0 打印一行并按窗口重置：

```
PointFlow phases (iter 20): step_forward 5123.4ms | step_data 1840.2ms | pf_sonata 1610.5ms
                          | step_pack 341.7ms | pf_codec_decode 9.2ms | step_loss 7.1ms
```

已埋点的相位：

| 相位 | 覆盖范围 |
| --- | --- |
| `step_data` | `get_data_and_condition`（含 `pf_build_batch` 与 VAE latent 读取/编码） |
| `step_pack` | `_pack_input_sequence`（text + 各模态打包） |
| `step_noise` | `_add_noise_to_input` + `_replace_clean_with_noised` + `to_cuda` |
| `step_forward` | `denoise`（整个 MoT 前向，含 `pf_sonata` / `pf_codec_encode` / `pf_positions` / `pf_attach_tokens` / `pf_codec_decode`） |
| `step_loss` | `_compute_losses`（含 `pf_loss`） |
| `pf_build_batch` | `build_pointflow_batch` |
| `pf_add_noise` | `pointflow_add_noise` |
| `pf_sonata` | `SonataGeometryEncoder.forward`（PTv3 稀疏编码） |
| `pf_codec_encode` / `pf_codec_decode` | PointFlowCodec 的编码与解码 |
| `pf_positions` / `pf_attach_tokens` | 点位置计算与 token 拼入 packed sequence |
| `pf_dec_inputs` / `pf_dec_blocks` | 解码的输入准备 / 逐 motion block 的 MLP 循环 |

反向路径用**标记（mark）**而不是嵌套相位：`training_step` 结尾打 `begin`，`preds_pointflow` 的 tensor hook 打 `head`，若干参数的 `register_post_accumulate_grad_hook` 打 `trunk` / `point_encode` / `sonata`，训练器调用 `on_after_backward` 时打 `end`。相邻标记的差即该段耗时，在 `on_after_backward` 里折算成相位名：

| 反向相位 | 区间含义 |
| --- | --- |
| `bwd_head` | backward 起点 → 梯度到达 point head 输出 |
| `bwd_trunk` | → 梯度到达第 0 层 q_proj（即整个主干 28 层反向走完） |
| `bwd_point_encode` | → 梯度到达 codec 的 motion encoder（codec 编码侧反向） |
| `bwd_sonata` | → 梯度到达 Sonata stem（PointFlow 分支反向结束） |
| `bwd_end` | → `backward()` 返回（其余 embedding/norm 的梯度） |

标记参数按名字**后缀**匹配（兼容 FSDP 的 `_fsdp_wrapped_module.` 前缀），找不到时只告警不报错。验证阶段复用 `training_step` 但不做反向，因此 `begin` 只在 `self.training` 时打；`grad_accum_iter > 1` 时多个 micro-batch 共用一个 `begin`，该窗口的反向拆分不可用。

计时用 `cuda.synchronize()` 包夹，每步约 30 次同步（~1.5 ms 量级），会轻微扰动测量；定位到问题后建议设 `POINTFLOW_PROFILE=0`。

**参考基线**：同一 recipe 在加入 PointFlow 之前（2026-08-25 的 offline wandb）为 `mfu/avg_time_per_step_s = 5.73`、`mfu/H100 = 0.632`；当前为 ~9.35 s、0.33。而同期 `vision_token_length` 反而从 107,712 降到 83,232，可见回归来自 PointFlow 路径本身，与视频序列无关。

## 已知限制

- **多卡 FSDP 未验证**：若某些 rank 的 batch 没有 PointFlow 样本（`pointflow_data is None`），各 rank 反向图不一致会挂起。当前单卡运行不受影响；上多卡前需要按 action 的写法补一个 `0.0 × sum(params)` 的 dummy loss。
- `sample_pointflow` 每次 eval step 要跑 16 步完整前向，validation 开销不低（可用 `POINTFLOW_EVAL_EVERY` 控制）。
- PointFlow 仍要求 CP=1、two-way attention、无 temporal causal / KV memory；`_add_noise_to_input` 里对 CP 有显式 guard。
- 采样/rollout 的正确性尚未验证，`PointFlowEvalCallback` 产出的是"给定干净 GT 视频/action 的条件拟合诊断"，不是闭环 rollout。
