# 任务 11：单卡 PointFlow 训练与 eval 可视化

已提供一条单卡 overfit 入口：

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled MAX_ITER=100 \
  bash examples/launch_pointflow_train_1gpu.sh
```

启动脚本固定 `NPROC_PER_NODE=1`，复用 Edge-DROID action recipe，并默认启用 validation start 和每次 validation 的 PointFlow callback。`MAX_ITER`、`OUTPUT_ROOT`、`POINTFLOW_MANIFEST`、`POINTFLOW_SONATA_CHECKPOINT`、`POINTFLOW_VIDEO_DECODER` 可覆盖。

训练 recipe 现在把任务 5 的 mixed manifest 传入 single-right-hand dataset，自动从数据中的 dense episode 读取 PointFlow；video cache 存在时仍可使用缓存。`POINTFLOW_SONATA_CHECKPOINT` 设定后，模型 materialize 阶段自动挂载 Sonata + PointFlow Codec；没有该变量时保持原 Cosmos recipe。

训练每一步执行 Cosmos video/action RF 和 PointFlow RF。PointFlow 的输出 `preds_pointflow` 放入 output batch，`PointFlowEvalCallback` 在 rank 0、W&B 已启动时记录：

- `pointflow/val_loss`：valid 点的 masked XYZ flow loss；
- `pointflow/valid_count`：有效监督数量；
- `pointflow/val_trajectory`：最多 128 个点的三维预测速度轨迹图。

Cosmos trainer 的 `validation_step` 复用同一 data/noise/model/loss 路径，因此 validation 使用与训练一致的 PointFlow mask 和 batch 对齐。callback 不改变优化器状态。

W&B 训练：

```bash
WANDB_MODE=online WANDB_API_KEY=... MAX_ITER=1000 \
  CUDA_VISIBLE_DEVICES=0 bash examples/launch_pointflow_train_1gpu.sh
```

单卡建议先用 1–2 个 episode、`MAX_ITER=100` 验证 loss 是否下降，再扩大数据。任务 11 仍是 overfit/训练工程验证；没有把 CP、temporal-causal、KV memory、CFG rollout 或 multi-GPU FSDP 宣称为已验证。

验证结果：PointFlow RF 专项测试 2/2 通过；任务 7 网络/分支回归测试 13/13 通过；Ruff 与 `git diff --check` 通过。当前工作节点的 `.venv` 缺少 `hydra`，因此无法在此节点执行 `train --dryrun`；实际启动前请在 Cosmos training 环境安装完整训练依赖（按 `cosmos3-post-training` 的 `cu130-train`/`cu128-train` 组）。

实现位置：PointFlow RF 在 `model/generator/pointflow_training.py`；训练接线在 `model/generator/omni_mot_model.py`；W&B callback 在 `callbacks/pointflow_eval.py`；单卡入口在 `examples/launch_pointflow_train_1gpu.sh`。
