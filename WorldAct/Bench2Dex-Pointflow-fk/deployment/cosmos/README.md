# Cosmos Task21：5090 部署开发参考

可复用的是 `backend=training_bundle` 路径，已在 4090 上完成真实 checkpoint 推理及仿真闭环联调。5090 尚未实测。部署代码随本仓库提交；新机器拉取同一分支后按下文准备外部模型文件。

## 代码入口

| 文件 | 用途 |
| --- | --- |
| `policy/Cosmos/bundle_policy.py` | 实际使用的模型封装、当前状态输入、推理计时与显存统计 |
| `policy/Cosmos/contract.py` | 三视角、52 维关节状态及按名字映射的输入约束 |
| `policy/Cosmos/deploy_policy.py` | `get_model` 的 training_bundle 分支及 `LocalSession`；其余 Robolab 路径是早期方案，不是本次 baseline 的验证路径 |
| `script/policy_model_server.py` | 模型进程入口 |
| `script/policy_rpc.py` | 同机 TCP 请求/返回动作块；无需云端 |
| `deployment/cosmos/client_example.py` | 独立客户端示例，读真实观测 NPZ、请求动作、保存 NPY，不驱动机器人 |
| `tools/prepare_cosmos_bundle.py` | 在目标机器重新生成绝对路径配置 |
| `tools/check_cosmos_flash_attention.py` | 在目标 GPU 上核对实际 Flash2 四类算子的数值 |
| `tools/check_cosmos_local_inference.py` | 从真实三视角 HDF5 检查输入或执行模型推理 |
| `run_policy.py`、`tools/run_cosmos_local_sim.sh` | 仿真客户端；运行时需另行准备 IsaacLab 环境及场景资产 |

外部训练配套源码 `source/cosmos_framework/inference/robot_policy/bench2dex.py` 实现实际 batch 构造和关节映射；`adapters.py` 负责模型加载和采样。仓库保存 `policy/Cosmos/bundle_compat.patch`，用于修补原始配套源码；本机源码已经修补，不要重复应用。这不是最新版 NVIDIA 官方部署实现。

## 固定数据接口

- `joint_names`：52 个唯一名字。`joint_action.vector`：对应顺序的当前实测关节角，float32、弧度。输入按名字转到训练顺序，输出转回请求顺序。
- `observation.cam_overhead.rgb`、`cam_wrist_left.rgb`、`cam_wrist_right.rgb`：每路 uint8 `[480,640,3]` RGB；OpenCV BGR 必须转 RGB。应取同一时刻的观测。
- 三视角拼成宽 640、高 720；训练预处理后视频张量为 `[3,33,640,640]`。仅首帧填观测，未来帧占位。
- 动作首行是当前实测 qpos 条件；32 个未来目标为 **52 维绝对关节角**，不是 delta、速度或末端位姿。模型内部 pad 到 64 维不改变机器人动作维度。
- 此 baseline 训练、推理均不归一化；不要额外反归一化。模型适配器已丢弃首行状态，客户端不要再删一行。
- 20 Hz、完整预测 32 步；配置 `execute_steps` 决定返回前几步，当前默认 4。`num_steps=4` 是扩散采样步数，含义不同。
- 实际任务文字来自 `$BUNDLE/metadata/manifest.json` 的 `task_text`；当前 backend 不支持通过每次请求的 `language` 切换任务。
- 默认 EMA 权重、BF16、不量化、关闭 compile；仍联合生成视频 latent，但不解码未来 RGB。该 baseline 不输入 PointFlow/FK。

## 搬到 5090

拉取仓库后，还需要已有的原始 `cosmos_task21_inference_bundle.tar` 中的 `model_assets/`（VAE、tokenizer 等）及训练 checkpoint 的 `model/` 目录。这些外部文件和 Python 环境不提交到 Git；原始配套 tar 同时提供匹配 checkpoint 的训练源码与配置。

原始配套 tar 的 SHA256：
`2e60255c64a7a24d36d94f5d344b18d0919f9db772badfe266ca3662b727e13b`。

将已有原始 tar 解压到 `outputs/cosmos_local/`，得到 `cosmos_task21_inference_bundle/`（包含 `source/`、`training/`、`metadata/`、`model_assets/` 和 `prepare_local.py`）。不要从本机复制 `checkpoint_compat/` 的绝对路径软链接；传原始 checkpoint，在目标机器按需转换元数据。

本机验证环境为 Python 3.11 / Torch 2.7.0+cu128 / FlashAttention 2.8.3。`.venvs/cosmos-local` 继承了 IsaacLab 环境，不能直接搬运。优先复用 5090 现有兼容环境；下述命令不会安装、升级或下载 Torch。4090 的 wheel 测试不能证明 5090 内核可用，先在目标机器跑下面的算子检查，再跑真实模型。

以下从本仓库根目录运行，`PY` 指向目标机器已经准备好的推理 Python，`CKPT` 指向原始 DCP `model/`：

```bash
export PY=/path/to/inference-env/bin/python
export CKPT=/path/to/iter_000020000/model
export BUNDLE="$PWD/outputs/cosmos_local/cosmos_task21_inference_bundle"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# 仅对新解压、未修补的原始源码执行以下两条；dry-run 失败先检查是否已修补。
patch --dry-run -d "$BUNDLE" -p1 < policy/Cosmos/bundle_compat.patch
patch -d "$BUNDLE" -p1 < policy/Cosmos/bundle_compat.patch
"$PY" "$BUNDLE/prepare_local.py"
"$PY" tools/prepare_cosmos_bundle.py --bundle "$BUNDLE" --checkpoint "$CKPT"
"$PY" tools/check_cosmos_flash_attention.py --source "$BUNDLE/source" \
  --output outputs/flash_attention_check.json
```

若加载旧 DCP 时出现 `pathlib._local` / `pathlib._abc` 元数据兼容错误，再执行以下转换并重新生成配置；不修改或复制权重张量：

```bash
"$PY" tools/prepare_cosmos_dcp_compat.py --source "$CKPT" --destination checkpoint_compat/model
"$PY" tools/prepare_cosmos_bundle.py --bundle "$BUNDLE" --checkpoint checkpoint_compat/model
```

配置生成在 `outputs/cosmos_local/deploy_baseline.json`，同目录另有 `baseline_contract.json`。先用真实三视角样本测试单次推理：

```bash
"$PY" tools/check_cosmos_local_inference.py --config outputs/cosmos_local/deploy_baseline.json \
  --hdf5 /path/to/observations.hdf5 --frame 1 --run-model \
  --output outputs/inference_check.json
```

HDF5 需要 `robot/joint_names`、`robot/qpos`、`meta/instruction` 和三路 `cameras/<name>/rgb` JPEG 数据集。首轮加载与稳态推理分开计时；多取真实观测检查动作幅度与连续性。

随后启动模型服务：

```bash
"$PY" script/policy_model_server.py --config outputs/cosmos_local/deploy_baseline.json --host 127.0.0.1 --port 9000
```

观测 NPZ 包含 `qpos`、Unicode 数组 `joint_names` 和以三个相机名命名的 RGB 数组，可用 `np.savez` 写入。在另一个终端执行：

```bash
"$PY" deployment/cosmos/client_example.py --observation /path/to/observation.npz \
  --contract outputs/cosmos_local/baseline_contract.json --manifest "$BUNDLE/metadata/manifest.json" \
  --output outputs/actions.npy
```

如果不需要进程隔离，可直接 `get_model(config)`、`model.reset(seed)`、`model.get_action(observation)`。此路径推理时无需启动仿真环境。

## 真机开发边界与已测结果

现有仿真会暂停物理推进，等模型返回再执行默认 4 步。真实机器人不会暂停时间，因此真机控制器需要独立的固定频率执行线程、带采集时刻的观测和动作块、对过期动作的处理，以及基于实际关节限制的速度/加速度约束和超时停止。当前代码没有实现这些，也没有机器人驱动。不要把模型输出直接当成已经可执行的真机控制策略。

当前 RPC 为简单同步 JSON/base64 TCP，共享一个模型 session，按单客户端使用；没有多机器人 session 隔离、认证或实时调度。它适合参考接口及同机联调。客户端示例不会自动执行动作。

4090 完整联调：1000 控制步、250 次查询，推理中位数约 1.145 秒；1 秒间隔采样的整卡显存峰值约 16.83 GiB（模型＋Isaac）。任务失败、0/3 放置阶段完成，动作存在抖动。因此已验证的是推理/仿真链路，不是策略质量。训练旧 RGB 与本地仿真 anchor 的场景外观不同，正式 benchmark 需对齐配置。本机 `outputs/cosmos_local/closed_loop_episode01/` 保留原始验证报告（不提交到 Git）；不能将这些数值当作 5090 测量。

下一步在 5090 先通过算子检查和单次推理，再测连续请求的延迟分布与动作质量，最后实现异步控制器。增大 `execute_steps` 可以延长单次动作覆盖时间，但必须评估更长开环执行对成功率和抖动的影响。
