# Cosmos 本地推理 + 本地部署（单臂）

> 叙事版 quickstart（架构 + 环境准备 + 四次真机故障的现象/根因/修复）见
> [`docs/cosmos_local_quickstart.md`](./cosmos_local_quickstart.md)；本文档侧重
> 配置字段与当前已验证 profile 的操作参考。

把 Cosmos 单臂策略的推理从云端 HTTP 服务改为**本机同机双进程**：推理服务跑在宿主机
（WorldAct checkout 的 uv 环境），ROS 控制链路不变（docker 容器，network_mode: host，
容器内 `127.0.0.1` 即宿主机）。protocol-v2 wire 契约不做任何改动。

```
相机/Tianji/WujiHand 状态
        │
deployment_node (docker) ── pickle/HTTP POST 127.0.0.1:8000/v1/robot-policy ──▶ Cosmos 推理服务 (host)
        │◀────────────── 32 步 joint action chunk @15Hz ──────────────│
        ▼
120Hz 插值 → /right_arm/external_joint_target + /right_hand/joint_commands
```

## 一次性准备

1. **推理环境**（WorldAct-sft-pointflow-fk 仓库根）：

   ```bash
   cd /home/pjlab/ros2_ws/worktrees/WorldAct/WorldAct-sft-pointflow-fk
   uv sync --all-extras --group=cu130-train --no-install-package megatron-core
   ```

   `megatron-core` 是 GitHub git 依赖（代理传输不稳、仅训练用），跳过不影响推理服务。
   若必须补装：重试 git clone 或从能稳定访问 GitHub 的机器拷 wheel。

2. **模型资产**（不在 git 里，需拷贝）：

   | 资产 | 本机位置（约定） | 说明 |
   |---|---|---|
   | 策略 DCP checkpoint | `~/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/<run>/iter_XXXXXXXX` | 需含 `model/.metadata` 与 `net_ema.*` 键 |
   | frozen 训练 config.yaml | 同 run 目录 | 缺省时服务端回退到仓库 experiment 配置重建 |
   | Edge 模型包 | 任意，如 `~/code/ckpt/cosmos3-edge-droid` | tokenizer/processor，约 11G，经 `EDGE_DROID_MODEL_PATH` 引用 |
   | Wan2.2 VAE | `<repo>/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth` | 未预置则首次启动自动从 HF 下载 |

3. **任务 prompt 与 model_id**：部署清单的 `model.task`（必须与训练 prompt 逐字
   一致）和 `deployment.model_id` 要与实际 checkpoint 对应；机器人侧
   `src/wuji_data_pipeline/config/cosmos_protocol_v2.yaml` 的
   `policy_http_expected_model_id` 同步。换任务建议复制一份新清单再改。

## 当前已验证配置（sandwich，16n-1001 / iter_000020000）

2026-10-06 在 RTX 5090 上实测通过（frozen config 重建架构 + EMA 权重）：

| 项 | 值 |
|---|---|
| checkpoint | `WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-16n-1001/iter_000020000` |
| 部署用 config | 同目录 `config.deploy.yaml`（frozen `config.yaml` 的副本，仅改写 `tokenizer_type`×2 与 `vae_path` 三处集群路径为本机路径；原文件不动） |
| Edge 模型包 | `WorldAct/models/cosmos3-edge-droid`（VAE 经 symlink `vae -> cosmos3-edge-droid/vae` 解析） |
| 训练 prompt | `make a sandwich`（逐字） |
| 动作空间 | **EEF**（right_chest 系 xyz + quat xyzw + 20 手关节；wire 上位置 metre、手 degree）。sandwich-100 raw 配方训练即 EEF；`joint` 是 dropper 数据集专用空间，两者不可混用 |
| model_id | `singlerighthand-sandwich-edge-droid-16n-1001-iter-000020000` |
| 服务端清单 | `WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_sandwich_edge_protocol_v2.yaml`（EEF profile） |
| 机器人侧清单 | `src/wuji_data_pipeline/config/cosmos_protocol_v2_sandwich.yaml`（EEF profile，经 `/right_arm/external_target_pose` 执行位姿目标） |

实测数据：启动总耗时 ~33 s（DCP 读盘 17 s + 首次 compile/warmup ~14 s）；常驻
显存 9.65 GiB / 32 GiB；稳态单次推理 ~563 ms、完整 RTT ~735 ms（云端参考
~1019 ms），满足 15 Hz 分块预取 ~1.07 s 的预算。attention 后端 flash2（sm120）。
真机 hold 阶段已通过（25 请求 25 响应零错误，RTT 中位 5.4 ms）。

**事故记录（2026-10-06，small_motion 抖动急停）**：首版 sandwich profile 误沿用
dropper 的 `action_space: joint`，把 EEF 训练的 checkpoint 当关节空间部署——模型
收到关节角当作 EEF xyz+quat，输出垃圾动作（±5°/关节钳制内仍足以危险抖动），
操作员急停，无损伤。修复：两侧清单改回 EEF（机器人侧恢复 `linear_slerp` 插值与
horizon 24 / lead 3-7 的原 EEF 时序参数）。教训：**新 checkpoint 部署前先确认训练
数据的空间**——数据缓存 manifest 的 `arm_action_space` 字段（`eef`/`joint`），
posttrain 文档的 Fixed data contract 一节也会写明。

**故障记录（2026-10-06，TARGET_HOLD 自动停车）**：EEF 真机首跑 ~1.5 s 后
`TARGET_HOLD → DISABLED`。trace 显示第二块 chunk `pending_chunk_activation_failed`
加一条响应过期被弃，命令流断粮，看门狗先保持目标再下电（非碰撞、非模型内容问题，
small_motion 钳制全程有效）。根因：沿用了原云端 EEF profile 的
`open_loop_horizon: 24`——时间对齐预算 = 32 − horizon = 8 步（0.53 s），小于本机
~0.78 s RTT（≈12 步），新 chunk 永远无法对齐激活。修复：horizon 16 / lead 7-9 /
lease 1.06 s（延迟等价的 joint profile 实测值；时序参数只与 RTT 有关，与动作
空间无关）。教训：换部署环境（云→本地）必须按实测 RTT 重算
`horizon ≤ 32 − ceil(RTT_s × 15)`。

**故障记录（2026-10-06，full 模式高频颤动）**：时序修复后 full 模式动作连贯但
"一抖一抖抓不准"。trace 定量：边界处平滑（blend 生效、边界步长 < 块内），
RTT 被对齐机制吸收——病灶是模型 15Hz 路点自身噪声（臂去趋势残余 ±0.55cm、
每 3.5 步反转 ≈ 7.5Hz；手指 69% 步反转、p95 11°）。修复：sandwich 清单启用
仓库既有的 `action_smoothing_method: butterworth`（三阶零相位 Butterworth，
cutoff 3Hz，作用于完整 chunk 的 15Hz 路点，含四元数 rotvec 局部坐标与手 20 维；
单测抖动曲率降 <10%，filtfilt 零相位不加延迟）。注意：只开这一项——PI05 记录
过 Butterworth+PCHIP+velocity_continuous 叠加被判过度平滑回退；dropper 清单
保持 `none`（两条 profile 独立调参）。备选旋钮（未启用）：`COSMOS_NUM_STEPS=6`
（采样质量提升，RTT ~0.98s 仍在预算内）、`COSMOS_GUIDANCE=2.0`、cutoff 改 2Hz、
`pchip_slerp` 插值。若组合手段仍抖，疑似 checkpoint 本身（iter_000020000 未获
训练侧开环评估确认），需训练侧提供验证过的 iter 与推荐采样参数。

**故障记录（2026-10-06，unsafe chunk 拒绝连锁停车）**：butterworth 首跑 ~6 s，
臂加速下潜（z 0.27→0.09 m）后一块 chunk 被客户端安全校验拒绝
（`unsafe right policy position jump`，相邻路点 >0.15 m），唯一在途响应被拒 →
断粮 → `TARGET_HOLD → DISABLED`（看门狗 ~0.7 s 无恢复即下电）。已定性：
服务端 0.15 m 校验放行了该 chunk，但客户端校验失败——无法从日志区分是
"原始 chunk 大跳变（客户端锚点不同）"还是"butterworth 边缘振铃放大"。曾给 `deployment_node.py` 加诊断埋点（纯日志、无行为变更：幅度/waypoint 索引/
raw-vs-post-smoothing 阶段标签）用于区分上述两种可能；**埋点未 commit，当日已
按用户要求 git checkout 回滚，代码恢复原生**。后续运行该拒绝未再出现，按孤例
处理；若复发可重新添加以定性。另记录在案：单次拒绝 ~0.7 s 即连锁下电的鲁棒性
缺口（拒绝后应立即重请求），待评估后改。

合成观测冒烟客户端（不碰真机，hello + 单次推理全链路）：`/tmp/cosmos_synth_smoke.py`，
用 WorldAct venv 的 python 直接跑（服务端 key 为 `test` 时；当前为 EEF 版请求）。

## 启动（一体化）

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
COSMOS_DEPLOYMENT_CONFIG=/home/pjlab/ros2_ws/worktrees/WorldAct/WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_sandwich_edge_protocol_v2.yaml \
./src/scripts/start_local_cosmos_deployment.sh \
  --checkpoint-dir /home/pjlab/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-16n-1001/iter_000020000 \
  --model-config-file /home/pjlab/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-16n-1001/config.deploy.yaml \
  --model-package /home/pjlab/ros2_ws/worktrees/WorldAct/models/cosmos3-edge-droid \
  --service-mode full \
  --config /home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2_sandwich.yaml
```

（`COSMOS_DEPLOYMENT_CONFIG` 环境变量选择服务端清单；`--config` 是容器内的机器人
侧清单路径。跑 dropper profile 时两者都省略即用默认值。）

编排脚本会：生成/校验 `COSMOS_POLICY_API_KEY` → 后台起推理服务（日志在
`<WorldAct>/logs/cosmos_policy_server_*.log`）→ 轮询 `/readyz`（模型加载+warmup
可能要几分钟，默认超时 900s，`COSMOS_SERVER_READY_TIMEOUT_S` 可调）→ 验证容器
连通性 → 启动交互式部署会话（单臂 right）。会话退出时自动停止推理服务。

会话按键沿用现有约定：`r` Recovery、`a` Enable、`x` standby、`q/e/Ctrl+C` 退出。

其他用法：

- 复用已在跑的服务：`--attach-existing --port 8000`
- 只验协议不加载模型：`--service-mode hold`（无需 checkpoint）
- 有界小动作诊断：`--service-mode small_motion`（EEF profile：位置钳制 ±1 cm、
  旋转 ±5°、手 ±5°；joint profile：±5°/关节）

## 分阶段真机验证

1. `hold`：确认 `/wuji_deployment/ready` 成功、`/wuji_deployment/status` 里
   `server_ready: true`、model_id 匹配、机械臂保持不动；
2. `small_motion`：确认方向/单位正确；
3. `full`：完整任务；
4. 每阶段分析 `deployment_trace_*.jsonl`（完整 RTT、server `inference_ms`、
   prefetch hit/miss、时间对齐 skip、边界跳变）。

## 常见问题

- **`torch._C` import 报错**：忘了清 `LD_LIBRARY_PATH`（启动脚本已内置
  `env -u LD_LIBRARY_PATH`，手动跑 python 时注意）。
- **首次启动慢**：torch.compile warmup；若长时间卡住，查日志里 inductor 编译。
- **HF 下载慢**：`export HF_HUB_DISABLE_XET=1`，保持本机代理开启。
- **内网镜像**：pjlab 镜像 `http://pypi.i.h.pjlab.org.cn/brain/dev/+simple/` 是精选
  索引且当前实测需网关认证（407），不要依赖；torch cu130/flash-attn/natten 只能从
  pyproject pin 的 download.pytorch.org / nvidia-cosmos.github.io 拉（走代理）。
- **端口被占**：`ss -tlnp | grep :8000` 找到残留进程后清理。
