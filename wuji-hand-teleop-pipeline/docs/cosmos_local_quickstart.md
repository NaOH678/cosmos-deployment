# Cosmos 本地推理 + 本地控制 Quickstart（单臂 sandwich）

> 本文档记录把 Cosmos 单臂策略从"云端推理 + 本地控制"迁移为"**本地推理 + 本地控制**"
> 的完整方案：架构、环境准备、启动方法、实测基准，以及部署过程中遇到的四个真实
> 故障的现象/证据/根因/修复。面向后续要在真机上复现或接手维护的人。
>
> 适用 checkpoint：`singlerighthand-edge-droid-16n-1001/iter_000020000`（EEF 空间，
> 任务 prompt `make a sandwich`）。运行环境：RTX 5090（32G）、docker 容器
> `wuji-hand-teleop`（ROS 控制链路）。日期：2026-10-06。

## 当前配置与完整实验记录（2026-10-07）

当前保留：**单卡5090、50k/40000、Omni、async+blend、stride16、smoothstep臂手同权**。原生后端也已在相同衔接配置下真机运行且无断流。完整优化过程、采纳／未采纳项、论文适配、各会话结论、证据路径、启动和回退见：

**[Cosmos 5090 推理加速与异步衔接实验记录](cosmos_5090_optimization_experiments_20261007.md)**。

上方16n/20000简介及下方早期架构属于历史部署，不能作为当前50k/40000参数依据。以下实现章节保留开发时背景，真机结果以新实验记录为准。

## 2026-10-07：async+blend 权重单变量实验（smoothstep，已完成首轮真机）

首轮线性配置会话 `20261007T112817Z-54fbff90`：用户反馈停顿缓解很多，仍有一点块状感，未明确报告任务是否成功。34 次激活（含首块）、33 次交接，未记录断流或激活拒绝；最长 ROS 发布间隔 17.66 ms，推理耗时中位数 412.77 ms。详细分析保存在该会话的 `analysis_async_blend_first_run.json`。

当前仍使用同一个 `cosmos_protocol_v2_50k_retrain_async_blend.yaml`，仅新增并设置 `paper_async_weight_curve: smoothstep`。重叠区新权重为 `3*u*u - 2*u*u*u`，旧权重为 `1 - 新权重`；u 是该重叠区归一化时间。臂手共同权重，姿态 SLERP。32 步预测、15 Hz、stride16、时间对齐、受保护端点及推理配置不变，启动命令不变，重新启动部署客户端后生效。

回退线性只需将同一 YAML 的 `paper_async_weight_curve` 改为 `linear`；代码缺省值仍为 linear，确保历史配置语义不变。`paper_async_activate.weight_curve` 和 `pending_chunk_activate.blend_method` 会记录实际曲线，状态信息也包含该配置。

该曲线是本地对照实验，不是论文指定的唯一公式。历史轨迹局部反事实计算中，交接处目标速度变化 P95 从 0.219 降至 0.153 m/s，但不模拟后续闭环，也不保证整条轨迹速度连续。此次 92 项离线回归通过，包括两种曲线的发布回调、相同时间轴与端点、臂手同权。随后用户完成 smoothstep 首轮真机（114022Z），反馈“感觉不错”；原生对照（115439Z）也已完成，详见上方实验记录。

以下章节记录首版线性实现的设计；当前曲线选择以上述配置为准。

## 2026-10-07：论文 async+blend 对照实现（线性版本已完成真机）

依据 [World Action Models in Real Time，第 3–4 节](https://arxiv.org/pdf/2608.01880) 的输出动作重叠加权方法实现。这里只改变客户端调度和块间衔接，服务端权重、Omni 优化、采样与视频录制沿用当前部署。没有修改去噪过程，也不是 RTC。

新配置：`src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_async_blend.yaml`。原来的 `cosmos_protocol_v2_50k_retrain_single24_sequential_blend8_hand2.yaml` 保留作为顺序对照；不新增启动脚本。

### 时间轴与动作融合

- 保持本模型 32 个未来动作、15 Hz；完整保留预测尾段，不再先截成 24 步。
- `paper_async_stride_steps: 16` 定义相邻请求的观测时间原点间隔目标。下一次请求最早为上次观测原点加 `16/15` 秒，仍限制一个在途请求。它不是“激活以后再执行 16 步”，也不是“剩下 16 步才预取”；迟到时不补发积压请求。
- 首块保持原来从第一个动作开始执行的启动方式，首块的请求时间表以执行原点建立。后续块按相机观测原点映射到同一单调时钟；服务端去掉条件状态行后，返回第 0 行对应 `观测原点 + 1/15 秒`。
- 新块返回后保留正在插值到的旧端点，在旧路点的时间上采样新预测，再对剩余重叠轨迹加权。旧轨迹使用实际已安装、可能已经混合过的计划。旧权重从受保护端点的 1 线性降到重叠尾端的 0；位置和手指使用同一权重，姿态使用 SLERP。之后执行新预测的非重叠尾段。
- 原来的固定跳 6 步、固定锚点臂 8／手 2 混合、客户端滤波、额外 settle、空间匹配和 early-splice 均不参与这个模式。延迟通过本次观测到交接时间计算，并非新的固定 skip 参数。超时耗尽旧计划时等待，不外推动作。

### 与论文一致的部分及明确的实现选择

对应论文的 **async+blend**：异步请求、按时间对齐、对最终动作的重叠部分加权，不更改训练或去噪。论文没有给出该基线完整代码或唯一权重曲线，因此不能称为逐行复现。

以下是本地适配，实验记录必须保留：模型仍为 H=32、15 Hz，s=16 是本地请求间隔选择，并非论文 H=24、10 Hz、s=4 的原参数；线性曲线、姿态 SLERP、保护正在插值端点、首块启动方式也是实现选择。多相机使用最旧有效图像作为保守时间原点，记录各源时间及偏差，这并不表示相机与机械臂状态完全同步。该模式不支持仿真时钟；缺失、过期或异常相机时标会拒绝请求，而不是默默退回请求构建时间。

### 启动与核查

在 `wuji-hand-teleop-pipeline` 目录沿用现有命令，仅选择新客户端配置：

```bash
COSMOS_DEPLOYMENT_CONFIG=../WorldAct/WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_50k_retrain_v1_edge_protocol_v2.yaml \
./src/scripts/start_local_cosmos_deployment.sh \
  --backend omni \
  --port 18006 \
  --record-video \
  --checkpoint-dir ../WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/iter_000040000 \
  --model-config-file ../WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/config.deploy.yaml \
  --config /home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_async_blend.yaml
```

trace 新增 `paper_async_request_clock`、`paper_async_activate`、`paper_async_activation_rejected` 和队列耗尽时一次性记录的 `paper_async_underrun`。检查观测原点、请求触发迟到量、camera-to-handoff 延迟、实际重叠长度、逐步权重、新动作采样索引，以及旧／新／安装轨迹。`pending_chunk_activate` 继续保留，`blend_method=moving_overlap_linear` 区别于历史固定锚点混合。

验证：新时间轴／观测时钟测试与旧协议、插值、trace 回归共 **90 项通过**（`test_paper_async_blend.py`、`test_paper_observation_clock.py`、`test_async_splice_independent.py`、`test_deployment_protocol.py`、`test_deployment_trace.py`、`test_deployment_splice_analysis.py`、`test_deployment_state_trace.py`）。包括实际发布回调离线模拟、保护端点连续性、过期会话清理、容量超限前置拒绝和观测时间倒退检查。宿主环境缺少 `wujihand_msgs`，未运行完整 `test_deployment.py`，不等于全项目测试通过。

离线测试只能确认时间轴、队列和混合实现。随后用户完成线性、smoothstep及原生后端真机对照，结果已记录；抓取成功率仍需重复实验，不能从离线测试或一次正面反馈推出。

## 2026-10-07：Omni 推理实现优化

本节补充当前 50k/40000、单卡 Omni 部署；下文早期 sandwich 的参数与结论属于历史记录，不能作为当前配置依据。

当前 Omni 链路接入三项实现优化：只构建真实首帧的 CPU packet、在 CPU 入口验证 domain 以减少重复 GPU 同步，以及完整 28 层 GEN 网络的 CUDA Graph。仍使用串行 CFG、4 步采样和原动作后处理，没有采用实验中的合批 CFG。

离线组合实验（5 份真实观测、ABBA 四阶段、每种方案 40 个正式请求，关闭观测与视频录制）：完整 HTTP 中位延迟 501.21 → 494.50 ms，P95 513.16 → 506.75 ms；60 对对应返回动作逐元素一致。最坏样本未改善，不代表已解决控制衔接或验证真机任务成功率。证据见 `WorldAct/omni-wam-lab/combined_optimization/full_graph_combo/report.json`。

这些优化由现有 `--backend omni` 入口默认启用，无需新增启动脚本。需要诊断回退时，`WAM_FULL_GEN_GRAPH=0` 关闭整段图，`WAM_CPU_FIRST_FRAME_PACKET=0` 恢复原 CPU packet 构建；两者不更改动作调度。已经运行的服务不会自动更新，`--attach-existing` 仍连接旧进程，需要重启推理服务才能加载修改。

50k 服务配置的预热尺寸为 head `[480, 640, 3]`、right_wrist `[480, 848, 3]`，来源是最近实际观测。更改相机分辨率后需要同步检查预热配置；未预热的新尺寸可能在首请求重新捕获图。

---

## 1. 架构

同机双进程，protocol-v2 wire 契约**不做任何改动**，控制端复用既有 `deployment_node`：

```
相机 / Tianji 臂状态 / WujiHand 状态
        │
进程2: deployment_node (docker, network_mode: host)
        │  pickle/HTTP POST http://127.0.0.1:8000/v1/robot-policy
        │  （hello 握手 → 15Hz 观测 → 预取下一块）
        ▼
进程1: Cosmos 推理服务 (宿主机 uv venv, RTX 5090)
        │  返回 32 步 EEF 动作块 @15Hz
        │  （右臂 xyz+quat xyzw right_chest 系 + 右手 20 维 degree，左侧实测保持）
        ▼
Butterworth 路点滤波 → 时间对齐(skip≈12) → 边界 blend → 120Hz 插值
        ▼
/right_arm/external_target_pose + /right_hand/joint_commands
```

关键点：

- 容器 `network_mode: host`，容器内 `127.0.0.1:8000` 即宿主机服务（已实测连通）。
- 推理服务是 WorldAct 仓库**原生** protocol-v2 服务
  （`cosmos_framework/scripts/action_policy_server_protocol_v2.py` +
  `inference/robot_policy/{adapters,server,protocol,config}.py`），一行未改。
- 控制端（同步采集、异步预取、120Hz 执行、过期处理、关节限位、推理超时）全部
  沿用 pipeline 仓库既有 `deployment_node.py`，仅加了 3 处日志埋点（见故障 4）。
- Bench2Dex 的 training_bundle / Robolab / TCP RPC 路径是 52D 双臂仿真专用，
  **不适用于本部署**，不要从那里起步。

### 文件清单（本次新建/修改）

推理服务侧（`WorldAct/WorldAct-sft-pointflow-fk/`）：

| 文件 | 性质 | 作用 |
|---|---|---|
| `script/start_cosmos_local_policy_server.sh` | 新建 | 推理服务启动脚本（参数校验、`env -u LD_LIBRARY_PATH` 起服务） |
| `examples/deployment/cosmos_singlerighthand_sandwich_edge_protocol_v2.yaml` | 新建 | sandwich 服务端清单（EEF profile） |

控制/编排侧（`wuji-hand-teleop-pipeline/`）：

| 文件 | 性质 | 作用 |
|---|---|---|
| `src/scripts/start_local_cosmos_deployment.sh` | 新建 | 一体化编排：起服务→等 `/readyz`→容器连通检查→交互会话，退出自动清理 |
| `src/wuji_data_pipeline/config/cosmos_protocol_v2_sandwich.yaml` | 新建 | sandwich 机器人侧清单（EEF、时序、butterworth） |
| `src/wuji_data_pipeline/wuji_data_pipeline/deployment_node.py` | **零改动**（曾加 3 处日志埋点，当日按用户要求 git checkout 回滚，见故障 4） |
| `docs/cosmos_local_deployment.md` | 新建 | 操作参考（配置字段细节） |

模型资产（不在 git）：

| 资产 | 本机位置 | 说明 |
|---|---|---|
| DCP checkpoint | `WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-16n-1001/iter_000020000` | 20G，`model/.metadata` + `net_ema.*` 键（EMA 权重用于推理） |
| frozen config | 同目录 `config.yaml`（原件不动）+ `config.deploy.yaml`（副本） | 副本仅改写 3 处集群路径（`tokenizer_type`×2、`vae_path`）为本机路径 |
| Edge 模型包 | `WorldAct/models/cosmos3-edge-droid` | 只需 5 个 tokenizer/processor 小文件 + `vae/Wan2.2_VAE.pth`；**base 权重（~7.6G）推理用不到**（权重全部来自 DCP） |
| VAE symlink | `models/cosmos3-edge-droid/vae -> cosmos3-edge-droid/vae` | 兼容 `$EDGE_DROID_MODEL_PATH/vae/` 路径约定 |

**dropper 相关的既有文件（`cosmos_protocol_v2.yaml`、WorldAct 侧 dropper 清单）
一行未动——sandwich 与 dropper 是两条独立 profile，各自调参。**

---

## 2. 一次性准备

### 2.1 推理环境（WorldAct-sft-pointflow-fk 仓库根）

```bash
cd /home/pjlab/ros2_ws/worktrees/WorldAct/WorldAct-sft-pointflow-fk
uv sync --all-extras --group=cu130-train --no-install-package megatron-core
```

- RTX 5090（sm120）必须走 `cu130` 组。
- `megatron-core` 是 GitHub git 依赖（`pyproject.toml:345` pin），代理 clone 不稳
  且**仅训练用**——跳过不影响推理；服务端四个模块已验证无 megatron 可 import。
- 冒烟验证：torch 2.10.0+cu130（CUDA 可用）、flash_attn 2.7.4、transformers 4.57.6、cv2。
- 网络：pjlab 内网镜像（10.102.254.2）实测 80 拒连/443 需网关认证（407），不可用；
  本机代理 `127.0.0.1:7897` 可用（pypi/pytorch/HF），HF 加 `HF_HUB_DISABLE_XET=1`。
  **访问内网地址时不要走代理**（curl/wget 加 `--noproxy '*'` 或清 env）。

### 2.2 模型资产（从集群拷贝）

```bash
# 在能访问集群的机器上执行（只拷必需文件，不要拷整个 11G）
DEST=/home/pjlab/ros2_ws/worktrees/WorldAct/models/cosmos3-edge-droid
scp <集群>:/data/shichaojian/models/cosmos3-edge-droid/{processor_config.json,preprocessor_config.json,video_preprocessor_config.json,tokenizer.json,chat_template.jinja} $DEST/
scp -r <集群>:/data/shichaojian/models/cosmos3-edge-droid/vae $DEST/
# frozen config 与 checkpoint（iter 目录整个）
scp -r <集群>:<run>/iter_000020000 /home/pjlab/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-16n-1001/
scp <集群>:<run>/config.yaml /home/pjlab/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-16n-1001/
```

### 2.3 部署前必须向训练侧确认的三件事（血泪教训，见故障 1）

1. **训练数据的动作空间**：数据缓存 manifest 的 `arm_action_space` 字段
   （`eef` / `joint`）。sandwich-100 raw 配方 = EEF（xyz+quat xyzw + 20 手关节）；
   dropper 是 joint。两者都是 27 维，**光看维度分不出来**。
2. **训练 prompt 逐字**：部署清单 `model.task` 必须逐字一致（本任务
   `make a sandwich`）。
3. **该 iter 是否验证过**：开环评估视频/指标、推荐 iter、推荐采样参数
   （steps/guidance/shift）。

---

## 3. 启动

### 3.1 一体化启动（唯一入口）

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

- `COSMOS_DEPLOYMENT_CONFIG`：服务端清单（宿主机路径，环境变量透传）。
- `--config`：机器人侧清单（**容器内**路径）。
- `COSMOS_POLICY_API_KEY` 未设时自动生成随机 key（loopback 专用，两侧共享）。
- 会话按键：`r` Recovery、`a` Enable、`x` standby、`q/e/Ctrl+C` 退出；退出自动停服务。

可选环境变量旋钮（透传到服务端，不改文件）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `COSMOS_NUM_STEPS` | 4 | 采样步数；6 时 RTT ~0.98s 仍在 1.07s 预算内（8 会超） |
| `COSMOS_GUIDANCE` | 3.0 | CFG 强度；2.0 更快可能更稳，但改变策略行为 |
| `COSMOS_SERVICE_MODE` | full | 同 `--service-mode` |
| `COSMOS_SERVER_READY_TIMEOUT_S` | 900 | 就绪等待上限 |

### 3.2 分阶段验证流程（真机）

| 阶段 | `--service-mode` | 加载模型? | 验证什么 |
|---|---|---|---|
| 0 合成冒烟 | full（先单起服务） | 是 | 不碰真机：`/tmp/cosmos_synth_smoke.py` 发合成观测，验 hello/推理/线格式 |
| 1 hold | `hold` | 否（秒起） | ROS 链路+握手+臂保持不动（hold 返回实测位姿） |
| 2 small_motion | `small_motion` | 是 | 方向/单位（EEF 钳制 ±1cm/±5°/手 ±5°；注意**钳制防不住累计漂移**） |
| 3 full | `full` | 是 | 完整任务；清空工作区、急停在手 |

每阶段后分析 trace（见 §6 排障）。

### 3.3 实测基准（RTX 5090）

| 指标 | 值 |
|---|---|
| 服务启动总耗时 | ~33 s（DCP 读盘 17 s + 首次 compile/warmup ~14 s） |
| 常驻显存 | 9.65 GiB / 32 GiB |
| 稳态单次推理 | ~563 ms（UniPC 4 步 @7.8 it/s + VAE/组批开销） |
| 完整 RTT（真机） | 中位 ~720 ms，P95 ~790 ms |
| 15Hz 分块预取 | 可行（预算 ~1.07s/块） |
| attention 后端 | flash2（sm120） |

---

## 4. 故障史：现象 → 证据 → 根因 → 修复

> 四次真机故障，每次都靠 `deployment_trace_*.jsonl` 定量定性。这一章是本文档
> 最有价值的部分——接手者遇到类似现象时可直接对号。

### 故障 1：small_motion 剧烈抖动、险撞物（操作员急停）

- **现象**：EEF checkpoint 用初版 profile（沿用 dropper 的 `action_space: joint`）
  部署，small_motion 阶段臂高频乱抖并持续漂移，差点碰到周边物体，急停。
- **证据/根因**：**训练/部署动作空间不匹配**。sandwich-100 raw 配方训练的是
  EEF 空间（right_chest xyz + quat xyzw + 20 手关节；数据集
  `singlerighthand_raw_dataset.py:356-363` 按缓存 manifest 的 `arm_action_space`
  分支，`tools/prepare_singlerighthand_raw.py:22` 默认 `eef`；posttrain 文档
  "Fixed data contract" 一节也写明 EEF）。部署侧却用 joint 适配器，把 7 维臂
  **关节角**喂给期望 EEF 的模型，输出再被当关节目标执行——±5°/关节钳制内仍是
  垃圾轨迹。
- **修复**：两侧清单改为 EEF（`arm_command_mode`/`action_space: eef`；机器人侧
  `action_interpolation_method: linear_slerp`；服务端 robot_layout 的
  `action_layout` 用 `eef` 键、units 用 `action.eef.position: metre` /
  `action.eef.quaternion: xyzw`）。执行通道随之从关节命令切到
  `/right_arm/external_target_pose`。
- **教训**：**27 维 ≠ 关节空间**（EEF 7+20 与 joint 7+20 同维）。部署新
  checkpoint 前第一件事是确认训练数据空间，不要从别的任务的 profile 抄。

### 故障 2：EEF 首跑 1.5 秒后 TARGET_HOLD → DISABLED（自动停车）

- **现象**：空间修正后首跑，~1.5s 系统自动 `TARGET_HOLD`（保持最后目标）随后
  `DISABLED`，无碰撞、无急停。
- **证据**：trace 显示 `pending_chunk_activation_failed` + 1 条
  `stale_policy_response`；state trace 显示 `right.arm_external_target` 等命令流
  全部 stale——**动作流断粮**，看门狗先保持后下电。
- **根因**：沿用原云端 EEF profile 的 `open_loop_horizon: 24`。时间对齐预算
  = 32 − horizon = 8 步（0.53s），而本机 RTT ~0.78s ≈ 12 步——新 chunk 到达时
  观测已过旧，永远无法对齐激活。（云端时代推理只要 0.22s，时序参数按那个延迟
  调的，直接搬来必然饿死。）
- **修复**：机器人侧时序改为延迟等价实测值：`open_loop_horizon: 16`（预算
  16 步=1.07s > 12 步）、prefetch lead 7-9 初始 8、`max_observation_age_s: 1.06`、
  `boundary_blend_steps: 4`。修复后 26 块连续激活、26 秒零失败。
- **教训**：prefetch 时序参数**只与 RTT 有关，与动作空间无关**。换部署环境必须
  按实测 RTT 重算：`horizon ≤ 32 − ceil(RTT_s × 15)`。

### 故障 3：full 模式"一抖一抖抓不准"

- **现象**：full 模式动作连贯、朝任务方向，但臂和手高频颤动，无法对准抓取。
- **证据**：412 个 dispatch 路点定量分析——**边界处平滑**（blend 生效，边界步长
  中位 0.85cm < 块内 0.97cm）、RTT 被对齐吸收；病灶是模型 15Hz 路点自身噪声：
  臂去趋势残余 ±0.55cm（p90 1.1cm）、每 3.5 步方向反转（≈7.5Hz 奈奎斯特边缘）；
  **手指 69% 的步反转、步幅 p95 11°**。
- **根因**：checkpoint（iter_000020000，2 万步）的原始采样噪声大，服务端
  binomial5 压不住；dropper 那个 30 万步的 checkpoint 就扛得住同样的采样参数。
- **修复**：sandwich 清单启用仓库既有但历代未开的
  `action_smoothing_method: butterworth`——三阶 Butterworth + `sosfiltfilt`
  **零相位**滤波（`deployment_protocol.py:429-558`），作用于完整 32 步 chunk 的
  15Hz 原始路点（臂 xyz + 四元数 rotvec 局部坐标 + 手 20 维），在时间对齐/
  blend/插值之前。单测：交替抖动曲率降至 <10%，**零相位不加控制延迟**；
  执行窗口（skip 12 后取第 12–27 步）避开 filtfilt 两端边缘效应。
  cutoff 3Hz / order 3 沿用默认。
- **后续修正（2026-10-06，频谱实测后推翻噪声模型）**：butterworth 上线后颤动
  无改善。容器内验证滤波器确实在跑（yaml 加载正确、代码 hardlink 同步）；用
  真实轨迹离线复算滤波效果：手反转数 2868→2856 **几乎不变**。频谱分析（去趋势
  + Hanning 窗 FFT）：臂/手的能量 **96-97% 集中在 <1Hz**——"抖动"的本体是
  **0.1-1Hz 慢速游走**（臂 ±~1cm、周期 ~10s 级），与任务动作同频段，
  **任何滤波都无法分离"想动"与"乱动"**；>3Hz 能量仅 ~0.2%（butterworth 可管的
  频段里根本没有病灶）。最终定性：**不是平滑问题、不是延时问题，是策略输出
  质量上限**（iter_000020000 欠训/未验证的候选证据）。butterworth 保留无害
  （臂反转 265→191 略降），cutoff 无需再调。
- **新增待确认（输入一致性）**：部署适配器每个请求仅喂 **1 帧真实图像 + 32 帧
  全黑**（`adapters.py:585-586`），而训练样本是 **33 帧真实序列**——若训练侧
  eval 管线使用全帧历史，部署条件天生不对等，慢速游走可能部分源于输入信息
  缺失。已列入训练侧问题清单。
- **教训**：①只开一项平滑——PI05 记录过叠加平滑被判"过度平滑"回退；
  ②sandwich/dropper 两条 profile 独立调参，此开关 dropper 保持 `none`；
  ③**先测频谱再选滤波器**——交替型高频噪声（单测场景）和低频游走（真实病灶）
  外观相似但疗法完全不同。
- **备选旋钮**（不根治，按需）：`COSMOS_NUM_STEPS=6`（RTT ~0.98s 仍在预算内）、
  `COSMOS_GUIDANCE=2.0`；根治依赖训练侧提供验证过的 iter 与采样参数。

### 故障 4：unsafe chunk 拒绝 → 0.7 秒连锁下电

- **现象**：butterworth 首跑 ~6s，臂从 z=0.27m 加速下潜到 z=0.09m（朝目标物
  果断运动），随后 `TARGET_HOLD → READY → TARGET_HOLD → DISABLED`。
- **证据**：trace 里 `policy_request_failed: "unsafe right policy position jump"`
  （相邻路点 >0.15m，瞬移级离群点——注意 20cm/s 的连续下潜每步才 1.3cm，远低于
  阈值，所以是离群跳变不是"动得太快"）。唯一在途响应被拒 → 断粮 → 看门狗下电。
- **悬疑**：服务端自己也有 0.15m 校验（`adapters.py:_validate_action_chunk`）
  却放行了这块 chunk。两种可能：①模型原始输出瞬移（但与服务端校验结果矛盾，
  除非两侧"前一路点"锚点不同）；②原始合规（13-14cm），butterworth 的 filtfilt
  边缘振铃放大过线。当时的日志无法区分。
- **修复（诊断埋点，当日已回滚）**：曾给 `deployment_node.py` 加三处纯日志增强
  （unsafe 消息带幅度、`waypoint N` 索引、`raw`/`post-smoothing` 阶段标签），
  用于区分"模型离群 chunk"与"滤波器振铃"。埋点未 commit，当日已按用户要求
  `git checkout` 回滚，代码恢复原生——回滚不改变任何行为（仅日志字符串）。
  后续运行该拒绝未再出现，按孤例处理；若复发，可按上述三条重新添加以定性。
- **待办（鲁棒性缺口）**：单块 chunk 被拒后 ~0.7s 即连锁下电，太脆。合理改法
  是 prefetch 被拒后**立即重新请求**而非等下个周期——需动控制循环，待评估实施。
- **教训**：安全校验链（拒绝→保持→下电）工作正常是好事；但要在"安全"和
  "一块坏 chunk 就终止任务"之间加一层重试。

### 附：小坑速查

| 现象 | 原因/解法 |
|---|---|
| 后台服务"无声"死亡（日志干净、exit -1） | 共享机器上被外部 kill（或按指示 `kill <pid>`），不是崩溃；查 `ss -tlnp \| grep :8000` |
| 冒烟客户端 415 | 服务端只收 `Content-Type: application/octet-stream`（pickle body） |
| 观测被拒 `must be a numpy array` / `dtype float32` | 协议校验要求 numpy float32 数组，不是 list |
| 冒烟 401 | B3 自动生成随机 key；直连测试用 `COSMOS_POLICY_API_KEY=test` 起的服务 |
| `torch._C` import 报错 | 残留 `LD_LIBRARY_PATH`；启动脚本已内置 `env -u LD_LIBRARY_PATH` |
| 端口起不来 | `ss -tlnp \| grep :8000` 清残留；B3 会话退出会自动停服务 |

---

## 5. 排障手册（trace 怎么用）

trace 位置（宿主机）：`wuji-hand-teleop-pipeline/datasets/tianji_wuji/diagnostics/`
（容器内 `/home/wuji/datasets/tianji_wuji/diagnostics`），每次会话一对
`deployment_trace_*.jsonl`（策略/调度事件）+ `deployment_state_*.jsonl`
（120Hz 状态快照）。`action_dispatch` 事件含**实际下发的 EEF 目标和手度数**，
是分析运动内容的一手数据。

| 关心什么 | 看哪个事件/字段 |
|---|---|
| 链路是否通 | `policy_request_send` vs `policy_response_ready` 计数、`policy_request_failed` |
| 时序是否健康 | `pending_chunk_activate` 连续性、`pending_chunk_activation_failed`、`stale_policy_response` |
| 延迟 | `complete_rtt_ms`、`server_inference_ms` |
| 边界 | `prefetch_boundary_miss`（miss 多但步长小=可吸收）、边界 vs 块内步长 |
| 抖动 | `action_dispatch` 的 `arm_eef.right` 逐轴方向反转次数、去趋势残余；`hand_deg` 反转率 |
| 安全拒绝 | `policy_request_failed` 的 `error` 字段（埋点后含幅度/waypoint/阶段） |
| 自动停车原因 | state trace 的 `lifecycle` 事件序列 + 前一行快照的 `stale_streams` |

服务端日志：`WorldAct-sft-pointflow-fk/logs/cosmos_policy_server_*.log`（启动参数、
`Robot policy ready` 行确认 model_id/mode/action_space/weights=ema）。
部署日志（容器内）：`/tmp/wuji_cloud_deployment.log`。

---

## 6. 当前状态与待办

**已验证通过**：环境安装、协议自测、合成冒烟、hold 真机、EEF 空间修正、时序修正
（26 块连续激活）、butterworth 启用、安全校验链（拒绝→保持→下电）。

**待办**：

1. 据埋点输出给故障 4 定性（raw = 模型离群 / post-smoothing = 滤波振铃），再决定
   是动模型侧还是滤波器参数。
2. 鲁棒性改进：chunk 被拒后立即重请求（待评估，动 `deployment_node.py` 控制循环）。
3. 问训练侧：iter_000020000 开环评估证据、推荐 iter、推荐采样参数；若离群 chunk
   频繁出现，基本坐实 checkpoint 上限。
4. 离线开环回放验证（训练 episode 对比专家轨迹）需要原始数据——在集群
   （`/data/shichaojian/raw_data/singlerighthand_sandwich_100`），本地只有 FK 标注。
5. `/tmp/cosmos_synth_smoke.py` 冒烟客户端如需保留请移入仓库（/tmp 重启即失）。

**第二条 profile（50k-retrain-v1，2026-10-06 加入）**：前云端生产 checkpoint
`singlerighthand-edge-droid-50k-retrain-v1-0831/iter_000040000` 的独立部署通道——
服务端 `examples/deployment/cosmos_singlerighthand_50k_retrain_v1_edge_protocol_v2.yaml`，
机器人侧 `src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain.yaml`
（model_id `singlerighthand-edge-droid-50k-retrain-v1-iter-000040000`）。使用该 run 自己的
`config.deploy.yaml`，由同目录原始 `config.yaml` 复制，仅将两处 tokenizer 路径和
一处 VAE 路径改为本机模型包位置，原件不动。启动时 `--model-config-file` 会覆盖
服务端清单的 `config_file`，必须传入
`/home/pjlab/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/config.deploy.yaml`。
此前复用 16n-1001 配置时，加载+合成冒烟已预验证
（~30s 就绪、推理 ~566ms、线格式正确）。逐项比对原始配置后，两者整个 model
段仅本机 tokenizer/VAE 路径不同；不能据此把任务表现差异归因于配置不匹配。

**50k 分步修正，第 1 步：预取时序（短会话已验证衔接，任务未完成）**

- 基线 `deployment_trace_20261006_180244_a4b1934d.jsonl`：24 次 chunk 激活、
  23 次边界 miss，完整 RTT 中位约 741ms。原提前量初始 8、上限 9，最多仅覆盖
  600ms；时间对齐成功不代表边界没有断流。
- 50k profile 改为初始 12、自适应范围 6–15，仍以 P99 RTT 加 2 步计算；
  horizon 保持 16，客户端 Butterworth 和 4 步边界混合保持当前值。
- 用该会话的 24 个 RTT 按顺序做离线预算检查，修改后提前量为 12/14，
  超出提前时间的样本从 24/24 降至 0/24。最小名义余量约 10ms，
  不包含未来调度抖动；这不是闭环回放，也不能代替真机验收。
- 下一次会话重点检查 `prefetch_boundary_miss`、`command_publish` 的间隔、
  `pending_chunk_activation_failed` 和 `TARGET_HOLD`。先确认断流消失，
  再单独对照客户端滤波和切块过渡；不同输入、不同执行时序的两次运行不能
  单凭表现差异认定 checkpoint 欠训。

**第 2 步：原始 EEF chunk 使用请求观测作为校验锚点（待真机复验）**

- 19:48 和 19:53 两次会话均前 6 块正常衔接，等待为 0；第 7 次响应拒绝后才断流。
- `deployment_trace_20261006_195313_165c192e.jsonl:953` 明确记录
  `raw policy chunk: waypoint 0` 位置跳变 0.203830m。邻近状态快照显示，这个
  首目标距离请求时实测位姿约 0.0088m，却距离响应时旧命令 0.2038m；原校验
  混用了两个时刻。该拒绝发生在客户端 Butterworth 之前。
- 仅 50k profile 启用 `first_step_anchor_on_measured_pose: true`。滤波前、后
  整块校验均使用请求携带的 EEF/手状态；实际执行窗口激活仍从当前命令检查
  对齐和混合后的轨迹。保留 0.15m/75° 阈值、块内逐步检查和看门狗。
- 原始 chunk 首步和内部真实超限仍会被拒绝；本次修正不解决所有可能的
  离群输出或执行边界过大问题。部署节点测试 85 项通过，尚未真机复验。

**第 3 步：恢复旧版客户端滤波设置并记录完整动作块（待真机复验）**

- `deployment_trace_20261006_200058_62439604.jsonl`：9 块激活，0 次请求拒绝、
  0 次边界 miss，命令发布间隔中位 8.33ms、最大 11.41ms；前两项修正已在
  该短会话生效，但大幅运动仍存在。第 8 块对齐后、混合前差距 21.83cm，
  混合后相邻路点最大 7.47cm，异常在送入 IK 之前已经出现在 EEF 目标中。
- 对比旧会话之前的 `0443070` 与当前源码：EEF 活动控制路径、IK 调用的
  `unit='m'` 与硬件臂控制实现未见对应功能变化；变化主要涉及 Recovery、
  joint 模式交接和手首次使能 ramp。不能把目标大幅运动直接归因于 IK。
- 仅 50k 的 `action_smoothing_method` 恢复为 `none`，保留服务器 binomial5；
  预取、观测锚点和 4 步边界混合保持本轮设置，避免同时改变多个轨迹变量。
- 开启 `diagnostic_policy_chunk_enabled`，`policy_action_chunk` 事件记录完整
  服务端返回动作和本次请求的臂/手状态，以 request_id 关联激活与下发事件。
  `server_output` 已含服务端平滑，不能称为未经处理的模型输出；若开启客户端
  滤波，还会记录 `post_smoothing`。使用有界异步 trace 队列，不记录图像或密钥。
- 记录快照不会被后续处理修改；修正 trace 的 numpy 数组序列化顺序，避免
  多元素数组被 `.item()` 错误处理。部署、协议和 trace 测试合计 110 项通过。
- 本步用于恢复旧版处理条件并建立离线对照，尚未证明能消除大幅切块分歧；
  原始新块与执行轨迹连续性仍待记录验证，不能宣称整个运动问题已修复。

50k 启动入口（先进入部署会话，由操作员执行 Recovery/Enable）：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
COSMOS_DEPLOYMENT_CONFIG=/home/pjlab/ros2_ws/worktrees/WorldAct/WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_50k_retrain_v1_edge_protocol_v2.yaml \
./src/scripts/start_local_cosmos_deployment.sh \
  --checkpoint-dir /home/pjlab/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/iter_000040000 \
  --model-config-file /home/pjlab/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/config.deploy.yaml \
  --model-package /home/pjlab/ros2_ws/worktrees/WorldAct/models/cosmos3-edge-droid \
  --service-mode full \
  --config /home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain.yaml
```

**第 4 步：完整动作离线复算（2026-10-06 20:09 / 20:11 新记录）**

- 两次会话已经确认 `action_smoothing_method: none` 和完整 chunk 记录生效。
  20:11 会话 29 次激活、0 次边界 miss，最长发布间隔约 11.00ms，仍有大步幅。
  因此，关闭客户端 Butterworth 并没有消除切块问题。
- 新增 `src/wuji_data_pipeline/wuji_data_pipeline/deployment_splice_analysis.py`，
  按 generation/request_id 关联服务端返回与激活，使用记录的 skip 截取窗口，
  以边界前实际发布的命令为锚点，调用生产代码 `blend_action_prefix` 复算。
  分开报告服务端块内、客户端处理后、对齐窗口内、实际下发路点间距，
  并对照固定输入下 1/4/8/16 步混合的最大步幅（包含锚点到首点）。
  不访问 ROS、模型服务或硬件；缺失原始动作/滤波快照/锚点时明确报告无法复算。
- 20:09 的 6 块/94 个下发路点及 20:11 的 29 块/457 个下发路点，
  所有已记录路点的复算位置误差均为 0。这里仅验证 EEF 位置，不代表旋转、
  手指或实测机械臂运动的完整回放，也没有完成新旧推理引擎同输入 A/B。

| 20:11 会话 | 对齐窗口内部最大步幅 | 与上一命令的边界差距 | 4 步混合后的实录最大步幅 | 固定输入改为 1 步混合 |
|---|---:|---:|---:|---:|
| 第 26 块 | 1.77cm | 17.10cm | 5.67cm | 17.10cm |
| 第 29 块 | 1.31cm | 14.24cm | 6.26cm | 14.24cm |

第 29 块只执行了前 9 个路点，复算误差仅对照这 9 个已记录路点。
这些样本证明大步幅由跨块差距在短时间内混合追赶产生，不能据此认定模型输出
整体正确，也不能解释新旧预测为何产生这么大的跨块偏差。直接恢复旧版 1 步
混合会放大这些样本的边界跳变；8/16 步的固定输入结果虽更小，却不代表真实
闭环效果，因此本步未修改执行窗口、时间对齐、插值、混合参数或底层控制。

运行离线分析（在仓库根目录，系统 Python 需有 numpy/scipy/opencv）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/wuji_data_pipeline \
  /usr/bin/python3 -m wuji_data_pipeline.deployment_splice_analysis \
  datasets/tianji_wuji/diagnostics/deployment_trace_20261006_201131_a2834822.jsonl \
  --output /tmp/cosmos_splice_201131.json
```

新增分析测试与协议测试合计 29 项通过，包括解析关联、解析式混合预期、
缺失快照、跨 generation 隔离、清流后的锚点失效及不完整下发记录。
后续仍需定位跨块偏差的来源：降低并分解推理延迟、采集同一观测做新旧引擎
离线对照；现有动作记录未包含图像，不能直接用于模型同输入推理对照。

**恢复旧推理实现：50k 通道不再默认导入 PointFlow/FK 工作树**

此前只确认 HTTP 适配层相同，未把完整推理引擎恢复到旧仓库；局部切块统计
不能代替整条链路的代码对照。此次修改实际的源码选择，不调整混合步数：

- `src/scripts/start_local_cosmos_deployment.sh` 检测到 checkpoint 路径包含
  `singlerighthand-edge-droid-50k-retrain-v1-0831` 时，默认选择同级
  `WorldAct/WorldAct-sft` 作为完整推理源码根目录。其他 run 保留原默认选择。
- 模型 Python 环境仍使用 `WorldAct-sft-pointflow-fk/.venv/bin/python`；
  本地启动适配脚本仍位于该工作树。新增 `--inference-repo DIR` 将源码与环境
  分开，并在执行前设置工作目录/PYTHONPATH、核验 `cosmos_framework` 导入位置。
  服务入口、模型、序列打包、注意力和采样代码均随源码根目录切换。
- 上面的 50k 完整启动命令无需改参数；新服务日志应显示
  `Inference repository: .../WorldAct/WorldAct-sft` 及
  `Verified inference import: .../WorldAct/WorldAct-sft/cosmos_framework/__init__.py`。
  可显式添加 `--inference-repo /home/pjlab/ros2_ws/worktrees/WorldAct/WorldAct-sft`。
  `--attach-existing` 不改变已有服务的代码，恢复旧源码需要重新启动服务。
- 在命令末尾添加 `--check-only` 只检查路径和导入来源，不加载模型、不启动
  HTTP/Docker/ROS 或硬件。该检查已用实际 50k 路径通过；旧服务入口 `--help`
  在共享本地环境中通过，未验证 GPU 模型加载与闭环任务效果。当前工具环境
  `nvidia-smi` 无法访问驱动，不能把源码导入检查描述为 GPU 推理验收。

完整对照中的本地控制边界：

- 当前 ROS 源码底座为 `baa6193`（2026-08-28），并非 Kimi 另写了一套 EEF
  控制器；新增本地入口主要改变服务地址、清单及客户端时序/滤波配置。
- 从 `ba2b32f` 取出旧 `deployment_node.py` / `deployment_protocol.py`，与当前
  实现分别运行相同的 7 项离线检查：单右手观测、预取请求门限、对齐窗口、
  边界混合、末路点调度、单右侧发布、120Hz 插值。两组共 14 项通过。
  这是选定 EEF 路径检查，不是整套新旧系统数值等价或真机证明。
- 对比 `ba2b32f`，底层控制节点差异位于 Recovery 和 joint 模式交接；
  右臂工具动力学更新在 `f46e835`（8 月 14 日）已存在，不能因拿更早的
  `0443070` 对照看到差异，就把它误认成此次本地回归新增的参数。
- 仍保留本机延迟适配的 horizon=16 / lead=12–15 和观测时刻校验修正；
  没有将旧云端的 24/7 时间预算直接用于目前约 0.74s 的本地 RTT。
  因此这是恢复旧推理源码的步骤，不宣称全部运行条件已经复原。

启动路由的 6 项测试通过，覆盖 50k 自动选择、其他 run、显式覆盖、
只检查时不执行入口，以及跨目录复用环境时确实执行指定源码。

**H200 云端与 5090 本地的延迟排查**

用户确认旧云端卡为 H200；本机 `/proc/driver/nvidia/gpus/0000:01:00.0/information`
确认 RTX 5090。20:45 切回旧源码后，模型计时中位 588.14ms，旧云端记录
238.99ms，约 2.46 倍。此次采样进度日志稳态约 7.4–7.5 it/s，4 步循环约
0.53–0.54s，占整个模型计时约九成；这是进度日志的近似墙钟值，不是 CUDA
算子级 profiling，不能直接把循环耗时全部归到某个 GPU 算子。

NVIDIA 规格为 H200 4.8TB/s、5090 1.792TB/s 显存带宽，约 2.68 倍。
来源：[H200](https://www.nvidia.com/en-us/data-center/h200/) /
[RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)。
硬件差异是重要候选解释，但带宽比与延迟比接近不证明 bandwidth-bound，
也不能排除算子后端、计算吞吐、CPU 提交开销、频率/功耗限制或其他 GPU 负载。
当前本地日志是 BF16、4 步 UniPC、guidance=3、flash2/sm120；更换源码后
仍约 588ms。没有旧云端逐算子 profile，不能宣称精确解释了全部 350ms。

新增纯推理基准 `src/scripts/profile_cosmos_inference.py`：复用真实 adapter、
同一 checkpoint/配置，固定合成图像和状态，预热 3 次后计时 10 次；可额外
执行一次 torch profiler，记录输入准备、视觉编码、打包、velocity 调用及
CUDA/CPU 算子。额外 profiling 的耗时含埋点开销，和无 profiler 的基准分开；
各阶段为包含子阶段的统计，不可相加。输出不记录 API key 或真实相机图像。

在本机能访问 GPU 的部署终端、仓库根目录执行（无需 Recovery/Enable）：

```bash
env -u LD_LIBRARY_PATH \
  /home/pjlab/ros2_ws/worktrees/WorldAct/WorldAct-sft-pointflow-fk/.venv/bin/python \
  src/scripts/profile_cosmos_inference.py \
  --torch-profile --output-dir /tmp/cosmos-5090-profile
```

输出 `report.json`、`operators_gpu.txt`、`operators_cpu.txt`、`torch_trace.json`。
报告含 GPU 频率/功耗/占用快照、依赖版本、固定输入 hash 和基准中位数。
它不启动 HTTP、Docker、ROS 或机械臂。先结束其他模型服务再测，可避免
另一份模型的显存占用和并发负载污染结果。H200 上同样可运行，路径通过
`--inference-repo/--config/--checkpoint-dir/--model-config-file/--model-package`
显式提供；同输入 hash、同采样参数下的报告才适合做直接对照。

早期受限工具检查中 `/dev/nvidia*` 不可见，PyTorch `cuda_available=false`。后续已通过获准的宿主机执行访问 GPU，并完成下节实测；此段仅保留早期检查背景。
仅已通过脚本语法、CLI 及 `--environment-only` 分支检查，GPU 分支未实际运行；
环境报告位于 `/tmp/cosmos-inference-profile/report.json`。这不代表用户终端的
5090 不可用，也没有证据要求重装驱动。

相关文档：操作参考与配置字段细节见 `docs/cosmos_local_deployment.md`；
部署记录见 `src/record/cosmos/`。


## 2026-10-06：首帧预处理优化与 CUDA Graph 实测

已在本地 `WorldAct-sft` 和云端 `/mnt/afs/WorldAct-cosmos3-edge-droid-sft` 同步首帧预处理优化。
只对真实首帧缩放和反射填充，再在目标分辨率补齐 32 帧零值未来图像。模型仍收到相同的 33 帧、状态条件及 prompt。
9 组相机尺寸/图像内容组合逐元素等价；两边 robot_policy 测试均 30 passed，修改文件 Ruff 检查通过。

同权重、固定合成输入、BF16/EMA、UniPC 4 步、guidance 3、shift 5，各 50 次非 profiler 计时：

| 输入处理到动作返回，中位数 | RTX 5090 | A800-SXM4-80GB 单卡 |
| --- | ---: | ---: |
| 优化前 | 675.5 ms | 693.4 ms |
| 首帧优化，Graph 关闭 | 576.0 ms | 558.5 ms |
| 首帧优化，Graph 开启 | 574.8 ms | 559.8 ms |

此计时不含 HTTP、网络、ROS；不代表任务成功率。CUDA Graph 的单次 profiler 均记录到 264 次 cudaGraphLaunch，但无明显性能收益。
独立实验配置开启 Graph 后，固定输入的最大 EEF 位置差异为 5090 58.7 mm、A800 36.0 mm。
补充重复原输入、水平翻转图像、改变状态、回到原输入的对照：每种模式自身可重复，模式之间差异可复现，变化输入最大约 65.6 mm。
尚未定位 Graph 数值差异的具体原因。实际部署保持 `model.use_cuda_graphs: false`（默认值）；只有独立实验 YAML 为 true。
不能把这次实验视为 Graph 通过正确性验证，也没有调整机器人控制参数。

本机报告位于 `/tmp/cosmos-5090-firstframe-03/report.json`、`/tmp/cosmos-5090-graph-04/report.json`；
云端报告位于 `/data/shichaojian/test_cosmos3-edge-A800/firstframe_a800_03/report.json`、`graph_a800_04/report.json`。
基准脚本现保存 `actions.npy` 供输出核对。所有离线测试已结束，未启动 HTTP 服务或 ROS 控制。


## 2026-10-07：加速与异步接续并行实验

本轮不启动 HTTP/ROS 或真机，不修改原 `cosmos_protocol_v2_50k_retrain.yaml` 默认行为。

### 异步接续实验

新增独立配置 `src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_early_splice.yaml`。
它只比原 50k 配置多 `early_splice_*` 开关和阈值，仍是 15 Hz 动作、120 Hz linear_slerp、原权重和采样。

- 默认 `early_splice_enabled=false`。实验配置显式开启；仅支持 HTTP EEF、预取和观测时间对齐，边界方法为 smoothstep。
- 新结果可在下一模型路点到期时接续，不再必须等待整个执行窗口耗尽。保留正在插值指向的路点；仅替换其后的未来部分。
- `early_splice_bridge_max_steps` 代码默认 0；实验配置为 8。先尝试直接接续，不通过则从 2 步起搜索最短合格 Hermite 过渡，最多 8/15=0.533 秒。
- 过渡精确保留 committed 首点、第 K 步原目标及其后轨迹，只修改 1..K−1。新检查覆盖全部修改邻域及出口；耗尽或启动检查全部混合点及后两个点，并包含历史速度或静止测量锚点。原全轨迹合法性检查仍保留。
- 检查平移段速度、平移速度有限差分、姿态角速度和手指角速度。它是命令筛选，不是连续加速度限制器，也不能证明真机跟踪或任务效果。
- 拒绝候选后保留旧计划，丢弃候选并经过冷却再请求；旧 generation 的迟到结果不能释放新请求槽。首次和队列耗尽后的激活不能绕过检查。
- 队列耗尽仍沿用已有无新命令时的 watchdog/TARGET_HOLD 行为；没有新增主动减速或持续保持发布，不能称已经保证无停顿恢复。
- `alignment_clock` 明确是请求建立时的单调时钟；不是相机曝光时间。trace 增加替换数量、接续检查指标，status 增加拒绝原因与耗尽等待状态。

代码默认的 1 m/s² 有限差分阈值不能被当作已验证生产参数：对历史已发布完整轨迹筛选，旧成功会话 34/34 块、当前会话 8/8 块都有超限。
实验配置使用 0.4 m/s、5 m/s²、120 deg/s 姿态、200 deg/s 手指作为探索值；这些不是硬件安全限值，也不是为让所有旧轨迹通过。
对同样已发布整段目标筛选，当前会话第 7/8 块仍超限；旧会话大量边界也超限。因此真实候选接续、任务效果仍需另行验证。

最终在现有 ROS 依赖容器内合并运行节点、协议、独立反例、FDM 集成、拼接分析及审计测试：148 passed；宿主机隔离启动器测试另有 6 passed。全部不启动 ROS 图或硬件。
反例覆盖 120/15 Hz 时钟、保持 committed 目标、过期/不连续/跨 generation 拒绝、请求冷却、拒绝→耗尽→恢复、启动锚点、混合后段大跳和未修改远端模型段。

### 独立时序审计

新增 `src/scripts/audit_cosmos_async_trace.py`，不导入 ROS、不连接硬件。
在 20261006_204600 会话，预取没有断流，但返回后到计划激活仍等待中位 167.8 ms、最大 308.6 ms；请求观测建立到激活中位约 866 ms。
1013 个完整 READY 状态快照中，外部目标与最新实测位置差 P95 为 11.74 cm、最大 14.80 cm；这是接收时间的最新值配对，不是严格同步物理测量。
原始报告 `/tmp/cosmos-async-baseline-current.json` 和 `/tmp/cosmos-async-baseline-old.json` 保留默认阈值筛选；探索阈值结果另存 `/tmp/cosmos-async-screen-candidate-current.json` 和 `...-old.json`。
历史部署 trace 未保存相机像素，所以它只能做固定历史输入的时序/轨迹审计，不能据此复现完整闭环或任务成功率。

`src/scripts/replay_cosmos_early_candidates.py` 独立重建相同 7 个历史提前接续候选：只保留 committed 硬接新动作时 0/7 通过；启用最短过渡后 5/7 通过，第 2/3/4 块用 2 步，第 5/6 块用 4 步，第 7/8 块在最多 8 步内仍拒绝。
报告 `/tmp/cosmos-early-candidates-current.json` 与 `/tmp/cosmos-early-candidates-bridge-current.json` 分开保存。候选的激活时刻可前移 133 或 267 ms，但首个修改路点还在一个动作周期之后；这是固定原历史计划的反事实，不能当成闭环延迟或成功率实测。
helper 在这 7 候选上预热后重复 20 轮，成功 P95 2.11 ms、拒绝 P95 5.83 ms。此耗时不含完整 120 Hz 回调和争用，不能证明总预算 8.33 ms 有余；正式使用前需要检查回调耗时和发布间隔。报告 `/tmp/cosmos-early-bridge-cpu-timing.json`。
旧 `deployment_splice_analysis.py` 对 early_splice 明确标记不支持普通边界回放，避免把正确保留的旧端点/桥接误报成实现数值错误；原始已发布步幅统计仍可用。

### 采样步数实验

新增 `src/scripts/benchmark_cosmos_acceleration.py`：加载现有 adapter，读取 NPZ 图像/状态，明确指定手指输入单位，对 4/3/2 步分别预热、计时、保存动作和重复性指标。
本轮为 3 份真实 Dropper 录制观测（非原 50k 部署请求），用同一 50k checkpoint、BF16/EMA、guidance=3、shift=5、固定种子，在单张 A800 每设置测 10 次。

| 采样步数 | 输入处理到动作返回中位范围 | 相对同输入 4 步最大位置差（每样本） |
| --- | ---: | ---: |
| 4 | 579–590 ms | 基准 |
| 3 | 459–467 ms | 5.59–6.57 cm |
| 2 | 338–347 ms | 6.20–7.61 cm |

所有设置自身重复输出一致；计时不含 HTTP/网络/ROS。不同步数的动作偏差不直接证明哪条更正确，且观测来自不同任务；没有把默认 4 步改成 2/3 步。
本地报告 `/tmp/cosmos-cfg-diag-06/steps_real_06/report.json`，输入出处 `/tmp/cosmos-real-observations-20261007/manifest.json`。

### 双卡 CFG 诊断

新增 `src/scripts/diagnose_cosmos_cfg.py`，固定输入，保存预处理条件、初始噪声、每轮 CFG 分支速度及采样中间值；张量抓取开销不能用作性能结果。
初始噪声/条件/提示词/时间步一致；compiled 单卡无文本缓存与双卡比较，第一轮条件分支完全一致，无条件分支已出现差异。
关闭 compile、同时关闭文本缓存后，该固定输入下两卡四轮的两个分支和中间噪声全部逐元素一致。因此该案例差异来自 compiled 执行路径，而不是 CFG 通信/组合或初始种子不一致。
报告 `/tmp/cosmos-cfg-diag-06/eager_comparison.json`。这并不证明所有观测下等价；不直接开启双卡生产服务。


补充编译诊断：只将 MoT block 设为 `dynamic=False` 后，第一轮两个分支一致，但第二轮无条件分支仍分歧；在隔离进程统一所有 `torch.compile(..., dynamic=False)` 后也未消除。
因此不能将原因定性为“动态 shape 一个开关”，也没有把静态编译作为修复启用。逐轮结果见 `/tmp/cosmos-cfg-diag-06/allstatic_comparison.json`。
本轮推理生产源码与默认采样/编译配置未更改；新增的是可复现诊断和基准工具。单卡/双卡模型输出等价性的后续定位仍未完成。


## 2026-10-07：真机验证的完整记录包

`start_local_cosmos_deployment.sh` 现在默认启用本轮记录功能，不需要额外传记录开关。原来的 `--config` 可以继续指向基线或 early_splice 实验配置；启动器复制为每次运行独立的配置，只增加记录开关、目录和运行 ID，不改原配置的控制/模型参数。

统一宿主机目录：

```text
datasets/tianji_wuji/diagnostics/cosmos_runs/<UTC时间-唯一ID>/
  manifest.json                 配置来源、源码指纹、checkpoint路径与元数据指纹
  deployment.yaml               本次实际传给ROS的配置
  pipeline_source.yaml          原控制配置快照
  model_config.yaml             frozen config快照
  server_deployment.yaml        原服务配置快照
  server.log / deployment.log   服务日志、会话退出时归档的ROS日志
  gpu.csv                       每秒GPU负载、显存、功耗、温度和频率
  client/                       控制决策trace、120Hz状态/动作trace
  server/run_.../
    manifest.json               覆盖启动参数后的实际模型/采样配置、软件版本与GPU
    00000001.npz                本次实际解码RGB、27D state、后处理前raw_actions
    00000001.json               request/session ID、时长、测量状态及wire输出
    stats.json                 写入/丢弃/错误/关闭与排空状态
  recording_audit.json          退出时自动生成的完整性与请求关联检查
```

配置快照脱敏；不保存API key、认证header或完整环境变量。模型权重不重复复制/逐次全量hash，manifest明确区分checkpoint元数据hash与完整权重hash。
`--check-only` 仍不会生成运行目录或启动服务。`--no-recording` 可关闭此运行包和服务端捕获，原profile已有trace开关仍由profile控制。
录制完整模型需要实际 inference source 含 `robot_policy/recording.py`。当前已接 `WorldAct-sft`；PF工作树未同步，不会静默假称支持。
`--attach-existing` 只能配置本次客户端记录，不能替已经运行的服务开启捕获；hold/small_motion没有真实模型输出，manifest也不会声称存在raw动作记录。

记录范围：

- 服务端每次实际推理：原尺寸RGB uint8（不是有损再编码视频）、native state（EEF xyz米/xyzw/手rad）、模型去掉条件行后的32×27原始动作、后处理wire动作、实际采样配置、decode/batch/model/lock/postprocess时长。失败请求也保存可用输入和失败阶段；非有限raw数组保留在NPZ供定位。
- 控制端：原返回、时间对齐、过渡输入和最终窗口、保留端点、桥接长度、拒绝阶段/原因、请求/会话/generation/chunk身份、传感器源时间戳/年龄、观测构建耗时。
- 控制目标和实测状态：继续120Hz状态/动作trace，记录姿态、关节、手指、生命周期和数据新鲜度；新增ROS发布完成时间/间隔与每秒callback时长汇总。
- 每个组件写入统一recording_run_id；服务与控制端再通过session_id/request_id关联。目录创建或日志存在不代表数据已经录齐。

写盘都使用有界后台队列，满队列或磁盘失败不阻塞控制，但会明确记为丢弃/错误。SIGTERM正常退出会等待在途推理并排空writer；强杀、磁盘故障或drain超时仍可能不完整，由stats/终端summary及audit报告指出。
服务ready后，启动器会检查记录器manifest和初始化stats正常，再启动部署会话；recording初始化失败时不会继续悄悄运行。

录制并非零开销：离线测试接续快照增加约0.9ms中位处理时间；synthetic拒绝路径的接续函数P95约7.13ms，曾见14.44ms最大值，未含整个120Hz回调。因此必须用本次callback_timing和真实发布间隔判断影响，不承诺全程满足8.33ms预算。

退出后也可手动复查：

```bash
/usr/bin/python3 src/scripts/audit_cosmos_recording_bundle.py   datasets/tianji_wuji/diagnostics/cosmos_runs/<本次目录>   --finalized --output /tmp/cosmos_recording_audit.json
```

审计区分运行中、完整、不完整、缺失证据和明确错误；一个推理失败但已完整保存输入/失败信息的请求，不会被误判成录制写盘失败。
本次实现与测试均未启动模型推理或机器人运动。
