# Cosmos Deployment

将机器人控制、Cosmos WAM 推理、vLLM-Omni 加速和 PointFlow/FK 研究代码放在同一个仓库中，统一维护部署链路和实验结论。主要应用为 Tianji 机械臂 + Wuji 灵巧手的视觉闭环操作。

这是 **2026-10-07 本地工作源码的统一快照**，包含各项目当时尚未单独提交的修改；原项目目录和历史保持不变。这里没有合并各仓库的 Git 历史，也没有打包权重、Python 环境或真机录像。新仓库不是安装完成即可直接驱动机器人的镜像。

## 目录结构

```text
cosmos-deployment/
├── README.md
├── cosmos-deployment.code-workspace   # VS Code 统一工作区
├── docs/
│   ├── IMPORT.md                     # 导入范围、来源和复现边界
│   ├── EXTERNAL_DEPENDENCIES.md       # 权重、模型组件、驱动和环境
│   └── source_snapshot.json           # 源文件 SHA256、来源版本、排除清单
├── wuji-hand-teleop-pipeline/
│   ├── src/scripts/                  # 本地/云端启动、测速及诊断入口
│   ├── src/wuji_data_pipeline/        # ROS 数据管线、动作调度、配置、测试
│   ├── src/input_devices/            # 遥操作输入设备集成
│   ├── src/output_devices/           # 机械臂/灵巧手输出设备集成
│   ├── src/wuji-retargeting/          # 重定向代码快照
│   ├── sim/                          # 仿真相关代码
│   ├── docker/                       # ROS 容器和构建配置
│   ├── patches/                      # 外部 wujihandros2 的本地功能补丁
│   └── docs/                         # 部署指南和实验过程
└── WorldAct/
    ├── COSMOS_DEPLOYMENT_MAP.md       # 跨组件入口和调用关系
    ├── WorldAct-sft/                 # 当前 50k 原生推理、Omni HTTP 适配
    ├── WorldAct-sft-pointflow-fk/     # 服务启动器、部署 YAML、联合去噪扩展
    ├── WorldAct-sft_base/            # 保留的 SFT 基础工作树快照
    ├── WorldAct-bench2dex/            # Bench2Dex 相关 Cosmos 工作树快照
    ├── Bench2Dex-Pointflow-fk/        # 仿真、策略、数据处理和 PointFlow/FK 工具
    ├── omni-wam-lab/                 # WAM 加速、导出、profile、固定 Omni 源码
    ├── models/cosmos3-edge-droid/    # 模型包转换代码和配置；权重另外准备
    └── model_ch/real_ckpt/           # 训练/部署 frozen YAML；不含 checkpoint
```

这些 SFT 目录是独立代码变体，暂时保留目录名，避免错误合并训练与推理实现。它们现在均由本仓库管理，不再作为嵌套 Git 工作树。

## 当前部署如何调用

```mermaid
flowchart TD
  A[teleop: start_local_cosmos_deployment.sh] --> B[pointflow-fk: start_cosmos_local_policy_server.sh]
  B --> C[sft: protocol-v2 / native / omni_http]
  C --> D[omni-wam-lab: patched vLLM-Omni + WAM GEN graph]
  A --> E[teleop: ROS deployment_node]
  E <-->|HTTP 观测 / 动作块| C
  E --> F[async + smoothstep overlap blend]
  F --> G[Tianji / Wuji 控制接口]
```

50k 启动器默认使用 `WorldAct-sft` 的完整推理源码。`WorldAct-sft-pointflow-fk` 提供当前服务启动脚本及配置；命令引用它的 YAML **不表示启用了 PointFlow/FK 联合去噪**。Omni 的普通 video + action WAM 路线已经接入；PointFlow/FK 的 Omni 联合去噪仍需另行开发和验证。

重点代码：

| 功能 | 位置 |
|---|---|
| 启动参数、后端选择、会话记录 | [本地启动器](wuji-hand-teleop-pipeline/src/scripts/start_local_cosmos_deployment.sh) |
| 服务配置、Python 和源码目录选择 | [服务启动器](WorldAct/WorldAct-sft-pointflow-fk/script/start_cosmos_local_policy_server.sh) |
| 原生与 Omni 服务实现 | [robot_policy](WorldAct/WorldAct-sft/cosmos_framework/inference/robot_policy/) |
| 观测时钟、异步调度与混合 | [wuji_data_pipeline](wuji-hand-teleop-pipeline/src/wuji_data_pipeline/wuji_data_pipeline/) |
| CUDA Graph 与后端优化 | [omni-wam-lab](WorldAct/omni-wam-lab/) |
| 当前客户端配置 | [async_blend.yaml](wuji-hand-teleop-pipeline/src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_async_blend.yaml) |

## 当前保留的实验配置

- 单张 RTX 5090；50k-retrain-v1-0831 的 `iter_000040000`、EMA 权重。
- Omni：FP8 linear、cuDNN attention、串行 CFG、GEN CUDA Graph；UniPC 4 步、guidance 3、shift 5。
- 每块 32 个动作、15 Hz 模型时间；ROS 以 120 Hz 插值发布。120 Hz 不是模型推理频率。
- `async + blend`：按观测时间推进，stride 16；重叠区使用旧块剩余轨迹与新块混合，smoothstep 权重，手臂/手指同权。
- 顺序执行、固定跳步、8/2 等旧实验 YAML 仍保留用于回归，不应把它们的参数混用到当前方案。

两次真机 trace 的中位数对照：

| 相同 smoothstep 衔接方式 | Omni | 原生 |
|---|---:|---:|
| 服务端推理 | 411.10 ms | 590.75 ms |
| 客户端 RTT | 434.66 ms | 607.13 ms |
| 超过 100 ms 的命令发布间隔 | 0 | 0 |

这是指定会话的测量，不是所有场景的速度保证，也不表示 FP8 与原生数值等价。现场对 Omni 该轮反馈为“感觉不错”；不据此推导任务成功率。详细边界与会话编号见 [加速和异步衔接实验记录](wuji-hand-teleop-pipeline/docs/cosmos_5090_optimization_experiments_20261007.md)。

## 准备环境与启动

1. 按 [外部依赖说明](docs/EXTERNAL_DEPENDENCIES.md) 准备权重、VAE/tokenizer、Omni 导出模型、Python 环境与硬件驱动。
2. 按 [teleop README](wuji-hand-teleop-pipeline/README.md) 和 [部署 Quickstart](wuji-hand-teleop-pipeline/docs/cosmos_local_quickstart.md) 配置 ROS 容器与相机。容器必须挂载本仓库的 `wuji-hand-teleop-pipeline/src`。
3. 打开根目录 `cosmos-deployment.code-workspace` 可以同时看到所有组件。各子项目依赖环境不同，不能在根目录统一执行一次 `pip install` 代替它们的安装。

下面是完成环境准备后的 Omni 启动命令。路径按仓库内的相对布局书写；导入的 frozen 配置及部分旧研究脚本仍可能含原机器绝对路径，迁移前需核对。

```bash
cd wuji-hand-teleop-pipeline
COSMOS_DEPLOYMENT_CONFIG=../WorldAct/WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_50k_retrain_v1_edge_protocol_v2.yaml \
./src/scripts/start_local_cosmos_deployment.sh \
  --backend omni \
  --port 18006 \
  --record-video \
  --checkpoint-dir ../WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/iter_000040000 \
  --model-config-file ../WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/config.deploy.yaml \
  --model-package ../WorldAct/models/cosmos3-edge-droid \
  --inference-repo ../WorldAct/WorldAct-sft \
  --omni-root ../WorldAct/omni-wam-lab \
  --omni-model ../WorldAct/omni-wam-lab/artifacts/4w-ema-omni \
  --service-mode full \
  --config /home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_async_blend.yaml
```

添加 `--check-only` 可先检查路径与服务导入，不启动 GPU 推理、HTTP、Docker、ROS 或硬件；它不替代模型加载和真机验证。服务就绪后仍由操作者按 `r`/`a` 执行恢复和使能，`q` 退出。

原生对照使用 `--backend native`，去掉 `--omni-*` 和 `--record-video`，其他模型及客户端配置保持一致。不要同时启动两个占用相同 GPU／端口或控制同一机器人的会话。

## 原生 WAM 推理说明（2026-10-29 补充）

本节只说明原生 PyTorch 的 WAM 推理，供部署和移植到其他加速卡使用；Omni、PointFlow/FK 不在本节范围内。

### 1. 用到的目录

| 目录 | 作用 |
|---|---|
| `WorldAct/WorldAct-sft-pointflow-fk/cosmos_framework/` | 推理源码（下文简写为 `cf/`） |
| `WorldAct/WorldAct-sft-pointflow-fk/examples/deployment/*.yaml` | 服务端配置（相机、动作空间、采样参数） |
| `WorldAct/WorldAct-sft-pointflow-fk/script/start_cosmos_local_policy_server.sh` | 服务端启动脚本 |
| `wuji-hand-teleop-pipeline/` | ROS 客户端：采集观测、HTTP 调服务、执行动作块 |

调用链：ROS `deployment_node` 每次把头部 + 右腕两张图和 27 维右手状态通过 HTTP 发给服务端，服务端返回 32 步动作；客户端按观测时间异步执行，重叠区平滑混合。

### 2. 推理入口

一次推理就是一次函数调用：

```python
samples = model.generate_samples_from_batch(batch, guidance=3.0, seed=[0], num_steps=4, shift=5.0)
action  = samples["action"][0]          # [33, 27]，第 0 行是状态，取 [1:] 得到 32 步
```

- 模型加载：`cf/inference/inference.py::OmniInference.create(setup_args)`，读 DCP checkpoint 和训练时冻结的 `config.yaml`，用 EMA 权重。
- batch 构建：`cf/inference/robot_policy/adapters.py::_build_batch`（官方对应实现 `cf/inference/action.py::build_action_batch`）。
- 推理主体：`cf/model/generator/omni_mot_model.py::generate_samples_from_batch`。
- 服务封装：`cf/inference/robot_policy/server.py`（HTTP），`adapters.py::_infer_native`。

### 3. 输入 / 输出

| 名称 | 形状 / 内容 |
|---|---|
| 视频 | `[3, 33, 736, 544]` uint8。第 0 帧：腕部 480×640 在上、头部 480×640 在下拼接，等比缩放后反射填充到 736×544；第 1～32 帧全 0 |
| 动作 | `[33, 64]` fp32。第 0 行前 27 维是状态（右手 EEF 位置 3 + 四元数 xyzw 4 + 手指关节 20），其余全 0 |
| 文本 | JSON 格式 prompt，由 `ActionPromptJsonFormatter` 生成 |
| 其他 | `domain_id`、`conditioning_fps=15`、`image_size`、`sequence_plan`（mode=`wam`） |
| 输出 | `samples["action"][0]`：`[33, 27]`；视频 latent 不需要解码 |

32 步动作对应 15 Hz 下 2.13 s。

#### 归一化 checkpoint

如果训练 `config.yaml` 里有 `dataloader_train...singlerighthand.dataset.action_stats_path`，这是归一化训练的 checkpoint，部署端必须：
- 状态进模型前做 `(x - offset) / scale`；
- 输出动作做 `x * scale + offset`；四元数 4 维不变。

实现在 `cf/inference/robot_policy/action_normalization.py`，adapter 自动加载，会校验 `action_stats_sha256`。没有这个字段的 checkpoint 不做归一化。

### 4. 计算流程

1. **文本**：tokenizer 转成 token。文本 K/V 在 4 个去噪步之间缓存复用（`mot/inference_text_kv_memory.py`）。
2. **VAE 编码**：Wan2.2 VAE（`cf/model/generator/tokenizers/wan2pt2_vae_4x16x16.py`），时间 4×、空间 16× 压缩，48 通道。33 帧 → 9 个 latent，每个 46×34。只有第 0 个 latent 是条件，其余 8 个从噪声开始。
3. **组装序列**：latent 按 2×2 patchify，每个 391 token，共 3519 个视频 token；加 33 个动作 token 和文本 token 拼成一条序列，3D mRoPE 位置编码。
4. **去噪**：UniPC（`diffusion/samplers/fm_solvers_unipc.py`），4 步，shift=5.0。每步有条件/无条件各一次前向（CFG，guidance=3.0），共 8 次前向；guidance=1.0 时只需 4 次。视频和动作共用同一个 σ；第 0 个视频 latent 和第 0 个动作 token 保持干净。
5. **输出**：取动作 token，（如需）反归一化，得到 32 步动作。

### 5. 网络结构

主干 Nemotron-2B-Dense-VL，按 Mixture-of-Transformers 改造（`cf/model/generator/mot/unified_mot.py`）：
- 28 层，hidden 2048，16 个 query 头 / 8 个 KV 头（GQA），head_dim 128，MLP 9216，BF16。
- 每层两套权重：und 处理文本，gen 处理视频和动作。
- 注意力 `two_way`（`mot/attention.py::two_way_attention`）：文本之间因果；视频/动作 token 看全部 token，无 mask。batch=1 走 dense 分支，不需要 varlen。
- 动作输入/输出投影是 `DomainAwareLinear`（`mot/domain_aware_linear.py`），按 `domain_id` 选权重。
- VAE 内的注意力直接用 `F.scaled_dot_product_attention`。

### 6. 移植到其他加速卡要改的地方

| 位置 | 现状 | 建议 |
|---|---|---|
| 注意力后端 `cf/model/attention/backends.py` | 按 CUDA 架构号在 cudnn / flash2 / flash3 / natten 中选，没有纯 PyTorch 兜底 | 新增一个 SDPA 或厂商算子的后端：支持 GQA（16/8）、因果与非因果、BSHD 布局，只需 dense 路径 |
| torch.compile | 推理默认开（`inference/common/args.py`：`use_torch_compile=True`） | 先关：setup 参数传 `use_torch_compile=False`（需加到 `OmniSetupOverrides`） |
| CUDA Graph | 默认关 | 保持关 |
| 写死的 `cuda` | adapter 里的 `device="cuda"`、`torch.cuda.synchronize()`；`omni_mot_model` 里的 `packed_sequence.to_cuda()` | 改成厂商 device，或用厂商 torch 插件映射 |
| 分布式初始化 | 单进程也起 world_size=1 的 nccl 进程组（`adapters.py::_maybe_init_distributed`） | 换厂商通信后端或 gloo |
| BF16 | 全模型 BF16 | 需要 BF16 矩阵乘 |
| VAE | conv3d、groupnorm、SDPA | 一般不用改 |

推理路径上没有自定义 triton kernel。

### 7. 数值对齐

1. 在 5090 上 dump 一次推理的输入（视频、动作、prompt、seed）和输出（每步 velocity、最终动作）。
2. 新卡加载同一份输入，比较：第 0 帧 VAE latent、第 1 步有条件 velocity、最终动作。
3. 5090 自己跑两次动作最大差就有 0.5～0.7，新卡差异在这个量级即可。逐层排查先比注意力输出。

参考性能：5090，guidance 3.0、4 步，单次推理中位数约 591 ms，最好控制在 1 s 内。

### 8. 在 5090 上启动（参考）

checkpoint、`action_stats.json`、基座模型 `cosmos3-edge-droid`（tokenizer + VAE）需另外拷贝。训练产出的 `config.yaml` 复制为 `config.deploy.yaml`，把 `load_path`、`tokenizer_type`、`action_stats_path` 改成本机路径（sha256 不改）。

```bash
cd wuji-hand-teleop-pipeline
COSMOS_DEPLOYMENT_CONFIG=../WorldAct/WorldAct-sft-pointflow-fk/examples/deployment/cosmos_sandwich_normalized_94train_20k_protocol_v2.yaml \
./src/scripts/start_local_cosmos_deployment.sh \
  --backend native \
  --port 18006 \
  --checkpoint-dir <ckpt_root>/sandwich_normalized_94train/iter_000020000 \
  --model-config-file <ckpt_root>/sandwich_normalized_94train/config.deploy.yaml \
  --model-package ../WorldAct/models/cosmos3-edge-droid \
  --inference-repo ../WorldAct/WorldAct-sft-pointflow-fk \
  --service-mode full \
  --config /home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_async_blend.yaml
```

`--check-only` 只检查路径和导入，不加载模型。环境准备见 [外部依赖说明](docs/EXTERNAL_DEPENDENCIES.md)。

## 文档与维护

- [完整优化／实验记录](wuji-hand-teleop-pipeline/docs/cosmos_5090_optimization_experiments_20261007.md)：5090 后端优化、论文 async+blend、实测对照。
- [部署 Quickstart](wuji-hand-teleop-pipeline/docs/cosmos_local_quickstart.md)：操作入口及历史配置说明。
- [跨组件导航](WorldAct/COSMOS_DEPLOYMENT_MAP.md)：服务和控制的实际职责。
- [导入记录](docs/IMPORT.md)：包含/排除范围、来源版本和迁移限制。

后续功能按 feature 提交；涉及推理与控制的同一修改可在一个提交中原子更新。不要在这些源码目录内重新初始化 Git。运行记录、视频和大模型继续放在外部存储；提交实验结论及必要的小型统计报告，并写明会话 ID、配置和代码版本。

各组件和第三方代码沿用其原有 LICENSE、NOTICE 和 ATTRIBUTIONS。统一仓库不改变其许可或权重的使用条款。
