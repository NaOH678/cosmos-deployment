# Bench2Dex 单任务 Cosmos Edge 适配

实现范围：UR5/Wuji 双臂双手、52 维绝对关节位置（弧度）、20 Hz、俯视＋左右腕部 RGB、32 步预测、默认执行前 4 步。纯 RGB/state/action 基线，不使用 PointFlow/FK 标签。新增 domain ID 28，在现有 32-domain/64-action 容量内。

## 数据契约

`cosmos_framework/utils/bench2dex_contract.py` 定义唯一顺序：左臂6、左手20、右臂6、右手20。文件中的关节顺序通常交错，分别按 `robot/joint_names`、`action/action_names` 重排。

采集器 `collector/data_collector.py:before_step` 使用 obs_t→commanded_t 约定：当前观测对应即将执行的命令。本实现不移动命令一帧。初始无动作、非有限动作、非正常 sim_step 间隔的过渡会被标记无效，窗口不跨越它们；不删除中间帧。`meta/homing_start_sim_step` 必须存在且非负，所有 >= 此 sim_step 的帧从缓存排除。相机回放与状态的物理同步仍应在正式训练前抽查。

不进行 EEF/四元数变换，不进行弧度转角度，不使用真实数据归一化统计；第一版沿用 joint cache 的无归一化绝对弧度。动作处理记录保留 raw_action_dim=52，由原生 pipeline 补齐到 64，并在生成后还原维度。

三视图使用同一个 CPU `compose_rgb`：俯视在上，左右腕在下，保持宽高比居中黑边；之后训练和推理都调用原生 reflection resize/pad。默认合成内容 640×720，模型 resolution=480；不是将不同相机拉伸到同一宽高。

## 环境

在本 worktree 根目录执行。可以调用原环境 Python，但**显式 `PYTHONPATH=$PWD`**，不要重新安装 editable 包覆盖其他 worktree 的共享环境。GPU 推理依赖 Cosmos；Isaac 仿真在另一环境/机器运行。本文未安装独立 venv。

## 转换

只传入一个 task/robot 的 replay 目录，不要混入 origin 或其他增强副本。需要 HDF5 中有三路 RGB；无 RGB 时先用 Bench2Dex replay 生成。

```bash
cd /mnt/afs/WorldAct-cosmos3-edge-droid-sft-bench2dex
export PYTHONPATH="$PWD"
python tools/prepare_bench2dex.py \
  --source /path/to/task21/robot/replay \
  --output /path/to/new/task21-cosmos-cache \
  --task 21_condiment_box_loading \
  --task-text 'Load the condiment bottles into the box.'
```

输出 manifest.json、video_manifest.json、episodes/*.npz、video_frames/*.npy 和 runtime_joint_names.json。每个源帧顺序写入，支持无 writable mmap 文件系统；输出目录必须不存在，失败目录不能当作完整缓存。NPY RGB 较大，准备充足磁盘空间。

按原始轨迹组确定性划分（seed42，val比例0.1）。episode_000000 与 episode_000000_1 等同编号增强版本使用相同 trajectory_group，保证不跨 split。其他命名方式需显式核对来源关系。需至少足够 episodes 让验证集非空；单个样本只适合 split=full 的 CPU 检查。训练配方不自动开启闭环验证；验证 dataset 可通过同一 factory 的 split=val 构建。

## 训练

```bash
export BENCH2DEX_CACHE_ROOT=/path/to/task21-cosmos-cache
export EDGE_DROID_MODEL_PATH=/path/to/cosmos3-edge-droid
export BASE_CHECKPOINT_PATH=/path/to/cosmos3-edge-droid-dcp
export WAN_VAE_PATH="$EDGE_DROID_MODEL_PATH/vae/Wan2.2_VAE.pth"
export OUTPUT_ROOT=/path/to/runs/bench2dex-task21
export NPROC_PER_NODE=8
python -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/action_policy_bench2dex_edge.toml --dryrun
bash examples/launch_sft_action_policy_bench2dex_edge.sh
```

先 dryrun、2-step 保存/加载 smoke，再完整训练；步数覆盖参数沿用原生 train CLI，避免把未经测试的配置当生产超参。需要 PyTorch 导入修复时使用 `LD_LIBRARY_PATH=''`，TorchCodec 的动态库配置参考原仓库环境文档。

## 推理与仿真 RPC

编辑 `examples/deployment/bench2dex_cosmos.json` 中 checkpoint、manifest 和输出路径。

Bench2Dex `run_policy.py` 已补充在 RPC 观测中发送实际 `joint_names`，与 `_policy_qpos` 的 active/full 顺序对应。Cosmos 必须收到这一字段，否则拒绝推理；按实际名称即时映射，无需假定 Isaac 顺序恒定。`runtime_joint_names` 配置默认 null；若提供文件，则额外检查实际顺序与该文件完全一致。不允许用静态文件静默替代缺失的运行时元数据。

转换器输出的 runtime_joint_names.json 是录制时的顺序，作为审计参考，不代表在当前机器实际启动 Isaac 后的结果。

Cosmos 环境：

```bash
python -m cosmos_framework.scripts.bench2dex_policy_server \
  --bench2dex-root /mnt/afs/Bench2Dex \
  --config examples/deployment/bench2dex_cosmos.json \
  --host 127.0.0.1 --port 9000
```

该入口复用 Bench2Dex 的 DefaultRemotePolicySession 和 PolicyModelServer，不需要修改 Bench2Dex 的 policy 包。模型加载一次，每次收到当前 RGB/qpos，生成32步，只返回 execution_horizon 步。reset 清理本适配器状态；没有跨 episode 的轨迹缓存。只使用当前观测，不读取未来示范或物体真值。

Isaac 环境（同机示例）：

```bash
cd /mnt/afs/Bench2Dex
python run_policy.py --policy-type REMOTE \
  --task scenes/21_condiment_box_loading.yaml \
  --robot-key multi_ur5_wuji_with_flange \
  --remote-host 127.0.0.1 --remote-port 9000 \
  --enable-rgb --headless --seed 100000000 --num-episodes 1
```

跨机将服务绑定地址和 remote-host 改为可达地址；RPC 仅在可信网络使用。保留 benchmark 官方初始化、控制器、成功判定；确认三相机均有效后再增加评测 episodes。当前代码不会把离线预测误差等同闭环成功率。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest --noconftest -c /dev/null \
  -p no:cacheprovider tests/bench2dex_adapter_test.py -q
```

CPU 测试覆盖：关节重排、回零截断、无效动作窗口、同帧动作约定、三视图布局、episode split 与推理输出映射。GPU训练和Isaac闭环需单独实际运行，不能由CPU测试替代。

## 本次实际验证记录（2026-10-03）

- 新增适配测试6项＋原有双手缓存回归3项，共9项通过。
- Edge训练配置 `--dryrun` 成功，解析得到20Hz、52维数据契约、64维模型动作容量、32-domain容量。
- 使用本机任务21原始回放HDF5完成一次低分辨率转换／读取检查：713源帧截到707个回零前观测，674个有效32步窗口。低分辨率缓存仅用于CPU smoke，不应作为正式训练缓存。
- 训练/推理首帧图像、状态补齐、结构化提示内容、SequencePlan一致；推理未来输入为零占位，无未来标签泄漏。
- Python lint、shell语法及git diff空白检查通过。
- 随后完成8卡GPU训练2步及checkpoint恢复后的第3步，详见下方更新。未运行真实checkpoint动作采样或Isaac闭环成功率评测。

本次检查日志：`/tmp/bench2dex_cosmos_tests_final.log`、`/tmp/bench2dex_cosmos_dryrun.log`；这些临时文件不是长期实验产物。


## 并行、可续跑的完整任务准备

`tools/prepare_bench2dex_batch.py` 复用同一转换逻辑，支持 --workers 4。已完成 episode 的 manifest 可复用；未完成目录重命名保留。最后全部完成才发布根 manifest。--expected-files 可传入官方目录列表 JSON，工具等待文件下载后才转换（仅识别 .hdf5，不读取 .part）。正式运行仍需空间容纳未压缩视频。

运行时映射核对结果在 Bench2Dex 的 `analysis/bench2dex_cosmos_setup/joint_order_audit.json`；它区分已录制的运行时顺序与本机实时 Isaac 验证。


## GPU smoke 与完整数据更新（2026-10-03）

8张A800完成2步训练，恢复模型、优化器、scheduler及trainer状态后完成第3步，两次运行退出0。总损失依次17.4657、17.5219、17.3362；动作损失1.5550、1.5302、1.5135。最终保存iter_000000003。训练使用20个episode子集，20Hz、52维补齐至64维、每卡batch=1；尚未评估验证损失或闭环成功率，三步不能说明任务收敛。

完整任务100个文件也已下载、转换并通过CPU验收，共50个轨迹组、63,490帧、54,302个训练窗口及5,888个验证窗口。后续新增适配测试8项通过。详细日志、启动脚本、checkpoint路径及结构化结果统一放在 `/mnt/afs/Bench2Dex/analysis/bench2dex_cosmos_setup/README.md` 和同目录 `training_smoke_result.json`。


## 窗口latent训练接入（2026-10-03）

正式launcher默认读取 `${BENCH2DEX_CACHE_ROOT}/vae_window_latents/window_manifest.json`，可用 `BENCH2DEX_LATENT_ROOT` 指定兼容目录。缓存必须来自Bench2Dex有效窗口生成器；严格校验20Hz、32步、stride=1、resolution=480、VAE权重SHA256、数据清单及state指纹、有效窗口起始帧、NPY头与完整长度。按有效窗口offset读取对应的bfloat16位模式，模型保留原生latent裁边流程并跳过在线编码。推理仍按原生条件帧编码，不依赖训练缓存。

100个episode全部通过结构检查，6个实际训练窗口与在线batch=1编码逐位相同；14项CPU回归测试通过。8卡每卡bs16训练5步并保存checkpoint通过。完整结果见 `/mnt/afs/Bench2Dex/analysis/bench2dex_cosmos_setup/latent_training_result.json`，数值对照见同目录 `latent_integration_validation.json`。此记录更新前文“尚未接入latent”的历史状态。


## 真实checkpoint RPC验证（2026-10-03）

已用5步训练checkpoint的EMA权重、该训练run的config.yaml、本地VAE启动真实推理服务；Bench2Dex原生RemotePolicyClient通过TCP发起3次动作请求，均返回有限4×52动作，reset后重复输出及关节反序映射还原均完全一致。冷启动请求28.34秒，随后1.03/0.98秒。模型生成未mock；未启动Isaac仿真。

部署必须使用 `checkpoint_path=<run>/checkpoints/iter_XXXXXXXXX/model` 和 `config_file=<run>/config.yaml`。入口已补原生init_script初始化。使用训练保存配置避免实验默认配置重新下载VAE、或与实际训练设置不符。RTX机器负责Isaac+Bench2Dex仿真，A800集群负责模型；详细启动与SSH转发说明见 `/mnt/afs/Bench2Dex/analysis/bench2dex_cosmos_setup/RTX_EVALUATION.md`。


## FlashAttention 2性能迁移（2026-10-03）

从真机PointFlow worktree迁入COSMOS_FLASH2_VARLEN开关及验证工具；Bench2Dex训练launcher默认开启，设0可回退。4组bf16变长序列前向/梯度容差检查、有限性检查通过；attention fwd+bwd内核515.1ms→108.6ms（4.74倍）。

仅更换后端、保留1worker时20步训练通过，但稳定步耗时仍约14.5秒。只读采样栈发现主进程等待DataLoader，worker在视频reflection padding，说明CPU准备限制端到端速度。改为每卡4worker、prefetch_factor=2后，8卡bs16的12步训练通过、checkpoint保存成功；第3～11步3.80～3.92秒，中位数3.85秒，相对旧14.47秒约3.76倍。短测不代表长期I/O完全无波动。

正式20k命令：`bash /mnt/afs/Bench2Dex/analysis/bench2dex_cosmos_setup/train_task21_20k_flash2.sh`。每卡16，全局128，累积1，4worker，prefetch2，自动读取latent，20k学习率周期，每500步保存。新OUTPUT_ROOT默认`/mnt/afs/Bench2Dex/outputs/cosmos_task21_bs16_full_20k_flash2`。本次未启动长程训练。旧20k运行日志停在66步，检查时进程已不存在、无checkpoint，不能从66步恢复。测试GPU已释放。结果`flash2_migration_result.json`，数值验证`flash2_validation.log`，训练日志`train_bs16_flash2_w4.log`。

## 统一逐关节归一化（2026-10-05）

新实验与 Bench2Dex/PointFlow-FK 统一使用 **action-only q01/q99 + 0.05 rad尺度下限，不裁剪**。
这替代本仓库此前的action/state联合范围与±5尾部保护。旧统计和旧checkpoint保持原语义；
本次没有切换正在运行的训练，也没有重新生成视频或VAE latent。

### 公式和统计口径

```text
lo = action.q01                         # 第1百分位，逐关节
hi = action.q99                         # 第99百分位，逐关节
offset = (lo + hi) / 2
scale = max((hi - lo) / 2, 0.05 rad)
x_model = (x_radian - offset) / scale
x_radian = x_model * scale + offset
```

仅训练episode的全部`action_valid`帧参与统计，各帧一次，不按滑窗重复计权，也不只取窗口覆盖帧。
缓存转换时已经剔除homing。state统计使用相同有效帧掩码，供诊断，不影响offset/scale。
**条件state与目标action都使用这套action统计参数**，不是分别用state.q01/q99进行缩放。
先处理52维，再补零到64；未来推理占位保持0；生成后通过ActionProcessingRecord反变换回弧度。
不硬裁剪、不按极值扩大scale，因此state/action允许超出[-1,1]或±5；需检查离群值而不是静默修改算法。

统计文件包含`source`、`joint_names`、`action/state.{count,mean,std,q01,q99}`，兼容Bench2Dex读取字段；
另保存schema版本、offset/scale、训练分组与数据指纹，用于本仓库严格校验和可重复恢复。
本仓库v2加载时从action.q01/q99重新计算并校验offset/scale，采用与PointFlow-FK相同的float32运算。

### 新统计与训练入口

新版100条数据池仍按seed=42的轨迹组90训练/10验证划分，不复制PointFlow-FK的8/2统计数值。
统一的是算法与采样口径，不是不同训练集的统计数值。

```bash
export LD_LIBRARY_PATH=''
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/compute_bench2dex_action_stats.py \
  --cache-root /data/shichaojian/raw_data/bench2dex/task21/cosmos-cache \
  --output /data/shichaojian/raw_data/bench2dex/task21/action_stats_q01q99_v2.json
```

工具拒绝覆盖已有输出。统计加载校验关节顺序、split、窗口配置和训练NPZ指纹，验证和部署复用训练统计。
scale下限默认0.05，与另外两处一致；如调整该参数，视作新的实验，不可覆盖运行中的统计。

新20k运行（8卡、每卡16、累积1、Flash2；额外覆盖项直接追加，不再加`--`）：

```bash
export LD_LIBRARY_PATH=''
export BENCH2DEX_CACHE_ROOT=/data/shichaojian/raw_data/bench2dex/task21/cosmos-cache
export EDGE_DROID_MODEL_PATH=/data/shichaojian/models/cosmos3-edge-droid
export BASE_CHECKPOINT_PATH=/data/shichaojian/models/cosmos3-edge-droid-dcp
export BENCH2DEX_ACTION_STATS_PATH=/data/shichaojian/raw_data/bench2dex/task21/action_stats_q01q99_v2.json
export OUTPUT_ROOT=/data/shichaojian/runs/bench2dex/cosmos_task21_bs16_actionq01q99_20k_flash2
bash examples/launch_sft_action_policy_bench2dex_normalized.sh
```

启动器复制统计至`$OUTPUT_ROOT/action_stats.json`并将路径和SHA256固定到config.yaml。
新运行要求v2统计；已有运行只有hash一致才允许恢复。不要给旧输出目录换统计后继续训练。

### 旧模型兼容与部署

- 原`cosmos_task21_bs16_full_20k_flash2`：不归一化，保持原始弧度输入输出。
- 旧`action_stats.json`（schema v1）：action/state联合范围+极值/5保护，仅供旧归一化模型恢复/推理。
  旧运行副本`/data/shichaojian/runs/bench2dex/cosmos_task21_bs16_normalized_20k_flash2/action_stats.json`未改动。
- 新`action_stats_q01q99_v2.json`：action-only分位数，无极值保护。配新实验目录，从base DCP训练。

Bench2DexPolicy从对应checkpoint的config.yaml读取统计契约。部署JSON可设置
`"action_stats_path": "/local/action_stats_q01q99_v2.json"`重定位，但必须与配置中SHA256匹配。
自行写推理代码也必须使用同一action参数处理state，并把action反变换后再发给机器人。
PointFlow/FK的位移std缩放是独立机制，此次未修改。
