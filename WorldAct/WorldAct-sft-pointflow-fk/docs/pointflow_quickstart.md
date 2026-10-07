# PointFlow 开箱使用手册(新集群)

> 面向刚拿到这套代码的机器/人:从 clone 到跑起 pointflow 训练的最短路径。
> 本文档只做索引和关键操作,细节全部链接到现有文档。
>
> 当前工作区：`/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk`。
> 下文保留原pointflow分支的安装与实机入口；仿真实验使用§3的pointflow-fk入口。

## 0. 这是什么

项目目标是缓解video与action预测不一致的问题：从video提取PointFlow，
由action和URDF计算FK，将两种3D模态作为video与action之间的桥梁。
FK每手21个关键点，双手共42个；PointFlow与FK分别编码，通过局部(t,h,w,z)
位置编码建立空间关联，并与video/action联合去噪。

当前主线已在 `pointflow-fk` 工作区接入 video/action/PointFlow/FK 四模态联合去噪，
局部四轴位置编码使用 `partition` + Flash2 varlen，开启torch.compile。
PointFlow支持per-point和cluster；当前仿真实验采用cluster stage2 + skip1/2 + PB4。
实机500点的full/selective、lifted160等历史对照见
[局部四轴实验记录](./pointflow_fk_local_mrope_20261002.md)，不作为所有数据集的默认配方。

### 当前仿真实验（2026-10-05）

先用10条仿真数据（8训练、2验证）检查拟合和联合生成质量，效果通过后扩到100条。
当前保留：8×A800、每卡batch16、累积1、selective AC（Flash2+QKVO）、1024点、
双手FK42点、local RoPE、compile。训练读取完整VAE latent cache及首帧RGB；eval保留33帧。
默认20000步，学习率周期20000、warmup50，开启val on start，每500步eval/save。
验证集每个early/middle/late阶段取4个连续窗口，覆盖6.4秒（拼接MP4为6.45秒）；
两条验证episode共24个窗口，另有2个训练case。每窗独立联合生成，再拼接，非连续自回归。
中途主要根据joint dream生成质量判断是否停止，有效果则训满两万步。

30步短测第6–26步均值5.65s，范围5.60–5.71s；显存抽查约59.6GiB/卡。
这不是长训速度保证。缓存路径的联合eval已在full AC下验证；selective短测验证了
训练与checkpoint。详见[仿真数据与实验记录](./pointflow_sim_v2_data_20261004.md)。

Action保留Bench2Dex的action-only q01/q99归一化，scale下限0.05 rad，无硬截断；
当前统计仅来自8条训练episode。另一仓库WorldAct-bench2dex使用action/state联合
范围并按极值扩大scale，二者不要混用。代码、公式、文件路径和扩容步骤见
[两套action归一化对照](./pointflow_action_normalization_comparison_20261005.md)。

实机几何检查：PointFlow 保留自己的 DA3 内参，FK 沿用已有投影代码；
已核验各自重投影及进入视频 patch 的像素变换。检查范围和结果详见
[pointflow_fk_local_mrope_20261002.md](./pointflow_fk_local_mrope_20261002.md)。
形态对比的全部数据见
[pointflow_form_comparison_observations_20261001.md](./pointflow_form_comparison_observations_20261001.md),
数据管线和实验记录见[文档地图](#文档地图)。

## 1. 拿代码 + 装环境

以下clone命令为原pointflow分支入口；当前已配置的pointflow-fk工作区可直接使用§3仿真命令。

```bash
git clone https://github.com/liuhangxu-robin/WorldAct.git
cd WorldAct && git checkout WorldAct-cosmos3-edge-droid-sft-pointflow
```

- 集群公网慢/不通 → 按 [docs/setup_offline.md](./setup_offline.md):有网机
  `uv sync` 填满 cache → 拷 cache → 离线机 `uv sync --offline --locked`。
- 有网 → 按 [docs/setup.md](./setup.md):`uv sync --all-extras --group=cu130-train`。
- 驱动 CUDA 13.2 兼容 cu130 wheel(驱动 ≥ 构建版本即可)。
- **手动运行Python前**:`source .venv/bin/activate && export LD_LIBRARY_PATH=''`；仿真启动脚本已自动处理。

**pointflow 追加依赖**(sonata 编码器需要,不在 base 环境/lock 里):

```bash
uv pip install --python .venv/bin/python --default-index https://pypi.org/simple \
  --only-binary :all: 'addict==2.4.0' 'spconv-cu126==2.3.8'
uv pip install --python .venv/bin/python --no-deps --no-index \
  --find-links "https://data.pyg.org/whl/torch-$(python -c 'import torch;print(torch.__version__)').html" \
  --only-binary :all: torch-scatter
```

- spconv 没有 cu130 构建,`spconv-cu126` 实测可在 torch 2.10.0+cu130 /
  A800(sm80)上正常跑 GPU kernel(2026-09-28 验证);torch-scatter 按当前
  torch 版本选 PyG 官方 wheel(本集群为 `2.1.2+pt210cu130`)。
- 缺这三个包的典型症状:训练启动即 `ModuleNotFoundError: No module named
  'addict'`(`pointflow_geometry.py` → `auxiliary/sonata`)。

## 2. 数据摆放

以下都不在 git 里,需从旧集群拷贝(大小为参考值):

| 内容 | 旧集群路径 | 谁引用它 |
| ---- | ---------- | -------- |
| 模型包(VAE/tokenizer/transformer) | `models/cosmos3-edge-droid/`(11G) | `EDGE_DROID_MODEL_PATH` |
| 训练基座 DCP | `models/cosmos3-edge-droid-dcp/`(6.3G) | `BASE_CHECKPOINT_PATH` |
| sonata 权重 | `checkpoints/ptv3/sonata_small.pth`(148M) | `POINTFLOW_SONATA_CHECKPOINT` |
| 原始数据 | `raw_data/singlerighthand_{sandwich,dropper}_100/` | `SINGLERIGHTHAND_RAW_ROOT` |
| 缓存(manifest+video+window latents) | `datasets/singlerighthand-*-cosmos-cache/`(每套百 G 级) | `SINGLERIGHTHAND_CACHE_ROOT` |
| pointflow 轨迹(9.24 转换产物) | `pf_out/9.24/<dataset>/labeled/` | manifest 内的绝对路径 |

**路径不一致时的两处修改**:

1. 启动脚本顶部 env(`examples/launch_sft_action_policy_singlerighthand_edge.sh:10-20`)
   全部支持同名环境变量覆盖,不用改脚本;
2. pointflow manifest 里嵌了轨迹的绝对路径——挂载点变了就 sed 换前缀,或按
   [数据 pipeline 步骤 2](./pointflow_data_pipeline.md) 重建。

### 本集群实例(2026-09-28 起,inspur / A800 sm80)

- 仓库:`/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow`(pointflow 分支 checkout);
- 共享 venv:`/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv`(在 base checkout 下,
  其 editable 安装指向 base 代码——**手动跑脚本必须 `cd` 到本仓库并用
  `python -m cosmos_framework.scripts.xxx`**,否则 import 到 base 的旧代码;
  训练启动脚本自身会 `cd` + `-m`,不受影响);
- 数据根:`/data/shichaojian/`,布局与上表一致(`models/`、`checkpoints/ptv3/`、
  `raw_data/`、`datasets/`、`pf_out/9.24/`、`runs/`);
- 9.24 labeled 已转换:sandwich/dropper 各 101 集,`video` 字段已补;
- manifest 已按新路径重建:`pointflow_outputs/sandwich_924_20260928/manifest.json`、
  `pointflow_outputs/dropper_924_20260928/manifest.json`(含本机绝对路径,不入 git)。

本集群启动训练前的 env 覆盖(复制即用):

```bash
export LD_LIBRARY_PATH=''
export PYTHON_BIN=/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python
export EDGE_DROID_MODEL_PATH=/data/shichaojian/models/cosmos3-edge-droid
export BASE_CHECKPOINT_PATH=/data/shichaojian/models/cosmos3-edge-droid-dcp
export POINTFLOW_SONATA_CHECKPOINT=/data/shichaojian/checkpoints/ptv3/sonata_small.pth
# sandwich:log
export SINGLERIGHTHAND_RAW_ROOT=/data/shichaojian/raw_data/singlerighthand_sandwich_100
export SINGLERIGHTHAND_CACHE_ROOT=/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache
export POINTFLOW_MANIFEST=$PWD/pointflow_outputs/sandwich_924_20260928/manifest.json
export OUTPUT_ROOT=/data/shichaojian/runs/<每次必换>
bash examples/launch_pointflow_labeled29_sandwich.sh
# dropper 换:raw_data/singlerighthand_dropper_100、
#   datasets/singlerighthand-dropper-100-cosmos-cache、
#   pointflow_outputs/dropper_924_20260928/manifest.json,
#   启动脚本换 launch_pointflow_dropper101.sh
```

## 3. 启动训练

### 仿真 PointFlow-FK（当前配方）

```bash
OUTPUT_ROOT="/data/shichaojian/runs/sim_v2_stage2_bs16_20k_$(date +%Y%m%d_%H%M%S)" \
EXTRA_TAIL_OVERRIDES='' \
bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/examples/launch_sim_pointfk_v2.sh --background
```

上面的时间戳为新实验生成独立目录；续训时将OUTPUT_ROOT改为原目录。脚本已设置工作目录、共享venv、LD_LIBRARY_PATH、
数据路径及PointFlow/FK尺度，不必另行source环境。该入口固定单机8卡。
同目录重跑会自动resume；更换stage、选点或归一化统计应使用新实验目录。
配置：[TOML](../examples/toml/sft_config/action_policy_sim_pointfk_edge.toml)。

### 实机 sandwich/dropper（原有入口）

```bash
# 本集群(A800/SenseCore)全量 101 集:一切配置已固化,只给 OUTPUT_ROOT
OUTPUT_ROOT=/data/shichaojian/runs/<新目录,每次必换> bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

`launch_pointflow_sandwich101.sh` 固化了两层默认,均可从外部同名 env 覆盖:

- **集群布局**:PYTHON_BIN(共享 venv)、模型/DCP/sonata 权重、sandwich raw+cache、
  101 集 manifest 与 allowlist;
- **训练配置**:per_point 分层语义选点(N=500,手/物/台面=40/45/15,组内 GT 运动
  排名+缺额回补)、幻影守卫(幽灵点占比 7.5%→0.7%)、`POINTFLOW_EVAL_JOINT=true`
  (联合去噪 dream eval)、`COSMOS_FLASH2_VARLEN=1`、expandable_segments、
  batch 16/卡、val-on-start、scale 0.0528(101 集分层重扫)、窗口缓存、
  HSDP(节点内分片、跨节点复制,分片/复制度数自动等于 SenseCore 注入的每节点
  卡数/节点数)。

通用/旧集群的裸启动方式(全 env 手写)见第 2 节"本集群实例";其它wrapper:
`launch_pointflow_labeled29_sandwich.sh`(29 集)、`launch_pointflow_dropper101.sh`(dropper)。

- 脚本开头的 `Checking inputs...` 会核对 TOML、数据集、checkpoint、缓存,缺什么会直接报;
- `OUTPUT_ROOT` **每次显式换**(wrapper 未设置会直接报错);resume = 同一 `OUTPUT_ROOT` 重跑同一命令;
- WANDB 默认 offline;更多开关见脚本内注释和
  [docs/pointflow_data_pipeline.md 步骤 5](./pointflow_data_pipeline.md)。

### 关键 env / override 速查(实机通用入口)

以下为通用配置/历史入口默认值；仿真wrapper的有效配方以本页“当前仿真实验”为准。

| 开关 | 默认 | 作用 |
| ---- | ---- | ---- |
| `POINTFLOW_EVAL_JOINT=true` | false | eval 加跑部署形态联合 rollout:首帧+state action 干净,未来 video/action/point 联合去噪,解码梦视频做画布,产出 `<case>_joint/`(含 dream 拼接)。费显存费时长,第一轮 eval 盯 OOM |
| `POINTFLOW_SIGMA_SCAN=true` | false | eval 诊断:固定 sigma 网格上扫 x0_hat 的 ADE |
| `trainer.run_validation_on_start=true`(EXTRA_TAIL_OVERRIDES) | false | 开局先跑一轮 val(fresh 在 iter 0,resume 在断点 iter);默认等 `trainer.validation_iter`(=100) |
| `model.config.parallelism.data_parallel_shard_degree=8` + `..._replicate_degree=1`(EXTRA_TAIL_OVERRIDES) | 1/-1(纯 DDP) | FSDP 全分片,80GB 卡必须;语义同 DP,全局 batch 不变 |
| `COSMOS_FLASH2_VARLEN=1` | 关 | 解除 flash2 varlen 禁令(sm80 上 varlen 唯一替代是 natten,慢 ~4.8x/kernel、~2.2x/step);数值验证 `tools/check_flash2_varlen.py` |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | 关 | 碎片回收;另:DCP save 前已内置 `empty_cache()`(修偶发 NCCL "unhandled cuda error") |
| `NNODES=2 NODE_RANK=0/1 MASTER_ADDR=<node0>` | 单机 | 多机训练;16 卡配 `shard=8 replicate=2`(HSDP:分片留节点内 NVLink,节点间只梯度 allreduce);shard×replicate×CP 必须等于 WORLD_SIZE。细节见 [pointflow_multinode_training.md](./pointflow_multinode_training.md) |
| `POINTFLOW_SELECT_REGION_QUOTAS` | 空(扁平排名) | 分层语义配额,如 `2:0.40,3:0.45,4:0.15`(手/物/台面);需配合 `SELECT_TOP_N>0`,组内仍按 GT 运动排名,缺额全局回补 |
| `POINTFLOW_SELECT_PHANTOM_GUARD=true` | false | 幻影点(uv 冻结但 3D 漂移)降级到排名末尾;101 集实测选中点幻影占比 7.5%→0.7% |
| `POINTFLOW_WINDOW_CACHE_ROOT` | 空(在线算) | 预计算窗口缓存目录(数据管线步骤 3b);缓存后每样本读 ~120KB 而非 ~205MB |

### resume 语义(查证结论)

- **自动 resume**:同一 `OUTPUT_ROOT` 重跑同一命令 → 从最新 DCP 续训,模型/优化器/调度器/dataloader 状态全恢复;dataloader 有 per-rank pickle 状态,**续训不重复取数**;
- **skip 只有 EMA**:`keys_to_skip_loading=["net_ema."]`(experiment config:146),EMA 从主网络热启动。**pointflow 权重 resume 时完整恢复**;首次从 base DCP 加载时 pointflow 键不存在于 base,走 partial load 保持随机初始化(日志里的 "Skipping loading of key: net_ema.pointflow_branch..." 是 EMA 跳过,属正常);
- **eval case 固定**：普通模式使用`fixed_cases.json`；阶段模式使用`fixed_cases_stages_<N>windows.json`。同名清单会校验case数量与身份。窗口数1→4使用新的清单，可保留旧清单；更换episode数量等导致同名清单不兼容时，应使用新实验目录。
- **改 FSDP 拓扑**(shard 度数)后 resume 旧 checkpoint 由 DCP resharding 支持,但同一 run 内别改。

## 4. 看训练效果

仿真产物：`<OUTPUT_ROOT>/cosmos3_action/action_sft/action_policy_sim_pointfk_edge/pointflow_eval/step_*/`。
直接查看各case的MP4（包括`<case>_joint/dream_canvas/comparison.mp4`），保留现有视频格式。
结合GT视频、GT PointFlow投影、生成PointFlow与FK/action的协调性判断桥接质量；
指标仅作参考。以下为实机历史入口的产物约定：

- eval 产物:`<OUTPUT_ROOT>/cosmos3_action/action_sft/action_policy_singlerighthand_edge/pointflow_eval/step_*/`,
  含逐 case mp4、`validation_stages.html`(浏览器打开看网格);
- 缺拼接视频时离线补:`tools/stitch_pointflow_canvas.py --eval-dir <上述目录>`;
- 指标判据看 eval ADE/zero,跨 scale 的 loss 读数不可比(见数据 pipeline 运维规则);
- 可视化规则与投影约定:[docs/pointflow_eval_visualization.md](./pointflow_eval_visualization.md)。

## 5. 接入新数据 / 换数据集

完整操作手册:[docs/pointflow_data_pipeline.md](./pointflow_data_pipeline.md)
(cache 骨架 → allowlist → **9.24 efep 转换(步骤 1b)** → manifest → window
latents → scale 标定 + 幻影扫描 + 可视化验证 → 训练)。

仿真双手数据的固定窗口例外及已生成的 v2 缓存，见
[Bench2Dex task21 v2 数据准备](./pointflow_sim_v2_data_20261004.md)。

## 6. 历史状态快照（不是实时运行状态）

当前仿真配方见§0和§3，以下保留早期集群与实机实验背景。

### 6.0 2026-09-28：新集群贯通

- sandwich/dropper 的 cosmos 缓存(manifest/video/window latents)**已就绪**;
- 9.24 labeled 转换**已完成**(sandwich/dropper 各 101 集,`video` 字段已补),
  manifest 已按新路径重建(见第 2 节"本集群实例"),旧集群的 manifest 仅
  供查历史扫描结论;
- 冒烟已过:1×A800 3 步训练 loss 正常下降、点云数据路 8/8 样本各 300 点、
  checkpoint 正常保存;sonata 依赖(addict/spconv-cu126/torch-scatter)已在
  共享 venv 装好(第 1 节);
- 主干文档 `docs/pointflow_per_point_tokens_20260919.md` 有全部实验结论与
  待办(幻影守卫、scale 重标定、位置编码、FK 模态)。

### 6.1 2026-10-01：实机实验历史快照

- **当时在跑(16卡×2)**:E1 framescale(`perpoint_strat500_framescale_20260930` @5100,
  cond 平台与 v2 打平,但 val_02_joint 短板治好)、E2 cluster+skip+pb4 全量
  (`cluster_500p_skip_pb4_20260930` @3000,~5000 步见平台);
- **已结束**:v2(per_point 基线,停 @9200,平台化后回退)、B(cluster-origin,
  10000 跑满,等 iter 落后 + 平台更高)、E2s(方案 C 冒烟,停 @6800);
- **全 case 对比结论**(cond/joint/漂移四张表):cond 口径 per-point 全胜且大运动
  case 差距不收敛;joint 口径 case 分化,cluster 在 val_02/04 有固有优势;静态漂移
  是 cluster 系系统病灶。详见
  [pointflow_form_comparison_observations_20261001.md](./pointflow_form_comparison_observations_20261001.md);
- **当时计划**:E3(n4000 密集 cluster)一把定胜负 → point 形态定稿 → 切入
  FK/thwz 实施;dropper 迁移(E5 系列)数据准备可随时起(CPU)。
  队列与启动命令见 [pointflow_experiment_queue_20260930.md](./pointflow_experiment_queue_20260930.md)。

## 7. 排障速查

| 症状 | 看哪里 |
| ---- | ------ |
| `torch._C` import 报错 | 没 `export LD_LIBRARY_PATH=''`([setup.md](./setup.md#pytorch-import-issue)) |
| OOM / NCCL / 训练慢 | [docs/faq.md](./faq.md) |
| push 被 pre-push hook 拦 | 机器缺 git-lfs;conda base 里有,或确认无 LFS 文件后 `git push --no-verify` |
| 训练报 labeled 目录不存在 | manifest 死链,见第 2/6 节 |
| "labeled uv outside the tracker grid" | 相机 rig 变了,仿射常数重测(数据 pipeline 步骤 2) |
| `No module named 'addict'` / `spconv` / `torch_scatter` | sonata 依赖没装,见第 1 节"pointflow 追加依赖" |
| "Cannot open head video: None" | 转换时漏了 `--video`,补 labeled report.json 的 `video` 字段(数据 pipeline 步骤 1b) |
| 手动跑脚本行为怪异/签名对不上 | 用了脚本路径直接跑,被 base checkout 的 editable 安装劫持;改 `python -m` 从本仓库根跑(第 2 节) |

## 文档地图

**新来的 agent 推荐阅读顺序**：本页§0/§3 →
[仿真数据与当前配方](./pointflow_sim_v2_data_20261004.md) →
[action归一化对照](./pointflow_action_normalization_comparison_20261005.md) →
[数据pipeline](./pointflow_data_pipeline.md)。实机历史按需查形态对比与实验队列。

### 入口与状态(先读)

| 文档 | 内容 |
| ---- | ---- |
| [pointflow_sim_v2_data_20261004.md](./pointflow_sim_v2_data_20261004.md) | 当前仿真配方、数据路径、选点、性能与联合eval验证历史 |
| [pointflow_action_normalization_comparison_20261005.md](./pointflow_action_normalization_comparison_20261005.md) | 两套仓库的action统计/公式差异、实际加载文件、扩容约定 |
| [pointflow_form_comparison_observations_20261001.md](./pointflow_form_comparison_observations_20261001.md) | **形态对比观测档案**:per_point/cluster 四形态全 case 数据(cond/joint/漂移)、架构事实(file:line)、未解问题清单、方案池、数据位置 |
| [pointflow_experiment_queue_20260930.md](./pointflow_experiment_queue_20260930.md) | **实验队列**:A/B/E1~E5 状态与启动命令、产物路径、sandwich 共享数据产物 |
| [pointflow_per_point_tokens_20260919.md](./pointflow_per_point_tokens_20260919.md) | 主线实验记录(最长最全):从 300 点验证到 101 集全量的全部结论与误差分析 |

### 操作手册(动手前查)

| 文档 | 内容 |
| ---- | ---- |
| [pointflow_data_pipeline.md](./pointflow_data_pipeline.md) | 数据处理操作手册(接新数据必读):转换、manifest、缓存、仿射常数 |
| [pointflow_multinode_training.md](./pointflow_multinode_training.md) | 多机训练:SenseCore 注入兼容、HSDP 配置、提交命令、排障 |
| [pointflow_eval_visualization.md](./pointflow_eval_visualization.md) | eval 可视化与 stage viewer:dream canvas 投影规则(544x704 真相) |
| [pointflow_selection_visualization_20260920.md](./pointflow_selection_visualization_20260920.md) | 选点动态可视化工具用法(gif/mp4,验证训练所见) |
| [pointflow_window_latent_cache_20260913.md](./pointflow_window_latent_cache_20260913.md) | VAE 窗口 latent 缓存:构建、并发、多卡、注意事项 |
| [docs/setup.md](./setup.md) / [setup_offline.md](./setup_offline.md) | 在线 / 离线环境安装;sonata 追加依赖见本页 §1 |

### 设计与专题分析

| 文档 | 内容 |
| ---- | ---- |
| [pointflow_fk_local_mrope_20261002.md](./pointflow_fk_local_mrope_20261002.md) | **局部四轴 RoPE 方案 A**:共享相机位置、24/16/16/8通道、分组attention、FK接入、CPU/GPU验证与投影审计(§7) |
| [pointflow_position_encoding.md](./pointflow_position_encoding.md) | 原三轴(t,h,w)位置与action–point交互规则设计;实际legacy路径见其核查说明 |
| [pointflow_cluster_decode_20260929.md](./pointflow_cluster_decode_20260929.md) | 簇 token + 点级 decode(skip/pb):方案、实现、env 清单 |
| [pointflow_ptv3_cosmos_design.md](./pointflow_ptv3_cosmos_design.md) | sonata(PTv3)接入 cosmos 的原始设计 |
| [pointflow_displacement_scale_20260914.md](./pointflow_displacement_scale_20260914.md) | scale 根因与标定;§14 per-frame scale(逐帧归一化,env 向量) |
| [pointflow_motion_selection_20260914.md](./pointflow_motion_selection_20260914.md) | 运动选点策略分析(配额、幻影守卫的依据) |
| [pointworld_implementation_20260929.md](./pointworld_implementation_20260929.md) | PointWorld/EgoWAM 点处理机制调研(外部方案对照) |
| [pointflow_fit005_analysis_20260914.md](./pointflow_fit005_analysis_20260914.md) | 早期拟合实验(fit005)误差分析 |

### 历史档案(任务实施记录,按需查)

| 文档 | 内容 |
| ---- | ---- |
| task1~task8 系列(task1_data_interface … task8_training) | 数据接口/几何/codec/序列注意力/数据源/batch/网络/训练的分步实施记录 |
| [pointflow_task11_single_gpu_train_eval.md](./pointflow_task11_single_gpu_train_eval.md) / [pointflow_task12_optimizer_freeze_and_metrics.md](./pointflow_task12_optimizer_freeze_and_metrics.md) | 单卡训练 eval 贯通、优化器冻结与指标 |
| [pointflow_sonata_smoke.md](./pointflow_sonata_smoke.md) / [sonata_environment_validation.md](./sonata_environment_validation.md) | sonata 冒烟与环境验证 |
| [pointflow_alignment_audit_20260913.md](./pointflow_alignment_audit_20260913.md) | 训练/eval 输入对齐审计 |
| [pointflow_dense_fullseq_audit.md](./pointflow_dense_fullseq_audit.md) | 密集全序列数据审计 |
| [pointflow_bugfix_log_20260912.md](./pointflow_bugfix_log_20260912.md) | 早期 bugfix 日志(sigma 解耦、投影等) |

### 仓库通用

| 文档 | 内容 |
| ---- | ---- |
| [AGENTS.md](../AGENTS.md) | 仓库总地图(目录结构、命令、规则) |
| [docs/training.md](./training.md) / [docs/inference.md](./inference.md) / [docs/faq.md](./faq.md) | 通用训练 / 推理 / 排障 |
