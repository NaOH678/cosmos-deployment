# 原生 Cosmos 基线 · 开发机训练说明（单机 8 卡 / 双机 16 卡）

本文档说明如何在开发机上训练**原生（未改架构）Cosmos Edge-DROID single-right-hand 策略**，作为 `_base` 四模态工作区的对照基线。

**原则：除两处有记录的例外外，原生代码零修改。** recipe、batch、callback、checkpoint 节奏、日志与产物格式全部保持仓库原样。变动如下：

- **新增**环境适配 wrapper（`examples/launch_sft_action_policy_singlerighthand_edge_dev.sh`）和本文档。wrapper 不碰训练逻辑，只做三件事：
  1. 把本 worktree 自带的 `.venv/bin`  prepend 到 `PATH` —— 开发机 ambient `python` 是无 torch 的 Conda base，`/usr/local/bin/torchrun` 也不是本环境；
  2. 清空 `LD_LIBRARY_PATH`（见 `docs/setup.md` 的 torch._C gotcha，原 launcher 会自行补回所需前缀）；
  3. 把数据 / 模型 / 输出路径指到开发机的 `/data/shichaojian/...`（原 launcher 内置默认是集群路径，本机不存在）。
- **例外一（flash2 varlen 逃生门）**：`cosmos_framework/model/attention/flash2/checks.py` 移植了与 `_base` 完全相同的 `COSMOS_FLASH2_VARLEN` 开关（~20 行，默认行为不变，仅当 env=1 时解除上游对 flash2 varlen 的禁令）。原因：训练是 varlen sequence packing，A800(sm80) 上原生禁令使 attention 只剩 natten 可用，实测 13.3 s/step；开逃生门后选 flash2，实测 **10.1 s/step（1.32×）**，且与 `_base` 所有 run 的 attention 路径一致，对比更公平。wrapper 默认设 `COSMOS_FLASH2_VARLEN=1`，设 `=0` 即回到纯原生 natten 行为。上游禁令理由是 instability：`_base` 两条线各观测 >1700 步无 NaN，但请盯 loss 形态与 `skip_nan_step` 频率，异常即回退 `=0`。
- **例外二（per-window VAE latent 缓存）**：从 `_base` 移植了 window latent 缓存，共 4 处——dataset 产出（`singlerighthand_raw_dataset.py` 的 `vae_window_latent_root` 参数、manifest 校验、`_read_window_latent`）、`action_sft_dataset.py` 参数透传、实验 config 的两行 env wiring、以及 `omni_mot_model.py` 的消费端（有 `vae_latent_cache` 时跳过冻结 VAE 的在线 encode）。原生默认不设这两个 env，行为与之前完全一致；wrapper 在检测到 `$SINGLERIGHTHAND_CACHE_ROOT/vae_window_latents/window_manifest.json` 时自动开启（显式设 `SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT=""` 关闭）。缓存按窗口各自 encode、与无缓存运行 bit 一致；dataset 会校验 manifest 的 fps/chunk_length/sample_stride，缓存过期会拒绝启动而不是静默读错。重新生成缓存用 `_base` worktree 的 `tools/cache_window_vae_latents.py`（缓存目录两 worktree 共享）。

原 launcher 的所有环境变量开关（`NPROC_PER_NODE` / `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` / `EXTRA_TAIL_OVERRIDES` / 各路径变量）原样透传。

## 路径（开发机布局）

| 用途 | 路径 |
| --- | --- |
| 原始数据（恰好 101 集，与 `_base` 的 101 集 allowlist 同集合） | `/data/shichaojian/raw_data/singlerighthand_sandwich_100` |
| 预处理缓存（manifest + video cache，已就绪） | `/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache` |
| Edge-DROID processor bundle | `/data/shichaojian/models/cosmos3-edge-droid` |
| 基础 DCP checkpoint | `/data/shichaojian/models/cosmos3-edge-droid-dcp` |
| 默认输出根 | `/data/shichaojian/runs/cosmos/singlerighthand-edge-droid` |

硬件：单机 8 × A800-80GB，128 CPU。recipe 原生配置 `data_parallel_shard_degree=1 / replicate=-1`、`max_iter=10000`、`save_iter=500`、wandb offline，均与硬件无关，直接可用。

**唯一的适配性 override（wrapper 内置）**：recipe 原生 `max_samples_per_batch=32` 在本机 OOM（已验证：480p 下首个 backward 前已占 73 GiB / 79.25 GiB），wrapper 默认注入 `dataloader_train.max_samples_per_batch=16` 并开启 `PYTORCH_ALLOC_CONF=expandable_segments:True`。16/rank 恰好与 `_base` 对照 run 的 per-rank batch 一致。该默认值可被显式覆盖——在 `EXTRA_TAIL_OVERRIDES` 里再写一次 `dataloader_train.max_samples_per_batch=<n>` 即生效（同名覆盖后者优先）。

## 单机 8 卡

```bash
cd /mnt/afs/WorldAct-cosmos3-edge-droid-sft
bash examples/launch_sft_action_policy_singlerighthand_edge_dev.sh
```

指定输出目录（建议按日期区分 run）：

```bash
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/singlerighthand-edge-droid-$(date +%m%d) \
  bash examples/launch_sft_action_policy_singlerighthand_edge_dev.sh
```

冒烟（几步验证链路，不影响正式 run）：

```bash
EXTRA_TAIL_OVERRIDES="trainer.max_iter=6 scheduler.cycle_lengths=[6] scheduler.warm_up_steps=[2] checkpoint.save_iter=1000000" \
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/native-baseline-smoke-$(date +%m%d%H%M) \
  bash examples/launch_sft_action_policy_singlerighthand_edge_dev.sh
```

**实测记录（2026-09-30，本机 8 × A800-80GB，16/rank、全局 128）**：三轮冒烟均 exit 0；启动（编译 + DCP 加载）约 3~4 分钟；loss ≈ 14~15 健康，grad norm 跨配置一致（13.1~13.6）；token 长度三种配置完全相同（vision 53856 / action 528，证明 latent 缓存产物形状与在线 encode 一致）；结束自动保存 DCP。稳态步速：

| 配置 | 稳态步速 | 说明 |
| --- | --- | --- |
| flash2 varlen + window latent 缓存（wrapper 默认） | **~3.95 s/step** | 与 `_base` 数据/attention 路径同口径 |
| flash2 varlen，无 latent 缓存 | ~10.1 s/step | 每步在线跑冻结 VAE encode |
| natten，无 latent 缓存（纯原生 auto） | ~13.3 s/step | 上游对 varlen 禁 flash2 后的 fallback |

latent 缓存开关：wrapper 检测到 `$SINGLERIGHTHAND_CACHE_ROOT/vae_window_latents/window_manifest.json` 即自动开启；显式 `SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT=""` 关闭。

- 日志：`$OUTPUT_ROOT/logs/action_policy_singlerighthand_edge_sft.log`
- checkpoint：`$OUTPUT_ROOT/checkpoints/`，每 500 步一份 DCP
- **续训**：用同一个 `OUTPUT_ROOT` 重跑同一命令，自动从最新 DCP resume
- 全局 batch = 16/rank × 8 = 128（dev 默认；原生 recipe 的 32/rank 在本机 OOM，见上节）

## 双机 16 卡

### 前提

1. 两台机器都能访问**相同路径**的代码与数据：`/mnt/afs/WorldAct-cosmos3-edge-droid-sft`（含 `.venv`）和 `/data/shichaojian/{raw_data,datasets,models}`。`/data` 已确认是共享 inspurfs；`/mnt/afs` 若为各机独立 PVC，需在 node1 同路径放一份 worktree+venv。
2. `OUTPUT_ROOT` 指向 `/data`（共享）即可，两机一致。
3. 两机网络互通：node0 的 `MASTER_PORT`（默认 50012）对 node1 可达（无防火墙拦截）。
4. 两机 GPU 数相同（各 8 卡），主机名带数字后缀（`dev-<id>-0` / `dev-<id>-1`）。

### 命令（两机各执行同一条，与 `_base` 用法一致）

```bash
cd /mnt/afs/WorldAct-cosmos3-edge-droid-sft
NNODES=2 \
EXTRA_TAIL_OVERRIDES="trainer.max_iter=20000 scheduler.cycle_lengths=[20000]" \
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/singlerighthand-edge-droid-16n-$(date +%m%d) \
  bash examples/launch_sft_action_policy_singlerighthand_edge_dev.sh
```

**不要传 `NODE_RANK`**（与 `_base` 的红字规则同理）：wrapper 会从主机名推导——Kubeflow pod 名 `<job>-master-0` → rank 0 兼 rendezvous 主机、`<job>-worker-N` → rank N+1（`MASTER_ADDR` 默认 `<job>-master-0`）；开发机 `dev-e5bd7ae4-0` → rank 0、`dev-e5bd7ae4-1` → rank 1（`MASTER_ADDR` 默认 `dev-e5bd7ae4-0`）。生效时控制台打印 `>>> topology fallback: HOSTNAME=... -> NODE_RANK=... MASTER_ADDR=...`。平台若注入了 `SENSECORE_PYTORCH_NNODES/_NODE_RANK` 也会被识别。两种命名都不匹配的机器需显式传 `NODE_RANK` + `MASTER_ADDR`。两机先后启动均可，torchrun rendezvous 默认等 30 分钟。

> ⚠️ 该兜底初版只取主机名数字后缀，把 Kubeflow 的 `<job>-worker-0` 误推成 rank 0，导致双机互等 rendezvous 卡死（2026-10-01 `job-cosmos-raw` 任务）；已修复为区分 master/worker，逻辑与 `_base` 的 `_sft_launcher_common.sh` 一致。遇到该 bug 的任务作废重提即可。

### 16 卡下的配置变化（有意保持原生行为）

- `replicate=-1` 自动扩到 16 rank，dev 默认 16/rank 下**全局 batch = 16 × 16 = 256**（8 卡时是 128）。这是原生 recipe `replicate=-1` 的固有行为：全局 batch 随卡数翻倍，未做改动。
- 若想让 16 卡与 8 卡的全局 batch 一致（128），加：
  `EXTRA_TAIL_OVERRIDES="dataloader_train.max_samples_per_batch=8"`
- 学习率 / schedule 不随卡数缩放（recipe 固定 `lr=2e-5`、`cycle_lengths=[10000]`），改 batch 时请自行评估。

### 双机排障

| 症状 | 处理 |
| --- | --- |
| node1 卡在 rendezvous / `Connection refused` | 确认 `MASTER_ADDR` 是 node0 的本机 IP（不是 localhost）；`nc -zv <node0> 50012` 测连通；两机 `MASTER_PORT` 一致且未被占用（可换 `MASTER_PORT=29501`） |
| NCCL 卡住或选错网卡 | `export NCCL_SOCKET_IFNAME=<内网网卡名>`（用 `ip -o link` 查，如 `eth0`/`bond0`），两机都设；首次联调加 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT` 看选了哪条路 |
| 有 IB 但走了 TCP | 确认 `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` 与集群文档一致；A800 机器一般应走 IB |
| 两机 batch/数据不一致报错 | 检查两机 `OUTPUT_ROOT`、数据路径、代码版本（worktree HEAD）完全一致 |
| 一机掉队后全挂 | torchrun 默认共进退；修复后两机用同一 `OUTPUT_ROOT` 重跑即自动 resume |

## 已知无害噪音

- 启动早期会打印一段 `ModuleNotFoundError: No module named 'nvidia.npp'` traceback —— 原 launcher 探测集群专用 wheel 失败后会以空路径继续，本机不影响（torch 为 cu130 自带运行时，torchcodec 已验证可导入，ffmpeg 用 `/usr/bin/ffmpeg`）。

## 与 `_base` 四模态 run 的对照口径

- **数据同分布**：本机 raw root 恰好就是 101 个 episode，与 `_base` 的 `singlerighthand_101_episodes.txt` allowlist 同一集合；原生代码无 allowlist 功能也无需它。
- **batch 一致**：dev 默认 16/rank，与 `_base` 的 fk-point run 相同；8 卡全局 batch 同为 128，loss 曲线可按步直接对比。
- **attention 路径一致**：`COSMOS_FLASH2_VARLEN=1`（wrapper 默认）下与 `_base` 所有 run 同为 flash2 varlen，排除了后端差异这个混淆变量。
- **视频 token 来源一致**：window latent 缓存自动开启时，与 `_base` 各 run 相同，每窗口直接用预编码 latent（bit 一致于在线 encode），训练不再重复跑冻结 VAE。
- 产物（DCP checkpoint、`*_sft.log`、wandb offline 目录）均为原生格式，未引入 `_base` 的任何改动。
