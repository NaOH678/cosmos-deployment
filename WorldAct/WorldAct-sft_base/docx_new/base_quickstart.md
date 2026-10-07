# `_base` 四模态 开箱使用手册

> **2026-09-30 缓存接入更新**：四模态 launcher 现在默认使用 sandwich 的
> `pointflow_windows/` 现有缓存，最多 500 点，区域配额 `2:0.40,3:0.45,4:0.15`，
> `min_voxel_members=3`、`min_valid_steps=16`、幻影守卫开启，scale=`0.0528`。
> **token 模式保持 `cluster`**，500 是选点预算，不是 cluster token 数。
> `cluster/per_point` 不影响缓存兼容性；选点配置不匹配会报错，缺窗口才在线回退。
> 新配方请用新 OUTPUT_ROOT，不能按新 scale 续训旧 300 点 checkpoint。
> 下文 300 点、0.075255、步速和实验结果均是旧 run 的记录。


> 面向拿到这个 worktree 的机器 / 人：从工作区现状到跑起**四模态（video + action +
> FK + pointflow）**训练的最短路径。
>
> 本文只做索引和关键操作，细节链到同目录其它文档。
>
> 分支：`WorldAct-cosmos3-edge-droid-sft_base`；worktree：
> `/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base`。

---

## 0. 这是什么 —— 以及它和另外两个 worktree 的关系

`_base` 是 **FK 线与 PointFlow 线的合并工作区**。两条线各自把一种新模态接进 Cosmos
（互不知情、各自都假设"我是唯一的额外模态"），本 worktree 把两者并进**同一次去噪**：

```
[vision | action | sound | pointflow | FK]   ← 一条 denoise 循环，四段一起去噪
```

三个 worktree 的定位（**不是新旧关系，是两摊并行的活 + 一个合并点**）：

| worktree | 线 | 记忆/文档位置 |
|---|---|---|
| `WorldAct-cosmos3-edge-droid-sft_mano` | **FK**（右手 21 关键点） | `docs/fk_modality_design.md`、`docs/fk_multinode_training.md` |
| `WorldAct-cosmos3-edge-droid-sft`（`-sft`） | **PointFlow**（点云） | `docs/pointflow_quickstart.md`、`docs/pointflow_multinode_training.md` |
| **`_base`（本 worktree）** | **合并：四模态** | **本文件** + [`fk_pointflow_merge.md`](./fk_pointflow_merge.md) + [`base_multinode_training.md`](./base_multinode_training.md) |

> ⚠️ **`_base` 没有自己的 venv。** 它借用 `-sft` 那个，而那个 venv 里
> `cosmos_framework` 是 **editable 安装、`.pth` 硬编码指向 `-sft`**。
> 所以**跑任何东西都必须 `PYTHONPATH=.`**，否则静默执行 `-sft` 的代码。
> 启动脚本自己会处理（`launch_..._fk_point_...sh:155` 有导入自检），
> **手动跑 `python xxx.py` 时必须自己加**。详见 §1.2。

---

## 1. 拿代码 + 环境

### 1.1 代码状态：**全部改动未提交**

这是最重要的一条现状。`_base` 的合并成果**只存在于这一个工作区里，没有远端备份**：

```bash
cd /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base
git rev-parse --short HEAD     # bae6a43
git status --porcelain -uall | wc -l   # 22
```

22 项 = **15 个已跟踪文件被改 + 7 个新文件**：

| 类别 | 文件 |
|---|---|
| **已改** | `omni_mot_model.py`（四模态联合路径的主体）、`fk_sampling.py`、`cosmos3_vfm_network.py`、`action_policy_singlerighthand_edge.py`（recipe，D4 σ）、`dcp.py`、`flash2/checks.py`、`_sft_launcher_common.sh`、两个 launcher，以及 5 个 `*_test.py` |
| **新增** | `tools/run_fk_point_101.sh`、`tools/check_flash2_varlen.py`、`tools/check_multinode_nccl.py`、`examples/launch_sft_action_policy_fk_point_singlerighthand_edge.sh`、`examples/toml/sft_config/action_policy_fk_point_singlerighthand_edge.toml`、`docx_new/` 下两份文档 |

> 🔴 **不要在这个 worktree 上跑 `git checkout` / `clean` / `stash`。**
> 同理，**不要对 `-sft`（PointFlow worktree）做任何写操作** —— 它的 131 个文件
> 也全是未提交的，是 FK 合并的来源，只能单向读出。

### 1.2 解释器：借 `-sft` 的 venv

```bash
SIBLING_VENV="/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv"     # launcher:98
PYTHON_BIN="${PYTHON_BIN:-$SIBLING_VENV/bin/python}"
```

每次开 shell：

```bash
export LD_LIBRARY_PATH=''                     # 见 §7
export PYTHONPATH=.                            # 手动跑脚本时必加
```

**为什么 `PYTHONPATH=.` 是硬要求**：`python tools/foo.py` 会把 `tools/` 放进
`sys.path[0]`，editable 安装的 `cosmos_framework` 因此胜出 —— 你**静默执行了
`-sft` 的旧代码**，症状是"我改的代码没生效"。自检：

```bash
PYTHONPATH=. $SIBLING_VENV/bin/python -c \
  "import cosmos_framework, os; print(os.path.realpath(cosmos_framework.__file__))"
# 期望：路径在本 worktree 下，不是 -sft
```

`-sft` 的 venv 里已经装好了 PointFlow 需要的 sonata 依赖（`addict`、
`spconv-cu126`、`torch-scatter`），**四模态跑得起来就是证据**，不用重装。
（这三条的安装方式见 `-sft` 的 `docs/pointflow_quickstart.md` §1。）

### 1.3 单测是脚本，不是 pytest

venv 没装 `pytest-xdist`，仓库根的 `conftest.py` 加载不了 → `pytest` 报
**"no tests ran"**，看起来像通过。直接跑文件：

```bash
PYTHONPATH=. $SIBLING_VENV/bin/python cosmos_framework/model/generator/<name>_test.py
```

`_base` 现有的 18 个相关测试（`cosmos_framework/model/generator/` 下）：
`fk_batch_test`、`fk_branch_test`、`fk_independent_schedule_test`、`fk_joint_sampling_test`、
**`fk_pointflow_compose_test`（跨模态索引回归，见下）**、`fk_sequence_test`、
`omni_mot_model_test`、`pointflow_{batch,branch,codec,geometry,per_point,profiling,sampling,sequence,training,window}_test`。

> **`fk_pointflow_compose_test.py` 是合并新造 bug 的守卫。** 两个 `attach_*_tokens`
> 都写于"自己是唯一额外模态"的年代，只重映射 vision/action/sound + **自己**，
> 于是**后挂载的一方会让先挂载的一方的索引失效** —— 写入成功、无形状错误、
> loss 照降。撤掉修复它必须报
> `vision indexes collide with the new modalities: [...]`。

> **测试能"静默跳过"**：几个测试指向真实缓存，路径没了会**报成功但什么也没做**。
> 看退出码不够，要看它**打出的计数**。

---

## 2. 数据摆放

全部在 `/data/shichaojian/`（模型、数据、输出），源码和 venv 在 `/mnt/afs/`（`$HOME`）。
这个分界是有意的。

| 用途 | 路径 | 谁引用 |
|---|---|---|
| 基底 DCP | `models/cosmos3-edge-droid-dcp` | `BASE_CHECKPOINT_PATH` |
| 模型包（VAE/tokenizer/processor） | `models/cosmos3-edge-droid` | `EDGE_DROID_MODEL_PATH` |
| 原始视频（101 集） | `raw_data/singlerighthand_sandwich_100` | `SINGLERIGHTHAND_RAW_ROOT` |
| 视频/VAE 缓存 | `datasets/singlerighthand-sandwich-100-cosmos-cache`（含 `vae_window_latents/`） | `SINGLERIGHTHAND_CACHE_ROOT` |
| **FK 标注** | `raw_data/sandwich_fk21/<ep>/annotations/wuji_fk21.npz`（101/101） | `FK_ANNOTATION_ROOT` |
| Sonata/PTv3 权重 | `checkpoints/ptv3/sonata_small.pth` | `POINTFLOW_SONATA_CHECKPOINT` |
| **PointFlow manifest** | `/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/sandwich_924_20260928/manifest.json` | `POINTFLOW_MANIFEST` |
| 点云数据本体 | `pf_out/9.24/sandwich/labeled/` | manifest 内的绝对路径 |
| 输出 | `runs/cosmos/` | `OUTPUT_ROOT` |

以上 2026-09-29 实测**全部存在**。⚠️ 旧集群的
`/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/...` **已失效**。

**manifest 里嵌了轨迹绝对路径** —— 挂载点变了要 `sed` 换前缀或重建。
它**不在 git 里**（在 `-sft` 的 `pointflow_outputs/` 下）。

---

## 3. 启动训练

```bash
cd /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base

# 先看计划不启动
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-dryrun NNODES=1 DRY_RUN=1 \
  bash tools/run_fk_point_101.sh

# 单机 8 卡短程
MAX_ITER=60 OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-smoke-<date> NNODES=1 \
  bash tools/run_fk_point_101.sh

# 16 卡（2 节点）：每个节点执行完全相同的一条，不要传 NODE_RANK
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-101-<YYYYMMDD> NNODES=2 MAX_ITER=20000 \
  bash tools/run_fk_point_101.sh
```

启动链：

```
tools/run_fk_point_101.sh                                     # 自包含 wrapper：守卫 / HSDP / dry-run
  └─ examples/launch_sft_action_policy_fk_point_singlerighthand_edge.sh   # 两个模态的 gate + A800 环境 + 输入检查
       └─ examples/_sft_launcher_common.sh                    # torchrun 参数组装（多机拓扑在这里）
            └─ torchrun -m cosmos_framework.scripts.train --sft-toml=...
```

**多机的完整说明见 [`base_multinode_training.md`](./base_multinode_training.md)** ——
注入机制、拓扑、起后验证、排障、flash2 都在那里。

### 3.1 wrapper 的守卫（触发即拒启动，都是"静默出错"型的）

| 守卫 | 触发条件 |
|---|---|
| `OUTPUT_ROOT` | **必填且每次换新** |
| `checkpoints/` | 目录里已有 checkpoint 就拒（**查的是 `checkpoints/`，不是目录本身** —— 目录存在性在多机下是跨节点竞态，见多机文档 §4） |
| `POINTFLOW_MANIFEST` | 文件不存在 |
| **`pointflow_displacement_scale`** | **≠ 0.075255** |
| **`fk_displacement_scale`** | **≠ 0.083745** |
| allowlist / recipe | 文件不存在或为空 |

两个 scale 守卫是这批守卫里最要紧的：**scale 错了没有任何形状错误**，只是所有预测被均匀缩放。
scale 是**这套选择规则的属性、不是常量** —— 换 manifest 或换任一选择旋钮就必须重测。
详见 [`fk_pointflow_merge.md`](./fk_pointflow_merge.md) §F.8。

### 3.2 起后控制台横幅（照抄 `run_fk_point_101.sh:157-168`）

```
 fk + point four-modality training
   output root : <OUTPUT_ROOT>
   episodes    : 101  (<allowlist>)
   gpus        : 8 per node x 2 node(s) = 16 ranks
   batch       : 16 per rank -> global 256 samples/step
   topology    : shard=8 replicate=2
   attention   : COSMOS_FLASH2_VARLEN=1
   eval        : fk_eval + fk_rollout + joint arms, pointflow_eval + joint
   overrides   : ... trainer.max_iter=20000 scheduler.cycle_lengths=[20000] ...
```

> 当前配置中，`fk_eval` 主产物已改为联合去噪，无需设置 `FK_EVAL_JOINT`。
> 旧进程和旧日志仍保持启动时的条件去噪行为；下次启动才生效。

### 3.3 可覆盖项

`OUTPUT_ROOT`（必填）、`NNODES`、`NPROC_PER_NODE`、`PER_RANK_BATCH`（默认 16）、
`MAX_ITER`、`RESUME`、`DRY_RUN`，以及所有 `POINTFLOW_*` / `FK_*` / `SINGLERIGHTHAND_*`。

**`MAX_ITER` 必须显式给**：这个 wrapper **不做样本预算推导**（与 FK 线的 `run_fk_101.sh`
不同），不给就用 toml 里的 `10000`。

---

## 4. 看训练效果 —— **读数前必看这一节**

这个 repo 里最容易出错的地方不是代码，是**读数字**。

### 4.1 FK 联合评估

| 评估 | 路径 | 当前行为 |
|---|---|---|
| `fk_eval` 固定窗口 | `fk_eval/step_*/<case>/` | video + action + 可用 PointFlow + FK 联合去噪，保留首帧/状态条件 |
| `fk_rollout_joint_action` 整集窗口评估 | 顶层 `fk_rollout_joint_action/step_*/` | 保持原有联合评估 |

`fk_eval` 仍写原路径下的 `prediction.npz`、`metrics.json`，按原开关生成图和视频。
主采样默认 UniPC 4 步，首次参考改为联合 UniPC 16 步；不再额外跑条件分支。
新指标带 `sampling_mode="joint"`，旧日志中的条件去噪指标不能与它直接比较。
实现见 `cosmos_framework/callbacks/fk_eval.py`，开关由共享 recipe 的 `joint_only=True` 设置。

### 4.2 PointFlow 的两个 arm（都在跑）

| arm | 目录后缀 | `conditions` 字段 | 可读? |
|---|---|---|---|
| condition | 无 | `clean GT video/action + ...; conditional fit, not rollout` | ❌ 同样 OOD |
| **joint** | **`_joint`** | `first frame + state action clean; future video/action/point joint rollout` | ✅ 部署同款 |

每个 case 目录下的 `metrics.json` 用 `all_ade_mm` / `zero_all_ade_mm`（**没有
`ratio_to_zero` 键**，要自己除）；FK 侧的 metrics.json **有** `ratio_to_zero`。

### 4.3 三条读数纪律

1. **唯一裁决键是 `ratio_to_zero`（< 1 才算赢）**，不是 loss。
2. **`zero_ade_mm` 是纯数据属性**（由目标轨迹自己算），与模型无关。
   **两个 run 的 zero 不同 = 评的不是同一个窗口，不可比。**
   跨 scale 的 loss 读数也不可比。
3. **rollout 的 ratio 不能与固定窗跨比**（zero 量级不同：rollout ~820 mm vs 固定窗 20~240 mm）。

### 4.4 健康指标行

`callbacks` 每 `logging_iter` 步打一条 rank-0 行（`callback.py:518`）：

```
Iteration 60: Total Loss: 14.0387 | Video Loss: 0.2001 | Action Loss: 0.9853 \
  | PointFlow Loss: 1.1963 | PointFlow ADE: 102.23mm (zero 88.48mm) \
  | FK Loss: 0.9888 | FK ADE: 111.72mm (zero 103.05mm)
```

**两个模态的 loss 与 ADE 都在这一行里**，一眼可查。ADE 在训练早期贴着 zero 是正常的
（分支刚初始化）。

更细的分模态曲线在 wandb offline 的 `.wandb` 二进制里（键名 `train/fk_loss`、
`train/pointflow_loss`、`train@2_detail/flow_matching_loss_vision` 等），
不在 stdout 里。

---

## 5. 单机 8 卡冒烟（我用的那套）

```bash
MAX_ITER=60 \
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-smoke-<date> NNODES=1 \
FK_VAL_ON_START=true \
  bash tools/run_fk_point_101.sh
```

`FK_VAL_ON_START=true` 让**第 0 步就跑完整 eval**，这样不用等 500 步就能验四段布局。
验完的判据是这行（`omni_mot_model.py:4649`）：

```
FK joint arm: flat vector elements (sample 0) = vision 646272 | action 2112 | pointflow 28800 | FK 2016
```

**四段必须全非零、顺序正确**。若变成 `pointflow 0` 并带 `(no PointFlow payload packed)`，
就是 payload 没进来、**静默退回三段**了。
`28800 = 32×300×3`（`select_top_n=300`）、`2016 = 21×3×32`。

部署路径（`_prepare_inference_data`）另有一行：
`inference flat vector: [vision | action | pointflow | FK], 679200 elements (sample 0)`，
`646272 + 2112 + 28800 + 2016 = 679200` —— 与上面**逐位相加相等**。

> 这两行日志是**故意存在的**：四段尺寸全是推导出来的，任何**排列**上的手误都会让某段的
> `.reshape()` 失败、**不会静默**；能静默的只有"顺序本身被改错"，而指标分不出来。

---

## 6. 当前状态（2026-09-30）

- **16 卡生产 run `fk-point-101-0` 在跑**，健康（见 §6.1）。
- 四模态联合路径、部署推理路径、三处"剥掉对方模态"的撤销 —— **全部已完成并在 GPU 上验证过**。
- 全量单测 **26 PASS / 0 FAIL**。
- 全部改动**未提交**。

### 6.1 run `fk-point-101-0` 实况（2026-09-30 05:22 采）

| 项 | 值 |
|---|---|
| 步速 | **11.7 s/步**（含 eval/ckpt 停顿）→ 20000 步 ≈ **65 h** |
| 异常 | Traceback / OOM / NaN / KeyError **全 0** |
| loss | 总 loss 17.89（首点）→ ~1.6 |
| ckpt | 每 500 步一个 × **30 G**，**无任何保留策略 → 20000 步约 1.2 TB** |
| eval | `fk_eval` / `pointflow_eval` 各 0/500/…；rollout 在步 **0 / 1000 / 2000**（`FK_ROLLOUT_EVERY` 数的是验证事件） |

部署 arm 的实测轨迹（FK rollout `ratio_to_zero`，零基线 824/821 mm 不变）：

| step | 0 | 1000 | 2000 |
|---|---|---|---|
| episode_0033 | 0.192 | 0.044 | **0.038** |
| episode_0059 | 0.219 | 0.046 | **0.043** |

---

## 7. 排障速查

| 症状 | 看哪里 |
|---|---|
| `torch._C` import 报错 | 没 `export LD_LIBRARY_PATH=''` |
| "我改的代码没生效" | 没 `PYTHONPATH=.`，执行的是 `-sft` 的代码（§1.2） |
| `pytest` 说 "no tests ran" | 正常，单测是脚本不是 pytest（§1.3） |
| 测试"通过"但没打印计数 | 它静默跳过了（路径不存在） |
| `CUBLAS_STATUS_NOT_INITIALIZED`（**第一次 eval** 时） | `LD_LIBRARY_PATH` 污染，逃生口 `KEEP_LD_LIBRARY_PATH=1` |
| OOM / 显存碎片 | `PYTORCH_ALLOC_CONF=expandable_segments:True`（launcher 已默认） |
| `Fixed eval identity changed` | 输出目录被换数据集复用了，换新目录 |
| `metadata is None` 且原因被吞 | ckpt 被裁剪过，resume 回不去 |
| 步速在 6.5~20.8 s 之间抖 | **正常**，数据 I/O，不是故障（§6.1） |
| 步速 ~8 s 且日志有 natten | flash2 varlen 没生效，**立刻停**（1.85× 差） |
| 启动时日志 grep 不到某行 | **不代表没发生** —— 启动那一波日志会被丢，见多机文档 §6 |
| 训练报 labeled 目录不存在 | manifest 死链 |
| `No module named 'addict' / 'spconv' / 'torch_scatter'` | sonata 依赖没了，见 §1.2 |

---

## 文档地图

| 文档 | 内容 |
|---|---|
| [`fk_pointflow_merge.md`](./fk_pointflow_merge.md) | **合并的唯一依据**：冲突清单、四模态方案（§D）、执行状态（§E）、训练启动与集群迁移（§F） |
| [`base_multinode_training.md`](./base_multinode_training.md) | 多机训练：注入机制、HSDP、提交、起后验证、排障、flash2 |
| [`fk_new_cluster_setup.md`](./fk_new_cluster_setup.md) | 新集群迁移：绝对路径重映射、数据该拷什么、验证清单 |
| `-sft` 的 `docs/pointflow_quickstart.md` | PointFlow 线单模态（**它的 launcher 与 `_base` 不是同一个**） |
| `-mano` 的 `docs/fk_multinode_training.md` | FK 线单模态多机（**boot 机制同源，规模相关的数不可照抄**） |
| `AGENTS.md` | 仓库总地图（目录结构、命令、规则） |
| `docs/faq.md` / `docs/training.md` / `docs/setup.md` | 通用排障 / 训练 / 安装 |
