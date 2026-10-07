# `_base` 四模态 多机训练（SenseCore 集群）

> **2026-09-30 更新**：base 已接入现有 sandwich 500 点分层缓存，保持 cluster 模式，
> scale 改为 0.0528（launcher 覆盖）。当前默认见 [base_quickstart.md](./base_quickstart.md) 开头；
> 本文 300 点配置及运行测量保留为历史记录。


> 本文覆盖**四模态（video + action + FK + pointflow）**在 2+ 节点 A800 上的启动方式、
> HSDP 拓扑、步数、起后验证与排障；第 9 节是环境坑，第 10 节是**注意力后端 flash2 varlen**
> （决定步速 1.85×）。单机用法见 [`base_quickstart.md`](./base_quickstart.md)。
>
> 这份文档与两份单模态同题文档**同源** —— 启动链路和注入机制**完全一样**（共用同一份
> `examples/_sft_launcher_common.sh`），差别只在 recipes：
> - FK 线：`WorldAct-cosmos3-edge-droid-sft_mano/docs/fk_multinode_training.md`
> - PointFlow 线：`WorldAct-cosmos3-edge-droid-sft-pointflow/docs/pointflow_multinode_training.md`
>
> ⚠️ **机制可以照抄，规模相关的数不能**：那两份的前提是单模态 recipe，本节的
> `run_fk_point_101.sh` 是**另一支 wrapper**（步数算法、守卫、可覆盖项都不同）。

---

## 1. 启动链路

```
tools/run_fk_point_101.sh                                          # wrapper：守卫 / HSDP / 步数 / dry-run
  └─ examples/launch_sft_action_policy_fk_point_singlerighthand_edge.sh   # 两个模态的 gate + A800 环境 + 输入检查
       └─ examples/_sft_launcher_common.sh                          # torchrun 参数组装（多机在这里）
            └─ torchrun -m cosmos_framework.scripts.train --sft-toml=...
```

torchrun 跨节点组网要 5 个量：`--nproc_per_node`、`--nnodes`、`--node_rank`、
`--master_addr`、`--master_port`。单机时全部缺省即可；多机时**前三个必须每节点正确、
后两个全任务一致**。

---

## 2. 本集群的注入机制（实测）

SenseCore 的 PyTorch 任务实际注入：

| 变量 | 例子 | 说明 |
|---|---|---|
| `NNODES` | 2 | 节点数 |
| `MASTER_ADDR` | `job-xxx-master-0.job-xxx` | pod.service 短名 |
| `MASTER_PORT` | 23456 | rendezvous 端口 |
| `RANK` | 0/1 | pod 序号（master=0, worker-N=N） |
| `WORLD_SIZE` | 2 | ⚠️ 是**节点数**，不是 GPU 数 |
| `SENSECORE_ACCELERATE_DEVICE_COUNT` | 8 | 每节点卡数 |
| `SENSECORE_PYTORCH_NNODES` / `_NODE_RANK` | — | **部分任务类型没有** |

**不注入 `NODE_RANK`。** `_sft_launcher_common.sh` 的解析顺序：

1. 显式 `NNODES`/`NODE_RANK` 优先，其次 `SENSECORE_PYTORCH_NNODES`/`_NODE_RANK`；
2. 若 `NNODES` 已给但 `NODE_RANK` 仍空 → 按 Kubeflow pod 名推导：
   `<job>-master-0` → rank 0 且自己当 rendezvous 主机；`<job>-worker-N` → rank N+1、
   `MASTER_ADDR=<job>-master-0`；兜底生效时控制台有一行
   `>>> topology fallback: HOSTNAME=... -> NODE_RANK=...`；
3. `MASTER_ADDR`/`MASTER_PORT` 透传，缺省 `50012`。

> 🔴 **`run_fk_point_101.sh` 故意不给 `NODE_RANK` 兜底默认值。**
> 如果默认成 0，第 2 条的 pod 名推导永远不触发（它要求变量为空）→
> **所有 worker 都以为自己是 rank 0**，各自去当 master → 全体卡在 rendezvous。
> 这是踩过的坑（FK 线第一次提交）。

---

## 3. 并行拓扑（HSDP）

约束：**`shard × replicate × CP = WORLD_SIZE`**（world = 节点数 × 每节点卡数）。

wrapper 自动推导：**shard = 每节点卡数、replicate = 节点数**。即节点内 FSDP 分片走
**NVLink**（每层一次 all-gather，延迟敏感），节点间只做**梯度** allreduce
（每步一次，量是分片后的）。

**不要把 shard 设成跨节点**（如 16）—— 那会把每层 all-gather 推上 IB。
本机 `nvidia-smi topo -m` 显示 8 卡之间全是 `NV8`，正是为这个拓扑准备的。

| 规模 | shard | replicate | 每卡 batch | 全局 batch |
|---|---|---|---|---|
| 8 卡单机 | 8 | 1 | 16 | 128 |
| **16 卡（2 节点）** | **8** | **2** | **16** | **256** |
| 32 卡（4 节点） | 8 | 4 | 16 | 512 |

> recipe 的 TOML 里 `data_parallel_shard_degree = 1`（单机不分片）—— **16 卡会不满足
> 上面那个不变式**，必须由 wrapper 通过 `EXTRA_TAIL_OVERRIDES` 覆盖。wrapper 已做。

---

## 4. 提交命令

任务模板的启动命令栏（bash -c 模式）填一行：

```bash
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-101-<YYYYMMDD> NNODES=2 MAX_ITER=20000 \
  bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/run_fk_point_101.sh
```

**每个节点执行完全相同的一条，不需要手动区分 rank。**

- `OUTPUT_ROOT` **必须显式给且每次换新**。这是提交式的唯一安全形态：wrapper 里那套
  "目录已存在就加后缀"的逻辑是**各节点独立状态**，node0 建完目录后 node1 会看到它存在而选
  `-2`、node2 选 `-3`，**三边各写一个 run、每个只拿到 1/3 数据**，而 loss 曲线看起来像训练 bug。
- **守卫查的是 `checkpoints/`，不是目录本身。** 目录里已有上次训练的 checkpoint 就拒绝启动；
  要继续旧 run 加 `RESUME=1`。
  ⚠️ **为什么不能直接查目录存在性**：那是跨节点的竞态 —— 每个节点都跑这个脚本且同时启动，
  node0 会先到 `_sft_launcher_common.sh`，其中的 `mkdir -p "$LOG_DIR"` 把目录建出来，
  后到的 worker 看到目录存在就拒绝启动。`checkpoints/` 是正确信号 —— 它只在真的训练过之后
  才存在，并发节点建个 `logs/` 不会触发。
- `NNODES` 显式给（pod 名兜底需要它）。
- **不要传 `NODE_RANK`**（理由见 §2 的红字）。
- 脚本自己 `cd` 到 worktree，所以提交时的 CWD 无关。

**提交前先 dry-run**（打计划不启动，任一节点）：

```bash
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-dryrun NNODES=2 MAX_ITER=20000 DRY_RUN=1 \
  bash tools/run_fk_point_101.sh
```

核对三处：`gpus: 8 per node x 2 node(s) = 16 ranks`、`topology : shard=8 replicate=2`、
`overrides` 里 **`trainer.max_iter` 与 `scheduler.cycle_lengths` 同时出现且相等**
（只来一个的话调度器会在 `find_in_interval` 返回 `None` 后崩，见 §11.1）。

---

## 5. 步数是怎么算的（四模态特有）

### 5.1 这个 wrapper **不做**样本预算推导

FK 线的 `run_fk_101.sh` 按"数据预算"反推步数：

```
预算 = REF_STEPS × REF_BATCH = 10000 × 256 = 2,560,000 样本
MAX_ITER = 预算 ÷ (每卡batch × 卡数 × 节点数)
```

**`run_fk_point_101.sh` 没有这套逻辑 —— 直接用 toml 里的 `10000`。**
后果：**8 卡与 16 卡跑同一份 toml 都是 10000 步**，但总样本分别是 128 万和 256 万。
想要"每个实验固定样本量"，就必须显式传 `MAX_ITER`。

### 5.2 本次的目标规模

| 量 | 值 |
|---|---|
| 步数 | **20000**（显式 `MAX_ITER=20000`） |
| 全局 batch | 256（每卡 16 × 16 卡） |
| 总样本 | 5,120,000 |
| 墙钟 | **≈ 65 h（2.7 天）** |
| checkpoint | 每 500 步 × **30 G** → 40 个 ≈ **1.2 TB** |

### 5.3 为什么每卡 batch 钉在 16

每步耗时由**每卡** batch 决定，不是全局 batch。加卡只增加每步样本数，不减少每步时间
（实测并行效率 ~99%）。想要更快只能**少做工作**，不是加卡。

---

## 6. 起后验证清单

> 🔴 **先读这条：多机的这个日志文件**在启动那一波**会丢行，grep 返回 0 不代表没发生。**

实测（run `fk-point-101-0`，2026-09-29/30）：

| 行 | 现在 grep 到的条数 | 实情 |
|---|---|---|
| `RankPartitionedDataLoader: world_size: 16 and rank: N` | **16 / 16** ✅ | 稳态行，**完整** |
| `RankPartitionedDataLoader allocation` | **16 / 16** ✅ | 同上 |
| `get_config_module` | **8** | 16 个 rank 只留 8 条 |
| `FK branch installed` | **0** ⚠️ | 启动时**亲眼读到过** |
| `PointFlow branch installed` | **0** ⚠️ | 同上 |
| `Attention backend selected` | **0** ⚠️ | 同上 —— 这是 flash2 的判据行 |

文件本身**没有截断**（12248 行、时间跨度 `09-29 17` → `09-30 05` 连续、0 个 NUL 字节），
所以不是"文件被改小"，而是**并发写期间的行被互相覆盖**：16 个进程、跨 2 个节点、各自
append 到同一个网络文件系统（`/data` = `inspurfs`）上的文件，而 **`O_APPEND` 的原子性只在
本地文件系统成立**。丢失**集中在启动那一波**（16 个进程在几秒内同时打开同一个文件），
稳态（每步的 `iter_speed`、每 rank 一次的 dataloader 行）不受影响。

**所以：**

- **启动类检查（分支装没装上、attention 后端）→ 用平台控制台**（每个 rank 各自 stdout，
  不会互相覆盖）**，或启动时就 `tail -f`**。事后 grep 只能当"最好不要当真"。
- **稳态类检查（`world_size` 是不是 16）→ 事后 grep 可靠**，因为那些行活下来了。
- **grep 返回 0 要读作"未知"，不是"不存在"。**

### 6.1 控制台（wrapper 回显）

- [ ] `gpus: 8 per node x 2 node(s) = 16 ranks`
- [ ] `topology : shard=8 replicate=2`（数字须符合 §3 的表）
- [ ] `[topology-env] ...` 平台注入的实际值；兜底生效时应有
      `>>> topology fallback: HOSTNAME=... -> NODE_RANK=...`

### 6.2 训练日志 `<OUTPUT_ROOT>/logs/action_policy_fk_point_singlerighthand_edge_sft.log`

- [ ] **`RankPartitionedDataLoader: world_size: 16 and rank: <n>`** —— 出现
      `world_size: 8` 就是**没组网、各自单机跑了，立刻杀**（共享 `OUTPUT_ROOT` 会在
      save 时互相写坏）
- [ ] `RankPartitionedDataLoader allocation (16 GPUs)`
- [ ] 两条安装日志都在（**各 2 条**：`net` bf16 + `net_ema` fp32）⚠️ 见上面的丢行警告
- [ ] 指标行**同时**含 `PointFlow Loss` 与 `FK Loss`
- [ ] **flash2 grep —— 唯一可靠的判据，不要看步速**：

```bash
grep -oE "Attention backend selected: [a-z0-9]+ \(sm80, requires_grad=True[^)]*\)" \
  <OUTPUT_ROOT>/logs/action_policy_fk_point_singlerighthand_edge_sft.log | sort -u
```

必须出现 `flash2 … requires_grad=True … varlen=True`；若是 `natten` **立刻停**重交
（1.85× 步速差）。⚠️ `requires_grad=False` 的那些本来就是 flash2，验不出东西来。

### 6.3 组网失败的判别

日志里 **`world_size 8` 而不是 16**，或同一 step 被打印两遍**且内容对不上**
（各自的 loss / 迭代号不一致）→ 两节点没组网、各自单机跑了。**杀掉重交。**
（注意区分：每个 rank 各写一条**内容相同**的行是**正常**的，不是共享日志。）

### 6.4 首次提交前先跑连通性测试

多机握手失败**不会快速报错**，会一直卡到 rendezvous 超时（默认 30 min）或 NCCL watchdog
（1800 s），日志里只有"超时"不说原因。

```bash
# 每个节点上各跑一次，NODE_RANK 各不同（0..节点数-1）
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0 ip> PYTHONPATH=. \
  /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/check_multinode_nccl.py
```

它依次验：握手 → 各边 world size / 每节点卡数是否一致 → 真实 NCCL all-reduce → 带宽数字。
全对才打 `PASS`。

⚠️ **只有这个测试脚本要手填 `NODE_RANK`。** 它直接开 `init_process_group`，不经过
`_sft_launcher_common.sh`，所以没有 pod 名兜底。**训练（§4）不带 `NODE_RANK`。**

> 💡 **顺手做一个更有价值的变体**：加 `KEEP_MASTER_PORT=1` 会让它改用**训练要用的那个端口
> （50012）**而不是默认的 50013 —— 这样连通性测试**同时证明那个端口是空的**。
> 提交前占端口是那种"跑到 rendezvous 才炸"的失败。

---

## 7. 排障

| 症状 | 原因与处理 |
|---|---|
| `world_size 8`、双份 rank 输出交错 | 没组网，各自单机跑了。杀掉按 §4 重交 |
| `NNODES=2 but no NODE_RANK` | 平台没注入、pod 名也不匹配 `<job>-{master,worker}-N`。手动传 `NODE_RANK=<0..N-1>` |
| `[c10d] ... hostname ... err=-3` | k8s pod 主机名反查失败，**良性警告**，忽略 |
| `IPv6 network addresses ... (gai error)` | c10d 先试 IPv6 失败会自动回退 IPv4，**良性**；若随后卡死才是真 DNS 不通 |
| rendezvous 卡住/超时 | worker 解析不到 master 名：`MASTER_ADDR` 换成 FQDN（`<pod>.<svc>.<ns>.svc.cluster.local`）或 master pod IP |
| 组网成功但 step 时间翻倍 | 跨节点走了 TCP 而非 IB：加 `NCCL_DEBUG=INFO` 重启，看 `NET/IB` 还是 `NET/Socket` |
| `CUBLAS_STATUS_NOT_INITIALIZED`（**第一次 eval**） | `LD_LIBRARY_PATH` 污染，见 §9.1 |
| `torch.OutOfMemoryError` | 见 §8 / §9.2 |
| 启动类日志 grep 不到 | **先看 §6**：启动那一波会丢行，不代表没发生 |

---

## 8. 硬件差异：A800 vs H200（这条线必须知道）

| | 旧集群 | 新集群 |
|---|---|---|
| 型号 | NVIDIA H200 | NVIDIA A800-SXM4-80GB |
| 架构 | Hopper (sm_90) | Ampere (sm_80) |
| 显存 | **141 GB** HBM3e | **80 GB** HBM2e |
| 带宽 | ~4.8 TB/s | ~2.0 TB/s |
| CUDA 栈 | cu128 | cu130 |

**每卡吞吐差约 4.3×**（实测：batch 32 时 H200 3.83 s/步、A800 16.56 s/步）。

后果：

1. **同一份 config 在 A800 上会 OOM。** `-101` 在 8×H200 上是 4.61 s/步，同样的
   8×A800 从没跑完过 —— FK 线的 run `-10-3` 在 iteration 42 就 `OutOfMemoryError`
   （69.26 GiB allocated，剩 ~0.5 GiB）。显存小 43% 是主因。
2. **`data_parallel_shard_degree` 决定 8 卡能不能用上。** recipe 的 toml 把它钉成 **1**
   （不分片）—— 那种情况下**每卡显存与 1 卡完全相同**，8 张卡只买吞吐不省显存。
   wrapper 现在按每节点卡数自动设 shard，把参数 + fp32 master + 两阶 Adam 从 ~25 GB
   压到 ~3 GB/rank。
3. **新旧数字不可直接比**：不同架构上算子实现和数值路径都不同。要并排看 ADE 时先记下这条。

---

## 9. 环境坑（都在启动器里修掉了）

### 9.1 `LD_LIBRARY_PATH` 必须清空

NGC 风格容器里 `LD_LIBRARY_PATH` 以 `/usr/local/nvidia/lib*:/usr/local/cuda/lib64` 开头，
最后一个含 CUDA 13.2 toolkit 的 `libcublasLt` **13.4.0.1**，而 PyTorch 自带的
`nvidia-cublas` wheel 是 **13.1.0.3**。实测（读 `/proc/self/maps`）只有**一个**被遮蔽：

```
libcublas.so.13   -> venv nvidia/cu13    (13.1.0.3)
libcublasLt.so.13 -> /usr/local/cuda-13.2 (13.4.0.1)   ← 错配
```

表现为 FK 线的 run `-10-3` 在**第一次验证**就死在
`CUBLAS_STATUS_NOT_INITIALIZED ... cublasLtMatmulAlgoGetHeuristic`。

**修在哪**：`examples/launch_sft_action_policy_fk_point_singlerighthand_edge.sh` 里
`export LD_LIBRARY_PATH=`。**故意不放在 `_sft_launcher_common.sh`** —— 那个文件被
~18 个 recipe 共用，而 `launch_sft_action_policy_singlerighthand_nano.sh` 是**故意要设**
这个变量的，放公共里会一次性覆盖掉所有 recipe 的意图。逃生口 `KEEP_LD_LIBRARY_PATH=1`。

### 9.2 显存碎片

FK 线的 `-10-4` / `-10-5` 两次跑到 iteration 1 结束、iteration 2 OOM：

```
69.26 GiB is allocated by PyTorch, and 7.07 GiB is reserved by PyTorch but unallocated
```

那 7 GiB 是分配器攥着但不能复用的 —— OOM 时只差 1.97 GiB，所以这是最大的一块可回收量。
修法是 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（PyTorch 自己在该 OOM 消息里
给的建议），已在启动器里默认开启。纯分配器行为，无数值影响。

⚠️ `PYTORCH_CUDA_ALLOC_CONF` 和 `PYTORCH_ALLOC_CONF` 两个名字 torch 2.10 **都认**
（已用非法值投毒实测）。

### 9.3 不要用环境里的 `python3`

FK 线的 wrapper 早期版本用 `python3 -c "import tomllib..."` 读配方。**job pod 上的
`python3` 比 3.11 老**（`tomllib` 是 3.11+ 才有的），提交后直接死在
`ModuleNotFoundError: No module named 'tomllib'` —— 而**开发机上每条 dry-run 都能过**，
因为那台机器的 `python3` 恰好是 Conda 3.14。

**教训**：拿一台机器上的环境解释器去验一个要在另一台机器跑的脚本，**测的是机器，不是脚本**。

`run_fk_point_101.sh` 里没有解释器调用，但仍 pin 了共享 venv 的 `PYTHON_BIN`
以防将来有人加回去。

---

## 10. 注意力后端：flash2 varlen（默认开启）

这条线的**训练 varlen attention** 有两个可选后端，**相差 1.85× 步速**。

### 10.1 为什么默认不是 flash2

上游代码**明确禁用** flash2 的 varlen 路径：

```python
if is_varlen:
    target_fn("Flash Attention v2 (flash2) varlen is banned due to instability.
               Please choose another backend.", exception=ValueError)
```

在 sm80（A800）上这就让 **natten 成为唯一的 varlen 后端**。

### 10.2 逃生口

`cosmos_framework/model/attention/flash2/checks.py` 里有一个本地逃生口
（**从 PointFlow worktree 移植过来**，那边已在该集群验证过）：

```python
def _flash2_varlen_allowed() -> bool:
    return os.environ.get("COSMOS_FLASH2_VARLEN", "").strip().lower() in {"1","true","yes"}

if is_varlen and not _flash2_varlen_allowed():   # 原来只是 if is_varlen:
```

`run_fk_point_101.sh` **默认设为 1**。

**为什么默认开、而不是让提交命令带前缀**：那个 `COSMOS_FLASH2_VARLEN=1` 前缀**已经被漏掉过
一次**（FK 线的 `fk-101-16gpu-xxx`），而漏掉的提交看起来完全正常 —— 唯一的信号是日志早期
**一行** `Attention backend selected:`。写在脚本内部的默认值不会被任务模板改写或过滤掉。

### 10.3 实测收益

| 后端 | 每卡 batch 16 的步速 |
|---|---|
| natten | 7.95 s |
| **flash2** | **4.30 s** |

**1.85×**（FK 单模态的数；四模态的绝对值见 §11.2，比例同源）。

### 10.4 风险与回退

⚠️ **上游禁用它的理由是 "due to instability"，这个禁令不是随手写的。** PointFlow 在他们
的 workload 上验过，**FK 的 pack 不同**。实测 1747 步内 loss 与 natten 逐步贴合
（step 100: 9.5712 vs 8.9527；step 182: 3.3759 vs 3.3163，flash2 每次略低），无发散、无 NaN
—— 但 1747 步**不足以证明长程稳定**。

回退：提交命令前加 `COSMOS_FLASH2_VARLEN=0`。

**盯着这几条**：

- loss 形状正常下滑（**不要跟 natten run 逐点比**，不同后端浮点结果本就不同）
- 出现 `NaN`，或 `skip_nan_step` 开始**频繁**触发 → 立刻回退重交
- 步速应稳定；若突然跳回 natten 的量级，说明后端没生效或被覆盖

### 10.5 一个尚未解释的崩溃（与本节无关）

FK 线的 run `fk-101-16gpu-flash2` 在 **iteration 1747** 死于数据管线，不是 flash2：

```
KeyError: (74, 1046)  at singlerighthand_raw_dataset.py:447  frame = self._composed_frame_cache[key]
```

`(episode 74, 帧 1046)`，而 episode 74 = `episode_0065_20260731_142119` 只有 **752 帧**
（真实 mp4 / `state` / `video_frames` 缓存 / manifest 四方一致，101 条 episode 的
`num_frames` 逐条核对**零不符**）。帧 1046 在该 episode 里不存在。

**已排除**：不是 flash2（只有这一次 run 撞到过；`fk-singlerighthand-edge-101` 同样的
101 条数据跑满 10000 步、KeyError 0 处）。

**未查明**：机制。唯一与"跑满 10000 步的那些 run"不同的是**多机（16 ranks）**在
`ActionIterableShuffleDataset.__iter__` 里的分片（`total_shards = shard_world_size × num_workers`）
—— **这是假设，未经证实**。

---

## 11. 四模态特有的实测（两份单模态文档覆盖不到）

### 11.1 `max_iter` 与 `scheduler.cycle_lengths` 必须同步

`LambdaWarmUpCosineScheduler.find_in_interval` 越过最后一个 cycle 后返回 `None`，
`schedule` 随即索引 `lr_warm_up_steps[None]`：

```
omegaconf.errors.KeyValidationError: ListConfig indices must be integers or slices, not NoneType
```

**只抬 `max_iter` 会训到旧上界、然后死在下一步。** 有一个 run 正是死在 step 10001。
`run_fk_point_101.sh` 从**一个变量**同时钉住两者，dry-run 里核对那两个 20000 相等即可。

### 11.2 步速与墙钟（16 卡实测）

**11.7 s/步（含 eval/ckpt 停顿）→ 20000 步 ≈ 65 h。**

这个数用**时间戳差**算（步 100→679 实耗 1.88 h / 579 步 = 11.72；同时段 52→679 得 11.78，
两窗口一致），把 eval/checkpoint 的停顿自动摊进去了，比 `iter_speed` 的单窗口均值可信。

⚠️ **别拿头两步估墙钟。** 步速是**冷热交替**的（数据 I/O），单机 60 步全样本：
**中位 13.49、均值 13.07、区间 6.47~20.82 s**。我最初只看了两个 6.5 s 的样本，据此说了
"6.6 s/步、墙钟 37.8 h" —— **错了，差 2 倍**。16 卡与 8 卡是**同分布、不同数字**
（跨节点梯度 allreduce 只多 ~0.1–0.2 s）。

⚠️ **也别拿 4.3 s/步比** —— 那是 **FK 单模态**的数。四模态多出的部分是 PointFlow 的
Sonata/PTv3 编码器成本。

### 11.3 checkpoint 占用（无保留策略）

**每 500 步一个 × 实测 30 G = 20000 步约 1.2 TB。**

toml 里 `save_iter` 的注释曾写着 *"old checkpoints are deleted as the run goes"* ——
**与代码不符**。实测 `cosmos_framework/utils/checkpointer.py` 里删除类调用 **0 条**；
全仓 `rmtree` 只出现在训练路径之外；三个启动器都不删；`fk-101-16gpu-flash2` 跑完实打实
留着 **5 个完整 ckpt**（各 30 G，`model/optim/scheduler/trainer` 齐全）。

想省：调大 `save_iter`（用崩溃可恢复的粒度换磁盘），或自己定期删旧的 ——
但**删过的 iteration 就 resume 不回去了**（裁剪过的 ckpt 会报 `metadata is None`，
而且它的真实原因被吞掉），所以至少留最近一两个。

### 11.4 scale 纪律

`pointflow_displacement_scale = 0.075255` 与 `fk_displacement_scale = 0.083745`
是**这套选择规则的属性，不是常量**。换 manifest 或换任一选择旋钮 → **必须重测**，
否则整个预测被均匀缩放错，而且**没有任何形状错误**。wrapper 里有守卫，值不一致就拒绝启动。
