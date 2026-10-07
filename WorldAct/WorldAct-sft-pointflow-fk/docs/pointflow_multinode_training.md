# PointFlow 多机训练(SenseCore 集群)

> 本文记录多机训练的脚本兼容层、HSDP 配置、提交命令与排障。环境为 SenseCore
> 调度的 2+ 节点 A800 任务;单节点用法见 [pointflow_quickstart.md](./pointflow_quickstart.md)。

## 1. 启动链路

```
examples/launch_pointflow_sandwich101.sh        # wrapper:钉住集群路径+训练配方+拓扑推导
  └─ examples/launch_sft_action_policy_singlerighthand_edge.sh   # env 默认值+输入检查
       └─ examples/_sft_launcher_common.sh      # torchrun 参数组装(多机在这里)
            └─ torchrun -m cosmos_framework.scripts.train --sft-toml=...
```

torchrun 跨节点组网需要 5 个量:`--nproc_per_node`、`--nnodes`、`--node_rank`、
`--master_addr`、`--master_port`。单机时全部缺省即可;多机时前三个必须每节点
正确、后两个全任务一致。

## 2. 本集群的注入机制(实测)

SenseCore 的 PyTorch 任务(2 节点)实际注入:

| 变量 | 例子 | 说明 |
| ---- | ---- | ---- |
| `NNODES` | 2 | 节点数(launcher 直接消费) |
| `MASTER_ADDR` | `job-xxx-master-0.job-xxx` | pod.service 短名 |
| `MASTER_PORT` | 23456 | rendezvous 端口 |
| `RANK` | 0/1 | pod 序号(master=0,worker-N=N) |
| `WORLD_SIZE` | 2 | 注意:是**节点数**,不是 GPU 数 |

**不注入**:`NODE_RANK`、`SENSECORE_PYTORCH_*`(文档示例里的名字,交互 pod 和
部分任务类型里没有)。

`_sft_launcher_common.sh` 的解析顺序(`SENSECORE_PYTORCH_*` → 显式 `NNODES`/
`NODE_RANK` → pod 名兜底):

1. `NNODES`/`NODE_RANK` 优先取显式设置,其次 `SENSECORE_PYTORCH_NNODES`/
   `SENSECORE_PYTORCH_NODE_RANK`;
2. 若 `NNODES` 已给但 `NODE_RANK` 仍空,按 Kubeflow 命名从 `HOSTNAME` 推导:
   `<job>-master-0` → rank 0 且自己当 rendezvous 主机;`<job>-worker-N` →
   rank N+1、`MASTER_ADDR=<job>-master-0`;
3. `MASTER_ADDR`/`MASTER_PORT` 透传,缺省 50012。

## 3. 并行拓扑(HSDP)

约束:`shard × replicate × CP = WORLD_SIZE`(world = 节点数 × 每节点卡数)。

wrapper 自动推导:**shard = 每节点卡数、replicate = 节点数**,即节点内 FSDP
分片(NVLink,每步多次 all-gather,延迟敏感),节点间只做梯度 allreduce(每步
一次,量是分片后的)。不要 shard 跨节点——那会把每层 all-gather 推上 IB。

| 规模 | shard | replicate | 全局 batch(batch 16/卡) |
| ---- | ----- | --------- | ----------------------- |
| 8 卡单机 | 8 | 1 | 128 |
| 16 卡(2 节点) | 8 | 2 | 256 |
| 32 卡(4 节点) | 8 | 4 | 512 |

## 4. 提交命令

任务系统(PyTorch 任务模板)的启动命令栏(bash -c 模式)填一行:

```bash
OUTPUT_ROOT=/data/shichaojian/runs/<新目录> NNODES=<节点数> bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/examples/launch_pointflow_sandwich101.sh
```

- `OUTPUT_ROOT` 必填且每次换新(wrapper 未设置会拒绝启动);resume = 同名重跑;
- `NNODES` 显式给(兜底推导要用);平台若注入了 `SENSECORE_PYTORCH_NNODES` 则可省略;
- 其余一切(数据/模型/选点/flash2/FSDP/联合去噪 eval)都在 wrapper 里有默认值,
  要改哪个就用同名 env 前缀覆盖,例如 `POINTFLOW_SELECT_TOP_N=300 ...`;
- 每个节点都会执行同一条命令,不需要手动区分 rank。

## 5. 起后验证清单

控制台开头(wrapper 回显):

- `topology: shard=8 replicate=2`(数字须符合上表);
- `[topology-env] NNODES=... / MASTER_ADDR=...`(平台注入的实际值);
- 兜底生效时有一行 `>>> topology fallback: HOSTNAME=... -> NODE_RANK=...`。

训练日志(`<OUTPUT_ROOT>/logs/action_policy_singlerighthand_edge_sft.log`):

- `RankPartitionedDataLoader allocation (16 GPUs)` —— 16 卡组网成功;
- "PointFlow branch installed" 每 rank 一次,不再出现双份交错(双份 = 两个
  独立任务在共享日志,见排障);
- step 时间:batch 16/卡时 ~8s/步为正常(A800+flash2)。

## 6. 排障

| 症状 | 原因与处理 |
| ---- | ---------- |
| 日志里 `world_size 8`、双份 rank 输出交错 | 两节点没组网,各自单机跑了(注入缺失且没给 NNODES)。杀掉,按第 4 节重交。**两任务共享 OUTPUT_ROOT 会在 save 时互相写坏** |
| `[c10d] ... hostname ... err=-3` | k8s pod 主机名反查失败,**良性警告**,忽略 |
| `IPv6 network addresses ... cannot be retrieved (gai error)` | c10d 先试 IPv6 失败,会自动回退 IPv4,**良性**;若随后卡死才是真的 DNS 不通 |
| rendezvous 卡住/超时 | worker 解析不到 master 名:把 `MASTER_ADDR` 换成 FQDN(`<pod>.<svc>.<ns>.svc.cluster.local`)或 master pod IP |
| 组网成功但 step 时间翻倍 | 跨节点走了 TCP 而非 IB:任务 env 加 `NCCL_DEBUG=INFO` 重启,看 `NET/IB` 还是 `NET/Socket`,按集群文档补 `NCCL_IB_HCA` 等 |
| step 时间 8s/20s 冷热交替 | 数据 I/O,不是网络:确认 `POINTFLOW_WINDOW_CACHE_ROOT` 指向已构建的窗口缓存(见 quickstart 第 3 节) |
