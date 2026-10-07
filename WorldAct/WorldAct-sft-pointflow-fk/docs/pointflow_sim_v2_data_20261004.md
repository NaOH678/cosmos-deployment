# Bench2Dex task21 v2：数据与训练记录（更新至2026-10-05）

按 [数据 pipeline](pointflow_data_pipeline.md) 处理，本记录包含数据、VAE 缓存及后续训练与性能实验。

## 当前方案与阅读顺序

- 当前目标：10条（8训练/2验证）验证拟合和四模态联合生成效果，通过后扩至100条。
- 当前配方：stage2、1024点、每卡batch16、累积1、selective Flash2+QKVO、compile，
  8×A800；默认20000步、warmup50、val on start、每500步eval/save。
- 已保留优化：CPU span映射、PB Flash2 varlen、latent快速路径（训练只读首帧RGB）；
  异步数值检查收益不足，已撤回。短测均值5.65s，详见末尾性能对照。
- Action归一化采用Bench2Dex的action-only分位数方案，当前文件来自8条训练数据。
  [两套归一化差异及100条扩容步骤](pointflow_action_normalization_comparison_20261005.md)。
- 启动与产物位置见下文S1；快捷命令见[quickstart](pointflow_quickstart.md#3-启动训练)。

以下按实施顺序保留记录。历史段落中的“尚未验证”“默认”“下一步”描述的是当时状态；
当前配置以上述摘要、S1入口和末尾用户确认约定为准。

## 路径与数据约定

- 数据包解压根：`/data/shichaojian/sim_data/bench2dex_task21_pointfk10_rgb3_v2`
- Cosmos cache：`<根>/datasets/bench2dex-task21-cosmos-cache`
- 新 VAE cache：`<cache>/vae_window_latents/window_manifest.json`
- PointFlow manifest：`<根>/pointflow_outputs/bench2dex_task21/manifest.json`
- 原生 PointFlow 窗口缓存：`<cache>/pointflow_windows`
- 划分：episode 000000–000007 训练（177 窗）；000008–000009 验证（43 窗）。
- 20 Hz，每窗当前帧 + 32 个未来帧；双手 FK 42 点，动作 52 维。
- 三路 RGB 按原生 `compose_dualhand_views` 合成 640×720：俯视在上、双腕在下。
- 原始交付候选点：当前有效 anchor，无未来运动排名，每窗 4424–8202 点；训练另选1024点，见S1。
  数据带 `uses_gt_query_mask=true`，因此不能把“无未来运动选点”解读为在线分割已解决。

窗口只允许 `window_index.json` 中的 220 个 frame_ids 序列。轨迹 ID 是窗口内 ID，
不能拼成整集轨迹，也不能套用单右手 stride=1 的枚举公式。已存在的 labeled、
PointFlow 缓存、FK 和动作保持不变，不重新做跟踪或未来运动选点。

旧 `/mnt/afs/Bench2Dex/data/task21/cosmos-cache/vae_window_latents` 来自另一份
场景外观和相机外参，不复用。相同 episode 名称、qpos 和时间索引不保证视频一致。

## VAE 生成

新增 [工具](../tools/cache_sim_pointfk_window_latents.py)，使用当前工作区原生
`reflection_pad_to_target` 和 `Wan2pt2VAEInterface.encode`。每个窗口单独编码，
batch=1，8 个 GPU worker，seed=42。归一化为 `float32 / 127.5 - 1` 后转 bf16。

```bash
LD_LIBRARY_PATH='' PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/cache_sim_pointfk_window_latents.py \
  --bundle /data/shichaojian/sim_data/bench2dex_task21_pointfk10_rgb3_v2 \
  --vae-path /data/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth \
  --output-root /data/shichaojian/sim_data/bench2dex_task21_pointfk10_rgb3_v2/datasets/bench2dex-task21-cosmos-cache/vae_window_latents \
  --devices 0,1,2,3,4,5,6,7 --seed 42
```

从仓库根运行。已完成的集按源文件、配置及输出 SHA256 校验后跳过；每集写临时
文件、完成独立验证后原子发布，全部完成才写总 manifest。

缓存 schema 为 `sim_pointfk_explicit_window_latents_v1`。**训练读取时必须通过
每集 `start_frames` / `frame_ids` 映射到 NPY 行号，不能把 start_frame 当作行号，
也不能直接指给只支持均匀窗口枚举的旧 loader。** 正式训练 recipe 的接入另做。

## 已完成验证

- 包内 4717 个文件 SHA256 全通过，177 训练窗 + 43 验证窗数据接口通过。
- 全量解码 30 个相机视频，每个视角合计 6932 帧。
- 8×A800 生成 10 集、220 个 latent，单窗口 `[48,9,40,40]`，uint16 存 bf16 位模式。
- 原生空间变换 `image_size=[640,640,640,569]`，包含右侧 reflection padding。
- 数组净数据量 304,128,000 字节，全部数值有限。
- 每集独立验证一个窗口：原始 HDF5 JPEG → 三视图合成 → 空间变换 → 新鲜 VAE 编码，
  与视频缓存和 latent 缓存分别逐位一致（10/10）。不使用有重编码损失的 MP4 当精确输入。
- 全部编码任务已退出，8 张 A800 显存回到 0 MiB。
- 当前工作区重新读取全部 220 窗，PointFlow/FK 打包通过，缓存 miss 为 0；
  帧 ID 严格对应，绝对仿真时间与相对 episode 时间的间隔一致。
- 穷举 177 个训练窗重算 pooled std：PointFlow `0.041089133704910685 m`，
  FK `0.0748351679186022 m`，与交付统计一致，验证集未参与。
- 6 个训练/验证窗口生成三视图 Point/FK 投影 MP4；anchor XYZ 使用归一化内参
  投影并恢复到 640×480 像素后，与 anchor UV 的平均偏差均小于 `0.00003 px`。
  这证明该数据内的投影约定一致，不等于未来轨迹或所有 FK 接触关系均准确。

当前仓库 `PointFlowSource` 原本仅支持 `frame_id / fps` 时间戳；v2 清单显式记录
`timestamp_offset_sec`（例如首集 -1/60 秒）。补齐该字段读取和有限值检查，默认仍为 0，
保留严格错帧检查。回归测试 4 项通过。

数据报告、latent 报告和 Point/FK 投影视频见
`pointflow_outputs/sim_v2_preparation_20261004/`。
完整日志及投影视频也放在数据包根目录的 `preparation_20261004/`。

## 1024 点 FPS 预览

`tools/visualize_sim_point_fps.py` 从当前工作区的 `PointFlowSource` 缓存路径读取
每窗有效 anchors，再仅按当前 XYZ 进行确定性的最远点采样，首点取最接近质心者。
这是当时的第一版选点预览；后续采用FK引导的第二版并重算scale，见S1。

```bash
LD_LIBRARY_PATH='' PYTHONPATH=. OMP_NUM_THREADS=2 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/visualize_sim_point_fps.py \
  --bundle /data/shichaojian/sim_data/bench2dex_task21_pointfk10_rgb3_v2 \
  --episode episode_000000 --points 1024 --output /data/shichaojian/sim_v2_fps1024_preview
```

按选点可视化文档绘制：头相机，左 RGB、右选点，5 帧尾迹，颜色表示窗口内平均
GT 位移（只用于着色，不参与选点）。完整 713 帧、20 FPS；1–706 帧有轨迹覆盖，
第 0 帧和末尾回零帧保留原视频并标注无监督窗口。窗口之间重新选点、重置尾迹，
不把不同窗口的轨迹 ID 接成一条轨迹。JSON 保存每窗选中的原始 point IDs。

产物目录：`/data/shichaojian/sim_v2_fps1024_preview`。

第二版预览增加 `--mode hand_guided --hand-radius 0.05`：先全局 FPS 512 点，
再从当前左右手各 21 个 FK 关键点的 5 cm 邻域中分别 FPS 至多 256 个未选点。
重叠邻域按最近手分配；不足的名额用全局 FPS 补齐，全部点互不重复。
该集 23 窗的 5 cm 邻域候选中位数分别为左手 988、右手 728（扣除全局选点前），
但右手部分窗口可为 0，所以必须允许名额回退，不放大半径强行凑数。
新旧版本使用同一全局 FPS 的运动量颜色尺度。选点仅用当前 anchor 与当前 FK。
产物目录：`/data/shichaojian/sim_v2_handguided1024_preview`，包含新版本完整视频与
旧全局 FPS / 新 FK 邻域 FPS 的双联对比。仍为方案预览，未改变训练缓存。

## S1：FK 引导 1024 点拟合实验

已确认采用第二版选点：512 全局 + 左右手各至多 256 点（当前 FK 5 cm 邻域），
无重复、不足回退全局。选点逻辑在 `cosmos_framework/data/pointflow_anchor_selection.py`，
预览和训练预处理共同调用，不依赖未来位移或未来可见性。

推理约定：当前 RGB、深度、关节状态和相机标定 → 候选表面点与当前 FK → 同一选点。
第一阶段在仿真中允许使用当前 GT query mask，先测策略拟合和闭环；之后再接在线
感知候选区域并匹配训练分布。每次重新规划重新选点，窗口内部固定点身份。

独立训练缓存：`<cache>/pointflow_fk_handguided1024`，由
`tools/prepare_sim_pointfk_training.py` 生成。第一集全部 23 窗的 point IDs 与已确认
的新版视频逐项一致。训练 177 窗重算 scale：PointFlow `0.05269760925058551`，
FK `0.07483516791860219`。沿用已验证的 VAE latents。

训练入口：`examples/launch_sim_pointfk_v2.sh`；配置：
`examples/toml/sft_config/action_policy_sim_pointfk_edge.toml`。
当前保留方案（2026-10-05）：8×A800，每卡 batch16、accum1、seed42；cluster stage2、skip1/2、PB4、local RoPE、
Flash2 varlen、torch.compile、selective Flash2+QKVO checkpoint。FK 使用 42 个独立 ID，
动作 domain28、52 维补齐至64，按训练集统计做可逆归一化。默认固定 20000 step，学习率周期20000、warmup50，
启动 eval 开启，每500 step eval/save；8集训练、2集验证，共2个训练和24个验证窗口；两个验证episode各有early/middle/late三个阶段，
每阶段4个连续窗口，拼接覆盖6.4秒（129帧/20fps，MP4时长6.45秒）。
仿真 eval 使用记录的相机内参，双手骨架分别绘制，不使用真机 D435 投影。

```bash
EXTRA_TAIL_OVERRIDES='' bash examples/launch_sim_pointfk_v2.sh --background
```

默认输出：`/data/shichaojian/runs/sim_v2_handguided1024_stage2_selective16_8gpu_20261005`。
后台 PID 写入 `<OUTPUT_ROOT>/train.pid`；日志位于 `<OUTPUT_ROOT>/logs/`。
持久脚本放 `tools/` 或 `examples/`，检查记录放 `pointflow_outputs/sim_v2_preparation_20261004/`，
不依赖 `/tmp` 下的启动脚本、日志或 PID 文件。

### 历史：stage1首轮与OOM排查

首轮 batch16 启动联合 eval 完成，但首次训练 backward 时显存约 81 GB，
NCCL 报 `Cuda failure 2 'out of memory'`。诊断记录位于
`pointflow_outputs/sim_v2_nccl_probe/`。当时将默认每卡 batch 改为 8，
仍为 accum1；1024 点选点和模型配置保持原配方。

batch8 启动联合 eval + 两步训练通过（2026-10-04），第二步 4.27 s；
首步包含启动验证和编译，不作为稳定速度。日志在
`pointflow_outputs/sim_v2_bs8_startup_probe/logs/action_policy_sim_pointfk_edge_sft.log`。

显存差异核查：selective 配置一致；旧 latent 为 46×34、新为 40×40，视频 token
规模接近。按 Sonata stage1 的 grid_coord // 2 分组统计，旧 sandwich 抽查前8集
首窗为24–45簇（500点），本次177个训练窗为125–233簇、中位183（1024点）。
PointFlow 每簇有1个条件+8个运动 token，故旧抽查216–405个，新1125–2097个，
中位1647；另 FK 从21到42点。点数翻倍不代表 cluster token 只翻倍；本次覆盖
更广使 occupied voxel/cluster 显著增加。统计见同目录 `token_count_comparison.json`。
这些数据解释了序列负担的增加，但尚未对各模块显存增量进行隔离测量。

### 30 step 性能对照（2026-10-04）

两组均8卡、全局batch128、compile开启、完成30次参数更新和checkpoint保存，exit0。
统计rank0第6–29步，排除启动/编译与第30步保存；显存为5秒间隔nvidia-smi采样最大值，
并非PyTorch allocated峰值。正式默认配方尚未随本次性能对照修改。

| AC | 每卡batch | 累积 | 平均 / 中位 step | 采样最大显存 |
|---|---:|---:|---|---:|
| selective Flash2+QKVO | 8 | 2 | 8.377 / 8.330 s | 47.51 GiB |
| full | 16 | 1 | 8.270 / 8.235 s | 58.74 GiB |

输出分别为 `/data/shichaojian/runs/sim_v2_bs8_accum2_30step_20261004` 和
`/data/shichaojian/runs/sim_v2_bs16_fullac_noeval_30step_20261004`。
前者启动eval通过；后者按要求关闭启动eval加快测速。full AC首次带启动eval的测试
由用户要求中止并重启，其目录 `sim_v2_bs16_fullac_30step_20261004` 不作为完成结果。
汇总JSON在 `pointflow_outputs/sim_v2_preparation_20261004/`。

full AC训练期间读取到的rank0第6–19步wall-clock分段均值：dataloader 0.0028 s、
forward 2.6018 s、backward 5.3919 s、optimizer 0.2320 s。
这些计时不是CUDA kernel精确拆分，不能排除模型内部CPU调度或通信空隙；但表明
当前数据加载等待不是主要瓶颈，不宜仅因GPU利用率波动就增加worker。

### Stage2 性能与 PyTorch Profiler

保持1024点、skip1/2、PB4、full AC、每卡batch16、accum1；仅stage1改为stage2。
启动脚本支持 `POINTFLOW_SONATA_STAGE=2`（当时默认仍为1）。177个训练窗的stage2簇数
44–84，中位64，PointFlow token中位576；stage1中位183簇/1647 token。
30步测试关闭启动eval，完成并保存checkpoint；第6–29步均值6.593s、中位6.590s、
范围6.47–6.78s，显存采样最高55.72GiB。比stage1 full AC平均8.270s减少20.3%。
性能不代表生成质量已验证。
输出：`/data/shichaojian/runs/sim_v2_stage2_bs16_fullac_30step_20261004`。

按要求另跑PyTorch Profiler，wait6/warmup1/active2，采集第8–9个更新step，
导出rank0/1，10步正常结束并保存checkpoint。路径：
`/data/shichaojian/runs/sim_v2_stage2_torchprofile_20261004/cosmos3_action/action_sft/action_policy_sim_pointfk_edge/torch_trace/iteration_9/`。
分析脚本：`tools/summarize_training_trace.py`；摘要在
`pointflow_outputs/sim_v2_preparation_20261004/stage2_trace_rank{0,1}_summary.json`。

观察（两步合计；profiler有扰动，不能当作无profiler的节省量预测）：

- rank0/1 CUDA活动并集占时间线88.68%/87.85%，即完全无CUDA活动约11–12%。
  活动包含NCCL等待，不等于有效计算利用率，也不同于nvidia-smi瞬时采样。
- rank0约10883次aten::item、8919次cudaStreamSynchronize、60289次cudaLaunchKernel。
  FlashAttention kernel实际存在，运行进程确认Flash2/varlen/partition全部开启。
- `pointflow_sequence.py:125` 与 `fk_sequence.py:152` 用 `int(remap[...])`
  逐span读取CUDA标量，各2688次/两步，合计5376次。适合改为CPU元数据上的区间映射，
  保持GPU张量重排和跨sample边界检查语义。
- `pointflow_codec.py:120` 的finite检查在decode路径两次累计等待1.58s；
  FK `_sigma` 检查也会同步。该时间含等待此前排队的GPU计算，不能宣称删除检查即可
  加速1.58s；应优先复用已验证输入或评估异步检查，并保留错误检测。
- 最大完全空档约39/45ms位于FusedAdam CPU处理，另有27/31ms位于step尾部。
- rank0 NCCL kernel累计2.40s、rank1约1.19s，存在跨rank等待/工作量差异；
  累计kernel时间有跨stream重叠，不能直接与墙钟相加。

优先方向：消除逐span CUDA标量回读，再检查热路径同步和通信重叠；当前尚未修改
这些训练逻辑，需通过正确性与同配方测速确认收益，不据GPU利用率波动直接加worker。

### CPU span 映射修复（2026-10-04）

已将PointFlow/FK `modality()` 中逐span `int(remap[...])` 改成共享
`sequence_span_remap.SpanRemapper`：从CPU sample长度构造分段偏移，以二分定位索引。
GPU张量重排不变，保留原端点连续性检查（包括跨过零插入sample仍连续的情况）。
13项针对性测试通过，覆盖dense参考映射、空sample、非法索引、两种拼接顺序、
输出与梯度。原有两个完整attention CPU测试无兼容后端，未计入通过数；
随后8卡full AC/batch16/accum1/stage2训练30步及checkpoint保存通过，exit0。

输出：`/data/shichaojian/runs/sim_v2_stage2_cpu_spans_30step_20261004`。
第29步采集profiler，速度统一比较无采集的第6–26步：原均值6.5895s、中位6.59s；
修复后均值6.6110s、中位6.59s。**未测到端到端加速**。
profiler归一到每步：aten::item约5442→2584，cudaStreamSynchronize约4460→1554。
两次采集窗口/stack选项不同，不用profile时长推断加速；调用次数下降与代码改动一致。
修复消除了不必要回读，但剩余计算/通信仍占主导，GPU波动并未因此全部解决。
有限值同步检查暂未更改，避免同时改变错误检测语义和混淆单项结果。
记录：`span_remap_targeted_tests.log`、`cpu_spans_speed_comparison.json`、
`cpu_spans_trace_summary.json`，均位于 `pointflow_outputs/sim_v2_preparation_20261004/`。

### 小 kernel 来源核查（2026-10-05）

使用 `tools/attribute_small_kernels.py` 对带Python stack的原stage2两步trace归因：
kernel External id关联CPU op，反向通过autograd Sequence number追溯前向位置。
短kernel定义为GPU执行时间≤10µs。结果保存在同记录目录
`small_kernel_attribution.json`，不是新训练结果。

- 两步67554个kernel，其中39584个≤10µs。
- PB `pointflow_point_decoder.py:28` 的attention前向15488个kernel、反向16752个；
  短kernel合计18559，占全部短kernel约46.9%。两步归因GPU累计时间约1.174s，
  此值不是可直接节省的墙钟时间。
- Sonata Hilbert encode有2656个短kernel，但两步GPU累计约10.8ms；
  padding/inverse有1130个短kernel，GPU累计约2.0ms。优先级低于PB attention。
- trace明确包含1024次 `aten::_scaled_dot_product_attention_math`，对应全部1024次
  SDPA前向；没有该PB路径的fused SDPA调用。CPU span修复后单步trace仍有512次math SDPA。
- 源码 `pointflow_codec.py:190` 串行8个时间块，PB有4层，
  `pointflow_point_decoder.py:34` 再逐16个sample调用SDPA，故每step共512次。
  输入q/k/v为三维[heads,N,head_dim]，该运行实际选择math后端；
  主干Flash2 varlen环境变量不控制这条独立的PyTorch SDPA路径。

下一优先项：PB attention使用显式Flash2 varlen，把同一时间块内16个sample
打包、由cu_seqlens隔离，前向调用512→32次/step。保持8个时间块串行，先不扩大
跨时间batch，控制显存；需验证输出/梯度数值一致、空segment与ragged边界，再测速。
本次仅定位并记录，未修改PB计算代码。

### PB Flash2实现与速度波动复核（2026-10-05）

随后按用户要求实现PB Flash2 varlen：CUDA bf16/fp16用packed QKV，CPU/float32
保留参考路径；同一时间块各sample由cu_seqlens隔离，空segment跳过，4层共用边界。
14项测试通过（含GPU1 ragged/空样本、输出、梯度和隔离检查），8卡30步训练完成。
输出 `/data/shichaojian/runs/sim_v2_stage2_pb_flash2_30step_20261005`；
第29步trace中math SDPA消失，kernel总数17619。第6–26步均值6.3738s、中位6.41s，
范围5.91–6.75s；对照CPU span版本均值6.6110s，观测降幅3.6%，并非稳定5.94s。

重新读取本轮W&B逐步计时，发现数据等待已不可忽略。rank0第6–26步：

| 时间段 | 平均 | 最小–最大 |
|---|---:|---:|
| dataloader | 0.2150s | 0.0022–0.8036s |
| forward | 2.0861s | 1.7957–2.4500s |
| backward | 3.7908s | 3.7573–3.8402s |
| optimizer | 0.2215s | 0.2034–0.2486s |

21步有11步取数超过100ms。第14步5.91s（rank5为5.94s），取数0.0032s、
反向3.7983s；第24步6.75s，取数0.8036s、反向3.8164s，主要差异由取数等待解释。
CP关闭，trainer计时的fetch直接调用next(dataloader)，不是CP数据广播。
不能继续用旧full AC trace的约3ms取数时间概括本轮，也不能仅凭簇数变化解释波动。
本轮forward也波动，尚未细分是主线程准备还是跨rank等待；GPU时钟未采集，勿推断。

当前每rank4workers、prefetch_factor2、loader batch_size2、in_order=true；
虽使用VAE缓存，dataset仍读取并变换33帧RGB，因此预取供给/CPU处理值得进一步定位。
可先同配方测试增加预取/worker，或缓存确定性视频预处理；尚未证明具体是哪一个环节。
完整逐步证据在 `pointflow_outputs/sim_v2_preparation_20261004/pb_flash2_step_audit.json`。

### 数据加载CPU实测（2026-10-05）

`tools/profile_sim_dataset.py` 使用真实tokenizer、单CPU线程，跨窗口准备20个样本，
首个样本/初始化排除。报告 `dataset_cpu_profile.json`：单样本平均1.1475s，
RGB读取0.3226s、resize0.6090s、pad0.1694s、Point/FK读取0.0058s、latent读取0.0091s。
transform总计0.7840s（包含resize/pad，不能重复相加）。输出RGB uint8
[3,33,640,640]，每样本40,550,400 bytes。单线程串行结果不能当作8rank实际吞吐。

明确开销：`sim_pointfk_dataset.py` 调用read_frame逐帧读取33帧，每次都打开文件并
解析NPY头，然后对全33帧bicubic缩放、反射padding。VAE缓存没有绕过这些步骤。
下个优先项是缓存确定性的resize/pad结果并批量读取连续帧，保留全部真实RGB及
相同image_size/PointFlow投影变换，不能随意删掉未来帧或改条件视频语义。

额外发现：当前max_block_size=4产生47个shuffle block，而8卡×8workers需要64个
分片。直接增加到8workers会有17个worker的固定分片索引永远超出block数，从而
反复切epoch但不yield；不能仅改num_workers。若测试8workers，需要同时把block
拆细（max_block_size=2为91块）或修复空分片处理。当前4workers为32分片无此空分片。
PackingDataLoader异步batch构建默认关闭；未来可测试异步打包/更深预取，但它们
只能重叠工作或缓解抖动，不能消除持续CPU缩放开销。本次没有改训练配置。

### 训练直接使用 latent cache，跳过未来 RGB（2026-10-05）

已修正上节发现的冗余处理，无需缓存全部 resize 后的 RGB。当前 WAM 训练使用
cached VAE latents 作为完整视频目标；RGB 只读取、缩放首帧，保留真实 anchor。
新增 `cached_video_num_frames=33`，文本时长、SequencePlan、packing token 预算仍按
完整窗口计算。image_size 和 PointFlow 投影仿射保持原样。缓存训练也不再计入
未执行的 VAE encoder FLOPs。

`get_sim_pointfk_sft_dataset` 默认启用 `cached_video_only=True`，仅在
`split=train` 且 `iterable_shuffle=True` 时生效；实验配置显式记录此开关。
Point/FK eval 重建 train/val case 时均使用 `iterable_shuffle=False`，因此仍读取
完整33帧，支持原有 joint dream/GT 视频。模型对短 RGB 路径增加限制：必须提供
latent cache、使用 latent_index 时间坐标，不能传入推理条件索引后回退到 VAE。
现有运行中的进程不会自动切换此路径，重启后生效。

验证：`tools/verify_sim_cached_video.py` 在实际 train 窗口0、17、176比较两条路径，
首帧像素逐值一致，除此之外所有样本字段以及真实 collator/packing 输出逐值一致
（新增完整帧数元数据除外）；通过 eval 构造函数核验 train/val 仍为33帧。
记录：`pointflow_outputs/sim_v2_preparation_20261004/cached_video_verification.json`。

`tools/profile_sim_dataset.py --cached-video-only`，单CPU线程、20样本、排除首样本：

| 项目 | 原全33帧RGB | 首帧RGB + latent cache |
|---|---:|---:|
| 单样本准备均值 | 1.1475s | 0.08464s |
| RGB读取 | 0.3226s | 0.04883s |
| resize | 0.6090s | 0.01216s |
| pad | 0.1694s | 0.00241s |
| 输出RGB bytes/sample | 40,550,400 | 1,228,800 |

新报告：`pointflow_outputs/sim_v2_preparation_20261004/dataset_cached_cpu_profile.json`。
两次串行测量存在文件缓存/系统负载差异，不能直接换算8卡step加速。实施时8卡均
被已有任务占用，未中断它们；当时尚未跑GPU训练、反向或完整joint eval生成；后续验证见下一节。

### 缓存训练8卡实测与联合eval验证（2026-10-05）

`/data/shichaojian/runs/sim_v2_stage2_cached_rgb_30step_20261005`：stage2、full AC、
每卡batch16、累积1、compile+Flash2 varlen，30步完成并保存checkpoint，exit 0。
启动eval关闭，在第30步执行联合eval：2个train和6个val case均生成PointFlow/FK视频。
本轮验证执行完整性，不以30步生成质量判断拟合效果。

第6–26步，共21步（不含编译、保存、eval）：

| 项目 | 前版全RGB（PB Flash2） | 首帧RGB+缓存 |
|---|---:|---:|
| step均值 | 6.3738s | 5.8219s |
| step中位数 | 6.41s | 5.83s |
| step范围 | 5.91–6.75s | 5.74–5.90s |
| rank0取数均值 | 0.2150s | 0.002699s |
| rank0取数最大 | 0.8036s | 0.003454s |
| forward均值 | 2.0861s | 1.8019s |
| backward均值 | 3.7908s | 3.7897s |
| optimizer均值 | 0.2215s | 0.2032s |

观测step下降8.7%，吞吐提升约9.5%；反向基本不变，数据等待和forward波动明显下降。
前版第29步开启profile，本版没有；统计均排除该区间，但属于分开运行的比较。
证据：`cached_rgb_speed.json`，可由 `tools/summarize_sim_training_speed.py` 重建。
用户仍观察到利用率瞬降，继续采集优化后的rank0/1训练trace，不能仅凭nvidia-smi
利用率断定缺数据，也不能以rank0取数2.7ms排除其他rank或主线程预处理等待。

### 缓存路径剩余GPU空档（2026-10-05，profile）

独立10步profile：`/data/shichaojian/runs/sim_v2_cached_rgb_profile_20261005`，
第9步采集rank0/1、开启stack和shape，训练成功退出。rank0 trace跨度6.178s，
GPU活动并集5.505s（89.1%）；rank1为88.4%。统计包含NCCL kernel，不等于SM
有效计算占比；profiler也增加CPU开销，不能据此承诺可提速11%。rank0大于1ms
空档64处、合计278ms；最大42ms位于FusedAdam的CPU处理，另有约34ms位于
前向准备、26ms位于EMA相关阶段。HtoD kernel合计仅约3.7ms。

rank0仍有2602次aten::item、1568次cudaStreamSynchronize。标量读取调用栈：
PointFlowCodec._blocks两次inclusive 793.7ms、日志GradScaler读取141.2ms、
FKBranch._sigma四次52.0ms。它们包含等待先前GPU计算完成的时间，不能直接
相加为优化收益。Sonata padding的895次item累计17.4ms；仍值得后续合并边界
读取，但不是几百ms的数据饥饿证据。

先测试PointFlow/FK的有限值和sigma范围检查使用torch._assert_async：GPU保留
检查、不强迫Python读取布尔值；形状检查和CPU ValueError语义保留。CUDA失败
异步报告，可能产生device-side assert。10项CPU codec/FK测试通过，实际性能
另用30步无profile对照判断。工具：`tools/attribute_training_sync.py`。

异步检查对照完成：30步+checkpoint，exit 0；第6–26步均值5.7952s、中位5.78s、
范围5.73–5.92s，取数均值2.76ms。相比5.8219s仅快0.46%，无法排除运行波动。
不保留此实验性改动，已恢复原来的同步检查，避免引入异步报错语义而无明确收益。
报告`cached_rgb_async_speed.json`；输出目录
`/data/shichaojian/runs/sim_v2_cached_rgb_async_checks_30step_20261005`。
接下来测试既有selective AC（Flash2 forward+QKVO形状缓存）在stage2、首帧RGB路径下
能否承载batch16，尝试直接减少反向重算。未修改默认启动脚本配方。

### stage2缓存路径可恢复selective AC、batch16（2026-10-05）

运行`/data/shichaojian/runs/sim_v2_cached_rgb_selective16_30step_20261005`，复用既有
selective配置：`save_ops_regex=["_flash_attn.*forward"]`，
`save_mm_shapes=[[2048,2048],[2048,1024]]`。没有保留异步检查实验改动。
8卡、batch16、累积1、stage2，30步训练完成。第6–26步21步统计：

| 项目 | 缓存+full AC | 缓存+selective AC |
|---|---:|---:|
| step均值 | 5.8219s | 5.6486s |
| step中位数 | 5.83s | 5.64s |
| 范围 | 5.74–5.90s | 5.60–5.71s |
| 取数均值 | 2.699ms | 2.697ms |
| forward均值 | 1.8019s | 1.7941s |
| backward均值 | 3.7897s | 3.6298s |
| optimizer均值 | 0.2032s | 0.1977s |

观测step再降2.98%；相对最初全RGB+full AC的6.3738s下降11.38%，吞吐提高12.84%。
稳态nvidia-smi抽查最大61046MiB（59.6GiB），不是连续采集的峰值，30步没有OOM。
推荐此stage2/batch16配方使用selective AC；不把这个结果外推到stage1或更大点数。
本轮关闭eval；缓存路径的完整联合eval已在前面的full AC运行验证，selective本轮
验证的是训练前向/反向和checkpoint，不声称另外验证了selective的joint eval。
证据：`cached_rgb_selective16_speed.json`。正式长训仍需按任务需求保留val on start。

剩余利用率波动不能直接等同于数据未到：取数2.7ms且拷贝很小，trace显示还存在
主线程准备、优化器/EMA小算子和通信等待。若继续，优先对FSDP通信重叠、Sonata
padding边界批量化做独立profile/对照；本次未实施这些额外改动，也未提高worker数。

### 保留方案（用户确认，2026-10-05）

已将仿真启动脚本默认stage改为2、TOML每卡batch改为16；保留selective Flash2+QKVO、
compile、Flash2 varlen、累积1及latent快速路径（训练仅读首帧RGB，eval读完整33帧）。
正式配方固定20000步、学习率周期20000、val on start开启、每500步eval/save；性能测试的30步和关闭
eval覆盖项没有写入默认配置。默认输出目录改为
`/data/shichaojian/runs/sim_v2_handguided1024_stage2_selective16_8gpu_20261005`，
避免自动续训旧stage1目录。此次只保存配置和记录，没有启动新的长训或提交git。

### 长训步数约定（用户确认，2026-10-05）

后续本项目正式训练默认以20000步为上限，本仿真配方已将`trainer.max_iter`和
`scheduler.cycle_lengths`同步设为20000，warmup仍为50步。每500步评估并保存，
主要结合joint dream中的视频、PointFlow和FK生成质量判断效果；中途确认无效果
可及时停止，有效果则继续训满两万步。不以单一指标设置自动早停。短期性能测试
仍可显式覆盖max_iter，不能把测试步数写回正式训练默认值。

### 每阶段恢复4窗口eval（2026-10-05，用户确认）

之前仿真配置覆盖为val_windows=1，使每个阶段只有33帧/20fps=1.65秒。
用户确认每阶段4窗口即可，已改为val_windows=4。每阶段4个连续33帧窗口，
拼接去掉重复边界帧后为129帧，物理覆盖6.4秒，MP4时长6.45秒，保持20fps。
两个验证episode、early/middle/late三个阶段共24个验证窗口，另有2个训练case；
格式保持joint/joint_dream MP4。每个窗口独立联合生成四模态，不是连续自回归rollout。

运行中的进程须重启续训才会加载新配置；阶段case清单按窗口数命名，新配置使用
fixed_cases_stages_4windows.json，保留旧1windows清单。之前拟用5窗口覆盖8秒的
方案未采用；当前以用户确认的4窗口为准。
