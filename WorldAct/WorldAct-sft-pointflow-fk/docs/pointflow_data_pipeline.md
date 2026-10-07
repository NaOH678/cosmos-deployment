# PointFlow 数据处理 Pipeline 操作手册 — 2026-09-24(2026-09-28 修订:补 9.24 efep 转换步骤;2026-09-30 修订:4a scan 工具并行化 + per-frame 向量)

每接入一批新数据(新任务/新集数/新导出的 flow)都按本文档走一遍。主线实验
记录在 `docs/pointflow_per_point_tokens_20260919.md`,本文档只讲操作。

## 总览

```
raw_data/<dataset>/episode_*/          原始数据(lmdb + videos/{head,right_wrist}.mp4)
pf_out/9.24/<dataset>/efep_seg_v61/    9.24 起:Track4World 3d_efep + SAM2 逐帧导出
episode_*/                             (ragged obs_*.npy + track_label + report.json)
        │
        ▼ 步骤 1b(9.24 起必做)
pf_out/9.24/<dataset>/labeled/         flat labeled 交付(position/uv_px/valid/
episode_*/                             region_labels/frame_indices/timestamps_sec/report.json)
        │
        ▼ 步骤 0(通常已备好,见步骤 0)
datasets/<dataset>-cosmos-cache/       manifest.json + episodes/*.npz + video_frames/
        │
        ▼ 步骤 1-4(本文档主体)
allowlist + pointflow manifest + vae_window_latents + pointflow_windows(3b)
+ scale 标定 + 可视化验证
        │
        ▼ 步骤 5
训练(launch 脚本,env 切换数据集)
```

> 9.24 之前的旧导出(`pf_out/sandwich/labeled/`、`pf_out/dropper_new_9.9/labeled/`
> 等)已随 pf_out 重构**删除**,指向它们的旧 manifest 均为死链,只能用于读
> 历史扫描结论,不能再用于训练。

## 步骤 0:cache 骨架(一次性,通常已存在)

`datasets/<dataset>-cosmos-cache/` 需要 `manifest.json`(集名/帧数/fps/
task_text/arm_action_space)、`episodes/*.npz`(state/action)、`video_frames/`
+ `video_manifest.json`(716×544 uint8 帧)。由 `tools/prepare_singlerighthand_raw.py`
生成:

```bash
.venv/bin/python tools/prepare_singlerighthand_raw.py \
  --raw-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/<dataset> \
  --output-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/<dataset>-cosmos-cache \
  --task-text "<任务指令文本>" \
  --arm-action-space eef    # 或 joint;dropper 用 joint,sandwich 用 eef
```

**arm_action_space 在这里确定**,数据集代码从 manifest 自动读
(`singlerighthand_raw_dataset.py:154`),toml 不用改。dropper 与 sandwich
均已完成此步。

## 步骤 1:episode allowlist

纯文本,一行一个 episode 名,`#` 开头为注释。必须与 raw 目录、labeled 目录、
cache manifest 三方对齐:

```bash
ls /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/<dataset> \
  | grep '^episode_' | sort > examples/pointflow_<dataset>_episodes.txt
```

现有文件:`examples/pointflow_dropper_all_101_episodes.txt`(101 集)、
`examples/pointflow_sandwich_labeled_29_episodes.txt`、`pointflow_sandwich_all_101_episodes.txt`、
`pointflow_sandwich_10_episodes.txt`。

## 步骤 1b:efep_seg_v61 → flat labeled 转换(9.24 起必做)

9.24 的交付是 ragged efep 格式(每帧一串观测,靠 `obs_track` 串成持久轨迹),
训练读的 flat labeled schema 由 `tools/convert_efep_labeled.py` 转出:按整集
有效观测数过滤轨迹(默认 `--min-valid-obs 32`,任何窗口都凑不够有效步数的
轨迹只浪费存储),无效帧用最后已知位置/uv 前填(valid=0,与旧交付的
parked-value 语义一致,幻影守卫照常工作)。纯 CPU,逐帧 seek 读,内存占用小。

```bash
# 单集
.venv/bin/python tools/convert_efep_labeled.py \
  --data-dir /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pf_out/9.24/<dataset>/efep_seg_v61/<episode> \
  --output-dir /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pf_out/9.24/<dataset>/labeled/<episode> \
  --video /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/<raw_dataset>/<episode>/videos/head.mp4

# 批量(约 101 集/数据集,顺序跑即可)
for ep in /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pf_out/9.24/<dataset>/efep_seg_v61/episode_*; do
  name=$(basename "$ep")
  out=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pf_out/9.24/<dataset>/labeled/$name
  [[ -f "$out/report.json" ]] && continue   # 已转换的跳过,中断可续
  .venv/bin/python tools/convert_efep_labeled.py --data-dir "$ep" --output-dir "$out" \
    --video /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/<raw_dataset>/$name/videos/head.mp4
done
```

⚠️ `--video` 必须显式给:efep 的 report.json 不记录视频路径,缺省会把
labeled report.json 的 `video` 写成 `null`,而 `prepare_window` 训练时要用它
打开 head 视频读锚点帧,直接报 "Cannot open head video: None"。已生成的
labeled 可以不重转,把 report.json 里的 `video` 改成
`<raw_root>/<episode>/videos/head.mp4` 即可。

输出目录约定为与 `efep_seg_v61/` 平级的 `labeled/`。产物:position/uv_px/
valid(流式 npy)+ region_labels/query_ids/frame_indices/timestamps_sec +
report.json(含 `native_pixel_queries=640*448`,供步骤 2 的仿射校验)。
转换完成后接步骤 2 重建 manifest,再按 4a 重标 scale(选点分布变了)。

## 步骤 2:pointflow manifest

把 flow 交付挂到 episode 名单上,并记录 tracker 画布 → 训练视频的仿射
(硬编码常数,相机 rig 不变则沿用;换相机必须重新测量并投影验证):

```bash
.venv/bin/python tools/build_pointflow_manifest.py \
  --raw-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/<dataset> \
  --labeled-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pf_out/9.24/<dataset>/labeled \
  --output pointflow_outputs/<tag>/manifest.json
```

⚠️ 旧产物 `pointflow_outputs/dropper_101_20260923/dropper_manifest.json`、
`pointflow_outputs/manifest_sandwich_labeled_20260921.json` 指向的 labeled
目录已随 9.24 重构删除,**失效,仅供查历史扫描结论**;9.24 数据必须按
步骤 1b 转换后重建 manifest。

已知坑:老版 labeled 交付的 report.json 可能缺 `native_pixel_queries`
(dropper episode_0001),`pointflow_window.py::_episode_metadata` 已回退为
校验首帧 uv 是否越界 640×448;若报 "labeled uv outside the tracker grid"
说明这批数据的 tracker 画布不是 640×448,仿射常数必须重测。

## 步骤 3:window latent cache(GPU)

训练从 cache 读每个窗口自己的 VAE latent(与无 cache 逐位一致)。缺
`<cache>/vae_window_latents/` 时必须跑这步,8 卡 101 集约 40-50 分钟:

```bash
bash examples/cache_dropper_window_latents.sh        # dropper 101 集
# 或通用形式(注意:两个 env 必须和 bash 在同一行,换行粘贴会丢):
VIDEO_CACHE_ROOT=<cache路径> ALLOWLIST=<allowlist> bash examples/cache_singlerighthand_window_latents.sh
```

脚本内 env:`WORKERS`(默认 8)、`DEVICES`、`BATCH_SIZE`(默认 1,不要调大,
batch>1 在该 VAE 下结果不逐位一致)、`OVERWRITE`。已处理的集会跳过,中断后
重跑同一条命令即可续。

## 步骤 3b:pointflow 窗口缓存(9.29 起,CPU,~8 分钟/101 集)

`prepare_window` 在线计算每窗口要读 ~205MB 全量轨迹 + 每窗口一次视频 seek,
大规模训练时数据 I/O 会顶穿共享存储(101 集实测 step 8s/20s 冷热交替)。
窗口缓存把选点结果预计算成小文件(~120KB/窗口,101 集约 23GB),训练时
直接读;miss 自动回退在线计算(结果逐位一致)。

```bash
PYTHONPATH=. .venv/bin/python tools/build_pointflow_window_cache.py \
  --manifest pointflow_outputs/<tag>/manifest.json \
  --output <cache路径>/pointflow_windows \
  --episode-allowlist <allowlist> \
  --top-n 500 --region-quotas 2:0.40,3:0.45,4:0.15 \
  --min-voxel-members 3 --min-valid-steps 16 --phantom-guard \
  --workers 32
```

- 流式单遍构建(轨迹每集读一遍 + 视频一遍前向解码),与在线逐窗口逐位一致;
- `cache_manifest.json` 记录枚举+选点全部配置,训练侧校验不一致直接拒绝——
  **选点配置变了要重建**(改目录或 `--overwrite`);
- 训练侧开关:`POINTFLOW_WINDOW_CACHE_ROOT=<cache路径>/pointflow_windows`;
- scale 扫描加 `--window-cache-root` 后从缓存读,秒级完成。

## 步骤 4:选点相关的四件套

### 4a. scale 标定:`scan_pointflow_selection.py`(选点规则每变一次就必须重测)

`pointflow_displacement_scale` = 米/模型单位,意图是让监督位移的 std ≈ 1
与单位噪声匹配。**选点规则、数据集、点数任一变化都要重测;跨 scale 不可
resume,必须新 OUTPUT_ROOT。**

工具同时产出**标量**候选和 **per-frame per-channel 向量**(96 值,供
`POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE`,原理见
`pointflow_displacement_scale_20260914.md` §14):

```bash
PYTHONPATH=. .venv/bin/python tools/scan_pointflow_selection.py \
  --manifest <manifest.json> \
  --cache-root <cache路径> \
  --episode-allowlist <allowlist> \
  --top-n 500 --min-voxel-members 3 --min-valid-steps 16 \
  --region-quotas 2:0.40,3:0.45,4:0.15 --phantom-guard \
  --windows-per-episode 256 --workers 64 \
  --window-cache-root <cache路径>/pointflow_windows \
  --output <scan.json>
```

**并行设计(2026-09-30 重写)**:按 episode 分片 + `ProcessPoolExecutor`;
统计不保留原始值,改为**直方图(0.5mm 桶,+0.6m 量程)+ (n, sum, sumsq)
累加**,worker 合并只是数组相加。因此:

- std **精确**且与 worker 数无关(32-worker vs 串行逐值差 = 0,已验证);
- 分位数(q01/q99/尾部)从 0.5mm 桶读,误差远低于下游噪声;
- 内存恒定(每 worker 几 MB 直方图),不随窗口数增长;
- 速度:101 集 × 64 窗口 19 分钟 → **32 秒**(32 workers);× 256 窗口 51 秒。

**三个输出文件**(`--output <scan.json>` 时):

| 文件 | 内容 |
| ---- | ---- |
| `<scan.json>` | 每窗口一行(点数/体素/运动/间距/退化等),供表格与排查 |
| `<stem>_frame_scales.json` | per-frame per-channel 的 std/robust/样本数,供分析 |
| `<stem>_frame_scales_env.json` | **模型直接读取的规范文件**(`steps/channels/stat/scales` 96 值),多选点配置时按 label 分文件;打印里给出 `POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE=<绝对路径>` |

**采样密度与收敛协议**:窗口是训练语义的真窗口(同一 `PointFlowSource`),
用 linspace 等距抽样,**不要**用训练那种 stride=1 的全重叠集合。密度按
**收敛序列**判定而不是拍脑袋:两档密度对比,逐值偏差进 1% 即收敛。
基准数据(101 集 strat500):w64/w8 偏差 max 13%(8 不够),w256/w64 偏差
mean 0.4%(64 已收敛)——**以后一律 `--windows-per-episode 256 --workers 64`**。

历史标定点:sandwich 旧 manifest 0.0432 → sandwich labeled29+mvs16 0.0740 →
dropper 101 集 top300+mvs16 0.0482 → sandwich 101 集 strat500+守卫 **0.0528**
(per-frame 向量见 scale 文档 §14)。

### 4b. 幻影漂移扫描(数据质量体检,两个工具是一对)

幻影点 = uv 锁死(移动 <2px)但 3D 漂移,是深度估计失败,不是真运动。
两个扫描互补:

- **原始总体扫描** `scan_pointflow_phantom_raw.py`:直接读原始数组,按
  ~267ms 步长统计"幻影步"占比(阈值 40mm/2px),不过训练路径,衡量的是
  **交付数据本身**的总体质量。注意 `--labeled-root` 默认硬编码 sandwich,
  扫别的数据集必须显式传:

```bash
.venv/bin/python tools/scan_pointflow_phantom_raw.py \
  --allowlist <allowlist> \
  --labeled-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pf_out/9.24/<dataset>/labeled \
  --output <scan.json>
```

- **训练路径扫描** `scan_pointflow_phantom_drift.py`:走 `prepare_window`
  真路径(含选点),衡量**选点之后模型实际吃到**的幻影暴露(阈值 30mm/2px,
  窗口级):

```bash
.venv/bin/python tools/scan_pointflow_phantom_drift.py \
  --manifest <manifest.json> --allowlist <allowlist> \
  --select-top-n 300 --min-voxel-members 3 --select-min-valid-steps 16 \
  --windows-per-episode 12 --output <scan.json>
```

两者都按语义区域拆分(L1 指尖 / L2 手 / L3+ 物体)。训练路径参考值:
sandwich 29ep 总计 12.1%(集中在物体);dropper 100ep 总计 23.4%(集中在
手部 30.2%)。注意幻影指标覆盖不了"低漂移背景噪声点"(dropper 的幕布块,
漂移 ~20mm 低于阈值)——那个要靠 4d 的无过滤全量渲染看。

### 4c. 幻影守卫(训练侧选点降级)

`prepare_window` 的 `select_phantom_guard=True` 把幻影点在运动排名中降级
(与 min_voxel_members 守卫同机制,阈值 30mm/2px 与扫描一致)。读未来标签,
属训练期规则。**目前只接到 PointFlowSource 构造参数,尚未接 recipe env,
首轮训练默认不开。**

### 4d. 选点可视化验证(每次接入新数据必做)

`tools/visualize_pointflow_selection.py` 一个工具两种用法:

**默认用法 = 看训练选到什么**(走 `PointFlowSource.load` 真路径,选点、
种子、对齐断言全是训练代码本身):

```bash
.venv/bin/python tools/visualize_pointflow_selection.py \
  --manifest <manifest.json> --episode <episode名> \
  --start-frames 100 600 --output <输出目录>
# 加 --phantom-guard 渲染守卫开启后的对照(文件名带 _noghost)
```

**无过滤用法 = 看交付里有什么**(上限拉到超过全集点数、守卫全关,
`prepare_window` 全保留;锚点有效性/法线检查仍生效,那是数据不是过滤):

```bash
.venv/bin/python tools/visualize_pointflow_selection.py \
  --manifest <manifest.json> --episode <episode名> \
  --start-frames 0 400 800 1200 \
  --max-points 20000 --select-top-n 20000 \
  --min-voxel-members 0 --min-valid-steps 0 --output <输出目录>
```

产物 mp4/gif/mid.jpg:左原帧、右叠加(实心圆点 + 5 帧尾迹,颜色=运动量,
红=快)。核对三件事:点是否准确落在手/物体上(仿射验证);守卫是否剔掉
幻影簇;无过滤版里有没有大块背景点(如 dropper 的幕布块)混在交付里。
参考:ep0002 无过滤 w0 选出 6043 点,w400 只剩 2279 点且存活率 0.61
(轨迹中途大量死亡)。

## 步骤 5:训练启动

```bash
bash examples/launch_pointflow_dropper101.sh          # dropper 101 集
bash examples/launch_pointflow_labeled29_sandwich.sh  # sandwich labeled 29 集
```

脚本钉死了 dataset/cache/manifest/allowlist/scale override;`OUTPUT_ROOT`
必须每次显式换(防目录冲突与 resume 混淆),`NPROC_PER_NODE` 默认 8。
WANDB 默认 offline。

## 运维规则速查

1. 选点规则/数据集/点数任一变化 → 重测 scale(4a,`--windows-per-episode 256 --workers 64`)+ 新 OUTPUT_ROOT;
2. 跨 scale 的 loss 读数不可比,判据只看 eval ADE/zero;
3. eval case 跟随 OUTPUT_ROOT 生成(fixed_cases),换数据集后旧 case 不可比;
4. 换相机 rig → 仿射常数重测 + 4d 投影验证;
5. eval 产物在 `<OUTPUT_ROOT>/cosmos3_action/action_sft/action_policy_singlerighthand_edge/pointflow_eval/step_*/`,缺失的拼接视频用 `tools/stitch_pointflow_canvas.py --eval-dir <该目录>` 离线补;
6. per-frame 向量文件(`*_frame_scales_env.json`)与训练绑定,resume 必须同一路径;重扫写新文件名,别原地覆盖。
