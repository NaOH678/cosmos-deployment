# PointFlow 数据处理 Pipeline 操作手册 — 2026-09-24

每接入一批新数据(新任务/新集数/新导出的 flow)都按本文档走一遍。主线实验
记录在 `docs/pointflow_per_point_tokens_20260919.md`,本文档只讲操作。

## 总览

```
raw_data/<dataset>/episode_*/          原始数据(lmdb + videos/{head,right_wrist}.mp4)
pf_out/<export>/labeled/episode_*/     Track4World flow 交付(position/uv_px/valid/
                                       region_labels/frame_indices/timestamps_sec/report.json)
        │
        ▼ 步骤 0(通常已备好,见步骤 0)
datasets/<dataset>-cosmos-cache/       manifest.json + episodes/*.npz + video_frames/
        │
        ▼ 步骤 1-4(本文档主体)
allowlist + pointflow manifest + vae_window_latents + scale 标定 + 可视化验证
        │
        ▼ 步骤 5
训练(launch 脚本,env 切换数据集)
```

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

## 步骤 2:pointflow manifest

把 flow 交付挂到 episode 名单上,并记录 tracker 画布 → 训练视频的仿射
(硬编码常数,相机 rig 不变则沿用;换相机必须重新测量并投影验证):

```bash
.venv/bin/python tools/build_pointflow_manifest.py \
  --raw-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/<dataset> \
  --labeled-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pf_out/<export>/labeled \
  --output pointflow_outputs/<tag>/manifest.json
```

现有产物:`pointflow_outputs/dropper_101_20260923/dropper_manifest.json`、
`pointflow_outputs/manifest_sandwich_labeled_20260921.json`。

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

## 步骤 4:选点相关的四件套

### 4a. scale 标定(选点规则每变一次就必须重测)

`pointflow_displacement_scale` = 米/模型单位,意图是让监督位移的 std ≈ 1
与单位噪声匹配。**选点规则、数据集、点数任一变化都要重测;跨 scale 不可
resume,必须新 OUTPUT_ROOT。**

```bash
.venv/bin/python tools/scan_pointflow_selection.py \
  --manifest <manifest.json> \
  --cache-root <cache路径> \
  --episode-allowlist <allowlist> \
  --top-n 300 --min-voxel-members 3 --min-valid-steps 16 \
  --windows-per-episode 8 --output <scan.json>
```

取输出里 `std` 候选。历史标定点:sandwich 旧 manifest 0.0432 → sandwich
labeled29+mvs16 0.0740 → dropper 101 集 top300+mvs16 0.0482。

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
  --labeled-root /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pf_out/<export>/labeled \
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

1. 选点规则/数据集/点数任一变化 → 重测 scale(4a)+ 新 OUTPUT_ROOT;
2. 跨 scale 的 loss 读数不可比,判据只看 eval ADE/zero;
3. eval case 跟随 OUTPUT_ROOT 生成(fixed_cases),换数据集后旧 case 不可比;
4. 换相机 rig → 仿射常数重测 + 4d 投影验证;
5. eval 产物在 `<OUTPUT_ROOT>/cosmos3_action/action_sft/action_policy_singlerighthand_edge/pointflow_eval/step_*/`,缺失的拼接视频用 `tools/stitch_pointflow_canvas.py --eval-dir <该目录>` 离线补。
