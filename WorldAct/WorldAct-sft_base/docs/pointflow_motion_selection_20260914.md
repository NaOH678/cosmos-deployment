# PointFlow 运动选点(GT 排序取 top-X% 点)

**日期**:2026-09-14
**状态**:已实现并验证(10 集数据扫描 + 80 窗口可视化)
**代码归属**:标注 `[本仓库]` 的是本仓库自有代码;`action_sft_dataset.py` 是 NVIDIA 同步文件,本次只做参数透传。
**与其它文档的关系**:
- `pointflow_alignment_audit_20260913.md` —— 位置编码核查。本文是**数据侧**的另一个改动,不重叠。
- `pointflow_window_latent_cache_20260913.md` —— 逐窗口 latent 缓存(同一天的另一件事)。
- `pointflow_eval_visualization.md` —— eval 可视化的既有说明。本文补充**选点之后**它的变化。
- `pointflow_bugfix_log_20260912.md` —— 更早的缺陷修复。
- **`pointflow_fit005_analysis_20260914.md`** —— 用本方案跑出的 7320 步训练分析。
  它指出:选点让信号强 10 倍,但**有效监督只剩 18%**(选中的快速点未来经常被遮挡/出画),
  且采样轨迹从未赢过 zero 基线(根因在采样器,不在选点本身 —— 见该文 §6–§8)。

---

## 1. 为什么要选点

目标是**验证 PointFlow 分支能不能学会预测运动**,而不是训练一个完整可用的模型。两个障碍:

**① 点太多。** 每个窗口 8192 个点、PTv3 之后 ~450 个簇、point token = `9 × K`。

**② 大部分点几乎不动。** 实测(10 集 × 6 窗口):

```
每点平均位移幅度     p10=0.24cm  p50=0.93cm  p90=2.55cm  p99=5.89cm  max=24.7cm

>0.1cm  99.99%        ← 几乎没有真正静止的点,但 p10 那批大概率是逐帧深度噪声
>1cm    46.00%
>2cm    23.63%
>5cm     1.98%

全点平均 = 7.6 ~ 13 mm     ← zero_ade_mm 的量级
```

**预测全零点就能拿到 7.6mm 的 ADE**,而这个分数里一大半是不可学的噪声。指标被稀释了。

⇒ 只拟合**真正在运动**的点。

---

## 2. 方案:按 GT 位移取 top-X% 点

### 2.1 实现位置

`cosmos_framework/data/pointflow_window.py::prepare_window` `[本仓库]`,参数 `select_motion_fraction`。

放在 `prepare_window` 里,是因为它同时持有 anchor、未来帧和体素映射 —— 选点需要的一切都在这个函数内,而且已经按 `sha256(seed:episode:frame)` 定种子,**天然确定**。

```python
if select_motion_fraction > 0 and len(ids):
    magnitude = np.linalg.norm(target, axis=-1)              # [steps, N]
    counts = target_valid.sum(0)                             # [N]
    per_point = np.where(counts > 0,
                         (magnitude * target_valid).sum(0) / np.maximum(counts, 1), 0.0)
    keep_count = max(3, int(round(select_motion_fraction * len(ids))))
    order = np.argsort(-per_point, kind="stable")
    keep_point = np.zeros(len(ids), dtype=bool)
    keep_point[order[:keep_count]] = True
    if keep_point.sum() >= 3 and (~keep_point).any():
        ids = ids[keep_point]
        anchor_xyz, anchor_uv = anchor_xyz[keep_point], anchor_uv[keep_point]
        normals, colors = normals[keep_point], colors[keep_point]
        target, target_valid = target[:, keep_point], target_valid[:, keep_point]
        # 子集化后**重跑体素化** —— 比重新编号既有的映射更省事也更不易错
        shift, centered, grid, representatives, inverse, features = voxelize(anchor_xyz, colors, normals)
```

要点:
- 每点的运动 = **只在该点有效的步上**的平均位移幅度
- 排序 `kind="stable"` ⇒ 平手时按原点序,**完全确定**
- 选完**重跑体素化**,而不是去改 `original_to_voxel` 的编号

### 2.2 为什么不选 top-K 体素(实测否决)

先实现的是"按体素平均运动取 top-K 个体素、保留其成员点"。**实测是坏的:**

```
以最快体素为中心取空间邻域:
  w0   : 最快体素 181mm,R=8cm → 15 点(运动被稀释到 14mm)
  w600 : 最快体素 1516mm(追踪伪影),R=15cm → 1 点
  w300 : 最快体素 248mm,R=15cm → 2 点

对比(同一批窗口):
  top-K 体素 K=4..32 :   4 ~ 33 点,  间距 77–302 mm,  中位运动 147–338 mm
  top 5% 点          :       409 点,  间距  4–10 mm,   中位运动  48–99 mm
```

两个原因:
1. **最快的体素是孤立的离群点** —— w600 那个"运动 1516mm"显然是伪影,周围一个点都没有。极值尾部选的是伪影,不是运动部件。
2. **把一个体素的其余成员丢掉之后,点集太稀,PTv3 没有邻域可池化**(间距 77–302mm)。

**运动在空间上是相关的**(手臂整块在动),所以"最快的 5% 点"彼此挨得很近;而"最快的 4 个体素"不是。

### 2.3 为什么按"比例"而不按绝对计数

`N = min(max_points, 可用点数)` 实测恒为 8192 ⇒ **`X%` 直接给出恒定的点数**(2%→164、5%→410、10%→819)。

这很重要:ragged pack 的大小非常稳定。top-K 体素那版是 4~33 乱跳。

---

## 3. 实测:10 集扫描

```bash
.venv/bin/python tools/scan_pointflow_selection.py \
  --manifest pointflow_outputs/task5/mixed_manifest.json \
  --cache-root .../singlerighthand-sandwich-100-cosmos-cache \
  --episode-allowlist examples/pointflow_sandwich_10_episodes.txt \
  --fractions 0,0.02,0.05,0.10 --windows-per-episode 6
```

纯 CPU(选点在 PTv3 之前,不需要 GPU)。

```
  select   pts min  pts p50  pts max  vox p50  motion p50  motion max  间距 p50  退化窗口
  off          0     8192     8192     4715         7.6      3035.6       8.5        7
   2%          0      164      164      123        91.2      3035.6      19.4        7
   5%          0      410      410      272        62.8      3035.6      12.5        7
  10%          0      819      819      602        47.7      3035.6      14.7        7
```

三个要点:

**① 退化窗口 = 7 在每一档都一样,包括 off** —— 那 7 个是**本来就空的窗口**(全量也是 0 点),不是选点造成的。

**② 信号** —— `motion p50` 从全量的 **7.6mm** 升到 **91 / 63 / 48 mm**,强 6–12 倍。

**③ `间距` 那一列看着非单调**(19.4 → 12.5 → 14.7),**不是 bug**。
它是"跨 60 个窗口的中位数的中位数",而**各档统计的是不同的点群体**:全量的中位数一半来自稀疏的墙面/背景,选中的中位数全来自密集的手臂区域。**单窗口实测是 4–5mm,比全量的 6.5mm 更密**:

```
全量 median NN  6.48 mm
top  2%         4.52 mm
top  5%         4.01 mm
top 10%         5.04 mm
```

**选出的点比全量还密**,因为运动快的点都在近处 → PTv3 有充分邻域 ✓

嵌套关系正确:`0.02 ⊆ 0.05 ⊆ 0.10`(逐窗口验证过)。

---

## 4. 可视化验证

### 4.1 选点接到可视化链路上

`cosmos_framework/scripts/validate_pointflow_sonata.py` 新增 `--select-motion-fraction`,传进 `prepare_window`。因为选点在 PTv3 **之前**,所以**编码、簇、渲染全部基于选中的点** —— 和训练数据路径一致。

```bash
bash examples/launch_pointflow_motion_scan.sh
```

一条命令:10 集 × 每集按比例铺开的 9 个窗口,每窗口两步:

```
① validate_pointflow_sonata.py --select-motion-fraction 0.05  → window.npz + encoding_enc3.npz
② visualize_pointflow_motion.py --top-motion-clusters 8       → .mp4 / .gif / _preview.jpg
```

**不传固定簇 id**:簇 id 只在产生它的窗口内有效(PTv3 从那个窗口的点云导出)。同一 episode 窗口 0 有 454 个簇、帧 300 起有 312 个 —— **id 完全不通用**。所以选簇是一条**逐窗口的规则**(`--top-motion-clusters` 按该窗口自己的 GT 运动排序),不是列表。

### 4.2 80 个窗口的实测结果

```
  指标              min    p25    中位    p75     max     均值
  选中簇数          3.0    6.0    7.0    8.0     8.0     6.7
  成员数           44.0  113.0  201.0  277.2   389.0   194.7
  平均运动 mm      29.6   62.8   75.1   98.4   243.0    88.0
  最小运动 mm      25.0   48.8   59.0   79.2   162.0    67.1
```

- **每个窗口信号都够**:平均运动 29.6–243mm,只有 1 个窗口低于 30mm
- **簇的成员数是真的**:44–389(中位 201),不是空壳
- **top-8 基本能满足**:39/80 给满 8 个,中位 7 个
- **连最慢的选中簇都在动**:每窗口最小运动 25–162mm

---

## 5. 训练侧

### 5.1 开关

```bash
POINTFLOW_SELECT_MOTION_FRACTION=0.05 bash examples/launch_sft_action_policy_singlerighthand_edge.sh
```

透传链:`toml/env` → `action_policy_singlerighthand_edge.py` → `action_sft_dataset.py` → `singlerighthand_raw_dataset.py` → `PointFlowSource` → `prepare_window`。

### 5.2 开销(实测)

```
                              off        fraction=0.05     差
prepare_window               337.9 ms      338.8 ms      +0.9 ms  (+0.3%)
PointFlowSource.load         340.1 ms      343.4 ms      +3.3 ms  (+1.0%)

下游张量(每样本)
  点云 + displacement         3.14 MB       0.16 MB      小 20×
```

- 340ms 里 **337ms 是读帧**;选点只占 3ms
- `prepare_window` 跑在 dataloader worker 里(`num_workers=8`),不在关键路径
- 本配方的注释记着:**这台机器上 6MB 的张量拼接,空闲时 2.6ms,有内存压力时 230ms** —— 瓶颈是分配/搬运,小 20 倍直接打在这上面

**GPU 侧也省,但没 20 倍**:86% 的 token 是 video,point token 从 `9×K` 变小,粗估 attention 工作量降约 30%。(沙箱无 GPU,这一条是推断,不是实测。)

---

## 6. eval 侧

### 6.1 选点**同样生效**

eval 的 case 由 `pointflow_eval_cases.stage_windows(dataset, …)` 选取 —— **经过 dataset**,所以走同一条 `PointFlowSource.load` → `prepare_window`,选点自动应用。

⇒ eval 报的指标是**在选中的点上**算的,和训练一致。

### 6.2 输出

每个 case 一个目录 `pointflow_eval/step_XXXXXXX/<case_id>/`:

| 文件 | 内容 |
|---|---|
| `comparison.mp4` | 三格动图 33 步:`GT \| tracker UV` / `Pred` / `GT + Pred` |
| `comparison.png` | 三格 × 4 个时间点 |
| `error_map.png` | 首帧 + 每点平均 ADE 着色 |
| `position_grid.png` | 点 token 落在视频网格何处(画布对齐诊断) |
| `prediction.npz` | 原始 record |

### 6.3 ⚠️ 三个掩码塌缩成一个

`trajectory_metrics` 按三个掩码分别报:

```python
for label, mask in [("all", valid), ("moving", valid & moving[None]), ("static", valid & ~moving[None])]
# moving = max|gt| >= 1cm
```

选中的点每窗口**最小运动 25–162mm**,全部 ≥ 1cm ⇒

```
moving == all
static_count == 0
static_drift_mm 不再上报(有 if static.any() 保护)
```

渲染里"静态上下文点"也消失(`render_case` 按 moving/static 分组显示),画面只剩运动点的轨迹。

### 6.4 ⚠️ 开/关选点之后,指标不能横向比较

```
不开选点:  all_ade_mm = 在 8192 个点上平均(大部分几乎不动)
开选点:    all_ade_mm = 在 410 个运动点上平均
```

**同名不同义。** 要对比必须两次跑同样的配置。这与 §1 里 `zero_ade` 被稀释是同一个问题。

---

## 7. 已知边界

**① 每集末尾的窗口是空的。** 实测 9 个窗口报 `Fewer than 3 valid observed anchor points` —— 那些帧的点云本来就是空的(数据集性质,不是选点造成的)。可视化脚本现在会安静跳过并说明原因。训练侧用 `allow_empty=True`,这些窗口作为空窗口保留。

**② 这是诊断,不是可部署的规则。** 排序读的是**未来 GT**;推理时没有未来,选不出来。所以用它训出来的模型不能直接回到推理。目的是回答"这条分支能不能学会运动"。要可部署,得换成空间规则(如 `--focus-uv` 那种按图像位置选),那是另一件事。

**③ `select_motion_fraction=0` 是默认值**,即完全保持原行为。所有改动都是 opt-in。

---

## 8. 参数与工具

### 参数

| 参数 | 位置 | 默认 | 建议 |
|---|---|---|---|
| `POINTFLOW_SELECT_MOTION_FRACTION` | 训练启动脚本 | `0`(关闭) | **`0.05`** |
| `--select-motion-fraction` | `validate_pointflow_sonata.py` | `0.0` | `0.05` |
| `--top-motion-clusters` | `visualize_pointflow_motion.py` | 无(需显式给) | `8` |
| `--min-members` | `visualize_pointflow_motion.py` | `8` | `8` |

**为什么是 0.05**:410 点、间距最密(4.0mm)、信号 63mm(全量 7.6mm 的 8 倍)。2% 信号更强(91mm)但点更少(164),10% 点数翻倍而信号降到 48mm。

### 工具

| 文件 | 用途 | 需要 GPU |
|---|---|---|
| `tools/scan_pointflow_selection.py` | 10 集 × 抽样窗口的选点统计 | ❌ |
| `examples/launch_pointflow_motion_scan.sh` | 10 集 × 按比例铺开的窗口,出动图 | ✅ (PTv3/spconv) |

### 改动清单

| 文件 | 改动 |
|---|---|
| `pointflow_window.py` `[本仓库]` | 核心:按运动比例选点 + 重跑体素化 |
| `pointflow_source.py` `[本仓库]` | `select_motion_fraction` 参数与校验 |
| `singlerighthand_raw_dataset.py` `[本仓库]` | 参数透传 |
| `action_sft_dataset.py` (NVIDIA) | 参数透传(沿用已有的 pointflow 参数模式) |
| `action_policy_singlerighthand_edge.py` | 配置项,env 驱动 |
| `launch_sft_action_policy_singlerighthand_edge.sh` | 环境变量 |
| `validate_pointflow_sonata.py` | `--select-motion-fraction`(把选点接到可视化链路) |
| `visualize_pointflow_motion.py` | `--top-motion-clusters` / `--min-members`(逐窗口按运动选簇,替代固定 id) |
| `examples/launch_pointflow_motion_scan.sh` | 新增:10 集扫描 + 渲染 |
| `tools/scan_pointflow_selection.py` | 新增:选点统计 |
| `action_policy_singlerighthand_edge.toml` | **顺带**:`logging_iter` 1 → 10 |
