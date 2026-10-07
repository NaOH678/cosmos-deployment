# DA3 深度使用真实 D435 RGB 内参：B 方案与尺度恢复诊断

更新日期：2026-10-05。

当前选择：继续采用 **B＋旧的末段尺度恢复**。逐段尺度恢复仅保留为诊断对照，不切换为当前使用方案；后续深度误差诊断也以 B＋末段尺度为基准。

## 1. 目标和当前状态

目标：**仍然使用 DA3 从 RGB 估计深度，把 Metric 米制尺度换算和最终 XYZ 反投影使用的估计内参替换为真实 D435 RGB 内参**，观察 PointFlow 手部点云是否更贴近 FK。

本次实现是独立诊断，不训练 DA3，不使用 D435 实测深度，也不根据 FK 拟合尺度、平移或旋转。

当前已完成：

- 使用原始 Track4World AnyView 权重重新计算 Metric 校准比例。
- 固定原始 PointFlow 的手部选点、有效性和去重结果，生成 B 校准点云。
- 生成原始 / B 两行图和整集连续视频。
- 独立增加逐段尺度恢复模块，生成原始 / B＋旧尺度 / B＋逐段修正三行图、连续视频和整集指标。

**尚未完成或不属于本次实现的内容：**

- 尚未将 B 接入生产数据流水线，也未替换训练数据或窗口缓存。
- 未完整重新运行时序追踪、有效性筛选和训练数据导出。
- 未将真实内参作为 AnyView 网络内部的相机条件。
- 未修复已经在不同归一化单位之间计算的跨段学习式三维运动。

因此，下文的 B 可视化应称为“固定原始选点的缓存几何校准重建”，不能声称是生产流水线完整重跑后的精确输出。

## 2. A、B 两种替换方式

| 方案 | Metric 尺度换算 | 最终 XYZ 反投影 | 对 Z 深度的影响 |
|---|---|---|---|
| 原始 | DA3 估计焦距 | Track4World 导出的估计内参 | 原始米制深度 |
| A | 保持原样 | 使用真实 RGB 内参 | Z 不变，主要改变 X、Y |
| **B，本次选择** | **使用真实 RGB 焦距** | **使用真实 RGB 内参** | **重新换算米制尺度，Z 也会改变** |

不能把 B 描述为“原来的深度数值完全不变，只换一个相机矩阵”。**保留的是 DA3 作为深度来源；Metric 换算后的深度值会变化。**

## 3. 原始 DA3 / Track4World 深度链路

本项目使用 Nested DA3：

1. AnyView 分支从多帧 RGB 预测深度几何、相机内参和外参。
2. Metric 分支预测用于米制尺度参考的深度。
3. 根据焦距，将 Metric 分支的深度换算到对应相机的尺度。
4. 在非天空、满足深度和置信度条件的像素上，把 AnyView 深度对齐到 Metric 参考深度。
5. Track4World 进行几何归一化、运动估计以及导出前的尺度恢复和反投影。

当前 Metric 焦距换算为：

\[
D_{metric}=D_{metric,raw}\cdot\frac{(f_x+f_y)/2}{300}.
\]

其中 `fx、fy` 必须对应 **DA3 当前输入分辨率**。`300` 是当前 DA3 实现中的换算常数，不是这台 D435 的真实焦距。

对齐的基本形式是：

\[
a=\arg\min_a\sum_{i\in M}(aD_{any,i}-D_{metric,i})^2,
\qquad D_{aligned}=aD_{any}.
\]

`M` 是实现选出的有效对齐像素集合；它不是 FK 关节或手部拟合区域。替换内参后应重新执行此步骤，不能把任意一个焦距比直接当作所有数据的最终深度比例。

代码依据：

- `/data/shichaojian/Track4World_portable/Track4World/track4world/nets/external/depth_anything_3/utils/alignment.py:118`：Metric 焦距换算。
- 同目录 `model/da3.py:399` 附近：`_apply_metric_scaling`、`_apply_depth_alignment`、天空处理。
- `/data/shichaojian/Track4World_portable/Track4World/track4world/nets/model.py:792`：加载 Track4World 权重并映射到 Nested AnyView 分支。

**权重必须一致：**原始 PointFlow 使用 Track4World checkpoint 覆盖 AnyView 分支，Metric 分支保留 Nested DA3 的预训练权重。只加载通用 DA3 checkpoint 会改变几何预测，不能作为仅替换内参的对照。

## 4. 真实内参来源与分辨率变换

示例 episode：`episode_0013_20260731_133649`。

元数据文件：

```text
/data/shichaojian/raw_data/singlerighthand_sandwich_100/episode_0013_20260731_133649/auxiliary_camera/metadata.json
```

读取字段：

```text
capture_metadata.cameras.head.streams.color.intrinsics
```

该 episode 的 RGB 标定值：

| 参数 | 数值 |
|---|---:|
| width × height | 640 × 480 |
| fx | 605.5706176757812 |
| fy | 604.4129638671875 |
| cx / ppx | 324.4994812011719 |
| cy / ppy | 238.25637817382812 |
| 畸变系数 | 全部为 0 |

**使用 color 内参，不是 depth 流内参。** DA3 的输入是 RGB，输出深度与 RGB 像素对应。本方案也不需要读取 `depth.lmdb` 或应用 depth→color 外参。

本例处理尺寸：

```text
原始 RGB 640×480
  → Track4World 画布 640×448
  → DA3 patch-14 输入 630×448
```

按像素中心约定，将图像从 `W×H` 缩放到 `W'×H'`：

\[
f'_x=f_x\frac{W'}W,\quad f'_y=f_y\frac{H'}H,
\]

\[
c'_x=(c_x+0.5)\frac{W'}W-0.5,\quad
c'_y=(c_y+0.5)\frac{H'}H-0.5.
\]

Metric 换算使用 `630×448` 对应的真实焦距。最终缓存点云的 `obs_uv` 位于 `640×448` 画布；当前绘图实现先将它映射回 `640×480` 原始 RGB 像素，再用原始 RGB K 反投影：

\[
u_{rgb}=(u_{canvas}+0.5)\frac{640}{640}-0.5,
\quad v_{rgb}=(v_{canvas}+0.5)\frac{480}{448}-0.5,
\]

\[
X=\frac{u_{rgb}-c_x}{f_x}Z,\quad
Y=\frac{v_{rgb}-c_y}{f_y}Z,\quad Z=Z.
\]

真实内参应按 episode 元数据读取，不能把本例数值无条件硬编码到其他相机、分辨率或裁剪方式。当前脚本针对零畸变数据；非零畸变需要先建立一致的去畸变/投影流程。

## 5. 当前 B 方案具体怎么实现

### 5.1 固定原始点云支持集

绘图基于用户指定的 `_mano/tools/render_fk_vs_pointflow.py` 扩展，在 base 保存独立版本。读取：

```text
/data/shichaojian/pf_out/9.24/sandwich/efep_seg_v2/<episode>/
```

每帧按 `frame_offsets` 提取观测，选点条件保持：

```python
(obs_label == 2) & obs_valid & obs_unique
```

保留原始 `obs_uv` 和 DA3 缓存 Z 的局部形状，只改变尺度和反投影。FK 沿用已验证的 base→camera 变换，不随 B 调整。

导出元数据含 `world_depthanythingv3` 字样，但不能仅凭模式名判断 `obs_pos` 的坐标系。当前四帧缓存点用导出内参投影后，与 `obs_uv` 最大偏差小于 0.335 像素，支持按最终相机点图处理；不要盲目再次应用 `c2w`。

### 5.2 重算校准比例

脚本：[`tools/pointflow_original_b_scale.py`](../tools/pointflow_original_b_scale.py)。

加载：

```text
/data/shichaojian/checkpoints/DA3NESTED-GIANT-LARGE-1.1
/data/shichaojian/checkpoints/track4world_da3.pth
```

复用相同的 AnyView / Metric 原始预测，对照估计焦距和真实焦距的 Metric 换算。只在 Metric 换算时临时替换 K；为隔离校准效应，中间归一化几何仍使用原预测射线，最终导出反投影才使用真实 RGB K。

这**不是**把真实 K 从网络入口送入 AnyView 重新预测相机几何。

为复现旧封装的尺度行为，本例先重算最后一个分段 `[1152,1192)`，得到：

```text
原始 Metric 对齐比例：1.1684721708
B 的 Metric 对齐比例：1.2845703363
B / 原始：            1.0993589479
```

固定缓存支持集的 B 使用：

```text
Z_B = Z_cached × 1.0993589479
XYZ_B = 使用真实 RGB K 对 (obs_uv, Z_B) 反投影
```

**该系数只属于本例和这次重建，不能推广到其他 episode。**

重算归一化焦距为 `0.811553816`，缓存为 `0.813095868`，相差约 **0.19%**。因此当前结果是带有复现不确定性的近似重建，而不是历史导出的逐位复现。

## 6. 独立问题：逐段归一化却只恢复末段尺度

这不是 DA3 的要求，而是当前 Track4World 封装中的状态覆盖问题：

```text
model.py:962 附近：每段 scale 写入 self._metric_scale，覆盖前段值。
model.py:2586 附近：导出整段时统一乘 self._metric_scale。
```

记第 k 段相机点平均范数为 `s_k`，实际除数是 `s_k + 1e-6`。

旧逻辑：

\[
P_{cached,k}=\frac{P_k}{s_k+\epsilon}s_{last}.
\]

正确逆操作：

\[
\frac{P_k}{s_k+\epsilon}(s_k+\epsilon)=P_k.
\]

为了将这个问题和真实内参替换分开，对照分为三种：

| 对照 | 内参校准 | 尺度恢复 |
|---|---|---|
| 原始 | 原始估计内参 | 旧逻辑 |
| B＋旧尺度 | 真实 RGB K | 保留末段恢复，以复现旧行为 |
| B＋逐段修正 | 真实 RGB K | 每段恢复自己的尺度 |

独立实现：[`tools/track4world_scale_fix.py`](../tools/track4world_scale_fix.py)。

- `ChunkScaleLedger` 记录每段实际除数和长度。
- 支持恢复相机点、世界点以及相机平移；相机旋转不缩放。
- `correct_cached_depth` 用于固定旧缓存的诊断重建。
- 原始 Track4World `model.py` 未覆盖，生产入口没有自动切换。

若第 k 段 B/原始 Metric 对齐比例为 `r_k`，缓存诊断重建采用：

\[
Z_{corrected,k}=Z_{cached,k}\cdot\frac{s_k+\epsilon}{s_{last}}\cdot r_k.
\]

各段尺度来自原始时间分段的重算，不根据 FK 拟合。由于历史运行状态不完全可复现，这仍不是严格的历史缓存精确逆算。

**跨段运动需要另行处理。** 不同单位的归一化坐标进入运动网络后，事后逐段乘回尺度不能保证学习式 scene flow 正确。当前模块和图表只处理几何单位，不宣称完成三维运动修复。

## 7. 当前实验结果

### 7.1 四帧对照

指标：每个 FK 关节到可见手部点云的最近三维距离，再对该帧 21 个关节取中位数。

| 原始帧号 | 原始 | B＋旧尺度 | B＋逐段修正 |
|---|---:|---:|---:|
| 0 | 52.1 mm | 17.0 mm | 33.9 mm |
| 300 | 70.6 mm | 18.9 mm | 22.7 mm |
| 650 | 88.3 mm | 39.6 mm | 82.0 mm |
| 1000 | 36.1 mm | 18.6 mm | 26.5 mm |

[两行图](../outputs/fk_vs_pointflow_B/episode_0013_20260731_133649_fk_vs_pointflow_B.png) · [三行图](../outputs/fk_vs_pointflow_scale_fix/episode_0013_20260731_133649_fk_vs_pointflow_three_rows.png)

### 7.2 整集 1192 帧对照

对每帧取上述中位距离，再对全体帧汇总；以下 P90 也是“帧中位距离”的 P90。

| 情况 | 全帧平均 | 全帧中位数 | P90 |
|---|---:|---:|---:|
| 原始 | 74.5 mm | 73.2 mm | 126.4 mm |
| B＋旧尺度 | 36.6 mm | 29.2 mm | 70.6 mm |
| B＋逐段修正 | 43.8 mm | 34.2 mm | 88.0 mm |

逐段修正相对 B＋旧尺度：

- 平均帧中位距离增加 **7.25 mm**。
- 以变化超过 1 mm 为明显变化：438 帧改善、664 帧变差、90 帧基本相同，即 **36.7% / 55.7% / 7.6%**。
- 并非每个阶段都变差；约 12.8～21.3 秒、34.1～38.4 秒的分段平均距离有所改善。

本例中 B＋旧尺度在这个贴合指标上最好，**不能因此认定旧恢复逻辑正确**。错误缩放可能抵消其他深度或外参误差；这是可能解释，不是已经分离验证的因果结论。

指标还受以下因素影响：FK 关节位于手内部、点云仅包含可见表面、遮挡、分割/追踪、FK 外参及 DA3 局部深度误差。不同变体的最近表面点也可能不同。因此它不是对应关节的真实误差，也不是模型预测 ADE，不能用来证明真实内参或尺度修复后的绝对深度一定正确。

产物：

- [三路连续 MP4：1192 帧、30 fps、约 39.7 秒](../outputs/fk_vs_pointflow_scale_fix/episode_0013_20260731_133649_three_continuous.mp4)
- [逐帧误差曲线](../outputs/fk_vs_pointflow_scale_fix/episode_0013_20260731_133649_three_continuous.metrics.png)
- [整集与分段指标 JSON](../outputs/fk_vs_pointflow_scale_fix/episode_0013_20260731_133649_three_continuous.summary.json)
- [逐帧 CSV](../outputs/fk_vs_pointflow_scale_fix/episode_0013_20260731_133649_three_continuous.csv)
- [全关节距离及 XYZ 残差 NPZ](../outputs/fk_vs_pointflow_scale_fix/episode_0013_20260731_133649_three_continuous.npz)

### 7.3 不应继续引用的早期实验

`outputs/da3_d435_intrinsics/` 下早期实验直接把四张相隔很远的 RGB 输入通用 DA3，再使用 SAM2 掩码，没有复现原始 Track4World 权重、时序上下文及选点流程。估计内参基线的手型已经改变。

早期“尺度增加约 8.4%、距离改善约 2.1 cm、XY 改善约 35%”仅描述那个独立实验，**不能作为原始 PointFlow 替换内参后的测量结果**。本文件采用后续原始缓存支持集的结果。

## 8. 复现命令

从 base 仓库根目录执行。所有命令输出到项目 `outputs/`，不覆盖原始数据。

### 8.1 利用现有尺度报告绘制两行图

```bash
LD_LIBRARY_PATH='' MPLCONFIGDIR=/tmp/da3-matplotlib \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python \
  tools/render_fk_vs_pointflow.py \
  --frames 0,300,650,1000 \
  --b-scale-report outputs/fk_vs_pointflow_B/scale_report.json \
  --out outputs/fk_vs_pointflow_B --seed 0
```

不要加 `--d435`：原脚本的这个选项使用 D435 实测深度，不是 B。

### 8.2 绘制三行图

```bash
LD_LIBRARY_PATH='' MPLCONFIGDIR=/tmp/da3-matplotlib \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python \
  tools/render_fk_vs_pointflow_three_rows.py \
  --frames 0,300,650,1000 \
  --b-scale-report outputs/fk_vs_pointflow_B/scale_report.json \
  --chunk-scale-report outputs/fk_vs_pointflow_scale_fix/chunks_all.json \
  --out outputs/fk_vs_pointflow_scale_fix --seed 0
```

### 8.3 生成整集三路视频和指标

```bash
LD_LIBRARY_PATH='' MPLCONFIGDIR=/tmp/da3-matplotlib \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python \
  tools/render_fk_pointflow_three_video.py \
  --scale-report outputs/fk_vs_pointflow_B/scale_report.json \
  --chunk-report outputs/fk_vs_pointflow_scale_fix/chunks_all.json \
  --output outputs/fk_vs_pointflow_scale_fix/episode_0013_20260731_133649_three_continuous.mp4
```

视频从完整原始缓存逐帧读取，不是从四帧 `comparison_points.npz` 插值生成。视频为显示而抽样点云，指标计算使用全部有效手部点。

### 8.4 重新计算尺度报告，需要 GPU

以下脚本当前针对上述 episode；其他 episode 需要同步修改脚本的 `EP` 和路径配置，不能直接复用现有报告。

仅末段 B 校准比例：

```bash
LD_LIBRARY_PATH='' MPLCONFIGDIR=/tmp/da3-matplotlib \
PYTHONPATH=/data/shichaojian/Track4World_portable/Track4World/track4world/nets/external \
CUDA_VISIBLE_DEVICES=7 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python \
  tools/pointflow_original_b_scale.py outputs/fk_vs_pointflow_B
```

整集分段尺度：

```bash
LD_LIBRARY_PATH='' MPLCONFIGDIR=/tmp/da3-matplotlib \
PYTHONPATH=/data/shichaojian/Track4World_portable/Track4World/track4world/nets/external \
CUDA_VISIBLE_DEVICES=7 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python \
  tools/track4world_chunk_scale_diagnostic.py \
  outputs/fk_vs_pointflow_scale_fix --all-chunks --low-memory
```

GPU 编号按实际空闲情况修改。`--low-memory` 保留完整 128 帧注意力上下文，仅对逐 token MLP 分块，并将块权重、暂存特征等在不使用时卸载至 CPU；它不保证与历史 GPU/软件环境逐位一致。

分段脚本会复用输出目录已有的 `chunk_XXXX.json`。更换权重、算法、episode 或配置后，应使用新目录，避免混用旧报告。

## 9. 代码与验证范围

| 文件 | 作用 |
|---|---|
| `tools/pointflow_original_b_scale.py` | 原始权重下重算末段 B 校准比例 |
| `tools/render_fk_vs_pointflow.py` | 用户指定绘图脚本的 base 独立扩展，支持 B 两行图 |
| `tools/track4world_scale_fix.py` | 独立分段尺度记录、恢复与缓存修正函数 |
| `tools/track4world_chunk_scale_diagnostic.py` | 原始时间分段下计算原始/B 范数及 Metric 比例 |
| `tools/render_fk_vs_pointflow_three_rows.py` | 三行静态对照 |
| `tools/render_fk_pointflow_three_video.py` | 三路整集连续视频 |
| `tools/evaluate_fk_scale_video.py` | 全帧、全关节距离及分段评估 |
| `tools/track4world_memory.py` | 诊断用中间特征 CPU 暂存 |

已验证：分段归一化的逆操作、相机/世界点与相机平移恢复；实际重算几何往返误差低于 `2.4e-7 m`。整集原始/B 两路逐帧指标与之前两路视频一致，1192 帧指标全部有限；MP4 已核对帧数、帧率、时长。相关新增脚本通过 Ruff 检查。

后续若要用于正式训练数据，应在独立流水线中完成：真实 K 的一致接入、分段尺度记录与正确恢复、跨段运动单位处理、完整追踪和筛选重跑、独立输出导出及缓存重建，并在更多 episode 和独立几何参照上验证。**本次更好的 FK 表面贴合结果本身，不等于已经完成这些生产改动。**
