# Action 归一化：两套仓库实现对照（2026-10-05）

## 结论与当前选择

当前 PointFlow-FK 仿真实验保留 **Bench2Dex 的 action-only 分位数方案**。
先用10条数据（8训练、2验证）检查拟合与联合生成质量，通过后扩至100条。
扩容时沿用公式，按新的训练划分重算统计；验证和部署复用训练统计。
这是当前设计选择，尚无两套归一化的生成质量对照，不能声称某套效果一定更好。

**按仓库区分算法，不用“10条版/100条版”代指算法。** 两套代码都可以对不同规模
的数据计算统计。此前提到的 action/state 联合范围属于 WorldAct-bench2dex，
不属于后来提供的 Bench2Dex 代码；公式相同也不代表统计文件或训练样本相同。

## 代码与统计口径

| 项目 | `/mnt/afs/Bench2Dex` | `/mnt/afs/WorldAct-cosmos3-edge-droid-sft-bench2dex` |
| --- | --- | --- |
| 统计入口 | `tools/finalize_sim_batch.py:55` | `tools/compute_bench2dex_action_stats.py:28` |
| 应用入口 | `utils/bench2dex_action.py:11` | `cosmos_framework/utils/bench2dex_normalization.py:17` |
| action样本 | 训练episode中全部action_valid帧 | 训练有效窗口覆盖的action帧，重叠窗口去重 |
| state样本 | 同一action_valid掩码对应的state | 实际作为条件的窗口起始state |
| 参数参考 | 仅action的q01/q99 | action与state的q01/q99联合范围 |
| scale下限 | 每关节0.05 rad | 每关节0.05 rad |
| 极端值 | 不额外调整尺度 | 扩大scale，使拟合样本绝对归一化值不超过5 |
| 硬截断 | 无 | 无 |
| 逆变换 | `x = y * scale + offset` | 相同 |

代码链接：[Bench2Dex统计](/mnt/afs/Bench2Dex/tools/finalize_sim_batch.py:55)、
[Bench2Dex应用](/mnt/afs/Bench2Dex/utils/bench2dex_action.py:11)、
[WorldAct统计](/mnt/afs/WorldAct-cosmos3-edge-droid-sft-bench2dex/tools/compute_bench2dex_action_stats.py:28)、
[WorldAct应用](/mnt/afs/WorldAct-cosmos3-edge-droid-sft-bench2dex/cosmos_framework/utils/bench2dex_normalization.py:17)。
这些绝对路径指向本集群相邻仓库，未随本仓库分发。

### Bench2Dex公式（逐关节）

```text
lo = action.q01
hi = action.q99
offset = (lo + hi) / 2
scale = max((hi - lo) / 2, 0.05)
y = (action - offset) / scale
```

mean/std同时写入JSON，但不用于这套变换。0.05下限避免放大小幅关节噪声；
不裁剪目标，因此归一化值允许超出[-1,1]，也可能超过±5。
当前仓库在 `cosmos_framework/data/generator/action/datasets/sim_pointfk_dataset.py:56`
读取同一文件并构建相同的 `ActionAffineNormalization`。

### WorldAct-bench2dex公式（逐关节）

```text
lo = min(action.q01, state.q01)
hi = max(action.q99, state.q99)
offset = (lo + hi) / 2
lower = min(action.min, state.min)
upper = max(action.max, state.max)
tail_scale = max(abs(lower - offset), abs(upper - offset)) / 5
scale = max((hi - lo) / 2, 0.05, tail_scale)
y = (action - offset) / scale
```

这里5是默认max_abs_target，约束拟合数据的幅度，不是clip，也不保证新数据不越界。
此方案同时覆盖条件state和目标action；少数极值可能扩大尺度，压缩正常动作的数值范围。
由于offset也可能变化，不能声称每个样本的绝对归一化值都比前者小。

## 文件路径与已核验来源

| 文件 | 用途 |
| --- | --- |
| `/data/shichaojian/sim_data/bench2dex_task21_pointfk10_rgb3_v2/normalization_train_only.json` | 当前仿真训练实际加载；数据包自带 |
| `/data/shichaojian/raw_data/bench2dex/task21/action_stats.json` | 另一条WorldAct-bench2dex训练路径的统计，90训练/10验证划分 |
| `/data/shichaojian/runs/bench2dex/cosmos_task21_bs16_normalized_20k_flash2/action_stats.json` | 上述WorldAct训练目录保存的统计副本 |

当前文件已用provenance中episode 0–7的5508条有效action复核：mean完全一致，
q01/q99最大差异小于3.4e-7。它不是完整100条的统计，也不是此次接入时重新生成的文件。
证据：`pointflow_outputs/sim_v2_preparation_20261004/action_normalization_origin_audit.json`。

两种JSON契约不同：前者读取action.q01/q99再计算参数；后者直接保存offset/scale。
迁移不能只改文件名，必须同时核对加载逻辑、关节顺序、单位和统计划分。

## PointFlow/FK尺度是另一件事

`/mnt/afs/Bench2Dex/tools/organize_sim_pointfk.py:91`在训练窗口的原始稠密点上统计
有效PointFlow位移；FK使用窗口中相对于首帧的位移。它输出全局std及32×3分帧分轴std。
当前 `tools/prepare_sim_pointfk_training.py:62` 在FK引导选出1024点后，对有效位移
重新统计全局标量std，训练加载该标量，并非action的分位数归一化。

| 尺度（当前10条数据包） | 原稠密点统计 | 选点后实际使用 |
| --- | ---: | ---: |
| PointFlow | 0.0410891 m | 0.0526976 m |
| FK | 0.0748352 m | 0.0748352 m |

两边全局std都按`sqrt(E[d²] - E[d]²)`计算；模型位移除以std，没有减去均值。
PointFlow变化来自选点分布变化。当前FK按valid掩码统计，原脚本直接计入全部FK位移；
本批数值基本一致。

## 扩至100条前

1. 统一episode级train/val划分，确认输入数据和关节顺序。
2. 修复当前选点准备脚本 `tools/prepare_sim_pointfk_training.py:23` 的`episode < 8`
   硬编码，改为读取manifest；Bench2Dex finalize的默认count=10及8/2描述也需更新。
3. 仅用新训练集计算action统计；检查归一化后超过±5的比例与对应动作，再判断是否需要换算法。
4. 按实际1024点选择重新统计PointFlow/FK scale，固定统计文件供训练、验证和部署共同使用。
5. 新统计配新实验目录；不要在当前运行中静默替换归一化文件。

训练入口和性能记录见[仿真实验文档](pointflow_sim_v2_data_20261004.md)，快捷启动见
[quickstart](pointflow_quickstart.md#当前仿真实验2026-10-05)。
