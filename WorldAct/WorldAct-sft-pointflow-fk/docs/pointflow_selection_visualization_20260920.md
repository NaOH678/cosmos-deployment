# PointFlow 选点可视化(与训练逐比特一致)— 2026-09-20

## 用途

把训练数据路径**实际选中的点**(GT 位移排序 top-N + 体素成员护栏)画在头相机
视频上,输出 mp4 + gif + 中间帧 jpg,用于人工检查"模型到底在被监督学什么"。

与 `examples/launch_pointflow_motion_scan.sh` 的区别:那个渲染的是 PTv3 **簇**,
需要 GPU 跑 Sonata;本工具渲染的是**选点本身**——选择发生在
`prepare_window`(`cosmos_framework/data/pointflow_window.py`)里、在编码器之前,
因此全程 CPU 可跑,每个窗口几秒。

## 与训练一致的保证(重要)

`tools/visualize_pointflow_selection.py` **不重新实现选择逻辑**,而是实例化训练用的
`PointFlowSource`(`cosmos_framework/data/generator/action/pointflow_source.py`)、
加载训练 manifest、调用它的 `load()`——和 trainer 跑同一份代码:

- **种子派生一致**:每个窗口的采样种子是
  `sha256(f"{pointflow_seed}:{episode}:{起始帧}")`(`pointflow_source.py:90-91`),
  不是裸的 recipe seed。种子影响 8192 候选点的随机下采样,进而影响 top-N。
  教训:第一版直接用 `seed=42` 渲染,选中集与训练不同(同一窗口 max 位移
  220mm vs 106mm),已废弃。**不要绕过 PointFlowSource 自己调 prepare_window
  来做"训练一致性"可视化。**
- **旋钮一致**:`max_points=8192`、`voxel_size=0.02`、`select_top_n`、
  `min_voxel_members`、`supervise_cluster_n=0`、`allow_empty=True`,默认值即
  recipe `action_policy_singlerighthand_edge.py` 的值。
- **对齐强制校验**:`load()` 断言窗口与视频的 frame_id/时间戳一致
  (`pointflow_source.py:119-122`),错位直接报错,不会悄悄画错。
- **timing 一致**:`PointFlowTiming()` 默认 fps=15 / steps=32 / steps_per_token=4,
  与 recipe 的 dataset/tokenizer 配置相符(改训练 timing 时必须同步改这里)。

渲染只读返回样本的 `point_ids` 画 GT 轨迹,不做任何额外过滤。

**边界**:窗口**位置**由工具参数的 `--start-frames` 决定,训练时的窗口采样由
dataloader 决定。同一窗口内选点逐比特一致,但渲染的窗口不一定是训练采样到的
那些。eval 回调的固定窗口由 `pointflow_eval_cases.py` 的 `val_stage_fractions`
(0.2/0.5/0.8)选出,如需渲染那些窗口,按同一规则换算起始帧即可。

## 用法

```bash
python tools/visualize_pointflow_selection.py \
    --episode episode_0013_20260731_133649 \
    --start-frames 0 400 800 \
    --output pointflow_outputs/selection_vis_v2
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--episode` | (必填) | manifest 里的 episode 名(不是路径) |
| `--start-frames` | `0` | **源视频帧号**,可多个;越界窗口跳过不报错 |
| `--manifest` | `pointflow_outputs/task5/mixed_manifest.json` | 训练 manifest |
| `--select-top-n` | `300` | 与 `POINTFLOW_SELECT_TOP_N` 对应 |
| `--min-voxel-members` | `3` | 与 `POINTFLOW_MIN_VOXEL_MEMBERS` 对应 |
| `--max-points` / `--voxel-size` / `--seed` | `8192` / `0.02` / `42` | recipe 的 pointflow_* 参数 |
| `--trail-steps` | `5` | 拖尾帧数 |

批量跑 10 个 sandwich episode:

```bash
while read -r ep; do
  python tools/visualize_pointflow_selection.py --episode "$ep" \
      --start-frames 0 400 800 --output pointflow_outputs/selection_vis_v2
done < examples/pointflow_sandwich_10_episodes.txt
```

## 输出与读法

每个窗口三个文件:`<episode>_w<起始帧>_top<N>.mp4` / `.gif` / `_mid.jpg`。

画面为双联:左 = 原始 RGB;右 = 选中的点 + 5 帧拖尾。

- **颜色 = 该点的平均 GT 位移**(TURBO 色图:蓝=几乎不动,红=最快,
  归一化到该窗口 p95,标题栏给出上限 mm)
- **遮挡/丢失时该点消失、拖尾断开**,复现也不续接——与训练 loss 的
  valid mask 语义一致
- 标题栏:`selected N pts | visible M` —— M 随遮挡下降,是
  `valid_fraction` 的可视化对应物
- stdout 每个窗口打印一行:kept 点数、mean motion p50/max(mm)、valid fraction

## 已观察到的典型模式(sandwich 10 episode,top-300)

- 选点集中在夹爪/手和正在被操作的物体上(生菜、面包),背景只有零星低速点
  ——`min_voxel_members=3` 护栏有效,孤立跟踪离群点被排在最后
- 运动越剧烈的窗口 valid fraction 越低(w800:0.68;episode_0014 w0:0.59),
  即"选中的快速点未来常被遮挡",这是数据侧的固有属性,解释 eval 时记得
  ADE 只在 valid 点上计算
- top-300 凑不满 300 个高运动点时,排名尾部会落入低速背景点(画面上青色散点)
  ——属预期,训练数据本就如此
