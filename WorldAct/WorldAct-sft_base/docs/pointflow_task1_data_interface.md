# 任务 1：训练窗口数据接口

2026-09-09。本任务提供可独立使用的 Dataset、时间契约与 ragged collate；不接入 Cosmos 训练循环，不新增 Sonata 编码封装、point token 或 attention。

## 时间配置

`PointFlowTiming.from_cosmos(dataset_config, tokenizer_config)` 从已解析的 Cosmos 配置取得 `fps`、`chunk_length`、`temporal_compression_factor`，并校验 `chunk_length+1` 与 `encode_exact_durations`。当前配方是 15 Hz、32 个未来状态、每 token 4 步。禁止另设独立 30 Hz 预测目标。

调用方应将现有 Cosmos 配方的 dataset/tokenizer 配置传入，而非长期维护另一份常量。配置位置：`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py:30`、`:77`；时间压缩配置：`cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py:135`。

## 使用方式

```python
from torch.utils.data import DataLoader
from cosmos_framework.data.pointflow_dataset import PointFlowWindowDataset, collate_pointflow_windows

# 从已解析的 Cosmos 实验配置取得，避免导入模型/加载权重来准备数据。
dataset_cfg = experiment["dataloader_train"]["dataloader"]["datasets"]["singlerighthand"]["dataset"]
tokenizer_cfg = experiment["model"]["config"]["tokenizer"]
dataset = PointFlowWindowDataset.from_cosmos(
    root="../datasets/sandwich_dense_fullseq_10_0298_20260908/outputs",
    episodes=train_episode_names,
    dataset_config=dataset_cfg,
    tokenizer_config=tokenizer_cfg,
    max_points=8192,
    voxel_size=0.02,
    seed=0,
)
loader = DataLoader(dataset, batch_size=2, shuffle=True,
                    collate_fn=collate_pointflow_windows, num_workers=0)
batch = next(iter(loader))
```

`episodes` 必须显式提供且无重复，由调用方固定 train/val episode 划分，Dataset 不自行混合或随机重切。Dataset 是 map-style，支持普通/分布式 sampler；窗口顺序由 sampler 控制。起点按 Cosmos 的 sample_stride（原始帧步长）枚举，只要求完整时间覆盖，不根据未来 valid 或运动量挑窗口。每个窗口的选点 seed 由全局 seed、episode 和 raw 起点稳定生成，跨 worker/访问顺序可复现；不随 epoch 改变点采样。

当前严格支持连续原始帧号、等间隔时间戳且源 FPS 为目标 FPS 整数倍的数据；不静默重采样不规则序列。窗口内目标按真实时间戳再次验证。太短的 episode 没有可采样窗口；整个列表没有窗口时报错。

## Batch 契约

| 容器 | 字段与含义 |
|---|---|
| inputs | 当前 point_ids、相机 anchor_xyz、tracker anchor_uv、RGB、normal；现有 reader 的体素 coord/feat/grid_coord |
| inputs | original_to_voxel、voxel_representatives：collate 后为 batch 全局索引 |
| inputs | point_offsets/voxel_offsets：累计末端偏移；point_batch/voxel_batch：各行所属样本 |
| inputs | coord_shift、intrinsics_normalized、image_size_wh 按样本堆叠；has_geometry 指示非空样本 |
| targets | displacement `[H,sum(N),3]`，单位米；valid `[H,sum(N)]`，逐步标签质量掩码 |
| metadata | episode、start_frame、原始帧号/时间戳、相对状态时间、时间片右端时间、seed、timing、empty_anchor、geometry_source |

原始 point_id 只在同一 episode 中有意义，跨样本须结合 point_batch/metadata 解释。原始 XYZ、UV 与体素预处理坐标分开，不能因 Sonata 中心化覆盖原始几何。尚未将 UV 转换为 Cosmos 拼图或 patch 坐标。

未来 valid 与位移位于独立 targets 容器，模型输入不能读它来筛选点。未来非有限/非正深度标签置为安全零并 mask 掉；失效后允许恢复。同一窗口全程按相同原始 ID 取目标。空 anchor 保留 `[H,0,3]` / `[H,0]`，可与非空样本共同 collate；后续几何编码器需用 has_geometry 跳过空样本，不能把重复 offset 不加处理地送入 sparse attention。

本任务没有计算 loss；全空批次的连图零损失将在联合损失任务实现。没有添加前景筛选。当前几何来源仍是离线 full-sequence Track4World，metadata 明确标记；不声称真实在线观测链路已完成。

现有 `prepare_window` 默认参数及严格空点报错保持兼容 smoke 脚本。训练 Dataset 显式传入 timing 和 `allow_empty=True`。读取错误、坏时间戳仍报错，不伪装成空点样本。

## 验证

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m unittest cosmos_framework.data.pointflow_window_test -v
```

覆盖时间配置校验、10 Hz/8 步非默认窗口、窗口尾端、NPY 截断、未来 NaN 不影响输入、有效性恢复、固定 seed、原始 ID、不同 N/体素数的 offset、空样本和全空 batch。

真实数据验证报告位于 `pointflow_outputs/task1/validation.json`。该检查包含第 596 帧的普通窗口及全无效尾段窗口，不运行 GPU 或 Cosmos 训练。

任务 1 完成后停止；下一任务是 Sonata 几何编码接口（enc3、簇中心与变长 batch），需要用户启动后才继续。
