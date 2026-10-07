# 任务 5：真实数据源与 Cosmos 样本对齐

任务范围：把任务 1 的 dense window 接入 SingleRightHandRawDataset 和 ActionTransformPipeline。复用任务 1–4；训练 batch、DataAndCondition、模型和联合 loss 在后续任务接线。

## 数据流

`Cosmos window index → episode + observation/action 原始帧索引 → video/action + PointFlowSource → ActionTransformPipeline → 单样本字典`

PointFlow 使用 video 的 observation_indices 读取轨迹，并逐项检查 frame IDs、绝对时间戳。默认 Cosmos 配置是 15 Hz、33 个状态、32 个未来点位置，30 Hz 原视频对应 0,2,…,64。时间参数来自数据集 fps/chunk_length 和 video_temporal_downsample；不另起 LingBot 时间窗口。

新增样本字段为 episode_name、raw_frame_ids、action_frame_ids、timestamps_sec、pointflow。pointflow 内保留 inputs/targets/metadata 三个容器；XYZ、UV、ID 和体素映射来自 anchor，未来 displacement/valid 只作为 targets。无标注 episode 的 pointflow 为 None；有标注但 anchor 为空仍返回空点窗口。两者不同。

## Mixed manifest

独立 sidecar schema_version=1，episodes 每行包含 name 和 pointflow_source。选中 split 内每个 episode 都必须列出；无标注显式写 null，拼错或遗漏会报错。划分 train/val 继续由原 action dataset 按 episode 完成。

```json
{
  "schema_version": 1,
  "episodes": [
    {"name": "example_labeled", "pointflow_source": {
      "path": "relative/path/to/dense_episode",
      "video_size_wh": [640, 960],
      "uv_to_video": [[1, 0, 0], [0, 1.07142857, 480.03571429]]
    }},
    {"name": "example_unlabeled", "pointflow_source": null}
  ]
}
```

上面的矩阵仅示范 tracker 640×448、head 640×480、wrist 640×480 的情况。实际生成器读取两个 MP4 的尺寸，按数据集的共享宽度和 round 后高度计算。path 相对于 manifest 所在目录解析。验收脚本生成当前数据的 mixed_manifest.json，并显式列出无 dense 标注的 episode。

## UV 如何跟随图像

uv_to_video 是 tracker 像素中心到数据集返回 video 像素中心的 2×3 仿射：先恢复 head 尺寸，再共享宽度缩放，最后加上上方 wrist 的高度。缩放采用像素中心约定：u' = s(u+0.5)-0.5。

ActionTransformPipeline 按真实取整后的 resize 宽高复合矩阵；右侧、下方 padding 不增加 UV 平移。原 anchor_uv 保持 tracker 坐标，只有 metadata 中的 uv_to_video 和 video_size_wh 更新。进入任务 4 的 patch 坐标还需后续按实际 VAE/patch 配置转换，这里不硬编码 patch 步长。

使用预计算 video 时，manifest 必须提供对应缓存画布的完整映射，不能直接套原始 MP4 的映射。加载时校验画布尺寸。当前拒绝未记录变换的随机图像 augmentation。

## 使用与验证

```bash
bash examples/launch_pointflow_source.sh
```

CPU 验证，无需 GPU 或 Sonata 权重。默认输出 pointflow_outputs/task5/mixed_manifest.json 和 report.json。DATA_ROOT、SINGLERIGHTHAND_RAW_ROOT、SINGLERIGHTHAND_CACHE_ROOT、OUTPUT_DIR、PYTHON_BIN、SEED 可覆盖。

验证读取真实 head/wrist MP4、真实 action NPZ、真实 dense 轨迹，并运行 ActionTransformPipeline；检查有标注和无标注样本、时间轴、目标形状和 UV 画布范围。脚本显式使用 OpenCV；当前环境的 TorchCodec 缺少 libnppicc.so.12。数据集默认仍使用 torchcodec，可配置 video_decoder="opencv"。

数据集工厂新增 pointflow_manifest、pointflow_max_points、pointflow_voxel_size、pointflow_seed、video_temporal_downsample 参数。启用 source 仅表示样本接口完成；变长 mixed batch 的整理属于任务 6，尚不能直接用原有 default_collate 启动联合训练。

单元测试使用 `--noconftest -o addopts=''`，因为本环境缺少仓库全局 conftest 所需的 xdist/custom-exit 插件。覆盖帧/时间错位、manifest 缺项与重复、像素中心变换和既有数据接口兼容性。
