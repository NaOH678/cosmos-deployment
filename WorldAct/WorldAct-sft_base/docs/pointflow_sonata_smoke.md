# 真实 PointFlow 窗口与 Sonata 编码验证

入口：[examples/launch_pointflow_sonata_smoke.sh](../examples/launch_pointflow_sonata_smoke.sh:1)。这是几何编码器验证，不会启动 Cosmos 联合训练或优化器更新。

## 启动

在已分配的 H200 节点、仓库根目录运行（单 GPU 即可）：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/launch_pointflow_sonata_smoke.sh
```

默认取 `episode_0013_20260731_133649` 的 raw frame 0 起点，最多 8192 个当前点、2 cm 体素。按当前 Cosmos 单右手配方固定读取 15 Hz 的 33 个状态，输出 32 步目标位移。窗口不足或时间戳不匹配直接报错；后续接联合训练时再将这些时间参数绑定到解析后的 Cosmos 配方。

验证两个样本的变长体素 batching 和跨样本隔离：

```bash
CUDA_VISIBLE_DEVICES=0 START_FRAME=120 MAX_POINTS=16384 \
OUTPUT_DIR=pointflow_outputs/sonata_batch \
bash examples/launch_pointflow_sonata_smoke.sh \
  --episodes episode_0013_20260731_133649 episode_0014_20260731_133743
```

没有 GPU 时只验证真实数据准备与渲染：

```bash
DEVICE=cpu OUTPUT_DIR=pointflow_outputs/sonata_cpu \
bash examples/launch_pointflow_sonata_smoke.sh
```

支持环境变量 `DATA_ROOT`、`SONATA_CHECKPOINT`、`OUTPUT_DIR`、`START_FRAME`、`MAX_POINTS`、`VOXEL_SIZE`、`SEED`、`PYTHON_BIN`。默认数据和权重来自仓库的相邻目录，不会重新下载。默认输出 `pointflow_outputs/sonata_smoke`；重跑相同输出目录会覆盖同名产物，比较实验时使用不同 OUTPUT_DIR。脚本使用当前仓库 `.venv` 并清空 LD_LIBRARY_PATH。

## 数据与映射

- NPY 逐帧 seek/read，不 mmap 或整集加载；原始数据只读。
- 输入 ID 仅由当前 valid、有限 XYZ/UV、正深度、图像范围和法向量可计算性决定，最多 MAX_POINTS，固定 seed 可复现。
- 当前 XYZ 的近邻 PCA 估计法向量，朝向相机；未来点和 valid 不参与输入选择。原查询二维网格不能用于当前法向量差分。深度边界附近的法向量质量仍需后续评估。
- 从 COMPLETE 指定的 head 视频解码当前帧，直接 resize 到 tracker 推理分辨率，再用当前 UV 双线性采样 RGB。不会用首帧 query 行列替代当前 UV。
- 保留相机系 XYZ、UV 和原始 point_id，Sonata 特征使用平移后的 XYZ、RGB/255、法向量。每体素选固定代表，保留全部选中原始点的 original_to_voxel。
- 未来位移由相同 ID 对齐，未来非有限或无效点用 mask 排除并写安全零占位；失效后可恢复有效。
- GPU 模式将各样本体素点拼接并构造 offset；组合 Sonata 所有 pooling_inverse，得到每个原始选中点的最终簇映射，并检查所属 batch 没有串样本。
- 簇 XYZ/UV 对原始选中成员求平均，不对多层簇中心做无权重平均。显示的是 tracker/head 图像坐标，尚未转换为 Cosmos 拼图/patch MRoPE 坐标。

## 产物

每个 episode 子目录包含：

| 文件 | 内容 |
|---|---|
| window.npz | 原始点 ID、XYZ/UV/RGB/normal、Sonata 输入、体素 inverse、33 个帧号/时间戳、32 步位移与 valid |
| anchor_points.png | 当前 head RGB 上的输入点位置，最多显示 4096 点 |
| encoding_encN.npz | GPU 模式：点 ID → 簇、簇成员数、相机 XYZ/UV 中心、对应层特征 |
| point_clusters_encN.png | GPU 模式：同簇点使用同一颜色 |

顶层 `report.json` 记录点数/体素数/簇数、标签覆盖、XYZ 重投影对 UV 的误差、数据准备时间、GPU 型号、前向/反向耗时和峰值 allocated 显存。投影误差仅衡量字段自洽，不证明轨迹标注准确；overlay 也不是模型预测。

GPU 模式严格加载 checkpoint，以 eval 模式执行带梯度前向，用随机线性 probe 验证反向，检查输入层非零梯度与所有已有梯度有限性。该 probe 不是 PointFlow loss；未进行优化器更新，也未验证 train 模式统计量、FSDP 或 Cosmos 联合 attention。

## 本地验证记录

当前节点无可见 GPU，已完成 CPU 真实窗口和独立单元测试。首个 episode、start=0、seed=0：8192 个输入点、3961 个体素点；raw IDs 为 0,2,…,64；未来标签有效率 0.98410；投影中位误差 0.0778 px、P95 0.1659 px。已查看生成的 anchor_points.png。

单元测试覆盖时间戳对齐/短窗口拒绝、NPY seek/截断检测、未来 NaN 不影响输入选择、无效后恢复和原始 ID 对齐。运行：

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m unittest cosmos_framework.data.pointflow_window_test -v
```

H200 的编码/反向及点簇可视化由上面的 GPU 启动命令执行；当前未报告 GPU 通过。

权限说明：仓库原有 `outputs/` 属于 nobody:nogroup 且权限为 755，当前用户无法创建子目录。启动脚本默认改用仓库下独立的 `pointflow_outputs/sonata_smoke`，也可通过 OUTPUT_DIR 指定可写位置；不会修改原目录权限。


## 比较 enc2 / enc3 / enc4

```bash
CUDA_VISIBLE_DEVICES=0 START_FRAME=596 \
OUTPUT_DIR=pointflow_outputs/sonata_middle_stages \
bash examples/launch_pointflow_sonata_smoke.sh --stages 2 3 4
```

`--stages` 使用零起始 stage 编号，默认 4。一次完整前向后沿 pooling_parent 取各层特征，分别组合该层到输入的映射；enc2/3/4 分别为 128/256/512 维。反向 probe 包含所选层。耗时仍是完整 encoder 前向，不能当作截断在中间层后的耗时。

每层输出 `encoding_encN.npz` 和 `point_clusters_encN.png`，report 的 samples/stages 记录各层簇数和维度。

CPU 模式输出 `geometry_preview_encN.png/npz`，只按当前 small checkpoint 四次 stride=2 的空间分组规则预览，不计算神经特征，报告为 geometry_preview。第 596 帧、8192 输入点：enc2=1471 簇，enc3=454 簇，enc4=140 簇；这些不能冒充 GPU 中间特征验证结果。

## 运动动画（CPU，无需重新编码）

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m cosmos_framework.scripts.visualize_pointflow_motion \
  --episode ../datasets/sandwich_dense_fullseq_10_0298_20260908/outputs/episode_0013_20260731_133649 \
  --encoding-dir pointflow_outputs/sonata_middle_stages/episode_0013_20260731_133649 \
  --output pointflow_outputs/sonata_motion --stage 3 --focus-uv 450 380
```

输出三栏 MP4、循环 GIF、首/中/末帧预览图和参数 JSON：左侧原视频，中间固定簇颜色的动态点，右侧指定簇黄色高亮和短轨迹，其余点置灰。点沿源数据每一帧的 UV 移动，簇成员和颜色固定；无效或出界点隐藏，轨迹不会跨接无效间隔。

`--focus-uv U V` 根据 anchor 簇中心选最近且至少有 8 个成员的簇（没有则在全部簇中选）；单位是 tracker 图像像素，不是 Cosmos patch 坐标。这只是选择展示位置，不是手部语义检测。也可用 `--cluster-id ID` 精确指定，或 `--stage 2` 查看 enc2。`--max-display-points` 默认 2048；高亮簇显示全部当前有效成员。

动画仅展示离线 Track4World 标签，不是 Sonata/Cosmos 的运动预测，也不会修改输入筛选或训练标签。33 个状态的首末间隔为 2.13 秒；每个状态显示一帧，15 FPS 媒体总时长为 2.2 秒。GIF 用 60/70 ms 交替时长近似 15 FPS，MP4 按 15 FPS 输出。

## 单点整段轨迹（不依赖编码窗口）

若要看原来的三栏**簇动画**而非单点，请使用上一节脚本的整段/多簇模式：

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m cosmos_framework.scripts.visualize_pointflow_motion \
  --episode ../datasets/sandwich_dense_fullseq_10_0298_20260908/outputs/episode_0013_20260731_133649 \
  --encoding-dir pointflow_outputs/sonata_middle_stages/episode_0013_20260731_133649 \
  --output pointflow_outputs/sonata_motion \
  --stage 3 --cluster-ids 252 244 --full-sequence
```

`--full-sequence` 读取全部原始帧（当前序列 1192 帧、30 Hz），保持编码窗口 anchor（本例第 596 帧）确定的簇成员和原始 point IDs，一直向前/向后追踪，不逐帧重新聚类。它只覆盖该 anchor 编码时选中的点，不会加入其他新点。

`--cluster-ids` 支持一次指定多个簇，右栏使用不同高亮色并标注 ID；左栏仍为原视频，中栏仍为全部展示点的固定簇颜色。原来的 `--cluster-id 252` 单簇用法继续有效。无效点暂时隐藏，恢复后使用相同簇身份。输出文件带 `_full` 后缀，避免覆盖窗口版。完整 GIF 宽度为 960，MP4 保留原三栏分辨率；建议优先看 MP4 细节。

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m cosmos_framework.scripts.visualize_pointflow_track \
  --episode ../datasets/sandwich_dense_fullseq_10_0298_20260908/outputs/episode_0013_20260731_133649 \
  --output pointflow_outputs/full_track \
  --select-frame 596 --focus-uv 450 380
```

在第 596 原始帧选择距 tracker 像素 (450,380) 最近的有效点，再沿固定 ID 导出整段序列，而非从 596 开始。也可通过 `--point-id ID` 直接指定原始查询 ID，绕过像素选点。选点实际距离记录在 JSON，不自动判断手或物体类别。

输出 `track_ID.mp4`（逐帧标点）、`track_ID_xyz.png`（三维路径和 XYZ 时间曲线）、`track_ID.npz`（完整 UV/XYZ、帧号、时间戳与有效性）和 JSON。默认保留最近 30 个原始帧的轨迹尾迹，`--trail-steps 0` 显示全部历史有效线段。当前无效点不显示，缺失区间不连线；恢复后继续同一 ID。导出使用原始时间采样（当前数据 30 Hz），与 Cosmos 15 Hz 训练窗口无关。

单点读取采用逐帧定位对应槽位，避免加载整集 dense 点云。图像越界与三维无效分别保存为 visible/valid。轨迹来自离线标注；完整导出并不保证点全程有效或跟踪身份始终准确。
