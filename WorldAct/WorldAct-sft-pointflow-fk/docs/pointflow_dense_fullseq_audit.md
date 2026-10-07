# 新 dense full-sequence PointFlow 数据检查

检查日期：2026-09-08。只读检查原始数据，未修改或重新生成标签。

数据目录：`/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/sandwich_dense_fullseq_10_0298_20260908/outputs`。

## 1. 结论与检查范围

这批数据是固定首帧查询槽位上的 dense 轨迹，不是每帧独立重新采样、没有对应关系的点云。每帧有效点集确实不同；固定槽位和动态有效性同时存在。

检查了全部 10 个序列的 COMPLETE 元数据、NPY 文件头、完整帧号/时间戳/有效比例/内参/位姿数组；对大型 XYZ、UV、valid、confidence 数组抽查首帧、中帧、最小有效比例帧、末帧以及全无效尾段起点前一帧。未完整扫描所有 dense 数值。计数表来自有效比例乘查询数后取整，所有抽查帧均与实际 valid.sum() 一致。

结构化记录：[pointflow_dense_fullseq_audit.json](pointflow_dense_fullseq_audit.json:1)。

## 2. 字段与点 ID

| 文件 | 实际 shape | 含义 |
|---|---|---|
| position.npy | [T,448,640,3], float32 | 首帧查询点在各时刻的绝对 XYZ，不能直接当位移 |
| uv_px.npy | [T,448,640,2], float32 | 同一查询点在各时刻的像素位置 |
| valid.npy | [T,448,640], bool | 逐帧、逐查询点有效性 |
| confidence.npy | [T,448,640], float32 | 置信度；具体导出阈值未由 COMPLETE 说明 |
| intrinsics.npy | [T,3,3], float32 | 归一化图像坐标内参，不能直接当像素内参 |
| c2w.npy | [T,4,4], float32 | 模型估计位姿，实测非严格恒定 |
| frame_indices.npy / timestamps_sec.npy | [T] | 连续原始帧号，30 Hz 时间戳 |
| valid_ratio_per_frame.npy | [T], float32 | 全网格的有效点比例 |

所有序列 metadata 为 query_frame=0、pixel_queries=286720、mode=3d_ff、coordinate=camera_depthanythingv3、metric_scale=true、inference_calls=1、external_chunks=0。source_demo 指向 Track4World_portable；本次按用户要求用 lingbot third_party 的 full-sequence 实现解释机制，未证明两份代码逐字相同。

参考代码根目录 T：`/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-lingbot-va-pointflow/third_party/Track4World`。

- `T/demo.py:482`：forward_video3d_ff 对选定全序列调用 infer。
- `T/track4world/nets/model.py:1636`：固定 fmap/pointmap anchor 为第 0 帧，内部 sliding window 不重置查询帧。
- `T/track4world/nets/model.py:2313`：3D 相对 flow 加首帧 pointmap，输出绝对位置。

网格坐标 (y0,x0) 是首帧的查询身份，可以展平为 point_id=y0*640+x0。同一槽位对应模型追踪的同一首帧查询，不保证估计轨迹永远正确。当前像素位置必须读 uv_px[t,y0,x0]；不能把数组行列当作第 t 帧的像素。

## 3. 有效点数量

| episode | 帧数 | 首帧有效点 | 最少有效点 | 最多有效点 | 全无效尾段起始帧 | 尾段帧数 |
|---|---:|---:|---:|---:|---:|---:|
| episode_0013_20260731_133649 | 1192 | 258559 | 215443 | 258897 | 无 | 0 |
| episode_0014_20260731_133743 | 1211 | 246223 | 181088 | 246520 | 无 | 0 |
| episode_0015_20260730_170501 | 1367 | 252467 | 0 | 252840 | 1120 | 247 |
| episode_0015_20260731_133841 | 1751 | 258135 | 0 | 258404 | 1296 | 455 |
| episode_0016_20260730_170616 | 1424 | 262262 | 0 | 262541 | 1344 | 80 |
| episode_0017_20260731_134058 | 1459 | 252700 | 0 | 253033 | 1416 | 43 |
| episode_0018_20260730_170833 | 1419 | 266245 | 0 | 266465 | 992 | 427 |
| episode_0018_20260731_134203 | 1201 | 255684 | 217279 | 255715 | 无 | 0 |
| episode_0019_20260730_171242 | 1342 | 262031 | 78764 | 262418 | 无 | 0 |
| episode_0019_20260731_134304 | 1430 | 252931 | 0 | 252931 | 1360 | 70 |

帧号从 0 开始。6 个序列有连续全无效尾段，共 1322 帧。COMPLETE 表示导出完成，不保证所有帧有可用监督。例如 episode_0015_20260731_133841 从第 1296 帧开始无有效点，末帧 position 与 confidence 全为非有限值；不是仅仅因为可见点少。尚未定位根因，不能断言是自然遮挡或某个滑窗 bug。

也有少数 valid=True 的点深度非正，例如 episode_0013 的第 596 帧有 6 个。数据适配器需另外执行有限值和正深度校验；用于监督的未来校验不得决定输入点集。

## 4. 内参与相机系

第一段序列内参为 fx=0.812833、fy=1.161190、cx=cy=0.5，10 段各自的内参在时间维上均恒定。这是归一化图像内参。参考构造/投影代码：`T/track4world/nets/model.py:2358`、`:2409`。

像素内参应通过图像尺寸转换：

$$
K_{px}=\operatorname{diag}(640,448,1)K_{norm}
$$

对抽查帧中 valid 且有限、正深度的点，使用此内参投影 XYZ 并比较已有 UV，误差中位数约 0.08–0.11 px，P95 约 0.13–0.27 px。该结果验证当前字段之间的投影自洽，不能代替与真实场景对照的三维精度评估。

相机物理固定，但模型估计的 c2w 各帧略有变化，最大矩阵元素变化约 0.0038–0.0117（旋转和平移分量混合，不能当作位移米数）。不要把该估计变化当成真实相机运动再次施加到 position 上；首版按统一固定相机系处理，单独审计背景漂移。

## 5. Cosmos 接入调整

1. 新增 dense_full_sequence_npy 数据适配器，旧 global/* H5 与 chunk H5 作为兼容格式。新数据不要求旧 H48 sidecar 属性或 fixed_camera_repaired_v1 标记。
   时间窗口由 Cosmos 当前配方决定：15 Hz、33 帧视频、32 步动作，PointFlow 同样按 15 Hz 读取 raw offsets 0、2、…、64 的 33 个轨迹状态并预测 32 步位移。原始文件保持 30 Hz，不重新生成数据；按 VAE 时间压缩 4 推导 point 时间片 q=4，共 8 个未来片。旧 H48 chunk 无法提供该窗口完整监督，不能拼接不同身份的 chunk 或缩短 Cosmos 窗口来迁就它。参见 [设计文档第 4 节](pointflow_ptv3_cosmos_design.md:111)。
2. 在窗口起点 r 选择当前有效、有限、正深度的查询 ID，可进行只依赖当前 XYZ/UV 的空间采样或 voxel 聚合。保留 ID，未来各帧使用同一组 ID gather。不同窗口 N 不同，不能硬编码 1120。
3. 起点读取 position[r]、uv_px[r]；目标为 position[r+2*k]-position[r]（k=1…32，实际按 Cosmos 对齐后的 raw_frame_ids 读取）。每个未来时刻独立保存 valid/finite/depth 监督 mask。不得每帧 independently compact 后按新数组序号相减，也不得只挑未来全程有效的点作为输入。
4. 不强制 once-lost-always-invalid：原始 valid 按帧使用。有效数回升并不能证明新增轨迹；固定 slot 允许查询估计再次变为有效。新出现但首帧没有 query 对应的表面仍没有独立新 ID。
5. 若一次输入多个观测帧，PTv3 batch/offset 可支持每帧不同点数；相应 ID、UV 和时间必须保留。当前首版只编码当前窗口起点，未来有效数变化用 loss mask 表达。
6. 用 voxel/输入点预算与 pooled token 预算分别控制成本。建议实测 8K/16K/32K 起点输入的覆盖、速度和误差，也可实测全有效点；这些是性能消融，不改变 dense 原始数据。原始 286720 点不是 286720 个 Cosmos tokens。
7. 用户接受 H200 上约 50M 模型，默认改为 PTv3 encoder-only（约 38.7M）端到端训练，不以旧浅 encoder 为首选。PTv3 权重容量与 dense 点数/稀疏算子成本分别实测；未来噪声状态不输入 PTv3 重建邻接。
8. 全无效尾段无 PointFlow 监督，不应让它贡献虚假的零轨迹目标；可保留视频/action 训练，point loss 连图置零，并记录失效比例。PointFlow 专项评估单独报告标签覆盖，不能只看剩余有效点误差。

全批 NPY 约 92.10 GiB。当前存储测试 mmap 返回 ENODEV，检查采用 seek/read。训练读取宜先做分帧缓存或独立的训练 sidecar，避免随机窗口反复读取所有 dense 数据；本次没有执行数据转换。

本次未安装 PTv3 依赖、未运行 tracker、未启动训练，也未修改原始数据。
