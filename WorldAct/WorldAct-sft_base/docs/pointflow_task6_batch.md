# 任务 6：变长训练 batch 与数据容器

任务 6 已接通：任务 5 单样本 → Cosmos 两条 collator 路径 → 混合 batch → GenerationDataClean.pointflow → PackedSequence.pointflow_data。

这里的 DataAndCondition 在代码中实际是 GenerationDataClean / GenerationDataNoised。任务 6 定义数据接口，任务 7 接入可训练模块；任务 8 负责正式噪声、loss、采样策略。

## 混合 batch

pointflow 在 collator 中按样本保留 list[dict | None]，不执行 default_collate、不删除 None。兼容 legacy JointDataLoader 与 VFMListCollator，包括不同内层 batch 先后遇到无标注/有标注样本的情况。

build_pointflow_batch 在模型 get_data_and_condition 中调用，得到 PointFlowBatch：

- inputs：拼接的原始点、体素、映射、UV 仿射、画布信息。
- displacement：[H, sum(N), 3]，单位米；valid：[H, sum(N)]，仅用于监督。
- labeled：[B]，有没有 dense 标注；has_point：[B]，anchor 是否非空。
- metadata：保留所有 B 个槽位，无标注为 None；timing：批内统一 Cosmos 时间配置。

例如 N=[256,0,0,256]，第二个样本无标注，第三个样本有标注但 anchor 为空：

| 字段 | 内容 |
|---|---|
| labeled | [true,false,true,true] |
| has_point | [true,false,false,true] |
| point_offsets | [256,256,256,512] |
| point_spans | [[0,256],[256,256],[256,256],[256,512]] |
| point_batch | 前 256 个为 0，后 256 个为 3 |
| displacement | [32,512,3] |

point_spans/voxel_spans 是左闭右开的区间。original_to_voxel 与 voxel_representatives 分别加入全局体素、全局点偏移，并检查样本内索引边界。未来 valid 全 false 也不减少点数、不改变 token 预算。

全无标注返回 None；有标注但全空返回合法 [H,0,3] 容器。设备移动保持整数 ID、布尔 mask 和 FP32 几何，不套模型的 BF16 dtype。供模型使用的 UV 仿射已进入 inputs；审计 metadata 留在 CPU。

## clean、noised 与 sequence

GenerationDataClean.pointflow 保存 PointFlowBatch。GenerationDataNoised.pointflow 保存 PointFlowNoised，其字段 xt、epsilon、velocity_target 为 [H,sum(N),3]，sigma 为原始 B 个样本的 [B]。支持形状/有限值校验与设备搬运；任务 6 不确定 RF 正负号、缩放或 sigma 抽样策略。

ActionTransformPipeline 设置 SequencePlan.has_point。pack_input_sequence 检查它与 PointFlowBatch 一致，把原始数据保存在 PackedSequence.pointflow_data；to_cuda 同步搬运。沿用任务 4 的 PackedSequence.point 专门保存后续编码出的 token payload，二者用途明确。

任务 6 不把 XYZ 或监督标签直接当成 2048 维 token。真正的 Sonata → Codec → attach_point_tokens 在任务 7 接入主模型；任务 4 已完成该局部链路。当前不应据此启动完整 PointFlow 联合训练。

## Token 预算

数据加载器不运行 Sonata。利用簇数 K ≤ 输入体素数 V，预留 V×(1+H/q) 个 point token，计入 legacy/dataflow 两条预算路径。空点与无标注预留 0。这个保守上界可能降低装箱利用率；它不是精确簇数，也不是 padding 后的真实序列长度。任务 7 模型内取得实际 K 后，通过任务 4 的序列逻辑生成精确长度。

当前只允许 two-way packing。批内不同 timing 或 has_point 错位直接报错。

## 验证

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/launch_pointflow_batch.sh
```

默认读取任务 5 生成的 pointflow_outputs/task5/mixed_manifest.json，可用 POINTFLOW_MANIFEST 覆盖；输出 pointflow_outputs/task6/report.json。没有 manifest 时先运行 launch_pointflow_source.sh。

本机真实 dense 窗口验证 CPU PASS：point_offsets=[256,256,256,512]，voxel_offsets=[245,245,245,491]，displacement=[32,512,3]。视频/action latent 和 noised state 是合成输入；实际调用 Cosmos pack_input_sequence。当前节点无可用 GPU，因此 CUDA 搬运需在 H200 上复跑同一脚本。脚本有 GPU 时自动执行 to_cuda 和 dtype/形状验证。

测试覆盖 mixed collator、空点/无标注区分、跨内层 batch 的占位、offset/mapping、未来监督 mask 隔离、真实 packer 的数据保留与 has_point 错位拒绝。

本轮相关回归测试共 23 项通过；新增/改动的数据接口通过 Ruff 检查，git diff --check 通过。
