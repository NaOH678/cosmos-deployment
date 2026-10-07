# 任务 3：Point token 与逐点解码器

状态：独立内容编码/解码已实现；不包含 Cosmos attention、序列 packing、RoPE、RF 加噪/采样或训练 loss。

接口位于 `cosmos_framework/model/generator/pointflow_codec.py`。调用方式：

```python
codec = PointFlowCodec(timing=timing, geometry_dim=256, hidden_dim=2048)
tokens = codec.encode(geometry, noisy_displacement, sigma)
# 下一任务将 noisy_tokens 接入 Cosmos，取得对应的 point_hidden。
velocity = codec.decode(geometry, inputs, point_hidden, noisy_displacement, sigma)
```

默认对应 enc3（256 维）、enc0 局部特征（32 维）、Cosmos hidden 2048、32 步预测、每片 4 步。构造支持其他维度及时间配置。模型和输入需位于同一设备；当前入口使用 FP32，混合精度尚未验证。

## 编码

几何内容为线性投影后的 Sonata 特征，加相机系簇 XYZ 的 MLP 编码；XYZ 除以固定 `xyz_scale`（默认 1 米，保存为 buffer），不逐帧重新中心化。

带噪位移保持 `[H,N,3]`，单位是外部调用方定义的模型空间（物理位移乘 flow_scale）。按时间顺序转换为 `[H/q,N,3q]`，先逐点通过 motion MLP，再根据 original_to_cluster 按原始成员数求平均。与几何拼接，经过 LN/point2llm，加模态 embedding 和 sigma embedding，产生 `[H/q,K,D]` noisy_tokens。

sigma 为 `[batch_size]`，含空样本占位，范围 [0,1]。本模块用固定正弦/余弦频率加 MLP 表达去噪进度，不表示真实时间。未来 valid/moving 不出现在接口中，不参与 token 聚合。输入 noisy state 必须有限且覆盖所有选中的原始点。

可选 anchor_tokens 为 `[K,D]`，仅由几何产生，不依赖未来噪声或 sigma。返回 cluster_batch/cluster_offsets，尚未把时间片与样本维展平为 Cosmos 序列；不能直接跨样本拼入 attention。

## 解码

输入 point_hidden 为 `[H/q,K,D]`，由调用方从未来 Cosmos 输出中取回。对每个原始点 gather 簇 hidden，与 enc0 局部特征、相对簇中心 XYZ/UV、该点有序带噪位移及 sigma embedding 拼接，再用逐点 MLP 输出 q×3 个速度，还原 `[H,N,3]`。UV 差值除以对应样本 tracker 图像宽高，XYZ 差值除以固定 xyz_scale。

因此同簇内点可预测不同运动，不是将一个簇速度广播给全部成员。此输出是待训练的 RF velocity，不是已经去噪完成的轨迹；本任务没有实现积分或损失。全空 N/K 支持零行张量与零梯度路径。

## 验证与启动

CPU 自动化测试已通过：时间顺序与 reshape 往返、变长/空样本、样本隔离、anchor 对噪声/sigma 不变、同簇逐点差异、几何/局部特征/噪声和全部 codec 参数梯度、全空 batch 反向及非法 sigma/shape。

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m unittest cosmos_framework.model.generator.pointflow_codec_test -v
```

H200 上验证真实数据→Sonata→codec 的完整梯度链路：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/launch_pointflow_codec.sh
```

报告为 `pointflow_outputs/task3/report.json`。脚本使用 512 点、空样本和 256 点混合 batch，将编码 token 直接作为 decoder hidden 的占位，检查形状、有限性和梯度。它不模拟 Cosmos attention，不构成预测质量评估，也没有优化器更新。新版 GPU 验证需用户运行；不能用 CPU 单元测试代替。

任务 3 完成后停止。下一任务才是 Cosmos 序列与位置编码接入。
