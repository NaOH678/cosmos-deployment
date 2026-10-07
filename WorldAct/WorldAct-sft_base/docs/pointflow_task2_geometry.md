# 任务 2：Sonata 几何编码接口

2026-09-09。提供独立 `nn.Module`，消费任务 1 的 `batch['inputs']`，不访问 targets 或未来轨迹；尚未接入 Cosmos 主干、point token、XYZ MLP 或位置旋转。

```python
from cosmos_framework.model.generator.pointflow_geometry import SonataGeometryEncoder

encoder = SonataGeometryEncoder(
    "../checkpoints/ptv3/sonata_small.pth", stage=3, freeze=False
).cuda()
inputs = {k: v.cuda() for k, v in batch["inputs"].items()}
geometry = encoder(inputs)
```

默认 enc3，输出 256 维特征；stage 可选 0–4。构造时根据本地 checkpoint 配置创建模型并严格加载，限制为当前 Sonata small encoder。无自动联网、无全局模块路径修改。默认参与训练；`freeze=True` 禁止 backbone 梯度，并在外层调用 train() 时保持 backbone 为 eval。正常 PyTorch `no_grad()` 语义保留。

## 输出

| 字段 | 说明 |
|---|---|
| cluster_features | `[sum(K), C_stage]`，保留梯度 |
| voxel_features | `[sum(V),32]`，enc0 特征，后续逐点 decoder 可通过 original_to_voxel 取回 |
| original_to_cluster / voxel_to_cluster | 原始点 / 输入体素 → batch 全局簇索引 |
| cluster_batch / cluster_offsets | 簇所属原 batch 样本和累计结束偏移，包含空样本的位置 |
| cluster_counts | 每簇原始点数量 |
| cluster_xyz / cluster_uv | 原始相机 XYZ 和 tracker UV 的成员均值 |
| relative_xyz / relative_uv | 每个原始点相对所属簇中心的差值 |
| point_ids / point_batch / point_offsets / has_geometry | 原输入身份与变长边界 |

统计按所有原始成员进行，不能把不同成员数的体素中心等权平均。XYZ 保持米制固定相机坐标，UV 仍是 tracker 图像像素，尚非 Cosmos 拼图 patch 坐标。

## 空样本和映射

传入 Sonata 前，根据 has_geometry 去除空样本的重复 offset，把非空样本重新编号为连续 batch。按 pooling_parent 找到目标层，组合 pooling_inverse，随后恢复输出 cluster_batch 为原 batch 编号，并验证所有原始点没有跨样本映射。

全空批次不运行 sparse/FlashAttention，返回零行特征与完整空 offset；未冻结模型的空特征包含参数关联的零项，使调用方可对 feature.sum() 反向得到零梯度。冻结且全空时没有模型梯度路径，调用方不应单独对此反向。

当前为保持与已验证官方调用一致，仍执行完整五层 encoder 后取中间层，尚未截断计算或融合 enc4。若只对 enc3 输出施加 loss，后续 enc4 参数可能没有梯度；未来 FSDP/优化器任务必须处理，不代表本任务已经验证分布式训练。

## 验证入口

在 H200 上：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/launch_pointflow_geometry.sh
```

该脚本使用任务 1 Dataset，构造两个普通真实窗口（512 点和 256 点）及一个全无效尾段窗口。验证全空 batch 反向、混合 batch 原始编号恢复、输出有限性及输入层非零梯度。默认 stage=3，可设置 STAGE=2；DATA_ROOT、SONATA_CHECKPOINT、OUTPUT_DIR、DEVICE、SEED、PYTHON_BIN 可覆盖。默认报告 `pointflow_outputs/task2/report.json`。

CPU 验证：

```bash
DEVICE=cpu bash examples/launch_pointflow_geometry.sh
LD_LIBRARY_PATH='' .venv/bin/python -m unittest cosmos_framework.model.generator.pointflow_geometry_test -v
```

本地已通过官方权重严格加载、真实全空窗口反向，以及原始成员加权、相对坐标恢复、跨样本错误检测、冻结模式测试。CPU 测试以小型假层级独立验证中间层映射和空样本重新编号，不模拟 CUDA 算子。新版封装的 GPU 混合 batch 尚待上述 H200 命令验证，不能以此前 smoke 的 GPU 成功代替。

任务 2 实现完成后停止；尚未进入任务 3 的运动 MLP、point token 或逐点解码。
