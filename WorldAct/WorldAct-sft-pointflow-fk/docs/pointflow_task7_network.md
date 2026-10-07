# 任务 7：PointFlow 接入 Cosmos 主网络

任务 7 将任务 2–4 的几何编码、运动 Codec 和 point attention 接入 Cosmos3VFMNetwork.forward。任务 6 的 PackedSequence.pointflow_data 现在可以在网络内编码，并输出原始点的速度预测。

## 前向数据流

1. 输入任务 6 的 packed sequence，以及显式的 pointflow_displacement [H,sum(N),3]、pointflow_sigma [B]。
2. PointFlowBranch.geometry 读取 anchor inputs，在网络内执行预训练 Sonata，取得实际簇数 K、enc3 特征、enc0 特征与原始点映射。
3. Codec 编码 anchor 和 noisy displacement，输出 anchor [sum(K),D]、motion [H/q,sum(K),D]。
4. 从实际视频 sequence position_ids 提取每个样本的时间/空间原点；将任务 5 的 uv_to_video 复合为 patch 坐标，生成 point 位置。
5. 复用 attach_point_tokens，将每个样本的 point tokens 插入对应 full-attention 段；统一更新视频/action/text 索引。
6. 原有 VFM video/action 投影和 Cosmos MoT attention 正常执行。point token 走 generation expert；video↔point 使用完整时空 RoPE，action↔point 使用时间 RoPE。
7. gather point hidden，再通过共享 MLP 解码到原始 N 个点，输出 preds_pointflow [H,sum(N),3]；preds_vision、preds_action 继续输出。

网络只读取 pointflow_data.inputs 和 timing，不读取未来 displacement target 或 valid。Noisy displacement 和 sigma 必须由调用者显式传入，缺少时直接报错。速度的归一化尺度与正式 RF loss/sampler 属于任务 8。

## 安装接口

在**基础网络已 materialize、初始化并加载基础权重之后，optimizer/FSDP 创建之前**调用：

```python
net.install_pointflow(sonata_checkpoint, timing=timing, stage=3)
output = net(
    packed_sequence,
    pointflow_displacement=noisy_displacement,
    pointflow_sigma=sigma,
)
velocity = output["preds_pointflow"]
```

Sonata 和 Codec 注册为 net.pointflow_branch 子模块，进入 parameters/state_dict。安装只创建新分支，不调用基础网络 init_weights。Sonata 初始保持 FP32，Codec 匹配网络参数 dtype。可通过 freeze_geometry 冻结 Sonata。

现有 OmniMoTModel.build_net 的 meta → FSDP → to_empty 流程尚未自动调用此接口。任务 9 会接入初始化、分布式包装、EMA、checkpoint/resume；不能把本接口直接插进现有 meta 构造块，否则可能丢失 Sonata 权重。本任务验证直接构造 materialized VFM 网络。

## 位置与长度

令 S=VAE spatial compression×latent patch size，则视频像素中心到 patch 中心采用 p=(u+0.5)/S−0.5。复合已有 tracker→video 仿射后，再加上本样本实际视频的空间原点；不猜测 wrist/head 拼接高度。

时间位置沿用任务 4：以实际视频 anchor 时间原点为起点，未来 block 的物理秒数乘 24/4。校验视频 latent 时间轴与 PointFlow timing 相同，避免忘记启用 FPS modulation。

支持 CP=1、two-way、单个拼接视频 item、默认 q=4；不支持 temporal causal、KV memory、controls、CUDA graphs。保留变长 offsets，无标注样本不插入 point token；全空 anchor batch 仍能返回 [H,0,3]。

实际 K 在网络中产生，task 6 的 V×(1+H/q) 预算只是保守上界。这里不重新定义点的生命周期，也不启用后续 chunk 最大槽位方案。

## 验证与边界

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/launch_pointflow_network.sh
```

默认读取任务 5 的 mixed_manifest.json 和 sonata_small.pth；可用 POINTFLOW_MANIFEST、SONATA_CHECKPOINT、OUTPUT_DIR、SEED 覆盖。输出 pointflow_outputs/task7/report.json。

GPU 脚本调用真实 Cosmos3VFMNetwork.forward、真实单层 PackedAttentionMoT、预训练 Sonata 和 Codec。Cosmos attention 使用随机权重；video/action latent 是合成输入。为对应小型 latent 网格，显式将轨迹 UV 仿射缩放到合成的 16×16 画布。该验证不是完整 Cosmos 基础 checkpoint 或真实分辨率训练验证。

检查输出形状、基础 VFM 权重保持不变，以及 point 预测梯度到 Sonata、Codec、attention、video/action 输入投影的连通性。损失为合成梯度探针，不是联合训练 loss。

本节点没有 CUDA。CPU 测试使用可微几何替代模块，实际运行 VFM 的 video/action heads 和 point attention，验证全链路梯度、未来标签隔离、空 batch、位置映射与错误配置拒绝。真实 Sonata GPU 前向/反向需要在 H200 上运行上述脚本。

下一步任务 8：在训练循环中构造 point noised state、加入 masked RF loss，并接入 video/action/point 联合采样。

本轮 13 项 CPU 集成/回归测试通过，记录在 pointflow_outputs/task7/cpu_report.json。Ruff 检查与 git diff --check 通过。
