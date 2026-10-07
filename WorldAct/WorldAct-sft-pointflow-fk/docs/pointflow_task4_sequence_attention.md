# 任务 4：Point token 接入 Cosmos 序列与位置 attention

本任务接通 `PointFlowCodec → PackedSequence → Cosmos 生成分支 attention → point_hidden → 逐点 decoder`。不包含联合训练 loss、RF 采样、数据集与训练配方的自动装配；这些留到后续任务。

## 序列与变长

`attach_point_tokens()` 位于 `cosmos_framework/model/generator/pointflow_sequence.py`。输入是原 Cosmos `PackedSequence`、任务 3 的 codec 输出和 point 的连续位置。每个样本仍有一段 causal text 和一段 full generation；在该样本的 full 段末尾追加 anchor（可关闭）及按时间片排列的 noisy point tokens。

样本 b 有 K_b 个簇、L 个未来时间片时，新增 `(L+1)K_b` 个 token（不使用 anchor 则为 `LK_b`）。不会把不同样本补齐到同一个 K。K_b=0 时该样本的视频和动作仍保留。原 text/vision/action/sound 的序列索引、监督索引、payload spans 均重映射；原 payload 内容不变。函数返回新对象，不原地修改输入序列。

VFM 将 point 内容直接填入 hidden buffer，送入已有生成分支的 QKV、输出投影及后续 Transformer 层。`forward()` 额外返回 `point_hidden[L, ΣK_b, D]`，按原簇顺序恢复，可以直接传给 `codec.decode()`；anchor 不混入 noisy hidden。

## 位置如何接起来

`point_positions()` 返回 `[1+L, ΣK_b, 3]`，最后一维按 Cosmos 的 `(t,h,w)` 排列，保持浮点数。

$$
\begin{bmatrix}w_j\\h_j\end{bmatrix}
=A_b\begin{bmatrix}\bar u_j\\\bar v_j\\1\end{bmatrix},\qquad
 t_{b\ell}=t_b^{\mathrm{video},0}+\frac{24}{4}\Delta s_\ell.
$$

A_b 是 **实际图像变换得到的** tracker UV 到视频 patch 坐标的 2×3 仿射矩阵，包含 tracker/head 的尺寸转换、crop、resize、wrist/head 拼接偏移、padding、VAE 与 patch 网格映射，以及所用的像素中心约定。接口要求显式传入该矩阵，不根据图像名字猜变换。若视频空间 mRoPE 不从零开始，可额外传入 `(w,h)` 的 `spatial_origin`。

这里的簇 UV 是原始成员 anchor UV 的平均值。仿射变换与求平均可交换，因此与先逐点变换再聚合一致。非线性相机畸变校正不属于该仿射接口，需先逐点处理再聚合。

时间原点必须取对应视频首个 latent 的位置；24/4 是 Cosmos 当前 mRoPE 基准时间倍率，不是把轨迹重新采样为 24 Hz。默认 15 Hz、q=4 时，anchor 与未来块的相对时间位置为 0、1.6、…、12.8。输入 `block_seconds` 是未来结果时间；不会改动已有 action 的控制区间起点位置。

XYZ 和带噪运动仍由任务 3 编入 token 内容。本任务不加入 GT future UV，也不根据未来有效掩码选择 token。

## Attention 规则

`attention_mode="pairwise_point_mrope"` 是默认模式：

| token 对 | Q/K 旋转 |
|---|---|
| video–point、point–point | 时间 + anchor UV |
| action–point（双向） | 仅时间；空间通道保留未旋转内容 |
| 其余组合 | Cosmos 原有 mRoPE |

`legacy_mrope` 可作为消融，所有组合使用原生完整 mRoPE。没有 point payload 时仍走原来的 dispatch。

实现读取 Edge 配置中的 `mrope_section`，按其 **交错频率索引** 识别 H/W 通道，并同时处理两个 rotary 半区。不会把通道错误地切成连续 T/H/W 三段。

$$
S_{ij}=\frac{1}{\sqrt d}
\begin{cases}
\langle Q_i^{t},K_j^{t}\rangle,& (i,j)\in\{(a,p),(p,a)\},\\
\langle Q_i^{thw},K_j^{thw}\rangle,&\text{其他组合},
\end{cases}
\qquad O_i=\sum_{j\in\mathcal V_i}\operatorname{softmax}_{j\in\mathcal V_i}(S_{ij})V_j.
$$

所有合法 key 共享一次 softmax。不同样本完全隔离；causal text 只能看本样本此前 text，generation 看本样本全部 token。保留 GQA，以及 Cosmos 对 generation→text 的额外 K normalization。

## 验证与启动

CPU 检查使用真实 `PackedAttentionMoT` 与 Edge rotary，另有逐 query/head 的独立数值 oracle、变长索引恢复、空点、梯度及 VFM forward 接口测试。VFM 接口测试使用轻量合成模态编码器，不加载完整基座。

```bash
LD_LIBRARY_PATH='' OMP_NUM_THREADS=4 .venv/bin/python -m unittest \
  cosmos_framework.model.generator.pointflow_sequence_test -v
```

H200 验证入口：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/launch_pointflow_sequence.sh
```

输出 `pointflow_outputs/task4/report.json`。沿用任务 3 的真实 512/空/256 点窗口与 Sonata 权重，经过 2048 维 **真实 Cosmos 单层 attention**，再逐点解码并反向验证梯度。视频和动作 token 是合成输入，attention 权重随机初始化；UV 仿射也显式标记为合成验证变换。这不是完整 Cosmos checkpoint 推理或训练质量评估。

GPU 结果需在 H200 上运行后确认；当前开发环境无可见 GPU，不能以 CPU 测试替代。

## 当前边界

参考实现支持 Edge、CP=1、two-way；不支持 temporal-causal、KV memory、多控制权重及 CUDA graph padding，相关 VFM 路径会明确报错。尚未验证 torch.compile、FSDP 与整模型训练。

参考 attention 用 FP32 logits 和按 query 分块计算，适合核对规则与小规模联调。计算仍为平方复杂度，训练反向保存的 attention 中间量也可累积为平方规模；分块不等于 FlashAttention 的显存开销。完整视频的大规模训练需要后续优化，不能据此声称已达到正式训练吞吐。

生产数据管线的实际 UV 仿射元信息、模型子模块注册、联合 loss 和采样调用，将在后续任务接线；本任务提供明确参数接口及 Cosmos 内部消费路径。

## 后续改进保留项：chunk 内固定最大点数 + mask（未实现）

当前实现保持“固定首帧 anchor 集合 + 逐样本变长 K_b”，这样不会引入固定矩形 mask 规则，便于先验证语义链路。  
若后续希望改为“同一 chunk/episode 内按统一 token 预算训练”，可保留该思路为下一阶段任务：

- 在 chunk 级别预统计 `K_max(chunk)`（或用全局上分位数作为预算）。
- 每个样本在 `attach_point_tokens()` 阶段补齐到该 chunk 的 `K_max`，不足部分补 `pad` 点（不做几何/运动有效性聚合）。
- 新增 `point_active_mask[b, j]∈{0,1}`：
  - Attention 读写使用 `point_active_mask` 屏蔽 pad 行；
  - Loss 统计使用 `point_active_mask & target_valid` 作为分母/分子掩码；
  - 可视化与指标忽略无效 slot。
- 帧间新出现点仍默认不动态加入；chunk 内新增点可以映射为“inactive slot”，在下一窗口重选 anchor 时才纳入。

这类实现可更容易利用静态形状优化（例如某些 kernel/编译路径），但会增加数据流复杂度：需要保证 packer、loss、attention、sampler、eval 指标都消费同一套 `point_active_mask`，且不能把它与 `target_valid` 混用。当前版本先不展开实现，以免在时空编码与时间对齐未完全收敛阶段引入额外回归风险。
