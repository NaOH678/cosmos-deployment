# PointFlow：独立运动编码与逐点几何—运动联合编码

> 日期：2026-10-01。状态：联合编码已实现为默认关闭的可选路径；训练效果尚未验证。
> 本文比较现有 cluster + skip/pb 与拟议的池化前联合编码；不改变正在运行的 stage0 实验。

## 1. 符号与现有结构

每个预测窗口包含 32 个未来状态，每 4 步组成一个时间块，共 8 个未来时间块。设：

| 符号 | 含义 |
| --- | --- |
| $i$ | 原始点索引 |
| $C_j$ | 第 $j$ 个簇的成员点集合，由 anchor 几何确定 |
| $b\in\{1,\ldots,8\}$ | 未来时间块索引 |
| $\sigma$ | 扩散噪声等级，与物理时间分开 |
| $\mathbf d^\sigma_{b,i}\in\mathbb R^{12}$ | 点 $i$ 的四步带噪位移，按时间顺序拼接 XYZ |
| $\mathbf f_i$ | Sonata level0 特征，通过 original-to-voxel 映射取得；同体素内的点可以共享此特征 |
| $\mathbf r_i=(\mathbf x_i^0-\bar{\mathbf x}_j^0)/s_{xyz}$ | 归一化的簇内相对位置 |
| $\mathbf G_j$ | 簇级几何内容，含 Sonata 特征与簇中心 XYZ 编码 |
| $K$ | 实际簇数 |

几何内容可写为：

$$
\mathbf G_j
=W_F\mathbf F_j^{\mathrm{Sonata}}
+\operatorname{MLP}_{xyz}\left(\bar{\mathbf x}_j^0/s_{xyz}\right).
$$

带噪位移由模型空间的干净位移与噪声混合得到：

$$
\mathbf d^\sigma_{b,i}
=(1-\sigma)\mathbf d^{\mathrm{clean,model}}_{b,i}
+\sigma\boldsymbol\epsilon_{b,i}.
$$

干净位移先按配置的标量或逐帧逐通道 scale 归一化。联合采样时，输入是当前去噪迭代的状态，不需要未来 GT。

代码依据：`cosmos_framework/data/pointflow_window.py:13`、`cosmos_framework/model/generator/pointflow_codec.py:17`、`cosmos_framework/model/generator/pointflow_codec.py:123`、`cosmos_framework/model/generator/pointflow_training.py:9`。

## 2. 当前方案：运动独立编码，在簇级与几何融合

每个点的四步带噪位移先通过同一个 motion MLP，再对簇内成员求平均：

$$
\mathbf E_{b,j}
=\frac{1}{|C_j|}\sum_{i\in C_j}\phi\left(\mathbf d^\sigma_{b,i}\right).
$$

将运动聚合结果与簇级几何拼接，再映射为主干 token：

$$
\boxed{
\mathbf z_{b,j}
=P\left([\mathbf G_j;\mathbf E_{b,j}]\right)
+\mathbf e_{\mathrm{point}}+\tau(\sigma)
}
$$

其中 $\phi$ 是 motion MLP，$P$ 是 LayerNorm 加线性投影，$\tau$ 是噪声等级编码。

```text
每点四步带噪位移 → motion MLP → 簇内平均 ─┐
                                        ├→ 拼接 → 主干 token
anchor 点云 → Sonata → 簇级几何内容 ──────┘
```

这里平均的是非线性运动特征，并非直接平均位移。问题不能简化为“正负位移抵消”：关键是聚合时没有显式绑定各点的几何与运动。

代码依据：`cosmos_framework/model/generator/pointflow_codec.py:119`。

## 3. 建议方案：逐点融合几何与运动，再聚合

先将每个点的几何特征、簇内相对位置与带噪运动联合编码：

$$
\widetilde{\mathbf E}_{b,j}
=\frac{1}{|C_j|}\sum_{i\in C_j}
\psi\left([\mathbf f_i;\mathbf r_i;\mathbf d^\sigma_{b,i}]\right).
$$

主干 token 的后续形成方式保持不变：

$$
\boxed{
\widetilde{\mathbf z}_{b,j}
=P\left([\mathbf G_j;\widetilde{\mathbf E}_{b,j}]\right)
+\mathbf e_{\mathrm{point}}+\tau(\sigma)
}
$$

$\psi$ 是联合非线性编码器。最小方案中，$\sigma$ 仍沿用现有 token 级编码；是否额外将其输入 $\psi$ 是独立设计选项。

```text
每点 [level0 几何、簇内相对 XYZ、四步带噪位移]
                        ↓
                    联合 MLP
                        ↓
                     簇内平均
                        ↓
              与原有簇级几何内容拼接
                        ↓
                    主干 token
```

第一版可以保持簇分配、输出特征维度、anchor token、位置编码、decoder、loss 与 scale 不变，仅替换运动聚合前的编码。每簇仍对应一个 anchor token 和八个未来 token，进入主干的数量仍为 $9K$。

## 4. 数学上的关键区别：是否保留几何—运动配对

固定一个簇及其几何，令 $\pi$ 为簇内点的任意排列。只交换运动的归属，几何保持原位：

$$
\mathbf d^\sigma_{b,i}\longrightarrow\mathbf d^\sigma_{b,\pi(i)}.
$$

当前方案必然满足：

$$
\sum_{i\in C_j}\phi(\mathbf d^\sigma_{b,\pi(i)})
=\sum_{i\in C_j}\phi(\mathbf d^\sigma_{b,i}).
$$

因此，这种交换不会改变该簇、该时间块的主干 token。模型可以感知簇内运动特征的集合，却不能通过这一聚合辨认这些轨迹分别属于哪个几何位置。

新方案不再被强制满足上述不变性：

$$
\sum_{i\in C_j}\psi([\mathbf f_i;\mathbf r_i;\mathbf d^\sigma_{b,\pi(i)}])
\;\not\equiv\;
\sum_{i\in C_j}\psi([\mathbf f_i;\mathbf r_i;\mathbf d^\sigma_{b,i}]).
$$

这里的“不恒等”表示有能力区分，并不保证任意参数、任意输入下结果都不同。

两种方案都应对点的存储顺序保持不变。如果几何与运动一起按同一排列重排，它们仍是同一个点集，聚合结果应相同。新方案保留的是配对信息，而不是点编号或数组顺序。

## 5. 为什么需要非线性交互

记 $\mathbf a_i=[\mathbf f_i;\mathbf r_i]$。如果只做线性拼接投影：

$$
\psi(\mathbf a_i,\mathbf d_i)=A\mathbf a_i+B\mathbf d_i,
$$

则聚合后仍然是：

$$
\frac{1}{|C_j|}\sum_i\psi(\mathbf a_i,\mathbf d_i)
=A\bar{\mathbf a}+B\bar{\mathbf d}.
$$

几何与运动重新分离，无法解决上述配对问题。即便是两个独立非线性编码器相加，$u(\mathbf a_i)+v(\mathbf d_i)$，结论也一样。

用一个示意性交互项可以看清需要增加的表达能力：

$$
\psi(\mathbf a_i,\mathbf d_i)
=A\mathbf a_i+B\mathbf d_i
+C\operatorname{vec}(\mathbf a_i\mathbf d_i^\top).
$$

聚合后包含：

$$
\widetilde{\mathbf E}
=A\bar{\mathbf a}+B\bar{\mathbf d}
+C\operatorname{vec}\left(
\frac{1}{|C_j|}\sum_i\mathbf a_i\mathbf d_i^\top
\right).
$$

最后一项是几何与运动的联合统计。公式中的外积用于解释机制，不要求实现时显式计算；对拼接输入使用联合非线性 MLP，可以学习此类交互。

## 6. 与当前 skip/pb 的关系

现有 decoder 已经获得逐点信息。省略部分实现细节，可写为：

$$
\widehat{\mathbf V}_b
=D_{\mathrm{pb}}\left(
\left\{
[\mathbf H_{b,c(i)};
\mathbf f_i^{\mathrm{skip}};
\mathbf r_i;\Delta\mathbf u_i;
\mathbf d^\sigma_{b,i};\tau(\sigma)]
\right\}_{i=1}^{N}
\right).
$$

$\mathbf H_{b,c(i)}$ 是主干输出按簇归属广播后的 hidden state；$D_{\mathrm{pb}}$ 在每个样本、每个时间块内对点做自注意力，再预测逐点 velocity。因此，上文的运动交换即使不改变主干输入，也可能改变现有 decoder 的输出。

建议改动的目的，是让逐点几何—运动配对在生成 $\mathbf H$ 之前，就参与 video/action/point 的联合交互，而不只在末端 decoder 中使用。它并不意味着现有 skip/pb 无法区分点，也不意味着当前最终输出必然相同。

代码依据：`cosmos_framework/model/generator/pointflow_codec.py:142`、`cosmos_framework/model/generator/pointflow_point_decoder.py:29`。

## 7. 与 stage0 实验的关系及验证边界

| 改动 | 主要检验的问题 |
| --- | --- |
| stage1 → stage0 | 更细的空间分组能否减少混合、提高效果；同时也改变几何特征层级与维度 |
| 池化前联合编码 | 在给定分组下，保留几何—运动配对是否有收益 |
| decoder skip/pb | 利用逐点特征和点间交互，能否从主干表示恢复更准确的逐点运动 |

现有记录包含大运动 case 的误差和部分静态点漂移，但尚未证明这些现象由上述编码不变性主导。高噪声时运动输入接近随机，新增配对信息也未必提供有效信号；有限维度的均值聚合仍可能丢失细节。

因此本文是一个可检验的结构假设，不是效果保证。验证时应保持其他设置一致，同时检查 cond/joint ADE、静态漂移、按运动幅度分组的误差与额外计算开销。

实验依据：`docs/pointflow_form_comparison_observations_20261001.md:149`。簇粒度扫描：`docs/pointflow_cluster_decode_20260929.md:111`。

## 8. 实现与启用

设置 `POINTFLOW_GEOMETRY_MOTION_FUSION=true` 启用，默认 `false`。开启后 motion MLP
输入依次为 `[四步带噪位移(12); level0特征(32); 相对XYZ(3)]`，即47维；隐藏和输出
仍为256维。相对XYZ沿用固定 `xyz_scale`，不是运动的 displacement scale。第一层增加
8960个权重；实际时间和显存开销需上卡测量，不能由参数量直接推断。

`PointFlowBranch` 向 codec 传递现有 inputs，供 `original_to_voxel` 映射使用；
训练、cond采样和joint采样共享这条网络路径。安装日志打印 `geometry_motion_fusion=True/False`，
可确认实际开关。额外 decoder skip 特征不进入本次融合。

首次实验使用 stage1 + skip(1,2) + pb4，和已有8卡 E2s 对照：

```bash
OUTPUT_ROOT=/data/shichaojian/runs/cluster_500p_stage1_geomotion_skip_pb4_20261001 \
POINTFLOW_TOKEN_MODE=cluster \
POINTFLOW_SONATA_STAGE=1 \
POINTFLOW_CLUSTER_TOKEN_CAP=128 \
POINTFLOW_GEOMETRY_MOTION_FUSION=true \
POINTFLOW_DECODE_SKIP_LEVELS=1,2 \
POINTFLOW_DECODE_POINT_BLOCKS=4 \
POINTFLOW_DISPLACEMENT_SCALE=0.0528 \
EXTRA_TAIL_OVERRIDES="trainer.grad_accum_iter=1" \
NPROC_PER_NODE=8 NNODES=1 \
bash examples/launch_pointflow_sandwich101.sh
```

切换开关改变 motion encoder 权重形状，必须用新 OUTPUT_ROOT 从基座开始；续训需保持
开关一致。默认关闭时保留原参数键、形状及计算路径，可继续加载旧的未融合 checkpoint。
当前运行中的 stage0 实验无需调整。本次实现不自动提交训练任务。

CPU测试见 `cosmos_framework/model/generator/pointflow_geometry_motion_test.py:1`：
运动单独交换的敏感性、几何与运动一起重排的不变性、编码器梯度、样本隔离、空样本、
点级decoder兼容及默认路径checkpoint兼容。主网络集成的开关双路径测试见
`cosmos_framework/model/generator/pointflow_branch_test.py:44`。

验证边界：codec/decoder 的14项CPU测试通过。默认主网络注意力在本机CPU上没有兼容
backend，设置 `POINTFLOW_REFERENCE_ATTENTION=true` 后，开关双路径的2项主网络
前向/反向及未来标签隔离测试通过；
这不等于生产GPU注意力后端已验证。另核实全空点batch配合pb时，原有PointDecoder
的norm1/qkv参数不产生梯度，融合开关两侧一致；本改动未扩展处理这一已有边界。
