# Cosmos PointFlow 位置编码方案

最终方案是：**UV 对齐视频、XYZ 表达三维几何、真实时间对齐三种模态；action–point 只比较时间位置。** 这是已确定的设计，尚未实现到训练代码。

> **实现核查见 `pointflow_alignment_audit_20260913.md`（2026-09-13）。** 结论：§1–§3 的
> **位置计算已实现且正确**（UV 走与视频一致的变换、时间与视频 latent 同格，均有运行时断言）；
> **只有 §4 的 action–point 规则未实现** —— 实际跑的是 `legacy_mrope`，action↔point 的空间通道照转。
> 该文还给出了**实测的位置三元组**和修法方案，以及一条重要对照：**原生 video↔action 用的是同一机制**，
> 所以这属于"设计与实现不一致"，不能简单称为缺陷。

## 1. 每个 point token 携带什么？

将当前点云通过 PTv3 聚成点簇，每个点簇在不同未来时间片生成 token：

$$
\mathbf z_{b,j}
=
\operatorname{Point2LLM}
\left(
[\mathbf G_j;\mathbf E_{b,j}]
\right)
+\mathbf e_{\mathrm{point}}
+\operatorname{TimeEmbed}(\sigma)
$$

其中：

- $\mathbf G_j$：PTv3 几何特征，加上**固定相机坐标系下的 XYZ 中心编码**。
- $\mathbf E_{b,j}$：该时间片内 **4 个三维带噪位移**，按顺序拼成 12 维后编码、聚合。
- $\sigma$：扩散去噪进度，与真实时间分开。

此外，token 携带用于 attention 的位置：

$$
\mathbf p_{b,j}=(t_b,\bar h_j,\bar w_j)
$$

也就是说，**XYZ 和运动进入 token 内容，时间与图像位置进入 Q/K 的 MRoPE**。参见[主设计文档：267](pointflow_ptv3_cosmos_design.md:267)。

## 2. 图像位置：使用当前 anchor 的 UV

当前点的 UV 经过与视频完全相同的 resize、拼图偏移、padding，再换算到视频 token 网格，得到连续浮点坐标 $(h,w)$。每个点簇使用成员点的平均坐标。

**同一簇的未来 tokens 都沿用当前 anchor UV**，含义是“从这里出发的点，未来怎样运动”。未来位置由预测位移表示，首版不使用未来 GT UV 做位置编码。

这个显式投影只对应 **head 视图**；与 wrist 的关系由模型学习。参见[主设计文档：234](pointflow_ptv3_cosmos_design.md:234)。

### UV 如何参与 attention？

**UV 是通过 attention 中 Q、K 的 MRoPE 旋转加入的。** 在当前方案中，它作为位置元数据传入；XYZ 和运动则编码进 token 内容。

**① 将像素 UV 转成视频 token 网格坐标。**

当前点的 UV 跟随视频做 resize、拼图偏移和 padding，再换算成连续的 $(h,w)$。点簇使用成员点坐标的平均值，其中 $C_j$ 表示第 $j$ 个点簇包含的原始点集合：

$$
\bar h_j=\frac{1}{|C_j|}\sum_{i\in C_j}h_i,
\qquad
\bar w_j=\frac{1}{|C_j|}\sum_{i\in C_j}w_i
$$

于是第 $b$ 个时间片、第 $j$ 个点簇的位置是：

$$
p_{b,j}=(t_b,\bar h_j,\bar w_j)
$$

**② 用这些坐标旋转 Q、K 中对应的通道。**

示意写法如下，其中 $T,H,W$ 分别表示时间、高度和宽度对应的通道：

$$
\widetilde{\mathbf q}
=
\left[
R_T(t)\mathbf q_T;\;
R_H(h)\mathbf q_H;\;
R_W(w)\mathbf q_W
\right]
$$

K 做相同处理。这里的“加入”是**改变 Q、K 的旋转相位**，而不是把 UV 数值直接加到特征上。公式按坐标轴分组便于解释；实际 Edge MRoPE 通道采用交错布局，若有未旋转通道则保持原值。

**③ 旋转后的 Q、K 决定 attention 分数。**

以高度通道为例，视频位置 $h_v$ 与点簇位置 $\bar h_j$ 的交互变为：

$$
\left(R_H(h_v)\mathbf q_H\right)^\top
\left(R_H(\bar h_j)\mathbf k_H\right)
=
\mathbf q_H^\top
R_H(\bar h_j-h_v)
\mathbf k_H
$$

因此，attention 分数能够感知**视频 patch 与点簇投影位置的相对位移**。宽度方向同理。不过，RoPE 并不保证距离越近、权重就越大，具体关联仍由模型学习。

这套 UV 旋转用于 **video–point 和 point–point**；**action–point 只做时间旋转**，空间通道使用未旋转的内容点积。同一簇的各个未来 token 沿用 anchor UV，时间位置随时间片变化。参见[主设计文档：317](pointflow_ptv3_cosmos_design.md:317)。

## 3. 时间位置：统一 Cosmos 的物理时间尺度

当前配置每秒对应 6 个 MRoPE 时间单位：

$$
t_{\mathrm{RoPE}}=t_{\mathrm{start}}+6\Delta t
$$

PointFlow 按 15 Hz 预测，每 4 个未来状态组成一片，使用**时间片右端点**作为代表位置：

| Point token | 覆盖目标状态 | 代表时间 | 相对 MRoPE 时间 |
|---|---|---|---|
| Anchor | 当前状态 | 0 秒 | 0 |
| 第 1 片 | 第 1–4 步 | 4/15 秒 | 1.6 |
| 第 2 片 | 第 5–8 步 | 8/15 秒 | 3.2 |
| … | … | … | … |
| 第 8 片 | 第 29–32 步 | 32/15 秒 | 12.8 |

这与视频 **9 个 latent 的名义时间位置**对齐。片内四步仍保留先后顺序；action 保留 Cosmos 原有 state/动作 offset，明确区分控制区间起点和结果状态时刻。参见[主设计文档：280](pointflow_ptv3_cosmos_design.md:280)。

## 4. 跨模态 attention：按 token 对选择位置规则

| Attention 关系 | 位置编码规则 |
|---|---|
| Video ↔ Point | 共享时间 + 图像 h/w 旋转 |
| Point ↔ Point | 共享时间 + anchor h/w 旋转；XYZ 几何在内容中 |
| Action ↔ Point | **只做时间旋转**；空间通道保留原始内容点积 |
| 原有 Video / Action 相互关系 | 保留 Cosmos 现有规则 |

一个 action token 是整个关节向量，没有唯一像素位置。因此，action–point 不应将动作的 `(0,0)` 与 point 的像素坐标计算空间相位差。

桥接关系由此形成：**Video 通过 UV 和时间关联 Point；Action 通过时间、三维几何与运动内容关联 Point。** 它是通过联合监督学习的对应关系；目前没有机器人外参和 FK 提供的显式空间对应。参见[主设计文档：317](pointflow_ptv3_cosmos_design.md:317)。

## 5. 统一形式：从点几何到跨模态 attention

位置编码通过两条路径汇合：**XYZ、局部结构和运动构成 token 内容；UV 与物理时间决定 Q/K 的位置旋转。** 两者在 attention 分数中共同起作用，随后聚合各模态的内容特征。

### 5.1 点簇的内容与位置

令 $i$ 为原始点，$j$ 为固定点簇，$b=1,\ldots,8$ 为未来时间片。当前可观测点具有相机坐标 $\mathbf X_i^0$ 和图像坐标 $\mathbf u_i^0$。$C_j$ 是仅根据当前观测确定的点簇成员集合，整个预测窗口保持不变。定义：

$$
\bar{\mathbf X}_j^0
=\frac{1}{|C_j|}\sum_{i\in C_j}\mathbf X_i^0,
\qquad
(\bar h_j,\bar w_j)
=\frac{1}{|C_j|}\sum_{i\in C_j}\mathcal T_{\mathrm{image}}(\mathbf u_i^0)
$$

$\mathcal T_{\mathrm{image}}$ 表示与视频一致的图像变换及 token 网格坐标换算。当前几何和未来带噪运动分别编码为：

$$
\mathbf G_j
=W_F\mathbf F_j^{\mathrm{PTv3}}
+\operatorname{MLP}_{xyz}
\left(\frac{\bar{\mathbf X}_j^0-\mathbf o}{s_{xyz}}\right)
$$

$$
I_b=\{4b-3,4b-2,4b-1,4b\},
\qquad
\mathbf E_{b,j}
=\frac{1}{|C_j|}\sum_{i\in C_j}
\operatorname{MLP}_{motion}
\left(\operatorname{Concat}_{k\in I_b}\mathbf d_{\sigma,k,i}\right)
$$

其中 $\mathbf o$ 为固定相机原点，$s_{xyz}$ 为固定空间尺度；每个点的运动输入是有序的 12 维向量。由此得到 point token 的内容和位置元数据：

$$
\boxed{
\begin{aligned}
\mathbf z^P_{b,j}
&=\operatorname{Point2LLM}
\left(\operatorname{LN}([\mathbf G_j;\mathbf E_{b,j}])\right)
+\mathbf e_P+\operatorname{TimeEmbed}(\sigma),\\
p^P_{b,j}
&=\left(t_{\mathrm{start}}+6\frac{4b}{15},\bar h_j,\bar w_j\right).
\end{aligned}
}
$$

这里显式写出主设计中的 LN，第 1 节为简写。anchor token 的时间为 $t_{\mathrm{start}}$，内容来自当前几何；未来 token 的相对时间为 $1.6b$，空间位置始终来自 anchor。$\sigma$ 只表征去噪进度，不进入物理时间坐标。

### 5.2 三种模态共享的 attention 形式

将 video、action、point tokens 放入联合序列。令 $a,c$ 为其中两个 token 的索引，$m_a,m_c\in\{V,A,P\}$ 为模态，$\mathbf z_a$ 为当前层输入 hidden state。先作内容投影：

$$
\mathbf q_a=W_Q\mathbf z_a,
\qquad
\mathbf k_c=W_K\mathbf z_c,
\qquad
\mathbf v_c=W_V\mathbf z_c
$$

以上省略层号和 head 下标。按 MRoPE 的轴分配，将 Q/K 记为时间部分 $T$、空间部分 $S$ 和可选的未旋转部分 $U$；这是逻辑分组，实际通道可以交错排列。令 $R_S(h,w)$ 表示高度、宽度各自对应的旋转。对涉及 point 的 token 对，空间分数定义为：

$$
\Phi_S(a,c)=
\begin{cases}
\left\langle R_S(h_a,w_a)\mathbf q_a^S,
R_S(h_c,w_c)\mathbf k_c^S\right\rangle,
& (m_a,m_c)\in\{(V,P),(P,V),(P,P)\},\\
\left\langle\mathbf q_a^S,\mathbf k_c^S\right\rangle,
& (m_a,m_c)\in\{(A,P),(P,A)\}.
\end{cases}
$$

于是这些 token 对的完整 attention 分数为：

$$
\boxed{
s_{ac}
=\frac{
\left\langle R_T(t_a)\mathbf q_a^T,R_T(t_c)\mathbf k_c^T\right\rangle
+\Phi_S(a,c)
+\left\langle\mathbf q_a^U,\mathbf k_c^U\right\rangle
}{\sqrt{d_{\mathrm{head}}}}
}
$$

没有 $U$ 通道时省略该项。所有时间坐标均使用 $t=t_{\mathrm{start}}+6\Delta t$，action 的 $\Delta t$ 遵循 Cosmos 原有 state/控制区间 offset。原有 video–video、video–action、action–action，以及 text/understanding 路径继续使用 Cosmos 原有分数规则。

对同一个 query，所有合法模态的 key 共同归一化并聚合：

$$
\alpha_{ac}
=\frac{\exp(s_{ac}+M_{ac})}
{\sum_{\ell}\exp(s_{a\ell}+M_{a\ell})},
\qquad
\mathbf o_a=\sum_c\alpha_{ac}\mathbf v_c
$$

$M$ 是样本隔离、padding 等既定 attention mask。这里表明 **UV 和时间经 Q/K 影响权重，XYZ 和运动经 token 内容影响 Q/K/V；它们最终在同一次 attention 聚合中汇合**。action–point 的空间通道仍参与内容匹配，只是不附加图像坐标旋转。

### 5.3 从联合特征回到逐点运动

经过 Cosmos 多层联合 attention 后，point hidden state $\mathbf H^P_{b,j}$ 同时包含视频、动作和点轨迹上下文。令 $c(i)$ 为原始点所属点簇，逐点解码可概括为：

$$
\widehat{\mathbf v}_{\sigma,I_b,i}
=D\left(
\mathbf H^P_{b,c(i)},
\mathbf f_i^{\mathrm{PTv3}},
\Delta\mathbf X_i^0,
\Delta\mathbf u_i^0,
\mathbf d_{\sigma,I_b,i},
\sigma
\right)
\in\mathbb R^{4\times3}
$$

其中 $\Delta\mathbf X_i^0$ 和 $\Delta\mathbf u_i^0$ 是原始点相对所属簇中心的 XYZ/UV，保留簇内细节；$\widehat{\mathbf v}$ 为 flow-matching 速度预测，与 attention 的 value 向量 $\mathbf v_c$ 含义不同。经去噪采样得到位移后，在物理单位下恢复轨迹：

$$
\widehat{\mathbf X}_{k,i}
=\mathbf X_i^0+\widehat{\mathbf d}^{\mathrm{cam}}_{k,i},
\qquad k=1,\ldots,32.
$$

因此，整套方案的连接关系是：**当前 XYZ 提供三维几何内容，当前 UV 提供与视频一致的空间相位，物理时间提供跨模态时间相位；联合 attention 将这些信息与动作、运动内容关联，再由逐点解码器恢复未来轨迹。** 这种桥接由联合训练学习，不预设 action 具有像素坐标，也不使用真实未来 UV 作为位置条件。相关定义见[主设计文档：178](pointflow_ptv3_cosmos_design.md:178)与[位置编码设计：209](pointflow_ptv3_cosmos_design.md:209)。
