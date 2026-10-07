# 当前 local RoPE 的 Q/K/V 矩阵

第 1–3 节解释三轴/四轴混合规则；第 4 节画出已实现的 160 维扩展方案。
160 维版已通过 CPU 和 A800 BF16 的输出、梯度、checkpoint、compile 检查；测速记录见局部四轴文档。
只画同一样本的四种生成模态、一个 query head 及其对应的 KV head；
为看清矩阵，文字及其 causal mask 暂不画。实际代码仍保留文字的独立 K normalization、
因果 mask 和不同样本之间的隔离，不能直接对整个 packed batch 无掩码相乘。

## 1. 每一行对应一个 token

- `I`：video，`A`：action，`P`：PointFlow，`F`：FK。
- 一个模态通常有很多 token，所以图中的每一行是一个矩阵块，而不是只有一个 token。
- 每个 head 的 Q/K/V 各有 128 个特征维度。上标 3/4 表示旋转使用三轴/四轴，
  不表示张量只有 3/4 维，也不表示额外学习了一套 Q/K 投影。

先由主干相应的投影得到 Q、K、V，再旋转 Q/K。V 不做 RoPE：

$$
Q^{(3)} = \begin{bmatrix}Q_I^{(3)}\\Q_A^{(3)}\\Q_P^{(3)}\\Q_F^{(3)}\end{bmatrix},
\qquad
K^{(3)} = \begin{bmatrix}K_I^{(3)}\\K_A^{(3)}\\K_P^{(3)}\\K_F^{(3)}\end{bmatrix},
\qquad
V = \begin{bmatrix}V_I\\V_A\\V_P\\V_F\end{bmatrix}.
$$

额外只有几何 token 需要四轴 Q/K：

$$
Q_G^{(4)} = \begin{bmatrix}Q_P^{(4)}\\Q_F^{(4)}\end{bmatrix},
\qquad
K_G^{(4)} = \begin{bmatrix}K_P^{(4)}\\K_F^{(4)}\end{bmatrix},
\qquad G=P\cup F.
$$

`Q_P^(3)` 与 `Q_P^(4)` 来自同一份旋转前的 Q，只是位置旋转不同；K 同理。
不存在需要相加的两套 V。

## 2. 最终分数矩阵：只有右下角用四轴

行表示 query，列表示 key。每个格子对应一类 query/key 配对：

```text
                         Key
                  video   action   PointFlow   FK
                ┌────────┬────────┬───────────┬────────┐
       video    │   3    │   3    │     3     │   3    │
Query  action   │   3    │   3    │     3     │   3    │
       PointFlow│   3    │   3    │     4     │   4    │
       FK       │   3    │   3    │     4     │   4    │
                └────────┴────────┴───────────┴────────┘

3 = 用三轴 Q 与三轴 K 点积
4 = 用四轴 Q 与四轴 K 点积
```

完整矩阵为：

$$
S=\frac{1}{\sqrt{128}}
\begin{bmatrix}
Q_I^{(3)}(K_I^{(3)})^T & Q_I^{(3)}(K_A^{(3)})^T & Q_I^{(3)}(K_P^{(3)})^T & Q_I^{(3)}(K_F^{(3)})^T\\
Q_A^{(3)}(K_I^{(3)})^T & Q_A^{(3)}(K_A^{(3)})^T & Q_A^{(3)}(K_P^{(3)})^T & Q_A^{(3)}(K_F^{(3)})^T\\
Q_P^{(3)}(K_I^{(3)})^T & Q_P^{(3)}(K_A^{(3)})^T & Q_P^{(4)}(K_P^{(4)})^T & Q_P^{(4)}(K_F^{(4)})^T\\
Q_F^{(3)}(K_I^{(3)})^T & Q_F^{(3)}(K_A^{(3)})^T & Q_F^{(4)}(K_P^{(4)})^T & Q_F^{(4)}(K_F^{(4)})^T
\end{bmatrix}.
$$

**每个格子只取一种分数，没有把三轴与四轴分数相加。**
可以把它理解为：先画一张全三轴分数矩阵，再把右下角几何×几何块替换为四轴分数。
这是数学示意；生产实现不会先计算完整三轴分数矩阵再覆盖它。

把 video/action 合成 O、PointFlow/FK 合成 G，矩阵就只有四块：

$$
S=\frac{1}{\sqrt{128}}
\begin{bmatrix}
Q_O^{(3)}(K_O^{(3)})^T & Q_O^{(3)}(K_G^{(3)})^T\\
Q_G^{(3)}(K_O^{(3)})^T & \boxed{Q_G^{(4)}(K_G^{(4)})^T}
\end{bmatrix}.
$$

这里能直接看到为何不能简单地为每个 token 选一套 128 维 Q/K 后就做普通 QKᵀ：
同一个几何 query 的左半行需要 Q3，右半行需要 Q4；几何 key 的上下两部分也有对应要求。
这不排除通过额外通道改写点积，但那是另一种实现。

## 3. 统一按行 softmax，再乘同一套 V

$$
W=\operatorname{softmax}_{\text{row}}(S+M),
\qquad
Y=WV=
\begin{bmatrix}W_{OO}&W_{OG}\\W_{GO}&W_{GG}\end{bmatrix}
\begin{bmatrix}V_O\\V_G\end{bmatrix}
=
\begin{bmatrix}W_{OO}V_O+W_{OG}V_G\\W_{GO}V_O+W_{GG}V_G\end{bmatrix}.
$$

`M` 在合法位置为 0，禁止位置为负无穷。每一行的 softmax 跨越这行所有合法 key，
**不能把左右两块分别 softmax 后直接相加**。
上式最后的加法是不同 key 的 value 加权求和，是普通 attention 的输出计算；
不是三轴/四轴分数相加。

当前 Flash2 路径分组计算，再用 LSE 恢复统一 softmax，等价于以上矩阵公式。
图里的 N×N 分数矩阵仅用于解释，生产代码不把它完整存进显存。

代码对应：

- `cosmos_framework/model/generator/pointflow_fk_attention.py::fused_partition_attention`：
  非几何 query 对全部 key 用三轴；几何 query 的非几何/几何 key 分别用三轴/四轴。
- `cosmos_framework/model/attention/flash2/two_group.py::_TwoGroupVarlen`：
  两组 LSE 合并，以及使用全局 output/LSE 的反向。
- [原始打分定义](./pointflow_fk_local_mrope_20261002.md#3-成对打分)。

## 4. 扩展到 160 维，用一次点积得到相同的分数矩阵

### 4.1 额外的 32 维从哪里来

当前每个 head 有 128 维。三轴变四轴只替换 8 对旋转通道，即 16 个特征维度，
其余 112 维不变。令被替换的维度集合为 D：

- `Q3` / `K3`：原来的完整 128 维三轴 Q/K。
- `Q3_D` / `K3_D`：从中提取 D 对应的 16 维。
- `Q4_D` / `K4_D`：同一组 D 维度改用四轴旋转后的值，仍然是 16 维。

这里是抽取通道和拼接，不增加新的 QKV 学习参数，也不用 MLP。

### 4.2 扩展后的 Q/K 矩阵

O 表示 video/action，G 表示 PointFlow/FK。每行是一个 token 对应的一行向量；
下图按模态分块，多行 token 使用相同规则。

```text
Q160                  128 维         16 维        16 维
                 ┌────────────────┬────────────┬────────────┐
video/action     │     Q3_O       │     0      │     0      │
PointFlow/FK     │     Q3_G       │   Q4_G,D   │   Q3_G,D   │
                 └────────────────┴────────────┴────────────┘

K160                  128 维         16 维        16 维
                 ┌────────────────┬────────────┬────────────┐
video/action     │     K3_O       │     0      │     0      │
PointFlow/FK     │     K3_G       │   K4_G,D   │  -K3_G,D   │
                 └────────────────┴────────────┴────────────┘
                                                      ↑
                                      这一块取负号，抵消原三轴贡献
```

矩阵写法为：

$$
\widetilde Q=
\begin{bmatrix}
Q_O^{(3)} & 0 & 0\\
Q_G^{(3)} & Q_{G,D}^{(4)} & Q_{G,D}^{(3)}
\end{bmatrix},
\qquad
\widetilde K=
\begin{bmatrix}
K_O^{(3)} & 0 & 0\\
K_G^{(3)} & K_{G,D}^{(4)} & -K_{G,D}^{(3)}
\end{bmatrix}.
$$

两者的形状均为 N×160。token 数 N 没有增加。

### 4.3 一次相乘后为什么仍然只有右下角变成四轴

只要 query 或 key 有一端属于 O，额外两块的乘积就是 0。因此 O→O、O→G、G→O
仍然完全使用原三轴分数。

两端都属于 G 时，点积为：

```text
完整的三轴点积（128 维）
+ 被替换通道的四轴点积（16 维）
- 被替换通道的三轴点积（16 维）
= 完整的四轴点积
```

因为其余 112 维在三轴/四轴之间完全相同，故有：

$$
Q_G^{(3)}(K_G^{(3)})^T
+Q_{G,D}^{(4)}(K_{G,D}^{(4)})^T
-Q_{G,D}^{(3)}(K_{G,D}^{(3)})^T
=Q_G^{(4)}(K_G^{(4)})^T.
$$

因此：

$$
\frac{\widetilde Q\widetilde K^T}{\sqrt{128}}
=\frac1{\sqrt{128}}
\begin{bmatrix}
Q_O^{(3)}(K_O^{(3)})^T & Q_O^{(3)}(K_G^{(3)})^T\\
Q_G^{(3)}(K_O^{(3)})^T & Q_G^{(4)}(K_G^{(4)})^T
\end{bmatrix}=S.
$$

这是点积的代数等价变换，随后统一 softmax。负号只用于抵消分数里的旧贡献，
不是 softmax 后使用负的 attention 权重。

### 4.4 V、缩放和 Flash2 调用

当前安装的 Flash2 varlen 接口采用相同的 Q/K/V head_dim，因此 V 末尾补 32 个零：

$$
\widetilde V=[V\mid 0_{N\times32}],
\qquad
\widetilde Y=\operatorname{softmax}_{\mathrm{row}}
\left(\frac{\widetilde Q\widetilde K^T}{\sqrt{128}}+M\right)\widetilde V
=[Y\mid0].
$$

输出取前 128 维即可继续走原来的输出投影。
必须显式设置 `softmax_scale=128**-0.5`，不能用扩维后的默认 `160**-0.5`。

这样生成分支可由一次 Flash2 varlen 完成，免去局部四轴造成的额外 attention 分组和 LSE 合并。
文字自身的因果分支仍保留独立调用；生成 query 对文字 key 的 normalization 也保持原样。

实现入口：`cosmos_framework/model/generator/pointflow_fk_attention.py::lifted_partition_attention`。
设置 `POINTFLOW_FK_LOCAL_ROPE=true POINTFLOW_FK_ATTN_IMPL=lifted160` 启用；
`POINTFLOW_FK_ATTN_IMPL=partition` 保留分组实现，仍是默认值。

代数等价不保证 BF16 逐位一致。A800 四类用例的输出最大绝对误差不超过 0.00741，
梯度不超过 0.02420（相对于 float64 dense oracle），checkpoint + fullgraph 动态 compile 检查通过。
记录：`pointflow_outputs/fk_lifted160_compile_20261002.json`。
点积维度增加 25%，实际内核可能按更大的维度档位执行，V 的计算也会扩展；
能否用减少搬运抵消这些开销，必须实测，详见
[局部四轴实现与测速记录](./pointflow_fk_local_mrope_20261002.md)。
