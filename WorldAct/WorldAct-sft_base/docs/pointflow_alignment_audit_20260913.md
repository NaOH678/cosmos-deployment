# PointFlow 对齐检查报告(位置编码 / 跨模态桥接)

**日期**:2026-09-13
**范围**:2D UV 对齐、3D 信息、跨模态位置编码、动点数变化
**方法**:逐行读代码 + 与 `pointflow_position_encoding.md` 的设计逐条对照 + **实测位置三元组**
**代码归属**:标注 `[本仓库]` 的是本仓库自有代码;其余 `cosmos_framework/**` 来自 NVIDIA 同步
(`e723d67 Sync NVIDIA cosmos-framework main at 5e67049`)。
**与其它文档的关系**:
- `pointflow_position_encoding.md` —— **设计意图**。本文是它的**实现核查**。
- `pointflow_window_latent_cache_20260913.md` —— 同一天的另一件事(数据侧缓存),**与本文无关**。
- `pointflow_motion_selection_20260914.md` —— 次日的**运动选点**(按 GT 位移取 top-X% 点)。它改变的是**送进模型的数据规模**,不改变本文讨论的位置编码。
- **`pointflow_fit005_analysis_20260914.md`** —— 次日的训练分析。它给本文的**发现 1** 添了新的分量:
  那里记录的 action↔point 虚假空间相位,当时定性为"设计未实现、危害未证实";如果先修好
  采样器的 `shift` 之后轨迹**仍不跟随**,它就升级为首要嫌疑 —— 因为 action 正是告诉点
  "手臂要往哪动"的信号源。
- `pointflow_bugfix_log_20260912.md` —— 09-11~09-12 的缺陷修复,时间更早,不重叠。

---

## 1. 结论摘要

| # | 检查项 | 结论 |
|---|---|---|
| 1 | 2D UV 与视频对齐 | ✅ 正确 |
| 2 | 3D 信息进入 token 内容 | ✅ 正确 |
| 3 | 时间轴对齐三种模态 | ✅ 正确,且有运行时断言 |
| 4 | **action ↔ point:只做时间旋转** | ❌ **未实现 —— 实际跑 legacy mRoPE**(危害未证实,见 §4) |
| 5 | 动点数变化(输入/输出) | ✅ 设计正确 |
| 6 | 少量点落在视频网格外 | ⚠️ 约 3%,且护栏阈值偏松 |
| 7 | 点的组织 / 输入格式 / 预测路径 | ✅ 正确,详见 **§6**(含真实张量清单与数字) |

> ### ⚠️ 后记(2026-09-14 晚)
>
> 本篇核查的**位置编码本身没有问题** —— 第 4 项那条"设计未实现"是事实,但**危害未证实**,
> 而且它**不是**训练训不动的根因。
>
> 真正的根因是 **PointFlow 的 clean 样本没有归一化**(`pointflow_displacement_scale = 1.0`,
> 位移 std 0.113 m vs 噪声 std 1.0 m),导致 flow-matching loss 90% 在度量噪声而不是位移。
> 见 [`pointflow_displacement_scale_20260914.md`](pointflow_displacement_scale_20260914.md)。
>
> **本篇的 §5(UV/3D/时间/动点数/网格外)、§6(点的组织与预测路径)、§7(mRoPE 通道布局实测)**
> 都是独立的基础事实,继续有效,可以放心引用。
> **§8 的修法方案暂时搁置** —— 在分支能学会运动之前无法判断它值不值得做。

**阅读提示**:§2–§4 是"位置编码是否存在问题";**§6 是"这些点到底以什么形式进模型、最后怎么变成预测"** ——
两者独立,§6 是理解前者的必要背景。

§6 内部有**两条正交的轴**,看的时候别混:

| | 讲什么 | 在哪 |
|---|---|---|
| **空间 / 样本维** | "哪一行属于哪个样本、哪个体素" —— CSR 拼接 | §6.1 – §6.2 |
| **时间维** | "每个 token 落在哪个时刻" —— `1+8` 分片、block-major、mRoPE | §6.3、**§6.7** |

---

## 2. 实测:三方的位置三元组

全部由本仓库真实函数算出,非推算:

```
① video latent   grid_t=9, grid_h=22, grid_w=17
   共 3366 个 token   t = 0, 1.6, …, 12.8     h∈[0,21], w∈[0,16]     ← 真实网格

② action token   grid_t=33, grid_h=1, grid_w=1
   共 33 个 token     t = 0.4, 0.8, …, 13.2   h = 全 0, w = 全 0      ← 空间退化成单格

③ point token    anchor UV,9 个时间片(anchor + 8 blocks)
   t = 0, 1.6, …, 12.8(与 video latent 同格)
   (h,w) = 点簇投影,**9 个片共用同一份**
```

复现命令(不需要 GPU):

```python
from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens as ids
v, _ = ids(grid_t=9,  grid_h=22, grid_w=17, temporal_offset=0, fps=15.0, base_fps=24.0,
           temporal_compression_factor=4)
a, _ = ids(grid_t=33, grid_h=1,  grid_w=1,  temporal_offset=0, fps=15.0, base_fps=24.0,
           temporal_compression_factor=1, base_temporal_compression_factor=4, start_frame_offset=1)
```

### 两个容易看漏的细节

- **action 的时间整体偏移 +0.4 秒**:`t = 0.4·(j+1)`,因为 Cosmos 原生 `start_frame_offset=1`
  ("action[0] 对齐 vision frame 1")。设计文档说的"action 保留 Cosmos 原有 offset"就是这个。
- **point 的 9 个时间片共用 anchor UV**:`point_positions` 里 `wh.flip(-1)[None].expand(len(times),-1,-1)`,
  符合设计"同一簇的未来 tokens 都沿用当前 anchor UV"。

---

## 3. action ↔ point 的分数分解

在**当前实际运行**的 `legacy_mrope` 下:

```
s(a,p) = ⟨ R_T(t_p) q_P , R_T(t_a) k_A ⟩_T        ← ① 时间:相对量 (t_p − t_a)
       + ⟨ R_S(h̄_j,w̄_j) q_P , R_S(0,0) k_A ⟩_S   ← ② 空间:相对量 = (h̄_j − 0, w̄_j − 0)
       + ⟨ q_P^U , k_A^U ⟩_U                       ← ③ 未旋转通道的纯内容点积
```

②中 `R_S(0,0)` 是单位阵,于是**相对旋转退化成点簇的绝对图像坐标**。实算三个簇:

```
簇0  (h,w)=(11.98,  8.02)  →  相对旋转 (11.98,  8.02)
簇1  (h,w)=(20.52,  1.64)  →  相对旋转 (20.52,  1.64)
簇2  (h,w)=(17.68, 15.45)  →  相对旋转 (17.68, 15.45)
```

**簇0 与簇1 在空间相位上被拉开约 (8.5, 6.4)** —— 而它们与 action token 的时间关系、
三维几何关系可以完全相同。

### 谁拿到的是"有意义的相对量"

| 对子 | 相对旋转 | 有意义吗 |
|---|---|---|
| video ↔ point | `(h̄_j − h_v, w̄_j − w_v)` | ✅ 真实相对位移 —— UV 对齐要的就是这个 |
| point ↔ point | `(h̄_j − h̄_j', w̄_j − w̄_j')` | ✅ 两簇在图上相距多远 |
| **action ↔ action** | `(0, 0)` | ✅ **空间项自动退化成纯内容点积 —— 天然就是"时间-only"** |
| action ↔ video | `(h_v, w_v)` 绝对量 | 原生如此,已烧进预训练权重 |
| **action ↔ point** | `(h̄_j, w̄_j)` 绝对量 | 设计与 action↔action 一致;实现没有 |

> **关键观察**:`action↔action` 今天**已经是时间-only**了,因为两边都是 `(0,0)`,相对量恒为 0。
> 所以设计要的 action↔point 规则**不是新机制**,而是"把 action↔point 归到 action↔action 那一类"。

---

## 4. 发现一:设计 §4 的 action↔point 规则未实现

### 4.1 设计要什么

`pointflow_position_encoding.md` §4:

> | Action ↔ Point | **只做时间旋转**;空间通道保留原始内容点积 |
> 一个 action token 是整个关节向量,没有唯一像素位置。因此,action–point **不应**将动作的
> `(0,0)` 与 point 的像素坐标计算空间相位差。

### 4.2 实际跑什么

**`legacy_mrope`** —— 所有模态共用一套完整 mRoPE,action↔point 的空间通道**照转**。

### 4.3 证据链

```
pointflow_attention.py:28   reference_attention_enabled() 读 POINTFLOW_REFERENCE_ATTENTION,默认 "false"
       ↓
pointflow_attention.py:32   default_attention_mode() → "legacy_mrope"
       ↓
pointflow_sequence.py:70    attach_point_tokens(..., attention_mode=None) → 取默认值
       ↓
pointflow_branch.py:138     调用时不传 attention_mode
       ↓
全仓库 grep:POINTFLOW_REFERENCE_ATTENTION **只被读取,从未被设置**;启动脚本里也没有
```

后果:point token 的位置直接写进 `sequence.position_ids`,和 video/action 走**同一套**
`dispatch_attention_fn`。

`pairwise_point_attention`(实现该规则的参考版本)存在且正确,但**默认关闭**。

### 4.4 ⚠️ 重要纠正:原生 video↔action 用的是**同一机制**

**最初的判断("注入纯噪声 / 三重错误")说过头了。** 原生 Cosmos 的 video↔action 就是同一结构:

```
video:  (t, h_v, w_v)       action: (t, 0, 0)
空间项 = ⟨R_S(h_v,w_v) q_V , R_S(0,0) k_A⟩ = ⟨R_S(h_v,w_v) q_V , k_A⟩
```

**形式与 action↔point 完全一样,而且原生就这么训出来的、能工作。**所以不能称其为"错误"。

它在原生里不构成问题的原因:**位置对两种 token 的意义不同**。

| | video token | action token |
|---|---|---|
| 位置是它的**身份**吗 | **是**(网格固定,`(h,w)` 就是"我是哪个 patch") | 不是(占位符) |
| 实际效果 | action→video 注意力上的一个**按 patch 位置的固定偏置** —— 可学习、有信息量 | —— |

**所以那是冗余/先验,不是噪声。**

### 4.5 修正后的定性

| | 最初说法 | **修正后** |
|---|---|---|
| 性质 | "违背设计的**缺陷**" | **"设计未实现"** —— 这是事实,可复现 |
| 危害 | "注入纯噪声" | **未证实**。原生对 video↔action 用同一机制且能工作 |
| 严重性 | 高 | **未知,可能低于最初判断** |

**事实部分不变**:设计写的是 A,代码跑的是 B。**变的是定性** —— 这不是"抓到一个 bug",
而是"实现与设计文档不一致,而哪个更好没有证据"。

设计给 action↔point 换规则的可能理由(可信度递减):

1. **预训练约束** —— video↔action 的相位已烧进 base checkpoint,改不得;action↔point 是全新关系,
   规则是自由选择。这条最能解释那个不对称。
2. **身份来源不同** —— video token 的唯一身份是位置(旋转必需);point token 的身份是三维几何
   (在内容里),位置旋转至多是冗余。
3. **桥接语义** —— 设计希望 action↔point 的耦合只由「时间 + 内容」决定。

**结论:该不该改,应由实验决定,不是由推理决定。**

---

## 5. 其余检查项

### 5.1 ✅ 2D UV 对齐正确

```
tracker 像素 (640×448 的 uv_px)
  → uv_to_video 仿射                → 画布像素 (544×736)
  → / pixel_stride (=32) + 半像素修正  affine[:,:,2] += 0.5/s − 0.5
  → patch 连续坐标 [0,17)×[0,23)     ← 与视频 token 索引同一坐标系
```

- 半像素修正 `(v+0.5)/s − 0.5` 正是"像素中心 ↔ 格子索引"的标准换算,**与视频 token 的整数索引同约定**
  ⇒ `R_S(h̄_j − h_v)` 的差值有意义 ✅
- 仿射与 `validate_pointflow_chain.py` L6 验过的 composed affine 一致
  (`x'=0.85u`, `y'=0.911096v+307.859`) ✅
- 同一簇的所有未来 token 共用 anchor UV ✅;未来 GT UV **没有**进入位置编码 ✅
- 运行时护栏 `check_points_inside_video_grid`(阈值 75%)

### 5.2 ✅ 3D 信息在内容里,位置在 Q/K 里

```
内容: g = W_F·PTv3特征 + MLP_xyz(cluster_xyz / xyz_scale)          ← G_j
      pooled = 簇内均值(MLP_motion(4 个带噪位移))                   ← E_b,j
      token  = point2llm(LN([g; pooled])) + e_P + TimeEmbed(σ)
位置: (t_b, h̄_j, w̄_j)
```

逐点解码的输入签名与设计 §5.3 一致:
`D(H^P[block, original_to_cluster], voxel_features, relative_xyz/uv, blocks, sigma_features)`

### 5.3 ✅ 时间对齐正确,且有运行时断言

```python
times = cat((zeros(1), seconds)) * (24.0/4.0)     # ×6,即 t = t_start + 6Δt
seconds = arange(1, blocks+1) * 4 / 15            # 1.6 … 12.8
# 且强制校验视频 latent 时间轴必须等于该序列:
target = times[0] + arange(expected) * steps_per_token / fps * 6
if not torch.allclose(times, target, atol=1e-4): raise ValueError(...)
```

σ 用的是**视频那个 σ**(`pointflow_add_noise`:"one shared video sigma per sample,
not diffusion forcing"),符合 WAM 联合去噪 ✅

### 5.4 ✅ 动点数变化的处理

**输入侧 —— 先聚类,再按簇出 token**

- 每样本 `(1 + 8)·K_b` 个 token,**K 逐样本不同 → 完全不 padding**("No padding of K"),靠
  sequence packing 的 per-sample `split_lens` 管边界
- 簇成员**只由当前帧决定,整个预测窗口不变**(`original_to_cluster` 在 8 个 block 里是同一份)
- 未来帧无效点 → `valid` mask,**不改变 token 数量**(有测试 `test_future_mask_does_not_change_topology_or_budget` 锁定)

**输出侧 —— 解码回原始点**

```python
hidden = point_hidden[block, original_to_cluster]   # 簇级 → 广播到每个原始点
decoder(cat((hidden, local, relative, blocks[block], sigma_features)))
```

输出形状 = 该样本的原始点数 `N_b`,也是逐样本不同;`restore_motion` → `[32, N_b, 3]`。

**损失** = masked MSE,只在 valid 点上算,按样本求均值。

> **两种"点数变化"处理方式不同:**
> - **样本间 K/N 不同** → ragged 序列,不 padding
> - **样本内逐帧有效点不同** → mask,拓扑不变
>
> **完整展开见 §6** —— 那里有真实的张量清单(8192 点 → 3908 体素 → K 簇)、
> CSR 风格索引的具体形式、`1+8` 与视频 latent 的对齐、anchor token 的角色,
> 以及簇级 hidden 如何回到逐点 velocity 的完整链路。

### 5.5 ⚠️ 约 3% 的点落在视频网格外

画布高 736 → `736/32 = 23` 行 patch,但视频 token 网格只有 **22** 行
(`_remove_padding_from_latent` 取整,而内容实际跨越 44.75)。落在那一带的点,其 mRoPE `h`
是**没有视频 token 占据的相位**。边界效应、有界。

**但护栏阈值偏松**:设在 0.75,正常 0.969、异常 0.158。若退化到 80% **不会报警**。
**建议提到 0.90。**

---

## 6. 点的组织、输入与预测路径

> 本节回答三个问题:点**怎么组织**、**怎么进模型**、最后**怎么预测**。
> 全部张量形状与数字来自实测(`episode_0013_20260731_133649` 的一个窗口)。

### 6.1 三层结构:点 → 体素 → 簇

```
8192 个原始点  ──体素化(voxel 0.02 m)──►  3908 个体素  ──PTv3 三次 pooling──►  K 个簇
```

实测单样本:

| 层 | 张量 | 形状 |
|---|---|---|
| 点 | `anchor_xyz` | `[8192, 3]` float32 |
| | `anchor_uv` | `[8192, 2]` float32 |
| | `point_ids` | `[8192]` int64 |
| 体素 | `coord` | `[3908, 3]` float32 |
| | `feat` | `[3908, 9]` float32 |
| | `grid_coord` | `[3908, 3]` int64 |
| 簇 | `cluster_features` | `[K, 256]`(PTv3 stage-3) |

`K` 由 PTv3 的 stage-3 输出决定,**逐样本不同**。
`N` 有**上界** `max_points = 8192`(实际还会被法向估计再筛掉少量退化点,见 §6.8);
实测 4/4 个 episode 都取到 8192。而 `V` 逐样本不同 —— 见 §6.2。

### 6.2 张量格式:全是矩阵 + 索引(CSR 风格)

**先纠正一个常见直觉** —— "点云不规则所以点这一层也不规则",**对本数据不成立**:

```python
# pointflow_window.py:113-133
xyz = read_frame("position.npy", rows[0]).reshape(-1,3)   # ← 三个数组都读锚定帧
uv  = read_frame("uv_px.npy",    rows[0]).reshape(-1,2)
valid = read_frame("valid.npy",  rows[0]).reshape(-1)
good = valid & isfinite(xyz) & (xyz[:,2] > 0) & isfinite(uv) & uv 在图内
candidates = np.flatnonzero(good)                          # 候选池
rng = np.random.default_rng(seed)
ids = np.sort(rng.choice(candidates, min(max_points, len(candidates)), replace=False))
# 再按法向估计筛掉邻域退化的点:
reliable = eigenvalues[:, 1] > 1e-12
ids = ids[reliable]                                        # ← 这里还会掉点
```

⇒ **`N = min(max_points, |candidates|) − 法向估计丢弃的退化点`**

所以 **`8192` 是上界,不是保证值**。实测 4/4 个 episode 都是 8192(那些数据上一点没丢),
但严格说这一层**并非严格的常数** —— padding 的"零浪费"结论因此要打个折扣(见本节末)。

选点的完整细节(seed 怎么来、一个 chunk 内怎么追踪同一批点)见 **§6.8**。
(顺带:`np.sort` 意味着点的顺序是**排序后的确定子采样**,不是随机排列 —— 见 §6.6 的"点序只有一个来源"。)

**真正逐样本变化的是体素层和簇层:**

```
episode_0013  V = 3908      episode_0014  V = 3725
episode_0015  V = 3748      episode_0017  V = 3711      ← V 逐样本不同
```

PTv3 三次 pooling 之后的 `K` 同样逐样本不同。所以**不能直接 stack 的是 V 和 K**;
本实现把**三层统一**成"扁平拼接 + 索引张量"。

#### 张量清单(四类)

**① 逐点张量 `[sum(N), …]`** —— 所有样本的点首尾相接

```
anchor_xyz   [sum(N), 3]    anchor_uv  [sum(N), 2]    normal  [sum(N), 3]
color        [sum(N), 3]    point_ids  [sum(N)]
```

**② 逐体素张量 `[sum(V), …]`**

```
coord  [sum(V), 3]     feat  [sum(V), 9]     grid_coord  [sum(V), 3]
```

**③ 索引张量 —— 这才是"哪行属于谁"**

```
point_offsets        [B]       每个样本点段的**终点**           ← 见下方图示
voxel_offsets        [B]       每个样本体素段的**终点**
point_batch          [sum(N)]  每一行的样本号
voxel_batch          [sum(V)]  每一行的样本号
original_to_voxel    [sum(N)]  点 → 体素(已加各样本的体素起点偏移)
original_to_cluster  [sum(N)]  点 → 簇(编码后才有)
```

> ⚠️ **`point_offsets` 长度是 `B`,不是 `B+1`** —— 它**只存终点**。起点由
> `PointFlowBatch.point_spans` 补出:`torch.cat((zeros(1), ends[:-1]))`。
> 这一点与教科书上的标准 CSR(长度 `B+1`、首元素为 0)**不同**,容易看错。

**④ 逐样本标量 `[B, …]`** —— 这些直接就是规整矩阵

```
uv_to_video [B,2,3]    image_size_wh [B,2]    has_geometry [B] bool
```

#### 具体例子(用真实代码构造 3 个样本跑出来)

```
   样本0        样本1        样本2
   4 个点       3 个点       5 个点      →  sum(N) = 12
   3 个体素     2 个体素     4 个体素     →  sum(V) =  9
```

**A. 逐点张量:首尾相接**

```
行号      0   1   2   3 │ 4   5   6 │ 7   8   9  10  11
          └── 样本0 ───┘ └─ 样本1 ─┘ └──── 样本2 ────┘

anchor_xyz     [12, 3]                  ← 12 = sum(N)
point_batch    [ 0, 0, 0, 0,  1, 1, 1,  2, 2, 2, 2, 2]     "这行属于哪个样本"
```

**B. 行指针:只存终点**

```
point_offsets = [4,  7,  12]        ← 长度 B = 3,**每个样本的终点**
                  │   │   │
                  │   │   └── 样本2 结束于 12
                  │   └────── 样本1 结束于 7
                  └────────── 样本0 结束于 4

起点 = 前一个终点(第一个是 0)   →   point_spans
                                      ┌───────┬───────┬────────┐
                                      │ [0,4) │ [4,7) │ [7,12) │
                                      └───────┴───────┴────────┘
```

**C. 跨层映射:点 → 体素**

样本**内部**是各自编号的,拼接时要**加各自体素段的起点偏移**(`pointflow_batch.py:168-177`):

```
样本内局部映射           加偏移                全局
样本0  [0,1,1,2]     +0(体素起 0)  →  [0,1,1,2]
样本1  [0,0,1]       +3(体素起 3)  →  [3,3,4]
样本2  [0,1,2,2,3]   +5(体素起 5)  →  [5,6,7,7,8]
                                        └─── original_to_voxel  [12]

体素起点 = np.r_[0, cumsum(voxels)][:-1] = [0, 3, 5]
```

对到图上:

```
   点行号     0  1  2  3 │ 4  5  6 │ 7  8  9 10 11
              │  │  │  │   │  │  │   │  │  │  │  │
              ▼  ▼  ▼  ▼   ▼  ▼  ▼   ▼  ▼  ▼  ▼  ▼
体素行号      0  1  1  2   3  3  4   5  6  7  7  8
              └─ 样本0 ─┘  └ 样本1┘  └─── 样本2 ───┘
   体素行     0  1  2 │ 3  4 │ 5  6  7  8
```

反向也有一个:`voxel_representatives` —— 每个体素指回它的一个代表点:

```
voxel_representatives = [0, 1, 3,  5, 6,  7, 9, 10, 11]
                          ↑              ↑
                    体素0 的代表点 = 点0
```

**D. 这些索引怎么被用 —— 为什么值得这么存**

```python
# pointflow_geometry.py:89   一行还原每样本规模
counts = torch.diff(inputs["voxel_offsets"], prepend=zeros(1))
#        diff([0,3,5,9]) = [3, 2, 4]                     ✓

# pointflow_batch.py:181     构造 batch 索引(自己就是它)
inputs["point_batch"] = torch.repeat_interleave(torch.arange(B), sizes)

# pointflow_geometry.py:93   PTv3 按样本切分
data["offset"] = counts[active].cumsum(0)
```

#### 等价形式:如果写成 padding 会是什么样

同样的数据,用 `[B, N_max, …]` + mask 表示:

```
              j=0    j=1    j=2    j=3    j=4
          ┌──────┬──────┬──────┬──────┬──────┐
 样本0(4) │ ●    │ ●    │ ●    │ ●    │ ░░░░ │
          ├──────┼──────┼──────┼──────┼──────┤
 样本1(3) │ ●    │ ●    │ ●    │ ░░░░ │ ░░░░ │
          ├──────┼──────┼──────┼──────┼──────┤
 样本2(5) │ ●    │ ●    │ ●    │ ●    │ ●    │
          └──────┴──────┴──────┴──────┴──────┘
   mask   ┌──────┬──────┬──────┬──────┬──────┐
 样本0    │  1   │  1   │  1   │  1   │  0   │      ░ = padding
 样本1    │  1   │  1   │  1   │  0   │  0   │
 样本2    │  1   │  1   │  1   │  1   │  1   │
          └──────┴──────┴──────┴──────┴──────┘
```

**两种表示承载的信息完全相同**,只是"不规则性"放在不同的地方:

| | padding 形式 | **CSR 形式(实际用的)** |
|---|---|---|
| 形状 | `[B, N_max, 3]` + mask | `[sum(N), 3]` + offsets/batch |
| 元素数(本例) | `3×5 = 15` | `12` |
| 不规则性体现在 | **形状里**(靠 mask 抹掉) | **索引张量里** |
| 是否必须额外掩码 | **必须** —— 否则 padding token 会被 attend 到 | **不需要** —— 它们根本不存在 |
| 谁能直接吃 | 普通算子 | **varlen kernel**(`cu_seqlens` 就是 `*_offsets`) |

> 📌 **本节图示的范围**:上面几张图只画了**点/体素两层的"行归属"结构** ——
> 也就是"哪一行属于哪个样本、哪个体素"。**它们完全不含时间轴。**
> 时间是怎么组织的,见 **§6.7**。

#### 为什么最终选 CSR

1. **V 和 K 真的逐样本不同** → 这两层**必须** ragged;三层用同一套约定比混着来简单
2. **不需要 mask** → padding 形式必须额外屏蔽 padding token 的注意力,ragged 没有这个问题
3. **varlen flash-attention 原生消费这个形式** → `cu_seqlens` 就是 `*_offsets`,零转换成本
4. **attention 是对集合求和**,不关心点与点的相邻关系 ⇒ 首尾相接**不引入虚假邻接**

> ⚠️ **一个原来写错的理由**:本报告早期版本说"`N_max ≫ 平均`,padding 会浪费大量算力"。
> **实测不成立** —— N 通常取满 `max_points = 8192`(但见 §6.8:它并非严格常数),
> 点这一层 padding 基本不浪费。真正成立的是上面 1–4 条(尤其第 1 条:V/K 层的必要性)。

### 6.3 token 怎么进序列:`1 + 8`

每个簇产出 **9 个 token**:`1` 个 anchor + `8` 个未来时间片,追加到该样本 **full split** 末尾。

**为什么是 1+8 —— 与视频 latent 的 9 对齐,而且有运行时断言:**

```python
# pointflow_branch.py [本仓库]
expected = timing.steps // timing.steps_per_token + 1        # 32 // 4 + 1 = 9
times = torch.unique(pos[0], sorted=True)                    # 视频 latent 的 mRoPE 时间
if len(times) != expected:
    raise ValueError("Video latent timeline does not match PointFlow blocks")

target = times[0] + torch.arange(expected) * timing.steps_per_token / timing.fps * 6
if not torch.allclose(times.float(), target.float(), atol=1e-4, rtol=0):
    raise ValueError("Video mRoPE must use Cosmos physical FPS modulation")
```

两句都是**拒绝启动**级检查:**数量必须正好 9**,**数值必须是 `0, 1.6, …, 12.8`**。

三个模态的 token 数本来就是同一件事的三种粒度:

| 模态 | token 数 | 条件 | 生成 | 时间压缩 |
|---|---|---|---|---|
| video latent | **9** | 1(latent 0) | **8**(latent 1..8) | 4 |
| action | 33 | 1(state) | 32(控制步) | 1 |
| **point** | **1 + 8 = 9** | 1(anchor) | **8**(blocks) | **4** |

```
32 个 action 步 ÷ 4(VAE 时间压缩) = 8 个 latent 间隔 = 8 个 point block
```

⇒ **point 的时间轴与 video 是同一根轴**,跨模态时间旋转 `R_T(t_p − t_v)` 才有意义。

### 6.4 anchor token 是 point 分支的**条件 token**

与 video latent 0 / action token 0 同角色。六条证据:

| 检验 | 结果 |
|---|---|
| 对 σ 的依赖 | 改 σ 后 `anchor_tokens` 变化 = **`0.000000`**(实测) |
| | 同期 `noisy_tokens` 变化 = `0.462` |
| 是否被加噪 | 从不进入 `clean.displacement` |
| 是否 loss 目标 | 否 —— loss 只在 `[32, sum(N), 3]` 上算 |
| hidden 是否用于解码 | 否 —— `hidden() = sequence[noisy_indexes]`,`noisy_indices` 从 `idx[count:]` 起跳过 anchor |
| 是否有位置 | 有,`(t_start, h̄_j, w̄_j)` |

**但它与 video/action 的条件有三处重要不同:**

**① 不是唯一条件通路,且"条件宽度"完全不同。** video 只条件于 latent 0,其余 8 个是纯生成目标;
point 则**每个簇的每个 noisy token 都带着当前几何**:

```
noisy_tokens 内容 = Point2LLM(LN([ G_j ; E_{b,j} ])) + e_P + TimeEmbed(σ)
                                  └─ 当前几何:PTv3 特征 + XYZ 编码
```

anchor token 是**额外的**一个"该簇当前状态"token,供其他 token 注意 —— noisy token 忙着表达未来运动 + σ,
不适合兼任。

**② σ 的施加机制不同。**

| | video / action 的条件 | point 的 anchor |
|---|---|---|
| 怎么保证干净 | **同一 token**,σ 被 `×(1−condition_mask)` 强制为 0 | **不同 token**,不同投影,不加 σ |
| σ=0 的 TimeEmbed | **有** TimeEmbed(0) | **无** |
| 排除出 loss | `mse_loss_indexes` | 不在 `clean.displacement` 作用域内 |

⇒ **不是"同一 token 在 σ=0",而是另一种 token 类型。**

**③ 输出被算出但丢弃。** anchor 在序列里,MoT 会算它的 hidden,但 `hidden()` 跳过它。
它纯粹当 **K/V**;梯度仍通(其他 token 的输出依赖它)。

**时间上:对齐的是 video latent 0,不是 action token 0**

```
video latent 0 :  t = 0
point anchor   :  t = 0        ← 对齐 ✓
action token 0 :  t = 0.4      ← 原生 start_frame_offset=1
```

设计 §3 明确写"Anchor | 当前状态 | 0 秒 | 0",action 那条说"保留 Cosmos 原有 offset"。**有意为之。**

### 6.5 预测路径:簇级 hidden → 逐点 velocity

```
MoT 输出所有 token 的 hidden
   │
   ├─ PointTokenPayload.hidden(sequence) = sequence[noisy_indexes]     →  [8, K, D]
   │                                                                    (block-major)
   ▼
decode:  输出 (block b, 点 i) 的每一项输入都索引到 i 自己
   hidden[b, i]    ← point_hidden[b][ original_to_cluster[i] ]   该点所属的簇
   blocks[b][i]    ← 该点自己的带噪位移
   local[i]        ← voxel_features[ original_to_voxel[i] ]      该点自己的 PTv3 特征
   relative[i]     ← 该点相对簇中心的 Δxyz / Δuv
   sigma           ← sigma[ point_batch[i] ]                     该点所属样本
   │
   ▼  MLP(每 block 一次)
[N_b, 12]  per block
   │
   ▼  restore_motion:block b → step [4b, 4b+3]
[32, N_b, 3]        ← 输出点数 = 原始点数 N_b,不是簇数
```

**输出规模回到原始点**,而 hidden 是簇级的 —— `original_to_cluster` 只把簇级信息**广播**到点,
不会把点 A 的位移配给点 B。

### 6.6 loss 对位:已验

**① 逐元素对齐(靠同一个掩码)**

```python
error = (prediction.float()[valid] - state.velocity_target.float()[valid]).square().mean(-1)
```

同一个布尔掩码同时作用在预测和目标上 ⇒ 逐元素对应,不可能串位。

**② 时间维度对齐(实测)**

```
block b 覆盖 step [4b, 4b+3] (0-based)  =  设计 I_b = {4b+1 … 4b+4} (1-based)   ✓
block b 的 mRoPE 时间 = 该片右端点      (block0 → 1.6 单位 = 4/15 s)             ✓
restore_motion(motion_blocks(x)) == x   逐位相同 ✓(已有测试 pointflow_codec_test.py:49)
```

**③ 护栏**

```python
displacement.shape == (steps, n, 3) 且 n 与 anchor 点数一致
prediction.shape != clean.displacement.shape → raise
decode 校验 point_hidden.shape == (blocks, K, D)
```

**残余风险(要说清)**:以上都是**形状**检查 —— 一次**全局一致的置换**能通过。

但端到端**只有一处产生点的顺序**,不存在两个独立来源需要对齐:

```python
# pointflow_window.py:123 —— 点序的唯一来源,而且是确定的
ids = np.sort(rng.choice(candidates, min(max_points, len(candidates)), replace=False))
```

`np.sort` 意味着这个顺序**不是随机排列**,而是"源点云下标的一个递增子集";
给定 seed 完全可复现。此后全链路(`original_to_cluster` / `valid` / `displacement`)
都按同一顺序索引,所以没有"预测点序"与"目标点序"不一致的可能。

### 6.7 时间维度是怎么组织的

> §6.2 的图只画了"行归属";**本节只画时间**。两者是正交的两个轴。

#### ① 目标张量:时间轴在最前面

```
displacement  [ 32 , sum(N), 3 ]
                ↑     └───┬───┘
                │     §6.2 画的 CSR 结构
                └── 32 个未来控制步(15 Hz,2.133 s)

valid         [ 32 , sum(N) ]          ← 掩码也是逐 (step, 点)
```

#### ② codec 之后:多出一层"时间片"

```
时间片:    0        1        2       …      8
         ┌────────┬────────┬────────┬─────┬────────┐
         │ anchor │ block1 │ block2 │  …  │ block8 │
         │  K 个  │  K 个  │  K 个  │     │  K 个  │
         └────────┴────────┴────────┴─────┴────────┘
           t = 0    1.6      3.2            12.8      (mRoPE 单位)

张量:  anchor_tokens  [K, D]        ← 只有一片(条件)
       noisy_tokens   [8, K, D]      ← block 在前,簇在后
```

#### ③ 为什么是 8 片 —— **片大小不是自选的,是 VAE 的时间压缩率**

```python
# pointflow_window.py:26  PointFlowTiming.from_cosmos
timing = cls(
    fps=float(dataset_config["fps"]),                                  # 15
    steps=dataset_config["chunk_length"],                              # 32
    steps_per_token=tokenizer_config["temporal_compression_factor"],   # ← 4,来自 VAE
)
durations = tokenizer_config.get("encode_exact_durations")
if durations is not None and timing.steps + 1 not in durations:
    raise ValueError("PointFlow states do not match Cosmos encode_exact_durations")
```

```
video latent = 4 帧   ← VAE 的 temporal_compression_factor = 4
point 片     = 4 步   ← 同一个 4
                       ⇒ 两者都是"**一个 VAE 时间单元**"
```

**"片分割"不是额外发明的一层**,而是把 VAE 对帧做的那件事原样套到 action 步上。
如果硬做 32 个 token(即 1+32),就是点分支自己发明了一个比 VAE 更细的时间粒度,代价:

**① 失去 token 级 1:1 对应** —— `point 9 ↔ video 9` 变成 `point 33 ↔ video 9`,只能靠模型隐式学。

**② mRoPE 的 T 轴不再干净**

```
现在:  video latent k → t = 1.6k ,  point 片 b → t = 1.6b
       匹配对 (k=b) 相对时间 = 0  → T 通道旋转 = 单位阵
       不匹配对        = 1.6(k−b) → 相位正比于"隔了几个片"

1+32:  point token s → t = 0.4s
       匹配对 (s=4k) 相对时间 ≠ 0,且相邻 4 个 token 挤在同一个 latent 时刻附近
       → "同时"从"相位为 0"退化成"相位很小",信号变糊
```

**③ token 数 ×4** —— `(1+8)·K = 9K` → `(1+32)·K = 33K`;`K=64` 时是 576 → 2112,
而视频是 3366 —— point 占比从 **17%** 涨到 **63%**,attention 开销是实打实的。

**而且"片"在内容/位置/输出三处都是同一个单位**(设计 §5.1):

```
内容:  E_{b,j} = 簇内均值( MLP_motion( Concat_{k∈I_b} d_{σ,k,i} ) )   ← 4 步 × 3 维 = 12 维**整体**编码
位置:  p^P_{b,j} = (t_start + 1.6b, h̄_j, w̄_j)
输出:  decoder 每片出 12 维 = 4 步 × 3 维
```

⇒ 不是"把 4 个独立 token 的位置平均一下",而是"这 4 步的运动被当成**一个整体**编码"。
这与 video latent k 代表 `[4k-3, 4k]` 这一**整段**是同一个语义:**代表一个区间的状态,不是某一帧**。

> 另一条没走的路:`1 + 32 = 33`,与 action token 同粒度。数字上整齐,但那样 point 就和 video
> 失去 token 级对应,而设计要的正是 **video↔point 的 UV + 时间双对齐** —— 那是"桥"的一半。

#### ④ 进序列时按 **block-major** 展开

```python
# pointflow_sequence.py:95-98
parts = [noisy[:, cstart:cend].reshape(-1, D)]   # [8,K,D] → [8K, D],block-major
parts.insert(0, anchor[cstart:cend])             # anchor 在最前
content = torch.cat(parts)
```

```
一个样本在 packed sequence 里的点段:

  ┌────────┬────────┬────────┬─────┬────────┐
  │ anchor │ block1 │ block2 │  …  │ block8 │     共 (1+8)·K 个 token
  │ K 个   │ K 个   │ K 个   │     │ K 个   │
  └────────┴────────┴────────┴─────┴────────┘

对应  noisy_indexes  [8, K]   ← 只记 noisy 那 8 片的全局下标
      hidden() = sequence[noisy_indexes]      →  [8, K, D]
```

#### ⑤ mRoPE 时间轴:与 video latent **同格**

```
t (mRoPE 单位)   0      1.6     3.2     4.8    …    12.8
                 ├───────┼───────┼───────┼─────┬──────┤
point            anchor  block1  block2  block3 …  block8
video latent     vl0     vl1     vl2     vl3    …  vl8      ← 9 个
action token     0.4     0.8     1.2    …            13.2   ← 33 个,原生偏移 +0.4

每个 block 覆盖 4 个 action 步:
   block b  →  step [4b, 4b+3]  (0-based)  =  设计 I_b = {4b+1 … 4b+4} (1-based)
```

**这张图就是 §6.3 那句运行时断言在管的东西** —— 视频 latent 的时间轴必须**正好 9 个**且**逐点等于**这个序列。

**实测对位表(9/9 全部一致):**

| 片 | 覆盖的步 | 右端点(秒) | mRoPE 单位 | video latent 时间 | 一致 |
|---|---|---|---|---|---|
| anchor | 当前 | 0.0000 | 0.0 | 0.0000 s | ✓ |
| 1 | 1–4 | 0.2667 | 1.6 | 0.2667 s | ✓ |
| 2 | 5–8 | 0.5333 | 3.2 | 0.5333 s | ✓ |
| 3 | 9–12 | 0.8000 | 4.8 | 0.8000 s | ✓ |
| 4 | 13–16 | 1.0667 | 6.4 | 1.0667 s | ✓ |
| 5 | 17–20 | 1.3333 | 8.0 | 1.3333 s | ✓ |
| 6 | 21–24 | 1.6000 | 9.6 | 1.6000 s | ✓ |
| 7 | 25–28 | 1.8667 | 11.2 | 1.8667 s | ✓ |
| 8 | 29–32 | 2.1333 | 12.8 | 2.1333 s | ✓ |

**但比"代表时刻相同"更强:覆盖的帧区间也完全相同。**

```
video latent k ← 15 Hz 帧 [4k-3, 4k]    (k≥1;latent 0 ← 帧 0 单独 prime)
point 片     b ← 步        [4b-3, 4b]    (1-based)
                 而  步 s  ≡  帧 s
                 ⇒ latent k 和 片 k 覆盖的是**同样那 4 帧**
```

由官方的 `get_pixel_num_frames(k) = (k-1)*4+1` 保证:

```
latent 0 需要 1 帧  → 帧 [0]
latent 1 需要 5 帧  → 帧 [0..4]  → 新增帧 [1..4]   ✓ 与片 1 同
latent 2 需要 9 帧  → 帧 [0..8]  → 新增帧 [5..8]   ✓ 与片 2 同
```

**所以不是"代表时刻碰巧一样",而是两种分割本来就是同一条时间栅格上的同一种切法。**

> 另一个结构性事实:**观测帧 1..32 就是预测目标步 1..32** —— 同一批帧。
> 模型学的是"把点云从帧 0 推到帧 1..32"的位移,而视频分支生成的那 32 帧与 point 分支监督的
> 那 32 步是同一个东西。

#### ⑥ 秒 ↔ mRoPE 单位:那两个数字是什么关系

上面那张表里每行都有**两列时间**,它们指同一时刻,只是单位不同:

```
0.26667 s   =   1.6 × (1/6 s)   =   1.6 个 mRoPE 单位
```

**换算公式**(`mrope.py:179`):

```python
tps      = fps / temporal_compression_factor        # 15/4 = 3.75
base_tps = base_fps / base_tcf                      # 24/4 = 6
scaled_t = frame_index / tps * base_tps             # = 真实秒数 × 6
```

**那个 6 是什么**:

```
base_fps = 24      ← 预训练时视频的帧率(edge_model_config: diffusion_expert_config.base_fps)
base_tcf = 4       ← 时间压缩
⇒ 1 秒 = 24/4 = 6 个 latent 帧
⇒ 1 个 mRoPE 时间单位 = 1/6 秒 ≈ 0.1667 s = "base 视频的一帧 latent"
```

| 列 | 含义 | 谁看 |
|---|---|---|
| **物理秒** | 真实世界过了多久 | 人读 |
| **mRoPE 单位** | 同一条轴,换成 **base 视频的尺度** | **位置编码吃这个** |

**为什么必须换算** —— 这才是关键:

mRoPE 的旋转频率 `θ_i` 是**固定的**,预训练时按 `base_fps=24` 标定。相位是 `角度 = 位置坐标 × θ_i`。
**若直接把"秒"当位置坐标**,15 fps 的数据会落在和预训练**完全不同的数值范围**上 ——
模型学到的"相邻 token 该差多少相位"就全乱了。

FPS modulation 做的就是:**把任何 fps 的时间折算回 base 时间尺度**,让同一段物理时间在任何帧率下给出相同相位。

**三个模态在同一把尺子上**:

```
秒      0     0.2667   0.5333   …   2.1333
×6      ↓      ↓        ↓            ↓
单位    0      1.6      3.2     …    12.8

video latent k   →  1.6 k        (每片 4 帧 @15 Hz = 4/15 s → ×6 = 1.6)
action token j   →  0.4 (j+1)    (每步 1/15 s → ×6 = 0.4;+0.4 是原生 start_frame_offset)
point 片     b   →  1.6 b        (与 video latent 同格)
```

**1.6 : 0.4 = 4 : 1,正好是 VAE 的时间压缩率。** "1 个 video latent 对应 4 个 action token"
这句话,在位置编码里就是这两个数。

#### ⑦ 解码时把 block 展开回 32 步

```
point_hidden                [8, K, D]
       │  point_hidden[b, original_to_cluster]      ← 簇级广播到点
       ▼
逐点 hidden                 [N, D]  (每 block)
       │  decoder
       ▼
[N, 12]  (每 block)          ← 12 = 4 步 × 3 维
       │  restore_motion      ← 与 motion_blocks 互为逆运算(逐位验证过)
       ▼
prediction                  [32, N, 3]              ← 回到 §6.2 那张图的形状
```

#### 一句话

> **§6.2 管"哪些行属于谁",本节管"每个 token 落在哪个时刻"。**
> 两者正交:`sum(N)` 是空间/样本维度上的拼接,`1+8` 是时间维度上的分片;
> 最终 `[32, sum(N), 3]` 的两个轴就分别是这两件事。

### 6.8 选点与追踪:一个 chunk 内是同一批点

#### ① 每个 chunk **重新随机抽点**,只从锚定帧抽

```python
# pointflow_window.py:113-123
xyz   = read_frame("position.npy", int(rows[0])).reshape(-1, 3)   # ← rows[0] = 锚定帧
uv    = read_frame("uv_px.npy",    int(rows[0])).reshape(-1, 2)
valid = read_frame("valid.npy",    int(rows[0])).reshape(-1)

good = valid & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0) & np.isfinite(uv).all(1)
good &= (uv[:,0] >= 0) & (uv[:,0] <= width-1) & (uv[:,1] >= 0) & (uv[:,1] <= height-1)
candidates = np.flatnonzero(good)

rng = np.random.default_rng(seed)
ids = np.sort(rng.choice(candidates, min(max_points, len(candidates)), replace=False))
```

**三个数组全部读 `rows[0]`** —— 候选池是**锚定帧**里所有有效像素(有限、z>0、落在图内)。
采样是**均匀无放回** `rng.choice`,不是"取前 8192 个"。

#### ② seed:每个 chunk 不同,但**完全确定**

```python
# pointflow_source.py:54
seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{episode}:{int(frame_ids[0])}".encode()).digest()[:8], "little")
```

⇒ 同 episode 同起始帧 → **同一批点**(可复现);**不同 chunk → 不同的一批点**。

#### ③ 追踪:点的身份 = **像素下标**

```python
# pointflow_window.py:174-179
for k, row in enumerate(rows[1:]):                                   # 未来的 32 帧
    future = read_frame("position.npy", int(row)).reshape(-1, 3)[ids]    # ← 同一组 ids
    mask   = read_frame("valid.npy",    int(row)).reshape(-1)[ids]
    mask  &= np.isfinite(future).all(1) & (future[:, 2] > 0)
    target_valid[k] = mask
    target[k, mask] = future[mask] - anchor_xyz[mask]                 # 位移 = 未来位置 − anchor
```

```
uv_px.npy 是**静态像素网格**(无 tracker 漂移)
position.npy 是 [T, H, W, 3] 的逐帧三维位置
⇒ 下标 i 永远指同一个像素
⇒ 同一组 8192 个像素在 33 帧里各读一次位置,差值就是位移
```

**"追踪"是隐式的 —— 没有显式的对应算法,点云本身按像素对齐。**

#### ④ 逐帧有效性 → loss 掩码

`valid.npy` 逐帧给出该像素是否可靠 ⇒ `target_valid [32, N]` ⇒ 就是 §6.6 里 loss 用的 `valid`。

**遮挡/失效的点在该帧被掩掉,但点集不变** —— 还是那 8192 个,只是某些帧某些点不监督。
这正是 §5.4 说的"**样本内逐帧有效点不同 → mask,拓扑不变**"。

#### ⑤ 全链只用 anchor ⇒ 拓扑在整个窗口内固定

```
候选池   ← rows[0] 的 valid
采样     ← rows[0] 的 candidates
coord_shift ← anchor_xyz 的 bbox
体素化   ← floor((anchor_xyz − shift)/0.02)          → original_to_voxel / voxel_representatives
法向     ← anchor 帧 xyz 的 kNN 邻域
```

**选点、体素化、法向、shift 全部只依赖锚定帧** ⇒ 簇的拓扑(`original_to_cluster`)在整个预测窗口里固定 ✓
符合设计:"`C_j` 是**仅根据当前观测**确定的点簇成员集合,整个预测窗口保持不变"。

> 体素化用 `np.unique(grid, axis=0, return_index=True, return_inverse=True)`:
> `inverse` = `original_to_voxel`,`representatives` = 每个体素的**首个**成员下标。
> `coord` / `feat` / `grid_coord` 只存代表点那几行 —— 这就是 §6.1 里 `V = 3908 < N = 8192` 的来源。

#### ⑥ `N` **不是严格常数**

采样之后还有一步**法向估计**,会丢掉邻域退化的点:

```python
tree = cKDTree(xyz[candidates]); _, neighbours = tree.query(xyz[ids], k=min(16, len(candidates)))
local = xyz[candidates][neighbours] - mean
eigenvalues, eigenvectors = eigh(cov(local))
reliable = eigenvalues[:, 1] > 1e-12
ids = ids[reliable]                       # ← 这里会掉点
normals = eigenvectors[reliable, :, 0]
```

所以:

```
N = min(max_points, |candidates|) − 法向估计丢弃的退化点
```

**`8192` 是上界,不是保证值。** 实测 4/4 个 episode 都取满 8192(那些数据上一点没丢),
但严格说这一层**并非严格的常数** —— §6.2 里"padding 零浪费"的结论因此要打个折扣:
它在本数据集上成立,不是一个结构性保证。

---

## 7. 附:`temporal_rotary` 的通道布局已实测验证

`pairwise_point_attention` 里的 `temporal_rotary` 按 Edge mRoPE 的交错布局置零 H/W 频率。
**这条假设已验证——实测调用真实实现 `Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope`:**

```
真实:  T = {0,3,…,57} ∪ {60,61,62,63}  (24 个)
       H = {1,4,…,58}                  (20 个)
       W = {2,5,…,59}                  (20 个)

temporal_rotary 的 spatial 掩码:  逐索引完全一致  ✅
```

复现:

```python
from ...reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRotaryEmbedding as A
obj = A.__new__(A)                                  # 只需这个方法
freqs = torch.zeros(3, 1, 1, 64); freqs[0],freqs[1],freqs[2] = 1.,2.,3.   # T/H/W
out = obj.apply_interleaved_mrope(freqs.clone(), [24,20,20]).reshape(-1)
# out==2 → H 通道;out==1 → T;out==3 → W
```

MoT 侧同样走这套(`unified_mot.py:949` 注释:"In both branches
Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope collapses the T/H/W axis"),
且 `unsqueeze_dim=1` 与 `temporal_rotary` 一致 ✅ **该假设不再是障碍。**

---

## 8. 修法方案(若决定实施)

### 8.1 关键洞察:按 key 集合切分,不需要任何 mask

所需旋转矩阵(**只有两个格子是 temporal**):

```
query \ key     video    action     point
video           full     full       full
action          full     full      TEMP
point           full    TEMP        full
```

把 key 按模态切成**互不相交、并集为全部**的几组,"排除"就变成"换一组 key 重算":

| query 组 | 调用数 | key 集合 | 旋转 |
|---|---|---|---|
| video | 1 | 全部 | full |
| action | 2 | `¬P`(video/action/text) | full |
| | | `P`(point) | **temporal** |
| point | 2 | `¬A`(video/point/text) | full |
| | | `A`(action) | **temporal** |

两组 key 不相交、并集为全部 ⇒ 两次 softmax 合并 = 一次完整 softmax,**精确等价**。

### 8.2 FLOPs 完全不变

```
video  查询:  n_v × n_all
action 查询:  n_a × (n_all − n_p) + n_a × n_p = n_a × n_all
point  查询:  n_p × (n_all − n_a) + n_p × n_a = n_p × n_all
                                        ───────────────────
合计 = (n_v + n_a + n_p) × n_all = N_full × N_all     ← 正好等于现在那一次 full attention
```

### 8.3 不换内核、不加依赖

`cosmos_framework/model/attention/frontend.py:45` 的 `attention()` **本来就支持 `return_lse=True`**,
且我们的路径已在用它的 varlen 参数。

⇒ **全程仍是 flash-attention**,不做以下任何一件:
- ❌ 不引入 FlexAttention(需额外 torch.compile、块对齐约束,与 `[model.compile] enabled=true` 交互有风险)
- ❌ 不依赖 NATTEN(合并用纯 torch 5 行 logsumexp 加权,天然可导)

### 8.4 实施

| # | 动作 | 文件 |
|---|---|---|
| 1 | 新增 `two_pass_point_attention(...)`,签名同 `pairwise_point_attention` | `pointflow_attention.py` |
| 2 | 按模态 gather Q/K/V + 逐组重建 varlen offsets | 同上 |
| 3 | 5 次 `attention(..., return_lse=True)`(可优化为 3 次 2-序列 varlen 批) | 同上 |
| 4 | 纯 torch logsumexp 合并 → 还原 pack 布局 | 同上 |
| 5 | `unified_mot.py:698` 换函数名(不改结构,NVIDIA 文件改动仍最小) | `unified_mot.py` |
| 6 | 语义翻转:默认走新实现;`POINTFLOW_LEGACY_MROPE=true` 取消除融做 A/B | `pointflow_attention.py` |
| 7 | **保留 `pairwise_point_attention` 不动** —— 它就是验证 oracle | —— |

**必须处理的细节**:

- `normalized_k`:gen→und 用归一化 K。`¬P`/`¬A` 两个 full 集合含 und key → 必须用归一化 K;
  两个 temporal 集合只含 gen 的 action/point → 用普通 K
- 合并时防 `-inf − (-inf) = nan`
- `FlexAttention` 不用,故无块对齐约束
- und(causal)路径**完全不动**(text 里没有 action/point,那格子不存在)

---

## 9. 验证计划

| # | 验证 | 判据 |
|---|---|---|
| **V1** | **对 oracle**:小规模下新实现 vs `pairwise_point_attention` | 逐位或 `atol=1e-5` 一致 |
| V2 | 消融:vs `legacy_mrope`,检查 video↔point / point↔point / text 分数**完全不变** | 只有 action↔point 变 |
| V3 | 开销 | step time 增幅 < 2% |
| V4 | 反传 | 梯度有限非零;`lse_B=-inf` 的行不产生 nan 梯度 |
| V5 | 语义 | action↔point 注意力图不再随点簇图像位置系统性变化 |

**V1 是最强判据** —— oracle 已存在且被证明是设计正确的实现。

### 决定要不要做:建议 A/B

因为修法**零额外 FLOPs、不换内核**,可以直接:

1. 先按现状(legacy)跑一版基线
2. 修好后跑同一配置
3. 对比 `pointflow_ade_mm` / `zero_ade_mm` 与 action↔point 注意力图
4. **用数据决定,而不是推理**

唯一成本是训练时间。
