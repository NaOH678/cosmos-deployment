# FK 作为独立模态接入 Cosmos — 方案

> 状态：**已定稿**（§5 九项全部确认）。
> 范围：小样本快速迭代（10 条数据），v1 **只加 FK**，不在点云基础上叠加。
>
> **一句话概括 v1**：把机器人右手 21 个关键点从基座系换算到相机系，
> 每个点编成 `索引嵌入 + 三维位置编码`（**相加**）得到一个锚点 token，
> 再照点云的做法加噪展开成 32 步预测；位置编码**只给时间轴**
> （空间轴照 action 退化为 `(t,0,0)`）；训练与评估的节奏完全照抄点云。
>
> **FK 的位置信息来源只有一条**：三维坐标进 **token 内容**（`MLP_xyz`），
> 不走 uv、不走 RoPE 空间轴 —— 这是相对点云的**有意识简化**，理由见 §1.4(a)。
>
> **改动落在哪**：全部在 **mano** worktree；sft（点云）只读不写。见 §2.0。

---

## 0. 已核实的现状（点云是怎么接的）

### 0.1 数据链路

```
raw head.mp4 ──(Track4World / DA3 离线)──> datasets/sandwich_dense_fullseq_.../outputs/<ep>/
                                             position.npy [T,448,640,3] f32
                                             uv_px.npy    [T,448,640,2] f32
                                             valid.npy    [T,448,640]   bool
                                             frame_indices / timestamps_sec / intrinsics .npy
                                                        │
        pointflow_outputs/task5/mixed_manifest.json  ◄──┘  (schema_version=1)
            每个 episode: {name, pointflow_source: null | {path, video_size_wh, uv_to_video}}
            101 条：10 条带标注，91 条显式 null（null 是"未标注"，不是缺失）
                                                        │
        PointFlowSource.load(episode, frame_ids, ...)  ◄─┘
            prepare_window() → 锚点云 [N] + displacement [32,N,3] + uv_to_video
                                                        │
        SingleRightHandRawDataset.__getitem__  →  sample["pointflow"] = {inputs, targets, metadata}
                                                        │
        JointDataLoader (sparse_data_keys) 保持 list[dict|None]
                                                        │
        omni_mot_model.build_pointflow_batch → PointFlowBatch
```

**窗口构造**（`singlerighthand_raw_dataset.py:266-282,338-342`）

- `source_stride = round(30/15) = 2`
- 窗口 = 33 个源帧号（锚点 + 32 未来），视频与点云**逐帧号严格相等**（`:119,121` 硬断言，不插值）
- `sample_stride=1` → 相邻窗口起点相差 1 个源帧（31/32 重叠）
- 10 条 allowlist → 8 训练 / 2 验证（`torch.randperm(10, generator=seed=42)`），共 10498 个窗口

**关键：坐标系**（最重要的一条）

- 所有几何都在 **DA3 头部相机坐标系**，单位**米**。原点=光心，x 右 / y 下 / z 前
- `pointflow_window.py:135-136` 硬断言 `COMPLETE.json` 里 `coordinate == "camera_depthanythingv3"` 且 `metric_scale` 为真
- **整个 PointFlow 代码里没有任何 robot base → camera 的变换**（`grep c2w / extrinsic / base_link` 零命中）
- 归一化内参：`fx/W = 0.8128, fy/H = 1.1612, cx/W = cy/H = 0.5`，画布 `[640,448]`

**批次张量**（N=锚点数，V=体素数，H=32）

| 字段 | 形状 | 坐标系 |
|---|---|---|
| `anchor_xyz` | `[ΣN,3]` f32 | **DA3 相机系，米** |
| `anchor_uv` | `[ΣN,2]` f32 | DA3 640×448 画布的像素 |
| `normal` | `[ΣN,3]` f32 | 相机系，朝向相机 |
| `color` | `[ΣN,3]` u8 | 在 `anchor_uv` 处 remap 的 RGB |
| `coord` / `feat` | `[ΣV,3]` / `[ΣV,9]` | 体素中心（平移过）；feat = 居中xyz+RGB/255+法向 |
| `displacement` | `[H,ΣN,3]` f32 | **相机系，米**，未来位置 − 锚点位置 |
| `valid` | `[H,ΣN]` bool | 监督质量掩码 |
| `uv_to_video` | `[B,2,3]` | DA3 640×448 → 训练视频 640×842 的仿射 |
| `intrinsics_normalized` | `[B,3,3]` | 归一化内参 |

### 0.2 模型链路

- **几何编码器 = Sonata PTv3**（`pointflow_geometry.py:40`，vendored 在 `cosmos_framework/auxiliary/sonata/model.py:567`）
- 输入 feat 9 维 = 居中 xyz(3) + RGB/255(3) + 法向(3)
- **2 cm 体素化**（`pointflow_voxel_size=0.02`）：N ≤ 8192 个点 → K 个唯一体素格
- token 数 = `K × (1 + H/q)` = `9K`（K 个锚点 token + K×8 个加噪 token），q=4 是 Wan VAE 时间压缩
- 编码器 5 级通道 `(32,64,128,256,512)`，默认取 stage 3 → 256 维
- **token 放置**：追加在每个样本 "full" 段的**末尾**（text|vision|action|sound|EOV 之后，`pointflow_sequence.py:112`）
- **mRoPE 三轴 (t,h,w)**：时间轴对齐视频首个 latent 时刻，空间轴来自 tracker uv 经 `uv_to_video` 仿射
- **模态编号**（仅用于成对注意力）：`0=其它, 1=action, 2=point`（`cosmos3_vfm_network.py:1097-1102`）
- **预测**：每个原始点、整个 horizon 的 **3D 速度** `[H, ΣN, 3]`；loss = 掩码逐点 MSE
- 采样：`sample_pointflow`（Euler/UniPC）

**install 的硬前提**（`cosmos3_vfm_network.py:186-194`）：
`joint_attn_implementation == "two_way"`、`video_temporal_causal == False`、
`enable_fps_modulation`、`base_fps == 24`、`temporal_compression_factor_vision == 4`、`steps_per_token == 4`

### 0.3 训练与评估

- 入口：`torchrun -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/action_policy_singlerighthand_edge.toml`
- **TOML 只是覆盖层**，没有 `[data]`/`[eval]` 段；真正的配方在
  `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py`
- 分支靠**环境变量**打开：`if os.environ.get("POINTFLOW_SONATA_CHECKPOINT"): net.install_pointflow(...)`
- 评估：`validation_iter=100`、`max_val_iter=1`、`run_validation_on_start=False`
- 产物：`<job.path_local>/pointflow_eval/step_%07d/{train_00,train_01,val_00..val_11}/`
  每个 case 8 个文件：`prediction.npz / metrics.json / comparison.png / comparison.mp4 / error_map.png / error_curve.png / error_curve.json / position_grid.png`
- **判据**：`all_ade_mm / zero_all_ade_mm < 1`（相对"零运动"基线），**不看 loss**
- 优化器：`pointflow_branch` 必须在白名单里，`lr_multipliers["pointflow_branch.codec"] = 25.0`

---

## 1. FK 模态的设计

### 1.1 数据从哪来

**FK 原始输出**：`wuji-mjlab` 的 MuJoCo 管线，从 episode 的 qpos 算出 21 个关键点在 `Link_Base` 下的位置。

现成标注在 `raw_data/sandwich_fk21/<ep>/annotations/wuji_fk21.npz`，10 条 allowlist **全部齐备**。

> ✅ **【已确认 A = 直接用现成标注】**
>
> 数据源：`raw_data/sandwich_fk21/<ep>/annotations/wuji_fk21.npz`
> （`positions` `[T,2,21,3]` f32，右手取 `[:,1]`；`coordinate_frame == "Link_Base"`，单位米）
>
> 10/10 条齐备。本会话已独立验证：用 `wuji-mjlab` 的 MuJoCo 管线重算，
> 与现成标注**逐点误差 = 0**（2015 帧 × 21 点），因此可直接信任。
>
> **换算放在训练侧**（dataloader 内），不预处理成缓存：
>
> ```python
> # 启动时算一次（全局常数）
> R, t = base_to_head_camera(URDF)      # 含已修正的光心
> R = roll_about_z(180.0) @ R
> t = roll_about_z(180.0) @ t
>
> # 每个窗口：p_cam = p_base @ R.T + t   （33×21×3 的一次矩阵乘，开销可忽略）
> ```
>
> ⚠️ 因为换算留在代码里，**`fk_coordinate_mode` 开关必须预留**（v2 切 DA3 系时不用改数据）。

**21 个关键点的解剖身份**（决定位置编码，见 1.4）：

```
0  wrist
1  thumb_cmc   2  thumb_mcp   3  thumb_ip    4  thumb_tip
5  index_mcp   6  index_pip   7  index_dip   8  index_tip
9  middle_mcp 10  middle_pip 11  middle_dip 12  middle_tip
13 ring_mcp   14  ring_pip   15  ring_dip   16  ring_tip
17 pinky_mcp  18  pinky_pip  19  pinky_dip  20  pinky_tip
```

### 1.2 ⚠️ 坐标系（最关键的技术决定）

```
FK 原生：     机器人基座坐标系（Link_Base），米
点云/目标：   DA3 头部相机坐标系，米
              ↑
        整个 PointFlow 分支只认这一个系，且没有任何 base→camera 的变换
```

**所以 FK 必须转换到 DA3 相机系**，否则两套几何互相矛盾。

**可用的换算链**：

```
基座 ──[URDF 相机外参，本会话已修正光心]──> 真实 D435 相机系
                                              │
                                              │ 这一步不是刚体变换
                                              ↓
                                          DA3 相机系
```

**为什么最后一跳不是刚体变换**：DA3 是模型**猜**的相机，而真实 D435 有出厂标定。
两者对同一个像素给出的三维点相差一个比例 —— 而且**横纵比例不一样**：

```
横向： x_da3 / x_real = 605.57 / 520.21 ≈ 1.164
纵向： y_da3 / y_real = 564.12 / 520.21 ≈ 1.084      ← 不是 1.164！
```

> ⚠️ **纵向为什么是 564.12 而不是 604.41**：Track4World 把 640×480 **整幅
> `cv2.resize` 压扁到 640×448**（纵向 ×0.9333），纵向焦距跟着被压。
> 详见 §7 末尾的「对 §1.2 的修正」—— 那条修正**推翻**了本节早先写的 1.162。

**另外两个偏差同时存在**（都在 §7 末尾详列）：

- **光心不重合**：真实 448 画布 (324.50, 222.37) vs DA3 (320, 224)，横向差 **4.5 px**
  —— 是**平移**，乘系数吸收不掉
- **DA3 用的是写死的默认内参**：1192 帧里一个数都没变，光心正好在画布正中

**所以这一跳是「各向异性的缩放 + 平移」，不是单一比例。**

> ✅ **【已确认 B = 沿用可视化时的换算方法】**
>
> ```
> 基座系 (Link_Base)
>    │
>    │  base_to_head_camera(URDF)   ← 本会话已修正光心 (0.0325, 0, 0.0043)
>    │  然后 roll_about_z(180°)      ← 把机械 link 系转成图像约定
>    ↓
> 真实 D435 彩色相机系   ← 和可视化时骨架落点一致的那一套
> ```
>
> 这套已经**在视觉上验证过**（骨架落在黑手套上），是当前最可信的换算。
>
> ✅ **【已确认 B2 = v1 先不做 DA3 那一跳】**
>
> **v1 全程使用真实 D435 相机系**（三维 + uv 都是），优先把工程链路跑通。
>
> ### v1 的 mRoPE 空间轴怎么处理
>
> 点云的 `uv_to_video` 仿射期望的是 **DA3 640×448 画布**的 uv。v1 我们手里是
> **真实相机 640×480 画布**的 uv，两者不能直接套同一个仿射。
>
> **v1 的做法（近似）**：把 FK 的 uv **按各自画布归一化** ——
> `(u/W_real, v/H_real)` → 再乘到视频 patch 网格尺寸。
> 这样不依赖任何画布假设，但**和点云 token 的空间坐标不是严格同一套**。
>
> ### ⚠️ 风险（必须记录）
>
> | 风险 | 后果 |
> |---|---|
> | FK 三维在真实系、点云三维在 DA3 系 | 模型看到的几何不同源，「空间桥」可能学不出来 |
> | FK 的 mRoPE 空间位置是近似的 | FK token 和视频/点云 token 的空间对齐不严格 |
> | **因此 v1 的训练结果不能用来评判方案好坏** | v1 的目标是**验证工程链路**，不是验证效果 |
>
> **v2 必须补上**：三维横向 ×1.164（焦距比）+ 用 DA3 内参投影得 DA3 画布 uv。
> 相关代码要**预留开关**（如 `fk_coordinate_mode: "real" | "da3"`），避免 v2 时大改。

### 1.3 ⚠️ 编码器：**不能用 PTv3**（回答需求 2）

**PTv3 是为"无序大点云"设计的**，它的核心步骤是 2 cm 体素化：

```
手部 21 个关键点，相邻间距约 2~3 cm
        ↓ 过 2cm 体素化
MCP / PIP / DIP 可能落进同一个格子 → 被合并成一个 token
        ↓
"这是食指还是中指" 的索引信息【直接丢失】
```

**而 FK 的 21 个点是有固定解剖身份的**，身份信息必须保留。

> **建议：索引嵌入 + 三维位置编码【相加】，结构与点云一一对应**
>
> ```
> token_i = W_idx · e_index(i)  +  MLP_xyz( p_i / s_xyz )  +  e_FK  +  TimeEmbed(σ)
>           └─ "我是哪一节" ─┘     └──── "我在哪" ────┘
> ```
>
> 理由：
> - 点数固定（21），不需要置换不变性
> - 身份是**先验知识**，应该喂进去而不是丢掉
> - 计算量极小（21 个 token），不需要 PTv3 的多级池化

> ✅ **【已确认 C = 索引嵌入 + 三维位置，且编码器接口做成可替换】**
>
> **⚠️ 2026-09-16 修订：原来是「拼接」，现在改成「相加」。**
>
> 理由是对齐点云的分支结构（`pointflow_codec.py:102-104`）：
>
> ```python
> # 点云：两条支路【相加】
> g_j = W_F · F_PTv3_j  +  MLP_xyz( X̄_j / s_xyz )
> #     └ 局部身份 ─┘       └── 三维位置 ──┘
>
> # FK：一一对应
> token_i = W_idx · e_index(i)  +  MLP_xyz( p_i / s_xyz )  +  e_FK + TimeEmbed(σ)
> #         └ 局部身份 ─┘         └── 三维位置 ──┘
> ```
>
> | | 拼接 `MLP([xyz ; e_idx])` | **相加 `W·e_idx + MLP_xyz(xyz)`** |
> |---|---|---|
> | 身份 vs 位置 | 混在一个 MLP 里，模型得自己拆开 | 在**不同子空间**，可独立处理 |
> | 与点云的对应 | 对不上 | **一一对应** ← 这就是"FK 与点云平等"的具体体现 |
> | 可解释性 | — | 位置那一路可单独消融 |
>
> `W_idx · e_index(i)` 扮演的角色，正是点云里 `W_F · F_PTv3` 的角色 ——
> **"这一小块的局部身份"**。
>
> ```python
> # 接口（其他代码只认这个形状）
> class FKEncoder(Protocol):
>     def __call__(self, xyz: Tensor, index: LongTensor) -> Tensor: ...
>
> # v1 实现
> class MLPIndexEncoder(FKEncoder):
>     # token_i = W_idx @ e_index(i) + MLP_xyz(p_i / s_xyz)
> ```
>
> - **21 个可学习的身份嵌入**（1 个对应 1 个关键点），随训练更新
> - 以后若 MLP 容量不够，只写一个新类实现同一接口，其余代码不改
>
> **PTv3 已排除**：它的 2 cm 体素化会把相邻指节合并成一个 token，
> 且它的置换不变性把"第几个关键点"当噪声丢掉 —— 而那正是我们要保留的先验。
> 即使把体素调到 1 mm 避免合并，PTv3 也退化成"逐点 MLP + 空间池化"，
> 且 token 顺序仍由空间填充曲线决定，不是解剖顺序。

### 1.4 ⚠️ 位置编码（回答需求 4）

**两套位置信息都要，但来源不同：**

**(a) 空间位置 → mRoPE 的 (h, w) 分量**

> ✅ **【已确认 G1 = v1 不给 FK 空间轴，走 action 的 `(t,0,0)`】**
>
> ### ⚠️ 先分清：点云是【两条路同时用】，FK 只取其中一条
>
> 这一点很容易混。点云的空间信息走**两条独立的路**：
>
> ```
> 内容（token 向量）：  ① MLP_xyz(X̄_j) + PTv3 特征        ← FK 要走的
> RoPE（Q/K 旋转）：    ④ (t, h̄_j, w̄_j)，h̄/w̄ 来自 uv     ← FK v1 不要的
> ```
>
> 所以 **FK 不是"模仿点云"，而是"取点云的一半"**。这没问题，但要清楚取掉的是什么：
>
> | | 内容里的 xyz（①） | RoPE 里的 (h,w)（④） |
> |---|---|---|
> | 表达 | **绝对**三维位置 | **相对**的画面位置 |
> | 作用 | "我是谁、我在三维的哪儿" | "我该看画面的哪一块" |
> | 机制 | 模型**学**出来的 | RoPE **结构上**给的 |
>
> **去掉 (h,w) 的真实代价**：FK↔视频的注意力**失去相对空间绑定**。
> 因为 21 个 FK token 的 h/w 都是同一个常数，它们相对某个视频 patch 的空间相位差
> **完全相同** → RoPE 帮不上"哪个 FK 点该看哪个 patch"的忙，只能靠内容学。
>
> **但 v1 这么选是划算的** —— 见下面的理由表（画布本来就是错的）。
> **v2 必须做消融**验证这个假设。
>
> **再澄清一个容易混淆的点**：FK 和点云在**三维语义**上确实共用同一个相机系，
> 但 mRoPE 工作在**注意力层**，不是语义层 —— FK token 和点云 token 是序列里
> **两个不同的 token**，每个 token 都需要**自己的** (h,w)。FK 无法"继承"点云的
> 位置，哪怕两者描述的是三维中的同一只手。
>
> **但 (h,w) 也确实不是"FK↔point 对齐"的手段**，两者的不对称在于：
>
> | | 三维从哪来 | uv 的性质 |
> |---|---|---|
> | 点云 | 从视频像素**反投影** | **原生**（"这个点来自哪个像素"） |
> | FK | 从机器人**运动学**算出 | **推导**（投影回去才知道） |
>
> ### v1 的决定
>
> **mRoPE 只给 FK 时间轴 `t`，空间轴退化成常数 —— 走 action 那条现成的路。**
>
> ⚠️ **2026-09-16 修订：不是"自己设常数"，而是复用 Cosmos 给 action 用的同一套机制。**
>
> **证据**：action 就是 Cosmos 里"没有画面位置"的模态，它的做法是：
>
> ```python
> # sequence.py:405-416 —— action 的 mRoPE
> action_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
>     grid_t=action_split_len,
>     grid_h=1,        # ← 空间网格退化成 1×1
>     grid_w=1,
>     ...
> )
> # 注释：action tokens share the temporal space with vision tokens
> ```
>
> 而 `mrope.py:196-203`：
>
> ```python
> h_index = torch.arange(grid_h, ...)   # grid_h=1 → [0]
> w_index = torch.arange(grid_w, ...)   # grid_w=1 → [0]
> ```
>
> **所以 action token 的 mRoPE 位置就是 `(t, 0, 0)`。** FK 照抄：
>
> ```python
> fk_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
>     grid_t=<片数>, grid_h=1, grid_w=1,          # 空间退化
>     temporal_offset=<视频首 latent 的 mRoPE 时间>,
>     fps=..., base_fps=24, temporal_compression_factor=4,
> )
> # → 每个 token 得到 (t, 0, 0)
> ```
>
> **比"自己设常数"多三个好处**：
>
> | # | 好处 |
> |---|---|
> | 1 | 时间轴走**同一个 FPS 调制函数**，不存在"我这边算错了"的可能 |
> | 2 | **代码是现成的**，action 那条路已经跑通，不用新写位置编码逻辑 |
> | 3 | **顺手避开 §4 的坑** —— FK 和 action 在空间轴上**都是 0**，所以"action 的 (0,0) 与 FK 的像素坐标算相位差"这个问题**天然不存在** |
>
> 理由：
>
> | # | 理由 |
> |---|---|
> | 1 | v1 的 uv **画布本来就是错的**（真实 640×480 vs DA3 640×448）。给一个**错误的**空间先验，比不给更糟 |
> | 2 | 避开误差累积：不引入 投影→画布映射 这两步额外变换 |
> | 3 | FK 与 point 的对齐走**三维内容**（token 特征里的 xyz），也就是"三维空间是共同基质"那条路 |
> | 4 | 少一个变量，v1 更容易定位问题 |
>
> ### v2 补上并做消融
>
> 画布问题修好后（见 1.2 / v2）再加回 (h,w)，并用数据回答"空间先验到底有没有用"：
>
> ```
> v2 消融：  A: FK 有 (h,w)   vs   B: FK 无 (h,w)   → 比较 ADE
> ```
>
> **v2 的换算链**（届时才启用）：
>
> ```
> p_i(DA3相机系) ──[DA3 归一化内参]──> (u_i, v_i) ∈ [0,640]×[0,448]
>                                     ↓ uv_to_video 仿射（与点云完全同一个）
>                               视频 patch 网格的 (h, w)
> ```
>
> ⚠️ **代码里必须预留开关**：`fk_spatial_mode: "none" | "uv"`，v2 直接切换，不改结构。
>
> **注意**：v1 不给 (h,w) 还顺带回避了一个依赖 —— **已确认 H2：v1 不带点云分支**，
> 因此 `uv_to_video` 仿射**根本不存在**，v1 想给 (h,w) 也给不了。
> v2 的空间轴分支**依赖点云存在**（仿射来自点云的 `metadata`），代码里要写清这个约束。

**(b) 时间位置 → mRoPE 的 t 分量**

点云的做法：以视频首个 latent 的 mRoPE 时间为原点，加 `k × steps_per_token / fps` 秒
按 `base_fps/tcf = 24/4 = 6` 位置单位/秒缩放。

**FK 完全照抄同一套** —— 这样 FK、点云、视频三者的时间轴严格对齐。

> ✅ **【已确认 D = 直接复用同一个函数】**
>
> FK 的 mRoPE 时间单位**直接复用点云的** `video_aligned_point_positions`
> （`pointflow_branch.py:53-100`），不是"照抄一份代码"，而是**调用同一个函数**。
>
> 理由：只有调用同一个函数，`base_fps / tcf / steps_per_token / 视频首 latent 原点`
> 这几个量才不可能出现"两边各改一半"的漂移。**这是 §6 第 4 步要显式单测的点。**

**(c) 关键点身份 → 不能进 mRoPE，只能进 token 内容**

mRoPE 只编码 (t,h,w) 三个轴。**"这是第几个关键点"不是空间位置，不能塞进 mRoPE**，
只能作为 1.3 里的**索引嵌入**加进 token 的特征向量。

> 这正是需求 4 说的"fk 单独需要对'具体是手掌还是手指'这个序号信息进行编码"。

### 1.5 ⚠️ token 数：21 个点 → 21 个 token（回答需求 5）

> ✅ **【已确认 E = E1：21 个点 → 21 个锚点 token】**
>
> 考虑过的三个方案：
>
> | 方案 | token 数 | 优点 | 缺点 |
> |---|---|---|---|
> | **E1 ✅** | 锚点 21 + 加噪 168 = **189** | 每个关键点一个 token，粒度和 FK 的解剖粒度一致 | 相比点云的 9K 个 token，占比可忽略 |
> | E2 ❌ | 整个手压成 1 个 token | 极省 | 丢掉每根手指的信息 |
> | E3 ❌ | 每根手指 1 个 + 手腕 = 6 | 折中 | 丢失指节粒度 |
>
> **选 E1 的理由**：
>
> - FK 的 21 个点**本身就是有身份的实体**（21 个关节），一对一映射语义最干净
> - E2/E3 的"压缩"不是省算力，而是**主动丢掉先验** —— 而保留先验正是 FK 相对点云的唯一优势
> - 189 个 token 对序列长度的影响可以忽略（点云动辄 9K）
>
> ⚠️ **注意 189 这个数是"锚点 21 + 8 个运动块 × 21"**，
> 其中 8 = `H/q = 32/4`，和点云同源（`q=4` 是 Wan VAE 时间压缩）。

### 1.6 预测目标（回答需求 1「和点云平等」）

> ✅ **【已确认 F = F2】FK 也做预测，与点云完全平等。**
>
> 由此确定的分支形态：
>
> | 项 | 值 |
> |---|---|
> | 加噪 | `xt = σ·ε + (1-σ)·target`，每样本一个 σ（照抄 `pointflow_add_noise`） |
> | 锚点 token | 21 个（纯几何） |
> | 加噪 token | 21 × 8 = 168 个（H=32, q=4 → 8 个运动块） |
> | **序列内 token 总数** | **21 + 168 = 189** |
> | 预测 | 每点每步 3D 速度 `[32, 21, 3]` |
> | loss | 掩码逐点 MSE（`[32,21,3]` vs GT 速度） |
> | 采样器 | 仿 `sample_pointflow`（Euler/UniPC，与视频同一个 shift） |
> | 评估 | ADE / FDE，与 `zero_all_ade_mm` 基线比 |
>
> 对照点云：点云是 `9K` 个 token（K 可达数千），FK 是固定的 189 个 —— **两者量级不同，
> 但 token 的语义结构（锚点 + 加噪块）完全一致**，这正是"平等"的含义。

### 1.7 uv 是否需要（回答需求 3）

用户的判断：**uv 冗余**。 → **同意**。

> ✅ **【已确认 G = uv 既不进 token 特征，v1 也不用于位置编码】**
>
> | 用途 | FK 的 uv |
> |---|---|
> | token 的特征向量 | ❌ 不进（只有 xyz + 索引嵌入） |
> | mRoPE 的 (h,w) | ❌ **v1 不用**（见 1.4a，v2 再加，做消融） |
>
> **不进 token 特征的三个理由**：
>
> 1. uv 可以从 三维位置 + 内参 **算出来**，送进去是纯冗余
> 2. 点云自己的 9 维特征里**也没有 uv**（`pointflow_window.py:200`），FK 保持一致
> 3. 喂一个可推导的量，模型会走**捷径**绕过"学三维"，反而有害
>
> **v1 也不进位置编码的理由**：见 1.4(a) —— 画布不匹配 + 误差累积。
>
> **唯一保留 uv 的地方**：v2 的 `fk_spatial_mode="uv"` 分支（预先写好接口，
> v1 不启用）。这样"到底要不要 uv"这个问题最后由 ADE 数据回答，而不是靠推理。
>
> ### ⚠️ 2026-09-16 修订：v1 的"不给空间轴"有了明确的实现形式
>
> 原来只写了"h = w = 常数"（模糊 —— 常数取什么？）。现在定为：
>
> **复用 Cosmos 给 action 用的同一套机制** —— `get_3d_mrope_ids_vae_tokens(grid_h=1, grid_w=1)`。
>
> 为什么这样更好：
>
> | # | 好处 |
> |---|---|
> | 1 | **不是新发明，是家规** —— action 同样是"没有画面位置的模态"，它的 mRoPE 位置就是 `(t,0,0)` |
> | 2 | 时间轴走同一个 FPS 调制函数，**不可能算错** |
> | 3 | 代码现成，不用新写位置编码逻辑 |
> | 4 | **顺手避开 §4 的坑** —— FK 和 action 空间轴都是 0，"action 的 (0,0) 与像素坐标算相位差"这件事对 FK 天然不存在 |
>
> 详见 §1.4(a)。

### 1.8 需求 7：小样本数据

- 10 条 episode：`examples/pointflow_sandwich_10_episodes.txt`
- 8 训练 / 2 验证（seed 42 固定）
- FK 标注**全部齐备**（已核实：10/10）

---

## 2. 需要改动的文件清单

### 2.0 ⚠️ 工作区分工（先读这条）

点云的代码和 FK 的代码**在两个不同的 git worktree 里**：

```
读：  WorldAct-cosmos3-edge-droid-sft/          ← 点云怎么写的（参考）
写：  WorldAct-cosmos3-edge-droid-sft_mano/     ← FK 的新代码（本次全部改动）
      sft:  一个字都不动
```

**两个 worktree 的实际状态（2026-09-16 核实）：**

| | sft（点云） | mano（FK） |
|---|---|---|
| 提交历史 | 完全相同（`git log` 差 0 条） | 同左 |
| 新增文件 | **71 个未跟踪**（`pointflow_*.py` 全套） | 0 |
| 修改文件 | **24 个已跟踪但未提交**（约 1100 行） | **0**（原始状态） |
| 自己的新文件 | — | 14 个未跟踪的 `tools/*.py` |

**关键：点云那 1100 行改动从未提交过**，所以它们只存在于 sft 的工作区，
mano 里对应文件是**原始版本**。这不影响读取（未跟踪 ≠ 读不到），但影响两件事：

**① 好消息 —— H2 让问题消失**

v1 是 **H2（不带点云）**，FK 的挂载点和点云**没有共存关系**，各写各的。
原文 §2.4「容忍无 point」的三处改造（⑯⑰⑱）**直接作废**：
mano 里压根没有 point 路径，没有东西需要"容忍"。

**② 坏消息 —— sft 里混着【不属于点云的通用修复】**

sft 那 1100 行里，**并非全是点云专属**。已抓到一个确凿的：

```python
# cosmos_framework/utils/misc.py  —— 与点云毫无关系
- results[key] = sum(value_list) / len(value_list)
+ if value_list:
+     results[key] = sum(value_list) / len(value_list)
```

除零保护：某个指标一条都没攒到时不再崩。
**若不移植，FK 的 eval callback 某个 metric 偶尔为空时会 `ZeroDivisionError` 崩掉，
而现象会被误判成"FK 代码写错了"。**

> ⚠️ **开工前必须做的事**：把 sft 那 24 个文件的 diff **逐个过一遍**，
> 把「通用修复」挑出来单独移植到 mano，**不要**把点云专属的部分一起搬。
>
> 已知的另一类：`omni_mot_model.py` / `attention.py` / `unified_mot.py` 里
> **import 块被整体挪位**（各占十几行删除）—— 那是点云新增 import 引发的
> 循环依赖。FK 加自己的 import 时可能撞上同类问题，按同样手法处理即可。

**③ 已核实无需处理**：FK 依赖的四个视频侧硬前提
（`joint_attn_implementation` / `video_temporal_causal` / `enable_fps_modulation` /
`base_fps` / `tcf` / `steps_per_token`）定义在 `edge_model_config.py` 等
**未被点云改过**的文件里，mano 与 sft 一致，不用管。

### 2.1 数据侧

| # | 文件 | 改动 |
|---|---|---|
| ① | **新增** `cosmos_framework/data/generator/action/fk_source.py` | 仿 `pointflow_source.py`：读 FK npz、做坐标变换（1.2）、构造窗口、返回 `{inputs, targets, metadata}` |
| ② | `cosmos_framework/data/generator/action/datasets/singlerighthand_raw_dataset.py` | `__init__` 加 `fk_*` 参数；追加 `sample["fk"] = ...`。**mano 里没有 `mixed_manifest.json` 那套，不用管点云的加载逻辑** |
| ③ | `cosmos_framework/data/generator/action/datasets/action_sft_dataset.py` | 工厂函数透传 `fk_*` 参数 |
| ④ | `cosmos_framework/data/generator/joint_dataloader.py` **和** `dataflow/collators.py` | `"fk"` 加进 `list_collate_keys` / `sparse_data_keys`（**两处都有**，见 §2.4 分类修正）；加 token 预算项 |
| ⑤ | `cosmos_framework/data/generator/sequence_packing/packers.py` | 追加 `packed.fk_data` |
| ⑥ | **新增** `cosmos_framework/data/fk_batch.py` | 仿 `pointflow_batch.py`：`FKBatch` + `build_fk_batch`。**必须用独立缓冲，不能复用点云那套 shape-keyed 池** —— 那套池曾把同形状的 `anchor_xyz` 和 `normal` 混用，毁掉整个 run |
| ⑦ | **新增** `tools/compute_fk21.py` | 用 `wuji-mjlab` 的 MuJoCo 管线算 10 条小样本的 21 点（虽已确认用现成 npz，此工具用于**独立复算校验**，本会话已验证误差 = 0） |

### 2.2 模型侧

| # | 文件 | 改动 |
|---|---|---|
| ⑧ | **新增** `cosmos_framework/model/generator/fk_branch.py` | `FKEncoder` 接口 + `MLPIndexEncoder`：`token_i = W_idx·e_index(i) + MLP_xyz(p_i/s_xyz)`（**相加，镜像 `pointflow_codec.py:102-104`**）+ 解码器（F2）；`fk_spatial_mode` 开关（v1 走 `"none"`） |
| ⑨ | **新增** `cosmos_framework/model/generator/fk_sequence.py` | token 放置（仿 `pointflow_sequence.py:112`，追加在 "full" 段末尾）+ mRoPE 位置：**时间轴复用点云的 `video_aligned_point_positions`；空间轴复用 action 的 `get_3d_mrope_ids_vae_tokens(grid_h=1, grid_w=1)`** —— 见 §1.4(a) |
| ⑩ | `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py` | 仿 `install_pointflow`（`:172-206`）加 `install_fk`；`_encode_X` / `_decode_X`；token 放置 `:995`；MoE 路由 `:1026-1033`；**模态编号加 `3=fk`**（`:1097-1102`）；decode 输出键 |
| ⑪ | `cosmos_framework/model/generator/omni_mot_model.py` | **新增 env 门控 `FK_ENCODER_CHECKPOINT`**（仿 `:321-341`，与 `POINTFLOW_SONATA_CHECKPOINT` 并列且互不影响）；F2 的 noising `:1805`、loss `:1345-1362`、`sample_fk` 仿 `:3379` |
| ⑫ | `cosmos_framework/model/generator/mot/attention.py:67` + `unified_mot.py:680-708` | 若要 fk 参与成对注意力，加 token 范围字段和 dispatch 分支 |

### 2.3 训练与评估

| # | 文件 | 改动 |
|---|---|---|
| ⑬ | `configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py` | 加 `fk_*` 配置键；优化器白名单加 `fk_branch`（`lr_multipliers["fk_branch.codec"] = 25.0` 对齐点云）；callbacks 加 `fk_eval` |
| ⑭ | **新增** `cosmos_framework/callbacks/fk_eval.py` + `fk_eval_cases.py` + `fk_visualize.py` | 仿点云三件套：固定 case、每 100 步评估、ADE 指标、可视化 |
| ⑮ | **新增** `examples/launch_sft_action_policy_fk_singlerighthand_edge.sh` + `examples/toml/sft_config/action_policy_fk_singlerighthand_edge.toml` | 照点云的启动脚本与 TOML 的结构写一份，env 用 `FK_ENCODER_CHECKPOINT`，episode allowlist 仍是那 10 条 |

### 2.4 通用修复的移植 ✅ **已完成（2026-09-16）**

sft 那 24 个文件（约 1100 行）**逐个 hunk 判读过**，结论：

**✅ 已移植（8 个文件，+62/−10 行）**

| # | 文件 | 改动 | 类别 |
|---|---|---|---|
| ⑯ | `utils/misc.py` | `compute_average_results` 加 `if value_list:` 除零保护 | 通用修复 |
| ⑰ | `trainer/__init__.py` | 初始 validation 前后加 `log.info` | 通用诊断 |
| ⑱ | `singlerighthand_raw_dataset.py` | `_OpenCVFrameReader` + `video_decoder` 参数 + `_get_reader` 分发 | **环境必需** |
| ⑲ | `action_sft_dataset.py` | 透传 `video_decoder`（否则配置传参 `TypeError`） | 接线 |
| ⑳ | 配置 `action_policy_singlerighthand_edge.py` | 加 `video_decoder="${oc.env:SINGLERIGHTHAND_VIDEO_DECODER,opencv}"` | 接线 |
| ㉑ | `examples/launch_..._edge.sh` | 删掉 ffmpeg / torchcodec 两条硬检查 | **环境必需** |
| ㉒ | `tools/prepare_singlerighthand_video_cache.py` | 用 `_OpenCVFrameReader` + 加 `--episode-allowlist` | 环境必需 |
| ㉓ | `examples/toml/..._edge.toml` | `logging_iter` 的说明注释 | 纯注释 |

> ⚠️ **⑱⑳㉑ 是"环境必需"而不是"bug 修复"**：本机 **`torchcodec` 加载失败**
> （sft 的真实 venv `.venv` 里也是 `[end of libtorchcodec loading traceback]`）。
> 原始 mano 代码用 `_TorchCodecFrameReader`，启动脚本还会 `python -c "import torchcodec" || exit 1`
> —— **不移植的话 mano 根本跑不起来。**

**已实测验证**：`_OpenCVFrameReader` 在真实 `head.mp4` 上解码
`(4,3,480,640) uint8`，批量/单帧索引语义一致，`close()` 正常；
`config → factory → dataset` 的 `video_decoder` 参数链已用 `inspect.signature` 确认贯通。

**⏳ 判定为"不是通用修复"，未移植**

| 项 | 为什么不移植 |
|---|---|
| `video_temporal_downsample`、`limit_dataloader_worker_threads`、`vae_latent_cache` / `vae_latent_root` / `vae_window_latent_root` | **诞生在点云 hunk 里**，是点云功能的一部分 |
| `episode_allowlist` | 环境变量是 `POINTFLOW_EPISODE_ALLOWLIST` —— 点云的命名。**FK 需要自己的等价物**（见 §2.1 ②） |
| `val_dataloader` + `split_val_ratio=0.2` | FK 要自己的等价物，属于 §2.3 ⑬ 的实现工作 |
| import 块挪位（`omni_mot_model` / `attention` / `unified_mot` / `transforms` / `joint_dataloader`） | 点云新 import 引发的循环依赖；FK 撞上同类问题再照做 |
| `wandb_log_eval.py` 删掉 `assert len(dataset_name)==1` | 会**掩盖**问题。建议不移植，等 FK 真撞上再加 |
| `[model.compile] enabled = true` | 性能开关，不是修复。FK 调试期先别引入额外失败模式 |

**⬜ 纯点云，跳过**：`has_point` / `packed_seq.point` / `pointflow_token_upper_bound` /
`pointflow_modalities` / `data_and_condition` 的 pointflow 字段 / `model_config` 的
pointflow 权重 / `.ruff.toml` 的 sonata 排除 / `.gitignore` / `collators.py` / `batchers.py`

> ⚠️ **教训：hunk 级分诊不够，必须逐行。**
>
> **第一遍**按 hunk 分诊（含 `pointflow` 就整块跳过）→ **漏了东西**。
> **第二遍**改成逐行扫描：把点云 hunk 里**不含任何点云标识符的行**全捞出来
> → **396 行**，滤掉结构性噪音（`)、(`、被重排的参数列表）后，**在 3 个文件里发现真改动**：

| 文件 | 漏掉的内容 | 处理 |
|---|---|---|
| `action_policy_..._edge.py` | `run_validation=True` / `validation_iter=100` / `max_val_iter=1` 整块 + `wandb_mode` 环境变量化 | ✅ 已补（见下） |
| `examples/launch_..._edge.sh` | `NPROC_PER_NODE`、`PYTHON_BIN`/`.venv`、`SINGLERIGHTHAND_VIDEO_DECODER` 导出、`TORCHINDUCTOR_*` Triton 规避 | ⏳ 见下 |
| `joint_dataloader.py` | `limit_dataloader_worker_threads()` 整个函数（钉住 worker 线程数 + nice，治 CPU 争抢） | ⏳ 见下 |

**第二遍补进去的**：

| # | 位置 | 改动 |
|---|---|---|
| ㉔ | 配置 `action_policy_..._edge.py` | `wandb_mode="${oc.env:WANDB_MODE,disabled}"` |
| ㉕ | 配置同上 | `trainer.update(run_validation=True, run_validation_on_start=False, validation_iter=100, max_val_iter=1)` —— **FK 需求 6 的"每 100 步 eval"就靠这一块，mano 原始配置里完全没有** |

**第二遍发现但**没有**直接移植的（需要你定）**：

| 项 | 决定 |
|---|---|
| `PYTHON_BIN` / 解释器 | ✅ **已定：借 sft 的 venv + 加守卫**，见下 |
| `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` / `TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0` | ⏳ 通用的 Triton codegen OOM 规避，但**只在 `[model.compile] enabled=true` 时才需要**。没移植 compile，暂时不用；以后开 compile 会撞上 |
| `limit_dataloader_worker_threads` | ⏳ 通用的 CPU 争抢治理（与点云无关），但**不接线就是死代码**。等 FK 的 dataloader 出来再决定挂不挂 |
| `num_workers=6→8` / `prefetch_factor=3→2` | ⏳ 性能调优，实测依据是针对点云窗口的。FK 的 pipeline 类似但不完全一样，先不动 |

### 2.5 解释器与 worktree 守卫 ✅ **已完成**

**背景（关键）**：`cosmos_framework` 在 sft 的 `.venv` 里是 **editable 安装**，
`.pth` 里**硬编码**指向 sft：

```
.venv/lib/python3.13/site-packages/_editable_impl_cosmos_framework.pth
  → /mnt/.../WorldAct-cosmos3-edge-droid-sft
```

**所以那个 venv "认为自己服务的是 sft"。** 但借用它仍然可行，因为
launcher 的 `PYTHONPATH=.` **优先于 `.pth`**（实测 sys.path 首位是 CWD）：

```
sys.path = [ <CWD=mano>, mano绝对路径, ...conda..., ...site-packages... ]
                ↑ 赢在这
```

**两条路的成本对比**（决定了选哪条）：

| | 给 mano 建 venv | 借 sft 的（**已选**） |
|---|---|---|
| 磁盘 | **+21 GB** | 0 |
| 网络 | 要下载全部 wheel（uv 缓存实测只有 **9.6 MB**，等于空） | 不需要 |
| 磁盘余量 | ⚠️ 该卷 **59T 已用、只剩 833 GB（99%）** | — |
| 代码来源 | `.pth` 指向 mano（显式） | CWD 压过 `.pth`（**隐式**） |

**风险与对策**：隐式优先级的失败模式是"**静默跑到 sft 的代码**"，
现象表现为"我改的代码没生效" —— 极难查。
所以在 launcher 里加**两道守卫**：

```bash
# ① 必须从本脚本自己所属的 worktree 根目录运行
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(pwd)" != "$REPO_ROOT" ]]; then
    echo "ERROR: run this launcher from its own worktree root ($REPO_ROOT), not $(pwd)." >&2
    exit 1
fi

# ② 导入到的 cosmos_framework 必须在 CWD 之内
PYTHONPATH=. "$PYTHON_BIN" - <<'PY' || exit 1
import os, sys
import cosmos_framework
resolved, expected = os.path.realpath(cosmos_framework.__file__), os.path.realpath(os.getcwd())
if not resolved.startswith(expected + os.sep):
    sys.exit(f"ERROR: cosmos_framework resolved to {resolved}, outside {expected}.")
PY
```

**实测三种场景**：

| 运行目录 | 结果 |
|---|---|
| mano 根（绝对路径 / 相对路径） | ✅ 退出码 0 |
| sft 根 | ❌ 退出码 1（守卫①拦住） |
| `/tmp` | ❌ 退出码 1（守卫①拦住） |

> 只加守卫②不够：从 sft 根目录跑时 CWD 和 `.pth` 是同一个，守卫②会**放行** ——
> 那等于用 mano 的脚本参数跑 sft 的代码。守卫①才是关键。

**分类修正**：

- `collators.py` 原来归在「纯点云，跳过」→ **错**。它的 `list_collate_keys` / `sparse_data_keys`
  要加 `"fk"`，是 FK 的**挂载点**（§2.1 ④ 原来只写了 `joint_dataloader.py`，但**两个文件都有这两行**）。

**第三遍（把剩下 4 个大文件逐个 hunk 读完）** —— 结论：

| 文件 | 结论 |
|---|---|
| `unified_mot.py` | 纯点云（§4 的 action–point 规则，走新模块 `pointflow_attention.py`）+ import 重排 |
| `cosmos3_vfm_network.py` | 纯点云（`install_pointflow` / forward 钩子 / `preds_pointflow` 解码） |
| `omni_mot_model.py` | 纯点云（`pointflow_phase(...)` 埋点包住原有代码、install 门控、noising、`sample_pointflow`） |
| `singlerighthand_raw_dataset.py` | 纯点云（`PointFlowSource`、VAE latent 缓存）；**例外见下** |

**第三遍新发现（唯一的漏网）**：

```python
# singlerighthand_raw_dataset.py，被包在 if self._pointflow_source is not None: 里
sample["episode_name"]   = episode.name
sample["raw_frame_ids"]  = observation_indices.copy()
sample["action_frame_ids"] = action_indices.copy()
sample["timestamps_sec"] = observation_indices / episode.source_fps
```

这 4 个字段**本身是通用的**（eval 可视化要知道是哪一集、哪几帧），但**被放在点云块内**，
所以只有在开点云时才会被设置。FK 的 eval 要用就得自己加（或者把它们挪出条件块）。

**另外**：`unified_mot.py` 引用了一个之前没注意到的模块
`cosmos_framework/model/generator/pointflow_attention.py` —— §4 的 action–point 规则实现。
**v1 不需要它**（v1 只给 FK 时间轴，不做成对空间规则），但须知它在。

> ✅ **排查状态：24/24 文件逐行读完（2026-09-16）。**
> 结论：真正的通用修复只有 §2.4 表格里那些；其余全是点云专属或 FK 的挂载点。

> 作废说明：原文 §2.4 的「容忍无 point」三处改造（`packers` / `joint_dataloader` /
> `omni_mot_model`）**已删除** —— mano 里没有 point 路径，无需容忍。见 §2.0。

---

## 3. 训练与评估（需求 6）

**完全照抄点云的节奏**：

```
每 100 步（validation_iter=100）
  → 跑 1 个 val batch
  → FK eval callback 渲染固定 case
  → 写到 <path_local>/fk_eval/step_%07d/
       ├── <case>/prediction.npz
       ├── <case>/metrics.json        ← 主判据：ratio_to_zero
       ├── <case>/comparison.png      ← 4 时刻 × 2 视角：俯视 x-y / 侧视 x-z
       ├── <case>/error_curve.png     ← 误差沿 32 步的走势
       └── <case>/comparison.mp4
```

> **没有 `error_map.png`** —— 那是点云的产物（它把误差画回像素网格上）。
> FK v1 **没有像素空间**，画不出来，也不该画（见 §1.4a）。
> 完整的"看哪个文件、看哪个数"见 **§6.7**。

**评估判据**：与点云同款 —— `all_ade_mm / zero_all_ade_mm < 1`（相对"零运动"基线），
**不看 loss**。F 已确认 = F2，所以这条判据成立。

**可视化**：在 head 视频上叠加
- GT 的 21 个关键点（投影到画面）
- 预测的 21 个关键点
- 二者连线，一眼看出偏差

⚠️ 投影用**真实 D435 内参 + 已修正光心的外参**（就是本会话做可视化时验证过的那一套），
和 §1.2 的坐标换算**同源** —— 这样可视化里看到的偏差，就是模型真正学到的偏差，
不会掺进第二套换算的误差。

**H2 带来的评估侧差异**：点云的 eval 会顺带比较"点云预测 vs FK"，
v1 没有点云，所以 `fk_eval` 只画
①FK 预测 vs FK GT、②FK 预测投影到视频上 —— **不要照抄点云 callback 里依赖点云的部分**。

---

## 4. 风险与注意

| # | 风险 | 说明 |
|---|---|---|
| 1 | **坐标系不一致** | FK 在基座系，点云在 DA3 系。若不做 1.2 的换算，两套几何互相矛盾，训练必然学不到东西 |
| 2 | **缓冲区复用** | `pointflow_batch.py` 的 shape-keyed 缓冲池曾经把 `anchor_xyz` 和 `normal` 混用，毁掉整个 run。FK 若有同形状的字段，**必须用独立缓冲** |
| 3 | **单个 dataset 的限制** | `pointflow_eval_cases.py:91` 硬要求"只有一个 action dataset"，加 FK 时不要新增 dataset，要在同一个 dataset 上加字段 |
| 4 | **install 的硬前提** | `two_way` 注意力、`base_fps=24`、`tcf=4`、`steps_per_token=4` —— FK 分支若同样依赖这些，需一并满足 |
| 5 | **本会话已知的残余偏差** | FK 投影到真实相机画面仍残留约 8–20 px 的偏差（诊断显示不是相机滚转，一部分随手臂姿态变化）。**这个误差会直接进入 FK 模态的三维坐标**。v1 接受它（v1 目标是跑通链路，不是效果），但**它同时是 v2 必须解决的头号问题** |
| 6 | **手套** | URDF 是裸手模型，实机戴厚手套。若 FK 的 21 点要和点云里的手表面对齐（B2 方案），手套厚度会带进标定 |
| 7 | **sft 里的通用修复会被漏掉** | 点云那 1100 行里混着非点云的 bug 修复（已抓到 `misc.py` 的除零保护）。不筛一遍就开工，FK 会崩在莫名其妙的地方，**而现象会被误判成 FK 代码写错了**。见 §2.4 |
| 8 | **无 point ⇒ 无 `uv_to_video`** | v1 不给 FK 空间轴正好绕开；但 `fk_spatial_mode="uv"` 的 v2 分支**依赖点云存在**，代码里要写清这个约束，别让 v2 切开关时才发现拿不到仿射 |
| 9 | **`install_*` 的硬前提是视频侧属性** | `two_way` / `base_fps=24` / `tcf=4` / `steps_per_token=4` 一条都不能动，否则视频侧的 mRoPE 就错了，FK 的 t 轴跟着错 |

---

## 5. 确认清单

| # | 问题 | 结论 | 状态 |
|---|---|---|---|
| **A** | FK 用现成 npz 还是重新生成？ | 用现成 `wuji_fk21.npz`；本会话已用 `wuji-mjlab` 独立复算，逐点误差 = 0，可直接信任；换算放训练侧不预处理 | ✅ 已确认 |
| **B** | FK 换算到相机系的方案 | 沿用可视化时验证过的那套：`base_to_head_camera(URDF)`（含已修正光心）+ `roll_about_z(180°)` | ✅ 已确认 |
| **B2** | v1 是否补 DA3 那一跳 | **不补**，v1 全程真实 D435 相机系，优先跑通工程链路 | ✅ 已确认 |
| **C** | 编码器 | **索引嵌入 + 三维位置编码【相加】`W·e_idx + MLP_xyz(xyz)`**（与点云的分支结构一一对应），接口做成可替换；PTv3 已排除（2cm 体素化会合并相邻指节，置换不变性丢掉解剖身份）。**2026-09-16 修订：拼接 → 相加** | ✅ 已确认 |
| **D** | mRoPE 时间轴是否直接复用点云的 `video_aligned_point_positions`？ | **是**，直接复用同一函数，保证 FK / 点云 / 视频三者时间轴严格一致 | ✅ 已确认 |
| **E** | 21 点 → 21 token？ | **是**（E1）。锚点 21 + 加噪 168 = 189 个 token | ✅ 已确认 |
| **F** | **FK 只输入，还是也预测？** | **也预测**，与点云完全平等（F2） | ✅ 已确认 |
| **G** | uv 的用途 | **v1 既不进 token 特征，也不进位置编码**；空间轴退化为 `(t,0,0)`，**复用 action 的 `grid_h=grid_w=1` 机制**；留 `fk_spatial_mode` 开关，v2 加回 (h,w) 并做消融。**2026-09-16 修订** | ✅ 已确认 |
| **H** | **v1 是否还加载点云分支？**（需求 8「直接加 fk」的落地形态） | **H2：彻底不加载**，新增 `FK_ENCODER_CHECKPOINT` 门控，完全走 FK 自己的路径 | ✅ 已确认 |

> ✅ **【已确认 H = H2：点云分支彻底不加载】**
>
> ```
> v1 数据流：
>   video ─┐
>          ├─→ FK branch ─→ cosmos
>   FK ────┘
>   (point 完全不出现)
>
> env: FK_ENCODER_CHECKPOINT=<path>
>      POINTFLOW_SONATA_CHECKPOINT 不设
> ```
>
> **不再需要"容忍无 point"的改造** —— FK 写在 mano 里，那里本来就没有 point 路径。
> 原文 §2.4 的三处改造已作废，见 §2.0。
>
> ⚠️ **不变的前置**：`install_pointflow` 里的硬前提（`two_way` 注意力、`base_fps=24`、
> `tcf=4`、`steps_per_token=4`）是**视频侧**属性，FK 分支同样依赖（mRoPE 的 t 轴
> 要对齐视频），所以**这些配置保持原样，不要动**。已核实它们在 mano 里与 sft 一致。

---

## 6. 实施顺序

全部改动都在 **mano** worktree，sft **一个字不动**（见 §2.0）。

| 阶段 | 内容 | 完成标志 |
|---|---|---|
| **0** | ~~筛 sft 的 24 个 diff，移植通用修复~~ | ✅ **已完成** 2026-09-16，8 个文件 +62/−10，见 §2.4 |
| 1 | ~~⑦ `compute_fk21.py` + 独立复算校验~~ | ✅ **已完成** 2026-09-16 —— `tools/verify_fk21_allowlist.py`，10/10 条 **max = 0.000000 mm**，见 §6.1 |
| 2 | ~~①⑥ `fk_source.py` / `fk_batch.py` + ②③④⑤ 接线~~ | ✅ **已完成** 2026-09-16 —— 21 个测试通过 + 真实数据集端到端取样本，见 §6.2 |
| 3 | ~~⑧⑨ 编码器 + 序列放置~~ | ✅ **已完成** 2026-09-16 —— `fk_branch.py` / `fk_sequence.py`，28 个测试全绿，见 §6.3 |
| 4 | ~~⑩⑪⑫ 网络接线 + env 门控~~ | ⚠️ **本机可验证部分已完成** 2026-09-16 —— 28 测试全绿 + 分支/加噪/loss 端到端；**完整前向需在 GPU 节点跑**，见 §6.4 |
| 5 | ~~⑬⑭⑮ 配置 + eval callback + launch 脚本~~ | ✅ **代码已完成** 2026-09-16 —— 见 §6.5；**训练启动需在 GPU 节点跑** |
| 6 | 训练 + 看可视化 | `all_ade_mm / zero_all_ade_mm < 1` |

**第 0 步不要跳**：跳过它，FK 后续任何莫名其妙的崩溃都可能是这个原因，
而你会先怀疑自己刚写的 FK 代码 —— 排查成本远高于筛一遍 diff。

### 6.1 阶段 1 记录：FK 复算校验 ✅

**工具**：`tools/verify_fk21_allowlist.py`（新增）

**做什么**：对 allowlist 的 10 条 episode，用 `wuji-mjlab` 的 MuJoCo 管线
**独立重算** 21 个关键点，与现成 `wuji_fk21.npz` 逐点比对。

**为什么需要它**：方案 §1.1 决定**直接用现成标注**而不在训练时重算 ——
这个工具是证明该捷径安全的那道闸。判据不是"差不多"，是**逐点完全一致**。

**结果**：

| | |
|---|---|
| 完成 | **10/10 条** |
| 全局最大误差 | **0.000000 mm** |
| 判定 | ✅ 全部逐点一致，可直接使用现成标注 |

逐条（帧数 / max）：

```
episode_0013_20260731_133649   1192 帧   0.000000 mm
episode_0014_20260731_133743   1211 帧   0.000000 mm
episode_0015_20260730_170501   1367 帧   0.000000 mm
episode_0015_20260731_133841   1751 帧   0.000000 mm
episode_0016_20260730_170616   1424 帧   0.000000 mm
episode_0017_20260731_134058   1459 帧   0.000000 mm
episode_0018_20260730_170833   1419 帧   0.000000 mm
episode_0018_20260731_134203   1201 帧   0.000000 mm
episode_0019_20260730_171242   1342 帧   0.000000 mm
episode_0019_20260731_134304   1430 帧   0.000000 mm
```

报告落盘：`docs/fk21_allowlist_verify.json`

**⚠️ 运行环境（易错，记下来）**：复算需要 `mujoco==3.2.5` + `numpy` + `lmdb`
（`wuji-mjlab/requirements-replay.txt`），**没有任何一个 venv 三者齐全**：

```bash
PYTHONPATH=/mnt/.../shichaojian/.fk-replay-deps \
  /mnt/.../WorldAct-lingbot-va/lingbot-va/.venv/bin/python \
  tools/verify_fk21_allowlist.py
```

- `lingbot-va/.venv` 提供 **mujoco 3.2.5**（版本正好对上 requirements）
- `.fk-replay-deps`（**仓库外**，只装了 `lmdb`）补上缺的那一个
- 每条 episode 的 lmdb（16 MB）先拷到 `/tmp` 再读 —— GPFS 不支持 lmdb mmap
- 比 `sft` 的 `.venv` 多了个 `lmdb`，**但那个 venv 没有 mujoco**

**顺带**：`examples/pointflow_sandwich_10_episodes.txt` 原来只在 sft 里，
已复制到 mano（FK 与点云用同一批 10 条）。

### 6.2 阶段 2 记录：FK 数据侧 ✅

**新增 5 个文件**

| 文件 | 作用 |
|---|---|
| `cosmos_framework/data/fk_camera_extrinsic.py` | **生成物**：基座→相机的 `R`/`t` 常量 + import 时自检 |
| `cosmos_framework/data/fk_window.py` | `FKTiming`（`PointFlowTiming` 的对应物） |
| `cosmos_framework/data/fk_batch.py` | `FKBatch` / `FKNoised` / `build_fk_batch` / `fk_token_upper_bound` |
| `cosmos_framework/data/generator/action/fk_source.py` | `FKSource`：读 npz、换算、构造窗口 |
| `tools/export_fk_camera_extrinsic.py` | 生成上面那个常量模块（`--check` 查漂移） |

**改动的文件**

| 文件 | 改动 |
|---|---|
| `singlerighthand_raw_dataset.py` | 加 `fk_root` / `fk_steps_per_token`；`sample["fk"] = ...` |
| `action_sft_dataset.py` | 透传这两个参数 |
| `collators.py` + `joint_dataloader.py` | `"fk"` 加进 `list_collate_keys` / `sparse_data_keys` |
| `packers.py` | `packed.fk_data = gen_data_clean.fk` |
| `data_and_condition.py` | `GenerationDataClean.fk` / `GenerationDataNoised.fk` 字段（**slots 必须声明**） |

**验证**

```
pytest fk_batch_test.py fk_source_test.py  →  21 passed
```

真实数据集端到端（不是 mock）：

```
数据集构造 ✅   81 episodes / 79549 窗口
sample["fk"]:  anchor_xyz (21,3) f32   displacement (32,21,3)
帧号 33 个、严格递增、与 _window_indices(0,0)[0] 逐位相同  ✅
相机系 z 全为正 (0.740~0.901 m)  ✅
位移量级 0.07 ~ 23.37 cm  ✅
```

**⚠️ 四个踩过的坑（都验证过）**

1. **`pointflow_window.py` 在 mano 里不存在** —— FK 一开始 import 了
   `PointFlowTiming` 直接 `ModuleNotFoundError`。**mano 里一个点云模块都没有**，
   所以有了 `fk_window.py`。**FK 的任何代码都不能 import `pointflow_*`。**

2. **生成的 `R` 少写了每行的方括号**，变成扁平的 `(9,)` 数组 —— 直到 matmul
   才报"size 9 is different from 3"，指向调用方而非生成器。已修，并在生成的模块里
   **加了 import 时的形状/正交/行列式断言**，让这类错误当场炸。

3. **`ulimit -v` 会让 torch 段错误**。验证脚本加了 `ulimit -v 8000000` 直接
   segfault（无 traceback）。**跑含 torch 的脚本不要加 `ulimit -v`**；
   第 1 阶段那个纯 numpy/mujoco 的校验脚本加了没事。

4. **`GenerationDataClean` 是 `dataclass(slots=True)`** —— 不能动态挂属性，必须
   声明字段；而 `PackedSequence` **没有** slots，所以 `packed.fk_data` 可以动态赋值
   （点云也是这么做的）。

> **关于 vt 消融**：`R`/`t` 是**冻结的常量**而不是训练时算 URDF。
> 原因是训练侧不能依赖 wuji-mjlab 的 URDF；写成**文本字面量**而不是 `.npz`，
> 这样光心偏移的改动在 git 里看得见（见 §1.2 那次光心事故）。
> `t = [0.0325, 1.094353, 0.830049]` 与 §1.1 记录的逐位一致。

### 6.3 阶段 3 记录：编码器 + 序列放置 ✅

**新增 3 个文件**

| 文件 | 作用 |
|---|---|
| `cosmos_framework/model/generator/fk_branch.py` | `FKEncoder` 接口 / `MLPIndexEncoder` / `FKBranch`（encode + decode） |
| `cosmos_framework/model/generator/fk_sequence.py` | `FKTokenPayload` / `fk_positions` / `attach_fk_tokens` |
| `cosmos_framework/model/generator/fk_sequence_test.py` | 阶段 3 的闸门测试 |

**改动 1 个文件**：`sequence_packing/sequence.py` 的 `PackedSequence` 加两个字段
（`fk_data` 原始 batch + `fk` 学习后的 token）；`FKTokenPayload` 走 `TYPE_CHECKING`
避免与 `fk_sequence.py` 循环 import。

**闸门三条断言（全部通过）**

```
① token 数 = 189/样本        （21 anchor + 8 block × 21）
② FK 第 b 个 block 的 mRoPE 时间 == 视频第 b 个 latent 的时间
③ FK 的 h/w 全为 0，且与 action token 的取法一致
```

测试用**手工构造的 `PackedSequence`**（文本 2 + 视频 9×2×2 + FK 189），
时间轴由 `get_3d_mrope_ids_vae_tokens` 自己生成，所以第 ② 条是**真比对**，
不是拿同一个公式两边算一遍。

**`FKBranch` 与点云分支的结构对应**

```python
# 点云 pointflow_codec.py:102-104
g_j = geometry_projection(F_PTv3) + xyz_encoder(X̄ / s)
# FK fk_branch.py（MLPIndexEncoder）
g_i = index_projection(e_index(i)) + xyz_encoder(p_i / s)
```

encode 的 noisy token 与点云同构：`point2llm(cat([g, motion])) + e_FK + TimeEmbed(σ)`；
anchor token 只有几何（无运动、无 σ）。

**diff 时最该看的三处**

1. **`decode` 比点云少两样东西** —— 没有 `local`（体素特征）、没有 `relative_xyz`
   （每点相对簇中心的偏移）。因为 FK 一个 token 就是一个点，**K == N**，
   没有"簇内多个点"这回事，所以不需要 gather 也不需要补偏移。
   → 见 §「encoder 和 decoder 是反过来的吗」那段讨论。

2. **`motion_blocks` 的 12 个数是【有序】的** —— 一个 block 的 token 代表
   "这 4 步、按这个顺序"，置换会让模型只能从时间戳反推顺序。

3. **`fk_positions` 的空间轴不是"没写"，是走了 action 的 `grid_h=grid_w=1`** ——
   时间轴也由同一个函数产生，所以 block b 与 video latent b 天然同格。

**⚠️ 两个踩过的坑**

- **`point_offsets` 是【排他的累计末端】**（`[21, 42, ...]`，长度 B），
  不是 `[0, 21, 42, ...]`（长度 B+1）。我一开始照点云的
  `offsets[0] < 0` 误写成 `offsets[0] != 0` —— 那条检查在真实数据上必然失败。
  **测试抓住了它**；现在注释里写明了这个约定。
- 测试帮助函数里 `positions[k]` 的索引容易差一格：**row 0 是 anchor，row b 是第 b 个
  block**（不是 `KEYPOINTS*(1+b)`）。生产代码没错，是测试写错了。

### 6.4 阶段 4 记录：网络接线 + env 门控 ✅

**新增 1 个文件**：`cosmos_framework/model/generator/fk_training.py`
（`fk_add_noise` / `fk_loss` / `fk_ade` / sigma 分箱 —— `pointflow_training.py` 的对应物）

**改动 4 个文件**

| 文件 | 改动 |
|---|---|
| `cosmos3_vfm_network.py` | `install_fk`；forward 加 `fk_displacement`/`fk_sigma`；守卫块；encode→token 写入；`all_gen_indexes`；`fk_hidden` + `preds_fk` 输出 |
| `omni_mot_model.py` | `FK_ENCODER_CHECKPOINT` 门控；`fk_add_noise`；`_replace_clean_with_noised` 挂 `fk_noised`；`denoise` 传状态收预测；`_compute_losses` 加 FK loss |
| `sequence_packing/sequence.py` | `PackedSequence.fk_noised` 字段 |
| `configs/base/defaults/model_config.py` | `fk_loss_weight` / `fk_displacement_scale`。⚠️ **`fk_displacement_scale` 的默认值 `1.0` 是不能用的**，每个实验必须按自己的数据设置 —— 见 §6.8 |

**env 门控与点云【互不蕴含】**：`FK_ENCODER_CHECKPOINT` 走自己的判断，
不设 `POINTFLOW_SONATA_CHECKPOINT`。这是 v1"只加 FK"的代码体现。

**✅ 本机验证通过的部分**

```
28 个测试全绿（无回归）
真实 FK 数据 → 批 → FKBranch.encode → decode：
   anchor_tokens (42,64)   noisy_tokens (8,42,64)   velocity (32,42,3)
加噪 + loss + ADE 端到端跑通
```

**⚠️ GPU 自检（`tools/fk_gpu_smoke.py`，2026-09-16 在 H200 单卡上跑）**

它抓到的东西正是设计它要抓的 —— **只看 CPU 永远发现不了的那类**：

| 轮次 | 现象 | 性质 |
|---|---|---|
| 第 1 轮 | `mat1 and mat2 must have the same dtype, but got Float and BFloat16`（`encode` 里 `motion_encoder`） | **真 bug** —— `encode` 少了 `noisy_displacement.to(dtype)`；点云 `PointFlowBranch.encode` 有这一步 |
| 第 2 轮 | `encoder.index_embedding.weight` 无梯度 | **测试脚本 bug** —— decoder 吃的是凭空 `torch.randn` 的 `hidden`，编码器不在图里 |
| 第 3 轮 | `motion_encoder` 权重梯度为 0 | **测试夹具问题** —— `noisy` 全零时 `dL/dW = dL/dout · inputᵀ` 必然为零 |

**后两条的教训**：冒烟脚本的夹具必须和训练同构。"随便造个张量当主干输出"会让梯度断在
编码器之前；"从零开始"会让权重梯度恒为零。**两种都会把健康的代码报成坏的。**

**⚠️ 训练前最后查出的缺口（第 3 个真 bug）**

**`build_fk_batch` 从来没有被调用过。** 数据集的每个 slot 产出的是原始 dict，
必须有人把它变成 `FKBatch` 才能进 packer —— 点云是在
`get_data_and_condition` 里调 `build_pointflow_batch`，我漏了这一步。
症状会是训练启动几分钟后 packer 报 `gen_data_clean.fk` 不是 `FKBatch`。
已接上（`get_data_and_condition` 的 `GenerationDataClean` 构造处）。

> **教训**：加一个模态时，"数据能取到样本" ≠ "数据能进模型"。
> 中间这一步（原始 dict → 批对象）在点云那边藏在模型文件里，不在数据侧，
> 只顺着数据侧抄就会漏。

**⚠️ 集成验证过程中又抓到的 3 个（都只在真实链路上暴露）**

| # | 现象 | 性质 |
|---|---|---|
| 4 | `PackedSequence.to_cuda` 不搬 `fk_data` / `fk` / `fk_noised` | **配套改动漏了** —— sft 给它的模态对象加了 3 行搬运，我加字段时没加。**训练路径里隐形**（训练时批直接建在 CUDA 上），只有"CPU 建批再 to_cuda"的调用者会踩到 |
| 5 | `omni_mot_model.py` **没 import os**，但门控用了 `os.environ.get` | **会立刻崩** —— sft 的 `import os` 是在点云改动里加的，我当成"点云专属"跳过了，可 FK 门控也要用 |
| 6 | 验证脚本里 `sigma` 硬编码 3 个（批只有 2 个） | 脚本 bug（抄参考脚本时没跟着改样本数） |

**第 5 条用 ruff 兜住了**：对全部改动的文件跑
`ruff check --select F821,F811,F401` → **无未定义名**（唯一的 F401 在两个既有工具文件里）。
**建议把这条当规程**：改完用 ruff 扫一遍，`os` 这种漏导入靠读代码很容易漏。

**新增 `cosmos_framework/scripts/validate_fk_network.py`** —— 阶段 4 唯一没验证的闸门：

用**一层随机权重的 MoT** 搭出真实 `Cosmos3VFMNetwork`（**不需要 checkpoint**），
`install_fk` → 真实打包序列 → 前向拿 `preds_fk` → 反传检查梯度是否到达
编码器 / 位置 MLP / 运动 MLP / 解码器 / 注意力 / 视频 / 动作。
比"启动训练再看"便宜得多。**该脚本于 2026-09-16 在 H200 上跑通（PASS）**：

```
✅ 前向: preds_fk (32, 42, 3)  fk_hidden (8, 42, 2048)
   ✅ noise ✅ fk_encoder ✅ fk_position ✅ fk_motion
   ✅ fk_decoder ✅ attention ✅ video ✅ action
base_weight_preserved: true
```

8/8 梯度检查全过 —— 包括 `install_fk` 未扰动 `vae2llm` 权重，
以及**视频/动作的输入投影仍拿得到梯度**（证明 FK 的接入没有截断别人的梯度）。

**未训练分支的基线数字（有意义的对照）**

```
loss      = 1.006      ← 单位方差噪声主导：随机预测≈1.0，与设计文档一致
ade       = 799 mm     ← 一步干净估计
zero_ade  = 102.8 mm   ← 零运动基线（32 步约 10 cm 的手部位移，量级合理）
比值 ≈ 7.8             ← 方案判据是 < 1
```

**✅ 完整前向已验证**（见上）。剩下的唯一未验证项是**带真实 checkpoint 的完整训练**，
那属于阶段 6。

**⚠️ 抓到的真 bug（2 个，都由测试抓到）**

1. **`FKBranch.decoder` 的输入维度漏了 σ 项**。点云写的是
   `hidden_dim*2 + local_dim + 5 + 3*steps_per_token`（两项 `hidden_dim`：
   backbone hidden + `sigma_features`，**σ 是 `hidden_dim` 宽的向量不是标量**），
   我写成了 `hidden_dim + ...`，直到 `matmul` 报 `42x140 and 76x32` 才暴露。

2. **bf16 下的 dtype 不匹配（GPU 自检抓到的）**：

   ```
   RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
     at fk_branch.py: motion = self.motion_encoder(blocks)
   ```

   `fk_add_noise` 产出的 `xt` **是 float32**（刻意的：`FKBatch.to` 里写明
   "不要把度量几何转成 bf16"），而训练时参数是 bf16。
   **`encode` 少了一次 `noisy_displacement.to(dtype)`** —— 点云的
   `PointFlowBranch.encode` 有这一步，我漏了。
   **CPU 单测全是 float32，永远碰不到这个分支**，只有 GPU 上才会炸。

**由这两个 bug 加了一组回归测试**：`cosmos_framework/model/generator/fk_branch_test.py`
（7 个），其中 bf16 那条**在 CPU 上就能复现 GPU 的报错** —— 已验证：
撤掉修复 → `1 failed`（同样的 RuntimeError），还原 → 通过。

> **教训**：dtype 契约要当成接口来测。`float32 输入 + bf16 参数` 是训练时的**常态**
> 而不是边界情况，但本机只跑 float32 的话完全测不到。

### 6.5 阶段 5 记录：配置 + eval + 启动脚本 ✅ 代码完成

**新增 6 个文件**

| 文件 | 作用 |
|---|---|
| `cosmos_framework/callbacks/fk_visualize.py` | 指标（ADE / 每步 / 每关键点）+ 骨架图 + mp4 + 误差曲线 |
| `cosmos_framework/callbacks/fk_eval_cases.py` | 固定用例选择（只读标签、写盘、resume 校验） |
| `cosmos_framework/callbacks/fk_eval.py` | `FKEvalCallback` |
| `cosmos_framework/model/generator/fk_sampling.py` | 无标签 Euler/UniPC 采样 + shift 处理 |
| `examples/launch_sft_action_policy_fk_singlerighthand_edge.sh` | 启动脚本 |
| `examples/toml/sft_config/action_policy_fk_singlerighthand_edge.toml` | 对应 TOML |

**改动 3 个文件**：配置（`fk_*` 键 / 优化器白名单 / `lr_multipliers` / callback / `dataloader_val` /
`episode_allowlist`）、`singlerighthand_raw_dataset.py`（`episode_allowlist`）、
`action_sft_dataset.py`（透传）；`omni_mot_model.py` 补了 **`sample_fk`**（阶段 4 漏的，eval 依赖它）。

**⚠️ 可视化**是**自足**的（相机系骨架双视图 + 误差曲线），**不叠在视频上**：v1 刻意不解析
FK 落到画面哪里（§1.4a），叠视频就得用上那个被避开的变换 —— **一张依赖它的图可能看着对，其实是错的**。

**⚠️ 三个真 bug（都验证过）**

1. **`or None` 从不生效**。配置写的是
   `fk_root="${oc.env:FK_ANNOTATION_ROOT,}" or None` —— `or` 绑定在**字符串字面量**上，
   而字面量非空，所以永远短路返回字符串本身。未设置 env 时值是 **`''`**，
   而数据集用 `is not None` 判断 → **空串通过检查** → `Path('')` 就是当前目录，
   会拿 CWD 当标注根去找，报"缺标注"而不是干净地关掉 FK。
   **已改成真值判断 `if fk_root:`**（`episode_allowlist` 本来就是真值判断，那个是对的）。

2. **`split_val_ratio=0.03` 在 10 条数据上算出 0 个验证 episode** → `dataloader_val` 空、
   定期 eval 没东西可跑。已改 **0.2**（10 条里 2 条验证）。

3. **mano 没有 `dataloader_val`** —— `run_validation=True` 需要它。已补，
   `batch_size=1 / num_workers=0` 保证 eval 确定性。

**验证**

```
28 个测试全绿（无回归）
可视化用合成数据跑通：ADE 9.55mm / zero 23.31mm / ratio 0.410 → 判据读得对
  comparison.png (442 KB) / error_curve.png (29 KB) / comparison.mp4 (349 KB)
配置解析：optimizer 含 fk_branch、lr_multipliers=25.0、callbacks 含 fk_eval、
  dataloader_val 在、validation_iter=100、split_val_ratio=0.2
未设置 FK_ANNOTATION_ROOT → fk_root='' → 数据集判定「关掉 FK」✅
```

**启动命令（GPU 节点）**

```bash
cd /mnt/.../WorldAct-cosmos3-edge-droid-sft_mano
bash examples/launch_sft_action_policy_fk_singlerighthand_edge.sh
```

脚本开头会**硬检查**：标注根存在、allowlist 存在且**正好 10 条**、每条都有 `wuji_fk21.npz`。
产物落在 `$OUTPUT_ROOT/fk_eval/step_%07d/<case>/`，每 `validation_iter`（100）步一份。

**第 3 步的单测很关键**：

- **t 轴**：和视频差一点点，训练能跑但学不到东西，而且**从 loss 上看不出来** —— 必须显式断言
- **h/w 轴**：断言它们**恒为 0**，并且**和同一序列里 action token 的取法逐位相同**。
  这条断言的作用是防止将来有人"顺手"给 FK 补上空间轴而没走 `fk_spatial_mode` 开关 ——
  那会让 v1/v2 的消融失去意义

### 6.6 收尾：稀疏键与 token 预算 ✅

阶段 2 做 diff 三分类时，把 sft 里几处"看起来是点云专有"的行归到了"作废"。
穷尽复核（把 sft 每个含 `pointflow` 的文件与 mano 对应文件的 `fk` 引用逐一对齐）
发现其中**三处对 FK 同样必要** —— 它们不涉及几何，是**打包管线的通用契约**：

| 文件 | sft 的行 | 为什么 FK 也要 |
|---|---|---|
| `dataflow/batchers.py` | `num_tokens += pointflow_token_upper_bound(...)` | 打包器按这个计数**切序列**。漏掉就少预留 189 个 token，packed sequence 会溢出 |
| `joint_dataloader.py` | 同一句 + `_update_output_batch` 的 None 回填 | 同上；回填见下 |
| `dataflow/collators.py` | `collate` 的键归一化 + `_accumulate` 的 None 回填 | 同上 |

**为什么"稀疏键"必须回填**：只有部分 episode 带 FK 标注，一个 packed group 可能
**混着带与不带**的样本。两个累加器都是"首次出现才建列表"，于是

- 不带 FK 的样本排在前面 → 列表**晚一个样本才开始**，短一格
- 排在后面 → 列表**到第一个带 FK 的样本就停**，更短

两种情况都**不报错**，只是和 `sequence_plan` 静默错位 —— 所以必须有回填，
把两个方向都补成 `None`，保证 `output_batch["fk"]` 恒等于样本数。

`_MULTI_ITEM_KEYS` 两边**完全一致**（sft 没往里加点云），确认 FK 走的就是这条稀疏键路径。

**验证**（`dataflow/fk_sparse_key_test.py`，直接驱动真实的 `VFMListCollator.collate`）：

```
✅ 带 FK 在前、不带在后 -> len(fk) = 3  [有, None, None]
✅ 不带在前、带 FK 在后 -> len(fk) = 3  [None, 有, 有]
✅ 整组无 FK -> 不产生 'fk' 键
🔬 去掉回填（对照）-> len(fk) = 1  ← 正是被修掉的错位
```

最后一行是**对照**：把修复前的累加循环原样跑一遍，证明前面的断言确实有牙齿。

token 预算在两条路径上都实测为 189：

```
joint_dataloader: 无 FK 237 → 有 FK 426   (差 189)
batchers:         无 FK  20 → 有 FK 209   (差 189)
```

**复核中确认"不适用"（非遗漏）的三处**：

| 位置 | sft 里做什么 | 为什么 FK 不需要 |
|---|---|---|
| `transforms.py` 的 `resize_pointflow_metadata` | 重算点云的**像素仿射** | FK 几何是**相机系米**，无像素空间；metadata 里没有尺寸也没有内参 |
| `transforms.py` 的 `has_point` | 往 `SequencePlan` 写标志位 | mano 的 `SequencePlan` **没有这个字段**（只有 text/vision/action/sound），v1 无条件打包 |
| `mot/unified_mot.py` + `mot/attention.py` 的 `pointflow_modalities` | 供**成对参考注意力**使用 | `reference_attention_enabled()` **默认关闭**，回落到 `dispatch_attention_fn`（即全 mRoPE）；FK 走的正是这条路 |

### 6.7 阶段 6 使用手册：怎么看训练日志与 eval 产物

#### 为什么不能只看 loss

训练 loss 是"预测速度 vs 真实速度"的均方误差，而**噪声是单位方差的** ——
所以**一个恒输出 0 的模型 loss 就已经 ≈1.0**（未训练基线实测 `1.006`）。
loss 从 1.0 掉到 0.8 说明不了任何事：它可能只是在学怎么抹掉噪声，动作完全没动。

**判据因此必须换成 ADE**（毫米）：

| 键 | 含义 |
|---|---|
| `train/fk_ade_mm` | 训练中**一步**的干净估计误差：把预测反解成位移后与真值比，单位 mm |
| `train/fk_zero_ade_mm` | 同一套算法，但假设**预测恒为"不动"**。手 32 步本来就移动约 10 cm，所以"什么都不猜"天然就有这个误差 |
| 比值 | `fk_ade_mm / fk_zero_ade_mm` |

**`< 1` 的意思**：模型比"猜它不动"更接近真值 —— 即**真的在学手的运动**。
未训练基线 **≈7.8**（799 mm / 102.8 mm）。

> ⚠️ **`train/fk_ade_mm` 与 eval 里的 ADE 不是同一个数。**
> 前者是训练时的一步估计；后者把采样器完整跑 4 步。
>
> ⚠️⚠️ **但"后者更该信"是错的，已于 2026-09-17 用测量推翻。** 两者差的
> **不只是步数，还有视频条件**，而视频条件那一项是致命的：
>
> | | 视频 token | FK 状态 σ | 速度误差 |
> |---|---|---|---|
> | `training_step` | 非条件帧**加噪到 σ**、条件帧 σ=0 | 同一个 σ | **~27 mm**（σ∈0.8–0.9 分箱） |
> | `sample_fk` / `_fk_fitting_context` | **全干净、timestep=0** | 采样器的 σ | **~1200 mm**（σ=0.8333） |
>
> 训练每次只抽**一个** σ 同时给视频和 FK（`fk_add_noise` 明确要求 "one shared
> video sigma per sample"，`_add_noise_to_input` 再用 `1 - condition_mask` 把
> 观测帧归零）。而 `_fk_fitting_context` 把**整段视频**按干净态打包，于是 FK
> 令牌被问到的是一个训练里从未出现过的 (视频, 状态) 组合。
>
> `sample_fk` 的 docstring 里那句 "the FK tokens are meant to be denoised
> *inside* that loop" 才是部署形态（视频与 FK 同步去噪），而 eval 并没有那样跑。
>
> ⚠️ **上面"外推"的结论已于 2026-09-23 作废一半。** 那段推理建立在"训练每次
> 只抽一个 σ 同时给视频和 FK"之上，而本配方已经改成
> `independent_fk_schedule=True`（`action_policy_singlerighthand_edge.py:65`）：
> 视频和 FK 的 σ 现在是**各自独立**抽的。独立意味着联合分布是乘积分布，
> "视频干净 + FK 加噪"是它**支撑集内**的一个点，不是外推。
>
> 于是两个 arm 都落在训练分布里，只是位置不同：
>
> | arm | 视频 | FK σ | 位置 |
> |---|---|---|---|
> | 主 arm（`sample_fk`） | 全干净、timestep=0 | 采样器的 σ | 乘积分布的一个角 |
> | joint arm（`generate_video=True`） | **从噪声去噪**（i2v，首帧干净） | 同一个 σ | 对角线 σ_v = σ_FK |
>
> 两者都不是外推，**差别在视频内容而不在支撑集**——这正是它们可以对比的原因。
>
> ⚠️ 但有一个真实的伪影要记住：`independent_action_schedule` 仍是 `False`，
> 而 `_fk_fitting_context` 把 action 按干净态（timestep 0）打包。所以 joint arm
> 呈现的是 `(视频 σ, action 0)` 这个训练只在 σ≈0 时才产生的组合。**主 arm 也有
> 这个伪影**（它是 `(0,0)`，自洽），所以**两个 arm 之差仍可归因于视频**，但
> joint 的绝对值带有这一项，读数时要记得。
>
> **已完成**（2026-09-23）：`sample_fk(..., generate_video=True)` 加了第三个
> arm，与主 arm **共用同一个 FK 种子**，所以两者只差视频。开关是
> `FK_EVAL_JOINT`（默认关）。详见 `_make_joint_velocity` / `_sample_joint`。
>
> **已完成**（2026-09-24）：第四个 arm —— 把上面那个伪影去掉。`joint_action=True`
> 让**未来的 action 也从噪声去噪**（conditioning 的初始状态帧仍干净，与
> `_prepare_inference_data` 的 blend 一致），状态向量从 `[vision | FK]` 变成
> `[vision | action | FK]`，即**部署配置**。开关 `FK_EVAL_JOINT_ACTION` /
> `FK_ROLLOUT_JOINT_ACTION`（都默认关，且都要求各自的 joint 开关先开）。
> 它**不是**第三个采样次数：它是 joint arm 的一个模式，写进 `joint_action/`
> （rollout 写 `fk_rollout_joint_action/`），与两段 arm 的 `joint/` 并列。
>
> 与 `joint/` **同 case 同种子**相减 = 干净 action 值多少；与主 arm 相减 = 视频
> + action 一起值多少。**数字还没出**（本节点无 GPU）。
>
> ⚠️ 两段 arm 的伪影在四段 arm 里只剩一半：视频和 action 现在都在采样器的 σ 上，
> 与训练一致；`[vision | action | FK]` 的排序与部署 `[vision | action | sound]`
> 同构，所以 `* scale` 仍只切 FK 尾部。三段的边界由 `fk_sampling.JointLayout`
> 统一给出（`_fk_fitting_context` 建一次，seed 与 velocity 共用），
> `flatten_pieces` 在 `cat` 前校验每段长度——`torch.cat` 对长度是宽容的，段长错了
> 只会静默错位。
>
> **未做**：`vision_sigma=σ` 那条路（把干净视频加噪到固定 σ，2026-09-17 已加）
> 仍是"给定视频"的一族，与 joint arm 的"生成视频"不同，两者不要混用。
>
> ⚠️ **另外记一笔已发现的量纲问题**：`_fk_fitting_context` 往
> `vision.timesteps` 写的是 `float(vision_sigma)`（0–1），而网络按
> `timestep_scale = 1/num_train_timesteps` 期望 0–1000（`omni_mot_model.py:206-210`
> 的注释、`:1526` 训练侧写的是 `sigmas * max_timestep`）。`sample_fk` 传 `None`
> → 0，`0*1000 == 0` 所以不受影响；但经 `fk_velocity_field` 用 `vision_sigma`
> 的那些测量，视频 token 的 timestep embedding 会小 1000 倍。**未修**，本改动
> 也没有沿用那一行（joint arm 走 `_copy_timestep_to_template`，与部署路径一致）。

#### 训练日志

```
$OUTPUT_ROOT/logs/action_policy_fk_singlerighthand_edge_sft.log
```

默认 `$OUTPUT_ROOT=/mnt/.../runs/cosmos/fk-singlerighthand-edge`。
`Iteration N: ...` 行**每 `logging_iter`=10 步**一条，形如：

```
Iteration 1: Total Loss: ... | Video Loss: ... | Action Loss: ...
           | FK Loss: 1.0060 | FK ADE: 799.00mm (zero 102.80mm)
```

wandb 是 `offline`（toml 里 `wandb_mode`），落在
`<path_local>/wandb/`；推上去用 `wandb sync`，或把 `wandb_mode` 改成 `online`。

#### eval 钩子

`FKEvalCallback`，参考点云的 `pointflow_eval.py`。触发链：

```
validation_iter = 100        → 每 100 步一次 validation
FK_EVAL_EVERY   = 1（默认）   → 每次都触发 FK eval
FK_EVAL_CASES   = 2（默认）   → 每个 split 取 2 个窗口
```

`fixed_cases` **同时遍历 train 和 val 两个 split**，所以每 100 步产出 **4 个 case**：

| case | 是什么 | 掉队说明什么 |
|---|---|---|
| `train_00` / `train_01` | **拟合**：训练集窗口 | 拟合都不行 → 优化/容量问题 |
| `val_00` / `val_01` | **泛化**：验证集窗口 | 只有它们掉队 → 过拟合 |

挑选规则：**运动量最大的窗口优先，不同 episode 优先于同 episode**。选择结果写进
`fixed_cases.json`，**续训时校验**，保证前后看的是同一批 case。

**产物路径：**

```
<path_local>/fk_eval/
    fixed_cases.json              ← 4 个 case 的身份，选定后固定
    step_0000100/{train_00,train_01,val_00,val_01}/
    step_0000200/ ...
```

`path_local` = `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>`（`config.py` 的 `JobConfig.path_local`），
取 toml 的 **`job.name`**，所以本次是：

```
/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/fk-singlerighthand-edge
  /cosmos3_action/action_sft/action_policy_fk_singlerighthand_edge/fk_eval/
```

（`run_validation_on_start=False`，所以**没有 step_0**；每 100 步一个新目录。）

**每个 case 目录看哪个文件：**

| 文件 | 看什么 |
|---|---|
| `metrics.json` | **主判据**：`all_ade_mm` / `zero_all_ade_mm` / `ratio_to_zero`，外加 21 个关键点各自的误差 |
| `comparison.png` | **最该看的图**：4 个时刻 × 2 视角（俯视 x-y、侧视 x-z），实线真值、虚线预测。一眼分辨"整体没动"还是"动错方向" |
| `error_curve.png` | 误差沿 32 步的走势：整体偏，还是越往后越飘 |
| `comparison.mp4` | 32 步动画（`FK_EVAL_VIDEO=1`，默认开） |
| `prediction.npz` | 原始预测，想自己算别的指标 |

`ratio` 还会**同时打进训练日志**，不必翻文件：

```
FK eval val_00: ADE 42.31 mm (zero 102.80 mm, ratio 0.41)
```

**两个额外诊断维度**（都在 `metrics.json` 里）：

- `euler16_*` + `sampling_difference_mm`：**同一噪声起点**换采样器/步数再跑一遍 ——
  用来回答"是不是采样器不够好"，省掉一次实验
- `zero_per_step_ade_mm`：基线也按步给出，能看出哪几步本来就难预测

**调频率不用改代码**，跑前设环境变量：

```bash
# every_n 数的是 validation 次数，而 validation 每 100 步一次 —— 所以 =5 就是每 500 步
FK_EVAL_EVERY=5 bash examples/launch_sft_action_policy_fk_singlerighthand_edge.sh

# 每个 split 取 4 个窗口 → 4×2 = 8 个 case（train 4 + val 4）
FK_EVAL_CASES=4 bash examples/launch_sft_action_policy_fk_singlerighthand_edge.sh
```

> ⚠️ 这几个变量是**建配置时**由 `oc.env` 读的，必须在启动脚本**之前**设好
> （启动脚本自己不导出它们，靠的是环境继承）。
> `fixed_cases.json` 一旦写下就**固定**，改 `FK_EVAL_CASES` 会被校验拒绝 ——
> **要么换输出目录，要么删掉 `fk_eval/fixed_cases.json` 重来。**

**一句话用法**：每 10 步扫一眼日志的 loss（只是"在不在学"的粗信号）；
每 100 步去 `fk_eval/step_XXXXXXX/` 看 `metrics.json` 的 `ratio_to_zero` 和 `comparison.png`
（这才是"学得对不对"）。

---

> 📄 **本篇 §6.8 与 §6.9 的完整独立版本（含排除过的全部假设、验证数据、操作手册）见
> [`fk_summary_1.md`](./fk_summary_1.md)。**

### 6.8 `fk_displacement_scale`：移植时漏掉的一行（2026-09-17 已修）

#### 问题

`fk_displacement_scale` 是从点云那条线移植过来的，但**只搬了代码，没搬这个值** ——
它留在了 `model_config.py` 的默认值 **`1.0`**。点云孪生在
`action_policy_singlerighthand_edge.py` 里专门覆盖了它：

```python
model_config["rectified_flow_training_config"]["pointflow_displacement_scale"] = 0.0839
```

注释把原因写死了：

> Every other modality in this pipeline is already **unit-scale** — the video VAE
> latent measures std 0.872, and the action is normalised by `quantile_rot` — so
> this restores the convention rather than introducing a special case.
> ... A std rather than a q01/q99 span, because **the noise is unit-variance Gaussian**.

#### 为什么这必须是单位尺度

RF 的前向过程是

```
x_σ = σ·ε + (1−σ)·target        ε ~ N(0,1)  逐元素单位方差
```

`ε` 是单位方差的，所以 **`target` 也必须是单位尺度**，否则两项不可比。

FK 的标注是**米**（`displacement = camera[1:] − anchor`，未经任何归一化），
实测 per-element std = **0.074450 m**，也就是比噪声小 **13.4 倍**。

#### 实测的后果

| | 数值 |
|---|---|
| σ 处信号/噪声相等的位置 | `(1−σ)·0.0745 = σ·1.0` → **σ ≈ 0.069** |
| σ=0.83（训练分布 43.8% 的质量所在）处，位移占状态标准差的 | **1.5%** |
| 要把一步估计的误差压到位移自身的 10%，需要的速度相对精度 | **0.9%** |
| 训练实际达到的相对精度（MSE 0.000241） | **1.55%** |
| → 所以 `ratio_to_zero` 卡在 | **~0.23**，再也下不去 |

即：**模型几乎全程在学"预测噪声"**，位移那点信息只是 loss 里 ~7% 的残差，
梯度里真正有用的信号被稀释了 13 倍。采样器的 σ 网格
`[1.0, 0.9375, 0.8333, 0.625, 0.0]` 是按单位尺度校准的，每个节点都深埋在噪声主导区。

#### 修法

```python
model_config["rectified_flow_training_config"]["fk_displacement_scale"] = 0.074450
```

0.074450 = 训练 10 个 episode、26,522,496 个元素的 pooled per-element std。复现：

```bash
python tools/scan_fk_displacement_scale.py
```

该工具**自校验**：它同时打印"每窗口平均关键点位移范数"（实测 95.8 mm），
这个量就是训练日志里的 `train/fk_zero_ade_mm`（54–123 mm）。两者对不上就说明
滑窗枚举和训练不一致，那个 std 不能信。

#### 连带影响

- **旧 checkpoint 不能续训**。target 和分支学到的 scale 都变了（点云孪生同样注明）。
  改完 FK loss 的**读数会变大**、"猜不动"基线约翻倍 —— **只看 ADE 对零基线，别看 loss 值**。
- `fk_loss_weight` 保持默认 `1.0`：点云孪生也是默认值，只设 `loss_scale=10.0`。**不是漏项。**
- **`anchor_xyz` 的尺度是对的**：实测 per-element std 0.3172 m、每关键点范数均值 0.814 m，
  本来就是 O(1)，所以 `xyz_scale=1.0`（`FK_XYZ_SCALE`）无需改动。这是同一类 bug 的另一个
  候选，已查过，干净。

#### 防复发

新增 `cosmos_framework/configs/base/experiment/action/posttrain_config/fk_displacement_scale_test.py`：
断言配置值与实测 std 的比值落在 2× 内。**已验牙**：改回 `1.0` 时报
`比值 = 13.432，容差 2.0x` 并以退出码 1 失败。标注目录不存在时跳过而非失败。

---

### 6.9 σ 耦合：eval 发散的真正原因（2026-09-20 已修）

**点云那条线在 2026-09-19 独立发现并修复了同一个 bug**
（`WorldAct-cosmos3-edge-droid-sft/docs/pointflow_per_point_tokens_20260919.md`）。
症状逐条对得上，这不是类比，是同一个机制。

#### 机制

1. **训练时手部动作与视频共享同一个 σ**。`fk_add_noise(gen_data_clean.fk, sigmas, ...)`
   里的 `sigmas` 就是 vision 的。也就是"加噪的手"永远搭配"同一个 σ 加噪的视频"。
2. **eval 的条件组合在训练分布中概率为零**：`_fk_fitting_context` 打包**干净视频**
   （连未来都是真值），手却从纯噪声出发。
3. **模型学会了捷径**："视频干净 ⇒ σ≈0"。而 σ=0 时 `x0_hat = x_t`，
   即**把输入原样当答案**——于是输出 = 初始噪声 × 系数。
4. 这解释了全部现象：预测的 21 个点**沿各自的初始噪声方向飞出**，
   互不相干、不成手型；训练指标一路降（训练时条件匹配），eval 单调恶化。

对照点云文档的原话："`comparison.png` 呈**直线扇形炸开**：所有点的预测位移沿各自
初始噪声方向飞出，即输出 ≈ 缩放后的初始噪声"——与我们的"散乱、没有手型"同一现象。

#### 修复

新增 `rectified_flow_training_config.independent_fk_schedule`（默认 False）。
开启后每个 step 手部动作**从 vision RF 采样器独立抽一个 σ**
（`_get_train_noise_level_fk`）：

- **边缘分布不变**（同 waver/shift），只切断"手 σ ≡ 视频 σ"的耦合；
- "干净/低噪视频 + 加噪的手"从此出现在训练分布中；
- **eval 一行未改**——它是测量仪器，修的是训练侧；
- **模型形状不变**，可从旧 checkpoint 直接续训验证。

接线守护：`cosmos_framework/model/generator/fk_independent_schedule_test.py`
（11 项 AST + 配置检查，无需 GPU）。

#### 点云侧的验证结果（同款修复）

| step | train_00 | val_00 | val_04 | val_08 |
|---|---|---|---|---|
| 2500（共享 σ） | 1.53 | 3.48 | 1.93 | 5.92 |
| 3200 | 0.94 | 0.62 | 0.87 | 1.28 |
| 3600 | **0.82** | **0.53** | **0.84** | 1.14 |

单调发散**逆转**为收敛。FK 侧待用同样方式验证（见 §6.8 的 scale 已先行修好）。

---

## 7. 明确不在 v1 范围内（避免范围蔓延）

| 项 | 归到 |
|---|---|
| FK 换算到 DA3 相机系 | v2 —— ⚠️ **注意下面的修正** |
| FK 的 mRoPE 空间轴 (h,w) + 消融 | v2 |
| point + fk 同时存在 | v2 |
| 残余 8–20 px 偏差的根治 | v2（v1 接受） |
| 编码器换成 PTv3 或更复杂的结构 | 视 v1 结果，接口已预留 |

### ⚠️ 对 §1.2 的修正（2026-09-16 新查实）

§1.2 原文写：

> ~~纵向同理，`fy_real/fy_da3 = 604.41/520.21 ≈ 1.162`~~

**这是错的 —— 漏掉了 448 的纵向压扁。** 实情：

Track4World `demo.py:946-963` 把 **640×480 整幅 `cv2.resize` 到 640×448**
（不是裁剪，不丢视野），纵向压扁到 `448/480 = 0.9333`。
所以真实相机在 448 画布里的**等效**内参是：

```
fx' = 605.57              （横向不变，因为宽度没变）
fy' = 604.41 × 0.9333 = 564.12   ← 纵向焦距也被压了
cx' = 324.50
cy' = 238.26 × 0.9333 = 222.37
```

于是两个方向的比例**不一样**：

| 方向 | 真实 / DA3 | |
|---|---|---|
| 横向 | 605.57 / 520.21 = **1.164** | |
| 纵向 | 564.12 / 520.21 = **1.084** | ← 原文写的 1.162 是错的 |

**差 7.4%**，意味着 FK 的点**不能只乘一个统一系数**去对齐点云，横纵要分开。
（同样的道理：点云里的手不是等比的真手，而是横着比竖着多胖 7.4%。）

**另外两点同批查实：**

- **光心不重合**：真实 448 画布 (324.50, 222.37) vs DA3 (320, 224) —— 横向差 4.5 px。
  这是**平移**，乘系数吸收不掉。
- **DA3 用的是一套写死的默认内参**：`intrinsics.npy` 里 1192 帧
  `fx/fy/cx/cy` 一个数都没变，且 `cx/W = cy/H = 0.5` **正好是画布正中**。
  它从没真的看过这台 D435 的参数。

> v1 不做这一跳，所以不影响 v1；**但 v2 必须按修正后的值实现**。

---
---

*本文件位于 `WorldAct-cosmos3-edge-droid-sft_mano/docs/`（mano worktree）。*
