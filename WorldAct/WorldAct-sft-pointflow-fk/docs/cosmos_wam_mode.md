# Cosmos WAM(video + action 联合去噪)原理

**日期**:2026-09-13
**范围**:Cosmos3-Edge 单右手配方的 WAM 模式。**文中的数字来自本配方,公式是通用的** —— 换配方时按第 2 节的公式重算。
**代码归属**:引用的 `cosmos_framework/**` 均来自 NVIDIA 同步(`e723d67 Sync NVIDIA cosmos-framework main at 5e67049`);标注 `[本仓库]` 的除外。
**与其它文档的关系**:本文件讲**既有机制的原理**,不涉及 PointFlow 分支,也不涉及任何修复记录(那些见 `pointflow_bugfix_log_20260912.md`)。第 6 节解释了"训练为什么要 encode 全部 33 帧",其**工程实现**(逐窗口 latent 缓存)见 `pointflow_window_latent_cache_20260913.md`。

---

## 1. 一句话

WAM = World-Action Model:给定**当前一帧画面**和**当前关节状态**,在一个去噪过程里**同时**生成未来的画面和对应的动作。视频和动作不是两个模型,而是**同一条序列、同一个 attention 区域、同一个噪声水平 σ** 上的两组 token。

---

## 2. 时间网格

窗口是 **2.133 秒**:源帧 `r, r+2, …, r+64`(`chunk_length=32`,`source_stride=30/15=2`,共 33 帧 @15 Hz)。

| | 数量 | 每格 | 覆盖 |
|---|---|---|---|
| **video latent** | 9 | 0.2667 s | 0 … 2.133 s |
| **action token** | 33 | 0.0667 s | 0 … 2.133 s |

**4 个 action token = 1 个 video latent 间隔。**

### 公式(换配方时按这个推)

```
video latent 数   T_latent = 1 + (T_pixel - 1) // 4        # T_pixel = 33 → 9
1 个 mRoPE 单位   = temporal_compression_factor / base_fps  # 4/24 = 1/6 s
video latent k 的 mRoPE 时间 = k · tcf · (base_fps/tcf) / fps = k · 1.6 单位 = k · 0.2667 s
action token j 的 mRoPE 时间 = j · 1   · (base_fps/tcf) / fps = j · 0.4 单位 = j · 0.0667 s
```

来源:`wan2pt2_vae_4x16x16.py:1780`(`get_latent_num_frames`)、`sequence_packing/mrope.py:75`(`get_3d_mrope_ids_vae_tokens`)。action 的 `temporal_compression_factor=1`(见 `modalities.py:428-441`),所以它比 video 密 4 倍。

### 语义

- video latent 0 = 窗口**第 0 帧**;latent k 对应第 4k 个 15 Hz 帧处的画面
- action token 0 = **初始 state**;token 1..32 = 32 个控制步
- 两条都从 0 起、在 2.133 s 处结束,端点对齐

---

## 3. 训练:单步,随机 σ

```
t=0                                                      t=2.133 s
│
│  条件(干净,σ=0,不进 loss)
│    video  latent 0 = f(第 0 帧)      condition_frame_indexes_vision = [0]
│    action token 0  = 初始 state       condition_frame_indexes_action = [0]
│
│  生成(要预测)
│    video  latent 1..8    ●───●───●───●───●───●───●───●      0.267 s/格
│    action token 1..32    ●●┬●●┬●●┬●●┬●●┬●●┬●●┬●●┬●●┬●●┬●   0.0667 s/格
│
├─ 采 σ:rf.sample_train_time → [B,1],每样本一个
├─ 加噪(见下)
├─ 一次 MoT forward → v̂_vision(9 个 latent)+ v̂_action(33 个 token)
└─ loss = loss_scale × 视频 MSE + action_loss_weight × action MSE
          只在生成位置算(mse_loss_indexes 已排除条件帧)
```

条件位置的来源:`data/generator/action/transforms.py:297`(`wam` → `[0]`)与 `:337-339`(Case B:`action_length == video_length` → 只有初始 state)。

### 加噪的准确式子

`model/generator/diffusion/rectified_flow.py:208-209`:

```python
x_t     = x_0 * t + x_1 * (1 - t)      # σ·ε + (1−σ)·x0
dot_x_t = x_0 - x_1                    # ε − x0
```

> **命名陷阱**:这里 `x_0` 是**噪声**、`x_1` 是**干净数据**,和 diffusion 社区的直觉相反(代码注释里明确说了这一点)。不要看反。

条件位置通过 `× (1 - condition_mask)` 强制 σ=0(`omni_mot_model.py:1755`),所以它们原样进入、不参与 loss。

---

## 4. 推理:迭代,σ 从 1 到 0

```
输入(实时):
    当前帧   ──► f(当前帧) = video latent 0     ← 唯一真实的视频输入
    初始状态 ──► action token 0

其余从纯噪声起步:
    video  latent 1..8 = ε
    action token 1..32 = ε

采样器:unipc,4 步,guidance=3.0,shift=5.0
    (examples/deployment/cosmos_singlerighthand_protocol_v2.yaml:61-67)

    σ=1 ─► forward ─► v_video, v_action ─► 一起更新 ─┐
     ↑                                              │
     └──────────────────────────────────────────────┘ ×4

    σ=0 ─► 8 个未来 latent ──VAE decode──► 32 帧画面
           32 步 action
```

### 视频输入怎么构造

官方 `inference/vision.py:136-142`(`build_conditioned_video_batch`):

```python
video_data = torch.zeros(1, 3, num_frames, h, w, ...)
t_fill = min(t_cond, num_frames)
video_data[0, :, :t_fill] = conditioning_frames[:, :t_fill]
if t_fill < num_frames:
    video_data[0, :, t_fill:] = video_data[0, :, t_fill-1:t_fill].expand(...)   # 复制最后一帧,不是零
```

载入多少帧由条件数决定(`inference/inference.py:695-696`):

```python
num_condition_latent_frames = max(condition_frame_indexes_vision) + 1     # = 1
max_frames = tokenizer.get_pixel_num_frames(num_condition_latent_frames)  # (1-1)*4+1 = 1
```

**"要 1 个 latent → 载入 1 个像素帧"**,一步不多 —— 这是因果 VAE 给的:

```python
out, enc_cache = _run_chunk(x[:, :, :1], feat_cache=enc_cache)   # wan2pt2_vae_4x16x16.py:932
```

chunk 0 是**单帧 prime**,所以 latent 0 只由第 0 帧决定,后面的 chunk 影响不到它。

> `[本仓库]` 服务端 `inference/robot_policy/adapters.py::_build_batch` 用 `torch.zeros` 填后面 32 帧,而官方是复制第 0 帧。**当前等价** —— 生成位置的 latent 从纯噪声起步,编码出来的值不会被用到。但若要严格对齐官方,改成复制即可。

---

## 5. "联合"体现在三层

**① 一次 forward 出两个模态。** video token 和 action token 拼在同一条 packed 序列的**同一个 full-attention 区域**,互相可见;一次前向同时得到两组 velocity。

**② 共用一个 σ。**

```python
sigmas_for_action = sigmas if sigmas_action is None else sigmas_action    # omni_mot_model.py:1659
```

本配方 `independent_action_schedule: false` → `sigmas_action = None` → **action 直接用视频那个 σ**。

作用:模型不能"拿更干净的模态去猜更脏的" —— 两边永远同一噪声水平,学到的才是"同一时刻的画面与动作"的联合关系。

**③ 训练单步与推理迭代同构。**

| | σ 来源 | x0 来源 | 一次调用 |
|---|---|---|---|
| 训练 | 随机采一个 | 真值(有监督) | 一次 forward + 反传 |
| 推理 | 1→0 迭代 N 步 | 只有条件位置有 | N 次 forward,无梯度 |

同一张网络、同一套 attention、同一个 σ 定义,区别只在**起点**和**有没有监督**。

---

## 6. 为什么训练要 encode 未来那些 latent

这是最容易困惑的一点:**既然只条件于第 0 帧,为什么后面 8 个 latent 也要 encode?**

因为**它们在训练里不是"输入的一部分",而是监督目标本身**。`x0` 在训练里出现两次:

| 用途 | 式子 | 为什么需要 x0 |
|---|---|---|
| **喂给模型的输入** | `x_t = (1−σ)x0 + σε` | 要往"干净的未来"上加噪,才能造出半噪声未来 |
| **模型的监督目标** | `v* = ε − x0` | 目标速度含 x0,没有它算不出 loss |

**没有真值 → 加不了噪、算不出 loss → 模型对"未来长什么样"完全没有学习信号。**

### 推理为什么不需要

- **输入**:σ=1 时 `x_t = σ·ε + (1−σ)x0 = ε` —— x0 被**完全覆盖**,一个 bit 都不进输入
- **目标**:推理没有 loss,不需要 `v*`

推理从 σ=1 起步,之后每一步的输入都是**模型自己上一步走出来的**。

### 关键:训练的半噪声未来 == 推理的中间态

```
训练:  x0(真值) ──加噪──► x_t ──► 模型 ──► v̂  vs  v* = ε − x0
                 ↑
            σ 随机 → 见过从纯噪声到近乎干净的全谱

推理:  ε ──► 模型 ──► v ──► 更新 ──► x_{t−Δ}
              ↑                                  │
              └──────────────────────────────────┘
        走出来的中间态,正是训练时那种"半噪声的未来"
```

**所以真值必须编码** —— 训练要能造出"半噪声的未来"给模型看,而它只能从真值加噪得到。

### 这不是"作弊"

输入是**加过噪**的,模型抄不到:

| σ | 模型看到的 | 它要学的 |
|---|---|---|
| 0.9 | `0.1·x0 + 0.9·ε` | 大幅去噪 |
| 0.5 | 一半一半 | 补全部分信息 |
| →0 | 接近干净 | 微调 |

σ 随机 → 模型见到整个谱,而不是"永远给答案"。

---

## 7. 条件 vs 生成:同一个序列里的两种角色

| 位置 | 训练里 | 推理里 |
|---|---|---|
| **latent 0 / action token 0** | 干净的**输入**(σ=0,不进 loss) | 真值(实时观测)**输入** |
| **latent 1..8 / action token 1..32** | **监督目标**(加噪后当输入,真值算 loss) | 从噪声**生成** |

**前一个 token 是"输入",后面的是"答案"。** 这解释了训练/推理在数据准备上的不对称:

- 训练 encode **全部 33 帧** → 9 个 latent(8 个当目标 + 1 个当条件)
- 推理 encode **1 帧** → 只需 latent 0(条件);其余从噪声生成

同理 action:训练提供 32 步真值作监督,推理只给初始 state,其余生成。

---

## 8. 本配方的具体参数

| 项 | 值 | 位置 |
|---|---|---|
| `chunk_length` | 32(→ 33 帧观测) | 实验配置 |
| `fps` | 15 | 同上 |
| VAE 时间压缩 | 4 | tokenizer 配置 |
| `base_fps` | 24(FPS modulation 基准) | model 配置 |
| 视频 loss 权重 | `loss_scale = 10.0` | run config |
| action loss 权重 | `action_loss_weight = 10.0` | run config |
| 独立 action schedule | `false`(与视频共用 σ) | run config |
| 推理采样器 | `unipc`,`num_steps=4`,`guidance=3.0`,`shift=5.0` | `examples/deployment/cosmos_singlerighthand_protocol_v2.yaml:61-67` |
| 分辨率 | `"480"`(画布 544×736) | 同上 |

---

## 附:一张总图

```
训练(单步):
  [latent 0 │ latent 1..8]     条件 σ=0 ┊ 生成(随机 σ)
  [state    │ action 1..32]            ┊
                ↓  一次 MoT forward
       v̂_video, v̂_action
                ↓
       视频 MSE ×10  +  action MSE ×10   →   反传

推理(迭代 N=4):
  [f(帧0) │ ε×8 ]              条件 ┊ 生成
  [state  │ ε×32]
                ↓  一次 forward(第 n 步)
       v_video, v_action  →  更新  →  回到上一步
                ↓  σ=0
       8 个 latent ──VAE──► 32 帧画面
       32 步 action
```
