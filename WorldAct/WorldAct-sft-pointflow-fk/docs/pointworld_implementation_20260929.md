# PointWorld 实现全解(代码走读笔记)

> 调研对象:`/mnt/afs/PointWorld`(NVlabs/PointWorld,arXiv 2601.03782),2026-09-29。
> 动机:我们的 pointflow 模态目前"每个点一个 token"进 Cosmos DiT 主干,新数据点太多放不下;
> 调研 PointWorld 的"簇 token + 映射回点"做法是否可移植。
> **核心结论先行**:PointWorld 并没有"簇 token 进主干 + learned decoder"的两段式结构——
> 它是 PTv3(Sonata 血统)U-Net,全点进、全点出,中间靠体素池化压深层 token 数、
> 靠序列化 patch attention 压单层复杂度;"映射回点"靠池化时存的 assignment 索引广播(零学习)。
> 但它验证了我们要移植的两件事,见 §12。

## 0. 全流程形状图

PointWorld 做的事:**给 t=0 时刻的场景点云(每点带特征),以及未来 11 帧的机器人轨迹
(表示成点云),预测每个场景点未来 10 步的 3D 位移**。一次前向出全部步,不是自回归。

```
场景点 (B, 12000, 3+特征)  ┐
                           ├→ 拼成 ~17500 个点的大点集 → PTv3 U-Net → 逐点特征 (B,12000,256)
机器人点 (B, 11, 500, 3+特征) ┘                                    ↓
                                              逐点 MLP head → (B,11,12000,3) 位移
```

关键直觉:它**从头到尾都在"逐点分辨率"上工作**,只是在网络中间把点临时聚成簇降低算力,
出网络前再原样散开。没有"一个簇 token 负责预测一簇点"的瓶颈。

## 1. 一个训练样本长什么样(数据侧)

`pointworld/base.py:440-445` 的 unpack 展示了一个样本的全部字段:

```python
scene_coord0 = data_dict["scene_flows"][:, 0]      # (B, 12000, 3)  t=0 场景点坐标(真实轨迹的第0帧)
scene_feat0  = data_dict["scene_features"][:, 0]   # (B, 12000, Ds) 每点原始特征(颜色/法线/gripper_open/dist2robot)
scene_exists0 = data_dict["scene_exists"][:, 0]    # (B, 12000)     哪些点是 padding
robot_coord_seq = data_dict["robot_flows"]         # (B, 11, 500, 3)  未来11帧机器人表面点(FK算出来的)
robot_feat_seq  = data_dict["robot_features"]      # (B, 11, 500, Fr)
robot_exists    = data_dict["robot_exists"]        # (B, 11, 500)
```

三件事:

- **action 不是向量,是"点"**。机器人未来 11 帧的关节角被 URDF FK 转成 500 个表面点的
  3D 坐标(`robot_sampler.py`)。场景点靠 attention"看见"未来的机器人点,就知道动作要
  干什么——这就是 action 条件。
- **场景点只有 t=0 一帧进模型**,未来轨迹是监督标签不是输入。
- `scene_flows` 的 t≥1 帧在数据管线里被替换成 t-1 的复制(模型看不到未来,
  `dataset_components/transforms.py:381-409`),GT 相对位移 = `gt_flows[t] - gt_flows[0]`
  (`transforms.py:805`)。

### 为什么场景点不需要 time step 维?

1. **信息合法性**:推理时(部署时)只有 t=0 的场景观测可拿。场景的未来位置正是要预测
   的答案——当输入就是泄漏。
2. **机器人点为什么有 11 帧**:机器人的未来轨迹是**合法输入**——那是机器人自己的动作
   指令(遥操作/policy 的关节轨迹),部署时同样拿得到。它是"条件",不是"答案"。
3. **为什么机器人点需要 time embedding、场景点不需要**:11 帧的机器人 token **同时**存在
   于同一个 attention 集合里,相同构型在不同帧必须可区分("机器人 t=0 在哪" vs
   "机器人 t=10 要到哪"),所以加 sin-cos 时间嵌入 + 类型嵌入。场景点全部来自同一时刻,
   没有时间歧义。
4. **时序输出也不靠 token**:未来 10 步是 head 的**输出维度**(`Linear(128, 3×10)`),
   一个点的 token 一次性吐出全部 10 步位移。时序建模由"场景点 attend 不同时间嵌入的
   机器人点"承担。
5. 对我们的 pointflow 的对照:我们目前每个点在每个 chunk 帧上有带 (t,h,w) 位置的 token;
   PointWorld 证明"点 token 只代表 t=0,时间作为输出维度展开"也可行,token 数少一个 T 倍。

## 2. 模型总装:`PointWorldModel.forward`(`pointworld/base.py:434-487`)

四步:

```python
# (a) 场景特征编码(见 §3)
scene_feat0 = self.encode_scene_features(data_dict)          # (B, Ns, 256)

# (b) 机器人特征(见 §4)
robot_feat = robot_raw + time_emb + self.robot_type_emb      # (B, 11, 500, 256)

# (c) 主干预测(见 §5-§9)
out = self.dynamics_predictor(scene_coord0, scene_feat0, ...,
                              robot_coord_seq, robot_feat, ...)

# (d) 反归一化 + 残差加成绝对坐标
pred = self.unnormalize(out["pred"])                    # 每个 timestep 有独立的 mean/var
out["scene_flows"] = scene_coord0.unsqueeze(1) + pred   # (B,11,Ns,3) = t=0坐标 + 逐步位移
```

输出归一化是 **per-timestep** 的:每个预测步有自己的 mean/var(存
`stats/<domain>/norm_stats.json`,以 buffer 进模型,`base.py:381-386`;
读取逻辑 `pointworld/norm_stats.py:132-168`)。因为第 1 步的位移量级和第 10 步差很多,
共用一个统计会偏心。

## 3. 场景特征:冻结 DINOv3 投影 + 原始特征(`scene_featurizer.py:366-394`)

```python
backbone_scene_feat0 = self.scene_encoder(scene_coord0, scene_exists0, camera_data)
    # DINOv3 ViT-L 冻结:每个3D点投到图像,取 layers 4/11/17/23 的 2D 特征双线性采样
scene_feat0 = torch.cat([
    self.scene_encoder_norm(backbone_scene_feat0),             # 图像侧特征 (B,Ns,256)
    self.scene_raw_norm(self.scene_raw_feat_proj(scene_feat0)) # 几何侧特征(颜色/法线/dist2robot) (B,Ns,256)
], dim=-1)
return self.scene_proj(scene_feat0)   # 512 → 256
```

**每个场景点的输入 token = 256 维向量**,一半是"它在图像上长什么样",一半是
"它的几何/语义属性"。坐标 xyz 本身不进 feat(坐标只用来聚类和排序,见 §7/§8)。

## 4. 机器人特征:时间嵌入三件套(`base.py:467-473`)

```python
robot_raw = self.robot_proj(robot_feat_seq)                 # (B,11,500,256) 特征投影
time_emb  = self.time_embed(self.time_steps.view(1, 11))    # sin-cos 多周期 + MLP 的时间嵌入
robot_feat = robot_raw + time_emb + self.robot_type_emb     # + 可学习的"我是机器人"类型嵌入
```

两个可照搬的设计:①同一团点在不同帧靠**时间嵌入**区分;②**类型嵌入**让模型知道
哪些 token 是机器人、哪些是场景。(`pointworld/embeddings.py:32-42`)

## 5. DynamicsPredictor:把世界表示成"一袋子点"(`base.py:242-265`)

整个模型最核心也最简单的一步:

```python
coord = torch.cat([scene_coord0, robot_coord_seq.reshape(B, 11*500, 3)], dim=1)  # (B, ~17000, 3)
feat  = torch.cat([scene_feat0,  robot_feat.reshape(B, 11*500, 256)], dim=1)     # (B, ~17000, 256)

data_dict = {
    "coord": coord[exists],     # (ΣN, 3)   所有 batch 的点拉平成一长串
    "feat" : feat[exists],      # (ΣN, 256)
    "batch": batch[...],        # (ΣN,)     每个点属于哪个样本
    "grid_size": 0.015,         # 体素边长 1.5cm
}
point = self.predictor_model(data_dict)   # PTv3 U-Net
```

**场景点、现在和未来的机器人点,全部平等地成为点 token**,进同一个 transformer。
未来信息(动作)就这样通过 attention 泄露给场景点。

## 6. PTv3 的 "Point" 是什么

PTv3 的每一层操作的不是普通 tensor,而是一个叫 `Point` 的字典对象
(`ptv3/structure.py`),里面始终装着:

```
feat       (ΣN, C)   每点特征 —— 网络真正在算的东西
coord      (ΣN, 3)   每点坐标 —— 不算特征,只用于聚类/排序
batch      (ΣN,)     样本归属
grid_coord (ΣN, 3)   coord / grid_size 的整数体素坐标
serialized_order / serialized_inverse  序列化排序表(见 §7)
pooling_inverse / pooling_parent       池化时存的 assignment 和父层指针(见 §8/§9)
```

## 7. 序列化:把 3D 点排成一维队列(`ptv3/structure.py:48-88`)

PTv3 不做全局 attention(太贵),而是把点**按空间填充曲线排序**后切成 256 个一段,
段内做 attention:

```python
self["grid_coord"] = (coord - coord.min(0)) / grid_size   # 整数体素坐标
code = [encode(self.grid_coord, self.batch, depth, order=o) for o in order]
# order = ('z', 'z-trans', 'hilbert', 'hilbert-trans') 四种曲线
```

z-order / Hilbert 曲线的性质:**空间上接近的点,排完序后位置也接近**。所以
"256 个连续点一段"≈"空间中相邻的一团点"。每个 Block 轮换一种排序
(`order_index = i % 4`),不同层看到不同的邻居划分,弥补单一排序的边界问题。

attention 本体(`ptv3/ptv3.py:187-196`):

```python
qkv = self.qkv(point.feat)[order]          # 按曲线排序
feat = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=256, ...)  # 段内 attention
feat = feat[inverse]                       # 排回原来的点序
```

扛 17.5k token 的手段之一:**复杂度从 O(N²) 变成 O(N·256)**。位置感不靠 RoPE/RPE
(显式 `enable_rpe=False`,`base.py:198`),靠每个 Block 里一个 3×3×3 稀疏卷积(CPE)
和排序本身。

## 8. 簇的形成:GridPooling(`ptv3/ptv3.py:368-443`)

每过一个 stage,把点按**大一倍的体素网格**聚成簇。用小例子讲:

假设某 batch 有 8 个点,体素化后落在 3 个网格里:

```
点编号:    0  1  2  3  4  5  6  7
所属簇:    0  0  0  0  0  1  1  2      ← cluster(就是 pooling_inverse)
```

代码做的事:

```python
grid_coord = grid_coord // self.stride          # stride=2,网格边长翻倍(0.015→0.03→…→0.96m)
grid_coord, cluster, counts = torch.unique(grid_coord | batch<<48,
                                           return_inverse=True, return_counts=True)
#   cluster: (8,) = [0,0,0,0,0,1,1,2]   —— 每个点属于哪个簇,这就是 assignment
feat = segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce="max")   # 簇特征 = Linear 后 max-pool
coord = segment_csr(point.coord[indices], idx_ptr, reduce="mean")           # 簇坐标 = 成员均值
point_dict["pooling_inverse"] = cluster     # ← 存 assignment,decode 要原样散开
point_dict["pooling_parent"]  = point       # ← 存父层,skip 连接要用
```

池化后 `Point` 对象里的 8 个点变成 3 个"簇点",继续往下走更深的 stage。7 个 stage、
6 次池化后,17.5k 点 → 约 270 个簇,网格 0.015m → 0.96m(配置
`ptv3/ptv3_arch.yaml:13-23`,通道 [256,256,256,384,384,512,768])。**深层在
"簇 token"上工作,这是省算力的核心**;但这只是网络内部的事,输入输出都是逐点的。

## 9. 映射回点:GridUnpooling(`ptv3/ptv3.py:470-484`)

U-Net 下到最粗层后开始上采样。还是刚才的例子:decoder 某层有 3 个簇的特征
`F = [f_A, f_B, f_C]`(每个 256 维),要恢复成 8 个点:

```python
parent  = point.pop("pooling_parent")    # 池化前的 8 个点(encoder 同分辨率的特征)
inverse = point.pooling_inverse          # [0,0,0,0,0,1,1,2]

parent = self.proj_skip(parent)                                # skip:encoder 特征投影
parent.feat = parent.feat + self.proj(point).feat[inverse]     # F[inverse] = [f_A,f_A,f_A,f_A,f_A,f_B,f_B,f_C]
return parent
```

`feat[inverse]` 就是"广播":**每个点拿回自己簇的解码特征**,再加上它自己在 encoder
同层的特征(skip)。没有任何学习参数负责"猜簇内差异"——簇内差异由 skip 连接里
encoder 保留的逐点特征补上。恢复点数后再过几个逐点 Block(`dec_depths` 个,也是
patch attention)继续细化。6 次 unpool 后,`point.feat` 回到 (Σ17000, 256)——
**进多少点,出多少点**。

## 10. 出口:FiLM 调制 + 逐点 MLP head(`pointworld/base.py:267-329`)

主干出来的逐点特征,加两个调制项后过 head:

```python
scene_feat_output[scene_exists0] = point.feat[scene_mask]        # (B, 12000, 256) 只取场景点

# (a) 输入特征的 FiLM skip:γ·feat₀+β,让网络"记得"原始观测
skip_modulated  = scene_feat0 * self.skip_film_gamma + self.skip_film_beta

# (b) 机器人全局摘要 FiLM:全部机器人点 max-pool 成 (B,1,256),广播调制每个场景点
robot_modulated = robot_global_summary * self.robot_film_gamma + self.robot_film_beta

padded_scene_feat = scene_feat_output + skip_modulated + robot_modulated

dynamics = self.dynamics_head(padded_scene_feat)   # MLP(256→128→128) + Linear(128, 30)
                                                   # → (B, Ns, 10步×3坐标),最后一层小初始化
```

**每个点独立地、一次性地输出自己未来 10 步的 (Δx,Δy,Δz)**。前面补零帧(t=0 位移
为 0),最后 `scene_flows = coord0 + pred`。dynamics head 最后一层 kaiming 小初始化 +
bias 置零(`base.py:218-221`),保证训练初期预测接近恒等。

## 11. Loss:点级稠密监督(`pointworld/losses.py:35-62`)

```python
error_term = huber(output_norm, gt_target_norm)        # (B,11,Ns,3),δ=5
per_dim_loss = 0.5 * (error_term / var + w·log_var)    # 异方差 NLL:不确定的点允许误差大,
                                                       # 但 log_var 本身被惩罚,防止作弊
dynamics_loss = (per_point_loss * weights).sum()       # weights: moved 点权重大
```

- 监督范围:**所有存在且可见的点**(不是采样 query),`~context & exists & supervised`
  (`losses.py:125-131`;DROID 的 supervised = 可见性 & 深度有效,
  `dataset_components/transforms.py:828-835`);
- `weights`:按该点 GT 位移是否超过 5mm 做 sigmoid 软加权(τ=0.005m, scale=5,
  `dataset_components/collate.py:250-275`、`utils.py:124-141`)——静态点权重低,
  防被海量静止背景淹没;GT 相对位移 clip 到 ±0.5m(`transforms.py:842`);
- `log_var` clamp 在 [1e-6, 1e2],首帧 bias 初始化为 log(0.005²)=5mm 的不确定度
  (`base.py:225-230`)。

## 12. 与我们 pointflow 方案的对应关系

PointWorld 里**根本不存在**"主干只吃簇 token"的层——它是全点进、全点出,中间池化
省算力。但它验证了我们要的两件事:

1. **"簇特征广播回点 + 补每点局部信息 + 逐点 head"是够用的**。我们的版本把"主干"
   从全点换成只吃 K 个簇 token(Cosmos DiT 放不下上万点),decode 端用同样的
   `assignment 广播`;它的 skip 来自 encoder 同层逐点特征,我们对应的是
   **Sonata level-0 逐点特征 + 相对簇中心偏移**。decode 端有逐点分辨能力,这正是
   旧 cluster 模式(单 token 直接回归全簇坐标)缺的东西。
2. **action 物理化成点 + 时间/类型嵌入 + FiLM** 这套条件注入方式(`base.py:289-312,
   467-473`),和我们 FK 模态的设想(URDF+关节角→关键点 3D 坐标)同构,可直接参考。

训练侧可照搬的决策清单:

| 决策 | 具体做法 | 出处 |
|---|---|---|
| 监督粒度 | 点级稠密,Huber(δ=5) + 异方差 NLL(log_var head) | `pointworld/losses.py:35-56` |
| 点权重 | moved 软加权 sigmoid(\|Δ\|, τ=5mm),GT 位移 clip ±0.5m | `dataset_components/collate.py:250-275` |
| 子采样 | t=0 体素(0.015m)采样贯穿全序列保持对应,场景点硬上限 12000 | `dataset_components/transforms.py:122-223, 748-783` |
| 输出归一化 | per-domain、per-timestep mean/var,buffer 入 ckpt | `pointworld/norm_stats.py:132-168` |
| head 初始化 | 小缩放初始化,残差式 xyz_t = xyz_0 + Δ_t | `pointworld/base.py:218-221` |
| 训练配方 | AdamW lr=1e-4 / wd=0.01 / clip 5.0 / bf16,无 scheduler;三级 NaN 检测 + DDP 同步跳 batch | `training/trainer.py:438-510, 644-652` |
| 评测 | moved/static 分开报 L2 + confidence-filtered 主指标,seed=42 确定性 | `pointworld/metrics.py:60-119` |

## 附:关键数字汇总

| 项 | 值 | 出处 |
|---|---|---|
| 场景点数上限 Ns | 12000 | `arguments.py:112` |
| 机器人点数/帧 Nr | 500(共 T=11 帧) | `arguments.py:110`, `base.py:32-33` |
| 主干宽度 D | 256(predictor_dim) | `arguments.py:158` |
| 网格大小 | 0.015 m,stride 2 ×6 | `arguments.py:146`, `base.py:145-150` |
| attention patch | 256 点/组,4 种序列化 order 轮换 | `arguments.py:157`, `ptv3/ptv3.py:636` |
| base 结构 | enc [4,4,4,8,8,12,4] × 通道 [256,256,256,384,384,512,768];dec [2,2,2,2,2,2] | `ptv3/ptv3_arch.yaml:13-23` |
| drop_path | 0.3 | `base.py:37` |
| head | MLP(256→128)→Linear(128, 3×10),小初始化 | `base.py:217-221` |
| 输出 | (B, T=11, Ns, 3) 位移 + (B,T,Ns,1) log_var | `base.py:315-329` |
| batch_size / lr | 22 / 1e-4 常数 | `arguments.py:90`, `training/trainer.py:644-652` |
