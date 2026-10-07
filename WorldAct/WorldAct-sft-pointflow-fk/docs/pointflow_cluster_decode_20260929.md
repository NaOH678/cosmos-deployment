# 簇 token + 点级 decode:方案与实现记录(2026-09-29)

> 状态:代码已实现并通过 CPU 测试,GPU smoke 未跑。
> 前置阅读:[PointWorld 实现全解](./pointworld_implementation_20260929.md)、
> [主线实验记录](./pointflow_per_point_tokens_20260919.md)、
> [数据管线手册](./pointflow_data_pipeline.md)。

## 1. 动机

per_point 模式(一点一 token)在 300/500 点已验证可拟合,但进 DiT 的 token 数是
`N×(1+H/q)`(500 点 = 4500 token),新数据(9.24,点间距 ~1.5mm)密到几千点/帧,
token 预算直接爆炸。方向:**K 个簇 token 进 DiT(与 video/action 联合去噪,承担
"点 ↔ 视频/动作"桥梁),点级细粒度放在 DiT 外的 decode 侧补回**。

三个方案的谱系(调研结论):

| | 点进主干? | decode | 出处 |
|---|---|---|---|
| PointWorld | 全点进(内部体素池化) | encoder skip + 点级 block + 逐点 MLP | 扛 17.5k token |
| **本方案** | 只进 K 个簇 token | 广播 + skip 特征 + 点级 block(可选) | K×9 token |
| EgoWAM | 完全不进 | cross-attn 头(query=XYZ 投影) | 点分支不可 rollout |

关键证据:PointWorld 的成功件是 decode 侧的 skip + 点级 block(点级通路从未断);
EgoWAM 证明 decode 侧 cross-attn 头即使主干没见过点也能学会 flow——瓶颈主要在
decode 能补,不在主干。

## 2. 方案

```
N 点(500~数千)
  │  sonata encoder(stage 可调,产出簇特征 + assignment + 多分辨率逐点特征)
  ▼
K 个簇 token ──► Cosmos DiT(video/action/point 联合去噪)
  │                │
  │                ▼
  │           输出 K 个簇 token
  │                │ assignment 广播(现有,pointflow_codec.py)
  ▼                ▼
skip:多分辨率逐点特征 ──► concat[簇token; skip; Δxyz/Δuv; noisy; σ]
                           ▼
              点级 transformer blocks ×N(可选,新增)
                           ▼
              逐点 MLP head → 每点位移(监督/loss/scale 全部不变)
```

两个实验:

- **baseline**:现有 cluster 模式(`POINTFLOW_TOKEN_MODE=cluster`),零模型改动,
  与 per_point-500 同选点同 scale 做 A/B,回答"簇 token 瓶颈有多疼";
- **加强版**:baseline + skip 升级 + 点级 block,回答"decode 侧补回点级通路后能不能
  追平/反超"。

## 3. 实现(2026-09-29,全部 env 默认关,旧行为零变化)

### 3.1 点级 decode block(新)

- 新文件 `cosmos_framework/model/generator/pointflow_point_decoder.py`:
  `PointDecoder` = in_proj(fused→dim) + anchor xyz 编码 → N 个 pre-LN
  transformer block(帧内按样本分段的自注意力,ragged 安全)→ LN + 线性 head,
  **输出投影零初始化**(起步近似恒等,PointWorld/EgoWAM 同款惯例);
- `pointflow_codec.py`:decode 两条路——`point_blocks>0` 走点级 block,否则原
  2 层 MLP(两条路只构造其一,避免 DDP/FSDP unused params);`point_offsets.tolist()`
  只在启用时算(避免无谓 device sync);
- 注意力语义:每个 motion block(H/q=8 组)内,按 `point_offsets` 分样本做全注意力,
  **不串样本**(有测试覆盖)。

### 3.2 skip 升级:多分辨率逐点特征(新)

- `pointflow_geometry.py`:新纯函数 `voxel_level_features(levels, k)`——沿 sonata
  pooling 链(`pooling_inverse`)把第 k 层特征 gather 回体素分辨率;
  `SonataGeometryEncoder(skip_levels=(1,2,...))` 在 cluster / per_point / 空 batch
  三条路径都发射 `voxel_skip_features`(空 batch 接现有零梯度路径);
- codec 把 skip 特征按 `original_to_voxel` gather 后与 level-0 特征拼接进 local;
  缺字段/数量不匹配报 ValueError。

### 3.3 token 预算 cap(2026-09-29 先行修复,必修)

`pointflow_batch.py:pointflow_token_upper_bound` 原来按 N×(1+H/q) 预留 packing
预算;N=几千时虚高 1~2 个数量级,直接压垮 batch。cluster 模式下封顶
`POINTFLOW_CLUSTER_TOKEN_CAP`(默认 1024)。

### 3.4 env 清单

| 变量 | 默认 | 作用 |
|---|---|---|
| `POINTFLOW_TOKEN_MODE` | `cluster`(代码默认) | `cluster` / `per_point` |
| `POINTFLOW_SONATA_STAGE` | 3 | 簇化层级,见 §4 实测选 1 |
| `POINTFLOW_CLUSTER_TOKEN_CAP` | 1024 | cluster 模式 packing 预算封顶 |
| `POINTFLOW_DECODE_SKIP_LEVELS` | (空) | 如 `1,2`:追加 level-1/2 逐点 skip 特征 |
| `POINTFLOW_DECODE_POINT_BLOCKS` | 0 | 点级 transformer 层数(>0 启用) |
| `POINTFLOW_DECODE_POINT_DIM` | 256 | 点级 block 宽度 |
| `POINTFLOW_DECODE_POINT_HEADS` | 4 | 点级 block 头数 |
| `POINTFLOW_DISPLACEMENT_SCALE` | (0.0740 硬编码) | scale 环境覆盖(实验配置里) |

启动示例(加强版):

```bash
POINTFLOW_TOKEN_MODE=cluster POINTFLOW_SONATA_STAGE=1 POINTFLOW_CLUSTER_TOKEN_CAP=128 \
POINTFLOW_DECODE_SKIP_LEVELS=1,2 POINTFLOW_DECODE_POINT_BLOCKS=4 \
OUTPUT_ROOT=<新目录> NNODES=2 bash examples/launch_pointflow_sandwich101.sh
```

## 4. 关键事实(认知修正,别再用旧印象)

1. **scale 跟选点、不跟 token 模式**:两种模式目标都是逐点位移 [H,sum(N),3]
   (`pointflow_training.py` 无 token 模式概念)。同一选点(分层500+守卫)→ 同一
   scale 0.0528,两模式 ADE 直接可比。
2. **窗口缓存与 token 模式无关**:`build_pointflow_window_cache.py` 的
   cache_manifest config 只有选点参数,不含 token mode;cluster 模式读同一份缓存,
   分层配额/幻影守卫照常生效。
3. **stage-3 默认值在新数据上太极端**(500 点集中在手/物体/台面小区域,窗口缓存
   实测 24 窗口):

   | stage | 体素 | 簇数 p10/p50/p90 |
   |---|---|---|
   | 0 | 2cm | 53/60/72 |
   | 1 | 4cm | 27/30/36 |
   | 2 | 8cm | 12/16/20 |
   | 3 | 16cm | 5/9/11 |

   建议 **stage 1**(~30 簇,×9=270 token,仍是 per_point 的 1/16);stage-3 的
   ~81 token 还会让 `POINT_GRID_MIN_TOKENS=256` 的 canvas 校验静默跳过。
   sonata encoder 共 5 层(stage 0~4),stage 4 只剩 ~2 簇,无使用价值。
4. **checkpoint 不通用**:geometry 特征维度不同(per_point=32 / cluster=64~256),
   切模式必须从头训、新 OUTPUT_ROOT;加强版的点级 block 是新权重,同样新 root。
5. **decode 早就是广播+逐点 MLP**(`pointflow_codec.py` 的 broadcast 行),旧 cluster
   失败病灶在 encode 端池化 + 缺点级 decode block,不在"没有广播"——本次加强版
   补的正是后者。

## 5. 验证状态与风险

- CPU 测试:新增 `pointflow_point_decoder_test.py` 8 例全过(gather 链、零初始化、
  梯度回流、ragged 样本隔离、默认路径不变、skip 校验);pointflow 全量 64 过,
  6 个失败均为本改动之前已存在(5 旧桩 + 1 新集群环境,stash 验证与本次无关);
  ruff check/format 全绿。
- **GPU 未验证**:首次启动先单卡 smoke 几十 iter(显存、loss 形状、eval 首帧),
  再上多机。
- 风险:①点级 block 是逐样本 Python 循环 SDPA(实现简单优先),N 大/B 大时若
  成瓶颈再换 varlen fused kernel;②加强版零初始化 head 起步输出恒零,前期
  pointflow loss 曲线读数与老 run 不可比,看 ADE/zero 基线。

## 6. 实验计划(判据先行)

| run | 配置 | 回答的问题 |
|---|---|---|
| A(已有) | per_point, 500 点 | 对照组 |
| B | cluster stage1, 500 点 | 簇 token 瓶颈有多疼(与 A 唯一变量=token 模式) |
| C | B + skip(1,2) + 点级 block×4 | decode 补回点级通路后的收益 |
| D | cluster stage1, N=4000 | 密集红利(scale 需重扫、缓存需重建) |

判据:训练 ADE 破 zero 基线的幅度、eval joint rollout 的逐点方向余弦(旧 cluster
病灶)、高 σ 段是否破 zero 基线。B/C 同 scale 0.0528;D 必须重扫 scale +
重建窗口缓存。
