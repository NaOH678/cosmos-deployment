# PointFlow 形态对比观测档案(2026-10-01)

> 目的:把 per-point(v2)/ cluster-origin(B)/ cluster+skip+pb(E2s、E2)/ framescale(E1)
> 四个形态的全部观测数据、架构事实和未解问题集中落档,供后续分析(人或 agent)直接引用。
> 所有数据均可按文末"数据位置"节复核。指标口径:ADE = all_ade_mm(越小越好),
> drift = static_drift_mm(静态点漂移,越小越好),zero = 零位移基线(预测全不动的误差)。

## 1. 实验配置

| run | token 形态 | stage | skip | pb | cap | scale | 硬件 | iter 速度 | 最终 step |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| v2 | per_point(500 点) | —(level0) | — | — | — | 0.0528 标量 | 16卡 HSDP bs32 | 11.07 s | 9200(手动停,平台化+回退) |
| B | cluster 原始池化 decode | 1 | 无 | 无 | 1024(默认) | 0.0528 | 8卡 bs16 | ~5 s | 10000(跑满) |
| E2s | cluster + skip(1,2) + 点级 block×4 | 1 | 1,2 | 4 | 128 | 0.0528 | 8卡 bs16 | 4.9 s | 6800(停,被 E2 取代) |
| E2 | 同 E2s(16卡全量) | 1 | 1,2 | 4 | 128 | 0.0528 | 16卡 bs32 | 5.03 s | 进行中(@3000) |
| E1 | per_point + per-frame scale 向量 | —(level0) | — | — | — | 96 值逐帧向量 | 16卡 bs32 | ~11 s | 进行中(@5100) |

注意:E2s/B 是 8 卡(global batch 16),v2/E1/E2 是 16 卡(batch 32)——等 iter
对比时 batch 差一倍;实测 cond 口径对 batch 不敏感(E2 vs E2s 等 iter 几乎无差),
joint 后段可能敏感。

## 2. 架构事实(分析前提,均有代码出处)

- **簇 = sonata encoder 的空间网格池化,不是语义聚类**。5 级层级,
  channels (32,64,128,256,512),`stage∈0..4` 选用第几级网格当簇:level 0 最细
  (voxel 级,grid 0.02m),stage 4 最粗。stage↑ ⇒ 簇更大 ⇒ 簇内运动混合更严重,
  "用更深 stage 让簇更同质"方向相反。见
  `cosmos_framework/model/generator/pointflow_geometry.py:80-168`。
- **per_point 模式**用 level-0 特征、每原始点一个 token(不经过池化),即事实上的
  最细粒度,代价是 token 数 = 点数(500)。`pointflow_geometry.py:87-89,156-160`。
- **CLUSTER_TOKEN_CAP 只是 packing 预算上限**,不改实际簇数;默认 1024 "高于任何
  扫描过的窗口的 stage-3 簇数"。`cosmos_framework/data/pointflow_batch.py:56-73`。
- **skip_levels=(1,2)**:把 level1/2 的细粒度 voxel 特征绕过池化直送 PointDecoder;
  **pb=4**:decoder 内点级 transformer block 数。两者都只增强 decode 表达能力,
  不改变 token 形成(池化)阶段的信息损失。
- 选点:strat500(top-n 500、region quotas 2:0.40/3:0.45/4:0.15、mvm3、mvs16、
  phantom-guard),缓存 `…/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows/`
  (配置指纹见其 `cache_manifest.json`)。**配额按 region,无 motion 分层**。
- **换 stage 不需要动数据**:缓存落盘的是 level-0 voxel(选完 500 点后重新
  voxelize 的产物,`pointflow_window.py:411-419`),stage 是模型 forward 时沿池化
  链选层级;scale(按点位移统计)也与 token 化无关 → 窗口缓存、VAE latent、
  scale 全部复用,唯一代价是 geometry_dim 变(64→32)导致权重 shape 不兼容、
  不能 resume。
- **实测各级 token 数**(160 窗口,strat500 缓存):stage0 mean 59 / p90 76 /
  max 95(每 token ~8.5 点);**stage1(现役)mean 27 / p90 36 / max 46(每 token
  ~18.5 点)**;stage2 mean 13;stage3 mean 6。即现役 cluster 是"27 个 token 表达
  500 点",池化程度比直觉狠得多;stage0 只要 ~59 token(远在 cap 128 内),是
  几乎免费的 2 倍细化。
- eval case 共 14 个:train_00/01 + val_00~11,每 case 两种口径:
  cond(GT video/action 作条件,point 从噪声采样)和 joint(video/action/point 联合去噪)。
  **val_01/07/08/10 无静态点,无 drift 值。**

## 3. 全 case 对比(B@10000 / E2s@6800 / v2@6800)

### 3.1 cond ADE(越小越好)

| case | zero | B | E2s | v2 | 最短板上界 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| train_00 | 117.5 | 16.5 | 18.5 | **11.8** | v2 |
| train_01 | 8.7 | 8.9 | 8.5 | **8.4** | ≈ |
| val_00 | 28.1 | 9.1 | 10.7 | **7.4** | v2 |
| val_01 | 33.3 | **13.0** | 17.3 | 14.7 | B |
| val_02 | 26.3 | 8.5 | **7.5** | 7.8 | E2s |
| val_03 | 17.4 | 7.7 | 7.8 | **7.0** | v2 |
| val_04 | 70.0 | **7.2** | 10.7 | 7.5 | B |
| val_05 | 66.7 | 13.1 | 11.5 | **9.1** | v2 |
| val_06 | 49.0 | 9.2 | 10.3 | **8.3** | v2 |
| val_07 | 43.5 | 18.1 | 19.8 | **18.2** | ≈ |
| val_08 | 123.3 | 36.8 | 36.8 | **29.3** | v2 |
| val_09 | 26.9 | 8.5 | **7.9** | 9.1 | E2s |
| val_10 | 68.0 | 17.8 | **14.8** | 17.1 | E2s |
| val_11 | 97.5 | 20.8 | 19.6 | **15.1** | v2 |

v2 赢 8/14,大 margin 集中在**大运动 case**(train_00/val_08/val_11,zero 基线大);
E2s 只在小/中运动 case(val_02/09/10)小幅领先。B 在 val_01/04 上意外最好。

### 3.2 cond drift

| case | B | E2s | v2 |
| ---- | ---- | ---- | ---- |
| train_00 | 14.3 | 15.3 | **6.3** |
| train_01 | **2.8** | 3.8 | 5.0 |
| val_00 | 4.6 | 6.9 | **3.9** |
| val_02 | 3.8 | 3.1 | **2.0** |
| val_03 | 7.8 | **5.1** | 5.6 |
| val_04 | 4.5 | 4.8 | **3.8** |
| val_05 | 8.9 | 7.1 | **6.3** |
| val_06 | 5.5 | 4.0 | **2.9** |
| val_09 | 5.6 | 4.6 | **4.0** |
| val_11 | 6.7 | 7.6 | **6.9** |

### 3.3 joint ADE

| case | zero | B | E2s | v2 | 观察 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| train_00 | 117.5 | 15.9 | 21.1 | **12.1** | v2;skip/pb 比 B 还差 |
| train_01 | 8.7 | **6.2** | 7.8 | 6.4 | ≈ |
| val_00 | 28.1 | **14.5** | 16.0 | 16.5 | B 最好 |
| val_01 | 33.3 | **11.4** | 13.5 | 12.0 | B 最好 |
| val_02 | 26.3 | 38.6 | **28.6** | 46.5 | v2 全程锁死 44~48(>zero);E2s 破位 |
| val_03 | 17.4 | 18.2 | **14.0** | 16.9 | E2s |
| val_04 | 70.0 | **11.0** | 10.8 | 26.9 | **cluster 形态固有优势**(B 无 skip 也好) |
| val_05 | 66.7 | 33.2 | 32.4 | **30.7** | ≈,三家都差 |
| val_06 | 49.0 | **14.3** | 15.5 | 15.1 | ≈ |
| val_07 | 43.5 | 29.4 | 28.3 | **28.1** | ≈ |
| val_08 | 123.3 | 42.2 | 44.9 | **40.6** | v2 |
| val_09 | 26.9 | **15.2** | 18.9 | 12.0 | v2 最好;E2s 比 B 退化 |
| val_10 | 68.0 | 18.1 | 18.5 | **18.0** | ≈ |
| val_11 | 97.5 | 32.7 | 33.0 | **23.4** | v2;E2s 振荡剧烈(@2500 曾 22.5 最优) |

### 3.4 joint drift

| case | B | E2s | v2 | 观察 |
| ---- | ---- | ---- | ---- | ---- |
| train_00 | 9.2 | 12.2 | **6.1** | cluster 系病灶,skip/pb 加重 |
| train_01 | **1.6** | 2.4 | 1.9 | ≈ |
| val_00 | 3.0 | **2.6** | 3.4 | ≈ |
| val_02 | 3.3 | 4.1 | **2.5** | E2s ADE 大赢但漂移更差 |
| val_03 | **2.3** | 2.9 | 2.8 | ≈ |
| val_04 | **2.3** | 2.6 | 2.8 | ≈ |
| val_05 | 9.6 | 9.2 | **8.7** | 三家都高,case 本身难 |
| val_06 | 2.9 | **2.8** | 3.1 | ≈ |
| val_09 | **1.7** | 7.2 | 2.0 | **E2s 恶化 4 倍,最重要异常点** |
| val_11 | **5.5** | 7.8 | 11.2 | cluster 系反而好 |

### 3.5 E2(16卡全量)@3000 中期位置

cond @2500:10.9/9.4/8.7(val_00/02/03),等 iter 与 v2(10.2/9.8/8.5)、
E2s(10.9/10.2/8.7)持平;joint val_02 仍 47.7 未破位(E2s 在 2500~3000 破位),
val_11 已提前下到 22.1(v2 同期 43.8)。~5000 步见平台。

### 3.6 E1(framescale)@5100 关键读数

- cond 平台 7.6/8.1/8.2,与 v2 平台(7.4~7.9)打平,**cond 口径无收益**;
- **val_02_joint 4100~4300 步打到 13.3(全场唯一优于 zero 26.3 的读数)**,
  4400 后振荡回 ~33;v2 全程 44~48。framescale 的收益在 joint 口径但不稳定。

## 4. 效率账

- 同 16 卡同 batch:cluster(E2)5.03 s/iter vs per-point(v2)11.07 s/iter = **2.2×**;
- 到小/中运动 case 平台水平:v2 ≈ 246 GPU·h,E2s ≈ 74 GPU·h ≈ **3.3× 节省**;
- 定位:cluster+skip 是"3 倍便宜、~70% case 等效"的高吞吐形态;差距集中在
  大运动 case 和静态漂移,**且不随训练步数收敛**(E2s 6800 步观察)。

## 5. 已确认的观察结论

1. cond 口径:per-point 全 case 更强,**运动越大优势越大**(与池化丢信息假设一致)。
2. joint 口径 case 分化:cluster 系在 val_02/04 有优势(val_04 是形态固有,
   非 skip/pb 功劳;val_02 需 skip/pb 才破位),在大运动/val_09/val_11 退化。
3. 静态漂移是 cluster 系系统病灶:train_00(cond+joint 双口径)和 val_09(joint)
   最典型;**skip/pb 在 val_09 反而把漂移恶化 4 倍(1.7→7.2)**——增强 decode
   表达能力不等于补回池化丢失的"不动"约束。
4. skip/pb 净效果是 case 互搏,不是净赢(见 §3 各表 B vs E2s 列)。
5. cluster 系 joint 指标振荡大(val_02_joint 13.8~33 区间,val_11 @2500 22.5→
   @6800 33.0),单点读数不可靠,对比必须看轨迹。
6. val_02_joint 是 v2 的形态级缺陷(全程 >zero),E1(framescale)和 E2s(skip/pb)
   都能破,但都不稳定——两个独立改动都在 joint 口径收益更大,原因未明。
7. **[2026-10-01 新增] joint ADE 的大部分是"视频时间线发散"伪影,不是点预测误差。**
   证据(error_curve.json, val_02):GT 运动发生在窗口后半(zero 误差 8.7→174),
   而 joint 预测误差在前半就冲到 110~140、末帧回落到 ~10 ——预测"提前动了"。
   同 case cond 口径曲线完全健康(运动发生时点才被跟踪,q3/末帧 13~23,
   远优于 zero 的 84/174),证明点预测器本身时序正确。joint 下点跟随的是
   **自己生成的视频**(生成视频里手动得更早),与 GT 视频对表时中段必然虚高。
   ⇒ **joint 口径判读必须用 FDE(末帧)+ 静态漂移 + 可视化自洽性,**
   **mid-trajectory ADE 只能当参考**;"ADE 差但 FDE 好"是时间线发散的指纹。
8. **[2026-10-01 新增] E1(framescale) 可视化复核:与 v2 几乎无差别。**
   val_02_joint / val_08 cond 的 comparison.mp4 逐帧对比,两者点云形态、
   跟手时序、末端对齐肉眼不可分;val_08 两者都对大运动**欠采样式跟随**
   (预测簇滞后于 GT 簇的延伸)。framescale 没有可见收益,E1 结论维持"平台期、
   与 v2 打平",val_02_joint 的破位(~4k 最佳 16.6,之后回退)不是 framescale 的稳定功劳。
9. **[2026-10-01 新增] E2b(stage0)@2000/2100 同 step 逐 case 判读:
   运动侧真实受益、漂移侧反向恶化,均值表会掩盖这个结构。**
   与 E2s 同 step 对比(@2100 / @2000):val_02_joint fde **16.3/14.4 vs 20.5/22.1**、
   val_09_joint drift **4.3/4.7 vs 9.2/10.2**、val_00 drift **4.8 vs 7.1/6.5**、
   train_00 ade **19.6/20.9 vs 22.9/22.2**、train_01 **12.1/11.8 vs 12.8/13.7** 胜;
   val_08 cond 振荡(40.5 vs 44.8 胜 / 44.2 vs 43.4 平);**train_00 drift 16.0/17.9
   vs 15.8/15.3 负**;val_00 ade 11.5/11.7 vs 10.8/10.6 略负。
   三问判读:val_00 cond 压不到 9.1(✗);train_00 漂移到不了 v2(✗,反而更重);
   val_02_joint 在 **fde 口径**破位(✓,ade 口径因时间线发散不可判)。
   ⇒ 更细簇对"运动目标的跟踪分辨率"有真实收益,但对静态漂移零收益甚至为负——
   **漂移不是簇粒度问题,"动/静共簇绑架"假设正式否掉**(簇变细后共簇概率大降,
   漂移反而更重)。漂移机制转向 decode 注意力侧(静态点 attend 到含运动的 token)
   或生成视频侧(视频幻觉出微动、点跟随)。visual 抽查与数值互证:
   train_00_joint 的定向拖影在 stage0 中原样存在。

## 6. 未解问题(供分析)

1. **val_00 cond**:cluster 系压不下去(E2s 10.7 vs v2 7.4),且 B(9.1)比 E2s 好——
   skip/pb 在此 case 上有害?val_00 的数据特征(点数/静态占比/运动分布)是什么?
2. **大运动 case(train_00/val_08/val_11)差距不收敛**:是簇内运动混合导致位移被
   平均,还是选点侧大运动点覆盖不足(无 motion 配额)?需要按点位移幅度分层统计
   ADE 来区分这两个假设。
3. **val_09 joint 漂移 E2s 恶化 4 倍**:skip 特征是否让 decoder 过度贴合细粒度
   纹理运动而忽略静态锚定?可查 val_09 的簇构成(动/静点共簇比例)。
   **[2026-10-01 更新]** E2b(stage0)判读后,"动/静共簇"假设已否(见 §5.9):
   簇粒度不是漂移的因。新假设方向:decode 注意力侧(点 attend 到含运动的 token
   时被拖拽)或视频侧幻觉。几何—运动联合编码(池化前融合,见
   `pointflow_geometry_motion_encoding_20261001.md`)是当前押注的解法。
4. **"手到点动"现象**(历史观测):静态菜叶上的点预测跟随手运动——与 §3.4
   train_00/val_09 的漂移病灶是否同源(动/静共簇)?新数据(924 manifest)选点
   已不含菜叶点,但漂移指标仍差,说明问题不在选点而在 token 形成。
5. **E1 framescale 的 val_02_joint 破位-振荡机制**:逐帧归一化为何只在 joint
   口径产生收益、为何不稳定?
6. **E2 待观察**:5000 步平台时 (a) val_00 cond 能否压到 B 的水平(9.1);
   (b) train_00/val_09 漂移能否压到 v2 水平;(c) val_02_joint 是否复现 E2s 的破位。
   三条都不成立 ⇒ "decode 修补救不了池化"坐实,转向 token 形成阶段方案。

## 7. 候选方案池(已讨论,未实施)

| 方案 | 目标 | 改动面 | 备注 |
| ---- | ---- | ---- | ---- |
| motion-coherent 选点/配额 | 大运动覆盖 + 动/静分簇 | 数据管线 | 配合 E3a 重建缓存时一起做;不动聚类算法本身,只在选点侧加 motion band 保底 |
| 加密点数 n4000 + cap256(E3) | 簇更小更同质 | 数据 + 配置 | 已在队列(E3a/E3b),缓存构建 101 集仅 ~8 min,扫描是大头 |
| 簇内运动统计(mean/std)注入 token | 让 decoder 感知簇内分散度 | 模型 | 改 install_pointflow 特征拼接 |
| loss 大位移加权 / 静态零速加权 | 梯度再分配 | loss | 只动训练;不给模型新信息,治标(用户已否决优先级) |
| stage 0(voxel 级簇) | 更细的簇(每簇 18.5→8.5 点) | 仅配置 | **实测仅 ~59 token/窗口,远在 cap 128 内,几乎免费**;数据/缓存全复用,但不能 resume(geometry_dim 64→32) |
| stage↑(更深) | — | — | **方向错误**:stage 是空间池化层级,更深=簇更大(stage2 仅 13 token、stage3 仅 6),见 §2 |

## 8. 数据位置

| 内容 | 路径 |
| ---- | ---- |
| v2 run | `/data/shichaojian/runs/pointflow_sandwich101_strat500_v2_20260929` |
| B run | `/data/shichaojian/runs/cosmos/cluster_500p_origin_baseline` |
| E2s run | `/data/shichaojian/runs/cluster_500p_skip_pb4_smoke_20260930` |
| E2 run | `/data/shichaojian/runs/cluster_500p_skip_pb4_20260930` |
| E1 run | `/data/shichaojian/runs/perpoint_strat500_framescale_20260930` |
| eval 产物 | `<run>/cosmos3_action/action_sft/action_policy_singlerighthand_edge/pointflow_eval/step_XXXXXXX/<case>[_joint]/{metrics.json,error_curve.json}` + 可视化 mp4 |
| strat500 窗口缓存 | `/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows/` |
| manifest / allowlist | `pointflow_outputs/sandwich_924_20260928/manifest.json`、`examples/pointflow_sandwich924_all_101_episodes.txt` |
| scale 扫描 | `pointflow_outputs/scale_scan_924_strat500_perchannel_w256_20260930*.json` |
| 实验队列(状态跟踪) | `docs/pointflow_experiment_queue_20260930.md` |
| 选点/可视化工具 | `tools/build_pointflow_window_cache.py`、`tools/scan_pointflow_selection.py`、`tools/visualize_pointflow_selection.py`、`tools/rerender_dream_canvas.py` |

## 9. 主线提醒

主线是 **point + FK 模态 + (t,h,w,z) 位置编码** 三位一体的接入:point 形态对比
(per-point vs cluster)是为 FK/thwz 设计提供输入空间依据的前置工作——FK 关键点和
pointflow 轨迹共用一个 3D token 化与位置编码体系,point 形态没定稿之前,FK 侧的
token 预算和编码粒度无法最终确定。设计分析见 `docs/pointflow_position_encoding.md`。
point 侧收敛策略:E2 到平台即停 → E3(n4000 dense cluster)一把定胜负 → 定稿后
立刻切入 FK/thwz 实施。
