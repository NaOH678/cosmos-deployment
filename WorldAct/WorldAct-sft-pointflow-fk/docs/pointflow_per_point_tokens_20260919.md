# PointFlow 逐点 token(one point = one token)拟合验证 — 2026-09-19

## 动机

簇模式的 encode 把一个簇内所有点的加噪位移 `index_add` 平均池化进同一个 token
(`pointflow_codec.py` 的 `PointFlowCodec.encode`)。既有实验:`supervise_cluster_n=1`
(一簇监督一个点)可以拟合,一簇监督多个点不行——一个 token 承载了簇内相互矛盾的
监督信号。本分支验证:每点自成一 token 后,top-300 运动点全监督能否拟合。

**结论先行:能拟合。但随后发现并修复了一个更深的训练/评估条件错位问题
(point 与 video 的 σ 耦合),这是本文档的主要篇幅。**

## 改动

- `cosmos_framework/model/generator/pointflow_geometry.py`:新增
  `summarize_per_point_geometry`——恒等映射 `original_to_cluster = arange(N)`,
  逐点特征 = level-0 Sonata voxel 特征按 `original_to_voxel` 索引,`cluster_xyz/uv`
  = 逐点 anchor 值,relative 偏移为 0。`SonataGeometryEncoder(per_point=True)` 走此
  路径,`output_dim = channels[0] = 32`。codec/sequence/loss 全部按"簇"泛化书写,
  零改动:池化在恒等映射 + `counts=1` 下自动退化为逐点。
- `pointflow_branch.py` / `cosmos3_vfm_network.py::install_pointflow`:新增
  `token_mode="cluster" | "per_point"` 参数,默认 cluster(行为不变);
  install 时打印 `token_mode` 与 `geometry_dim`。
- `omni_mot_model.py`:`POINTFLOW_TOKEN_MODE` 环境变量门控,非法值报错。
- `data/pointflow_batch.py::pointflow_token_upper_bound`:改按点数估计(N≥V,两种
  模式都是安全上界)。
- `examples/launch_sft_action_policy_singlerighthand_edge.sh`:`POINTFLOW_TOKEN_MODE`
  默认 `cluster` 并 export。
- 测试:新增 `pointflow_per_point_test.py`(几何契约、无跨点池化串扰、token 拓扑、
  参数校验、预算按点数);修复 `pointflow_sampling_test.py` 的既有失败(mock 缺
  `rectified_flow_inference_config`,unipc 成为默认采样器后测试未跟上)。

## 实验设置

10 episode sandwich,单卡,top-300 运动点(`select_top_n=300`,
`min_voxel_members=3`)全监督。每样本 point token 数 = 300×(1+8) = 2700
(300 干净 anchor + 8 block × 300 noisy)。`pointflow_displacement_scale=0.0432`
(标定见下文),codec lr×25,eval 为固定 14 case(train 2 + val 12,
`fixed_cases_stages_4windows.json`),UniPC-4 主采样 + Euler-16 首轮对照。
eval 选点与训练逐比特一致:eval 回调 deepcopy 训练数据集 config 实例化
(`pointflow_eval_cases.py:93-95`),manifest、选点旋钮、种子派生全部相同。

## scale 必须跟着选点走(运维规则)

`pointflow_displacement_scale` 是"米 / 模型单位"的标定,意图是让模型空间位移
std≈1.0(与单位方差噪声同量级)。**位移分布由选点决定,所以选点(数量/方式)
一变,scale 就必须重新测量**——这是本次实验实际踩到的:0.0839 是为
`select_motion_fraction=0.05`(409 点)标定的,换 top-300 后位移更大,沿用旧值
时模型空间 std 只有 0.51,信号强度只有意图的一半,"loss 被度量噪声主导"的老
问题部分回归。

两个标定点(同一测量流程,`tools/scan_pointflow_selection.py` 与训练共享
`PointFlowSource` 种子派生):

| 选点 | std(= 应取的 scale) | n | 日期 |
|---|---|---|---|
| fraction=0.05(409 点) | 0.0839 m | 2,528,250 | 2026-09-14 |
| top-300 + min_voxel_members=3 | **0.0432 m** | 3,764,532 | 2026-09-20 |

复现(top-300;其他选点改 `--top-n`/`--fractions`):

```bash
python tools/scan_pointflow_selection.py \
  --manifest pointflow_outputs/task5/mixed_manifest.json \
  --cache-root .../singlerighthand-sandwich-100-cosmos-cache \
  --episode-allowlist examples/pointflow_sandwich_10_episodes.txt \
  --top-n 300 --min-voxel-members 3 --windows-per-episode 16
```

注意:scale 变更后旧 checkpoint 不可 resume(目标与模型学到的 scale 都变了),
必须新 OUTPUT_ROOT 重训;且跨 scale 的 loss 读数不可比,判据只看 ADE/zero。
recipe 注释(`action_policy_singlerighthand_edge.py`)里同步记录了这两个标定点。

## 现象:训练拟合与采样轨迹完全分离

两个指标讲出相反的故事。

**训练侧(训练 batch 上的一步 x0_hat ADE / zero 基线)**:一路下降,iter 670 时
~0.5(23mm vs 45-65mm),iter 2800 时 ~0.3(15-18mm vs 41-63mm)。**一点一 token
拟合 300 点全监督成立**——簇模式多点监督时这个比值压不下去(见
`pointflow_displacement_scale_20260914.md`)。

**eval 侧(UniPC-4 多步采样 ADE / zero 基线,14 个固定 case)**:单调恶化。
下表为 scale-0.0432 run;更早的 scale-0.0839 run 形态相同
(step 100:1.13-2.86 → step 600:1.71-6.83)。

| step | train_00 | val_00 | val_04 | val_08 |
|---|---|---|---|---|
| 100 | 1.05 | 1.66 | 1.51 | 2.58 |
| 2500 | 1.53 | 3.48 | 1.93 | 5.92 |

发散的形态(以 step 600 的 val_00 为例):

- 预测**首帧(+0.067s)误差即 ~150mm**,不是向终点累积——轨迹从一开始就错;
- `pred_outside_fraction`(预测点投影出画面的比例)从 12.8%(step 100)涨到
  32.2%(step 600);
- `comparison.png` 呈**直线扇形炸开**:所有点的预测位移沿各自初始噪声方向飞出,
  即"输出 ≈ 缩放后的初始噪声",与 scale 修复前记录的退化形态一致;
- **训练窗口(train_00)同样发散** → 不是泛化问题,而是评估条件本身落在训练
  分布之外。

## 根因:point 与 video 的 σ 耦合导致训练/评估条件错位

机制链条(每一环都有代码或数据佐证):

1. **训练时 point 与 video 共享同一个 σ**。`_add_noise_to_input` 的 pointflow
   分支把 vision 的 σ 直接传给 `pointflow_add_noise`(`omni_mot_model.py`)。
   因此训练样本中,"加噪的 point token"永远搭配"同一个 σ 加噪的 video token":
   σ=0.9 时 video 也是 90% 噪声,只有 σ=0 时 video 才干净。
2. **eval 的条件组合在训练分布中概率为零**。`sample_pointflow` 的设计是给干净
   GT video/action 当条件、point 从纯噪声出发去噪——"干净 video + 加噪 point",
   这个组合训练中从未出现。
3. **模型把"video 干净"当成了 σ≈0 的信号**。σ=0 时 x0_hat = x_t,即把输入
   (纯噪声)原样当答案——这正是扇形直线的来源:输出 = 初始噪声 × scale。
   显式 σ embedding 存在,但"video 噪声程度"是更强的统计相关信号,10 episode
   的小数据(action loss 同期塌到 0.001-0.005)让模型完全有余量学死这个
   虚假相关。
4. **逐点模式放大了错位**。簇模式下 token 的运动输入是簇内平均,ε 方差除以
   成员数,token 自身信噪比高;逐点模式每个 token 直接吃自己的 ε,模型被迫更
   依赖 video 上下文——而 video 的干净程度正是错位的变量。
5. **UniPC-4 的求值点恰好全压在最差区间**。shift=5 时四个速度求值点约在
   σ≈[1.0, 0.94, 0.83, 0.63];σ scan(resume 后开启)证实高 σ 档一步误差
   ≈ 甚至 > zero 基线(共享 σ 的权重、干净 video 条件下),轨迹终点全靠这几个
   点外推,所以采样崩而训练指标好(训练 σ 也在高位,但那时 video 同样加噪、
   条件匹配)。

**排除过程**:先把 scale 从 0.0839 重新标定为 0.0432(top-300 的真实位移 std,
n=3.76M,`pointflow_outputs/selection_scan_top300_20260920.json`;0.0839 是为
fraction=0.05 的 409 点标定的,沿用后模型空间 std 只有 0.51)。scale-only run
的 eval 仍单调发散(step 100:1.05-2.58 → step 2500:1.53-5.92)——uniform
rescale 不是根因,错位假设保留。

## 修复:独立 σ 调度(只改训练,eval 不动)

新增 `rectified_flow_training_config.independent_pointflow_schedule`
(`model_config.py`,仿照 `independent_action_schedule`),recipe 已开启。
机制:每个训练 step,pointflow 从 vision RF 采样器**独立**抽一个 σ
(`training_step` → `_get_train_noise_level_pointflow` →
`_add_noise_to_input(sigmas_pointflow=...)`)。

- **边缘分布不变**(同 waver/shift),只是切断 "point σ ≡ video σ" 的耦合;
  "干净/低噪 video + 加噪 point"从此出现在训练分布中;
- **eval 一行未改**——它是测量仪器;修复发生在训练侧,让 eval 的条件变成
  分布内;
- **模型形状不变**,run 可从旧 checkpoint 直接 resume 验证(iter 2500 续训);
- 测试:`pointflow_training_test.py::test_independent_pointflow_schedule_wiring`
  (AST 接线守护)与 `test_pointflow_noise_level_matches_video_marginal`
  (边缘分布行为)。

部署视角的注记:独立 σ 同时覆盖两种未来形态——point 接入 video/action 联合
去噪循环(video 同样从噪声起步)和"观测当前帧预测点运动"(video 干净),
两种条件组合都在训练分布内。

## 验证:发散逆转(iter-2500 resume,独立 σ + σ scan)

UniPC-4 ADE / zero 基线:

| step | train_00 | val_00 | val_04 | val_08 |
|---|---|---|---|---|
| 2500(共享 σ) | 1.53 | 3.48 | 1.93 | 5.92 |
| 2600 | 1.37 | 2.99 | 1.64 | 4.86 |
| 2800 | 1.20 | 1.78 | 1.14 | 2.59 |
| 3000 | 0.99 | 0.82 | 0.92 | 1.48 |
| 3200 | **0.94** | **0.62** | **0.87** | 1.28 |
| 3600 | **0.82** | **0.53** | **0.84** | 1.14 |

**收敛终态(step 5000,iter-5000 checkpoint,训练在此停止)**:全量 14 个 case
均值 0.82,9/14 破 1.0。按 stage 分组:

| stage | cases | UniPC-4 ADE/zero |
|---|---|---|
| train | train_00 / train_01 | 0.82 / 0.67 |
| early | val_00-03 | 0.48 / 0.64 / 0.38 / 0.46 |
| middle | val_04-07 | 0.81 / 0.69 / 0.67 / 1.15 |
| late | val_08-11 | 1.09 / 1.09 / 1.26 / 1.24 |

均值在 step 3900-5000 间平台(0.85→0.82):train/early/middle 早已收敛,
late 段(episode 后段、遮挡/大运动窗口)仍缓慢下降但未破 1.0,按当时速度
外推还需数千步,边际收益不足,停训。训练侧一步 ADE/zero 同期进入
~0.29 的平台——该指标在 waver σ 分布(中位 0.83)上平均,高 σ 段的
条件后验方差构成信息论下限,平台是预期而非异常。

σ=0.9 一步 scan 误差(resume 前的旧权重在该档 ≈ 或 > zero 基线):

| step | train_00 | val_00 | val_04 | val_08 |
|---|---|---|---|---|
| 2600 | 95.8 | 60.3 | 71.4 | 55.6 |
| 3200 | 67.4 | 20.7 | 44.7 | 21.5 |
| zero 基线 | 137.1 | 52.1 | 95.5 | 30.2 |

step 3200 起 σ=0.9 档**全部低于各自 zero 基线**;step 3000 起多数 case 的
采样 ADE/zero 破 1.0。**结论:训练/eval 的 σ 条件错位是采样发散的主因,
独立 σ 调度(边缘分布不变、仅解耦)修复了它。**

**全量 case 的 stage 结构(step 3900)**:难度随 stage 递增——
train 0.83/0.76;early(val_00-03)0.40-0.67;middle(val_04-07)0.72-1.17
(val_07 未破);late(val_08-11)1.11/1.12/1.41/1.24(全未破,val_10 最差)。
14 个 case 均值 0.85,10/14 破 1.0。episode 后段窗口更难,与手部遮挡加剧、
运动幅度更大(valid fraction 更低)一致;监控与后续分析应按 stage 分组,
不能只盯单个 case。

## 遗留

- late 段(val_08-11)终态 1.09-1.26 未破 1.0,是下一步的主攻对象;
  train_00(大位移 case)σ=0.9 误差 67mm,相对其 137mm 基线仍有空间;
- **下一个杠杆候选**(按预期收益/成本排序;2026-09-21 深入指标分析后已重排,
  见文末两节):
  1. ~~低 σ 加权~~——**已否决**:step-5000 的 σ-scan 显示所有 case 在 σ≤0.5 的
     单步去噪误差一致(σ=0.05 时 1.5-2.4mm),精修环节没坏;分化只在 σ≥0.7 的
     全局走向决策段。加权低 σ 治不了 late case;
  2. **history motion token**(干净的运动 state)——anchor 只给"点在哪",
     补上"点正在怎么动"(过去 q 步真实位移,σ=0 的 block token),直接针对
     方向盲区(stepCos≤0.32);改动涉及数据/codec/序列三层;
  3. 中期:point token 接入 video/action 联合去噪循环(task8 遗留的部署形态)、
     更多数据、多卡 FSDP;**前置条件**:部署形态是"video=当前观测(σ=0)、
     action 与 point 联合从噪声生成",但本 recipe `independent_action_schedule`
     仍为 False(action 与 video 共享 σ),"干净 video + 加噪 action"组合在训练
     分布中概率为零——与 point 当初的错位同病。接联合循环之前必须先开
     `independent_action_schedule`,否则 eval 发散会在 action 上重演;
- FDE 与逐点方向余弦未系统评估——之前簇模式就有"方向余弦 ~0.5、误差向终点
  累积"的问题,需在收敛后复查;
- 过拟合贡献未分离:10 episode 下错位被学死有小数据因素,独立 σ 是否同样
  帮到大数据场景待验证;
- eval 面板 `position_grid_inside_fraction=0.0` 是既有 artifact(簇模式 run
  同为 0.0,encode 时的 guard 通过、模型实际 mRoPE 位置正确),统计待修;
- euler16 对照只在首轮 eval 跑一次(`_compared` 标志,by design,
  `pointflow_eval.py:216`);
- 磁盘:checkpoint 30G/个 × save_iter 500,本 run 曾因配额在 iter ~2800 中断
  一次,注意清理。

选点的人工检查用 `tools/visualize_pointflow_selection.py`,见
[pointflow_selection_visualization_20260920.md](./pointflow_selection_visualization_20260920.md)。

## 深入指标分析(step-5000,2026-09-21)

对 step-5000 全部 14 个 case 的 `prediction.npz` 做了逐点/逐步分析(分析脚本
思路:逐步 err vs GT 运动曲线、步矢量方向余弦、幅度比、逐点 误差~运动量 回归、
逐点胜率、valid 随时间衰减)。结论:late case 的病根是**方向盲区**,不是拟合
或精修问题:

- **方向余弦**:最好的 early case 逐步方向余弦也只有 0.22-0.32(终点方向
  0.70-0.79);late case 逐步 ~0.0-0.1、终点 0.02-0.37。模型学会的是"锚点周围
  校准过的误差球 + 大幅显著运动的粗略漂移方向",没有学会逐步运动方向;
- **幅度缩水**:late case 预测幅度/GT 幅度 = 0.48-0.77(early ~0.9-1.0)——
  方向不确定时预测条件均值,多方向平均后输出小幅度无方向抖动;
- **误差结构**:early 是常数地板(回归 slope≈0,intercept 22-43mm),late 是
  比例误差(slope 0.5-1.1 ≈ 预测不动)——模型对 late 运动没有注入信息;
- **σ-scan 排除精修假设**:σ≤0.5 各 case 单步去噪误差一致(σ=0.05 时
  1.5-2.4mm);分化只在 σ≥0.7(全局走向决策段),late case 高 σ 误差 ≈ 甚至
  超过 zero 基线(val_10:56.2 vs 49.8mm);
- **点流失**:late 窗口点大量流出有效区域(val_11 t16 后 valid=0,val_10 t32
  只剩 10%)——训练侧同样的 mask 逻辑意味着 late 阶段长视界监督被截断;
- **逐点胜率**:early 0.79-0.81,late 0.41-0.51(val_11 为 0);
- 警示:12 个 val case 全部来自同一 episode,外推有限;train_00 t8 后点也全
  流失,作为训练侧探针价值有限。

## 新数据接入(labeled 29 集,2026-09-21)

`pf_out/sandwich`(见该目录 README)是同一 Track4World 3d_ff 管线的升级版:
逐帧 model mask + frame-t depth-edge 测试(修掉旧版的空洞和飞点),CPU 质量
过滤 + 首帧人工语义区域,**只交付手和被操作物体的点**(背景不写盘)。

质量对比(同一 episode_0013,新 labeled vs 旧 dense):飞点(步长>100mm/4帧)
占比 0.01% vs 0.24%(↓24 倍),max 步长 296 vs 557mm,valid 内无 NaN;窗口
尺度(66 帧)存活率 0.79-0.95;坐标系与旧数据完全一致(投影误差中位 0.12px,
锚点帧跨管线中位差 0.2mm,内参 fx=0.813/fy=1.162 跨管线一致)。eval episode
(0015_20260730_170501)不在新集内,继续是干净 held-out。

接入改动:

- `pointflow_window.py`:元数据双格式(COMPLETE.json / report.json);
  **intrinsics.npy 可选**(缺失时跳过投影守卫,intrinsics_normalized 与
  projection_error_px 置 NaN;模型与 loss 本就不用它,只有 eval 可视化用,
  eval 集仍在旧数据上不受影响);新增 `select_regions`(语义区域过滤,只读
  锚点时刻信息,可部署)与 `select_min_valid_steps`(留存守卫,motion 排名
  中降权窗口内标签存活不足的轨迹,针对点流失);
- `pointflow_source.py` / 数据集 / recipe 接线:env `POINTFLOW_SELECT_REGIONS`
  (逗号分隔,空=全部)、`POINTFLOW_SELECT_MIN_VALID_STEPS`;
  **注意:开 regions 后 legacy 集(含 eval 集)因缺 region_labels.npy 会报错,
  首轮实验不开**;
- manifest:`tools/build_pointflow_manifest.py` 生成
  `pointflow_outputs/manifest_sandwich_labeled_20260921.json`(101 集 =
  29 labeled + 7 legacy + 65 无标签);训练 allowlist 换
  `examples/pointflow_sandwich_labeled_29_episodes.txt`;
- scale 重标定(老规矩,数据+选点变了必须重测):labeled 29 集 top-300 std
  0.0751(mvs=0)/ **0.0740**(mvs=16,采用),比旧 0.0432 大 ~1.7 倍——候选
  全是手/物体点,top-300 不再被静态背景稀释。recipe 已更新为 0.0740,
  **与旧 manifest 不兼容,不可跨 scale resume**。

首轮新数据实验配置(不开 regions):`POINTFLOW_MANIFEST` 指新 manifest、
`POINTFLOW_EPISODE_ALLOWLIST` 指 29 集 allowlist、`POINTFLOW_SELECT_TOP_N=300`、
`POINTFLOW_MIN_VOXEL_MEMBERS=3`、`POINTFLOW_SELECT_MIN_VALID_STEPS=16`,
全新 OUTPUT_ROOT(不可 resume 旧 checkpoint)。

## 29 集 labeled 首轮训练:进展、工程修复与 TODO(2026-09-22)

**事实修正**(上文 2026-09-21 节的两处表述已过时):

- 本轮 eval 集不是 0015(legacy):29 集 allowlist 走 val 划分后 held-out 是
  **episode_0019_20260730_171242**(labeled),全部 12 个 val case 来自它,与旧
  run 的 case 不可逐号对比;
- labeled 交付物没有 intrinsics.npy,而 eval 预览需要投影:`pointflow_eval.py`
  的 `make_preview` 已改为 `_episode_metadata` 双格式 + **常量归一化内参回退**
  (fx=0.813/fy=1.162,实测自 efep_labeled/episode_0013,逐帧恒定;只用于
  overlay 投影,指标全在 3D xyz 空间)。**待验证**:不同集相机若不一致,预测
  overlay 会系统性偏移,拿到 eval 视频先看 pred/GT 是否对齐。

**首轮结果**(step 100→500,UniPC-4,条件拟合诊断):

- 无发散:多数 case 单调改善,pred_outside 0-1.4%(旧 run 同期 32% 并单调
  恶化至比值 1.5-6.8);约半数 case 击破 zero 基线(val_00 0.24、val_09 0.22、
  val_11 0.30);val_00 ADE 29mm vs 基线 122mm;
- 点流失大幅缓解:late case valid_fraction 0.73-0.93(旧数据 val_11 t16 后
  为 0)——labeled 跟踪持久性 + min-valid-steps=16 守卫起效;
- 异常:**train_01 比值 1.54 且仍在恶化**(训练集窗口劣于 zero 基线,反常,
  待查);val_02(1.05)/val_03(1.13)/val_08(1.29)未破 1.0;
- 注意:仅 step 500,旧 run 在 step 100 时同样看似正常、2500 才发散,结论
  需等到 step 2000+。

**本轮工程修复**(均已测,尚未 commit):

- 训练 dataloader `_load_video` LRU 淘汰 bug(shuffle 下"需要但陈旧"的帧被
  本窗口自己的插入淘汰 → KeyError,step 500 崩溃根因):插入前刷新命中帧
  recency;回归测试 `test_load_video_eviction_never_drops_frames_needed_now`
  (已验证旧代码必挂);
- 缓存工具 `cache_window_vae_latents.py`:任务钉卡(每卡独立单 worker executor,
  治"worker 跨卡拿任务 → 设备错配/OOM/挂死")、BrokenProcessPool 自动重试、
  跳过已完成集;101 集 window latents 已全量重算(batch=1 逐窗口路径,此前
  batch=8 产物已废弃;batch 差异未定性,指标疑似被近零元素放大,未定论);
- 启动/缓存脚本固化:`examples/cache_singlerighthand_window_latents.sh`、
  `examples/launch_pointflow_labeled29_sandwich.sh`(防止长命令粘贴折断)。

**TODO(按优先级)**:

1. **盯训练到 step 2500+**:确认不发散;重点看 train_01、val_02/03/08 的趋势
   和 eval 视频(判断数据问题还是模型问题);
2. **路线 B(部署形态 eval)**:把 point 接进 `generate_samples_from_batch`
   的联合去噪循环——噪声拼接 [vision|action|sound] 加 pointflow、point 的
   condition_mask 语义(anchor=1/位移=0)、各模态 σ 步进策略。接口已就位
   (PackedSequence.pointflow_noised 槽位、denoise 联合单步、
   pointflow_sampling 与主循环同 solver/shift);**前置决策**:部署形态选
   "video 一起做梦"(官方 wam,action 共享 σ 自洽)还是"观测窗口整段干净"
   (必须开 `independent_action_schedule`,否则 action 重演 point 的错位);
3. **双 eval 并存**:条件拟合诊断(现状)作隔离变量仪器,联合生成 eval 作
   部署验收;两者差距即 video/action 生成质量的度量;
4. **history motion token**:收敛后若 late case 仍不行,这是下一个结构杠杆
   (anchor 只给"点在哪",补"点在怎么动":过去 q 步真实位移作 σ=0 干净
   block token;改动涉数据/codec/序列三层);
5. **收敛后复查方向盲区**:逐点方向余弦/FDE 系统评估(旧 run stepCos≤0.32、
   late≈0),看 29 集 + mvs16 后是否缓解;
6. eval 面板既有 artifact `position_grid_inside_fraction=0.0` 待修;
7. **commit**:reader 双格式、缓存工具、eval 修复、dataloader LRU 修复、两个
   新脚本、测试,全部未提交;
8. 101 集全量训练(等 29 集结论后);regions 过滤首轮不开(legacy/eval 集缺
   region_labels.npy)。

**训练量与 eval 覆盖**(2026-09-22 补充):

- 29 集共 26,398 窗口,train split 21,118;**全局 batch = 32/卡 × 8 = 256**
  (`max_samples_per_batch`,不是 recipe 里的 `batch_size=2`)→ **≈83 step/epoch**,
  iter 889 ≈ 10.7 epoch,max_iter 10000 ≈ 121 epoch——plateau 出现在多轮遍历
  之后,是表征上限而非欠训;过拟合成为主要风险,需要 val 侧证据分辨;
- val split 按集切分实为 **6 集**(seed=42:0019/0080/0031/0028/0022/0099),
  但 `stage_windows` 只在第一个够长的 episode 块内选早/中/晚 → 12 个 val case
  全部落在 0019。**TODO-9:拓宽 eval case 到多集**(每集 2-3 窗口;改动会触发
  fixed_cases 校验,须新 OUTPUT_ROOT,本轮不动)。

### 深度漂移乌龙与问题正确定性(2026-09-22 晚)

**纠错**:当日早些时候的"菜叶点 1 米幻影漂移"结论是**分析错误**——
`target[k] = future_pos - anchor`(pointflow_window.py:255)是锚点相对位移,
分析时误当逐帧 delta 累加,虚增 ~20 倍。正确口径(每点最后有效步的锚点相对
位移):菜叶点实际位移 p50 = 51mm(w345)/ 101mm(w409),uv 首末位移
p50 = 0.2-0.3px。

**正确定性**:labeled 数据没有损坏。菜叶点的 GT 运动**沿相机射线为主**(z 向
50-100mm),透视投影下 2D 近乎不可见;eval 视频中"点跟着手动"的是紫色 pred
(模型对不可见 z 信号的可见化猜测),绿色 GT uv 静止(实测 0.2px)。这是一个
**部分可观测性问题**,不是数据污染:val_02/03 破不了 1.0 因为监督信号的大头
不在图像里。

**旧 run 未选中菜叶点的原因**:eval 集不同(0015 vs 0019)+ 旧数据飞点
(单点 1.6m)占据 top-300 名额;新 labeled 清掉飞点/背景后菜叶点首次入选。

**衍生 TODO**:
- 验证 z 向运动真实性(深度图时序 or 数据生产方确认 3d_ff 接触段稳定性);
- ADE 拆分为 uv 一致分量 / 纯 z 分量分别报告,否则可观测部分的进展被淹没;
- 若纯 z 分量确认不可学,loss 侧降权(与方向盲区分开治理)。

### 幻影漂移最终定性(2026-09-22 晚,抽帧目视确认后修正)

继上节纠错后,目视抽帧确认:**w345/w409 中菜叶未被抓取,全程静止**(手指仅
戳触叶尖)。最终结论:

- 菜叶点的 z 向位移(选中子集 p50=51-101mm)**是深度噪声**,不是真实运动——
  手指接触/遮挡叶尖时 3d_ff 的深度估计在接触区漂移;
- 但全集菜叶点是干净的(p50=0mm, p99=56mm):幻影集中在**长尾**;
- **核心机制:top-300 运动量排名在收割这个长尾**——静止菜叶上的真实点排不
  上号,幻影点全部入选(选中子集 vs 总体差一个数量级);
- 旧 run 未见此现象:eval 集为 0015 且旧数据飞点(单点 1.6m)占满 top-300;
  新 labeled 清掉飞点/背景后长尾浮现。**清洗改变了选点分布**,漂移本身两版
  数据都有;
- 修复方向(训练侧):选点排名加 uv-3D 一致性守卫(Δuv<2px 且沿射线位移
  >30mm 的点降权);规模量化见 `tools/scan_pointflow_phantom_drift.py` 的产出
  `pointflow_outputs/phantom_drift_scan_29ep_20260922.json`。

**幻影规模扫描结果**(2026-09-22,`tools/scan_pointflow_phantom_drift.py`,
29 集 × 12 窗口,判据:位移>30mm 且 Δuv<2px):

- **top-300 选中点中幻影占 12.1%**(12596/103800),分集 1.1%(0090)至
  31.4%(0017);
- **L3(菜叶)选中点几乎全是幻影**(0013: 140/141,0026: 295/299,0055:
  161/161,0063: 31/31);L4 部分集严重(0022: 409/455,0032: 398/498);
  L2(手)总体干净(<3%),L1 干净;
- 含义:物体点的运动监督信号大部分是深度噪声,eval val_02/03 卡住是个例
  表现,根因是选点在收割漂移长尾;
- 修复:选点加 uv-3D 一致性守卫(幻影降权)→ 重扫验证 → scale 重标定 →
  新 OUTPUT_ROOT 重训。扫描数据:`pointflow_outputs/phantom_drift_scan_29ep_20260922.json`。

**消融工具与"跟手"实锤**(2026-09-22 晚):pred 像素漂移方向与手的 GT 运动方向
余弦 0.90(val_02)/1.00(val_03),与径向(射线误表达的方向)反相关——"手动点动"
是学到的相关性,不是编码几何伪影。`PointFlowEvalCallback` 新增
`ablate_modes`(env `POINTFLOW_ABLATE_MODES=action,video,first_frame`):逐通道
中性化条件后重采样,`<mode>_ablation_dependence≈0` 表示该通道没被读;
`first_frame` 专测"未来帧是否被读"(wam 训练中未来帧是噪声目标而非上下文)。
同时确认:point token 的 mRoPE 位置 = (time, h, w) 由 anchor_uv 决定,**无深度**
(pointflow_sequence.py:30-61);监督目标是 Δxyz,uv/z 通道未分离。

### 三通道消融(step-2100,2026-09-22 晚,含 latent 缓存修复后的真实数据)

背景:首轮消融(19:30-19:59 的 step 2100/2200)video/first_frame 全 0.00 是
**伪影**——batch 带 `vae_latent_cache` 时 `get_data_and_condition` 跳过 VAE
编码直接用缓存(omni_mot_model.py:3832),像素级消融根本没到达模型。修复:
video 消融同时丢弃 latent 缓存强制重编码(`pointflow_eval.py`)。

结论(完整表见本节末尾记录):

1. **action 通道死亡**:全部 14 case dependence ≤0.05,点分支不读 GT 未来
   action,"点 token 是 video/action 桥"的设计前提不成立,需单独排查接线/梯度;
2. **video 是命门**:全黑后普遍崩向 zero 基线(dep 0.4-1.45);
3. **未来帧是真假成绩的开关**:
   - 最好的 4 个 case(train_00/val_00/val_09/val_11,比值 0.14-0.25)冻结未来帧
     后精确退回基线(0.95-1.01)——**好成绩几乎全靠读未来帧**;
   - 卡住的 case(train_01/val_02/val_08)不读未来帧,差是真实的差;
   - val_10 冻结后反从 0.91 改善到 0.60——未来帧在此净误导;
4. **部署诚实度看"仅首帧"列**:14 个 case 全部 ≥0.60,多数 ≈1.0——没有未来
   信息时模型打不过"预测不动"。主 eval 数字系统性高估,**路线 B(首帧条件/
   联合去噪 eval)升为最高优先级**;
5. "跟手"的信息源 = 首帧手的视觉内容 + 任务先验(非 action、非未来帧)。

| case | 正常ADE | zero | 比值 | 无action(dep) | 无video(dep) | 仅首帧(dep) | 首帧比值 |
|---|---|---|---|---|---|---|---|
| train_00 | 39.1 | 286.9 | 0.14 | 40.1 (0.01) | 275.2 (0.95) | 291.1 (0.99) | 1.01 |
| train_01 | 48.4 | 42.4 | 1.14 | 49.3 (0.03) | 58.3 (0.91) | 46.2 (0.48) | 1.09 |
| val_00 | 23.8 | 122.0 | 0.20 | 23.6 (0.01) | 117.4 (0.99) | 122.0 (0.99) | 1.00 |
| val_01 | 35.1 | 73.8 | 0.48 | 35.3 (0.01) | 81.3 (1.02) | 62.0 (0.75) | 0.84 |
| val_02 | 34.5 | 37.0 | 0.93 | 34.6 (0.02) | 65.0 (0.92) | 34.9 (0.30) | 0.94 |
| val_03 | 38.7 | 50.0 | 0.77 | 38.9 (0.02) | 52.3 (0.60) | 37.6 (0.46) | 0.75 |
| val_04 | 26.4 | 42.8 | 0.62 | 28.0 (0.05) | 72.6 (1.45) | 41.4 (0.65) | 0.97 |
| val_05 | 45.2 | 56.3 | 0.80 | 45.2 (0.02) | 67.1 (0.93) | 63.2 (0.76) | 1.12 |
| val_06 | 43.8 | 39.3 | 1.12 | 43.2 (0.03) | 74.5 (0.80) | 54.9 (1.09) | 1.40 |
| val_07 | 35.8 | 52.3 | 0.68 | 36.3 (0.02) | 72.0 (0.88) | 53.3 (0.67) | 1.02 |
| val_08 | 39.4 | 34.8 | 1.13 | 39.8 (0.03) | 52.6 (0.93) | 41.4 (0.36) | 1.19 |
| val_09 | 39.0 | 157.7 | 0.25 | 38.9 (0.01) | 160.7 (0.96) | 159.0 (1.00) | 1.01 |
| val_10 | 51.5 | 56.4 | 0.91 | 52.0 (0.02) | 58.9 (0.39) | 34.0 (0.51) | 0.60 |
| val_11 | 37.5 | 176.6 | 0.21 | 37.8 (0.01) | 202.0 (1.10) | 167.4 (0.95) | 0.95 |

(单位 mm;dep = 消融前后预测轨迹差异/正常预测幅度,≈0 表示模型没读该通道;首帧比值>1 表示冻结未来帧后打不过"预测不动")

### 路线 B 落地(2026-09-22 晚):point 接入主联合去噪循环

实现(omni_mot_model.py):`_prepare_inference_data` 把 point 纯噪声拼入主状态
向量尾部([vision|action|sound|pointflow],GT 目标打包前清零);`_get_velocity`
逐步切出 point 段、以循环时刻 σ=t/1000 组装 PointFlowNoised、preds_pointflow
作速度拼回;`generate_samples_from_batch` 返回 samples["pointflow"](米)。
pack template(Ray serve)路径暂不支持,遇 pointflow 显式报错。

eval:`POINTFLOW_EVAL_JOINT=true` → 每 case 输出 <case>_joint/(首帧+state action
干净,未来 video/action/point 联合采样,UniPC 4 步,guidance=1.0)。
**解读**:conditional vs joint 的 ADE 差 = 未来泄漏贡献;joint vs zero 才是部署
水平。守护测试 `test_joint_sampling_wiring`。**未经 GPU 验证**,首次 joint eval
可能需修。
  可视化:joint case 双画布——`<case>_joint/`(点叠 GT 视频:梦 vs 现实差距)与
  `<case>_joint/dream_canvas/`(点叠模型自生成视频:联合一致性,由
  `model.decode(samples["vision"][0])` 渲染)。判读:梦内自洽+现实差=video 生成
  瓶颈;梦内也不自洽=point 分支条件化问题。

### 阶段拼接补全与离线回拼(2026-09-23)

- 在线 eval 的阶段拼接从 3 段(conditional)扩到 9 段:每 stage(early/middle/
  late)各出 `{stage}_stitched.mp4`(条件拟合)、`{stage}_stitched_joint.mp4`
  (联合 rollout 叠 GT 视频)、`{stage}_stitched_joint_dream.mp4`(联合 rollout
  叠梦视频),`validation_stages.html` 改为按 kind 分行 × stage 分列的网格
  (`stage_viewer` 旧契约只收 3 段,9 段会直接 ValueError——已改,3 段仍走
  旧滑杆版式,测试 `test_grouped_stage_viewer_grid` 守护)。
- **离线回拼**:`tools/stitch_pointflow_canvas.py --eval-dir <run>/.../pointflow_eval`
  纯 CPU,按 `fixed_cases_stages_4windows.json` 的 case→stage 映射补拼缺失的
  joint/joint_dream 拼接视频并重建网格 html;已有文件跳过(`--overwrite` 重做),
  case 不完整的 step 自动跳过。已对 step_0003000-0005100 共 22 个 step 全部
  补齐(129 帧/段 = 4 窗 × 33 帧去重边界)。

### train/val 初始帧"投影偏移"争议:原生帧验证,结论——投影无错(2026-09-23)

起因:train_00(0044@698)三类画布(条件/joint/joint dream)首帧点都在锅沿、
菜叶正下方,疑似投影偏低;val_00(0019@217)点在面包上看着正常,"必有一个错"。
验证链(全部离线,未改代码):

1. train/val/conditional/joint/dream 共用同一份渲染代码与**同一个全局仿射常数**
   (`tools/build_pointflow_manifest.py:36-37`:canvas 640×842,v 向 scale
   480/448=1.0714、head 起始 362)——448 网格是 480 帧的 resize 而非裁剪,
   非 dream 直接缩放与 dream 仿射数学上落到同一相对位置(实测 0.936 vs 0.937);
2. 把 tracker 自己的 `uv_px` 不经过 eval 代码直接画上 `head.mp4` 原生帧:
   红点位置与 eval 渲染一致——点本来就在锅沿;
3. **决定性**:train_00 的 300 个选中点是**手套上的点**——同一批 point_id 的
   tracker uv 从 f698 的锅沿移到 f762 右侧黑手套上(位移中位 258px),GT 3D
   位移 p50=386mm。首帧手就在锅沿,点跟着手,没有投错;val_00 首帧手在面包
   上所以点在面包上。视觉上"点应该钉在菜叶上"是错觉:该窗口菜叶几乎不动,
   top-300 运动量选点不会选它。
- 唯一真实的小系统偏差:labeled 交付无 `intrinsics.npy`,pred 投影统一用 0013
  的常量内参,跨集 t=0 偏差 1-3px(train_00 1.2 / val_00 3.0),比视觉"偏移感"
  小一个数量级;指标在 metric xyz 计算,不受影响。
- 附带事实:train_01(0017@1059)首帧散点满屏是该窗口 65.7%(197/300)选中点
  为幻影漂移点所致(数据/选点问题,非投影),给幻影守卫 TODO 又添一条证据。

## Dropper 数据接入(101 集,2026-09-23)

数据源:`pf_out/dropper_new_9.9/labeled`(同一 Track4World 3d_ff 管线 + CPU
质量过滤,101 集全 labeled,无 legacy);raw `raw_data/singlerighthand_dropper_100`;
cache `datasets/singlerighthand-dropper-100-cosmos-cache`(manifest/episodes/
video_frames 早已备好,**只缺 vae_window_latents**,用
`examples/cache_singlerighthand_window_latents.sh` 换 `VIDEO_CACHE_ROOT` +
`ALLOWLIST` 两个 env 重跑即可)。三处 episode 名单 101/101/101 完全对齐。
点数规模 4.8k-15.8k/集(sandwich ~55k,稀疏约 4 倍),top-300 无压力,最差
窗口也能选出 228 点。

与 sandwich 的实质差异:

- **action space 是 joint**(arm 7 + hand 20 关节角,eef 是 sandwich 的);
  `singlerighthand_raw_dataset.py:154` 从 cache manifest 自动读,toml 不动;
- task_text 已在 cache manifest("draw liquid from the beaker ...");
- 相机 rig 完全相同(head 640×480 / wrist 848×480 / tracker 画布 640×448),
  `build_pointflow_manifest.py` 的硬编码仿射直接沿用,已用
  `tools/visualize_pointflow_selection.py` 验证:选中点准确落在手套/夹爪/
  滴管上(episode_0003 w100/w600,产物在
  `pointflow_outputs/dropper_101_20260923/vis/`)。

接入改动与产物:

- `pointflow_window.py::_episode_metadata`:老版 labeled 交付(dropper
  episode_0001,全 101 集中仅此一例)的 report.json 缺 `native_pixel_queries`
  字段,原实现直接 raise。改为字段缺失时回退校验首帧 valid uv 是否落在
  640×448 网格内(越界才报错);测试
  `test_labeled_delivery_without_native_pixel_queries` 守护。不修这条,
  episode_0001 在训练中必炸、扫描中静默丢集;
- manifest:`pointflow_outputs/dropper_101_20260923/dropper_manifest.json`
  (101 labeled / 0 legacy);allowlist
  `examples/pointflow_dropper_all_101_episodes.txt`;
- **scale 重标定**(老规矩):dropper 101 集 top-300+mvs16 实测 std
  **0.0482**(`dropper_101_20260923/scale_scan_top300.json`,n=21.5M,
  std/robust=0.81 尾部略抬但不改用 std 的约定),recipe 默认 0.0740 是
  sandwich-labeled29 的,dropper 必须显式覆盖——已写进
  `examples/launch_pointflow_dropper101.sh` 的 `EXTRA_TAIL_OVERRIDES`;
- 启动脚本:`examples/launch_pointflow_dropper101.sh`(全新 OUTPUT_ROOT
  `pointflow_dropper101_300_scale00482_20260923`,不可 resume sandwich
  checkpoint)。

**幻影漂移扫描(100 集,episode_0001 扫描时修复未落地被跳过):总计 23.4%
(84404/360000),约为 sandwich(12.1%)两倍,且分布完全倒挂**:

| 区域标签 | sandwich 29ep | dropper 100ep |
| --- | --- | --- |
| L1 指尖 | 5.2% (669/12968) | 90.2% (637/706) |
| L2 手 | 5.0% (3539/71269) | **30.2% (78381/259619)** |
| L3+ 物体 | 75.8% / 26.3% (L3/L4) | 5.4% (5386/99675) |

sandwich 的漂移集中在物体点(菜叶),dropper 集中在**手/指尖轨迹**——正是
top-300 运动量排名的主力军,意味着近三分之一的选中监督信号是"uv 锁死 +
深度漂移"的幻影。最差集:episode_0002 65.3%、episode_0023 55.2%、
episode_0089 46.6%。注意该启发式(Δuv<2px 且 3D>30mm)无法区分真·朝向
相机的伸展运动,但 sandwich 手部只有 5% 说明 30% 是数据问题而非任务形态。
**幻影守卫(排名时 Δuv<2px 且位移>30mm → rank_score=-inf)对 dropper 比
sandwich 更关键,但守卫会同时杀掉真·深度伸展,落地前需权衡;首轮先不开,
与 sandwich 配置对齐拿基线。**

## 幻影守卫落地 + dropper 可视化对照(2026-09-24)

实现(选点降级版,阈值与扫描一致:Δuv<2px 且末个 valid 步 3D 位移>30mm):

- `pointflow_window.py::prepare_window` 新增 `select_phantom_guard`
  (+ `phantom_guard_disp_mm=30` / `phantom_guard_uv_px=2.0`),幻影点在运动
  排名中置 -inf 降级,与 min_voxel_members 守卫同机制;读未来标签,是训练
  期规则。`PointFlowSource` 已接构造参数(env 接线留待决定开训时再做);
- 测试 `test_phantom_guard_demotes_uv_locked_drifters` 守护;16 个窗口测试
  全过;
- `tools/visualize_pointflow_selection.py --phantom-guard` 渲染对照(文件名
  带 `_noghost`)。

dropper 对照(产物 `pointflow_outputs/dropper_101_20260923/vis/`):

- **episode_0002 w200(65.3% 幻影集,最差)**:不开守卫时 top-300 几乎只剩
  右上角黑幕上的一簇红色幻影点;开后大漂移幻影簇消失,p50 47.3→24.2mm。
  **但进一步分析(2026-09-24)发现这只解决了一半**:守卫后 ep0002 各窗口
  仍有 70-94% 选中点锚定在右上角黑幕区——这些点 3D 漂移 ~20mm(低于守卫
  30mm 阈值)、uv 动 ~1.6px、存活率 100%,按现有全部规则都是"优质候选",
  但 z≈1.5m(工作区 0.87m)表明是远背景幕布纹理。它们挂着 L2 手标签
  (hand ROI 多边形覆盖手部入场区),**语义标签无法区分幕布与真实表面**;
  真正手套点只剩 14/800。ep0002 手部覆盖差是主因(黑手套+黑幕布,交付
  pipeline 的 appearance gate 本来就拦暗色表面);
- **episode_0089 w300(46.6% 幻影集)**:选点反而完美——300 点全部打在
  被夹爪举起的滴管蓝泡上,p50 102.7mm。"幕布区"分析在这集误报:该区域
  是真实的滴管(z≈0.82m,L3 标签);
- **episode_0003 w100**:守卫后点集中到手套手指;p50 59.0→51.3mm;
- 结论:守卫按设计工作(各预算下幻影 0 残留),但"幻影"只是 dropper 数据
  问题的一半,另一半是**低漂移背景噪声点抢占预算**,逐集差异极大。这直接
  影响语义配额设计:区域内体素均匀采样可稀释幕布份额,z 工作区闸口
  (锚点时刻可部署)能干净杀掉 ep0002 型远背景,窗口级质量过滤(锚点帧
  真实工作区点太少则跳过)是第三个抓手。

操作手册独立成文:`docs/pointflow_data_pipeline.md`(allowlist/manifest/
latent cache/scale 标定/幻影扫描与守卫/可视化/训练启动的全流程命令)。

### dropper 交付的更深缺陷(2026-09-24,无过滤整集渲染 + 逐标签量化)

用 `visualize_pointflow_selection.py` 无过滤用法(max-points/top-n 20000,
守卫全关)整集平铺渲染 ep0002/ep0003/ep0053,叠加逐标签量化:

- **L1 指尖通道整个是假的**:49 个点 uv 钉死在 (513,190)±3px,z=1.57m
  (幕布深度),1400 帧全程不动——指尖种子是 regions.json 里手工钉死的 3 个
  固定坐标,首帧手未入场,种子落在幕布上,tracker 锁死。幻影扫描 L1 90.2%
  即此。pipeline 自己的 `initial_candidate_regions.png` 可直接看到蓝色手部
  候选块整个画在幕布上;
- **L2 手部覆盖偏手掌/手背**(首帧可见部分),手指腹侧基本无点;集间差异
  取决于首帧手入画程度:ep0003/ep0053 首帧手已入画覆盖好,ep0002 首帧手
  未入场、整集手套区只剩 25 个有效点(f800);
- **L3 滴管质量好**:全程粘附蓝泡,运动着色准确;
- **幕布块普遍存在于所有抽样集**(z≈1.5m),挂着 L2/L3 标签,语义配额无法
  区分——z 闸口(候选 z<工作区上限)可同时杀掉幕布块和假 L1,dropper 配额
  只需分 手/物体 两类;
- 结论:flow 管"物体+手掌表面稠密运动",FK(URDF 21 关键点含五指指尖)
  恰好补上交付完全缺失的指尖/指节通道。

### dagger 的"白墙块"与手部翻转无点的裁决(2026-09-24)

用户发现 dagger ep0010 手翻转后看不到点,排查结论(逐帧单图验证:
绿=valid L2、黄叉=被门控杀、红=L1):

- **可视化无 bug**,点位置准确;翻转后没点是数据真相,两层叠加:
  1. 掌心/指腹侧首帧不可见 → 永远没有种子(同 dropper_101);
  2. **dagger 也有"幕布块",只是背景是白墙**:首帧 hand ROI 覆盖右上角
     白墙,墙点被种子+锁死,翻转帧上占 L2 有效点的 34-67%(f800 实测
     1813 个 L2 里 1209 个钉在墙上,z=1.04m vs 真手 0.66m)——翻转帧
     真手点只剩几百个且偏腕部;
- 质量门控逐帧波动大:L2 部分帧杀 30-45%,L1 指尖部分帧杀 70-90%
  (372→109);ep0010 全片点数 11683→2523 衰减,ep0013 则 16-22k 稳定,
  **集间质量差异大,点数衰减曲线可作训练集筛选指标**;
- z 闸口对白墙块同样有效,但墙-手 z 差只有 ~0.35m(幕布 0.7m),阈值要
  按数据集分别标定(建议 0.9-1.0m),不能一刀切;
- 病根统一:首帧 ROI 内背景被种子+锁定,与背景是黑幕还是白墙无关。

### 更正 + efep 格式才是病根裁决(2026-09-24)

**更正前文两处错误**:① 我说"官方 3D 可视化在翻转帧也看不到手点"是错的——
那是我 MP4 导出脚本的 RGB/BGR 通道 bug(get_render 返回 RGB,cv2 要写
BGR,已修),官方交互 HTML 一直是对的;② "手部稠密覆盖是 tracker 天花板"
也错了——同一 tracker,efep 格式(每帧新生轨迹,infer_pair 链接)在翻转帧
有 5.9-10.7 万个手部点,覆盖完整。

**病根重述**:flat labeled(训练在用)与 efep_labeled(官方 3D 在用)是同一
tracker 的两种导出:前者整集单次推理、query 固定在第 0 帧(身份跨窗口一致,
但后入场/翻转面永远无点,翻转帧真手点跌到几百个);后者每帧重种子(覆盖完整,
且 obs_track 仍给跨帧身份)。老管线还有"指定起点固定 query"形态
(track4world_dense_pointflow/episode_0008_f947_160:start_frame=947 导 160 帧)。

**路线(不破坏逐点 token 范式)**:离线转换 efep→flat labeled schema
(`tools/convert_efep_labeled.py`),候选池从"首帧存活点"变成"锚点帧存活
轨迹",prepare_window/manifest/扫描/可视化全部零改动。要点:track 按全集
有效观测数过滤(定 32:全集凑不满一个窗口的轨迹无训练价值;窗口级裁决仍归
prepare_window 的 mvs 守卫);invalid 槽位沿用 parked-value 语义(幻影守卫
不受影响);逐帧 seek 读 + 流式写,小内存机器可跑。验证中的第一集:dagger
ep0010(唯一现成 efep 集)。退路:若 efep 逐帧链接身份质量差,改走 chunk 化
3d_ff + 跨 chunk 身份拼接(数据侧工作)。

### efep→flat 转换器落地并验证 + 9.24 新数据(2026-09-24)

**9.24 新数据(数据侧,进行中)**:pf_out 已重组,`pf_out/9.24/` 是四个数据集的
官方 3d_efep + **SAM2 逐帧 mask** 重导(提交 94da2a9,README/CLAUDE.md 在
该目录):语义从"首帧手画多边形 ROI"升级为逐帧像素级 mask(墙/幕布块从源头
消失),label 2=手 3=被操作物体 4=台面/静止物;训练点 = obs_valid & obs_unique;
交付格式 efep_seg_v61(ragged obs 表 + track_label + c2w + intrinsics)。
**截至今天交付目录仍是占位(DUMMY_SKIP + obs_valid + report.json),导出
还在跑**,sandwich 96 / dropper 97 / dagger 48 / micropipette 90 集状态可查
(status/*.json)。老数据在 pf_out/old_before_9.24/(路径已变,勿混用)。

**转换器 `tools/convert_efep_labeled.py`(已验证)**:efep ragged → flat
labeled schema,逐帧 seek 读 + 流式写(小内存可跑);track 按全集有效观测
过滤(默认 32);invalid 槽位 parked-value 语义;obs_unique 存在时训练点
= valid&unique;--video 补老 report 缺失字段。修过的坑:npy 头重复 magic、
descr 结构化、query_ids 索引。

**dagger ep0010(老 efep_labeled)端到端验证结果**:

| 指标 | 老 flat labeled | 转换后(efep) |
| --- | --- | --- |
| f650 手点(翻转帧) | 3,343(34% 墙) | **106,739** |
| f800 手点(翻转帧) | 1,813(67% 墙) | **86,261** |
| w640 窗口候选 | 4,708 | 20,784 |
| 幻影率(top-300) | 23.4%(dropper 集口径) | **0.1%** |
| prepare_window 加载 | — | ✅ top-300 正常 |

翻转帧渲染(`efep_converted/vis/`)确认手套(含翻转掌心侧)被稠密覆盖;
老 efep 的"手"类仍含臂/墙(首帧 ROI 标签所致)——等 9.24 SAM2 数据到位
后标签问题从源头解决,届时批量转换 → manifest → 扫描 → 配额+z 闸口 → 开训。

### 再更正:手套空的凶手是老打标步骤,不是重构也不是转换器(2026-09-24)

用户指出 efep 转换渲染里手套一大块是空的、点多在手臂。逐层追查(dagger
ep0010 f650 黑手套手指紧致区 u330-400/v60-180):

- 重构 raw efep:**15,636 obs**(clean 14,925,conf p50 0.98)——黑手套手指
  重构正常,点存在;
- 老 efep_labeled 交付:**281**——老打标步骤(锚帧 0 多边形+邻域继承)丢掉
  98%。翻转/后入场表面在"新位置"出生的轨迹挂不上"手"标签被丢弃;
- 转换器无空间过滤,忠实继承;
- 官方 3D 里完整的"手形"实为**手臂**(白色重构好)+ 墙,L2 标签虚胖;
- "efep 转换后手部覆盖完整"结论作废:老 efep 交付里手套依然空。**9.24 的
  SAM2 逐帧 mask + mask 内全量取点才是真正的解**;
- 9.24 进度(很早期):交付目录全占位(DUMMY_SKIP),sam2_masks sandwich 5 /
  dropper 1 / dagger 0 / micropipette 0;
- 保留验证项:即使 9.24 保住手指点,黑手套深度质量仍需幻影扫描复验
  (conf 高 ≠ 深度准)。

### 3D 深度分层实锤:f650 "手"标签 = 手臂 + 墙板(2026-09-24)

用户在官方 HTML 21.6s(=f648)看到两大块蓝色"手形",质疑"手套空"的结论。
用深度着色渲染同帧"手"标签点(efep_converted/vis/depth_layers_650.png):
侧视图清楚分出两层壳——近 z≈0.55-0.75m(深蓝,手臂+手背)与远
z≈0.9-1.15m(橙红平板,墙),计数 19057/2295/18648。用户看到的两大块 =
手臂 + 挂"手"标签的墙板;手套手指/掌心仍空(2D 逐像素投影已证)。
三层证据齐了:2D 逐像素投影 / raw(15636) vs 交付(281) 计数 / 3D 深度分层。

### 9.24 新数据终极大考通过:dagger ep0002(2026-09-24)

9.24 仍在导出(sandwich 21 / dropper 4 / dagger 3 / micropipette 1 集完成,
status/*.json 可查),拿已完成的 dagger ep0002 做新旧对比(转换器直转
efep_seg_v61,obs_valid&obs_unique,min_valid_obs=32):

- **SAM2 标签根治墙/臂污染**:手点 z>0.9 占比 0%(老数据 34-67% 是墙),
  白色机械臂不再挂"手"标签;台面/静止物 16k 点/帧正确归入 L4;
- **翻转帧手套完整**:f800(掌心朝镜头,以前全空)五指轮廓根根分明
  (vis/new924_f500.jpg / new924_f800.jpg);
- **幻影率 10.4%**(373/3600,与 sandwich 12.1% 同档;老 dropper 最差集
  65%)——黑手套深度噪声仍在但可接受;
- 新标签体系:L2 手 / L3 被操作物体 / L4 台面静止物,**无 L1 指尖类**
  (配额设计按 L2/L3/L4 走;指尖细粒度归 FK 模态补)。

下一步:等 9.24 全量导出完成(或边出边转)→ 批量转换 → manifest →
scale 重标定(候选池完全不同,必须重测)→ z 闸口+配额采样 → 新
OUTPUT_ROOT 开训。

## FK 对账(2026-09-24):原料齐全,URDF 相机链需修正

**原料盘点(全部存在)**:

- 关节角:lmdb `/observations/qpos` [T,54],右 arm=qpos[27:34] rad、右
  hand=qpos[34:54] rad(qpos_layout 见 meta_info.pkl robot_layout);
  另有 `/observations/hand_joint_deg`(度)与 `/observations/eef`;
- URDF:`marvin_wuji_d435_complete.urdf`(83 joints;右臂 Joint1-7_R、
  右手 5 指×4 关节;**含头相机链** head_d435_link_optical_frame 与腕相机链);
- 独立深度:D435 实深 `auxiliary_camera/depth.lmdb`(head_depth uint16 PNG
  284 帧、right_wrist_depth 1420 帧)——depthanything 的第三方真值;
- MANUS 手套 `/teleop/manus/right/keypoints_21` [T,21,3]:手局部坐标系
  (|xyz|~0.1m),teleop 侧数据,不能直接给相机系位置,可用于关节角互验。

**FK 验证链与结论**(dagger ep0002):

1. 直接用 URDF 相机链:FK 21 点投影落在试管架上(偏移 ~15-20cm),且 FK
   手臂指向与真实手臂相反——最初疑似臂映射错;
2. 用 25 帧 FK 腕部(base 系)对 tracker 手点质心(相机系)做 Kabsch 拟合
   单一刚体变换:**残差 p50=30mm / max=59mm 全剧集恒定**,拟合变换投影后
   21 点精准落在手套上(/tmp/fk_fitted_proj_650.jpg 已存档 vis/);
3. 结论:**臂 FK + qpos 映射(right=27:34/34:54)正确;URDF 的 head 相机
   链与实物不符**(URDF 相机 t=(0.064,0,1.375) vs 拟合 t=(0.163,0.898,
   1.026),R 也不同);depthanything 深度与 FK 深度在拟合单一变换后 ~3cm
   (~4%)一致——**没有大 scale 因子**,深度噪声水平 ~3cm;
4. 注意:用 tracker 数据拟合相机修正对训练生产有循环依赖,正式方案应
   (a)用 D435 实深独立拟合/验证,或(b)找机器人侧要真实外参;
5. 后续:D435 三重验证、把 FK 修正变换做成按 rig 的标定产物、FK 21 点
   接入模态(干净、不加噪、锚点帧可用,局部 z 走 A/A'/B 编码)。

### 三方深度对账:D435 + FK + depthanything(2026-09-24)

sandwich ep0013 f650(D435 实深 6fps/uint16,经正式内参+depth→color 外参映射,
轮廓对齐已验证):深度排序 depthanything 表面 < D435 表面 < FK 关节中心。

- **depthanything − FK关节 = -19mm**:关节在组织内、表面在前 ~2cm,
  解剖学完美 → depthanything 深度真米级且系统正确,FK/pointflow 可共用
  坐标系,无需缩放;
- **D435 − FK关节 = -53mm**:偏多,D435 在黑手套上偏近 ~3cm(RealSense
  深色表面已知问题),不能当深度真值裁判,作旁证合格(关键点旁 96%
  有效、量级一致);
- 坐标系统一:base→camera 用 mano 仓库正式外参
  (`fk_camera_extrinsic.py`:修过 URDF 光学原点 32.5/4.3mm 错误 +
  180° roll 对齐图像约定);FK 标注 raw_data/sandwich_fk21(101 集,
  wuji_fk21.npz,[T,2,21,3] Link_Base 米制,右手 index=1);
- 遗留:mano 外参在 dagger ep0002 上投影残差 ~5-8cm(sandwich rig 标定,
  dagger 晚录一个月,rig 可能动过)——**每个采集批次的外参要重新核验**,
  FK 标注生产按批次走 verify_fk21_allowlist 流程。

### 外参按批次核验:dropper 通过,dagger 需重标(2026-09-24)

用 mano(sandwich rig)外参在 dropper(非 dagger)ep0002 上投影 FK 21 点:
f300/f800 全部精准落在手套上(vis/fk_dropper_f300.jpg / f800.jpg)——
**dropper 与 sandwich 同一 rig**,外参和 tracker 内参常数(fx=0.813/
fy=1.162)都直接迁移。dagger 批(20260830 录制,晚 ~3 周)投影残差
~5-8cm,rig 被动过,外参需按批次重拟合或找机器人侧要标定。

## 迁集群 + 性能/数据管线大修(2026-09-28/29)

集群迁移到 inspur A800(sm80)/SenseCore。代码、依赖、数据的调试记录全部
在 [pointflow_quickstart.md](./pointflow_quickstart.md)(第 1/2/6/7 节)和
[pointflow_multinode_training.md](./pointflow_multinode_training.md)。要点:

**性能**(labeled29 recipe,batch 32/卡基准):

| 改动 | 效果 |
| ---- | ---- |
| FSDP 全分片(shard=8,替代纯 DDP) | 修 80GB 卡 OOM,语义不变,全局 batch 不变 |
| flash2 varlen 解禁(`COSMOS_FLASH2_VARLEN=1`) | kernel 4.76x,端到端 35.2→16s/step;数值验证 `tools/check_flash2_varlen.py`(MHA/GQA×causal/full 全过);上游禁令是策略性保守,FA2 varlen 在 Ampere 上成熟 |
| `prepare_window` import 提升到模块级 | 窗口准备 4x(每次调用重 import cv2/scipy 在网络 FS 上是 1700 次 stat 风暴) |
| DCP save 前 `empty_cache()` | 修偶发 NCCL "unhandled cuda error"(裸 cudaMalloc 撞上 PyTorch 缓存预留) |

**HSDP 多机**:SenseCore 注入 `NNODES/MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE`
(不给 NODE_RANK;WORLD_SIZE 是节点数),launcher 按"显式 > SenseCore > pod 名
兜底(master-0/worker-N)"解析;wrapper 自动推 shard=每节点卡数、replicate=
节点数。16 卡(2 节点)world_size 16 组网验证通过。教训:交互 pod 上起"多机"
会各自独立成 8 卡任务且共享 OUTPUT_ROOT,日志双份交错,务必走任务系统。

**数据管线**(101 集 sandwich):

| 改动 | 效果 |
| ---- | ---- |
| 9.24 efep→labeled 转换(`--video` 必填,漏了会在训练时报 "Cannot open head video: None") | sandwich/dropper 各 101 集 |
| 分层语义选点 `select_region_quotas`(组内 GT 运动排名+缺额回补,总数恒定) | 扁平 top-500 实为 92% 手+8% 物+0% 台面;分层后精确 200/225/75(40/45/15) |
| 幻影守卫接线(`POINTFLOW_SELECT_PHANTOM_GUARD`) | 选中点幻影占比 7.5%→0.7%;幻影集中在物体区(label 3 无守卫时 13.9%)——深度噪声最坏恰在核心监督目标上 |
| 窗口预计算缓存(`tools/build_pointflow_window_cache.py`,流式单遍:轨迹每集读一遍+视频一遍解码) | 训练每样本读 205MB→120KB;101 集构建 473s;与在线逐位一致;miss 回退在线;配置不一致拒绝读取 |
| scale 重扫(101 集分层 500+守卫) | **0.0528**(扁平 top-300 的 0.0753 不再适用;中位位移 21mm→4.8mm) |

注意:选点配置(点数/配额/守卫)变化 → 窗口缓存要重建、scale 要重扫、
训练要新 OUTPUT_ROOT。

## A/B:per_point vs cluster + 误差三分解 + per-frame scale(2026-09-29/30)

**A/B 对比**(同数据同选点同 eval case,唯一变量 token 形态;判据 ADE/zero):

| | per_point-500 | cluster-origin(stage1) |
| ---- | ---- | ---- |
| 等 iter @5900:平均 ADE / 比值 | **14.74mm / 0.229** | 17.79mm / 0.280 |
| 等 wall-clock ~12h | **15.61 / 0.254**(@3900) | 17.53 / 0.270(@9400) |
| validation loss @5900 | **1.40** | 1.63 |
| 速度 | 11s/iter | 4.65s/iter(2.4x) |

cluster 五个 val case 全败,且 5900→9400 基本 plateau;train ADE 两者相当(~6.7mm),
差距全在泛化 —— 池化丢信息、靠记忆训练集。静态点漂移 cluster 明显更糟(val_00 静态
ADE 7.6 vs 4.2)。结论:500 点规模 per_point 全面占优;cluster 的价值在几千点、
per_point 放不下的场景,其病灶由 skip+点级 block 方案治
([pointflow_cluster_decode_20260929.md](./pointflow_cluster_decode_20260929.md))。

**"误差随时间涨"三分解**(逐帧误差曲线 + gt-cond/joint 对照):

| 成分 | 机制 | 证据 | 对策 |
| ---- | ---- | ---- | ---- |
| 机械量纲 | 目标是累计位移,GT 幅度 6.7→140mm,绝对误差同比涨 | gt-cond 比值 0.62→0.11 其实在变好 | 不用治,看比值 |
| 归一化/SNR | 全局单一 scale+共享 σ,前段帧信号被噪声淹没、loss 不扣分 | gt-cond t=0 比值 0.62,前段几乎没学 | **per-frame scale**(本节) |
| 条件漂移 | joint dream 视频偏快/偏离 GT,点自洽跟随 | joint 比 gt-cond 后段差 +20~33mm(t≥7) | video 主干问题,点侧治不了 |

条件漂移的另一面是好消息:点在 dream 里与视频自洽,说明 point-video 桥接是 work 的。
gt-cond 指标才是点分支自身的干净信号;joint 指标衡量的是整个系统。

**per-frame scale**(照搬 PointWorld per-timestep 归一化,目标参数化不变、不积分):
`POINTFLOW_DISPLACEMENT_FRAME_SCALES` = 逐帧 32 值向量,覆盖标量;7 个 scale 使用点
全部接入(train/eval/joint 一致);101 集实测 std 向量 0.00506→0.07583(15 倍),
全局 0.0528 对前 8 帧高估 3~10 倍。实现与向量见
[pointflow_displacement_scale_20260914.md](./pointflow_displacement_scale_20260914.md) §14。
注意:改目标参数化 → 从头训 + 新 OUTPUT_ROOT + resume 必须带同一 env;loss 读数跨
参数化不可比,看 ADE/zero。
