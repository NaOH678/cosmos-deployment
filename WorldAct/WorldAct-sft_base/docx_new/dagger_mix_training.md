# DAgger-mix 数据集变体(fk+point 四模态)

> 面向要复现/接手这条线的人:数据是什么、混采为什么不用改代码、scale 怎么来的、怎么启动。
> 通用四模态内容不重复,见 [`base_quickstart.md`](./base_quickstart.md) 与
> [`fk_pointflow_merge.md`](./fk_pointflow_merge.md);本文只记 **dagger-mix 特有的**。

## 0. 一句话

在 dropper 任务上用两批 DAgger 数据(51 + 20 = 71 集)训 fk+point 四模态,入口
`tools/run_fk_point_dagger_mix.sh`;两批数据按窗口粒度随机混采,**无任何代码改动**。

## 1. 数据是什么

| 批 | raw 目录 | 集数 | 采集日期 |
|---|---|---|---|
| dagger(旧) | `/data/shichaojian/raw_data/dropper_dagger` | 51 | 2026-08-30 |
| dagger_new(新) | `/data/shichaojian/raw_data/dropper_dagger_new` | 20 | 2026-09-03 |
| **mix(训练用)** | `/data/shichaojian/raw_data/dropper_dagger_mix` | **71** | 两批并集(diff 验证逐名一致) |

mix 不是软链合并,是真实目录;两批 episode 名带各自时间戳,不会撞名。

任务文本:`draw liquid from the beaker with a dropper and dispense it into the test
tube`(cache manifest 的 `task_text`,与 dropper-101 一致)。

## 2. 转换产物清单(训练前已齐,2026-10-02 核查)

| 产物 | 路径 | 状态 |
|---|---|---|
| cosmos cache | `/data/shichaojian/datasets/dropper-dagger-mix-cosmos-cache` | 71 集;动作空间 **joint**(7 arm + 20 hand,同 dropper-101,**非** sandwich 的 EEF) |
| FK 标注 | `/data/shichaojian/raw_data/dropper_dagger_mix_fk21/<ep>/annotations/wuji_fk21.npz` | 71/71 |
| pointflow 窗口缓存 | `<cache>/pointflow_windows/` | 71 集;配置与 sandwich/dropper 逐项一致(top500、quotas 2:0.40/3:0.45/4:0.15、min_voxel 3、min_valid 16、phantom_guard) |
| window VAE latent | `<cache>/vae_window_latents/` | 71 集;schema fps15/chunk32/stride1 与训练配置匹配 |
| pointflow manifest | `WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/dagger_mix_71_20261001/manifest.json` | 71 集,指向 `pf_out/9.24/dagger/labeled` + `pf_out/10.1/dagger_new` 的原始点数据,路径零缺失 |
| episode allowlist | `examples/dropper_dagger_mix_71_episodes.txt`(_base) | 71 行 |

## 3. 随机混合:为什么不用改代码

两批数据在 mix root 下就是普通 episode;数据集把 allowlist 全部 71 集的有效窗口索引成
**一个扁平列表**(`singlerighthand_raw_dataset.py` 的 `__len__` = 各集窗口数累加),
dataflow 在整个列表上 shuffle(`singlerighthand_raw_dataset.py:369` `get_shuffle_blocks`)。
每个 step 的 batch 因此按窗口粒度随机混合两批数据,采样频率 = 各批自然窗口数之比(约 51:20)。

⚠️ 若以后要**加权**(如 dagger_new 翻倍),平坦列表不支持,需要另加采样权重——目前没有。

## 4. 两个 scale(数据选择的属性,换数据必须重测)

| scale | 值 | 测量 |
|---|---|---|
| `FK_DISPLACEMENT_SCALE` | **0.043357** | `tools/scan_fk_displacement_scale.py --root .../dropper_dagger_mix_fk21 --episodes examples/dropper_dagger_mix_71_episodes.txt`,71 集 1.38 亿元素 |
| `POINTFLOW_DISPLACEMENT_SCALE` | **0.037493** | 对 `pointflow_outputs/scale_scan_dagger_mix_strat500_20261001.json` 的逐窗统计做 pooled per-element std;同一协议在 dropper-101 扫描上复现 0.038031,与生产值分毫不差 |

对照:sandwich 0.083745 / 0.0528,dropper-101 0.040438 / 0.03803。**不要跨数据集复用。**

## 5. 启动

参数全部固化在 `tools/run_fk_point_dagger_mix.sh`(数据路径、双 scale、
`POINTFLOW_TOKEN_MODE=per_point`);命令行只带拓扑和步数。

8 卡冒烟(100 步):

```bash
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-dagger-mix-smoke-$(date +%m%d%H%M) \
NPROC_PER_NODE=8 PER_RANK_BATCH=16 MAX_ITER=100 \
bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/run_fk_point_dagger_mix.sh
```

16 卡(两节点,两台执行同一条):

```bash
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-dagger-mix-16n-$(date +%m%d) \
SENSECORE_PYTORCH_NNODES=2 NNODES=2 NPROC_PER_NODE=8 \
PER_RANK_BATCH=16 MAX_ITER=20000 \
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/run_fk_point_dagger_mix.sh
```

预期 ~11.5 s/step(与 sandwich/dropper 同配置),20000 步 ≈ 64 h。

冒烟验证点:banner 显示 `71 episodes`、`per_point`、`fk_scale=0.043357`、
`pointflow_scale=0.037493`;日志无 FK/pointflow 缺数据报错。

## 6. 纪律与坑

- **不要 resume sandwich 的 checkpoint**:动作空间是 joint 不是 EEF,同 27 维不同语义。
- resume 本线 checkpoint 时 wrapper 会校验双 scale 与当前数据一致(继承自
  `run_fk_point_101.sh` 的守卫)。
- dagger 的评估语义:fk_eval / fk_rollout 的 val case 来自 mix 内的 episode;与
  sandwich run 的指标**不可直接比**(任务、动作空间、scale 都不同)。
- 该数据集没有单独的 baseline 原生 run;如需对照,用 dropper-101 的训练作参照系,
  但注意 71 集 vs 101 集、DAgger 分布偏移两点差异。

## 7. 变更清单(2026-10-02)

| 文件 | 说明 |
|---|---|
| `_base/tools/run_fk_point_dagger_mix.sh` | 新;dagger-mix 入口 wrapper,exec 进 `run_fk_point_101.sh` |
| `_base/examples/dropper_dagger_mix_71_episodes.txt` | 新;71 集 allowlist |
| 其余 | 零改动;sandwich / dropper-101 入口行为不变(各自 DRY_RUN 回归通过) |

## 8. DAgger 头部相机支架与 FK 投影（2026-10-02）

DAgger 使用不同的头部支架，不能直接使用 sandwich/dropper 的头部相机外参。
用户确认新 STL 为毫米制，仅替换支架，原底座、底部安装孔和相机安装方式不变。

- 专用 URDF：`assets/dagger_fk/marvin_wuji_d435_dagger.urdf`。
- 米制且已对齐原支架坐标系的网格：`assets/dagger_fk/head_camera_bracket_dagger_m.stl`。
- 重建：`python tools/build_dagger_camera_urdf.py`，依赖 numpy/scipy/trimesh。
- 几何推导：`assets/dagger_fk/construction.json`；验证和新旧相机外参：`assets/dagger_fk/validation.json`。

以新旧底部 50×20 mm 四孔对齐，按顶部 45 mm 双孔及接触面保持原相机安装关系。
`head_d435_mount_joint` 的 yaw 从 40° 变为约 50.998°（这是支架局部坐标系的角度），
xyz 从 `[-0.048788402190, 0.146132595160, 0.020000000000]` 变为
`[-0.062194646412, 0.139900323023, 0.019003303526]` m。光心偏置与原来的 180° 图像 roll 保留。
该结果由机械安装几何推导，没有根据视频人为平移骨架；旧、新两批各一集的投影明显改善，仍有局部残差。

只改变相机安装关节；20 组随机姿态验证手部和 TCP 基座系 FK 完全一致。
因此现有 `Link_Base` 系的 `wuji_fk21.npz` 不需重算。
**训练已接入：DAgger wrapper 固定设置 `FK_CAMERA_PROFILE=dagger`，经数据集配置传到 `FKSource`，
同时用于 anchor 和 displacement。默认 `legacy` 继续使用原 `fk_camera_extrinsic.py`，原文件未修改。**
新外参独立保存在 `cosmos_framework/data/fk_camera_profiles.py`，测试与专用 URDF 对照。
更换 URDF 后需同步该文件中的 DAgger 常量。旧外参训练的 checkpoint 不能直接 resume 到新配置，
启动脚本会检查保存的 camera profile；请使用新的 OUTPUT_ROOT。
新外参全量复扫 71 集后，六位小数的 FK scale 仍为 `0.043357`。
训练及使用相同数据集配置的 FK eval/rollout 均自动使用对应外参；无需重建 FK、PointFlow 或 VAE 缓存。

新支架未提供质量/惯量，专用 URDF 删除该固定支架的旧惯性参数，仅用于 FK/几何投影，
不是经过验证的动力学模型。其余网格使用原 `/data/shichaojian/wuji-mjlab/` 包的绝对路径。
