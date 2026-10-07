# RTX 新机器交接：RGB 二维跟踪 + 对齐深度 → PointFlow XYZ + FK

日期：2026-10-03。面向接手本仓库的新 agent。本文是实施任务书，**不是已完成实现的说明**。

## 0. 方法语义澄清：不能把仿真状态当作视觉桥接输入

已重新阅读训练仓库 `docs/pointflow_quickstart.md`：目标是缓解 video-action gap，point 与 FK 共享三维 token/位置编码，并与 video/action 联合去噪。当前 quickstart 还明确说 pointflow 分支未组合 FK 路径；不能假定四模态已在该分支全部就绪。

**本交接中的仿真状态访问只用于离线标签生成、可见性审计和明确标注的 oracle 对照，不是让模型在部署时读取物体真值位置。** 物体位姿来自 HDF5 `objects/<id>/pose_world` 或仿真状态读取接口，这不是从视频估计出的结果。若把这些值直接作为 PointFlow 输入，就改变了方法的观测条件，不能用于声称 RGB 到 action 的桥接成立。

**用户已确认主线为 RGB-D 视觉提取，不再需要重新询问方案选择：**

```text
RGB 视频 → 二维点跟踪 → UV 像素轨迹 + tracking_valid
                             ↓
               同时刻对齐深度 Z + 相机标定
                             ↓
                  XYZ 三维 PointFlow + valid

关节状态 + URDF → FK21 → 与 PointFlow 相同坐标系

mesh + 仿真状态 → 独立表面轨迹 GT → 仅监督/评估，不回填视觉观测
```

仿真使用渲染深度，真机对应 RGB 对齐后的 D435 深度；这是 RGB-D 方案，不是 RGB-only。渲染深度没有真实传感器的噪声和缺失，首版为理想 RGB-D 基线，不宣称已经消除 sim-to-real 差距。不使用 Track4World/DA3，但仍需要一个以 RGB 为输入的二维点跟踪器。

特别检查 anchor：即使未来轨迹只作标签，首帧 `anchor_xyz` 若来自完美渲染深度、anchor 语义若来自 instance 真值，也属于观测权限变化。真机使用 RGB-D 时应有对应传感器来源；RGB-only 则必须估计深度或去掉该真值条件。仿真实例 ID / link ID 只在标签生成器内部使用，不能自动成为模型输入。

quickstart 中的 GT 运动排名选点可能依赖未来轨迹。部署形态实验必须使用当前/历史可计算的 query 选择，并检查 eval 是否复用了离线未来排名结果。conditional eval 中 clean GT video/action 的诊断不能替代 joint rollout 或闭环评测。

按已确认的 RGB-D 主线直接实施阶段 B。分开保存 `labels_gt/` 与 `observations/`，独立审计数据来源；禁止用真实物体位姿、mesh 绑定点或 GT 可见性静默修正视觉跟踪输出。训练伪标签来自离线视觉轨迹，仿真 GT 可作额外监督/上限对照，必须标注来源。

## 1. 用户目标与边界

将用户的 PointFlow + FK 方法接入 Bench2Dex，随后训练并做仿真闭环评测，后续还要上真机。

已决定：

- 主 PointFlow 由 RGB 二维跟踪 + 同时刻对齐深度反投影生成，不使用 Track4World 或 DA3。mesh/状态生成独立 GT，仅用于监督和验收。
- 尽可能遵循真实相机的观测限制：只将当前可见的表面点作为视觉几何观测，不把完整机器人或物体背面点云直接喂给模型。
- RGB 同时用于二维跟踪和模型视觉输入。深度用于把 UV 提升为 XYZ，并辅助过滤边缘和不可靠观测；不是仅用来判断遮挡。
- 仿真可知道遮挡点的真实轨迹，但 `geometry_valid`、`visible`、`input_valid`、`target_valid` 必须分开。未来真值只能作为标签，不能泄漏到输入。
- 首先完成一个 episode 的数据导出和可视化验收，再接训练，再扩到多任务；不要一开始批量渲染或改模型。
- 不需要申请原始 Track4World/DA3 权重。仿真阶段不需要复跑以前的 DA3 实验。

状态生成的机器人表面 GT 是 FK 的稠密几何表达；视觉提取的 PF 是独立估计，不应强行令其等于 FK。FK21 是另一组稀疏关键点，不能要求表面点与关节点重合。物体运动是机器人 FK 本身不包含的信息，必须纳入后续对照。

## 2. 仓库与当前状态

```bash
git clone https://github.com/NaOH678/Bench2Dex-Pointflow-fk.git
cd Bench2Dex-Pointflow-fk
git switch -c feat/sim-pointflow
```

历史实验已推送的提交：`257113b28fe00658ab6fa4cb531766f03a571b21`。使用包含本文的新提交，并记录实际 HEAD；不要退回旧提交导致丢失本文。

仓库保留原 Bench2Dex 历史。`analysis/` 包含以前的几何审计、隔离 DA3 修复、结果与报告。约 30 GB 中间数组、权重、本机符号链接目标及大型资产未提交。**clone 完成不代表资产和数据已经就绪。**

可选的历史 Pi3 子模块与本任务无关，不要求递归初始化来运行本阶段。不要为它安装整套 Track4World 环境。

先阅读：

1. `README.md`：环境、数据来源和资产目录布局。
2. `replay.py`：运动学回放、相机采集、原始状态写入。
3. `collector/cameras.py`：相机模型、深度语义、外参输出。
4. `utils/replay_support.py`：相机数据写回 HDF5。
5. `configs/collect/default.yaml`：相机与采集配置。
6. `analysis/geometry_audit_20261002/audit_geometry.py`、`export_robot_surface.py`：历史 USD/URDF 几何变换参考。
7. `analysis/experiment_report/PointFlow_FK_实验总结/实验总结.md`：已有结论和附件。

历史审计脚本包含 `/tmp` 和旧机器绝对路径，只能参考逻辑，不能直接认为可移植。新的实现使用 CLI 参数和配置，不硬编码 `/mnt/afs`、`/data/shichaojian`。

## 3. RTX 环境与资产准备

### 环境

先记录 GPU 型号、驱动、操作系统、可用显存与磁盘空间，检查所选 Isaac Sim 版本的官方兼容要求。需要支持渲染的 GPU，不能只检查 `torch.cuda.is_available()`。先证明 headless RGB/depth 渲染能运行。

本仓库 README 当前声明的环境基线如下；这是仓库兼容基线，不是要求追逐软件最新版本：

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 Flask h5py
```

在仓库旁边克隆 IsaacLab `v2.3.2`，按 README 安装 `IsaacLab/source/isaaclab` 为 editable。环境与训练环境分开，不要用训练 PyTorch 版本直接覆盖 Isaac 依赖。额外几何依赖按实际实现安装并记录版本。

### 资产与 episode

官方资源入口（来自仓库 README）：

- https://huggingface.co/datasets/Bench2Dex/Assets
- https://huggingface.co/datasets/Bench2Dex/teleopdata
- https://modelscope.cn/datasets/Bench2Dex/Bench2Dex
- https://modelscope.cn/datasets/Bench2Dex/teleopdata

先检查远端文件列表，仅下载首个任务需要的资产、依赖和 episode；不要假定历史临时文件还存在，也不要直接下载所有 checkpoint。

首个验收目标：`21_condiment_box_loading`，episode `000000`。历史样本为 713 帧、20 Hz、52 个机器人关节，约 35.65 秒。新下载样本需要重新核对这些元数据，不把它们当作强制覆盖值。

旧机器曾使用下列本地文件名，它们不是远端下载路径，也没有随 Git 提交：

- `/tmp/bench2dex_origin21_ep0.hdf5`
- `/tmp/bench2dex_replay21_ep0.hdf5`
- `/tmp/bench2dex_wuji.urdf`
- `/tmp/bench2dex_usd/Multi_UR5_wuji_with_flange.usd`

记录下载仓库 revision、文件 SHA256、机器人 URDF/USD、对象 USD、场景 YAML、相机配置。解析 USD 引用和纹理路径，保证使用匹配版本；不要用“能加载的类似资产”替代。

`replay.py` 读取：

- `meta/scene_file`、`robot_key`、`fps`、`frame_count` 等；
- `robot/qpos` 和 `robot/joint_names`；
- `objects/<id>/pose_world`，其格式为 `[x,y,z,qx,qy,qz,qw]`；
- 有关节对象的 `joint_names` 与 `qpos`。

物体记录的四元数是 xyzw，Isaac 写入接口可能要求 wxyz，必须沿用回放代码中的显式转换。

## 4. 阶段 A：先验证真实可用的回放

第一轮用 `cam_overhead` 针孔相机，避免同时引入腕部鱼眼投影；通过后再扩到实际模型使用的视角。不要启用随机重采样。

在激活 Isaac 环境且回到仓库根目录后，以下为现有 CLI 的调用模板：

```bash
python replay.py \
  --hdf5 /ABS/DATA/episode_000000.hdf5 \
  --cameras cam_overhead \
  --enable-rgb --enable-depth --headless \
  --output /ABS/OUTPUT/episode_000000_rgbd.hdf5 \
  --save-sample-frames /ABS/OUTPUT/replay_samples
```

先替换全部占位路径。必要时指定匹配的 `--scene` 和 `--collect-config`。若原数据包含场景随机化记录，审查后用 `--restore-generalization` 恢复原场景，不能盲目用默认场景。没有相应记录时不要强行加此参数。

README 提到发布数据通常未保存深度，而代码存在 `--enable-depth` 采集路径。**这只能证明接口存在，不能证明当前机器实际渲染已成功**；必须检查生成 HDF5 里深度是否有正确尺寸、有限值、合理单位及帧数。

现有 CLI 没有 `--max-frames`；如需前 5–10 帧冒烟测试，先新增明确的帧范围参数并保存源帧编号，不要直接传不存在的参数。之后再跑完整 713 帧。

同步验收：同一 frame 的机器人状态、物体状态、相机位姿、RGB、depth、instance 标识必须属于同一次状态更新。回放以写状态 + render 为主，不额外 physics step 让物体漂移。检查渲染器是否有一帧延迟和充分 warmup。

历史曾发现腕部外参与 qpos 存在前一帧对应现象，但未证明 RGB 同样延迟。新机器必须用投影验证，不可盲目整体平移一帧。

## 5. 阶段 B：实现视觉 RGB-D PointFlow，另建独立 GT

建议新增 `tools/export_rgbd_pointflow.py`（主导出器）与 `tools/export_sim_pointflow_gt.py`（真值工具）。这些文件**尚未实现**，不是可直接执行的已有命令。

### 5.1 二维跟踪必须来自 RGB

新 agent 选择并记录一个合适的二维点跟踪器、模型版本/权重/许可证，先对少量帧做冒烟测试。不指定未经本环境验证的安装命令，也不为此重新引入 Track4World/DA3。

- 输入 RGB 和 query `(frame,u,v)`，输出固定视觉轨迹 ID、UV、跟踪有效性/可见性/置信度（依实际模型支持字段记录，不能伪造）。
- query 由当前或历史图像选取，先用像素网格/图像可计算的区域策略。仿真 instance ID、未来 GT 运动排名不得作为默认输入选点依据。
- 若使用 RGB 语义分割做手/物/背景配额，记录算法和误差；首版没有分割时可用通用网格。GT 语义只用于分组评估或单独标注的 oracle 对照。
- 首版先验证起始帧 query 的全段轨迹；之后按需求支持新出现区域的 query，保留 birth_frame，不用未来数据修改过去的 query 选择。
- 可用全视频离线跟踪生成训练标签，但这不是在线算法；闭环输入只能用当前/历史帧。不要将离线全序列平滑或未来排名结果当作可部署条件。
- 遮挡重识别依跟踪器能力；不使用仿真永久材料点 ID 替代视觉 track ID。

### 5.2 用同时刻深度将 UV 提升到 XYZ

每个有效二维观测 `(u_t,v_t)` 读取同一时刻、已对齐 RGB 的深度 `Z_t`。对于去畸变针孔图像、轴向 Z 深度：

```text
p_camera(t) = Z_t * inverse(K_t) * [u_t, v_t, 1]^T
X = (u-cx)*Z/fx; Y = (v-cy)*Z/fy
```

- 使用已标定 K，不调用 DA3 估计内参。RGB resize/crop 后同步变换 UV/K，明确像素中心约定。
- 先验证 `depth_semantics`：轴向 Z 与沿射线欧氏距离不是一回事。鱼眼需对应投影/反投影，不能套针孔公式。
- 浮点 UV 的深度采样要处理表面不连续，不盲目在前景/背景间双线性混合。参数化邻域、深度跳变、边界和无效值过滤。
- 二维轨迹被遮挡后，UV 处深度可能属于遮挡物。即便采样深度有限也不保证该点有效；结合跟踪可见性、置信度和深度质量判断，疑似遮挡标无效。GT 检查其错误率，但不得用 GT 可见性直接替换预测 mask 后仍声称是纯观测结果。
- `tracking_valid`、`depth_valid`、`xyz_valid` 分开。无有效深度时保留二维轨迹，不因深度缺帧切断 ID；不静默填 DA3、上一帧深度或真值 XYZ。
- 真机按真实 RGB/depth 时间戳匹配并设容差；约 30 Hz RGB / 6 Hz depth 时，首版只在可靠匹配时刻提供有效 XYZ。仿真先同步全帧验证，再按实际时间采样模拟低频深度。

初期使用固定 overhead 相机。若扩展到移动相机，记录每帧相机坐标和可获得的相机位姿，按训练接口统一成世界或窗口起始相机参考系；不能将各帧相机坐标直接当作同一固定空间。真机端使用同等可获得的相机位姿来源。

### 5.3 坐标格式：UV + XYZ，训练预测 ΔXYZ，不是 UVZ

已核对当前真实数据导出和 loader：

| 当前字段/表示 | 实际含义 |
|---|---|
| `obs_uv.npy` | `(M,2)` 像素 `(u,v)`，历史导出为 640×448 跟踪网格 |
| `obs_pos.npy` | `(M,3)` 三维 `(X,Y,Z)`，单位米；历史数据来自 DA3/Track4World |
| `anchor_xyz` | 窗口起始点三维坐标 |
| `target_displacement` | 对应未来 `XYZ_t - anchor_xyz`，后续经过训练归一化 |
| `(t,h,w,z)` | token 的位置编码设计，不是轨迹存储格式，也不是 ΔXYZ 的替代 |

新方案替换 XYZ 的生成来源，继续保留独立 UV，并适配原有 anchor/ΔXYZ 接口。不要把 `[u,v,Z]` 写进 `obs_pos`。

建议中间格式包含 `source_frame_id`、时间戳、`track_id`、`birth_frame`、`uv`、`depth_z`、`xyz_camera`、参考系、`tracking_valid`、`depth_valid`、`xyz_valid`、置信度、K/相机模型和 provenance。精确 dtypes、稀疏排序、类别编码及窗口 shape 以 converter/loader 为准。`depth_z` 可作诊断保留，但模型主位置仍是 XYZ。

GT 的 mesh/link/instance 绑定、真值可见性、永久材料点 ID 单独保存在 `labels_gt/`。无效占位坐标必须由 mask 排除 attention 空间索引和 loss。

### 5.4 独立真值工具：检验对应点，而不替代跟踪

- 在 query 起始帧沿真实相机射线与渲染 visual mesh 求交，建立同一 query 的真值表面绑定 `(object,link,mesh,triangle,barycentric)`。
- 用仿真物体位姿/关节状态更新同一材料点的 GT 轨迹，生成 GT UV/XYZ/visible，用于评估二维漂移、三维误差和遮挡误判。
- 不能在每帧沿**预测 UV**重新找最近表面作为唯一真值：即便轨迹已滑到另一处，该做法也可能报很小误差。主要轨迹指标必须比较初始绑定的同一材料点。
- visual origin、嵌套 USD 变换、缩放、实例、根位姿、矩阵行列方向必须正确；使用 visual mesh，不用 collision mesh 替代。状态四元数顺序按源格式转换。
- 对有独立 link 的关节物体逐 link 处理。第一版不宣称支持未保存变形状态的柔性物体。
- GT 与观测导出器分离；GT 缺失不应使视觉轨迹算法改用其他观测值。

### 5.5 FK 模态

从关节状态和 URDF 按现有 FK21 关键点定义计算，映射关节名称和关键点局部偏移；不能随意把 link origin 当成全部 FK21。FK 与视觉 PF 使用同一帧及相机参考系。

额外用同一个表面绑定点比较直接仿真 link 变换与独立 FK，以检查几何实现；这是 GT 管线一致性测试，不是视觉 PF 精度。视觉 PF 与 GT 的差异应如实报告，不能通过 FK 拟合把它抹掉。

## 6. 阶段 C：接用户现有训练仓库

本仓库不是完整 WorldAct 训练仓库。旧机器路径仅用于定位，不意味着新机器上已经存在：

- `/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow`
- `/mnt/afs/WorldAct-cosmos3-edge-droid-sft_mano`
- `/mnt/afs/WorldAct-pointflow-native`

历史观察到训练仓库 remote 为 `https://github.com/liuhangxu-robin/WorldAct.git`，对应工作分支包括 `WorldAct-cosmos3-edge-droid-sft-pointflow` 与 `WorldAct-cosmos3-edge-droid-sft_mano` 和 `WorldAct-cosmos3-edge-droid-sft-pointflow-fk`(这是当前最新版方案)。获取时重新确认所需分支、revision、访问权限和本地未提交改动，不能假定 remote 分支已包含旧机器所有代码。缺少训练仓库时先完成独立导出与验收，不阻塞阶段 A/B。

PointFlow 训练仓库首先阅读 `docs/pointflow_quickstart.md`。重点审查：

- `tools/convert_efep_labeled.py`：实际 schema、类别、轨迹编号与有效掩码。
- `cosmos_framework/data/pointflow_window.py`：窗口采样、anchor、有效步数和未来目标 mask。
- FK21 定义、FK/相机坐标变换与 joint loader。
- `fk_camera_extrinsic.py` 中真实数据外参只用于真实数据，不能套到仿真相机。

在 adapter 中对应现有 `obs_track`、`obs_uv`、`obs_pos` 等字段，确认其真实 shape/dtype/排序；不要仅凭字段名臆造兼容性。

首先明确 PointFlow 是模型的输入、未来预测目标，还是两者都有。只有标签分支可访问未来几何。仿真真值输入实验应明确标记为 oracle geometry，不能直接当作 RGB-only 真机能力。

第一轮保留原 benchmark 帧率，不伪造 30 Hz。后续对齐真机传感器时，实测 RGB 约 30 Hz、D435 深度约 6 Hz 仅是此前数据的观测值，需按目标采集配置设置；不能直接把 20 Hz 的每五帧当成 6 Hz。

当前主线是真机 D435 与仿真渲染深度对应的 RGB-D 管线。先做理想 RGB-D 基线，再依据实测传感器缺失、噪声和同步误差设计对照；若另做 DA3 输入实验，要单独标记，不混称为相同输入设置。

## 7. 验收标准与交付物

先完成以下检查，再扩数据和训练。阈值是初始工程目标，不是已经达到的结果；如果精度不满足，先解释原因而非放宽到厘米级掩盖坐标错误。

### 几何及状态

- 同一表面绑定点：直接 link 变换与独立 FK 结果误差以小于 1 mm 为初始上限，报告 median/P95/max；纯代数测试应接近数值精度。
- 点投影与 mesh/深度应一致，非边界样本重投影以约 1 pixel 为初始目标；分别报告不同视角与机器人/物体结果。
- 静态物体在世界坐标不漂移，移动相机下其相机坐标变化符合相机位姿。
- GT 固定 ID 不换表面；视觉 ID 是否滑点由同材料点 GT 评估。关节映射无漏项，四元数与 mesh scale 正确。
- 状态、RGB、depth、相机位姿同帧；专门做运动中的 ±1 帧对照检查。

### 可见性及观测限制

- 相机外、背面、自遮挡、物体遮挡、重新出现至少各有一个可视化案例。
- 深度无效、轮廓像素、多个表面竞争有明确处理。
- 分别保存估计 visible 与 GT visible，评估遮挡误判；输入没有未来新增点、未来 GT 排名或隐藏位置泄漏。
- 汇报各类可见点覆盖、轨迹寿命、缺失比例；不能只筛出少量好点报告极低误差。

### 视觉轨迹指标（与纯几何自检分开）

- 相同初始 query / 同一 GT 材料点：二维像素轨迹误差、三维 ADE/末帧误差，报告 median/P95、手/物/静态分组及覆盖率。
- 几何反投影再投回原 UV 是代数自检，不证明二维跟踪正确；不要用它代替同材料点误差。
- 额外提供 oracle 2D UV + 同一深度的诊断对照，区分二维跟踪误差与深度提升误差；该对照不得当作主实验。
- 验证缺失深度不改变 track ID，遮挡时不读取前景深度冒充原点，非因果离线结果不进入在线输入。

### 输出

建议新建 `analysis/sim_pointflow_validation/`，输出：

1. 环境与资产 manifest（版本、hash、帧范围、配置）。
2. 原始格式说明和 adapter 字段映射。
3. 该 episode 的 RGB 叠加点轨迹视频，标注 frame、ID、类别及可见性。
4. 机器人表面 PF / 同点 FK / FK21 的三维对照，清楚区分表面与骨架。
5. 物体轨迹与遮挡示例。
6. JSON 指标、失败帧清单、Markdown 小结及复现命令。
7. 数据体积、耗时和峰值显存，作为批量规划依据。

大数组、渲染深度、资产和权重不要提交 Git；保留生成脚本、小规模指标与代表媒体。不要覆盖历史实验结果，也不要提交凭据。

## 8. 训练与 benchmark 顺序

只有阶段 A–C 通过后才进行：

1. 单 episode 过拟合，验证 loader/mask/loss/预测轨迹。
2. 单任务多 episode，以 episode 划分训练与验证，避免相邻窗口跨集合泄漏。
3. 等预算对照：FK；FK + 机器人表面 PF；FK + 机器人及物体 PF。控制采样量并说明是否包含静态点。
4. 检查移除／打乱 PF 的消融，确认模型确实利用该模态。
5. 接 benchmark 动作接口，明确动作表示、单位、控制频率、关节顺序和闭环执行流程。未来真值不能作为策略条件。
6. 使用任务成功率等 benchmark 指标，而不只比较 PF loss；对比需要一致的观测权限。
7. 最后扩任务、相机和传感器扰动，并准备真实数据微调。

## 9. 发给新 agent 的启动指令

> 请先阅读本仓库 `SIM_POINTFLOW_AGENT_HANDOFF.md` 及 WorldAct PointFlow 仓库的 `docs/pointflow_quickstart.md`。用户已确认主线是“RGB 视频 → 二维点跟踪 → UV；同时刻对齐深度 + 相机标定 → XYZ PointFlow”，不使用 Track4World/DA3，不把仿真物体位姿当成视觉输入。先在 RTX 机器检查环境、下载任务 21 episode 000000 的匹配资产和数据，跑通短帧 RGB/depth 回放，再实现二维跟踪和深度提升。另建 mesh/状态真值工具，只用于同材料点轨迹评估和显式监督对照。保持 obs_uv 与 obs_pos/XYZ 分开，训练沿用 anchor_xyz 和 ΔXYZ；处理遮挡、无效深度、时间同步与首帧条件的信息边界。先交付完整 episode 的 RGB 叠加视频、独立 GT 对照、指标与复现命令，再接 WorldAct loader 和训练。方案无需再次确认；新工具尚未实现，旧机器路径不可假定存在。
