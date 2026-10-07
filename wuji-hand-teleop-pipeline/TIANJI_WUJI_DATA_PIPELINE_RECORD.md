# Tianji + Wuji 数采、遥操诊断与 Replay 开发记录

更新日期：2026-07-30

本文记录当前工作区中 Tianji 双臂、Wuji 双手、HTC Vive Tracker、MANUS
手套、三路直连相机、ROS 2机器人数据与Replay管线，包括需求边界、参考来源、数据格式、
同步方法、安全策略、实现文件和现场常用命令。

机械臂与机械手的控制方式不在本文重新定义。真机控制必须继续遵循：

- [TIANJI_VIVE_TELEOP_RECORD.md](TIANJI_VIVE_TELEOP_RECORD.md)
- [MANUS_WUJI_INTEGRATION_RECORD.md](MANUS_WUJI_INTEGRATION_RECORD.md)

当前完整链路为：

```text
数采：
SteamVR + 5 Tracker -----> openvr_input ---------> Tianji 遥操 TF
                              |                         |
                              +-> raw/corrected 诊断    +-> Tianji 双臂

MANUS -> ManusGlove ------> Wuji retarget ----------> Wuji 双手
              |                                      |
              +---------------> raw/21点诊断          |

三路相机 -> 独立 Camera Manager -> /dev/shm 有界环形缓冲
                                      |-> GUI 最新帧
                                      |-> Recorder RGB相机帧
                                      `-> 可选深度/头部双红外

机器人反馈 + 控制目标 + Camera Manager帧 + 遥操诊断
  -> wuji_teleop_recorder
  -> LMDB + MP4 episode

Replay：
LMDB episode -> replay_server/ZMQ policy 接口
             -> wuji_deployment
             -> Tianji external EEF/ZSP + WujiHand joint command
```

## 1. 当前结论与验证状态

- 遥操控制代码沿用已经验证的 Tianji Vive 和 MANUS Wuji 方案，没有另写一套
  Tianji/Wuji 控制器。
- 机器人、手和遥操诊断使用ROS 2 topic；相机图像默认不经过ROS 2。数据集同步频率
  为30 Hz。
- 当前无相机模式已经可以记录双臂和单只右手；未接入手保持 54 维格式并补零。
- 已增加独立 PyQt5 数采 GUI。GUI 统一启动 controller、手、相机和 recorder，
  不需要再单独执行一条遥操 launch 命令。
- Episode 采用“结束采集”和“确认保存”两阶段状态；结束后可保存，也可在下一次
  开始时丢弃。
- 已预留三踏板硬件适配接口，GUI 中的三个模拟踏板按钮走完全相同的状态保护。
- 相机位置上限固定为一个主视角和两个腕部视角，共三路彩色图像。GUI 只显示
  这三路 RGB；每条 episode 只把开始采集时在线的 RGB 子集作为训练图像。
- 三台 RealSense 的原始 `uint16` 深度和头部 D435 左右红外作为分析用辅助数据
  best-effort 保存，不加入训练 schema、readiness、GUI、deployment 或 Replay。
  辅助流缺失、掉帧或写队列满都不会拒绝或删除原训练帧。
- 阶段D已经将GUI、Recorder和Deployment的正式相机路径改为Camera Manager加
  `/dev/shm`有界环形缓冲；`camera_transport=ros`只作为迁移回退。
- LMDB 主 key 和核心单位语义保持与 `dexmanip_tool` 兼容，但维度按当前硬件调整。
- Replay 走部署管线，而不是由数据读取器直接调用 Tianji SDK。
- Replay 支持 `right`、`left` 和 `both`，单手模式不会等待或命令未接入手。
- Tracker 和 MANUS 原始/中间数据已经作为 `/teleop/*` 可选字段与训练帧对齐。
- 原始诊断数据缺失不会拒绝开始录制，也不会导致原训练帧被跳过。
- 旧 episode 不包含 `/teleop/*` 时仍能读取和 Replay。
- `openvr_input` 与 `wuji_data_pipeline` 构建成功。
- 当前累计相关回归测试最近一次结果为 `178 tests, 0 failures`。
- 已使用一条现有 599 帧 episode 完成 inspect 和 Replay dry-run 验证。

新增 OpenVR 诊断 topic 的真实硬件有效率仍应在下一条新采 episode 后通过
`inspect_episode` 确认。软件实现和协议测试已经完成，但开发时没有为了验证该
topic 而主动启动或移动机器人。

日常操作只需要记住下面两个入口：

```bash
# 0. 确保主容器运行（已经 Up 时重复执行无害）
docker compose \
  -f /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline/docker/docker-compose.yml \
  up -d

# 1. 启动独立数采 GUI（自动启动 OpenVR，默认双手 + 在线相机）
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline && \
  ./src/scripts/start_record_gui.sh

# 2. Replay 指定 episode
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline && \
  ./src/scripts/start_replay_session.sh right \
  /home/wuji/datasets/tianji_wuji/pick_red_block/episode_0000_YYYYMMDD_HHMMSS
```

GUI 的标准顺序是 `Recovery -> Enable -> 踏板1开始 -> 踏板1结束 ->
踏板2保存 -> 退出系统`；Replay 是 `r -> a -> q`。完整含义和安全条件见第
14、16、19 节。

## 2. 需求边界与设计原则

### 2.1 必须保持的行为

1. Tianji Recovery、Enable、阻抗参数和 Tracker neutral 映射保持既有实现。
2. MANUS 25 节点到 21 点、Wuji retarget 和 WujiHand 命令保持既有实现。
3. 启动程序不得自动 Recovery 或 Enable。
4. `qpos`、`action`、`eef` 等训练字段维度必须适配 Tianji 7 DoF 和 WujiHand
   20 DoF，不能照搬其他硬件维度。
5. 三路相机为 `head`、`left_wrist`、`right_wrist`，不增加虚构的第四路相机。
6. 三路 RGB 保持原训练格式；三路深度和头部双红外只作为独立辅助信息保存。
7. Replay 必须能够使用本地 server，也能够替换为云端 policy server。
8. 单手阶段仍保持双臂双手训练协议，缺失手明确补零并写入 metadata。
9. `Ctrl+C` 必须退出整个本次 session，不等待 recorder 的保存服务；需要保存时
   必须先结束采集，再执行保存并等待成功。

### 2.2 原始诊断数据的要求

新增 Tracker/MANUS 数据是为了分析遥操误差，而不是改变控制：

```text
手臂分析链：
OpenVR raw pose
  -> wrist offset / role correction 后 pose
  -> Tianji mapped target EEF
  -> 实际 EEF / 实际 joint

手部分析链：
MANUS raw 25-node skeleton
  -> 控制器实际使用的 21x3 keypoints
  -> 20-DoF retarget target
  -> 实际 WujiHand joint
```

因此诊断字段采用旁路设计：

- 不加入 recorder 的 required source 列表。
- 不影响 `s` 是否允许开始。
- 不影响原始同步误差统计。
- 不改变训练帧 anchor。
- 缺失或格式错误时使用填充值，并将 `available/valid` 置零。
- Replay 不读取这些字段。

## 3. 参考了 dexmanip_tool 的什么

参考仓库：

```text
/home/pjlab/code/dexmanip_tool
参考分支：dual-tele、tracker_remote
```

### 3.1 保留的参考语义

- LMDB scalar key 命名和 episode + video 的整体组织方式。
- observation/action 分离。
- `action` 中的机械臂部分保存机器人实际执行的 EEF 轨迹，而不是上游 Tracker
  原始目标。
- 手部 observation 使用弧度，手部 action 使用度。
- Python pickle + ZMQ REQ/REP 的部署协议形式。
- Replay server 伪装成 policy server，机器人侧使用同一 deployment client。

### 3.2 没有照搬的部分

- 没有使用 `dexmanip_tool` 中的 Tianji 控制代码。
- 没有替换已经验证的 Tianji Vive controller、Recovery、Enable 或 neutral 映射。
- 没有照搬其他机械手的 12 DoF 或其他维度。
- 没有照搬四路训练相机结构；新增深度/红外保存在独立辅助目录，不改变原
  `camera_names`、训练 LMDB 或 RGB 视频。
- 没有使用 PICO；当前输入设备是 HTC Vive Tracker + MANUS。
- 没有假定 `/manus_glove_0` 永远对应某一侧，左右路由使用 `msg.side`。
- 没有让 Replay 直接控制 SDK；Replay 始终通过 ROS 2 deployment/external 管线。

这部分边界很重要：参考工具决定数据和网络协议的兼容方向，真机控制行为则由
本仓库两份已经验证的遥操文档决定。

## 4. 数采软件结构

### 4.1 一键数采启动关系

宿主机入口：

```text
src/scripts/start_record_session.sh
```

它执行以下工作：

1. 检查主容器 `wuji-hand-teleop` 是否运行。
2. 检查独立 OpenVR 容器 `wuji-openvr-input` 是否运行。
3. 检查 OpenVR 容器是否使用 `ROS_DOMAIN_ID=112` 和 CycloneDDS。
4. 以交互式 TTY 进入主容器。
5. 主容器固定使用 `ROS_DOMAIN_ID=112`、FastRTPS 和禁用 ROS CLI daemon。
6. 启动 `wuji_data_pipeline record_session`。

`record_session` 再启动：

```text
record.launch.py
  +-> vive_arm_tf.launch.py
  +-> 已验证的 tianji_arm_controller
  +-> wuji_teleop_hand.launch.py
  |     +-> 单个 MANUS publisher
  |     +-> 选定侧的 hand controller / driver
  +-> 默认独立 camera_manager（无ROS图像topic）
  |     `-> camera_transport=ros时回退camera_launch.py
  +-> wuji_teleop_recorder
```

OpenVR 不在 `record.launch.py` 中重复启动，因为 SteamVR 输入由独立容器管理。
MANUS publisher 则由 hand launch 统一拥有，禁止在 launch 外再启动第二份。

### 4.2 为什么不需要必开的 lifecycle 终端

一键 cockpit 已订阅并显示：

```text
/tianji_arm/lifecycle_state
/tianji_arm/teleop_status
/wuji_teleop_recorder/status_text
```

它会显示 `INITIALIZING`、`RECOVERING`、`RECOVERY_READY`、`ENABLING`、`READY`
等状态，并在 `r` 前检查节点、TF 和机械臂状态。

因此单独的 lifecycle 终端不是必需启动项。额外终端只用于排障，不参与控制。

### 4.3 独立数采 GUI 的进程边界

宿主机入口为：

```text
src/scripts/start_record_gui.sh
```

GUI 自身运行在宿主机，原因是它需要拥有一个 PTY，并在该 PTY 中执行宿主机脚本
`start_record_session.sh`。后者再进入 `wuji-hand-teleop` 容器启动完整 ROS graph。
这样原有 cockpit 仍是 Recovery、Enable、standby 和子进程清理的唯一所有者，
GUI 只是发送与原键盘完全相同的单字符命令，不会再实现一套 Tianji 启停逻辑。

```text
PyQt5 GUI
  -> PTY
  -> start_record_session.sh
  -> docker exec -it
  -> record_session
  -> record.launch.py
```

GUI 进入前先选择旧任务或创建新任务。任务名只允许 ASCII 字母、数字、`_` 和
`-`，且首字符必须是字母或数字。进入准备前可选择 `both/left/right` 和是否启用
相机，默认值是双手和启用相机。启用相机表示启动相机管线并要求至少一路相机在线，
并不要求三个位置全部连接。

## 5. 核心训练数据格式

每个 episode 的主训练布局由 `RobotLayout` 写入 metadata。目前硬件布局为：

```text
Tianji arm: 7 DoF / side
WujiHand:   20 DoF / side
EEF pose:    7 values / side = xyz + quaternion xyzw
```

### 5.1 qpos / qvel / effort

```text
left arm  7  [0:7]
left hand 20 [7:27]
right arm 7  [27:34]
right hand 20[34:54]
total = 54
```

| 字段 | LMDB path | 形状 | 单位/语义 |
|---|---|---:|---|
| `qpos` | `/observations/qpos` | `N×54` | 双臂和双手实测关节位置，rad |
| `qvel` | `/observations/qvel` | `N×54` | rad/s；驱动缺失时用相邻实测位置差分 |
| `effort` | `/observations/effort` | `N×54` | 驱动实测 effort，缺失分量为 0 |
| `hand_joint_deg` | `/observations/hand_joint_deg` | `N×40` | 双手实测关节角，degree |

Tianji ROS topic 边界使用度，写入 `qpos/qvel` 前转换为弧度。WujiHand ROS topic
本身使用弧度。

### 5.2 action

```text
left EEF   7 [0:7]
left hand 20 [7:27]
right EEF  7 [27:34]
right hand20 [34:54]
total = 54
```

| 字段 | LMDB path | 形状 | 单位/语义 |
|---|---|---:|---|
| `action` | `action` | `N×54` | EEF 为实际执行轨迹；手为目标角度 degree |
| `action_eef` | `action_eef` | `N×14` | 左右实际 EEF |
| `action_bases` | `action_bases` | `N×6` | 当前无移动底盘，固定为 0 |

机械臂 `action` 保存实际 EEF，而不是 Tracker target，原因是 Replay 应复现机器人
真正执行到的轨迹，而不是可能不可达或未跟上的上游目标。

### 5.3 EEF 和辅助诊断

| 字段 | LMDB path | 形状 | 语义 |
|---|---|---:|---|
| `eef` | `/observations/eef` | `N×14` | 左 EEF 7 + 右 EEF 7，实际 pose |
| `robot_base` | `/observations/robot_base` | `N×6` | 当前固定为 0 |
| `commanded_eef` | `/diagnostics/commanded_eef` | `N×14` | Tianji mapped target EEF |
| `arm_joint_command` | `/diagnostics/arm_joint_command` | `N×14` | Tianji 关节目标，rad |
| `zsp` | `/diagnostics/zsp` | `N×6` | 左右臂 ZSP，各 3 维 |

## 6. 三路 RGB 与辅助深度/红外格式

原训练相机语义固定为：

| 名称 | 直连角色 | 类型 |
|---|---|---|
| `head` | Camera Manager `head` ring | D435 主视角BGR彩色图 |
| `left_wrist` | Camera Manager `left_wrist` ring | 左腕BGR彩色图 |
| `right_wrist` | Camera Manager `right_wrist` ring | 右腕BGR彩色图 |

每路保存为：

```text
episode_xxxx_.../videos/head.mp4
episode_xxxx_.../videos/left_wrist.mp4
episode_xxxx_.../videos/right_wrist.mp4
```

Camera Manager 按 RealSense 序列号或稳定 udev 路径识别相机，在
`/dev/shm/wuji_camera_v1` 中建立独立的有界环：

| ring | dtype/形状 | 用途 |
|---|---|---|
| `head`、`left_wrist`、`right_wrist` | `uint8 H×W×3` | 原 RGB 训练视频和 GUI |
| `head_depth`、`left_wrist_depth`、`right_wrist_depth` | `uint16 H×W` | 分析用原始深度 |
| `head_ir_left`、`head_ir_right` | `uint8 H×W` | 头部 D435 左右红外成像器 |

D435 是“一路彩色 + 双目红外 + 深度”，所以头部左右视角是灰度红外，不是两路
彩色图。深度值不直接等于米；每台相机真实的
`depth_scale_m`、内参、深度/红外到彩色的外参、实际 profile、序列号和 USB
连接类型均写入该 episode 的辅助 metadata，换算关系为：

```text
distance_m = raw_uint16_depth * depth_scale_m
```

GUI 仍然只读取前三个 RGB ring，不解码、不传输也不显示新增的深度/红外。
Recorder 的原 RGB 选择逻辑也不变：踩踏板1开始时检查最近1秒内在线的 RGB
子集，冻结为本条 episode 的 `camera_names`，至少需要一路 RGB 在线。之后插入的
RGB 相机从下一条 episode 生效，episode 中途分辨率变化仍会隔离该 episode。

辅助流是旁路 best-effort 数据：采用独立 `0.06 s` 最近邻对齐，不加入
`camera_names`、source readiness、`sync_skip_count` 或核心同步误差。某个辅助
流离线、写入队列满或单帧损坏时，只在辅助 metadata 中记缺失/丢帧，原训练帧和
RGB 视频继续保存。

当前实机头部 D435 连接为 USB 2.1。已验证在保持 RGB 约 30 Hz 的条件下，头部
Stereo Module 使用 6 Hz 保存深度和左右红外；右腕 D435 在 USB 3.2 下以 30 Hz
保存 RGB 和深度。左腕配置已经支持深度，但序列号仍为占位值，接入后需填写真实
serial。若以后把头部相机移到确认可用的 USB 3.x 链路，才应同步提高
`depth_fps`、`infrared_fps` 和 `infrared_frame_rate`。

旧ROS图像topic仅在显式设置 `camera_transport=ros` 时使用。

## 7. 帧同步策略

### 7.1 无相机模式

无相机时使用：

```text
/left_arm/joint_states
```

作为 anchor，并从高频反馈下采样到配置的 30 Hz。

### 7.2 有相机模式

有相机时使用本条 episode 在线 `camera_names` 中第一路作为 anchor。通常三路都
在线时是 `head`；只有腕部相机在线时则使用配置顺序中的第一路在线腕部相机。每个
新 anchor 帧只尝试生成一个训练帧。

### 7.3 必需来源

每一帧必须匹配：

- 左右臂 state、command、actual EEF、target EEF 和 ZSP。
- `active_hand` 指定侧的 hand state 和 command。
- 有相机模式下，本条 episode 开始时选定的在线相机子集。

当前基础同步容差为 `0.04 s`：

```text
arm scalar:  40 ms
camera:      80 ms
hand:       120 ms
```

任一必需来源匹配失败时，该 anchor 计入 `sync_skip_count`，不写训练帧。

### 7.4 可选遥操诊断来源

Tracker/MANUS 诊断使用独立 `0.10 s` 最近邻容差，不加入必需来源。

每个训练帧的 `sync_timestamps` 都记录：

- anchor 时间。
- 各必需 topic 时间。
- 可选诊断 source 时间和 alignment error。
- 写入时的系统时间。

原始同步误差统计只计算必需来源，不被可选诊断污染。

## 8. Tracker 原始数据实现

### 8.1 OpenVR 新增 topic

`openvr_input` 新增：

```text
/openvr/tracker_diagnostics
类型：std_msgs/Float64MultiArray
频率：30 Hz
```

Tracker 固定顺序：

```text
0 chest
1 left_wrist
2 right_wrist
3 left_arm
4 right_arm
```

每个 Tracker 一行 26 个值：

```text
raw pose 7
corrected pose 7
linear velocity 3
angular velocity 3
detected 1
connected 1
valid 1
tracking_result 1
device_index 1
corrected_valid 1
```

### 8.2 为什么不会改变原 TF 控制

现有 `get_poses()` 返回和 wrist offset/role correction 公式没有改变。

原始状态在同一次 OpenVR SDK pose 读取时复制到内部缓存，然后现有控制路径继续
在自己的返回矩阵上执行 wrist offset 和坐标修正。诊断 publisher 只读取缓存，
不会额外调用一次 OpenVR SDK。

诊断字段提取失败会被隔离，现有 pose 仍照常返回。诊断 publisher 也在原 TF
发布之后执行，异常只产生限频 warning，不会停止 TF。

### 8.3 LMDB Tracker 字段

| path | 每帧形状 | 说明 |
|---|---:|---|
| `/teleop/tracker/raw_pose` | `5×7` | OpenVR 原始 xyz + xyzw |
| `/teleop/tracker/corrected_pose` | `5×7` | 当前控制使用的修正 pose |
| `/teleop/tracker/linear_velocity` | `5×3` | OpenVR 原始线速度 |
| `/teleop/tracker/angular_velocity` | `5×3` | OpenVR 原始角速度 |
| `/teleop/tracker/detected` | `5` | 是否发现配置设备 |
| `/teleop/tracker/connected` | `5` | SDK connected 状态 |
| `/teleop/tracker/valid` | `5` | raw pose 是否有效 |
| `/teleop/tracker/corrected_valid` | `5` | corrected pose 是否存在 |
| `/teleop/tracker/tracking_result` | `5` | OpenVR tracking result |
| `/teleop/tracker/device_index` | `5` | OpenVR device index |
| `/teleop/tracker/available` | `1` | 本训练帧是否匹配到诊断消息 |
| `/teleop/tracker/source_timestamp` | `1` | recorder ROS 接收时间 |
| `/teleop/tracker/alignment_error_s` | `1` | 与训练 anchor 的时间差 |

所有保存数组最终形状都在上述每帧形状前增加 `N`。

## 9. MANUS 原始数据实现

### 9.1 不修改 MANUS 控制消息

recorder 直接旁路订阅现有：

```text
/manus_glove_0
/manus_glove_1
类型：manus_ros2_msgs/ManusGlove
```

没有修改 `ManusGlove.msg`，也没有修改 `wujihand_node.py` 的控制回调。
因为原消息没有 ROS Header，保存的 source timestamp 是 recorder 的 ROS 接收时间。

### 9.2 25 节点和 21 点同时保存

原始节点按 `node_id` 升序写入固定 25 个槽位，同时保存真实 `node_id`、
`parent_node_id`、chain/joint 类型编码和有效位。

21 点使用与控制器相同的语义映射：

```text
Wrist: Hand / Invalid
Thumb: MCP, PIP, DIP, TIP
Index/Middle/Ring/Pinky: PIP, IP, DIP, TIP
```

recorder 中单独实现纯数据转换，不 import 或调用控制器，以免数采对控制代码产生
运行时依赖。回归测试保证语义顺序与控制器一致。

### 9.3 LMDB MANUS 字段

以下 `{side}` 为 `left` 或 `right`：

| path | 每帧形状 | 说明 |
|---|---:|---|
| `/teleop/manus/{side}/raw_node_pose` | `25×7` | 原始节点 pose |
| `/teleop/manus/{side}/node_id` | `25` | 真实 node id |
| `/teleop/manus/{side}/parent_node_id` | `25` | parent id |
| `/teleop/manus/{side}/chain_type_code` | `25` | chain 类型编码 |
| `/teleop/manus/{side}/joint_type_code` | `25` | joint 类型编码 |
| `/teleop/manus/{side}/node_valid` | `25` | 节点槽位是否有效 |
| `/teleop/manus/{side}/keypoints_21` | `21×3` | 实际 retarget 输入语义顺序 |
| `/teleop/manus/{side}/keypoints_valid` | `1` | 21 个必需节点是否完整 |
| `/teleop/manus/{side}/ergonomics_value` | `20` | MANUS ergonomics |
| `/teleop/manus/{side}/ergonomics_type_code` | `20` | ergonomics 类型编码 |
| `/teleop/manus/{side}/ergonomics_valid` | `20` | ergonomics 有效位 |
| `/teleop/manus/{side}/raw_sensor_orientation` | `4` | raw sensor orientation |
| `/teleop/manus/{side}/raw_sensor_pose` | `5×7` | 最多 5 个 raw sensor pose |
| `/teleop/manus/{side}/raw_sensor_valid` | `5` | sensor 有效位 |
| `/teleop/manus/{side}/glove_id` | `1` | MANUS glove id |
| `/teleop/manus/{side}/reported_node_count` | `1` | 消息报告节点数 |
| `/teleop/manus/{side}/reported_sensor_count` | `1` | 消息报告 sensor 数 |
| `/teleop/manus/{side}/available` | `1` | 是否匹配到该侧消息 |
| `/teleop/manus/{side}/source_timestamp` | `1` | recorder ROS 接收时间 |
| `/teleop/manus/{side}/alignment_error_s` | `1` | 与训练 anchor 时间差 |

类型编码表保存在：

```text
metadata["teleop_diagnostics"]["manus_type_codebooks"]
```

未知类型编码为 `-1`。

## 10. 单手模式

启动参数：

```text
active_hand = right | left | both
```

例如 `right`：

- 只启动右侧 WujiHand controller 和 driver。
- 左手 state/command 不进入 required source。
- 左手 `qpos/qvel/effort/action/hand_joint_deg` 对应 20 维写零。
- 双臂仍然都记录和 Replay。
- metadata 写入 `active_hand_sides=[right]` 和 `zero_filled_hand_sides=[left]`。
- `/teleop/manus/left/*` 仍存在，但没有左手消息时 `available=0`。
- Replay 只向右手发布命令，不等待左手 driver。

这保持了当前训练和云端协议的双臂双手 54 维，不把暂时缺失硬件变成另一种
不兼容 schema。

只验证 MANUS/Wuji glove 到 WujiHand、完全不启动 Tianji 时，使用受控入口：

```bash
./src/scripts/start_teleop_hand_only.sh right
```

终端中依次按 `r` 让所选手从实测姿态移动到配置的初始姿态，等待
`RECOVERY COMPLETE` 后按 `a` 放行手套遥操；`x` 关闭关节，`q/e/Ctrl+C`
关闭关节并退出。该入口不会自动 Recovery 或 Enable。

## 11. Episode 持久化与失败处理

### 11.1 目录结构

```text
datasets/tianji_wuji/
  pick_red_block/            # 一个英文任务名对应一个目录
    episode_0000_YYYYMMDD_HHMMSS/
      lmdb/
        data.mdb
        lock.mdb
      meta_info.pkl
      sync_timestamps.json
      videos/                  # 原训练 RGB，格式不变
        head.mp4
        left_wrist.mp4
        right_wrist.mp4
      auxiliary_camera/        # 分析用旁路数据，存在在线流时生成
        depth.lmdb/            # 三台相机共用一个物理 LMDB
          data.mdb
          lock.mdb
        head_ir_left.mp4
        head_ir_right.mp4
        metadata.json
```

录制开始时目录名带：

```text
.inprogress
```

第一次踩踏板 1 结束采集后，目录仍带该后缀并进入 `pending_save`。只有踩踏板 2
且 finalize 全部成功后才去掉该后缀。GUI 显示的宿主机任务根目录是：

```text
/home/pjlab/ros2_ws/src/wuji-hand-teleop/datasets/tianji_wuji/<task>
```

容器内对应：

```text
/home/wuji/datasets/tianji_wuji/<task>
```

### 11.2 写入策略

- 核心 scalar 每帧使用一个 LMDB transaction 写入，结束时再写 aggregate array。
- 相机帧流式写入 MP4。
- `/teleop/*` 使用定长数值数组并在 finalize 时写 aggregate array，减少运行时
  LMDB 写放大；其纯数值 payload 约为 `6.3 MiB/min @ 30 Hz`，不含视频。
- 所有核心字段都做 shape 和 finite 校验。
- 可选诊断字段错误时替换为 neutral fill，不拒绝核心训练帧。
- 相机写入失败后 episode 被 quarantine，不允许伪装成完整 episode。
- 三路深度共用 `auxiliary_camera/depth.lmdb`，但 key 通过
  `depth/{camera}/{training_step}` 分开命名，不会混淆相机；图像以 PNG16
  无损编码保存。
- 头部左右红外分别写入两个 MP4，并在辅助 LMDB 中保存 video frame 到
  training step、时间戳和源 sequence 的映射。
- 辅助数据使用容量 64 的后台有界队列。采集线程只做非阻塞入队；队列满、编码
  或写盘失败不会反向阻塞或使原训练 episode 失败。
- 踏板2保存时会排空辅助队列并关闭文件；丢弃轨迹或退出时，辅助目录随同对应的
  `.inprogress` episode 一起删除。
- 零帧 episode 不 finalize，确认保存时直接删除。
- 关闭 GUI、`q/e` 或 `Ctrl+C` 会删除 active/pending 的未保存目录，不会删除已经
  finalize 的 episode。
- 新轨迹 writer 创建成功后，才删除被新一轮踏板 1 替换的 pending 轨迹。

### 11.3 退出语义

```text
s / 踏板1（idle）          开始采集
s / 踏板1（recording）     结束采集，进入 pending_save
z / 踏板2（pending_save）  确认保存，等待 SAVED
s / 踏板1（pending_save）  丢弃上一条并开始新采集
c / 踏板3                  断开/重新锚定 Tracker；采集中强制拒绝
q/e 或 GUI 退出            丢弃 active/pending 并退出整个 session
Ctrl+C                     与 q/e 的数据语义相同，随后有序关闭全部程序
```

`z` 的服务超时为 60 秒，用于等待 LMDB aggregate 和 MP4 finalize。数据必须保留时，
应先结束采集，再保存并等待 `SAVED:`，不要紧接着退出。

`Ctrl+C` 会先请求 Tianji standby、验证双臂 `state=0`，再依次使用 SIGINT、
SIGTERM、SIGKILL 清理本次 launch 进程组。它不会等待 recorder 保存 active episode。

## 12. Replay 与部署管线

### 12.1 为什么 Replay 走部署管线

Replay server 与云端策略使用同一接口：

```text
机器人 observation
  -> ZMQ request
  -> replay_server 或 cloud policy
  -> action response
  -> deployment_node
  -> Tianji external target / WujiHand command
```

这样本地 Replay 验证的是未来云端部署会使用的真实机器人端管线，而不是另一套
只服务于离线回放的控制代码。

### 12.2 Replay 行为

- 默认机械臂读取 `action` 中的 EEF pose；使用
  `--arm-command-mode joint` 时改读 `qpos` 中的实测电机关节角。手始终读取
  `action` 中的手指目标；忽略 `/teleop/*`。
- 启动前验证轨迹 shape 和有限值。
- Recovery 仍必须由操作者按 `r` 显式触发。
- Replay Enable 仍必须由操作者按 `a` 显式触发。
- 默认 EEF 模式把首帧刚体 rebase 到 Enable 时的实际 EEF，经 IK 后播放；
  joint 模式要求 episode 包含完整 `qpos` 且不做 rebase。
- 默认每秒推进 6 个录制源帧（30 Hz 数据的 0.2x），Replay server 输出 30 Hz
  waypoint，deployment/controller 以 120 Hz 插值和发送目标；500 Hz 只用于
  Tianji SDK 状态读取。
- Replay Enable 默认把选定 Tianji 臂切到 SDK `state=1` 位置模式，也可显式
  使用 `--arm-hardware-mode impedance` 进入 `state=3`；未选臂保持 `state=0`。
- 超过 250 ms 的旧 action 不发布。
- 外部目标断流时仍由 Tianji controller 安全逻辑请求 standby。

Replay 改变的是命令来源：

```text
record: Tracker -> IK -> Tianji state=3 impedance
replay(default): recorded EEF -> IK -> Tianji state=1 position
replay(joint): recorded qpos -> Tianji state=1 position
```

命令来源 `eef|joint` 与硬件模式 `position|impedance` 是两个独立开关，四种组合
均受支持；默认值是 `eef + position`。位置模式只允许 `control_source=external`，
不会改变 Tracker 遥操的默认阻抗模式。Recovery 和退出时回到 standby 的流程不变。
Replay 使用独立 `config/replay.yaml`，不会继承云端模型 profile 的 horizon、预取、
边界融合和平滑实验参数。

## 13. 主要实现文件

| 文件 | 作用 |
|---|---|
| `src/wuji_data_pipeline/wuji_data_pipeline/schema.py` | 54 维数据布局、单位转换 |
| `src/wuji_data_pipeline/wuji_data_pipeline/sync.py` | 时间 ring buffer、最近邻、速度差分 |
| `src/wuji_data_pipeline/wuji_data_pipeline/recorder_node.py` | ROS 订阅、同步、单手补零、诊断对齐 |
| `src/wuji_data_pipeline/wuji_data_pipeline/teleop_diagnostics.py` | Tracker/MANUS 可选 schema 和纯转换 |
| `src/wuji_data_pipeline/wuji_data_pipeline/episode.py` | LMDB/MP4 writer 和 reader |
| `src/wuji_data_pipeline/wuji_data_pipeline/auxiliary_camera.py` | 深度 PNG16 LMDB、头部双红外视频和后台有界队列 |
| `src/wuji_data_pipeline/wuji_data_pipeline/record_session.py` | 一键数采 cockpit 和有序退出 |
| `src/camera/stereocamera/shared_frames.py` | RGB、深度、单通道红外 typed 共享内存环 |
| `src/camera/stereocamera/camera_manager.py` | RealSense 多传感器采集、最新帧和标定 metadata |
| `src/wuji_teleop_monitor/wuji_teleop_monitor/ui/run_record.py` | 独立 PyQt5 数采 GUI、三路预览和 PTY |
| `src/wuji_teleop_monitor/wuji_teleop_monitor/ui/record_gui_core.py` | 任务目录、启动参数和踏板状态纯逻辑 |
| `src/wuji_teleop_monitor/wuji_teleop_monitor/ui/pedal_input.py` | 三踏板硬件中立适配接口 |
| `src/wuji_data_pipeline/wuji_data_pipeline/replay_core.py` | 轨迹拆分、校验、插值、rebase |
| `src/wuji_data_pipeline/wuji_data_pipeline/replay_server.py` | ZMQ Replay policy server |
| `src/wuji_data_pipeline/wuji_data_pipeline/deployment_node.py` | 机器人侧网络 client 和 external command |
| `src/wuji_data_pipeline/wuji_data_pipeline/replay_session.py` | 一键 Replay cockpit |
| `src/wuji_data_pipeline/launch/record.launch.py` | 已验证遥操 + recorder graph |
| `src/wuji_data_pipeline/launch/deployment.launch.py` | external controller + deployment graph |
| `src/input_devices/openvr_input/openvr_input/openvr_tracker_wrapper.py` | 同 SDK poll 缓存 raw state |
| `src/input_devices/openvr_input/openvr_input/openvr_input_node.py` | TF + 30 Hz 诊断 topic |
| `src/scripts/start_record_session.sh` | 宿主机一键数采入口 |
| `src/scripts/start_record_gui.sh` | 宿主机独立数采 GUI 入口 |
| `src/scripts/start_replay_session.sh` | 宿主机一键 Replay 入口 |

集中配置：

```text
src/wuji_data_pipeline/config/pipeline.yaml
```

## 14. 标准数采操作

### 14.1 启动主容器

已有容器且状态为 Up 时不需要重复执行。

```bash
docker compose \
  -f /home/pjlab/ros2_ws/src/wuji-hand-teleop/docker/docker-compose.yml \
  up -d
```

### 14.2 准备 SteamVR

先启动 SteamVR，确认基站和 5 个 Tracker 正常。无需再单独运行
`start_openvr_input.sh`；点击 GUI 的“进入准备”后，session 会自动启动或复用
`wuji-openvr-input`，等待 `/openvr_input` 在线后再启动其余数采节点。由本次
session 自动创建的 OpenVR 容器会在退出 GUI session 时一并关闭。

### 14.3 启动独立数采 GUI

另开宿主机终端：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
./src/scripts/start_record_gui.sh
```

每次打开先选择旧任务，或输入英文名创建新任务。主窗口默认：

- 手部配置：双手。
- 相机：启用；实际只显示并记录在线相机。

需要单手或无相机开发时，在点击“进入准备”之前修改。点击“进入准备”后配置锁定，
由 GUI 统一启动 OpenVR、controller、手、相机和 recorder，不再另开终端重复
启动 `record.launch.py`。此时 WujiHand 驱动只连接硬件、发布实测状态，关节保持
未使能；不会因为进入准备而立即跟随 MANUS。

新机器需要应用的 WujiHand 驱动 Patch、构建和验证步骤见
[`patches/WUJIHANDROS2_SUPERVISED_ENABLE.md`](patches/WUJIHANDROS2_SUPERVISED_ENABLE.md)。

相机画面不再嵌入数采控制主窗口。进入准备且启用相机后，GUI 会自动打开独立的
“相机监看”窗口；也可以点击主窗口的“打开相机监看窗口”重新显示。该窗口可以
拖到数采员使用的第二块屏幕，也可以点击“移到另一块屏幕”后全屏显示。窗口顶部
有醒目的轨迹状态条：踩 F7 后
红色显示“正在采集”并持续更新时长和帧数，再踩 F7 后橙色提示等待 F8 保存，
踩 F8 保存成功后绿色结果会一直保留到下一条轨迹开始；数采操作失败则显示红色
错误提示。该状态条只读取 recorder 已发布的缓存状态和 session 操作结果，不会
查询机械臂硬件，也不会增加控制回路负载。

相机窗口继续读取 Camera Manager 的共享内存最新帧，不重复启动相机、不复制
Recorder，也不把图像改回 ROS2 话题。窗口只显示实际收到画面的相机：

```text
1 路：单画面占满窗口
2 路：左右并排
3 路：主视角占左侧大画面，左右腕部在右侧上下排列
```

画面超过 2 秒没有更新时保留其位置并显示红色“图像超时”，便于数采员发现运行中
掉线；未连接过的相机不会占用画面位置。隐藏监看窗口不会停止相机或 Recorder。

### 14.4 GUI 按钮和踏板顺序

GUI 完整标准流程：

```text
1. 点击 Recovery，同时等待 Tianji lifecycle `7/RECOVERY_READY` 和所选手
   `recovery_state=2/READY`
2. 点击 Enable，等待 lifecycle 2/READY
   Tianji 到达 READY 后才放行所选 WujiHand 的外部控制；机械臂从当前实测位置
   使用约 6 秒渐进 handoff，手从 Recovery 初始姿态使用约 5 秒渐进接管
3. 踩踏板1，开始采集
执行遥操任务
4. 再踩踏板1，结束采集并进入 pending_save
5. 踩踏板2，确认保存并等待“最近轨迹结果：已保存”
6. 下一条回到第 3 步
7. 全部完成后点击“退出系统”
```

`r` 会实际移动机械臂和所选 WujiHand。执行前必须清空工作区、保持手指远离手部
机构并使实体急停可触达。手部 Recovery 期间外部 MANUS/replay/policy 目标被驱动
拒绝；任何一侧 Recovery 或使能失败时，session 会关闭已经使能的手并请求 Tianji
standby。`x`、退出和 `Ctrl+C` 都先关闭所选 WujiHand，再请求 Tianji standby。

踏板 1 和踏板 2 的替换逻辑：

```text
踏板1开始 -> 踏板1结束 -> 踏板2保存
踏板1开始 -> 踏板1结束 -> 不踩踏板2，直接踏板1
                            = 丢弃上一条并开始新一条
```

踏板 3 是 Tracker clutch：

1. 采集中踩踏板 3，GUI 和 cockpit 都强制拒绝。
2. 先用踏板 1 结束采集，再踩踏板 3 断开人体 Tracker 控制；Tianji 仍保持阻抗，
   不请求 standby。
3. 人调整到与机器人当前姿态一致。
4. 再踩踏板 3，按当前人体/机器人姿态重新建立 neutral 并恢复控制。
5. 踏板 3 不删除 pending 轨迹。

实体踏板协议尚未确定，因此当前 GUI 提供三个“踏板模拟”按钮。未来硬件适配器只
负责上报 `1/2/3`，所有拒绝条件仍由 GUI/cockpit 状态机执行，不能旁路。

### 14.5 正常停止

若当前数据需要保存：

1. 先踩踏板 1 结束采集。
2. 踩踏板 2。
3. 等待 GUI 显示已保存，或日志出现 `SAVED: ... (N steps)`。
4. 点击“退出系统”。
5. 等待 `Tianji state=0 confirmed; all session processes stopped`。

关闭窗口也走相同的有序退出，会先弹出未保存数据将被丢弃的确认。GUI 不直接
杀进程；它向唯一的 cockpit 发送 `q`，由 cockpit 请求 standby、确认双臂
`state=0`，再清理完整 launch 进程组。

独立 `wuji-openvr-input` 不属于 record session。所有数采结束后，在它自己的
终端按 `Ctrl+C`；若原终端已经丢失，可执行：

```bash
docker stop wuji-openvr-input
```

### 14.6 命令行兼容入口

GUI 不可用时仍可直接使用 cockpit：

```bash
# 默认无相机、单右手，保持旧命令兼容
cd /home/pjlab/ros2_ws/src/wuji-hand-teleop
./src/scripts/start_record_session.sh right

# 指定任务、双手和三路相机
./src/scripts/start_record_session.sh both \
  --task pick_red_block \
  --with-camera

# 仅用于迁移对比的旧ROS相机路径
./src/scripts/start_record_session.sh both \
  --task pick_red_block \
  --with-camera \
  --camera-transport ros
```

命令行按键与三个踏板一一对应：

```text
r Recovery
a Enable
s 踏板1
z 踏板2
c 踏板3
q/e 退出并丢弃未保存轨迹
```

GUI 会向统一入口额外传入 `--handoff-ramp-sec 6.0`。直接使用旧命令行入口时，
`record.launch.py` 仍保持已经验证的 `1.0 s` 默认值，不改变原操作习惯。

## 15. 常用检查与离线命令

以下命令除特别说明外均从宿主机执行。

### 15.1 查看容器

```bash
docker ps --filter name=wuji-hand-teleop
docker ps --filter name=wuji-openvr-input
```

### 15.2 查看数采子进程日志

```bash
docker exec wuji-hand-teleop \
  tail -n 200 /tmp/wuji_record_session_children.log
```

持续查看：

```bash
docker exec -it wuji-hand-teleop \
  tail -f /tmp/wuji_record_session_children.log
```

### 15.3 查看 lifecycle 和状态

在主容器内统一环境：

```bash
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=112
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS2CLI_DISABLE_DAEMON=1
unset CYCLONEDDS_URI
```

然后可选执行：

```bash
ros2 topic echo /tianji_arm/lifecycle_state
ros2 topic echo /tianji_arm/teleop_status
ros2 topic echo /wuji_teleop_recorder/status_text
```

查看 recorder service 状态：

```bash
ros2 service call /wuji_teleop_recorder/status \
  std_srvs/srv/Trigger "{}"
```

### 15.4 检查诊断输入

```bash
ros2 topic hz /openvr/tracker_diagnostics
ros2 topic hz /manus_glove_0
ros2 topic echo --once /manus_glove_0 --field side
ros2 topic echo --once /manus_glove_0 --field raw_node_count
```

如果连接两只手套，也检查 `/manus_glove_1`，并以消息 `side` 为准。

### 15.5 查看保存目录

宿主机：

```bash
find /home/pjlab/ros2_ws/src/wuji-hand-teleop/datasets/tianji_wuji \
  -mindepth 2 -maxdepth 2 -type d -name 'episode_*' | sort
```

容器内对应路径：

```text
/home/wuji/datasets/tianji_wuji
```

### 15.6 检查一条 episode

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
ros2 run wuji_data_pipeline inspect_episode \
  /home/wuji/datasets/tianji_wuji/pick_red_block/episode_0000_YYYYMMDD_HHMMSS
'
```

输出会包含：

- steps、frame rate、action/qpos/eef 维度。
- camera/video 信息。
- 各深度/红外流的实际保存帧数、辅助队列丢帧和写入错误。
- Tracker available/valid rate 和平均对齐误差。
- 左右 MANUS available、keypoints valid、raw node valid 和平均对齐误差。

### 15.7 Replay dry-run

该命令不打开 socket，也不启动或移动机器人：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
ros2 run wuji_data_pipeline replay_server \
  --episode-dir /home/wuji/datasets/tianji_wuji/pick_red_block/episode_0000_YYYYMMDD_HHMMSS \
  --dry-run
'
```

## 16. 一键 Replay 操作

### 16.1 启动单右手 Replay

始终显式传入要播放的容器内 episode 路径，不依赖脚本中的示例默认值：

```bash
cd /home/pjlab/ros2_ws/src/wuji-hand-teleop
./src/scripts/start_replay_session.sh right \
  /home/wuji/datasets/tianji_wuji/pick_red_block/episode_0000_YYYYMMDD_HHMMSS
```

默认命令源为 `--arm-command-mode eef`。只有需要对比电机关节角直放时才使用：

```bash
./src/scripts/start_replay_session.sh right \
  /home/wuji/datasets/tianji_wuji/pick_red_block/episode_0000_YYYYMMDD_HHMMSS \
  --arm-command-mode joint
```

默认以 6 source-frames/s 播放。只修改轨迹速度时使用明确参数：

```bash
./src/scripts/start_replay_session.sh right \
  /home/wuji/datasets/tianji_wuji/pick_red_block/episode_0000_YYYYMMDD_HHMMSS \
  --playback-rate-hz 12.0
```

这里 12 Hz 表示每秒推进 12 个录制帧；若 episode 是 30 Hz，即 0.4x。它不是
Tianji 控制频率。`--action-rate-hz` 默认保持 30 Hz，底层控制与状态读取仍分别为
120 Hz 和 500 Hz。

### 16.2 Replay cockpit 按键

按键在 Replay 终端直接按，不需要回车：

```text
r  guarded Tianji/WujiHand Recovery，等待 arm=7 且 hand=2
a  Enable 并开始 Replay，等待 lifecycle 2
x  Disable 并请求 standby
q/e 或 Ctrl+C 退出全部 Replay 进程
h  显示帮助
```

Replay 不需要启动 OpenVR、MANUS publisher 或 hand retargeter。它只需要：

- Tianji 双臂已上电、无急停、SDK 可连接。
- 选定侧的 WujiHand 已连接并配置了 20D `initial_position`。
- episode 已通过 dry-run。

### 16.3 Replay 日志

```bash
docker exec wuji-hand-teleop \
  tail -n 200 /tmp/wuji_replay_server.log

docker exec wuji-hand-teleop \
  tail -n 200 /tmp/wuji_replay_deployment.log
```

## 17. 构建与测试

普通 Python 源码修改不需要重建 Docker 镜像。进入主容器后：

```bash
cd /home/wuji/ros2_ws
source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select \
  camera \
  openvr_input \
  wuji_data_pipeline \
  wuji_teleop_monitor

source /home/wuji/ros2_ws/install/setup.bash
```

完整相关回归：

```bash
cd /home/wuji/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash

python3 -m pytest -q \
  src/camera/test \
  src/input_devices/openvr_input/test \
  src/wuji_data_pipeline/test \
  src/controller/test \
  src/output_devices/tianji_output/test \
  src/wuji_teleop_monitor/test
```

最近一次结果：

```text
178 passed
```

ROS package 测试：

```bash
colcon test --packages-select \
  camera openvr_input controller wuji_data_pipeline wuji_teleop_monitor \
  --event-handlers console_direct+
colcon test-result --verbose
```

## 18. 常见现象速查

| 现象 | 原因/边界 | 处理 |
|---|---|---|
| `wuji-hand-teleop container is not running` | 主容器未启动 | 执行 14.1 |
| `wuji-openvr-input is not running` | 独立 OpenVR 未启动 | 先执行 14.2 |
| OpenVR Domain/RMW 检查失败 | 启动环境不是文档值 | 停止 OpenVR 容器，按 14.2 重启 |
| `missing TF ...` | OpenVR/静态 TF 尚未建立或节点退出 | 等待初始化；仍失败则看 child log 和 OpenVR |
| Recovery 要求 `state=0`，当前为 `state=3` | 上一控制实例未正确 Disable | 先按 `x`/退出并确认 standby；不要反复按 `r` |
| WujiHand Recovery 拒绝或进入 `FAILED` | 所选手缺少 20D 初始姿态，或在时限内未收敛 | 关闭手部并检查配置、关节状态和实体机构；不要绕过 Recovery 门禁 |
| 踏板1提示 lifecycle 不是 READY | 尚未完成 Recovery/Enable，或 Tracker clutch 尚未重连 | 等待 lifecycle 2 |
| 踏板1提示 sources not ready | 核心arm/hand topic或Camera Manager帧缺失/过期 | 查看GUI Recorder状态、Camera Manager日志和child log |
| 踏板2无效 | 仍在采集或当前没有 pending 轨迹 | 先用踏板1结束当前采集 |
| 踏板3被拒绝 | 当前正在采集 | 先用踏板1结束；不得绕过 |
| Tracker diagnostics missing 但遥操正常 | OpenVR 仍是旧进程或新 topic 未发布 | 重启 `wuji-openvr-input`；不会阻塞核心数采 |
| MANUS diagnostics available=0 | 未收到该侧 ManusGlove 或 side 不符 | 查看 `_0/_1` 的 `side`、频率和 publisher 日志 |
| RGB 正常但某路辅助流没有保存 | 对应 depth/IR profile 不可用、设备离线或未在容差内匹配 | 查看 Recorder status 的 `auxiliary_camera` 和 episode 的 `auxiliary_camera/metadata.json` |
| 头部深度/红外只有约 6 Hz | 当前头部 D435 是 USB 2.1，为保护原 RGB 30 Hz 主动降频 | 属于当前已验证配置；迁到 USB 3.x 后再提高三个对应频率 |
| `queue_drops` 或 `write_errors` 大于 0 | 辅助编码/磁盘跟不上 | 原训练数据仍有效；检查磁盘和 CPU，再调整辅助频率或队列 |
| 踏板2保存暂时没有返回 | 正在 finalize LMDB/MP4 | 最多等待 60 秒；数据要保留时不要退出 |
| 运行中出现 `.inprogress` | active 或 pending 的正常临时目录 | 只有保存完成且后缀消失后才能 Replay |
| 新 episode 能 inspect 但 raw valid rate 为 0 | topic 存在但设备状态无效 | 检查 Tracker connected/valid 和 MANUS 节点 |
| Replay dry-run 报 unsafe jump | 相邻 action 突变超过保护阈值 | 不要绕过；检查 episode 同步和原始数据 |
| Replay `ENABLE_FAILED` | Tianji Recovery/Enable 安全检查失败 | 查看 lifecycle、arm status 和 deployment log |
| Replay 启动后没有动作 | 未到 READY、网络 action 未到或 episode 正在 hold | 查看两个 Replay log 和 lifecycle |
| Ctrl+C 后需要数秒退出 | 正在 Disable、确认 state=0 和清理进程组 | 等待最终退出信息；不是 recorder 保存等待 |

诊断字段缺失不会造成 `sources not ready`。若 `s` 被拒绝，应优先排查核心机器人、
手部topic或Camera Manager共享帧，而不是 `/teleop/*`。

## 19. 强制停止与安全边界

正常情况只使用 cockpit 的 `q` 或 `Ctrl+C`。

若机器人状态异常、程序无响应或无法确认 standby：

1. 先按实体急停。
2. 不要靠近机器人。
3. 急停确认后再停止容器：

```bash
docker stop wuji-hand-teleop
docker stop wuji-openvr-input
```

`docker stop wuji-hand-teleop` 不是正常数采退出方式，因为它不能替代 ROS
controller 的 standby 确认。禁止在机器人可能仍使能时用宽泛的 `pkill -f ros2`
当作日常停止命令。

## 20. 配置入口

主要配置位于：

```text
src/wuji_data_pipeline/config/pipeline.yaml
```

关键项：

```yaml
recording:
  output_dir: "~/datasets/tianji_wuji"
  frame_rate: 30.0
  sync_tolerance_s: 0.04
  require_cameras: true
  camera_names: [head, left_wrist, right_wrist]
  camera_transport: "direct"
  camera_shared_memory_dir: "/dev/shm/wuji_camera_v1"

teleop_diagnostics:
  enabled: true
  sync_tolerance_s: 0.10
  tracker_topic: "/openvr/tracker_diagnostics"
  tracker_roles: [chest, left_wrist, right_wrist, left_arm, right_arm]
  manus_topics: ["/manus_glove_0", "/manus_glove_1"]
  manus_node_count: 25
  manus_sensor_count: 5
  manus_ergonomics_count: 20

deployment:
  server: "tcp://127.0.0.1:5555"
  request_rate_hz: 30.0
  max_action_age_s: 0.25
  publish_rate_hz: 120.0
  camera_transport: "direct"
  camera_shared_memory_dir: "/dev/shm/wuji_camera_v1"

replay:
  rebase: true
  start_hold_s: 1.0
  loop: false
```

GUI 数采时，`start_record_session.sh --task NAME` 会把 recorder 的实际
`output_dir` 覆盖为 `/home/wuji/datasets/tianji_wuji/NAME`；YAML 中的值只作为
未显式传入任务目录时的兼容默认值。

注意：当前一键 `replay_session` 直接使用 `replay_server` 的同名 CLI 默认值，尚未
读取 `pipeline.yaml` 的 `replay` block；上面三项记录的是当前一致的默认语义。
若以后需要从 YAML 动态修改 Replay 参数，必须先把该 section 显式接入
`replay_session/replay_server`，不能只改 YAML 后假定已经生效。

硬件维度参数可以随硬件配置调整，但 Tracker topic 的角色顺序属于当前协议，修改
OpenVR publisher 时必须同步修改 recorder 配置和测试。

## 21. 已知限制和维护原则

- 阶段D已完成主视角D435直连验证；双腕序列号仍是placeholder，因此2/3路在线
  子集和带机械臂性能基线仍待现场验收。无相机模式使用左臂反馈作为30 Hz anchor。
- 当前实体踏板协议尚未确定；`PedalInputAdapter` 接口和 GUI 模拟按钮已经完成，
  但还没有绑定具体 USB/HID/串口设备。
- OpenVR 和 MANUS 消息没有统一硬件时间戳，当前使用 recorder ROS 接收时间。
- `/teleop/*` 是分析字段，不是训练主输入的强制组成部分。
- 可选诊断 aggregate 在 episode finalize 时写入；未正常保存的 `.inprogress`
  不保证包含完整 `/teleop/*` aggregate。
- 当前协议使用 pickle，云端只能部署在受信任内网、VPN 或认证隧道，不能把
  ZMQ 5555 直接暴露到公网。
- Replay 默认使用 EEF rebase，也可切换为录制关节角直放；Recovery、碰撞检查和
  急停仍不可省略。
- 当前没有环境碰撞规划。
- 不得为了 Replay 成功而放宽 Tianji IK 跳变、关节步长或 Recovery 安全阈值。
- 不得同时运行 MarvinPlatform、第二个 Tianji controller 或直接 Tianji SDK 工具。
- 不得同时运行两份 MANUS Core/publisher。
- 数采代码只能订阅和记录控制中间量，不能在诊断 callback 中发布机器人命令。
- 修改核心字段语义时必须同时更新 metadata、reader、Replay、inspect 和回归测试。
- 修改相机数量、手 DoF 或 Tracker 角色时，必须按真实硬件修改 schema，不能为了
  兼容参考工具而填入虚构设备。

本文记录的是当前保留实现。以后若改变硬件、时钟同步、相机数量或云端协议，
应在本文追加现场验证结论，而不是仅修改代码。
