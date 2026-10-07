# Tianji + HTC Vive 双臂遥操作改动与运行记录

更新日期：2026-07-16

本文记录当前工作区针对 Tianji 双臂真机控制所做的改动、现场验证结论、参数语义和常用命令。
当前主链路为：

```text
SteamVR + 5 个 Vive Tracker
  -> openvr_input
  -> ROS 2 TF
  -> tianji_arm_controller
  -> TianjiChestDriver
  -> Marvin SDK
  -> Tianji 双臂
```

本文只描述当前保留的实现。已经撤销的左臂 `r_align`、`position_sign` 和
`invert_rotation` 实验不属于最终方案。

## 1. 当前现场结论

- 右臂与左臂均已完成真机方向验证，可以随 Tracker 正常运动。
- 正常遥操使用 `state=3` 关节阻抗模式，不使用刚性位置模式持续遥操。
- 启动时保留 neutral 映射，操作者不需要把手腕绝对位置与机器人 init TCP 重合。
- 上臂 Tracker 的 ZSP 使用相对 enable-time neutral 的旋转，避免启动时切换 IK 分支。
- OpenVR 原始左右修正矩阵和 `static_transforms.yaml` 没有修改。
- 左右腕使用同一套 side-chest neutral-delta 公式，不再额外翻转左侧。
- 当前已验证的软件关节步长为 `0.5 deg/frame`，120 Hz 下理论软件上限约 `60 deg/s`。
- 当前真机验证命令没有覆盖阻抗速度比例，因此仍使用代码默认 `15/15`。

## 2. 主要代码改动

### 2.1 Marvin SDK 与同步确认

涉及文件：

```text
src/output_devices/tianji_output/tianji_output/_internal/lib/libMarvinSDK.so
src/output_devices/tianji_output/tianji_output/_internal/fx_robot.py
src/output_devices/tianji_output/tianji_output/tianji_chest_driver.py
```

改动内容：

- Linux Marvin 客户端库已替换为导出 `OnSetSendWaitResponse` 的版本。
- 只替换主机侧 `.so`，没有调用 `OnUpdateSystem`，没有刷新机器人固件。
- 控制系统实测版本为 `100343009`。
- `fx_robot.py` 新增 `send_cmd_wait_response()` 和能力检测。
- 状态切换、待机、阻抗参数写入和 Recovery 命令使用同步 ACK。
- 旧 SDK 没有同步接口时只能执行只读/待机路径，不允许进入受控 Recovery。

库文件记录：

```text
旧 SHA-256: 6ec35fb433e93bd7fd66d500651b248e82890098c7d6a0881913add7af6af087
新 SHA-256: a0b69541efc02209c15c6519db4dd44129824994d0109d2c2956fabe7952e778
旧库备份:  src/.backup/tianji-sdk/libMarvinSDK.so.legacy-6ec35fb4
```

### 2.2 默认禁止自动使能

涉及文件：

```text
src/controller/controller/tianji_arm_node.py
src/wuji_teleop_bringup/launch/wuji_teleop.launch.py
```

当前行为：

- `auto_enable` 默认并强制为 `false`。
- 启动 controller 只连接机器人并请求 `state=0` 待机。
- 真机运动必须通过明确的 Recovery 和 Enable 服务触发。
- `set_enabled data:true` 不再隐式执行 move-to-init。
- `set_enabled data:false` 会取消 Recovery/Enable/Teleop 并请求待机。

### 2.3 受保护的 recover-to-init

新增服务：

```text
/tianji_arm_controller/recover_to_init
/tianji_arm_controller/recover_left_to_init
/tianji_arm_controller/recover_right_to_init
```

Recovery 流程：

1. 必须从 `state=0` 开始。
2. 先把当前实测关节角写成 seed，避免继承历史位置目标。
3. 只在 Recovery 内部允许进入 `state=1`。
4. 左右臂按顺序单独恢复，不同时进行大范围刚性运动。
5. 使用绝对速度、绝对加速度、反馈前置量和每帧步长限制。
6. 持续检查状态、错误码、关节限位、内外编码器一致性、反馈速度、跟踪误差、
   反向运动和停滞。
7. 到达 init 后保持并确认，再回到 `state=0`。如果开始时已经处于 init 容差内，
   仍会进入受保护的 `state=1` 保持阶段；此时不会故意制造位移，但不会再瞬间
   报告 Recovery 完成。
8. 任一异常都请求双臂待机。

注意：Recovery 中的 `state=1` 仍然是刚性位置模式。低速只降低速度和动能，不提供阻抗柔顺；执行前必须清空工作区并保证急停可触达。

### 2.4 阻抗使能顺序

Enable 现在只接受已完成 Recovery 的双臂：

1. 确认双臂 `state=0`。
2. 确认关节与 init 的最大误差不超过 `enable_init_tolerance_deg`。
3. 先写入关节阻抗 K/D 和阻抗类型。
4. 再切换到 `state=3`。
5. 暂不发送 Tracker 命令，执行稳定性观察。
6. 记录实测关节和 Tracker neutral，之后才允许遥操。

当前默认阻抗参数：

```text
K = [14.0, 14.0, 14.0, 10.5, 5.6, 5.6, 5.6]
D = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
impedance_velocity_ratio     = 30
impedance_acceleration_ratio = 30
```

K/D 决定柔顺性和跟随强度，不应通过随意提高 K 来解决速度问题。

### 2.5 enable-time neutral 映射

使能完成时记录：

- 左右腕 Tracker neutral pose。
- 左右上臂 Tracker neutral rotation。
- 当前机器人关节快照及对应 FK 末端 pose。

腕部目标使用以下等价关系：

```text
p_target = p_robot_neutral
         + position_scale * (p_tracker - p_tracker_neutral)

R_target = (R_tracker * R_tracker_neutral^T) * R_robot_neutral
```

这意味着操作者使能时所在的位置成为零点，不要求人与机器人末端绝对重合。

### 2.6 官方绝对 pose 与 enable-time neutral 对比

这里的“绝对/相对”描述的是空间目标映射方式，不是速度参数中的绝对角度和 SDK 百分比。

#### 官方原始绝对映射

当前仓库原始官方 controller 直接把 TF 链计算出的 pose 送入 Tianji IK：

```text
T_target_left  = TF(left_chest  -> tianji_left)
T_target_right = TF(right_chest -> tianji_right)
```

也就是说，TF 输出的位置和姿态就是机器人的绝对末端目标。例如 TF 给出：

```text
p_tracker_chain = [0.45, 0.20, 0.55] m
```

IK 就直接求机器人末端到 `[0.45, 0.20, 0.55] m`。官方上臂约束同样使用绝对语义：

```text
zsp_left  = TF(left_chest  -> left_arm)  的局部 Y 轴
zsp_right = TF(right_chest -> right_arm) 的局部 Y 轴
```

这种方法成立的前提是：

- chest、wrist 和 Tianji IK base 坐标系已经完成准确标定。
- Tracker 角色和佩戴方向符合官方约定。
- 人体 Tracker 映射出的绝对末端 pose 位于机器人工作空间内。
- Enable 时的绝对目标已经接近机器人当前末端 pose。

如果使能时 Tracker 的绝对目标与机器人当前姿态差得很远，第一帧就可能产生较大的关节目标、IK 无解或 IK 分支跳变。

#### 当前 enable-time neutral 映射

当前方案在每次 Enable 时建立一对参考：

```text
人体侧参考：T_tracker_neutral
机器人参考：T_robot_neutral（实测关节 FK）
```

随后只把 Tracker 相对于 neutral 的变化施加到机器人参考 pose：

```text
p_target = p_robot_neutral
         + position_scale * (p_tracker - p_tracker_neutral)

R_delta  = R_tracker * R_tracker_neutral^T
R_target = R_delta * R_robot_neutral
```

例如 Enable 时：

```text
人体手腕位置： [0.20, 0.30, 0.40] m
机器人末端位置：[0.45, 0.20, 0.55] m
```

两者不需要相等。之后人体手腕沿某个已对齐的轴移动 `0.10 m`，机器人从自己的 neutral 沿对应轴移动 `position_scale * 0.10 m`。

该计算始终使用“当前值减 Enable 时快照”，不是逐帧积分，因此不会因为数值积分本身持续累积误差。不过 Tracker 漂移、胸部 Tracker 移动或 TF 数据异常仍然会影响结果。

#### 两种方法的差异

| 项目 | 官方绝对映射 | Enable-time neutral |
|---|---|---|
| Enable 第一帧目标 | Tracker TF 的当前绝对 pose | 机器人当前实测 FK pose |
| 人与机器人绝对位置是否需要接近 | 需要 | 不需要 |
| 固定位置零点偏差 | 依赖完整 TF 标定消除 | Enable 时自动吸收 |
| 固定佩戴旋转影响 | 依赖准确静态 TF | 相对空间旋转可抵消部分固定安装旋转 |
| 相同 Tracker pose 跨会话结果 | 基本固定 | 取决于每次捕获的 neutral |
| 绝对位置复现能力 | 较好 | 不适合做绝对位置复现 |
| 操作者站位和自然姿态自由度 | 较低 | 较高 |
| 启动时大跳风险 | 绝对目标未对齐时较高 | 较低，但仍受 IK/ZSP 连续性影响 |
| 主要用途 | 已完整标定的绝对空间跟随 | 人机形态不同、强调自然启动的相对遥操 |

#### Neutral 能解决和不能解决的问题

Neutral 可以吸收：

- Enable 时人体手腕与机器人末端之间的固定位置差。
- Enable 时人体手腕与机器人末端之间的固定姿态差。
- 一部分由 Tracker 固定安装旋转带来的零点差异。
- 操作者站位和机器人 init pose 不容易直接重合的问题。

Neutral 不能修复：

- 坐标轴方向、左右角色或 Tracker 序列号错误。
- OpenVR/static TF 中的镜像、轴交换和比例错误。
- 人体上下臂与机器人连杆长度不同造成的非线性姿态差。
- IK 奇异点、关节限位、碰撞和不可达目标。
- 人体肘角与机器人肘角的一一对应。

当前映射控制的是“手腕末端相对位移和姿态变化”，而不是“人体关节角直接复制给机器人关节”。因此可能出现机器人上臂和下臂形成 `90 deg` 时，人体肘角已经大于 `90 deg`。主要原因是人体与机器人连杆长度和肩部几何不同；`teleop_position_scale=1.0` 只代表米制位移 1:1，不代表关节角 1:1。

判断该现象是形态差异还是 Tracker 漂移的方法：

- 回到 Enable 时的相同人体姿态后，机器人也回到原 neutral：主要是人机形态/比例差异。
- 人体回到相同姿态后仍存在残余偏移：检查 chest Tracker、Tracker 漂移和 TF。

当前项目选择 enable-time neutral，是为了降低启动对齐难度并避免第一帧追逐远端绝对目标。官方 OpenVR 修正矩阵和 static TF 仍然保留，因为 neutral 只消除零点差，不能替代坐标轴标定。

### 2.7 上臂 Tracker 与 ZSP

最终保留的 ZSP 映射为：

```text
zsp_target = normalize(
    (R_arm * R_arm_neutral^T) * zsp_driver_default
)

left  zsp_driver_default = [0, -1, -0.5]
right zsp_driver_default = [0,  1, -0.5]
```

重要现场结论：

- 腕部使用 neutral 映射时，上臂也必须从 neutral 相对变化开始。
- 曾把上臂 ZSP 改成 Tracker 局部 Y 轴绝对值，同时继续使用腕部 neutral。
- 该混合模式曾导致左臂 IK 解发生明显不连续，旧保护实现会让左右臂同时停止下发。
- 恢复相对 neutral 的 ZSP 后，左右臂均恢复正常运动。

### 2.8 IK 下发语义

`2026-07-27` 对照 `wuji-technology/wuji-hand-teleop` 的
`origin/main@6478013` 后，真机 Tracker 控制恢复为原始仓库的下发语义：

- 有效 IK 解在 handoff 结束后直接发送，不再经过额外的软件关节限速；
- 某一侧 IK 无解时，只跳过该侧当帧命令，另一侧继续下发；
- `libKine` 报告目标越界或关节超限时，该侧同样不发送当帧命令；
- 启动时的 `handoff_hold_sec` 和 `handoff_ramp_sec` 继续保留；
- 已删除 0.45 m 目标偏移、首帧 30°、相邻 IK 35°和0.3°/帧四项后加保护。

`10/TARGET_HOLD` 现在只用于 Replay/云端部署的外部命令流短时陈旧，不再由
Tracker 目标或 IK 失败触发。

## 3. 生命周期状态

话题：

```text
/tianji_arm/lifecycle_state
/tianji_arm/teleop_status
```

| 数值 | 状态 | 含义 |
|---:|---|---|
| 0 | INITIALIZING | controller 初始化 |
| 1 | ENABLING | 进入阻抗并执行稳定性检查 |
| 2 | READY | Tracker 遥操已启用 |
| 3 | DISABLED | 双臂待机 |
| 4 | ENABLE_FAILED | 使能失败，已请求待机 |
| 5 | SDK_ERROR | SDK、通信或硬件错误 |
| 6 | RECOVERING | 正在恢复到 init |
| 7 | RECOVERY_READY | 双臂已在 init 且待机，可以 Enable |
| 8 | RECOVERY_FAILED | Recovery 失败，已请求待机 |
| 9 | RECOVERY_PARTIAL | 只完成一侧 Recovery |
| 10 | TARGET_HOLD | Replay/部署外部命令流短时陈旧，等待新命令 |

`set_enabled` 返回 `Enable sequence started` 只表示后台流程已开始；必须等待生命周期变为 `2` 才算真正进入遥操。

## 4. 当前参数及单位

### 4.1 当前真机验证命令使用的覆盖值

| 参数 | 当前值 | 单位/含义 |
|---|---:|---|
| `control_rate` | 120 | Hz |
| `state_publish_rate` | 500 | Hz |
| `teleop_position_scale` | 1.0 | 无量纲，1:1 位移 |
| `handoff_hold_sec` | 0.2 | s |
| `handoff_ramp_sec` | 1.0 | s |
| `recovery_max_speed_deg_s` | 5.0 | deg/s，只用于 Recovery |
| `recovery_max_accel_deg_s2` | 10.0 | deg/s^2，只用于 Recovery |
| `recovery_encoder_agreement_deg` | 1.0 | deg，内外编码器最大允许差值 |
| `impedance_max_drift_deg` | 3.0 | deg，使能稳定性检查 |

未在命令中覆盖的关键默认值：

```text
impedance_velocity_ratio     = 30
impedance_acceleration_ratio = 30
```

### 4.2 绝对量与相对量

绝对量：

```text
recovery_max_speed_deg_s
recovery_max_accel_deg_s2
```

相对量：

```text
impedance_velocity_ratio
impedance_acceleration_ratio
recovery_velocity_ratio
recovery_acceleration_ratio
teleop_position_scale
```

Marvin `velRatio/AccRatio=100` 表示机器人配置上限的 100%，不是 `100 deg/s`。
原仓库 `HEAD` 的阻抗遥操使用 `60/60`；原代码 Recovery 曾使用 `100/100`。当前受保护 Recovery 将比例硬限制为不超过 `10/10`，正常遥操使用 `30/30`。

### 4.3 不同关节速度不同的原因

即使每帧 IK 目标直接下发，不同关节也不会实际等速：

- IK 会按当前姿态把末端运动分配给不同关节。
- Tianji 配置中 J1-J2 最大加速度为 `450 deg/s^2`，J3-J7 为 `900 deg/s^2`。
- 肩部负载惯量更大。
- J5-J7 阻抗 K 较低，腕部更柔顺，反馈可能更慢。
- SDK 比例作用于各关节自身的速度/加速度配置。

不要同时修改 K/D；先区分是命令目标本身慢，还是实际关节反馈落后。

### 4.4 遥操速度与阻抗比例配置

handoff 完成后的稳态控制没有额外的软件关节步长限制。有效 IK 解直接发送，
实际速度主要受 Marvin 阻抗模式比例、机器人关节配置、K/D 和负载影响：

```text
SDK 关节速度上限  = 关节配置最大速度 * impedance_velocity_ratio / 100
```

Tianji `ccs_m6.MvKDCfg` 中左右臂 J1-J7 的最大速度均为 `180 deg/s`。SDK 比例
对应的理论速度上限为：

| `impedance_velocity_ratio` | SDK 理论上限 |
|---:|---:|
| 15 | 27 deg/s |
| 30 | 54 deg/s |
| 60 | 108 deg/s |
| 100 | 180 deg/s |

`impedance_acceleration_ratio` 按各关节自己的配置加速度缩放：

| 比例 | J1-J2（配置 450 deg/s^2） | J3-J7（配置 900 deg/s^2） |
|---:|---:|---:|
| 15 | 67.5 deg/s^2 | 135 deg/s^2 |
| 30 | 135 deg/s^2 | 270 deg/s^2 |
| 60 | 270 deg/s^2 | 540 deg/s^2 |
| 100 | 450 deg/s^2 | 900 deg/s^2 |

当前正常遥操没有单独的软件关节加速度限制；handoff smoothstep 只平滑使能后的最初一段时间。因此提高 `impedance_acceleration_ratio` 会直接改变机械臂追赶目标时的动态响应。

#### 配置 A：保守方向检查

用于首次确认 Tracker 方向或修改坐标映射后的真机检查：

```bash
-p teleop_position_scale:=0.25 \
-p impedance_velocity_ratio:=15 \
-p impedance_acceleration_ratio:=15 \
-p handoff_hold_sec:=0.5 \
-p handoff_ramp_sec:=2.0
```

其中 `teleop_position_scale=0.25` 主要缩小末端位移幅度，不是直接的关节速度限制。

#### 配置 B：当前采用的阻抗比例

```bash
-p teleop_position_scale:=1.0 \
-p impedance_velocity_ratio:=30 \
-p impedance_acceleration_ratio:=30 \
-p handoff_hold_sec:=0.2 \
-p handoff_ramp_sec:=1.0
```

该配置使用 `30/30` 阻抗比例；修改后的第一次真机使用应只做小范围动作，并
观察 `joint_commands` 与 `joint_states`。

#### 阻抗 K/D

当前配置：

| 关节 | K | D | 特点 |
|---|---:|---:|---|
| J1-J3 | 14.0 | 0.3 | 跟随较紧，肩部惯量仍然较大 |
| J4 | 10.5 | 0.3 | 中等刚度 |
| J5-J7 | 5.6 | 0.3 | 腕部更柔顺，可能比命令目标滞后 |

- K 是目标位置误差对应的恢复刚度，不是速度上限。
- D 是速度阻尼，用于抑制振荡和过冲。
- 提高 K 会降低柔顺性，不能把它当作普通的“加速参数”。
- 调速度时先调整 SDK 比例，不同时修改 K/D。

#### Recovery 参数与遥操参数分离

以下参数只作用于 recover-to-init，不影响 `state=3` 正常遥操速度：

```bash
-p recovery_velocity_ratio:=10 \
-p recovery_acceleration_ratio:=10 \
-p recovery_max_speed_deg_s:=5.0 \
-p recovery_max_accel_deg_s2:=10.0
```

Recovery 的比例在代码中硬限制为不超过 `10/10`，同时还受绝对速度和绝对加速度限制。

#### 参数生效方式

当前 controller 在启动时读取参数并保存到内部字段。不要依赖运行中的
`ros2 param set` 改变控制行为。修改速度/阻抗比例时：

1. 调用 `set_enabled data:false`，确认双臂回到 `state=0`。
2. 在 controller 终端按 `Ctrl-C`。
3. 使用新参数重新启动 controller。
4. 重新执行 Recovery，等待生命周期 `7`。
5. 再 Enable，等待生命周期 `2`。

## 5. 常用启动命令

### 5.1 Docker 说明

已有容器时不需要重复执行 `docker compose up`：

```bash
docker start wuji-hand-teleop
docker exec -it wuji-hand-teleop bash
```

首次创建或确实需要重建容器时才使用：

```bash
cd /home/pjlab/ros2_ws/src/wuji-hand-teleop/docker
docker compose up -d
```

### 5.2 启动 SteamVR Tracker 输入（宿主机）

先确认 SteamVR、基站和 5 个 Tracker 均正常，再执行：

```bash
cd /home/pjlab/ros2_ws/src/wuji-hand-teleop
export ROS_DOMAIN_ID=112
./src/scripts/start_openvr_input.sh
```

该脚本创建 `wuji-openvr-input` 容器，并使用 CycloneDDS。已经运行时不要重复启动。

检查：

```bash
docker ps --filter name=wuji-openvr-input
```

### 5.2.1 开发阶段纯机械臂遥操（不启动双手和数采）

已经按 5.2 节启动 `wuji-openvr-input` 后，在宿主机执行：

```bash
cd /home/pjlab/ros2_ws/src/wuji-hand-teleop
./src/scripts/start_teleop_arm_only.sh
```

默认仍控制双臂。只控制左臂或右臂时分别执行：

```bash
./src/scripts/start_teleop_arm_only.sh left
./src/scripts/start_teleop_arm_only.sh right
```

该入口只启动 `vive_arm_tf.launch.py` 和使用本文 5.5 节已验证参数的
`tianji_arm_controller`。它不会启动 recorder、相机、MANUS、Wuji 手驱动或手部
控制器，也不会创建任何 episode。单臂模式只对所选机械臂执行 Recovery、进入
阻抗、读取 Tracker TF、求解 IK 和发送关节指令；另一臂必须且始终保持
`state=0`。不带参数时为 `both`，因此原双臂行为不变。交互键为 `r` Recovery、
`a` Enable、`x` Standby、`q`/`Ctrl-C` 确认双臂 `state=0` 后退出；按键不需要
回车。

### 5.3 controller/服务/监控终端统一环境（主容器内）

每个新终端都执行：

```bash
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=112
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS2CLI_DISABLE_DAEMON=1
unset CYCLONEDDS_URI
```

服务调用方必须与 controller 使用相同的 `ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION`，否则可能出现服务实际执行但客户端一直等待的现象。

### 5.4 启动 Vive 到 Tianji 静态 TF（终端 A）

```bash
ros2 launch wuji_teleop_bringup vive_arm_tf.launch.py
```

不要同时启动第二份相同静态 TF。

### 5.5 启动 Tianji controller（终端 B）

当前真机验证通过的命令：

```bash
ros2 run controller tianji_arm_controller --ros-args \
  -p auto_enable:=false \
  -p control_rate:=120.0 \
  -p state_publish_rate:=500.0 \
  -p teleop_position_scale:=1.0 \
  -p impedance_velocity_ratio:=30 \
  -p impedance_acceleration_ratio:=30 \
  -p handoff_hold_sec:=0.2 \
  -p handoff_ramp_sec:=1.0 \
  -p recovery_max_speed_deg_s:=5.0 \
  -p recovery_max_accel_deg_s2:=10.0 \
  -p impedance_max_drift_deg:=3.0
```

看到 `STANDBY reached (cur_state=0)` 后再继续。不要同时运行 MarvinPlatform 或第二个 Tianji controller；Marvin SDK 连接端口只能由一个进程持有。

### 5.6 监控（终端 C）

```bash
ros2 topic echo /tianji_arm/lifecycle_state
```

另一个终端查看文字原因：

```bash
ros2 topic echo /tianji_arm/teleop_status
```

查看硬件状态：

```bash
ros2 service call /tianji_arm_controller/arm_status \
  std_srvs/srv/Trigger "{}"
```

检查 Tracker TF：

```bash
ros2 run tf2_ros tf2_echo left_chest tianji_left
ros2 run tf2_ros tf2_echo right_chest tianji_right
ros2 run tf2_ros tf2_echo left_chest left_arm
ros2 run tf2_ros tf2_echo right_chest right_arm
```

### 5.7 Recovery 到 init（会实际移动机械臂）

双臂依次恢复：

```bash
ros2 service call /tianji_arm_controller/recover_to_init \
  std_srvs/srv/Trigger "{}"
```

只恢复单臂：

```bash
ros2 service call /tianji_arm_controller/recover_left_to_init \
  std_srvs/srv/Trigger "{}"

ros2 service call /tianji_arm_controller/recover_right_to_init \
  std_srvs/srv/Trigger "{}"
```

等待生命周期：

```text
data: 7
```

### 5.8 进入阻抗遥操

保持 Tracker 姿态稳定，急停可触达，然后执行：

```bash
ros2 service call /tianji_arm_controller/set_enabled \
  std_srvs/srv/SetBool "{data: true}"
```

等待生命周期变成：

```text
data: 2
```

### 5.9 停止遥操/取消 Recovery

```bash
ros2 service call /tianji_arm_controller/set_enabled \
  std_srvs/srv/SetBool "{data: false}"
```

确认回到 `state=0` 后，再在 controller 终端按 `Ctrl-C`。ROS 服务无响应且机械臂状态异常时，使用实体急停。

## 6. 常用诊断命令

### 6.1 关节命令与反馈

```bash
ros2 topic echo --once /left_arm/joint_commands  --field position
ros2 topic echo --once /left_arm/joint_states    --field position
ros2 topic echo --once /right_arm/joint_commands --field position
ros2 topic echo --once /right_arm/joint_states   --field position
```

当前这些 Tianji 话题直接发布 SDK 关节值，数值单位为度；用于 RViz 的 `/tianji_alignment/*` 话题才转换为弧度。

查看频率：

```bash
ros2 topic hz /left_arm/joint_commands
ros2 topic hz /left_arm/joint_states
```

判断速度问题：

- `joint_commands` 本身变化慢：IK 分配或 Tracker 输入变化慢。
- `joint_commands` 快而 `joint_states` 明显落后：SDK 比例、阻抗 K/D、负载或硬件侧限制。

### 6.2 IK 无解与状态 10

```bash
ros2 topic echo /tianji_arm/teleop_status
```

Tracker 遥操中，IK 无解或 `libKine` 报告目标/关节超限时，只跳过对应侧的
当帧命令，不切换到状态10。相关 SDK 日志可能包含：

```text
Robot Inverse Kinematics Error
[LEFT_IK] FAILED
[RIGHT_IK] FAILED
```

状态 `10/TARGET_HOLD` 只表示 Replay/云端部署的外部命令流短时陈旧；Tracker
遥操不使用该状态。

### 6.3 DDS UDP 报错或服务卡住

典型输出：

```text
ddsi_udp_conn_write ... failed with retcode -1
```

这属于 ROS 2 DDS 网络，不是 Marvin SDK 到机器人 `192.168.1.190` 的控制连接。当前 OpenVR 脚本使用 CycloneDDS，而现场 controller 使用 FastRTPS；监控终端若继承了不同 RMW，可能出现噪声或服务等待。

处理：

1. 在报错的监控终端按 `Ctrl-C`。
2. 关闭遗留的 `ros2 topic/node/param` 监控进程。
3. 重新执行第 5.3 节环境配置。
4. 确认服务端和客户端使用同一 Domain/RMW。

宿主机 `192.168.1.100` 是连接机器人的本机网卡地址，不是机器人地址；机器人地址是 `192.168.1.190`。

## 7. 构建与测试

代码变化后，在主容器中执行：

```bash
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash

cd /home/wuji/ros2_ws
colcon build --symlink-install --packages-select \
  tianji_output controller wuji_teleop_bringup tianji_description
```

Controller 安全测试：

```bash
export PYTHONPATH=/home/wuji/ros2_ws/src/controller:${PYTHONPATH:-}
python3 -m pytest -q \
  /home/wuji/ros2_ws/src/controller/test/test_tianji_arm_safety.py
```

当前最近一次结果：

```text
21 passed
controller package build succeeded
```

## 8. 已知限制和操作红线

- 真机正常遥操只允许 `state=3` 阻抗模式。
- `state=1` 仅用于显式 Recovery，执行时工作区必须清空。
- 不得绕过 `RECOVERY_READY=7` 直接 Enable。
- 不得提高 `30/35 deg` IK 分支保护来解决正常速度问题。
- 不得同时运行 MarvinPlatform、直接 SDK 工具和 ROS Tianji controller。
- `TARGET_HOLD=10` 当前是双臂全局 HOLD：任一侧目标无效时两侧都保持。
- `tracker_alignment.launch.py` 是离线诊断/可视化实验，不是当前真机遥操的必需启动项。
- OpenVR、static TF、controller 和所有 CLI 最终应统一 RMW；当前现场仍存在 CycloneDDS/FastRTPS 混用。
- 当前没有环境碰撞规划，Recovery 和遥操都依赖现场清空工作区及实体急停。

更详细的 SDK 升级和 Recovery 事故记录见：

```text
src/controller/TIANJI_VIVE_SAFETY.md
```
