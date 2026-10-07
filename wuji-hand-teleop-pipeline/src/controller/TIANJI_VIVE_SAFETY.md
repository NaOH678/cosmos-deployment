# Tianji + Vive 遥操作安全记录

更新日期：2026-07-24

## Marvin SDK 升级记录

2026-07-15 将主机侧 Marvin 客户端库升级为参考分支
`vive-dual-arm-teleop` 中已提交的版本。此次操作只替换 Linux 客户端 `.so`，没有调用
`OnUpdateSystem`，也没有刷新机器人控制系统固件。

- 旧库 SHA-256：`6ec35fb433e93bd7fd66d500651b248e82890098c7d6a0881913add7af6af087`
- 新库 SHA-256：`a0b69541efc02209c15c6519db4dd44129824994d0109d2c2956fabe7952e778`
- 旧库备份：`src/.backup/tianji-sdk/libMarvinSDK.so.legacy-6ec35fb4`
- 新库已确认导出：`OnSetSendWaitResponse`、`CheckSDKTypeCompat`
- 控制系统实测版本：`100343009`
- 待机启动实测：同步 ACK 成功，双臂保持 `state=0`，退出后连接正常释放
- 构建测试：`32 tests, 0 errors, 0 failures`

## 适用范围

本文适用于以下链路：

`openvr_input -> TF -> controller/tianji_arm_node.py -> TianjiChestDriver -> Marvin SDK`

## 事故结论

此前出现过两种不同的启动异常：

1. 直接使用高速 `state=1` 回 init，机械臂可能突然执行较大的历史/初始化目标。
2. 完全禁止 `state=1` 后，曾尝试在低刚度 `state=3` 中慢速回 init。2026-07-15
   真机测试中，进入 `state=3` 约 0.43 秒后左臂已经偏离首个目标 16 度，保护逻辑
   随后返回 `state=0`。

第二次异常发生在 Tracker/IK 接管前。主要问题是低刚度阻抗不适合从任意远端姿态
完成大范围恢复，而且当时的实现把参考顺序反转成了“先进入 `state=3`，后写 K/D”。

参考分支 `vive-dual-arm-teleop` 的正式顺序是：

```text
state=1 position recovery -> move_to_init -> preload K/D and impedance type
-> state=3 -> tracker handoff
```

## 当前状态机

恢复和遥操使能已经分成两个独立阶段。

### Recovery

`recover_to_init` 从 `state=0` 开始，一次只恢复一条机械臂：

1. 读取当前状态、错误码、关节角、外编码器、速度和配置关节限位。内外编码器
   任一关节相差超过 1 度时拒绝 Recovery。
2. 在 `state=0` 预写当前关节角，避免进入 `state=1` 后继承历史目标。
3. 以 `velRatio=10`、`AccRatio=10` 进入 `state=1`。
4. 保持当前位置 2 秒；漂移超过 0.5 度立即失败。
5. 以反馈约束的规划参考回 init。默认命令速度上限 1 度/秒、加速度上限
   2 度/秒平方、控制周期 20 ms。规划参考独立匀速前进，但下发目标相对实测反馈
   最多前置 0.5 度，并且每帧命令仍受速度上限约束。
6. 检查反馈速度、跟踪误差、反向运动、停滞、控制状态、错误码和关节限位。
   只要保持有效进展就不使用按初始误差估算的固定总超时；连续停滞 5 秒才退出。
   运行中每 2 秒输出剩余最大角度、主导关节、反馈速度、进展速度和 ETA。
7. 到达 init 后保持 2 秒，再将该臂返回 `state=0`。即使 Recovery 开始时已经在
   init 容差内，也不能直接报完成：仍会进入 `state=1` 保持 2 秒，核对状态和
   两套编码器后才允许生命周期进入 `7`。这种情况下机械臂不应产生可见位移。
8. 左臂完成后才恢复右臂。任一异常都请求双臂 `state=0`。

普通服务不能直接进入 `state=1`。驱动要求内部 `recovery=True`，并硬限制恢复
速度/加速度比例不能超过 10。

Recovery 还强制要求 Marvin SDK 导出同步确认接口
`OnSetSendWaitResponse`。只有异步 `OnSetSend` 的旧 SDK 可以保持待机和读取状态，
但不得进入位置模式 Recovery，避免将含义不明确的异步返回值当作运动授权。

本机 100343 SDK 头文件将 `state=100` 定义为 `ARM_STATE_ERROR`，不是状态切换
过程。只有 `101/102/103/104/109` 是切换状态。检测到 `state=100` 或非零
`err_code` 时会立即失败并要求显式清错，不再等待超时或错误地报告 Recovery 完成。

### Teleop Enable

`set_enabled data:true` 不再执行 move-to-init。它只接受已经完成 Recovery 的状态：

1. 确认双臂均为 `state=0`，且相对 init 最大误差不超过 3 度。
2. 先写入 K/D 和阻抗类型，等待 0.5 秒。
3. 最后切换到 `state=3`。
4. 暂不发送 Tracker 命令，观察 3 秒；任一关节漂移超过 1 度立即回 `state=0`。
5. 稳定后建立 handoff，才允许 Tracker/IK 控制。

旧的 `TianjiChestDriver.move_to_init()` 已改为直接抛出异常，防止旧调用重新走
阻抗模式回 init 的危险路径。

## 生命周期

`/tianji_arm/lifecycle_state` 使用以下数值：

| 数值 | 状态 | 含义 |
|---|---|---|
| 0 | INITIALIZING | 控制节点初始化 |
| 1 | ENABLING | 正在进入阻抗并做稳定性检查 |
| 2 | READY | Tracker 遥操已允许 |
| 3 | DISABLED | 双臂待机 |
| 4 | ENABLE_FAILED | 阻抗使能失败，已请求待机 |
| 5 | SDK_ERROR | SDK/通信错误 |
| 6 | RECOVERING | 正在低速恢复 |
| 7 | RECOVERY_READY | 双臂已在 init 且为 `state=0`，允许 enable |
| 8 | RECOVERY_FAILED | 恢复失败，已请求待机 |
| 9 | RECOVERY_PARTIAL | 选定单臂已恢复，另一臂尚未恢复 |
| 10 | TARGET_HOLD | 当前 IK 目标被拒绝，保持最后安全指令；调整 Tracker 后自动恢复 |

`TARGET_HOLD` 不是 SDK 故障，也不会自动退出阻抗模式。控制器拒绝异常帧并继续检查
后续目标，同时在 `/tianji_arm/teleop_status` 发布具体原因。目标重新进入安全范围后，
生命周期自动回到 `2`。真正的 SDK、通信或硬件状态异常仍会退回待机。

## 操作顺序

启动控制节点后必须先确认双臂为 `state=0`。以下 Recovery 命令会实际移动机械臂，
使用前必须清空工作区、保证急停可触达并安排现场观察人员。

恢复双臂：

```bash
ros2 service call /tianji_arm_controller/recover_to_init \
  std_srvs/srv/Trigger "{}"
```

只恢复左臂或右臂：

```bash
ros2 service call /tianji_arm_controller/recover_left_to_init \
  std_srvs/srv/Trigger "{}"

ros2 service call /tianji_arm_controller/recover_right_to_init \
  std_srvs/srv/Trigger "{}"
```

另一个终端监控生命周期：

```bash
ros2 topic echo /tianji_arm/lifecycle_state
ros2 topic echo /tianji_arm/teleop_status
```

只有看到 `7` 后才能请求遥操使能：

```bash
ros2 service call /tianji_arm_controller/set_enabled \
  std_srvs/srv/SetBool "{data: true}"
```

返回 `Enable sequence started` 只表示后台检查已开始。看到生命周期 `2` 后才表示
阻抗稳定性检查通过并允许 Tracker 控制。

停止或取消 Recovery/Enable/Teleop：

```bash
ros2 service call /tianji_arm_controller/set_enabled \
  std_srvs/srv/SetBool "{data: false}"
```

软件停止后再次查询状态，确认双臂为 `state=0`。若 ROS 服务无响应，使用实体急停。

## 重要限制

- `state=1` 即使很慢仍然是刚性位置控制，低速只降低运动速度和动能，不会让碰撞变柔顺。
- 当前 Recovery 使用参考代码的关节空间直线路径，没有完整的环境碰撞规划。工作区必须清空。
- 不得提高 Recovery 的速度/加速度上限来缩短等待时间。
- 不得绕过 `RECOVERY_READY` 直接进入 `state=3`。
- 不得通过 `/tianji_arm/set_arm_state data:true` 直接请求 `state=1`。
- 单臂 Recovery 结束后若生命周期为 `9`，还需要恢复另一条机械臂。

## 参考实现差异

参考分支中的独立 `tools/move_to_init.py` 使用 25 秒、`vel=30`、`acc=30` 的开环
quintic 轨迹。当前实现更保守：比例上限 10、默认 1 度/秒，并以实际反馈推进目标。

历史 Python 实现提交 `8cf0a17` 使用的控制库二进制与当前 ROS 控制库不同。因此没有
直接覆盖 `.so`，而是保留当前库并使用返回值、状态和反馈闭环确认命令结果。
