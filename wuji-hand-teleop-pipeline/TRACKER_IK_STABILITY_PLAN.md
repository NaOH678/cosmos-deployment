# Tracker 与 IK 诊断计划

> 初始记录日期：2026-07-24
> 控制语义修订日期：2026-07-27
> 当前基线：阶段 B 控制实现 + 阶段 D 相机直连

## 1. 当前决策

经核对 `wuji-technology/wuji-hand-teleop origin/main@6478013`，Wuji 原始
稳态控制路径的行为是：

- 有效 IK 解直接发送；
- 某侧 IK 无解时只跳过该侧当帧命令；
- `libKine` 报告目标越界或关节超限时，该侧当帧不发送；
- 启动阶段使用 handoff 平滑过渡。

当前分支已经删除以下非原生控制保护：

- 0.45 m Tracker 目标偏移限制；
- 首帧 30° IK 偏差检查；
- 相邻 IK 解 35° 跳变检查；
- 0.3°/帧关节命令限速。

后续工作只做只读诊断。未经新的独立评审和真机实验，不再向控制链路加入
Tracker 拒绝、滤波、IK 跳变阈值或软件关节步长限制。

## 2. 已观察到的现象

实际操作动作幅度正常，但部分会话出现短时卡顿和大量
`Robot Inverse Kinematics Error`。

历史诊断中发现：

- 左臂 IK 无解和 IK 解大幅变化明显多于右臂；
- 部分 30 Hz Tracker 诊断帧出现二十至三十厘米的单帧位移；
- SteamVR `valid=true` 不能证明每一帧位置连续；
- 阶段 D 录制时控制仍约为 120 Hz，相机失败数为0，因此当时的 IK 异常不能
  直接归因于 Recorder 或相机负载。

这些历史数据用于定位问题，不再作为控制器拒绝目标的依据。

## 3. 保持不变的控制基线

```text
control_rate                    120 Hz
state_publish_rate              500 Hz
teleop_position_scale           1.0
impedance_velocity_ratio        30
impedance_acceleration_ratio    30
```

同时保持：

- 当前 Tracker 坐标轴、左右臂映射和 Enable neutral；
- 上臂 ZSP 相对 neutral 的映射；
- Recovery、Enable 和 Tracker clutch 逻辑；
- `qpos/action/eef` 训练数据格式；
- Tracker/MANUS 原始诊断数据记录。

## 4. 只读诊断内容

使用相同动作分别测试：

1. `arm_only right`；
2. `arm_only left`；
3. `arm_only both`；
4. GUI idle；
5. GUI recording。

每组记录和比较：

- Tracker 时间戳、有效位、单帧平移和旋转；
- 上臂 Tracker/ZSP；
- 左右 IK 成功标志；
- IK 候选关节解；
- 实际发送的左右关节命令；
- 左右关节反馈；
- 控制频率和控制周期；
- SDK IK 错误计数；
- 相机与 Recorder 性能统计。

诊断代码必须满足：

- 不改变 Tracker 输入；
- 不改变 IK seed 或求解结果；
- 不改变硬件命令；
- 不因为日志、磁盘或 GUI 阻塞控制循环；
- 历史缓存必须有界。

## 5. 验证流程

每次相关修改后执行：

1. 单元测试和构建；
2. `arm_only right` 60秒；
3. `arm_only left` 60秒；
4. `arm_only both` 60秒；
5. GUI idle 60秒；
6. GUI recording 60秒；
7. Tracker clutch断开与重连；
8. 保存并检查episode；
9. 正常退出并确认机械臂Standby。

验收目标：

- 控制稳定在约120 Hz；
- 正常动作方向不改变；
- 有效 IK 解在 handoff 后直接发送；
- 一侧 IK 无解时另一侧继续发送；
- Tracker遥操不进入`TARGET_HOLD`；
- `qpos/action/eef`格式保持不变；
- 原始Tracker数据继续完整保存。

## 6. 后续分析顺序

```text
固定当前原生控制语义
  → 左右臂与左右Tracker交叉对照
  → 对齐Tracker、ZSP、IK和关节反馈时间戳
  → 区分设备跳点、遮挡、奇异位形和映射问题
  → 输出诊断结论
  → 如确需改变控制行为，再单独提出方案并进行真机评审
```
