# 阶段 B：GUI 状态缓存与硬件查询分级

> 日期：2026-07-24  
> 状态：已完成并通过实机 GUI 性能验收

## 1. 目的

阶段 A 确认数采 GUI 每秒调用一次完整
`/tianji_arm_controller/arm_status`。该服务同步读取双臂状态、错误码和
14 个伺服错误，单次耗时约 57 ms，并运行在 Tianji 控制节点的 ROS
执行器中。

实测对应关系：

| 场景 | `arm_status` 调用 | 控制调度间隙 | 控制频率 |
| --- | ---: | ---: | ---: |
| `arm_only` | 0 | 1 | 119.98 Hz |
| `gui_idle` | 61 | 60 | 113.64 Hz |
| `gui_recording` | 60 | 61 | 113.96 Hz |

本阶段的目的，是停止 GUI 在 READY 状态周期性查询硬件，让 GUI 读取内存
状态快照，同时保留 Recovery、故障诊断和退出时的真实硬件检查。

## 2. 修改前

```text
GUI 每 1 秒
    │
    ▼
/tianji_arm_controller/arm_status
    │
    ▼
控制节点服务回调
    │
    ├── SDK 双臂状态读取
    ├── 左臂 7 个伺服错误
    └── 右臂 7 个伺服错误
         耗时约 57 ms
    │
    ▼
同一 ROS 执行器中的 120 Hz 控制回调等待
```

## 3. 修改后

```text
现有 500 Hz 状态读取
    │
    ├── 关节反馈
    ├── state code
    └── err_code
          │
          ▼
  RobotStatusSnapshot 内存缓存
          │
          ▼ 2 Hz，无 SDK 调用
/tianji_arm/status_snapshot
          │
          ▼
         GUI
```

完整硬件查询仍保留：

```text
/tianji_arm_controller/arm_status
```

但只允许现有安全流程和明确诊断使用，不再由 GUI 定时器调用。

## 4. 轻量状态快照

话题：

```text
/tianji_arm/status_snapshot
```

消息类型：

```text
std_msgs/msg/String
```

内容为 JSON，主要字段：

```text
schema_version
sequence
published_at_unix_s
lifecycle_state
teleop_status
arm_enabled
arm_state
active_arm
control_source
tracker_connected
motion_in_progress
target_hold
target_hold_reason
sdk_fault_latched
feedback_age_s
arms.left.state
arms.left.err_code
arms.right.state
arms.right.err_code
detailed_hardware_status.available
detailed_hardware_status.age_s
```

快照只读取 Python 内存字段。生成和发布快照的函数不得调用
`TianjiChestDriver` 或 Marvin SDK。

## 5. 复用现有 SDK 状态帧

当前 `get_current_joints()` 的同一次 Marvin `subscribe` 返回：

- 双臂关节反馈；
- 双臂 `cur_state`；
- 双臂 `err_code`。

阶段 B 只把同一帧中已经存在的 `cur_state/err_code` 缓存下来，没有增加
SDK 调用次数。GUI 因此仍可显示：

```text
left: state=<值>, err=<值>
right: state=<值>, err=<值>
反馈年龄=<值>
```

伺服详细错误不会以 2 Hz 重复读取。

## 6. 安全路径保持不变

以下流程继续调用完整硬件 `arm_status`：

1. Recovery 前的双臂状态检查；
2. Recovery 失败后的诊断；
3. 请求退出后的双臂 Standby 确认；
4. Replay/部署启动前的硬件检查；
5. 明确的故障诊断。

因此阶段 B 不会把 Recovery 或退出确认改成只相信 Python 缓存。

## 7. GUI 行为

GUI 的 1 Hz `_poll_system()` 现在只检查 ROS 节点是否在线，不再创建或调用
`arm_status` client。

GUI 单独订阅 2 Hz 缓存状态：

```text
/tianji_arm/status_snapshot
```

Lifecycle 和文字原因仍使用原来的事件话题：

```text
/tianji_arm/lifecycle_state
/tianji_arm/teleop_status
```

Recorder 状态、脚踏板逻辑、相机显示和会话控制均未改变。

## 8. 不在本阶段修改的内容

- 120 Hz 控制频率；
- 500 Hz 状态读取频率；
- IK 算法；
- Tracker 映射；
- Recovery/Enable；
- 阻抗参数；
- 相机 ROS 数据路径；
- Recorder 写入架构；
- 数据格式；
- Replay。

## 9. 代码测试

阶段 B 增加测试，确认：

1. 状态快照完全使用缓存，不接触 SDK；
2. 现有 joint feedback 的一次 SDK 帧同时缓存状态码和错误码；
3. GUI 订阅缓存话题；
4. GUI 源码不再包含周期性完整 `arm_status` client；
5. 完整 `arm_status` 服务仍然真实访问硬件并刷新详细诊断缓存；
6. 数采 launch 和 arm-only launch 都固定使用 2 Hz 快照。

首轮相关测试：

```text
77 passed
```

三个相关包完整测试：

```text
86 passed
```

五个受影响 ROS 2 包完成构建，并通过 colcon 完整回归：

```text
143 tests, 0 errors, 0 failures, 0 skipped
```

## 10. 验收目标

与阶段 A 相比，开启 GUI 后应达到：

```text
arm_status.calls                 0（正常 READY 期间）
control callback rate           ≥119 Hz
60 秒 control scheduling gaps   ≤5
控制周期不再每秒出现一次约 57 ms 停顿
```

操作员仍能在 GUI 看到双臂 state、err_code、反馈年龄、生命周期和错误原因。

## 11. 实机验收步骤

启动数采 GUI，完成 Recovery 和 Enable，进入稳定 READY。保持与阶段 A
相近的操作范围，在第二个终端执行：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
./src/scripts/collect_performance_baseline.sh stage_b_gui_idle 60
```

然后实际开始一条轨迹，再执行：

```bash
./src/scripts/collect_performance_baseline.sh stage_b_gui_recording 60
```

重点比较：

```text
counters.arm_status.calls
counters.control.scheduling_gaps
active_callback_rates_hz.control
series.control.period_ms.max
series.status_snapshot.duration_ms
```

## 12. 实机验收结果

在 GUI 中使用右手模式完成 Recovery 和 Enable，并分别进行 60 秒空闲测试
和 60 秒真实录制测试。结果如下：

| 指标 | 阶段 A `gui_idle` | 阶段 B `gui_idle` | 阶段 A `gui_recording` | 阶段 B `gui_recording` |
| --- | ---: | ---: | ---: | ---: |
| 控制频率 | 113.6391 Hz | 120.0019 Hz | 113.9649 Hz | 120.0001 Hz |
| 状态频率 | 466.8189 Hz | 497.0411 Hz | 468.4426 Hz | 497.6235 Hz |
| 控制调度间隙 | 60 | 1 | 61 | 0 |
| 推算丢失控制周期 | 383 | 1 | 363 | 0 |
| 完整 `arm_status` 调用 | 61 | 0 | 60 | 0 |
| 最大控制周期 | 70.6076 ms | 12.9480 ms | 67.0712 ms | 11.2049 ms |

缓存快照本身的开销：

| 场景 | 发布次数 | 平均耗时 | 最大耗时 |
| --- | ---: | ---: | ---: |
| 阶段 B `gui_idle` | 120 | 0.0601 ms | 0.1038 ms |
| 阶段 B `gui_recording` | 122 | 0.0622 ms | 0.1272 ms |

原始性能文件：

```text
/home/wuji/datasets/tianji_wuji/diagnostics/20260724_175318_stage_b_gui_idle.jsonl
/home/wuji/datasets/tianji_wuji/diagnostics/20260724_175318_stage_b_gui_idle.summary.json
/home/wuji/datasets/tianji_wuji/diagnostics/20260724_175624_stage_b_gui_recording.jsonl
/home/wuji/datasets/tianji_wuji/diagnostics/20260724_175624_stage_b_gui_recording.summary.json
```

录制期间实际完成了开始、结束和保存流程，最终轨迹为：

```text
/home/wuji/datasets/tianji_wuji/exchange_tubes/episode_0001_20260724_175600
```

轨迹包含 LMDB、`meta_info.pkl`、`sync_timestamps.json` 和 `videos/head.mp4`。
本次为右手模式，左手按既有规则补零；Tracker 原始数据和右侧 MANUS 原始
数据均已写入，训练帧同步跳过数为 0。

空闲测试期间仍记录到大量 TARGET_HOLD，但控制频率仍保持 120 Hz，且只出现
1 次调度间隙。这进一步证明：

- 阶段 B 已消除 GUI 硬件查询造成的调度阻塞；
- TARGET_HOLD 是独立的 Tracker/IK 可达性问题；
- HOLD 原因应在阶段 G 单独分析，不能与 GUI 性能问题混在一起。

## 13. 验收结论

阶段 B 的全部验收条件均已满足：

1. `gui_idle` 和 `gui_recording` 控制频率均达到 120 Hz；
2. 60 秒控制调度间隙分别为 1 次和 0 次；
3. READY 正常运行期间完整 `arm_status` 调用次数为 0；
4. 最大控制周期由约 67～71 ms 降至约 11～13 ms；
5. Recovery、Enable、真实录制、保存和退出 Standby 安全确认完成；
6. 退出后控制器、Recorder 和数采会话均已停止；
7. 控制参数、IK 保护、数据格式和相机录制逻辑均未改变。

退出日志明确记录：

```text
left,right STANDBY reached (cur_state=0)
Both arms at cur_state=0
Robot released
```

但控制器在完成上述安全动作后，最终进程退出码为 `-11`。该问题发生在
`Robot released` 和 `Safely exited` 之后，没有阻止双臂进入 Standby，也
没有留下控制器或 Recorder 进程；但它说明 Marvin SDK/Python 资源释放路径
仍需治理。该退出码问题不属于 GUI 状态缓存造成的实时阻塞，留到阶段 C 的
SDK 单一访问与生命周期隔离中处理和回归。

阶段 B 完成。下一阶段为阶段 C：Marvin SDK 单一访问与控制路径隔离。
