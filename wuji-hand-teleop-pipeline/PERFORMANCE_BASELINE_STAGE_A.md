# 阶段 A：Tianji 数采遥操性能基线

本阶段只增加性能观测，不修改遥操映射、IK、Recovery、阻抗参数、安全阈值、
数据格式、GUI 控制逻辑或 Replay。

## 1. 观测内容

控制节点每秒在 `/tianji_arm/performance_metrics` 发布一个 JSON 快照，包括：

- `control`：120 Hz 控制定时器的实际频率、周期、执行耗时和超期次数；
- `state`：500 Hz 状态定时器的实际频率、周期、执行耗时和超期次数；
- `eef`：60 Hz FK/末端状态发布定时器的实际频率和耗时；
- `driver.state.sdk_read_ms`：状态发布中的 SDK 关节读取耗时；
- `driver.pose.sdk_reference_read_ms`：每次 IK 前的 SDK 参考关节读取耗时；
- `driver.pose.ik_compute_ms`：双臂 IK 计算耗时；
- `driver.command.state_guard_ms`：命令前硬件状态保护检查耗时；
- `driver.command.sdk_send_ms`：SDK 发送关节命令耗时；
- `arm_status.duration_ms`：GUI 完整硬件状态查询耗时；
- `target_hold.*`、`sdk.errors`：HOLD 和 SDK 错误计数。

所有逐帧样本只保存在内存中的有界窗口里。每秒发布后清空窗口，不写控制日志，
不会无限占用内存。

## 2. 使用新 worktree

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
```

首次切换到这个 worktree 后，需要让容器重新绑定这里的源码：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline/docker
docker compose up -d --force-recreate
```

容器启动时会按现有入口逻辑构建 ROS 2 工作区。

## 3. 采集一组 60 秒基线

系统进入 `READY` 并保持正常遥操后，在另一个终端执行：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
./src/scripts/collect_performance_baseline.sh arm_only 60
```

脚本结束后会生成：

```text
datasets/tianji_wuji/diagnostics/<时间>_arm_only.jsonl
datasets/tianji_wuji/diagnostics/<时间>_arm_only.summary.json
```

JSONL 保存每秒快照，`summary.json` 保存整段统计结果。

## 4. 三组必须分别采集的场景

每组保持相似的手臂运动范围和速度，持续 60 秒：

1. `arm_only`：只启动 arm-only 遥操，不启动 GUI、相机和 Recorder；
2. `gui_idle`：启动数采 GUI 和已连接相机，进入 READY，但不开始轨迹；
3. `gui_recording`：启动数采 GUI、相机，并实际采集一条轨迹。

对应采集命令只有标签不同：

```bash
./src/scripts/collect_performance_baseline.sh arm_only 60
./src/scripts/collect_performance_baseline.sh gui_idle 60
./src/scripts/collect_performance_baseline.sh gui_recording 60
```

不要同时执行三条。切换到对应场景后再运行该场景的命令。

## 5. 首轮对比重点

主要比较三个 summary 中：

- `callback_rates_hz.control`；
- `active_callback_rates_hz.control`；
- `series.control.period_ms.worst_window_p99`；
- `series.control.duration_ms.worst_window_p99`；
- `series.driver.state.sdk_read_ms.mean/max`；
- `series.driver.pose.sdk_reference_read_ms.mean/max`；
- `series.driver.pose.ik_compute_ms.mean/max`；
- `series.driver.command.sdk_send_ms.mean/max`；
- `counters.control.scheduling_gaps`；
- `counters.control.execution_overruns`；
- `counters.arm_status.calls` 和 `series.arm_status.duration_ms.max`。

阶段 A 先用实测结果定位负载来源。降低 500 Hz 状态读取、缓存
`arm_status`、拆 SDK 线程、相机离开 ROS 和异步 Recorder 都属于后续阶段，
不能在本阶段混入，以免失去可比较的原始基线。

## 6. 2026-07-24 实测结果

本轮三组基线文件：

```text
20260724_172126_arm_only.summary.json
20260724_165940_gui_idle.summary.json
20260724_171134_gui_recording.summary.json
```

三组测试都使用双臂 Tracker 遥操，控制目标频率为 120 Hz，状态读取目标频率
为 500 Hz，EEF 发布目标频率为 60 Hz。

| 指标 | `arm_only` | `gui_idle` | `gui_recording` |
| --- | ---: | ---: | ---: |
| 实际控制频率 | 119.98 Hz | 113.64 Hz | 113.96 Hz |
| 实际状态频率 | 479.45 Hz | 466.82 Hz | 468.44 Hz |
| 实际 EEF 频率 | 59.98 Hz | 57.20 Hz | 57.75 Hz |
| 控制周期最大值 | 13.33 ms | 70.61 ms | 67.07 ms |
| 控制调度间隙 | 1 | 60 | 61 |
| 估算丢失控制周期 | 1 | 383 | 363 |
| `arm_status` 调用次数 | 0 | 61 | 60 |
| `arm_status` 平均耗时 | 无调用 | 57.20 ms | 57.50 ms |
| `arm_status` 最大耗时 | 无调用 | 62.40 ms | 60.76 ms |
| TARGET_HOLD 帧数 | 230 | 101 | 2138 |
| TARGET_HOLD 状态切换 | 7 | 8 | 1259 |

详细 SDK/IK 耗时：

| 指标（平均值） | `arm_only` | `gui_idle` | `gui_recording` |
| --- | ---: | ---: | ---: |
| SDK 状态读取 | 0.180 ms | 0.123 ms | 0.121 ms |
| IK 前 SDK 参考关节读取 | 0.181 ms | 0.129 ms | 0.122 ms |
| IK 计算 | 0.141 ms | 0.105 ms | 0.102 ms |
| SDK 命令发送 | 0.0012 ms | 0.0010 ms | 0.0010 ms |

## 7. 阶段 A 结论

### 7.1 GUI 完整硬件查询是已确认的周期性阻塞源

`arm_only` 在 59 秒内只有 1 次控制调度间隙，控制频率基本达到目标
120 Hz。两个 GUI 场景的控制频率都下降到约 114 Hz，并分别出现 60 和
61 次调度间隙。

GUI 场景同时每秒调用一次完整 `arm_status`。每次查询阻塞约 57 ms，
调用次数与控制调度间隙次数近似一一对应；控制周期最大值也从 arm-only
的 13.33 ms 增加到 67～71 ms。这是本阶段最明确的负载来源。

### 7.2 Recorder 不是本轮控制降频的主要来源

从 `gui_idle` 切换到 `gui_recording` 后，控制频率没有继续下降，控制调度
间隙只从 60 变为 61，完整硬件查询耗时也基本相同。因此在当前实现和本轮
数据量下，没有证据表明 Recorder 或磁盘写入是约 114 Hz 控制频率的主要
原因。

该结论只表示 Recorder 没有造成可见的额外控制降频，不表示当前同步保存、
队列容量和相机写入架构已经满足最终部署要求。

### 7.3 GUI 录制样本包含严重的独立 IK/HOLD 问题

`gui_recording` 中出现 2138 个 TARGET_HOLD 帧和 1259 次状态切换。日志显示
主要原因为左臂 IK 帧间跳变超过 35° 安全阈值。该问题会直接造成控制目标
间歇性保持，是遥操卡顿的另一条独立路径，不能归因于 Recorder 负载。
这是阶段 A 当时实现的历史测量；对应跳变保护已于 `2026-07-27` 删除，不代表
当前控制路径。

由于三组测试中的人体运动和 IK 可达性不可能完全一致，IK、SDK 单次执行
耗时不应用来单独证明 GUI 使计算变快或变慢。调度间隙、`arm_status` 调用
次数及阻塞耗时之间的对应关系更具诊断价值。

## 8. 阶段 B 的固定入口

阶段 B 首先处理已经由基线确认的问题：

1. GUI READY 状态下停止每秒执行完整硬件 `arm_status` 查询；
2. 控制节点维护轻量级状态缓存，GUI 只读取缓存快照；
3. 详细伺服错误仅在 Recovery、故障、Standby 或人工刷新时查询；
4. 优化前后重新执行相同三组基线，验收目标是 GUI 场景不再出现每秒一次
   的 57 ms 控制阻塞，实际控制频率恢复到接近 arm-only。

相机脱离 ROS、SDK 唯一访问线程和 Recorder 异步化继续保留为后续独立阶段，
避免在阶段 B 同时改变多个变量。TARGET_HOLD/IK 问题也单独处理，不通过放宽
安全阈值掩盖。

阶段 A 到此完成。本阶段没有修改遥操映射、IK、Recovery、阻抗参数、安全
阈值、数据格式或 Replay 行为。
