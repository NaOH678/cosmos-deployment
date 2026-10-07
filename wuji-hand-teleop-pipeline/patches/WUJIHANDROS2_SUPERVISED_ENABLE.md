# WujiHand 受控使能与 Recovery Patch 使用说明

## 1. 目的

数采和云端部署启动时会拉起 WujiHand 驱动。原始驱动连接硬件后会立即使能
所有手指，而且没有独立的固定初始姿态 Recovery，因此不能满足当前受控启动流程。

本 Patch 为一个以 `e513ea9` 为基线的累积补丁，同时提供受控使能和手部
Recovery：

1. `auto_enable:=false` 时只连接硬件、读取并发布状态，关节保持未使能。
2. `recover_to_initial` 服务从实测姿态平滑移动到配置的 20D 初始姿态；期间拒绝
   MANUS、replay 和 policy 外部目标。
3. `recovery_state` 以 transient-local topic 发布 `IDLE/MOVING/READY/FAILED`。
4. `require_recovery_before_enable:=true` 时，Recovery 未完成不能启用外部控制。
5. Recovery 完成后，手保持初始目标；session 再通过 `set_enabled` 放行外部控制。
6. 外部控制从使能瞬间的实测关节位置开始，沿用 `command_ramp_duration` 渐进接管。
7. 使能或 Recovery 失败时，session 关闭已经使能的手并请求 Tianji standby。
8. `x`、退出 GUI 或 `Ctrl+C` 时，先关闭 WujiHand，再关闭 Tianji。

该文件已取代早期只包含“延迟使能”的同名补丁；不要先应用旧版本再应用本版本。

Patch 文件：

```text
patches/wujihandros2-supervised-enable.patch
```

适用的 WujiHand 驱动基线：

```text
仓库：src/wujihandros2
commit：e513ea9c92c424ce3bfff5eeea4710c60d3a1ca8
branch：deploy-robotics54
```

## 2. 当前 Robotics54 机器

当前工作区的 `src/wujihandros2` 已经应用过本补丁，不要再次执行 `git apply`。

项目路径：

```text
/home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
```

可用下面的命令确认 Patch 已经存在：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline

git -C src/wujihandros2 apply --reverse --check \
  ../../patches/wujihandros2-supervised-enable.patch
```

命令无输出且退出码为 `0`，表示 Patch 已经应用，不需要重复操作。

## 3. 新机器或新工作区首次应用

先确认主项目和 WujiHand 驱动都已经准备好：

```bash
cd /path/to/wuji-hand-teleop

git rev-parse --short HEAD
git -C src/wujihandros2 rev-parse HEAD
```

推荐 `src/wujihandros2` 位于上面记录的 `e513ea9` 基线。先只检查，不修改文件：

```bash
git -C src/wujihandros2 apply --check \
  ../../patches/wujihandros2-supervised-enable.patch
```

检查成功后应用：

```bash
git -C src/wujihandros2 apply \
  ../../patches/wujihandros2-supervised-enable.patch
```

再确认已经应用：

```bash
git -C src/wujihandros2 apply --reverse --check \
  ../../patches/wujihandros2-supervised-enable.patch
```

## 4. 重新构建

Patch 修改了 C++ 驱动，必须重新构建，不能只重启 GUI：

```bash
docker exec wuji-hand-teleop bash -lc '
source /opt/ros/humble/setup.bash
cd /home/wuji/ros2_ws
colcon build --symlink-install --packages-select \
  wujihand_driver \
  wuji_teleop_bringup \
  wuji_data_pipeline \
  wuji_teleop_monitor
'
```

构建结果应为：

```text
Summary: 4 packages finished
```

关闭仍在运行的旧 GUI 和旧 session，然后重新启动：

```bash
cd /path/to/wuji-hand-teleop
./src/scripts/start_record_gui.sh
```

## 5. 真机验证顺序

保持实体急停可触达，并使用当前实际连接的手部模式。数采 GUI、Replay 和云端部署
session 都应遵循相同的 Recovery/Enable 顺序。

1. 点击“进入准备”。
   - WujiHand 驱动应在线并持续发布状态。
   - 手指不应使能，也不应跟随 MANUS。
2. 点击 Recovery。
   - Tianji 与所选 WujiHand 都开始各自的受控 Recovery。
   - 手部 `recovery_state` 应从 `1/MOVING` 进入 `2/READY`。
   - 手部移动期间不能接收 MANUS、replay 或 policy 目标。
3. Tianji 到达 `7/RECOVERY_READY` 且所选手全部为 `2/READY` 后点击 Enable。
   - Tianji 先执行 Enable。
   - Tianji 到达 `2/READY` 后，所选 WujiHand 才使能。
   - WujiHand 从当前实际姿态渐进过渡到 MANUS 或 policy 目标。
4. 点击退出或在终端按 `Ctrl+C`。
   - 日志应先出现 WujiHand disabled。
   - 随后 Tianji 回到 standby，双臂状态为 `0`。

单右手模式只调用：

```text
/right_hand/set_enabled
/right_hand/recover_to_initial
```

单左手模式只调用：

```text
/left_hand/set_enabled
/left_hand/recover_to_initial
```

双手模式会同时管理左右两只手；其中一只使能失败时，另一只也会回滚关闭。

## 6. 判断 Patch 当前处于什么状态

### 6.1 尚未应用

下面的命令成功：

```bash
git -C src/wujihandros2 apply --check \
  ../../patches/wujihandros2-supervised-enable.patch
```

### 6.2 已经应用

下面的命令成功：

```bash
git -C src/wujihandros2 apply --reverse --check \
  ../../patches/wujihandros2-supervised-enable.patch
```

### 6.3 两个检查都失败

说明 `src/wujihandros2` 的代码版本与 Patch 基线不同，或者文件还有其他未提交
修改。不要使用 `--reject` 或强制应用。先执行：

```bash
git -C src/wujihandros2 status --short
git -C src/wujihandros2 rev-parse HEAD
```

再根据实际版本人工合并。

## 7. 撤销 Patch

只有在确认需要同时移除受控使能和手部 Recovery 能力时才撤销：

```bash
cd /path/to/wuji-hand-teleop

git -C src/wujihandros2 apply --reverse \
  ../../patches/wujihandros2-supervised-enable.patch
```

撤销后同样必须重新构建 `wujihand_driver`。原始行为会恢复为驱动连接后立即使能
所有手指，并失去 `recover_to_initial`/`recovery_state`，不适合当前数采和部署流程。

## 8. 相关主仓库配置

Patch 只负责 WujiHand 驱动能力。主仓库还包含以下配套配置，缺一不可：

- `wuji_teleop_hand.launch.py` 配置 `auto_enable`、20D `initial_position`、Recovery
  时长/容差以及 `require_recovery_before_enable`。
- `record.launch.py` 在数采模式传入 `auto_enable:=false`。
- 手部 `command_ramp_duration` 为 `5.0` 秒。
- `deployment.launch.py` 同样以受控模式启动手部驱动。
- `record_session.py` 同时管理手臂和所选手的 Recovery 状态，并在 Tianji READY 后
  调用手部 `set_enabled` 服务。
- 退出路径先关闭手，再请求 Tianji standby。

主仓库配套文件与本 Patch 应一起纳入同一轮代码管理；不要只提交主仓库调用端而
遗漏本补丁。
