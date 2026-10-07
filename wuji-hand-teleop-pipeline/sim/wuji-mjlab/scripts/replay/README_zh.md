# Tianji + Wuji 遥操作轨迹 MuJoCo Replay
这个工具只读取已经完成的 episode，不依赖 ROS 2，不连接机器人，也不会启动或
修改遥操作节点。它把 LMDB 中实测的 `/observations/qpos` 映射到组合 URDF 的
54 个转动关节：

```text
数据顺序:
  left:  Joint1_L..Joint7_L + left_hand_finger1..5_joint1..4
  right: Joint1_R..Joint7_R + right_hand_finger1..5_joint1..4

URDF/MuJoCo 内部顺序:
  不作假设；程序按关节名字查询 qpos 地址。
```

## 最小环境

系统 Python 已经有这些包时可以直接运行。否则在本目录创建独立环境：

```bash
cd sim/wuji-mjlab
python3 -m venv .venv-replay
.venv-replay/bin/python -m pip install -r requirements-replay.txt
```

这套环境不安装 mjlab、PyTorch、CUDA 或 ROS。

## 检查数据和模型

```bash
python3 scripts/replay/replay_teleop.py --inspect
```

默认读取本仓库的 `tianji_wuji_data` 和
`marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf`。
仓库中的紧凑样例没有重复保存 `meta_info.pkl`，Replay 会从 LMDB 的
`meta_info`/`__metadata__` key 读取同一份 metadata；普通完整 episode 仍优先
读取目录中的 `meta_info.pkl`。
也可以显式指定：

```bash
python3 scripts/replay/replay_teleop.py \
  --episode-dir /path/to/episode_xxxx \
  --urdf marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf \
  --inspect
```

## 打开 MuJoCo Viewer

```bash
python3 scripts/replay/replay_teleop.py
python3 scripts/replay/replay_teleop.py --speed 0.5 --loop
python3 scripts/replay/replay_teleop.py --start 300 --stop 900
```

无显示器机器上的完整快速验证：

```bash
MUJOCO_GL=egl python3 scripts/replay/replay_teleop.py \
  --headless --no-realtime
```

## 导出 MJCF

MuJoCo 可以直接导入已解析资源路径的 URDF。下面命令同时检查模型，并把编译后
模型保存为 MJCF：

```bash
python3 scripts/replay/replay_teleop.py \
  --inspect --export-mjcf generated/marvin_wuji_d435.xml
```

`generated/` 是派生文件；源 URDF 和遥操作工程不会被修改。

## 导出 Tianji Base 坐标系下的 Wuji FK-21

```bash
python3 scripts/replay/export_wuji_fk21.py
```

默认输出：

```text
generated/tianji_wuji_fk21.npz
generated/tianji_wuji_fk21.json
```

`positions` shape 为 `(帧数, 2, 21, 3)`，左右手顺序是
`[left, right]`，单位为 metre，坐标系为 `Link_Base`。21 点顺序与
HaWoR/OpenPose/MediaPipe 相同。默认使用未裁剪的实机反馈做 FK；需要按 URDF
limit 裁剪时显式增加 `--clip-limits`。

当前样例只有右手是实测数据，`side_is_observed=[false,true]`；左手位置是“左臂
实测姿态 + 左手零填充关节”的模型结果，不能当作左手真实动作。

批量处理指定任务目录中的所有 `episode_*`：

```bash
python3 scripts/replay/batch_export_wuji_fk21.py \
  /path/to/task_directory
```

结果不写回原始数据目录，而是默认在输入目录旁建立 `<任务名>_fk21/`，并保持
`episode_xxxx/annotations/{wuji_fk21.npz,wuji_fk21.json}` 结构。例如输入
`sandwich/` 时输出到 `sandwich_fk21/`。可用 `--output-root /path/to/output`
指定其他位置。已有完整结果默认跳过；使用 `--force` 重新生成，使用
`--dry-run` 只检查将要处理的轨迹。

可视化机器人、FK-21骨架、21点短时轨迹和Tianji左右TCP轨迹：

```bash
python3 scripts/replay/visualize_fk21.py --loop
```

默认只画有真实手部反馈的一侧；增加 `--show-unobserved` 可同时显示零填充侧。
所有叠加点和TCP轨迹都是 `Link_Base` 坐标系下的米制坐标。

## 回放语义

- 使用 `qpos`（臂和手均为 rad），它是机器人实际关节观测。
- 不用 `action` 驱动关节：其中臂数据是末端位置 + xyzw 四元数，手数据是 degree。
- 这是逐帧 `mj_forward` 的运动学回放，用于核对数据、关节顺序和模型外观；它
  不表示 Tianji/Wuji 的电机、阻尼、接触和控制器动力学已经辨识。
- 超出 URDF limit 的实测值会仅在仿真副本中裁剪，结束时报告涉及帧数和最大
  超限；原始数据不变。
- `meta_info.pkl` 和 LMDB 使用 Python pickle，只应读取自己采集、可信的数据。

## 常用指令
给单条轨迹生成FK-21坐标
```
.venv-replay/bin/python scripts/replay/export_wuji_fk21.py \
    --episode-dir ../../datasets/tianji_wuji/sandwich/episode_0040_20260730_173530 \
    --output ../../datasets/tianji_wuji/sandwich_fk21/episode_0040_20260730_173530/annotations/wuji_fk21.npz
```
