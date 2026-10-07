# 当前样例的数据格式

样例目录 `tianji_wuji_data/` 是一条已 finalize 的 episode：

```text
tianji_wuji_data/
├── lmdb/{data.mdb,lock.mdb}     # 数值 aggregate + 逐帧 pickle
│   └── meta_info/__metadata__   # 内嵌 schema、维度、相机和诊断描述
└── sync_timestamps.json         # 每帧各数据源时间戳
```

这是用于运动学 Replay 的紧凑副本，因此省略了不参与关节回放的 RGB 视频，并
利用 LMDB 中原本就存在的 metadata，未重复提交 `meta_info.pkl`。从数采目录
传入普通完整 episode 时，程序仍优先读取外部 `meta_info.pkl`。

样例实际为 2266 帧、30 Hz（约 75.53 秒），一台 `head` 相机，视频
metadata 记录的原视频为 640×480、2266 帧，但紧凑副本不携带视频文件。只启用
了右 Wuji 手；左手的 20 个槽位按 schema 保留并填零。
双 Tianji 臂仍都存在（`arm_mode=dual`）。

## 训练/回放主数组

所有 aggregate 数组以完整 key 存在 LMDB 中；同一数据也以
`<key>/000000` 形式逐帧保存。数值类型是 float32。

| LMDB key | 样例 shape | 单位/含义 |
|---|---:|---|
| `action` | `(2266,54)` | 每侧 `[实测 EEF 7, 手目标 20]`；位置 m、四元数 xyzw、手 degree |
| `action_eef` | `(2266,14)` | 左、右实测 EEF，各 `[x,y,z,qx,qy,qz,qw]` |
| `action_bases` | `(2266,6)` | 保留的基座 action；当前全零 |
| `/observations/qpos` | `(2266,54)` | 每侧 `[臂关节 7, 手关节 20]`，全部 rad |
| `/observations/qvel` | `(2266,54)` | 对应关节速度，rad/s |
| `/observations/effort` | `(2266,54)` | 臂/手驱动返回的 effort |
| `/observations/eef` | `(2266,14)` | 左、右实测 EEF；本样例与 `action_eef` 相同 |
| `/observations/robot_base` | `(2266,6)` | 保留的基座状态；当前全零 |
| `/observations/hand_joint_deg` | `(2266,40)` | 左、右手实测关节角，degree |
| `/diagnostics/commanded_eef` | `(2266,14)` | 左、右臂命令 EEF |
| `/diagnostics/arm_joint_command` | `(2266,14)` | 左、右臂关节命令，rad |
| `/diagnostics/zsp` | `(2266,6)` | 左、右臂各 3 维 ZSP |
| `/sync_timestamps` | 2266 records | 与 JSON 相同的同步记录 |

54 维 `qpos` 的准确切片来自 `meta_info.pkl/robot_layout/qpos_layout`：

```text
[ 0: 7] left arm       [ 7:27] left hand
[27:34] right arm      [34:54] right hand
```

每只手的 20 维按 `finger1..finger5`，每指 `joint1..joint4` 映射。臂按
`Joint1..Joint7`。采集器当前只保存数值数组，没有把 ROS `JointState.name`
逐帧写进 episode，因此这个名字顺序是采集/驱动和组合 URDF 之间必须维持的
接口约定。

## 同步时间

每帧以相机帧为 anchor。样例记录 arm state/command、actual/target EEF、ZSP、
右手 state/command、head camera，以及可选的 tracker/MANUS 对齐信息。样例的
平均同步误差约 5.55 ms，最大约 34.75 ms，`frame_sync_vqe≈0.9882`，
`sync_skip_count=27`。

## 相机和可选遥操作诊断

视频在 LMDB 外保存，`meta_info.pkl/videos` 提供相对路径、帧数、尺寸、编码和
帧率。LMDB 的 `/teleop/*` 是分析诊断，不是 replay 必需输入，包括：

- 5 个 OpenVR 角色（chest、左右 wrist、左右 arm）的 raw/corrected pose、
  线/角速度、连接和 tracking 状态；
- 左右 MANUS 的 25 节点 raw pose、21 keypoints、20 ergonomics、5 个 raw
  sensor pose、ID/有效性/时间对齐信息；
- 缺失数据按 metadata 中的 fill value 填充，并配套 available/valid 标志。

样例左 MANUS 不可用，右 MANUS 可用。Replay 只读
`/observations/qpos`；相机 MP4 与上述诊断仍保留供可视化或离线分析。

## URDF 与 replay 的关系

组合 URDF 有 84 个 link、83 个 joint，其中恰好 54 个非固定 revolute joint：
双臂 14 + 双手 40。URDF 文件中的关节声明顺序是“双臂后双手”，与数据的
“左侧臂手、右侧臂手”不同。因此 replay 必须按关节名查询 MuJoCo qpos 地址，
不能直接假设 `model.qpos[:] = dataset_qpos`。
