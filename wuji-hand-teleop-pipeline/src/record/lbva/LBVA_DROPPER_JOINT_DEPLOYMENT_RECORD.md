# LBVA Dropper Joint 部署记录

## 1. 核心配置

以 `src/wuji_data_pipeline/config/lingbot_va_fdm.yaml` 当前内容为准。

| 配置项 | 当前值 |
|---|---|
| checkpoint | `lingbot-va/singlerighthand/dropper-100-joint/step-30000` |
| 协议 | FDM V3，HTTP `/v1/robot-policy` |
| 执行硬件 | 右臂 7 维 + 右手 20 维 |
| `action_mode` | `joint` |
| 模型动作频率 | 30 Hz |
| 底层控制发布 | 120 Hz |
| Tianji 状态读取 | 500 Hz |
| wire chunk | 48 步 |
| native horizon | 首段 48 步，后续 64 步 |
| feedback | 每 4 个动作一次，即 7.5 Hz |
| 插值 | `pchip_joint` |
| 额外平滑/边界融合 | 关闭，`initial_blend_steps=0`、`boundary_blend_steps=0` |
| 异步预取 | 开启，初始提前 12 步，范围 3～47 步 |
| state history | 开启 |
| 相机 | `head + right_wrist` |
| 图像传输 | 直接共享内存，不经过 ROS；JPEG 质量 90 |
| 网络请求超时 | 30 秒 |
| pending 缺失 | 保持最后目标，最长 5 秒后安全停止 |
| 反馈队列 | 64 组 |
| 反馈重试 | 5 次，间隔 0.1 秒 |

checkpoint 只需要修改 YAML 中的两处，并保持完全一致：

```yaml
policy_http_expected_model_id: "新checkpoint"

fdm_async:
  model_id: "新checkpoint"
```

## 2. Joint 与 EEF 版本的重点区别

| 项目 | EEF 版本 | Joint 版本 |
|---|---|---|
| 机械臂动作 | `ee_pos[3] + ee_quat[4]` | `joint_pos[7]` |
| 机械臂单位 | 米 + `xyzw` 四元数 | 弧度 |
| 手部动作单位 | 度 | 弧度 |
| 执行路径 | EEF → IK → 关节控制 | 直接关节控制，不经过 IK |
| 插值 | `pchip_slerp` | `pchip_joint` |
| 状态历史 | 可选 | 必须开启 |
| 反馈 | EEF 目标和手目标 | 关节目标和手关节目标 |

Joint 模式中，弧度只在最终调用 Tianji SDK 前转换为度；通信、日志和反馈始终保存弧度。

协议保持固定 54 维顺序：

```text
左臂 7 + 左手 20 + 右臂 7 + 右手 20
```

当前只执行右侧 27 维；左侧不会下发控制，但保留左侧协议槽位和真实左臂状态。每次反馈的 `qpos_history` 为 `(4, 54)`：`executed_actions` 表示下发目标，`qpos_history` 表示机器人真实反馈，二者不能互相替代。

## 3. 本次问题及解决方案

1. **ZMQ 能 ping 但无法连接云端端口**

   集群内部节点不能直接访问，改为平台暴露的 HTTP 在线服务域名，统一使用 `/v1/robot-policy`。

2. **`/wuji_deployment` 启动即崩溃**

   原因是 YAML 中两处模型 ID 不一致。现已增加启动检查，两处 checkpoint 不同会直接拒绝启动；测试不再硬编码具体 checkpoint。

3. **Joint 模式被错误地当作 EEF 模式**

   原因是 `action_mode` 缺失或启动脚本错误传递参数。现固定 `action_mode: joint` 和 `state_history.enabled: true`；`deployment_session` 自动从 YAML 选择 joint，不再要求额外命令行参数。

4. **`ROBOT_LAYOUT_MISMATCH` 或左臂 EEF 数据非法**

   Joint 只改变动作格式，不改变固定的机器人观测布局。保留双侧 54 维 qpos 和双侧有限 EEF 状态；动作布局改为 joint，动作单位全部改为弧度。

5. **握手一直未完成或服务返回 503**

   同类原因包括云端模型未就绪、相机未产生有效帧。当前要求 `head` 和 `right_wrist` 同时在线，并采用直接共享内存取图。

6. **READY 后立刻进入 DISABLED**

   ENABLING 期间 Tianji 状态话题会短暂停顿，首个反馈可能读到过期 qpos。现改为 READY 后等待第一帧新鲜机器人状态，再执行 W0。

7. **启动后立即按 `r` 提示缺少 ROS 节点**

   Tianji 初始化约需要 3 秒。现启动时最多等待 ROS 图 15 秒并清空等待期间的按键；
   超时会报告缺失节点，后续 Recovery preflight 仍会拒绝不完整的 ROS 图。

8. **Recovery/Enable 模式切换超时**

   若左右臂保持 `state=0`、伺服错误全为 0，但无法切换到位置模式 1 或阻抗模式 3，属于 Tianji SDK/控制权问题，不是模型动作问题。需要关闭竞争的上位机 SDK 客户端，检查外部控制模式、控制线，必要时重启控制柜。

9. **右手未张开或启用后立刻关闭**

   Joint 版本先完成右手 5 秒初始姿态恢复；Enable 时先使能右手，再使能 Tianji，保证首个 `qpos_history` 包含有效右手状态。
