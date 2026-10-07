# 统一仓库导入记录

日期：2026-10-07。来源是本机 `ros2_ws/worktrees` 下的工作源码快照，原目录未移动，原修改未回退。

## 导入原则

- 纳入 teleop-pipeline 与 WorldAct 下七个代码项目，并保留 Edge 模型包转换代码和配置、两个 checkpoint run 的 frozen YAML。
- 将嵌套项目源码作为普通文件纳入统一版本管理；不复制 `.git` 或 submodule 元数据。`wujihandros2` 保持外部依赖，仅保留父项目中的补丁和说明。
- 不导入权重、运行环境、缓存、录制数据、密钥、本机 live config、SDK 二进制和超过 5 MiB 的资产。二进制网格和图片等也不在这次源码快照内。少量已有实验结论 JSON 作为文档证据保留。
- 原仓库 `.gitignore` 被保留；首次导入使用经过筛选的显式文件清单，以保留被原项目忽略但确实属于本次研究的源码。后续不要用全目录强制添加绕过排除规则。
- 各目录原有许可文件沿用，没有给第三方项目重新授权。

## 来源与历史

`docs/source_snapshot.json` 记录每个纳入文件的**原始** SHA256、大小和排除原因。`projects` 中的 `source_head` 只表示能读取到的源 HEAD；不能代表导入文件没有本地修改。

Teleop 来源为 `3e94398a365e619b2175e272fcac92315d20f2dc`。SFT、PointFlow/FK、SFT_base、WorldAct-bench2dex 的 `.git` 指向已不可访问的主仓库 metadata，因此本次只能保存当前源码，不能证实它们当前与远端一致，也没有恢复或合并历史。Omni 上游 revision 见 `WorldAct/omni-wam-lab/upstream.json`，同时保留本地修改后的源码和补丁。

## 在导入副本内的调整

1. `wuji-hand-teleop-pipeline/src/scripts/start_local_cosmos_deployment.sh` 的默认 `WORLDACT_ROOT` 改为从自身目录定位兄弟 `WorldAct/WorldAct-sft-pointflow-fk`，避免新仓库启动时默默调用旧工作区。显式环境变量覆盖仍可用。
2. 添加根 README、忽略规则、外部依赖说明和统一 VS Code workspace；更新 WorldAct 导航对迁移状态的说明。
3. 模型计算、采样参数、控制算法和现有部署 YAML 未因本次导入而修改。

所以 snapshot 哈希表示复制前来源，不是对文档与上述启动器调整后的哈希声明。完整导入后版本由新仓库提交标识。

## 复现边界

保留各研究分支的绝对路径配置，便于追溯原实验；没有批量替换冻结训练配置。依赖缺失和历史脚本的路径需要迁移者逐项配置。原实验文档部分链接指向未纳入的录像、trace 或大资产，属于外部证据，不代表这些数据随仓库提供。

新仓库导入只做离线检查；原工作区的 ROS/GPU 验证结果不能被表述为已在新目录完成同样验证。
