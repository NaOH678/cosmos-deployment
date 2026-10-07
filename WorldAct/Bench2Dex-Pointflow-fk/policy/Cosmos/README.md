# Cosmos 部署入口

当前 Task21 baseline 使用 **`backend=training_bundle`**，已在 4090 上完成真实模型推理和 1000 步仿真闭环联调。部署代码、接口、5090 迁移步骤和验证边界统一整理在：

**[5090 部署开发说明](../../deployment/cosmos/README.md)**

实际调用链：`script/policy_model_server.py` → `LocalSession` → `BundlePolicy` → 训练配套源码 `Bench2DexPolicy`。

本机配置是 `outputs/cosmos_local/deploy_baseline.json`。`deploy_policy.yml` 和 `deploy_policy.py` 中的 Robolab 通用实现是早期适配方案，不是这次 Task21 checkpoint 的验证路径。新机器用 `tools/prepare_cosmos_bundle.py` 生成配置，不要复制本机绝对路径。

完整记录在 `outputs/cosmos_local/closed_loop_episode01/`：三视角各 1000 帧，动作/状态有限，250 次推理中位数约 1.145 秒；任务失败、0/3 阶段完成，仍有抖动。当前仿真同步等待推理，真机异步控制器尚未实现。

兼容修改见 `bundle_compat.patch`。新机器从原始训练配套 tar 解压源码后，按部署说明应用一次补丁。本机已有源码已经修补。
