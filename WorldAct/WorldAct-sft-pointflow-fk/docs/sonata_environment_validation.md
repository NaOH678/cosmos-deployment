# Sonata small 在 Cosmos 环境中的验证

日期：2026-09-08。最新状态：三个依赖已安装；Sonata 官方代码已接入，small checkpoint 严格加载和 CPU 预处理验证通过。GPU 前向/反向尚未验证。下文保留安装前检查及后续操作记录。

## 环境实测

使用本仓库 `.venv/bin/python`，并清空 `LD_LIBRARY_PATH`。该虚拟环境实际链接到 `../WorldAct-cosmos/.venv`，本次未修改共享环境。

| 项目 | 结果 |
|---|---|
| 当前主机 | scjdex |
| Python | 3.13.14 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA 版本 | 12.8 |
| 当前进程 CUDA 可用性 | False，device_count=0 |
| cosmos_framework | 导入成功 |
| timm | 1.0.25，导入成功 |
| flash_attn | 2.7.4.post1，导入成功；未验证 CUDA kernel |
| spconv.pytorch | ModuleNotFoundError: No module named 'spconv' |
| torch_scatter | ModuleNotFoundError |
| addict | ModuleNotFoundError |
| sonata | ModuleNotFoundError |

当前进程没有可见 GPU 不代表 H200 训练节点不可用；GPU 前向、反向及 CUDA 扩展兼容性仍需在实际训练节点验证。

## 权重读取

文件：`../checkpoints/ptv3/sonata_small.pth`，约 148 MiB。

通过 `torch.load(path, map_location='cpu', weights_only=True)` 成功完整读取，包含 `config` 和 `state_dict`。state_dict 有 273 个张量、38,648,992 个张量元素；此数值是 checkpoint 张量统计，尚非实例化模型后的参数统计。

关键配置：

- `in_channels=9`。
- `enc_depths=(2,2,2,6,2)`，`enc_channels=(32,64,128,256,512)`。
- `enc_num_head=(2,4,8,16,32)`，各级 `enc_patch_size=1024`。
- `enable_rpe=False`，`enable_flash=True`。
- `enc_mode=True`，`traceable=True`，`mask_token=True`。

## 与本地原版 PTv3 的差异

Sonata checkpoint 的输入层为 `embedding.stem.linear.weight`，形状 `(32,9)`，另有 linear bias 和 `embedding.mask_token`。

本地原版采用 5×5×5 的 `SubMConv3d` 输入层，默认 `in_channels=6`；构造函数使用 `cls_mode`，不接受上述 Sonata 专用的 `enc_mode`、`traceable`、`mask_token` 参数。来源：[原版输入层 model.py:753](../../PointTransformerV3/model.py:753)、[原版配置 model.py:786](../../PointTransformerV3/model.py:786)。依赖导入见 [model.py:11](../../PointTransformerV3/model.py:11)。

因此不能将该 checkpoint 的 config/state_dict 原样直接加载到本地原版类中。仅设置 `strict=False` 也不能保证保留预训练输入层的行为。

后续接入应使用匹配的 Sonata 模型定义及输入预处理，核对 9 维输入的特征组成和归一化，再在 Cosmos 环境补齐匹配 Python/PyTorch/CUDA 的依赖。在实际 GPU 上通过严格权重加载、最小前向和反向后，才能确认可用于联合训练。新增 XYZ MLP、motion MLP、point2llm 和轨迹解码器仍需单独初始化，不属于本次下载权重。

本次未安装依赖、未修改模型或启动训练，也未将缺少依赖时的静态检查当作模型运行成功。

## 后续安装：三个依赖已补齐

用户随后授权安装三个依赖。安装目标为本仓库 `.venv` 指向的共享 `../WorldAct-cosmos/.venv`。使用环境内的 `uv`；没有安装 pip，也没有替换 Python、PyTorch 或 CUDA 包。

内网 `http://pypi.i.h.pjlab.org.cn/brain/dev/+simple/` 的实际索引配置为精选 stage、`bases=[]`，不含这三个包。因此保留内网为优先索引，缺失包回退到 PyPI；torch_scatter 使用 PyG 官方预编译 wheel。

新增包版本：

| 包 | 安装版本 |
|---|---|
| addict | 2.4.0 |
| spconv-cu126 | 2.3.8 |
| torch-scatter | 2.1.2+pt210cu128 |
| cumm-cu126 | 0.7.11 |
| ccimport | 0.4.4 |
| fire | 0.7.1 |
| pccm | 0.4.16 |
| pybind11 | 3.1.0 |

实际安装命令（省略下载超时和临时缓存目录设置）：

```bash
.venv/bin/uv pip install --python .venv/bin/python \
  --index http://pypi.i.h.pjlab.org.cn/brain/dev/+simple/ \
  --default-index https://pypi.org/simple \
  --allow-insecure-host pypi.i.h.pjlab.org.cn \
  --only-binary :all: addict==2.4.0 spconv-cu126==2.3.8

.venv/bin/uv pip install --python .venv/bin/python \
  --no-deps --no-index \
  --find-links 'https://data.pyg.org/whl/torch-2.10.0+cu128.html' \
  --only-binary :all: 'torch-scatter==2.1.2+pt210cu128'
```

安装后验证（Python 运行时清空 `LD_LIBRARY_PATH`）：

- addict 导入及嵌套属性读写通过。
- spconv.pytorch 导入、`SubMConv3d(3,8,kernel_size=3)` 构造通过。
- torch_scatter 导入、CPU scatter_mean 前向/梯度检查、segment_csr 均通过。
- 本地原版 PTv3 导入并以 `cls_mode=True` 成功构造，实例参数量为 38,672,640；未加载 Sonata 权重，未执行 PTv3 前向。
- cosmos_framework 和 flash_attn 在安装后仍可导入。
- PyTorch 仍为 `2.10.0+cu128`，当前进程可见 GPU 数仍为 0；尚未验证 CUDA kernel。

本次没有将依赖写入 pyproject.toml 或 lock 文件；以上命令记录当前环境增量安装方式。

## Sonata 源码接入与严格加载

官方源码固定于 commit `18c09ff8d713494f78a8213792262b910977a65d`，放在 [cosmos_framework/auxiliary/sonata](../cosmos_framework/auxiliary/sonata/VENDOR.md:1)。12 个 Python 文件与下载的官方版本逐字节一致，保留 Apache-2.0 LICENSE 和原始 README。Ruff 排除这份第三方源码，避免自动改写；新增验证脚本已通过 Ruff 检查和格式检查。

使用仓库内包导入，无需额外 `pip install sonata`：

```python
from cosmos_framework.auxiliary import sonata

model = sonata.load("../checkpoints/ptv3/sonata_small.pth")
model.eval()
```

该本地路径不会触发权重联网下载。官方 loader 按 checkpoint 配置构造模型，并执行严格 state_dict 加载。

可复现验证命令：

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m cosmos_framework.scripts.validate_sonata \
  --checkpoint ../checkpoints/ptv3/sonata_small.pth --device cpu --seed 0
```

实测结果：参数量 **38,648,992**，273 个 checkpoint 张量，missing/unexpected keys 均为空；官方预处理对合成平面输出 `[1024,9]` 特征，有限值、RGB 范围、坐标特征及原始点 inverse 长度检查通过。这里使用合成点云验证接口，不代表真实 PointFlow 数据适配完成。

H200 节点上的后续验证命令：

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m cosmos_framework.scripts.validate_sonata \
  --checkpoint ../checkpoints/ptv3/sonata_small.pth --device cuda --backward --seed 0
```

GPU 模式检查输出有限值、逐层 pooling inverse 组合、输入层梯度及全部已生成梯度的有限性。CPU 模式明确不执行 sparse/FlashAttention 前向，不把 CPU 加载成功算作 GPU 测试通过。

### 对 PointFlow 设计的具体影响

Sonata small 的输入为 `[coord(3), color(3), normal(3)]`，不能直接套用原版的 3/6 维输入建议。官方 `NormalizeColor` 将 RGB 由 0–255 除以 255；`CenterShift` 平移坐标；默认 GridSample 以 0.02 米体素采样并返回 inverse。实现见 [transform.py:1205](../cosmos_framework/auxiliary/sonata/transform.py:1205)。

训练数据需从当前 head 图像取得 RGB，从当前几何估计法向量并核对方向约定。官方预处理所用坐标与保留给绝对 XYZ 编码/轨迹标签的原始相机坐标应分开保存；UV、点 ID 也必须保留并沿采样映射同步。当前官方默认变换存在体素内随机采样，不能直接当作已经满足确定性缓存和固定点 ID 的生产数据适配器。

本次只接入 Sonata 源码和验证入口，未实现 PointFlow reader、跨模态 attention 或联合训练。
