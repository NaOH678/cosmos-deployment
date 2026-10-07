# 离线环境配置(内网预下载 + uv 离线安装)

> 适用场景:训练集群走公网极慢或完全不通。流程是**在一台有网的机器上把依赖全部拉下来,整体拷贝到离线机器,再用 `uv sync --offline` 原地装出相同环境**。
> 在线安装见 [setup.md](./setup.md);磁盘占用参考 [faq.md](./faq.md)。

## 前提

- 两台机器**平台一致**:Linux x86_64、glibc ≥ 2.35(torch/natten 等 wheel 是平台相关的,跨架构搬运无效)。
- 仓库 pin 了 Python 3.13(`.python-version`)。离线机上没有 3.13 也没关系,uv 管理的 Python 可以一起搬。
- 全程使用同一份 `uv.lock`(随仓库走),保证两边解析结果一致。

## 第一步:有网机器(内网/办公机)预下载

```bash
# 1. 拿代码(私有库,这台机器需要有权限的凭据)
git clone https://github.com/liuhangxu-robin/WorldAct.git
cd WorldAct && git checkout WorldAct-cosmos3-edge-droid-sft-pointflow

# 2. 装 uv(静态二进制,之后一起拷走)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. 关键:把 uv cache 指到一个便携目录,所有 wheel 都落在里面
export UV_CACHE_DIR=$PWD/.uv-cache-portable

# 4. 一次性拉满依赖(CUDA 13.0 训练组;CUDA 12.8 用 --group=cu128-train)
uv sync --all-extras --group=cu130-train

# 5. 拉一份 uv 管理的 Python 3.13(默认在 ~/.local/share/uv/python)
uv python install 3.13
```

第 4 步完成后,`.uv-cache-portable` 里就有了全部 wheel——**包括 PyPI 上没有的那几个自定义源包**(`flash-attn-3-nv`、`natten`、`torchcodec`、`transformer-engine` 的 `+cu130.torch210` 构建,来自 download.pytorch.org 和 nvidia-cosmos.github.io,见 `pyproject.toml` 的 `cu130` 组)。这正是推荐"搬 cache"而不是"搬 pip 镜像"的原因:镜像站通常收不到这几个包,而 uv cache 按 `uv.lock` 记录的原样缓存,离线装的时候不挑来源。

## 第二步:拷贝到离线机器

| 内容 | 源路径 | 目标路径 | 大小量级 |
| ---- | ------ | -------- | -------- |
| 仓库(含 `uv.lock`、`.git`) | `WorldAct/` | 任意,建议共享盘 | <1 GiB |
| uv cache | `$UV_CACHE_DIR`(上文的 `.uv-cache-portable`) | 任意,记为 `$UV_CACHE_DIR` | ~20 GiB |
| uv 二进制 | `~/.local/bin/uv`(及 `uvx`) | `~/.local/bin/` | ~50 MB |
| uv 管理的 Python | `~/.local/share/uv/python/` | 同路径 | ~200 MB |

拷贝建议 `rsync -a`(保留符号链接;uv cache 里的 hardlink 断掉无影响,只是多占空间)。

不在 git 里、但必须单独拷的运行时数据(pointflow 训练):

- DCP 基座 checkpoint(如 `models/cosmos3-edge-droid-dcp`)和 Wan2.2 VAE 权重(`vae/Wan2.2_VAE.pth`);
- sonata 预训练权重(geometry encoder 用,见训练配置里的 checkpoint 路径);
- 原始数据(如 `raw_data/singlerighthand_sandwich_100`);
- VAE window latent 缓存(如 `datasets/singlerighthand-sandwich-100-cosmos-cache`,重算一遍要按小时计,务必搬走);
- (可选)训练输出 `runs/cosmos/pointflow_*`,resume 用。

## 第三步:离线机器安装

```bash
cd WorldAct && git checkout WorldAct-cosmos3-edge-droid-sft-pointflow
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=<拷贝过来的 cache 目录>

# --offline 只用 cache 不发网络请求;--locked 严格按 uv.lock
uv sync --all-extras --group=cu130-train --offline --locked

source .venv/bin/activate && export LD_LIBRARY_PATH=''
```

`uv sync` 会自动找到第二步拷来的 Python 3.13;找不到时用 `uv venv --python <python路径>` 显式指定。

## 验证

```bash
export LD_LIBRARY_PATH=''
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
.venv/bin/python -m pytest cosmos_framework/data/pointflow_window_test.py -o addopts="" --noconftest -q
```

训练启动脚本(如 `bash examples/launch_pointflow_labeled29_sandwich.sh`)开头的 `Checking inputs...` 会核对 TOML、数据集、checkpoint 路径,离线机上路径变了就改脚本顶部对应的变量。

## 常见问题

- **想直接用内网 PyPI 镜像而不是搬 cache**:可以 `UV_INDEX_URL=<镜像地址> uv sync ...`,但镜像必须同时收录 PyPI、download.pytorch.org(`+cu130` 构建)和 nvidia-cosmos.github.io(flash-attn/natten)三处来源,缺一个就会回源拉取失败。搬 cache 没有这个问题。
- **报错提示要联网**:说明 cache 不完整或平台不一致(比如在有网机器上用了不同的 Python 小版本解析)。在有网机器重跑 `uv sync --all-extras --group=cu130-train`(同一 `UV_CACHE_DIR`)补齐再拷。
- **`torch._C` import 报错**:忘了 `export LD_LIBRARY_PATH=''`,见 [setup.md → PyTorch Import Issue](./setup.md#pytorch-import-issue)。
