# 离线环境配置 — h 集群(内网预下载 + uv 离线安装)

> 适用集群:**h 集群**。内网镜像为 `/etc/pip.conf` 里配置的 `mirrors.i.h.pjlab.org.cn` / `pypi.i.h.pjlab.org.cn`。
> **pt 集群**(镜像为 `pkg.pjlab.org.cn`,可直接走内网 `uv sync`)见 [setup_offline.md](./setup_offline.md)。在线安装见 [setup.md](./setup.md);磁盘占用参考 [faq.md](./faq.md)。

本集群最稳的路径是**搬 cache**:在一台有网的机器上把依赖全部拉下来,整体拷贝到离线机器,再用 `uv sync --offline` 原地装出相同环境。**不要指望配个内网 PyPI 镜像就能直接 `uv sync`** —— 原因见下节,它与集群无关,是本仓库 `uv.lock` 的性质。

## 为什么内网镜像在这条路径上帮不上忙

`uv sync` 只有在**重新解析依赖**时才会使用镜像源(`UV_INDEX_URL` / `--default-index`)。而本仓库的 `uv.lock` **无法重新生成**:

> 锁里包含一个 `python_full_version >= '3.14'` 的 resolution split,但 `multi_storage_client==0.44.0`(由 `train` extra 钉死,见 `pyproject.toml`)在 pypi.org 上没有 cp314 wheel。因此任何重解析——无论是否配了镜像、在哪个集群——都会以 unsatisfiable 失败。

由此推出两条必须记住的行为:

1. `uv sync` / `uv lock` 会重解析 → **失败**。`--locked` / `--frozen` 复用锁 → 正常。
2. 在 `--frozen` 下,`UV_INDEX_URL` / `--default-index` **对下载完全无效**:uv 只会去取锁里记录的绝对 URL(`files.pythonhosted.org`、`download.pytorch.org`、`github.com`)。

**所以 `UV_INDEX_URL=<镜像> uv sync ...` 在本仓库跑不通**(本文档旧版曾这么建议,已更正)。此外锁里还有一批内网镜像收不到的包(`torchcodec` / `torchao` / `triton`,以及 `flash-attn` / `natten` / `transformer-engine` 的 `+cu130.torch210` 构建,只在 download.pytorch.org 与 nvidia-cosmos.github.io 上),必须在有网机器上取——这正是"搬 cache"而不是"搬镜像"的原因:uv cache 按 `uv.lock` 记录的原样缓存,离线装的时候不挑来源。

如果 h 集群上也能连到内网镜像并且想要锁级精确,**可以照搬 pt 集群的做法**(锁 URL 重写 + `--frozen`,脚本在 [`tools/rewrite_lock_to_mirror.py`](../tools/rewrite_lock_to_mirror.py)),把 `--mirror-base` 换成本集群的镜像地址即可;详见 [setup_offline.md 第四步](./setup_offline.md#第四步走内网安装推荐)。

### 本集群的内网镜像配置

供 `uv pip install` 路径或其它仓库使用(对 `uv sync` 无效,原因见上):

```bash
# 对应 /etc/pip.conf:
#   index-url       = http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/
#   extra-index-url = http://pypi.i.h.pjlab.org.cn/brain/dev/+simple
#   trusted-host    = mirrors.i.h.pjlab.org.cn  pypi.i.h.pjlab.org.cn
export UV_INDEX_URL=http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/
export UV_EXTRA_INDEX_URL=http://pypi.i.h.pjlab.org.cn/brain/dev/+simple
export UV_INSECURE_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"   # http 源需要
```

第一个源是 PyPI 代理,普通依赖都能走;第二个是所内自研包的源。注意这两个都是 **PyPI 范畴**的源,收不到上面点名的那几个 `+cu130` 构建。

## 前提

- 两台机器**平台一致**:Linux x86_64、glibc ≥ 2.35(torch/natten 等 wheel 是平台相关的,跨架构搬运无效)。
- 仓库 pin 了 Python 3.13(`.python-version`)。离线机上没有 3.13 也没关系,uv 管理的 Python 可以一起搬。
- 全程使用同一份 `uv.lock`(随仓库走),保证两边解析结果一致。
- 系统级依赖 `ffmpeg` 必须单独装(见第三步)。`torchcodec` 会 `dlopen` 系统的 `libavutil.so.56`,缺了就是 `import torchcodec` 直接失败,而这不是 pip 能补的;Ubuntu 22.04 自带的 `ffmpeg 4.4.2` 正好提供该 soname。

## 第一步:有网机器(内网/办公机)预下载

```bash
# 1. 拿代码(私有库,这台机器需要有权限的凭据)
git clone https://github.com/liuhangxu-robin/WorldAct.git
cd WorldAct && git checkout <你的分支>

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

第 4 步完成后,`.uv-cache-portable` 里就有了全部 wheel——**包括 PyPI 上没有的那几个自定义源包**(`flash-attn-3-nv`、`natten`、`torchcodec`、`transformer-engine` 的 `+cu130.torch210` 构建,来自 download.pytorch.org 和 nvidia-cosmos.github.io,见 `pyproject.toml` 的 `cu130` 组)。

> 有网机器在内网时,可以先 `export UV_INDEX_URL=...`(见上节)让大部分包走镜像加速;但如前面所述,`uv sync` 会因此触发重解析并失败,所以**别在 `uv sync` 上用它**。要么老实走公网,要么改用 pt 文档里的锁重写方案。

## 第二步:拷贝到离线机器

| 内容 | 源路径 | 目标路径 | 大小量级 |
| ---- | ------ | -------- | -------- |
| 仓库(含 `uv.lock`、`.git`) | `WorldAct/` | 任意,建议共享盘 | <1 GiB |
| uv cache | `$UV_CACHE_DIR`(上文的 `.uv-cache-portable`) | 任意,记为 `$UV_CACHE_DIR` | ~20 GiB |
| uv 二进制 | `~/.local/bin/uv`(及 `uvx`) | `~/.local/bin/` | ~50 MB |
| uv 管理的 Python | `~/.local/share/uv/python/` | 同路径 | ~200 MB |
| 系统 ffmpeg(离线机没装的话) | 内网 apt 源 | — | ~100 MB |

拷贝建议 `rsync -a`(保留符号链接;uv cache 里的 hardlink 断掉无影响,只是多占空间)。

不在 git 里、但必须单独拷的运行时数据(pointflow 训练):

- DCP 基座 checkpoint(如 `models/cosmos3-edge-droid-dcp`)和 Wan2.2 VAE 权重(`vae/Wan2.2_VAE.pth`);
- sonata 预训练权重(geometry encoder 用,见训练配置里的 checkpoint 路径);
- 原始数据(如 `raw_data/singlerighthand_sandwich_100`);
- VAE window latent 缓存(如 `datasets/singlerighthand-sandwich-100-cosmos-cache`,重算一遍要按小时计,务必搬走);
- (可选)训练输出 `runs/cosmos/pointflow_*`,resume 用。

## 第三步:离线机器安装

```bash
cd WorldAct && git checkout <你的分支>
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=<拷贝过来的 cache 目录>

# 系统依赖:torchcodec 需要它
apt-get install -y --no-install-recommends ffmpeg

# --offline 只用 cache 不发网络请求;--locked 严格按 uv.lock
uv sync --all-extras --group=cu130-train --offline --locked

source .venv/bin/activate && export LD_LIBRARY_PATH=''
```

`uv sync` 会自动找到第二步拷来的 Python 3.13;找不到时用 `uv venv --python <python路径>` 显式指定。

## 验证

```bash
export LD_LIBRARY_PATH=''
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 环境是否与锁完全一致(应输出 "Would make no changes")
uv sync --all-extras --group=cu130-train --frozen --dry-run

# 注意力后端的实测(不是只看 import)
.venv/bin/python -c "
import torch
from natten import na2d
from flash_attn import flash_attn_func
q = torch.randn(1,8,16,16,64, device='cuda', dtype=torch.float16)
print('natten  ', na2d(q,q,q,kernel_size=7).shape)
print('flash   ', flash_attn_func(q,q,q).shape)
"

.venv/bin/python -m pytest cosmos_framework/data/pointflow_window_test.py -o addopts="" --noconftest -q
```

> `natten` 在 `cu130` 组里钉的是 `.gb300` 构建(Blackwell 定向),但在 A800(sm_80)上实测可正常前向——`pyproject.toml` 给 Ada/Ampere 指定的后端本来就是 `flash-attn`,见 setup.md 的 "Custom torch/cuda versions"。

训练启动脚本(如 `examples/launch_pointflow_labeled29_sandwich.sh`)开头的 `Checking inputs...` 会核对 TOML、数据集、checkpoint 路径,离线机上路径变了就改脚本顶部对应的变量。

## 常见问题

- **`UV_INDEX_URL=<镜像> uv sync` 报 unsatisfiable / 下载仍走公网**:这是**预期行为**,原因见[上文](#为什么内网镜像在这条路径上帮不上忙)。前者是重解析撞上 py3.14 split,后者是 `--frozen` 下 index 参数被忽略。要走内网请用 pt 文档的锁重写方案,或改用 `uv pip install`(见下条)。
- **想不搬 cache、也不要锁级精确**:仓库文档化的 `uv pip install -r pyproject.toml --all-extras --group=cu130-train`(见 [setup.md](./setup.md) 的 "UV Pip: virtual environment")只按当前解释器解析,能绕开 py3.14 死结,并且**会**遵守 `UV_INDEX_URL`;代价是版本由 uv 现场解析,可能与锁有漂移。注意它不安装项目本体,需再补 `uv pip install -e .`。
- **报错提示要联网**:说明 cache 不完整或平台不一致(比如在有网机器上用了不同的 Python 小版本解析)。在有网机器重跑 `uv sync --all-extras --group=cu130-train`(同一 `UV_CACHE_DIR`)补齐再拷。
- **`import torchcodec` 失败,提示 `libavutil.so.56: cannot open shared object file`**:系统没装 ffmpeg,见"前提"与第三步。
- **`torch._C` import 报错**:忘了 `export LD_LIBRARY_PATH=''`,见 [setup.md → PyTorch Import Issue](./setup.md#pytorch-import-issue)。
- **`uv` 建的 venv 里没有 `pip`**:正常,uv 默认不装。用 `uv pip list --python .venv/bin/python` 或 `uv run`,别指望 `.venv/bin/pip`。

## 分支

本文档里 pointflow 相关的路径、测试文件和启动脚本对应 `WorldAct-cosmos3-edge-droid-sft-pointflow`;若你实际在别的分支上,把上面 `<你的分支>` 换成对应分支名即可——注意不同分支的 `uv.lock` 不同,第一步和第三步必须是同一个。
