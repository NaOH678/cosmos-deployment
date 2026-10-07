# 离线环境配置 — pt 集群(内网镜像 + uv 安装)

> 适用集群:**pt 集群**(内网镜像 `pkg.pjlab.org.cn`,可直接走内网 `uv sync`)。**h 集群**见 [setup_offline_h.md](./setup_offline_h.md)。
> 在线安装见 [setup.md](./setup.md);磁盘占用参考 [faq.md](./faq.md)。

本集群下有两种可行路径:

- **直接走内网装**(推荐,见[第四步](#第四步走内网安装推荐)):把锁 URL 重写到内网镜像再 `uv sync --frozen`,版本与锁逐字一致。
- **搬 cache**(第一~三步):通用做法,在有网机器上拉满 cache 再整体拷过来。没有内网、或要搬到完全离线的机器时走这条。

## 前提

- 两台机器**平台一致**:Linux x86_64、glibc ≥ 2.35(torch/natten 等 wheel 是平台相关的,跨架构搬运无效)。
- 仓库 pin 了 Python 3.13(`.python-version`)。离线机上没有 3.13 也没关系,uv 管理的 Python 可以一起搬。
- 全程使用同一份 `uv.lock`(随仓库走),保证两边解析结果一致。
- 系统级依赖 `ffmpeg` 必须单独装(见[第三步](#第三步离线机器安装))。`torchcodec` 会 `dlopen` 系统的 `libavutil.so.56`,缺了就是 `import torchcodec` 直接失败,而这不是 pip 能补的:Ubuntu 22.04 自带的 `ffmpeg 4.4.2` 正好提供该 soname。

## 关于内网 PyPI 镜像的一个重要限制

**先读这节,它决定了后面选哪条安装路径。**

`uv sync` 只有在**重新解析依赖**时才会使用镜像源(`UV_INDEX_URL` / `--default-index`)。而本仓库的 `uv.lock` **无法重新生成**:

> 锁里包含一个 `python_full_version >= '3.14'` 的 resolution split,但 `multi_storage_client==0.44.0`(由 `train` extra 钉死,见 `pyproject.toml`)在 pypi.org 上没有 cp314 wheel。因此任何重解析——无论是否配了镜像——都会以 unsatisfiable 失败。

由此推出两条必须记住的行为:

1. `uv sync` / `uv lock` 会重解析 → **失败**。`--locked` / `--frozen` 复用锁 → 正常。
2. 在 `--frozen` 下,`UV_INDEX_URL` / `--default-index` **对下载完全无效**:uv 只会去取锁里记录的绝对 URL(`files.pythonhosted.org`、`download.pytorch.org`、`github.com`)。**所以 `UV_INDEX_URL=<镜像> uv sync ...` 在本仓库是跑不通的**(曾写在本文档旧版的"常见问题"里,已更正)。

想要"用内网镜像装、同时保持锁级精确",只有[第四步](#第四步走内网安装推荐)那条路。

### 内网镜像的覆盖边界

集群内可达的镜像不是 `pip.conf` 里的那台。实测结论(2026-09):

| 主机 | 状态 |
| ---- | ---- |
| `pkg.pjlab.org.cn`(10.140.3.250,Nexus) | ✅ 可用。`no_proxy` 已覆盖 `.pjlab.org.cn` 与 `10.0.0.0/8`,直连不走企业代理。 |
| `mirrors.i.h.pjlab.org.cn` / `pypi.i.h.pjlab.org.cn`(10.102.254.2,`/etc/pip.conf` 里配的) | ❌ 在这些容器里不可用:80 端口拒绝连接,443 是要 Basic 认证的 Envoy 网关(`407 proxy-authenticate: Basic`)。`/etc/pip.conf` 在容器内也不存在。那套配置属于另一个网络区域。 |

`pkg.pjlab.org.cn` 上有用的仓库:

| 仓库 | 上游 | 覆盖 |
| ---- | ---- | ---- |
| `official-pypi-proxy/simple/` | pypi.org | 全部 PyPI 包,**有锁里所有精确版本**(优先用它) |
| `pypi-proxy/simple/` | mirrors.aliyun.com/pypi | 同上但滞后(缺 `multi-storage-client==0.44.0` 之类) |
| `pypi-pytorch/simple/` | aliyun `pytorch-wheels` | 只有 `torch` / `torchvision` 的 `+cu130`;**没有 `torchcodec` / `torchao` / `triton`** |
| `apt-jammy-proxy/ubuntu` | 清华 TUNA | apt 源,`sources.list` 已指向它 |

覆盖边界很重要:**内网能覆盖约 98% 的制品,但覆盖不了**
`torchcodec` / `torchao` / `triton`(download.pytorch.org 上与 torch 同源的辅包),以及
`flash-attn` / `flash-attn-3-nv` / `natten` / `transformer-engine` 的 `+cu130.torch210` 构建(只在 `nvidia-cosmos.github.io` / github releases 上)。
这些必须回源,所以**完全离线或纯内网镜像都装不出完整环境**。

> **测速时注意冷/热缓存**:该 Nexus 对未命中的包是 pull-through 回源,首次请求只有 ~2 MB/s,容易误判成"内网很慢";缓存命中后是 **~110–176 MB/s**。判断速度请对同一文件测第二次。

## 第一步:有网机器(内网/办公机)预下载

```bash
# 1. 拿代码(私有库,这台机器需要有权限的凭据)
git clone https://github.com/liuhangxu-robin/WorldAct.git
cd WorldAct && git checkout <你的分支>          # 见文末"分支"

# 2. 装 uv(静态二进制,之后一起拷走)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
#    没有外网时,可以直接从内网镜像取 wheel 解出二进制(uv wheel 里就是编译好的二进制):
#      W=$(curl -s http://pkg.pjlab.org.cn/repository/pypi-proxy/simple/uv/ \
#            | grep -oE 'href="[^"]*uv-[0-9.]+-py3-none-manylinux_2_17_x86_64[^"]*\.whl' \
#            | tail -1 | sed 's/href="//;s/"$//;s|#.*||;s|^\.\./\.\./|http://pkg.pjlab.org.cn/repository/pypi-proxy/|')
#      curl -sL "$W" -o /tmp/uv.whl && unzip -qo /tmp/uv.whl -d /tmp/uvx
#      install -m755 /tmp/uvx/uv-*.data/scripts/uv /tmp/uvx/uv-*.data/scripts/uvx ~/.local/bin/
#    注意 uv>=0.11.3(pyproject 里的 required-version)

# 3. 关键:把 uv cache 指到一个便携目录,所有 wheel 都落在里面
export UV_CACHE_DIR=$PWD/.uv-cache-portable

# 4. 一次性拉满依赖(CUDA 13.0 训练组;CUDA 12.8 用 --group=cu128-train)
uv sync --all-extras --group=cu130-train

# 5. 拉一份 uv 管理的 Python 3.13(默认在 ~/.local/share/uv/python)
uv python install 3.13
```

第 4 步完成后,`.uv-cache-portable` 里就有了全部 wheel——**包括 PyPI 上没有的那几个自定义源包**(`flash-attn-3-nv`、`natten`、`torchcodec`、`transformer-engine` 的 `+cu130.torch210` 构建,来自 download.pytorch.org 和 nvidia-cosmos.github.io,见 `pyproject.toml` 的 `cu130` 组)。这正是推荐"搬 cache"而不是"搬 pip 镜像"的原因:镜像站收不到这几个包,而 uv cache 按 `uv.lock` 记录的原样缓存,离线装的时候不挑来源。

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

# 系统依赖:torchcodec 需要它,并且 apt 源已经指向内网
apt-get install -y --no-install-recommends ffmpeg

# --offline 只用 cache 不发网络请求;--locked 严格按 uv.lock
uv sync --all-extras --group=cu130-train --offline --locked

source .venv/bin/activate && export LD_LIBRARY_PATH=''
```

`uv sync` 会自动找到第二步拷来的 Python 3.13;找不到时用 `uv venv --python <python路径>` 显式指定。

## 第四步:走内网安装(推荐)

> 做法是**锁 URL 重写 + `uv sync --frozen`**。前三步是**回退路径**(搬 cache),没有内网时才需要;本集群有内网,通常直接用这一步。

在没有 cache、但能连内网的机器上想直接装(而不是先搬 20 GiB cache),用这条路。原理:**uv 不动锁就用锁里的 URL,那就把锁里的 URL 换掉**。

改写的只有 URL,**版本号和 `sha256` 一个字节都不动**——所以 uv 下载后仍按原哈希校验,映射错了会直接报 `hash mismatch` 而不是静默装错。这也是可以放心改锁的原因。

```bash
# 1. 改写(脚本随仓库走;--verify-sample 会回下几个制品核对 sha256)
python3 tools/rewrite_lock_to_mirror.py uv.lock --verify-sample 8

uv sync --all-extras --group=cu130-train --frozen   # 2. 安装

git checkout -- uv.lock                      # 3. 还原,保持仓库干净
```

第 2 步的脚本把锁里每个制品 URL 换到内网对应路径。映射规则(两侧文件名一致,这是能对上的关键):

| 锁里(公网) | 内网 |
| ---- | ---- |
| `https://files.pythonhosted.org/packages/<a>/<b>/<sha>/<file>` | `http://pkg.pjlab.org.cn/repository/official-pypi-proxy/packages/<pkg>/<ver>/<file>` |
| `https://pypi.org/...` | 同上 |
| `https://download.pytorch.org/whl/cu130/<file>` | `http://pkg.pjlab.org.cn/repository/pypi-pytorch/packages/<pkg>/<ver>/<file>` |
| `nvidia-cosmos.github.io` / `github.com` | 内网无镜像,**保持原样**(走代理回源) |

`<pkg>` / `<ver>` 不建议硬拼,Nexus 的路径布局和 PyPI 不同;**正确做法是取 `http://pkg.pjlab.org.cn/repository/<repo>/simple/<pkg>/` 的索引页,按文件名匹配出真实 URL**。脚本实现在 [`tools/rewrite_lock_to_mirror.py`](../tools/rewrite_lock_to_mirror.py),可直接复用(支持 `--mirror-base` / `--pypi-repo` / `--pytorch-repo` 换镜像)。它会顺手做两件安全的事:改写前只认带扩展名的制品 URL(不会误伤 `source = { registry = ... }` 这类标识),`--verify-sample N` 改写后回下 N 个制品核对 sha256。

2026-09 在本仓库实测的效果:6783 个制品里 **6665 个走内网**(~110–176 MB/s),**118 个回源**(torchcodec 39 / torch 36 / torchvision 24 / triton 16 / torchao 3),另加 cosmos 系(github)回源。一次 `uv sync` 跑完,结果与锁逐字一致。

> 改写完可以用这两条确认"只动了 URL":`git diff --numstat uv.lock` 两侧行数相同;把两侧文件的 URL 字符串替换成占位符后应当**逐字节相同**。实测 6665 处替换,除 URL 外零差异。

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

- **`UV_INDEX_URL=<镜像> uv sync` 报 unsatisfiable / 下载仍走公网**:这是**预期行为**,原因见[上文](#关于内网-pypi-镜像的一个重要限制)。前者是重解析撞上 py3.14 split,后者是 `--frozen` 下 index 参数被忽略。要走内网请用[第四步](#第四步走内网安装推荐),或直接在 `uv pip install` 路径上用镜像(见下条)。
- **想不搬 cache、也不要锁级精确**:仓库文档化的 `uv pip install -r pyproject.toml --all-extras --group=cu130-train`(见 [setup.md](./setup.md) 的 "UV Pip: virtual environment")只按当前解释器解析,能绕开 py3.14 死结,并且**会**遵守 `--default-index`;代价是版本由 uv 现场解析,可能与锁有漂移。注意它不安装项目本体,需再补 `uv pip install -e .`。
- **报错提示要联网**:说明 cache 不完整或平台不一致(比如在有网机器上用了不同的 Python 小版本解析)。在有网机器重跑 `uv sync --all-extras --group=cu130-train`(同一 `UV_CACHE_DIR`)补齐再拷。
- **`import torchcodec` 失败,提示 `libavutil.so.56: cannot open shared object file`**:系统没装 ffmpeg,见"前提"与第三步。
- **`torch._C` import 报错**:忘了 `export LD_LIBRARY_PATH=''`,见 [setup.md → PyTorch Import Issue](./setup.md#pytorch-import-issue)。
- **`uv` 建的 venv 里没有 `pip`**:正常,uv 默认不装。用 `uv pip list --python .venv/bin/python` 或 `uv run`,别指望 `.venv/bin/pip`。

## 分支

本文档里 pointflow 相关的路径、测试文件和启动脚本对应 `WorldAct-cosmos3-edge-droid-sft-pointflow`;若你实际在别的分支上(例如 `WorldAct-cosmos3-edge-droid-sft`),把上面两处 `<你的分支>` 换成对应分支名即可——注意不同分支的 `uv.lock` 不同,第一步和第三步必须是同一个。
