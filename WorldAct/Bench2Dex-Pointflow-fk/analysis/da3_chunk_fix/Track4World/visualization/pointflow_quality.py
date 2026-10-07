"""STUB — 本机重建的占位模块，不是原文件。

原 `pointflow_quality.py` 只存在于 GPFS 的
`shichaojian/Track4World_portable/Track4World/visualization/`（自定义模块，未进 git，
官方 TencentARC/Track4World 仓库里也没有）。本机没有挂载 GPFS，无法拿到原文件。

9.24 v1 流水线（run_efep_export.py → sam2_segment.py → efep_seg_label.py →
drop_detached.py）只通过 `batch_repair` / `efep_label` 的模块级 import 间接引用本模块；
实际被调用的只有 batch_repair 里的 DATASETS / initial_regions /
rigid_landmark_filter（纯 numpy/cv2），下面的 6 个函数只在 legacy 3d_ff 修复
路径里用到，本流水线不会执行。若真被调用会立即报错，不会静默产生错误结果。
"""


def _missing(name):
    raise NotImplementedError(
        f"pointflow_quality.{name} 是 stub：原文件在 GPFS Track4World_portable 里，"
        "本流水线不应调用到它。若看到此错误，说明执行路径进入了 legacy 修复代码，"
        "请先从 GPFS 拷贝原始 pointflow_quality.py。")


class QualityConfig:  # noqa: D101
    def __init__(self, *a, **k):
        _missing('QualityConfig')


def observation_mask(*a, **k):
    _missing('observation_mask')


def reference_neighbor_graph(*a, **k):
    _missing('reference_neighbor_graph')


def selected_spatial_support(*a, **k):
    _missing('selected_spatial_support')


def valid_links(*a, **k):
    _missing('valid_links')


def isolated_spike_mask(*a, **k):
    _missing('isolated_spike_mask')
