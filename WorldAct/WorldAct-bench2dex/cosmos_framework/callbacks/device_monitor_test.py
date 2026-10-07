# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

from cosmos_framework.callbacks import device_monitor


def test_device_monitor_manages_nvml_lifecycle(monkeypatch, tmp_path):
    calls = []
    handle = object()
    monitor = device_monitor.DeviceMonitor()
    monitor.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(device_monitor.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(device_monitor.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(device_monitor.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(device_monitor.pynvml, "nvmlInit", lambda: calls.append("init"))
    monkeypatch.setattr(
        device_monitor.pynvml,
        "nvmlDeviceGetHandleByIndex",
        lambda rank: calls.append(("handle", rank)) or handle,
    )
    monkeypatch.setattr(device_monitor.pynvml, "nvmlShutdown", lambda: calls.append("shutdown"))

    monitor.on_train_start(model=None)
    monitor.on_train_end(model=None)

    assert monitor.handle is handle
    assert calls == ["init", ("handle", 3), "shutdown"]
