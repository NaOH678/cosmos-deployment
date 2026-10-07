# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from unittest.mock import patch

import pytest
import torch

from cosmos_framework.model.attention import backends


@pytest.fixture(autouse=True)
def clear_logged_backend_selections():
    backends._LOGGED_BACKEND_SELECTIONS.clear()
    yield
    backends._LOGGED_BACKEND_SELECTIONS.clear()


@pytest.mark.L0
def test_report_backend_selection_logs_each_configuration_once():
    with patch.object(backends.framework_log, "info") as info:
        for _ in range(2):
            backends._report_backend_selection(
                selected_backend="flash3",
                arch_tag=90,
                requires_grad=True,
                is_causal=False,
                is_varlen=True,
                deterministic=False,
                explicitly_requested=False,
            )

    info.assert_called_once()
    assert "Attention backend selected: flash3" in info.call_args.args[0]
    assert "source=auto" in info.call_args.args[0]


@pytest.mark.L0
def test_report_backend_selection_is_fullgraph_compile_safe():
    def report_from_compiled_region(x: torch.Tensor) -> torch.Tensor:
        backends._report_backend_selection(
            selected_backend="flash3",
            arch_tag=90,
            requires_grad=True,
            is_causal=False,
            is_varlen=True,
            deterministic=False,
            explicitly_requested=False,
        )
        return x + 1

    compiled = torch.compile(report_from_compiled_region, backend="eager", fullgraph=True)
    with patch.object(backends.framework_log, "info") as info:
        torch.testing.assert_close(compiled(torch.zeros(2)), torch.ones(2))
        torch.testing.assert_close(compiled(torch.zeros(3)), torch.ones(3))

    info.assert_called_once()
    assert "Attention backend selected: flash3" in info.call_args.args[0]
