# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Frontend APIs
"""

import torch
from torch._dynamo.comptime import comptime

from cosmos_framework.model.attention.cudnn.checks import cudnn_attention_check
from cosmos_framework.model.attention.flash2.checks import flash2_attention_check
from cosmos_framework.model.attention.flash3.checks import flash3_attention_check
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.natten.checks import natten_attention_check, natten_multi_dim_attention_check
from cosmos_framework.model.attention.utils import get_arch_tag
from cosmos_framework.model.attention.utils.environment import (
    filter_attention_backends,
    filter_multi_dim_attention_backends,
    is_torch_compiling,
)
from cosmos_framework.model.attention.utils.safe_ops import log
from cosmos_framework.model.attention.utils.safe_ops.functools import lru_cache
from cosmos_framework.utils import log as framework_log

BACKEND_CHECK_MAP = {
    "cudnn": cudnn_attention_check,
    "natten": natten_attention_check,
    "flash2": flash2_attention_check,
    "flash3": flash3_attention_check,
}

BACKEND_MULTI_DIM_CHECK_MAP = {
    "natten": natten_multi_dim_attention_check,
}

_LOGGED_BACKEND_SELECTIONS: set[tuple[str, int, bool, bool, bool, bool, bool]] = set()


def _emit_backend_selection_info(
    selected_backend: str,
    arch_tag: int,
    requires_grad: bool,
    is_causal: bool,
    is_varlen: bool,
    deterministic: bool,
    explicitly_requested: bool,
) -> None:
    key = (
        selected_backend,
        arch_tag,
        requires_grad,
        is_causal,
        is_varlen,
        deterministic,
        explicitly_requested,
    )
    if key in _LOGGED_BACKEND_SELECTIONS:
        return
    _LOGGED_BACKEND_SELECTIONS.add(key)
    framework_log.info(
        "Attention backend selected: "
        f"{selected_backend} (sm{arch_tag}, requires_grad={requires_grad}, "
        f"causal={is_causal}, varlen={is_varlen}, deterministic={deterministic}, "
        f"source={'explicit' if explicitly_requested else 'auto'})"
    )


def _emit_backend_selection_info_at_compile_time(ctx) -> None:
    _emit_backend_selection_info(
        selected_backend=ctx.get_local("selected_backend").as_python_constant(),
        arch_tag=ctx.get_local("arch_tag").as_python_constant(),
        requires_grad=ctx.get_local("requires_grad").as_python_constant(),
        is_causal=ctx.get_local("is_causal").as_python_constant(),
        is_varlen=ctx.get_local("is_varlen").as_python_constant(),
        deterministic=ctx.get_local("deterministic").as_python_constant(),
        explicitly_requested=ctx.get_local("explicitly_requested").as_python_constant(),
    )


def _report_backend_selection(
    selected_backend: str,
    arch_tag: int,
    requires_grad: bool,
    is_causal: bool,
    is_varlen: bool,
    deterministic: bool,
    explicitly_requested: bool,
) -> None:
    if is_torch_compiling():
        comptime(_emit_backend_selection_info_at_compile_time)
        return
    _emit_backend_selection_info(
        selected_backend=selected_backend,
        arch_tag=arch_tag,
        requires_grad=requires_grad,
        is_causal=is_causal,
        is_varlen=is_varlen,
        deterministic=deterministic,
        explicitly_requested=explicitly_requested,
    )


def is_backend_compatible(
    backend: str,
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType | None,
    is_varlen: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    """
    Input validation function a specified backend.
    Runs the common and backend-specific checks. Returns False if any checks fail, otherwise True.

    Parameters:
        backend (str): selected backend.

        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        is_causal (bool): whether or not causal masking is enabled.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        is_varlen (bool): whether or not a variable length (varlen) use case. Must be inferred
            beforehand based on arguments such as seqlens_{Q,KV} or cumulative_seqlen_{Q,KV} being
            passed.

        deterministic (bool): Deterministic backward pass required.

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is False.

    Returns:
        success (bool): whether use case is compatible with the backend.

    """
    if backend is None:
        raise ValueError("Cannot pass None backend to is_backend_compatible.")

    if backend not in BACKEND_CHECK_MAP:
        raise ValueError(f"Unrecognized backend name {backend}.")

    return BACKEND_CHECK_MAP[backend](
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=raise_error,
    )


def get_backend_list(arch_tag: int) -> list[str]:
    """
    Returns list of supported backends according to arch tag (attention.utils.get_arch_tag).
    Backends are ordered based on their known performance levels, so that the best-performing
    compatible backend is selected.

    The returned list can be filtered via environment variable.
    See `filter_attention_backends` for details.

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        backend_list (list[str]): a list of backend names (string). Empty if device is not supported.

    """

    if arch_tag < 75:
        log.debug(f"Minimum architecture supported for Attention is 75, got {arch_tag=}.")
        return []

    default_backends = []
    if arch_tag == 90:
        default_backends = [
            "flash3",
            "cudnn",
            "natten",
            "flash2",
        ]
    elif arch_tag in [100, 103]:
        default_backends = [
            "cudnn",
            "natten",
            "flash2",
        ]
    elif arch_tag in [110, 120, 121]:
        default_backends = [
            "cudnn",
            "flash2",
            "natten",
        ]
    elif arch_tag >= 80:
        default_backends = [
            "flash2",
            "cudnn",
            "natten",
        ]
    else:
        default_backends = ["natten"]

    # Apply environment variable filtering
    return filter_attention_backends(default_backends)


@lru_cache
def choose_backend(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType | None,
    is_varlen: bool,
    deterministic: bool = False,
    backend: str | None = None,
    raise_error: bool = True,
) -> str | None:
    """
    Selects a compatible backend, unless one is already selected, which runs its corresponding
    checks.

    Parameters:
        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        is_causal (bool): whether or not causal masking is enabled.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        is_varlen (bool): whether or not a variable length (varlen) use case. Must be inferred
            beforehand based on arguments such as seqlens_{Q,KV} or cumulative_seqlen_{Q,KV} being
            passed.

        deterministic (bool): Deterministic backward pass required.

        backend (str | None): selected backend, if any.

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is **True**.

    Returns:
        backend (str | None): selected backend, or None if no backends are compatible.

    """
    explicitly_requested = backend is not None
    if explicitly_requested:
        if is_backend_compatible(
            backend=backend,
            query_shape=query_shape,
            key_shape=key_shape,
            value_shape=value_shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            is_causal=is_causal,
            causal_type=causal_type,
            is_varlen=is_varlen,
            deterministic=deterministic,
            raise_error=raise_error,
        ):
            selected_backend = backend
            arch_tag = get_arch_tag(device)
            _report_backend_selection(
                selected_backend=selected_backend,
                arch_tag=arch_tag,
                requires_grad=requires_grad,
                is_causal=is_causal,
                is_varlen=is_varlen,
                deterministic=deterministic,
                explicitly_requested=explicitly_requested,
            )
            return selected_backend
        return None

    arch_tag = get_arch_tag(device)
    backend_list = get_backend_list(arch_tag)
    for candidate_backend in backend_list:
        if is_backend_compatible(
            backend=candidate_backend,
            query_shape=query_shape,
            key_shape=key_shape,
            value_shape=value_shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            is_causal=is_causal,
            causal_type=causal_type,
            is_varlen=is_varlen,
            deterministic=deterministic,
            raise_error=False,
        ):
            selected_backend = candidate_backend
            _report_backend_selection(
                selected_backend=selected_backend,
                arch_tag=arch_tag,
                requires_grad=requires_grad,
                is_causal=is_causal,
                is_varlen=is_varlen,
                deterministic=deterministic,
                explicitly_requested=explicitly_requested,
            )
            return selected_backend

    if not raise_error:
        return None

    raise ValueError(
        "Could not find a compatible Attention backend for this use case / device. "
        "Try running with debug logs to find out why."
    )


def is_multi_dim_backend_compatible(
    backend: str,
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    """
    Input validation function a specified multi-dimensional backend.
    Runs the common and backend-specific checks. Returns False if any checks fail, otherwise True.

    Parameters:
        backend (str): selected backend.

        query_shape (torch.Size): Shape of 4-D, 5-D, or 6-D query tensor (`[batch, *token_layout_shape, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D, 5-D, or 6-D key tensor (`[batch, *token_layout_shape, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D, 5-D, or 6-D value tensor (`[batch, *token_layout_shape, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        deterministic (bool): Deterministic backward pass required.

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is False.

    Returns:
        success (bool): whether use case is compatible with the backend.

    """
    if backend is None:
        raise ValueError("Cannot pass None backend to is_backend_compatible.")

    if backend not in BACKEND_MULTI_DIM_CHECK_MAP:
        raise ValueError(f"Unrecognized backend name {backend}.")

    return BACKEND_MULTI_DIM_CHECK_MAP[backend](
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        deterministic=deterministic,
        raise_error=raise_error,
    )


def get_multi_dim_backend_list(arch_tag: int) -> list[str]:
    """
    Returns list of supported multi-dimensional backends according to arch tag (attention.utils.get_arch_tag).
    Backends are ordered based on their known performance levels, so that the best-performing
    compatible backend is selected.

    The returned list can be filtered via environment variable.
    See `filter_multi_dim_attention_backends` for details.

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        backend_list (list[str]): a list of backend names (string). Empty if device is not supported.

    """

    if arch_tag < 75:
        log.debug(f"Minimum architecture supported for Multi-Dimensional Attention is 75, got {arch_tag=}.")
        return []

    # NATTEN is the only supported backend for now
    default_backends = ["natten"]

    # Apply environment variable filtering
    return filter_multi_dim_attention_backends(default_backends)


@lru_cache
def choose_multi_dim_backend(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    deterministic: bool = False,
    backend: str | None = None,
) -> str:
    """
    Selects a compatible multi-dimensional backend, unless one is already selected, which runs its
    corresponding checks.

    Parameters:
        query_shape (torch.Size): Shape of 4-D, 5-D, or 6-D query tensor (`[batch, *token_layout_shape, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D, 5-D, or 6-D key tensor (`[batch, *token_layout_shape, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D, 5-D, or 6-D value tensor (`[batch, *token_layout_shape, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        deterministic (bool): Deterministic backward pass required.

        backend (str | None): selected backend, if any.

    Returns:
        backend (str): selected backend.

    """
    if backend is not None:
        assert is_multi_dim_backend_compatible(
            backend=backend,
            query_shape=query_shape,
            key_shape=key_shape,
            value_shape=value_shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            deterministic=deterministic,
            raise_error=True,
        )
        return backend

    arch_tag = get_arch_tag(device)
    backend_list = get_multi_dim_backend_list(arch_tag)
    for backend in backend_list:
        if is_multi_dim_backend_compatible(
            backend=backend,
            query_shape=query_shape,
            key_shape=key_shape,
            value_shape=value_shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            deterministic=deterministic,
            raise_error=False,
        ):
            return backend

    raise ValueError(
        "Could not find a compatible Multi-Dimensional Attention backend for this use case / device. "
        "Try running with debug logs to find out why."
    )
