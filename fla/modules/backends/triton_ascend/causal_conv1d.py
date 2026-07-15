# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Causal 1D convolution kernels adapted for triton-ascend on Huawei NPU."""

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.utils import prepare_chunk_indices
from fla.utils import input_guard

STATIC_WARPS = 2
# Ascend Triton rejects grids whose product exceeds 65535 (see fla/modules/token_shift.py).
_NPU_MAX_TRITON_GRID = 65535
_ELEM_BLOCK = 2048
_FWD_DENSE_BT = 64
_FWD_DENSE_BD = 128
_FWD_DENSE_WARPS = 2
_BWD_DENSE_BT = 64
_BWD_DENSE_BD = 128
_BWD_DENSE_WARPS = 2
_BWD_REDUCE_T = 512
_TARGET_BF16_DIMS = (1024, 3072)


def _dense_backward_num_warps(x: torch.Tensor, weight: torch.Tensor) -> int:
    """Use the measured four-warp schedule only for production BF16 buckets."""
    if (
        x.shape[0] == 1
        and x.shape[1] >= 64
        and x.shape[2] in _TARGET_BF16_DIMS
        and x.dtype == torch.bfloat16
        and weight.shape == (x.shape[2], 4)
    ):
        return 4
    return _BWD_DENSE_WARPS


def _elementwise_launch_iters(numel: int):
    n_blocks = triton.cdiv(numel, _ELEM_BLOCK)
    for block_off in range(0, n_blocks, _NPU_MAX_TRITON_GRID):
        yield min(_NPU_MAX_TRITON_GRID, n_blocks - block_off), block_off * _ELEM_BLOCK


def _npu_chunk_size(T: int, BT: int) -> int:
    BT = min(max(BT, 1), 64)
    if BT not in (1, 2, 4, 8, 16, 32, 64):
        BT = triton.next_power_of_2(BT)
    # Ascend compiler requires power-of-2 BT; pad with mask when BT > T.
    if T not in (1, 2, 4, 8, 16, 32, 64):
        BT = min(triton.next_power_of_2(T), 64)
    else:
        BT = min(BT, T, 64)
    return BT


def _get_npu_max_grid() -> int:
    return _NPU_MAX_TRITON_GRID


def _iter_3d_grid_splits(B: int, NT: int, D: int, BD: int):
    """Yield complete B/NT/D-block slices whose grid products fit Ascend."""
    if min(B, NT, D, BD) <= 0:
        return

    DB = triton.cdiv(D, BD)
    limit = _get_npu_max_grid()
    b_start = 0
    while b_start < B:
        b_count = min(B - b_start, max(1, limit // (DB * NT)))
        remaining = max(1, limit // b_count)
        nt_start = 0
        while nt_start < NT:
            nt_count = min(NT - nt_start, max(1, remaining // DB))
            remaining_nt = max(1, remaining // nt_count)
            d_start = 0
            while d_start < DB:
                d_count = min(DB - d_start, remaining_nt)
                yield b_start, b_count, nt_start, nt_count, d_start, d_count
                d_start += d_count
            nt_start += nt_count
        b_start += b_count


def _npu_tile_config(
    T: int,
    BT: int,
    D: int,
    dtype: torch.dtype,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int]:
    BT = _npu_chunk_size(T, BT)
    BD = 16
    if D >= 8192:
        BD = 8
        BT = min(BT, 8)
    elif D >= 1024:
        # BD=4 overflows Ascend UB on large-D forward; cap BT to limit NT.
        BD = 8
        BT = min(BT, 32)
    elif D >= 512:
        BD = 8
    if dtype == torch.float16 and initial_state is not None:
        BD = min(BD, 8)
    if dtype == torch.bfloat16 and T <= 16:
        BD = 8
    return BD, BT, STATIC_WARPS


def _npu_bwd_tile_config(
    T: int,
    BT: int,
    D: int,
    dtype: torch.dtype,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int]:
    BT = _npu_chunk_size(T, BT)
    BD = 16
    if initial_state is not None:
        BD = min(BD, 8)
        BT = min(BT, 32)
    if D >= 2048:
        BD = 8
        BT = min(BT, 8)
    elif D >= 1024:
        BD = 8
        BT = min(BT, 16)
    elif D >= 512:
        BD = 8
        BT = min(BT, 32)
    if dtype == torch.bfloat16 and T <= 16:
        BD = 8
        BT = 32
    return BD, BT, STATIC_WARPS


def _is_dense_single_sequence(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
) -> bool:
    """Return whether the contiguous Ascend forward kernel can preserve semantics."""
    if x.dim() != 3 or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    B, T, D = x.shape
    W = weight.shape[1]
    if W not in (2, 3, 4) or weight.shape != (D, W) or not weight.is_contiguous():
        return False
    if bias is not None and (bias.shape != (D,) or not bias.is_contiguous()):
        return False
    if residual is not None and (residual.shape != x.shape or not residual.is_contiguous()):
        return False
    if initial_state is not None and (initial_state.shape != (B, D, W) or not initial_state.is_contiguous()):
        return False
    if cu_seqlens is None:
        return True
    if cu_seqlens_cpu is None or cu_seqlens_cpu.device.type != "cpu" or cu_seqlens_cpu.numel() != 2:
        return False
    bos, eos = cu_seqlens_cpu.tolist()
    return bos == 0 and eos == T


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
    }
)
@triton.jit
def causal_conv1d_fwd_dense_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    initial_state,
    T: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    NT: tl.constexpr,
    DB: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Compute one dense token/channel tile from a grid-safe 1D task."""
    task = tl.program_id(0).to(tl.int64) + tl.cast(TASK_OFFSET, tl.int64)
    batch = task // (NT * DB)
    tile = task % (NT * DB)
    time_block = tile // DB
    channel_block = tile % DB
    t = time_block * BT + tl.arange(0, BT).to(tl.int64)
    d = channel_block * BD + tl.arange(0, BD).to(tl.int64)
    token_mask = t < T
    channel_mask = d < D
    output_mask = token_mask[:, None] & channel_mask[None, :]
    output_offset = (batch * T + t[:, None]) * D + d[None, :]

    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for tap in tl.static_range(0, W):
        source_t = t + tap - W + 1
        source_mask = token_mask[:, None] & (source_t >= 0)[:, None] & channel_mask[None, :]
        source_offset = (batch * T + source_t[:, None]) * D + d[None, :]
        source = tl.load(x + source_offset, mask=source_mask, other=0.0).to(tl.float32)
        if USE_INITIAL_STATE:
            state_index = source_t + W
            state_mask = token_mask[:, None] & (source_t < 0)[:, None] & (state_index >= 0)[:, None] & channel_mask[None, :]
            state_offset = (batch * D + d[None, :]) * W + state_index[:, None]
            source += tl.load(initial_state + state_offset, mask=state_mask, other=0.0).to(tl.float32)
        coefficient = tl.load(weight + d * W + tap, mask=channel_mask, other=0.0).to(tl.float32)
        acc += source * coefficient[None, :]

    if HAS_BIAS:
        acc += tl.load(bias + d, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    if USE_ACTIVATION:
        acc *= tl.sigmoid(acc)
    if HAS_RESIDUAL:
        acc += tl.load(residual + output_offset, mask=output_mask, other=0.0).to(tl.float32)
    tl.store(
        y + output_offset,
        tl.cast(acc, dtype=y.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=output_mask,
    )


def _launch_fwd_dense(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    activation: str | None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch dense two-dimensional tiles from grid-safe 1D slices."""
    B, T, D = x.shape
    NT = triton.cdiv(T, _FWD_DENSE_BT)
    DB = triton.cdiv(D, _FWD_DENSE_BD)
    n_tasks = B * NT * DB
    y = torch.empty_like(x, memory_format=torch.contiguous_format) if output is None else output
    for task_off in range(0, n_tasks, _NPU_MAX_TRITON_GRID):
        grid = min(_NPU_MAX_TRITON_GRID, n_tasks - task_off)
        causal_conv1d_fwd_dense_kernel[(grid,)](
            x=x,
            y=y,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            T=T,
            D=D,
            W=weight.shape[1],
            NT=NT,
            DB=DB,
            USE_ACTIVATION=activation in ("swish", "silu"),
            TASK_OFFSET=task_off,
            BT=_FWD_DENSE_BT,
            BD=_FWD_DENSE_BD,
            num_warps=_FWD_DENSE_WARPS,
            multibuffer=False,
        )
    return y


def _is_dense_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    dht: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
) -> bool:
    """Return whether the dense backward kernels cover this call exactly."""
    return (
        x.shape[0] == 1
        and dy.shape == x.shape
        and dy.is_contiguous()
        and dht is None
        and activation in (None, "silu", "swish")
        and _is_dense_single_sequence(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
    )


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
    }
)
@triton.jit
def causal_conv1d_dpre_dense_kernel(
    x,
    dy,
    weight,
    bias,
    initial_state,
    dpre,
    T: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    NT: tl.constexpr,
    DB: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    task = tl.program_id(0).to(tl.int64) + tl.cast(TASK_OFFSET, tl.int64)
    batch = task // (NT * DB)
    tile = task % (NT * DB)
    time_block = tile // DB
    channel_block = tile % DB
    t = time_block * BT + tl.arange(0, BT).to(tl.int64)
    d = channel_block * BD + tl.arange(0, BD).to(tl.int64)
    token_mask = t < T
    channel_mask = d < D
    mask = token_mask[:, None] & channel_mask[None, :]
    output_offset = (batch * T + t[:, None]) * D + d[None, :]
    gradient = tl.load(dy + output_offset, mask=mask, other=0.0).to(tl.float32)

    if USE_ACTIVATION:
        pre = tl.zeros((BT, BD), dtype=tl.float32)
        for tap in tl.static_range(0, W):
            source_t = t + tap - W + 1
            source_mask = token_mask[:, None] & (source_t >= 0)[:, None] & channel_mask[None, :]
            source_offset = (batch * T + source_t[:, None]) * D + d[None, :]
            source = tl.load(x + source_offset, mask=source_mask, other=0.0).to(tl.float32)
            if USE_INITIAL_STATE:
                state_index = source_t + W
                state_mask = (
                    token_mask[:, None] & (source_t < 0)[:, None] & (state_index >= 0)[:, None] & channel_mask[None, :]
                )
                state_offset = (batch * D + d[None, :]) * W + state_index[:, None]
                source += tl.load(initial_state + state_offset, mask=state_mask, other=0.0).to(tl.float32)
            coefficient = tl.load(weight + d * W + tap, mask=channel_mask, other=0.0).to(tl.float32)
            pre += source * coefficient[None, :]
        if HAS_BIAS:
            pre += tl.load(bias + d, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
        sigmoid = tl.sigmoid(pre)
        gradient *= sigmoid * (1.0 + pre * (1.0 - sigmoid))

    tl.store(dpre + output_offset, gradient, mask=mask)


@triton.jit
def causal_conv1d_dx_dense_kernel(
    dpre,
    weight,
    dx,
    T: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    NT: tl.constexpr,
    DB: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    task = tl.program_id(0).to(tl.int64) + tl.cast(TASK_OFFSET, tl.int64)
    batch = task // (NT * DB)
    tile = task % (NT * DB)
    time_block = tile // DB
    channel_block = tile % DB
    t = time_block * BT + tl.arange(0, BT).to(tl.int64)
    d = channel_block * BD + tl.arange(0, BD).to(tl.int64)
    token_mask = t < T
    channel_mask = d < D
    output_mask = token_mask[:, None] & channel_mask[None, :]
    dx_value = tl.zeros((BT, BD), dtype=tl.float32)

    for delta in tl.static_range(0, W):
        output_t = t + delta
        dpre_mask = token_mask[:, None] & (output_t < T)[:, None] & channel_mask[None, :]
        dpre_offset = (batch * T + output_t[:, None]) * D + d[None, :]
        dpre_value = tl.load(dpre + dpre_offset, mask=dpre_mask, other=0.0).to(tl.float32)
        coefficient = tl.load(weight + d * W + W - delta - 1, mask=channel_mask, other=0.0).to(tl.float32)
        dx_value += dpre_value * coefficient[None, :]

    output_offset = (batch * T + t[:, None]) * D + d[None, :]
    tl.store(
        dx + output_offset,
        tl.cast(dx_value, dtype=dx.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=output_mask,
    )


@triton.jit
def causal_conv1d_dh0_dense_kernel(
    dpre,
    weight,
    dh0,
    T: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
):
    channel_block = tl.program_id(0).to(tl.int64)
    d = channel_block * BD + tl.arange(0, BD).to(tl.int64)
    channel_mask = d < D
    for state_index in tl.static_range(0, W):
        grad_state = tl.zeros((BD,), dtype=tl.float32)
        for t in tl.static_range(0, W - 1):
            if t < T and t < state_index:
                coefficient = tl.load(
                    weight + d * W + state_index - t - 1,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                gradient = tl.load(dpre + t * D + d, mask=channel_mask, other=0.0).to(tl.float32)
                grad_state += gradient * coefficient
        tl.store(
            dh0 + d * W + state_index,
            tl.cast(grad_state, dtype=dh0.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=channel_mask,
        )


def _reduce_dwdb_dense(
    x: torch.Tensor,
    dpre: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use fused vendor reductions for the dense depthwise parameter gradients."""
    B, T, D = x.shape
    W = weight.shape[1]
    gradients = []
    for tap in range(W):
        shift = W - 1 - tap
        prefix = initial_state[:, :, tap + 1 :].transpose(1, 2) if initial_state is not None else x.new_zeros(B, shift, D)
        gradient = torch.zeros(D, dtype=torch.float32, device=x.device)
        for start in range(0, T, _BWD_REDUCE_T):
            end = min(T, start + _BWD_REDUCE_T)
            if start < shift:
                boundary = prefix[:, start : min(end, shift)]
                source = torch.cat((boundary, x[:, : end - shift]), dim=1) if end > shift else boundary
            else:
                source = x[:, start - shift : end - shift]
            gradient.add_((dpre[:, start:end] * source).sum(dim=(0, 1)))
        gradients.append(gradient)
    dw = torch.stack(gradients, dim=1).to(weight)
    db = dpre.sum(dim=(0, 1)).to(bias) if bias is not None else None
    return dw, db


def _launch_bwd_dense(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    activation: str | None,
    poison: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Launch the dense backward pipeline without partial workspaces."""
    _, T, D = x.shape
    W = weight.shape[1]
    NT = triton.cdiv(T, _BWD_DENSE_BT)
    DB = triton.cdiv(D, _BWD_DENSE_BD)
    num_warps = _dense_backward_num_warps(x, weight)
    n_tasks = NT * DB
    fill = float("nan") if poison else None
    dpre = (
        torch.full_like(x, fill, dtype=torch.float32, memory_format=torch.contiguous_format)
        if poison
        else torch.empty_like(x, dtype=torch.float32, memory_format=torch.contiguous_format)
    )

    common = dict(
        T=T,
        D=D,
        W=W,
        NT=NT,
        DB=DB,
        BT=_BWD_DENSE_BT,
        BD=_BWD_DENSE_BD,
        num_warps=num_warps,
        multibuffer=False,
    )
    for task_off in range(0, n_tasks, _NPU_MAX_TRITON_GRID):
        grid = min(_NPU_MAX_TRITON_GRID, n_tasks - task_off)
        causal_conv1d_dpre_dense_kernel[(grid,)](
            x=x,
            dy=dy,
            weight=weight,
            bias=bias,
            initial_state=initial_state,
            dpre=dpre,
            USE_ACTIVATION=activation in ("silu", "swish"),
            TASK_OFFSET=task_off,
            **common,
        )

    dx = torch.full_like(x, fill) if poison else torch.empty_like(x)
    for task_off in range(0, n_tasks, _NPU_MAX_TRITON_GRID):
        grid = min(_NPU_MAX_TRITON_GRID, n_tasks - task_off)
        causal_conv1d_dx_dense_kernel[(grid,)](
            dpre=dpre,
            weight=weight,
            dx=dx,
            TASK_OFFSET=task_off,
            **common,
        )

    dw_value, db_value = _reduce_dwdb_dense(x, dpre, weight, bias, initial_state)
    if poison:
        dw = torch.full_like(dw_value, fill)
        dw.copy_(dw_value)
        db = torch.full_like(db_value, fill) if db_value is not None else None
        if db is not None:
            db.copy_(db_value)
    else:
        dw, db = dw_value, db_value

    dh0 = None
    if initial_state is not None:
        dh0 = torch.full_like(initial_state, fill) if poison else torch.empty_like(initial_state)
        causal_conv1d_dh0_dense_kernel[(DB,)](
            dpre=dpre,
            weight=weight,
            dh0=dh0,
            T=T,
            D=D,
            W=W,
            BD=_BWD_DENSE_BD,
            num_warps=num_warps,
            multibuffer=False,
        )
    return dx, dw, db, dh0, dpre


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "D", "B_OFFSET", "NT_OFFSET", "D_BLOCK_OFFSET"])
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    T: tl.int64,
    B_OFFSET: tl.int64,
    NT_OFFSET: tl.int64,
    D_BLOCK_OFFSET: tl.int64,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    stride_residual_n,
    stride_residual_t,
    stride_residual_d,
    D: tl.int64,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64)
    i_t_local = tl.program_id(1).to(tl.int64)
    i_b_local = tl.program_id(2).to(tl.int64)
    chunk_id = tl.cast(NT_OFFSET, tl.int64) + i_t_local

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + chunk_id * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + chunk_id * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * tl.cast(stride_x_t, tl.int64)
        p_y = y + bos * tl.cast(stride_y_t, tl.int64)
        if HAS_RESIDUAL:
            p_residual = residual + bos * tl.cast(stride_residual_t, tl.int64)
    else:
        i_n = tl.cast(B_OFFSET, tl.int64) + i_b_local
        i_t = chunk_id
        bos = i_n * tl.cast(T, tl.int64)
        p_x = x + i_n * tl.cast(stride_x_n, tl.int64)
        p_y = y + i_n * tl.cast(stride_y_n, tl.int64)
        if HAS_RESIDUAL:
            p_residual = residual + i_n * tl.cast(stride_residual_n, tl.int64)

    o_d = (tl.cast(D_BLOCK_OFFSET, tl.int64) + i_d) * BD + tl.arange(0, BD).to(tl.int64)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    o_t = i_t * BT + tl.arange(0, BT).to(tl.int64)
    m_t = (o_t >= 0) & (o_t < T)
    b_y = tl.zeros((BT, BD), dtype=tl.float32)

    for i_w in tl.static_range(-W + 1, 1):
        o_x = o_t + i_w
        m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
        b_yi = tl.load(
            p_x
            + o_x[:, None] * tl.cast(stride_x_t, tl.int64)
            + o_d[None, :] * tl.cast(stride_x_d, tl.int64),
            mask=m_x,
            other=0,
        ).to(tl.float32)

        if USE_INITIAL_STATE:
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d[None, :]
            b_yi += tl.load(
                initial_state + i_n * D * W + o_d[None, :] * W + (o_x + W)[:, None],
                mask=m_c,
                other=0,
            ).to(tl.float32)

        if HAS_WEIGHT:
            b_yi = b_yi * tl.sum(b_w * (o_w == (i_w + W - 1)), 1)[None, :]
        b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)[None, :]
    if ACTIVATION:
        b_y *= tl.sigmoid(b_y)
    if HAS_RESIDUAL:
        b_y += tl.load(
            p_residual
            + o_t[:, None] * tl.cast(stride_residual_t, tl.int64)
            + o_d[None, :] * tl.cast(stride_residual_d, tl.int64),
            mask=m_t[:, None] & m_d[None, :],
            other=0,
        ).to(tl.float32)

    tl.store(
        p_y
        + o_t[:, None] * tl.cast(stride_y_t, tl.int64)
        + o_d[None, :] * tl.cast(stride_y_d, tl.int64),
        tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_t[:, None] & m_d[None, :],
    )


@triton.jit(do_not_specialize=["ELEMENT_OFFSET", "ELEMENT_END"])
def _silu_kernel(
    x_ptr,
    y_ptr,
    ELEMENT_OFFSET: tl.int64,
    ELEMENT_END: tl.int64,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64) + ELEMENT_OFFSET
    mask = offs < ELEMENT_END
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.sigmoid(x)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["ELEMENT_OFFSET", "ELEMENT_END"])
def _add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    ELEMENT_OFFSET: tl.int64,
    ELEMENT_END: tl.int64,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64) + ELEMENT_OFFSET
    mask = offs < ELEMENT_END
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (a + b).to(out_ptr.dtype.element_ty), mask=mask)


def _launch_silu(y: torch.Tensor) -> torch.Tensor:
    y = y.contiguous()
    out = torch.zeros_like(y)
    n = y.numel()
    for grid, elem_off in _elementwise_launch_iters(n):
        _silu_kernel[(grid,)](
            y,
            out,
            ELEMENT_OFFSET=elem_off,
            ELEMENT_END=n,
            BLOCK=_ELEM_BLOCK,
            num_warps=STATIC_WARPS,
        )
    return out


def _launch_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    b = b.contiguous()
    out = torch.zeros_like(a)
    n = a.numel()
    for grid, elem_off in _elementwise_launch_iters(n):
        _add_kernel[(grid,)](
            a,
            b,
            out,
            ELEMENT_OFFSET=elem_off,
            ELEMENT_END=n,
            BLOCK=_ELEM_BLOCK,
            num_warps=STATIC_WARPS,
        )
    return out


@triton.jit(do_not_specialize=["T", "D", "ELEMENT_OFFSET", "ELEMENT_END"])
def _silu_bwd_kernel(
    y_ptr,
    dy_ptr,
    out_ptr,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    stride_out_n,
    stride_out_t,
    stride_out_d,
    T: tl.int64,
    D: tl.int64,
    ELEMENT_OFFSET: tl.int64,
    ELEMENT_END: tl.int64,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64) + ELEMENT_OFFSET
    mask = offs < ELEMENT_END
    rem = offs % D
    d = rem
    rem = (offs - d) // D
    t = rem % T
    b = rem // T
    y_off = (
        b * tl.cast(stride_y_n, tl.int64)
        + t * tl.cast(stride_y_t, tl.int64)
        + d * tl.cast(stride_y_d, tl.int64)
    )
    dy_off = (
        b * tl.cast(stride_dy_n, tl.int64)
        + t * tl.cast(stride_dy_t, tl.int64)
        + d * tl.cast(stride_dy_d, tl.int64)
    )
    out_off = (
        b * tl.cast(stride_out_n, tl.int64)
        + t * tl.cast(stride_out_t, tl.int64)
        + d * tl.cast(stride_out_d, tl.int64)
    )
    y = tl.load(y_ptr + y_off, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + dy_off, mask=mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(y)
    out = dy * s * (1.0 + y * (1.0 - s))
    tl.store(out_ptr + out_off, out.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_silu_bwd(y_pre: torch.Tensor, dy: torch.Tensor, *, poison: bool = False) -> torch.Tensor:
    fill = float("nan") if poison else 0.0
    out = torch.full_like(dy, fill, dtype=torch.float32, memory_format=torch.contiguous_format)
    B, T, D = dy.shape
    n = B * T * D
    sy_n, sy_t, sy_d = y_pre.stride()
    sdy_n, sdy_t, sdy_d = dy.stride()
    so_n, so_t, so_d = out.stride()
    for grid, elem_off in _elementwise_launch_iters(n):
        _silu_bwd_kernel[(grid,)](
            y_pre,
            dy,
            out,
            sy_n,
            sy_t,
            sy_d,
            sdy_n,
            sdy_t,
            sdy_d,
            so_n,
            so_t,
            so_d,
            T,
            D,
            ELEMENT_OFFSET=elem_off,
            ELEMENT_END=n,
            BLOCK=_ELEM_BLOCK,
            num_warps=STATIC_WARPS,
        )
    return out


def _postprocess_fwd(
    y: torch.Tensor,
    residual: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    if activation in ("swish", "silu"):
        y = _launch_silu(y)
    if residual is not None:
        if residual.stride() != y.stride():
            residual = residual.contiguous()
        y = _launch_add(y, residual)
    return y


def _use_seq_bwd(
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    initial_state: torch.Tensor | None,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    use_seq = cu_seqlens is None and initial_state is None and dht is None and dtype == torch.bfloat16 and T <= 16
    return use_seq and triton.cdiv(B * T * D, 1024) <= _get_npu_max_grid()


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["dw"] is not None,
        "HAS_BIAS": lambda args: args["db"] is not None,
    }
)
@triton.jit(do_not_specialize=["B", "TC", "D"])
def causal_conv1d_bwd_seq_kernel(
    x,
    weight,
    dy,
    dx,
    dw,
    db,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    B: tl.int64,
    TC: tl.int64,
    D: tl.int64,
    W: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    n_elements = tl.cast(B, tl.int64) * TC * D
    mask = offs < n_elements
    d = offs % D
    tmp = offs // D
    t = tmp % TC
    b = tmp // TC

    b_dx = tl.zeros((BLOCK,), dtype=tl.float32)
    for i_w in tl.static_range(0, W):
        t_dy = t + i_w
        dy_off = (
            b * tl.cast(stride_dy_n, tl.int64)
            + t_dy * tl.cast(stride_dy_t, tl.int64)
            + d * tl.cast(stride_dy_d, tl.int64)
        )
        b_dy = tl.load(dy + dy_off, mask=mask & (t_dy < TC), other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w_idx = W - i_w - 1
            b_w = tl.load(weight + d * W + w_idx, mask=mask, other=0.0).to(tl.float32)
            b_dx += b_dy * b_w
        else:
            b_dx += b_dy

    dx_off = (
        b * tl.cast(stride_dx_n, tl.int64)
        + t * tl.cast(stride_dx_t, tl.int64)
        + d * tl.cast(stride_dx_d, tl.int64)
    )
    tl.store(dx + dx_off, b_dx.to(dx.dtype.element_ty), mask=mask)

    if HAS_WEIGHT:
        x_off = (
            b * tl.cast(stride_x_n, tl.int64)
            + t * tl.cast(stride_x_t, tl.int64)
            + d * tl.cast(stride_x_d, tl.int64)
        )
        b_x = tl.load(x + x_off, mask=mask, other=0.0).to(tl.float32)
        i_tg = b * TC + t
        for i_w in tl.static_range(0, W):
            t_dy = t + i_w
            dy_off = (
                b * tl.cast(stride_dy_n, tl.int64)
                + t_dy * tl.cast(stride_dy_t, tl.int64)
                + d * tl.cast(stride_dy_d, tl.int64)
            )
            b_dy = tl.load(dy + dy_off, mask=mask & (t_dy < TC), other=0.0).to(tl.float32)
            w_idx = W - i_w - 1
            tl.store(
                dw + (i_tg * D + d) * W + w_idx,
                (b_dy * b_x).to(dw.dtype.element_ty),
                mask=mask,
            )

    if HAS_BIAS:
        i_tg = b * TC + t
        dy_off = (
            b * tl.cast(stride_dy_n, tl.int64)
            + t * tl.cast(stride_dy_t, tl.int64)
            + d * tl.cast(stride_dy_d, tl.int64)
        )
        b_dy0 = tl.load(dy + dy_off, mask=mask, other=0.0)
        tl.store(db + i_tg * D + d, b_dy0.to(db.dtype.element_ty), mask=mask)


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "STORE_DW": lambda args: args["dw"] is not None,
        "HAS_BIAS": lambda args: args["db"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "USE_FINAL_STATE": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(
    do_not_specialize=["T", "D", "B_OFFSET", "NT_OFFSET", "D_BLOCK_OFFSET", "NT_TOTAL"]
)
def causal_conv1d_bwd_kernel(
    x,
    weight,
    initial_state,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    T: tl.int64,
    B_OFFSET: tl.int64,
    NT_OFFSET: tl.int64,
    D_BLOCK_OFFSET: tl.int64,
    NT_TOTAL: tl.int64,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    D: tl.int64,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    STORE_DW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64)
    i_t_local = tl.program_id(1).to(tl.int64)
    i_b_local = tl.program_id(2).to(tl.int64)
    chunk_id = tl.cast(NT_OFFSET, tl.int64) + i_t_local
    if IS_VARLEN:
        i_tg = chunk_id
        i_n = tl.load(chunk_indices + chunk_id * 2).to(tl.int64)
        i_t = tl.load(chunk_indices + chunk_id * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * tl.cast(stride_x_t, tl.int64)
        p_dy = dy + bos * tl.cast(stride_dy_t, tl.int64)
        p_dx = dx + bos * tl.cast(stride_dx_t, tl.int64)
    else:
        i_t = chunk_id
        i_n = tl.cast(B_OFFSET, tl.int64) + i_b_local
        i_tg = i_n * tl.cast(NT_TOTAL, tl.int64) + i_t
        p_x = x + i_n * tl.cast(stride_x_n, tl.int64)
        p_dy = dy + i_n * tl.cast(stride_dy_n, tl.int64)
        p_dx = dx + i_n * tl.cast(stride_dx_n, tl.int64)

    o_d = (tl.cast(D_BLOCK_OFFSET, tl.int64) + i_d) * BD + tl.arange(0, BD).to(tl.int64)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    o_t = i_t * BT + tl.arange(0, BT).to(tl.int64)
    m_t = (o_t >= 0) & (o_t < T)

    if STORE_DW:
        b_x = tl.load(
            p_x
            + o_t[:, None] * tl.cast(stride_x_t, tl.int64)
            + o_d[None, :] * tl.cast(stride_x_d, tl.int64),
            mask=m_t[:, None] & m_d[None, :],
            other=0,
        ).to(tl.float32)
    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    for i_w in tl.static_range(0, W):
        o_dy = o_t + i_w
        m_dy = ((o_dy >= 0) & (o_dy < T))[:, None] & m_d[None, :]
        b_dy = tl.load(
            p_dy
            + o_dy[:, None] * tl.cast(stride_dy_t, tl.int64)
            + o_d[None, :] * tl.cast(stride_dy_d, tl.int64),
            mask=m_dy,
            other=0,
        ).to(tl.float32)

        if HAS_WEIGHT:
            b_wdy = b_dy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)[None, :]
            if STORE_DW:
                b_dw = tl.sum(b_dy * b_x, 0)
                if USE_INITIAL_STATE:
                    mask_head_rows = (o_t < i_w) & (o_t < T)
                    b_dy_head = tl.load(
                        p_dy
                        + o_t[:, None] * tl.cast(stride_dy_t, tl.int64)
                        + o_d[None, :] * tl.cast(stride_dy_d, tl.int64),
                        mask=(mask_head_rows[:, None] & m_d[None, :]),
                        other=0.0,
                    ).to(tl.float32)
                    o_c = W - i_w + o_t
                    mask_c = mask_head_rows & (o_c >= 1) & (o_c < W)
                    b_xc = tl.load(
                        initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                        mask=(mask_c[:, None] & m_d[None, :]),
                        other=0.0,
                    ).to(tl.float32)
                    b_dw += tl.sum(b_dy_head * b_xc, 0)
                tl.store(
                    dw + i_tg * D * W + o_d * W + W - i_w - 1,
                    b_dw.to(dw.dtype.element_ty),
                    mask=m_d,
                )
        else:
            b_wdy = b_dy

        if HAS_BIAS and i_w == 0:
            b_db += tl.sum(b_dy, 0)
        b_dx += b_wdy

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

    if USE_FINAL_STATE:
        if i_t * BT + BT >= T - W:
            start_tok = T - (W - 1)
            offset = i_t * BT + tl.arange(0, BT)
            tok_idx = offset - start_tok
            mask = (offset >= start_tok) & (offset < T)
            w_idx = 1 + tok_idx
            dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
            b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            b_dx += b_dht

    tl.store(
        p_dx
        + o_t[:, None] * tl.cast(stride_dx_t, tl.int64)
        + o_d[None, :] * tl.cast(stride_dx_d, tl.int64),
        tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_t[:, None] & m_d[None, :],
    )


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T", "D", "D_BLOCK_OFFSET"])
def causal_conv1d_dw_reduce_kernel(
    x,
    dy,
    initial_state,
    cu_seqlens,
    dw,
    N: tl.int64,
    T: tl.int64,
    D_BLOCK_OFFSET: tl.int64,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    D: tl.int64,
    W: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64) + D_BLOCK_OFFSET
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    o_w = tl.arange(0, BW)
    m_w = o_w < W
    b_dw = tl.zeros((BW, BD), dtype=tl.float32)

    for i_n in tl.range(0, N, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            seq_len = eos - bos
            p_x = x + bos * tl.cast(stride_x_t, tl.int64)
            p_dy = dy + bos * tl.cast(stride_dy_t, tl.int64)
        else:
            seq_len = T
            i_n64 = i_n.to(tl.int64)
            p_x = x + i_n64 * tl.cast(stride_x_n, tl.int64)
            p_dy = dy + i_n64 * tl.cast(stride_dy_n, tl.int64)

        for i_t in tl.range(0, seq_len, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
            i_t64 = i_t.to(tl.int64)
            o_d64 = o_d.to(tl.int64)
            b_dy = tl.load(
                p_dy
                + i_t64 * tl.cast(stride_dy_t, tl.int64)
                + o_d64 * tl.cast(stride_dy_d, tl.int64),
                mask=m_d,
                other=0.0,
            ).to(tl.float32)

            source_t = i_t + o_w - (W - 1)
            source_t64 = source_t.to(tl.int64)
            b_source = tl.load(
                p_x
                + source_t64[:, None] * tl.cast(stride_x_t, tl.int64)
                + o_d64[None, :] * tl.cast(stride_x_d, tl.int64),
                mask=m_w[:, None] & (source_t >= 0)[:, None] & (source_t < seq_len)[:, None]
                & m_d[None, :],
                other=0.0,
            ).to(tl.float32)

            if USE_INITIAL_STATE:
                state_index = source_t + W
                i_n64 = i_n.to(tl.int64)
                b_source += tl.load(
                    initial_state
                    + i_n64 * D * W
                    + o_d64[None, :] * W
                    + state_index.to(tl.int64)[:, None],
                    mask=m_w[:, None]
                    & (source_t < 0)[:, None]
                    & (state_index >= 0)[:, None]
                    & (state_index < W)[:, None]
                    & m_d[None, :],
                    other=0.0,
                ).to(tl.float32)

            b_dw += b_source * b_dy[None, :]

    tl.store(
        dw + o_d.to(tl.int64)[:, None] * W + o_w.to(tl.int64)[None, :],
        tl.trans(b_dw),
        mask=m_d[:, None] & m_w[None, :],
    )


@triton.heuristics(
    {
        "USE_ACTIVATION": lambda args: args["y"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "D", "N_OFFSET", "D_BLOCK_OFFSET"])
def compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    T: tl.int64,
    N_OFFSET: tl.int64,
    D_BLOCK_OFFSET: tl.int64,
    D: tl.int64,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64)
    i_n = tl.program_id(1).to(tl.int64) + tl.cast(N_OFFSET, tl.int64)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
        dy_base = dy + bos * tl.cast(stride_dy_t, tl.int64)
    else:
        seq_len = T
        dy_base = dy + i_n * tl.cast(stride_dy_n, tl.int64)

    o_d = (tl.cast(D_BLOCK_OFFSET, tl.int64) + i_d) * BD + tl.arange(0, BD).to(tl.int64)
    m_d = o_d < D

    for i_w in tl.static_range(1, W):
        b_dh0 = tl.zeros([BD], dtype=tl.float32)

        for t in tl.static_range(0, W - 1):
            if t < i_w:
                w_idx = i_w - 1 - t
                p_dy = (
                    dy_base
                    + t * tl.cast(stride_dy_t, tl.int64)
                    + o_d * tl.cast(stride_dy_d, tl.int64)
                )
                m_t = (t < seq_len) & m_d
                b_dy = tl.load(p_dy, mask=m_t, other=0).to(tl.float32)

                if USE_ACTIVATION:
                    if IS_VARLEN:
                        p_y = (
                            y
                            + bos * tl.cast(stride_y_t, tl.int64)
                            + t * tl.cast(stride_y_t, tl.int64)
                            + o_d * tl.cast(stride_y_d, tl.int64)
                        )
                    else:
                        p_y = (
                            y
                            + i_n * tl.cast(stride_y_n, tl.int64)
                            + t * tl.cast(stride_y_t, tl.int64)
                            + o_d * tl.cast(stride_y_d, tl.int64)
                        )
                    b_y = tl.load(p_y, mask=m_t, other=0).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

                b_w_col = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0).to(tl.float32)
                b_dh0 += tl.where(m_t, b_dy * b_w_col, 0)

        p_dh0 = dh0 + i_n * D * W + o_d * W + i_w
        tl.store(p_dh0, b_dh0.to(dh0.dtype.element_ty), mask=m_d)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "D", "W", "N_OFFSET", "D_BLOCK_OFFSET"])
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T: tl.int64,
    D: tl.int64,
    W: tl.int64,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    N_OFFSET: tl.int64,
    D_BLOCK_OFFSET: tl.int64,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64)
    i_n = tl.program_id(1).to(tl.int64) + tl.cast(N_OFFSET, tl.int64)

    o_d = (tl.cast(D_BLOCK_OFFSET, tl.int64) + i_d) * BD + tl.arange(0, BD).to(tl.int64)
    m_d = o_d < D

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
        p_x = x + bos * tl.cast(stride_x_t, tl.int64)
    else:
        seq_len = tl.cast(T, tl.int64)
        p_x = x + i_n * tl.cast(stride_x_n, tl.int64)

    o_w = W - BW + tl.arange(0, BW).to(tl.int64)
    m_w = o_w >= 0
    o_t = seq_len - BW + tl.arange(0, BW).to(tl.int64)
    m_t = (o_t >= 0) & (o_t < seq_len)

    b_x = tl.load(
        p_x
        + o_t[:, None] * tl.cast(stride_x_t, tl.int64)
        + o_d[None, :] * tl.cast(stride_x_d, tl.int64),
        mask=m_t[:, None] & m_d[None, :],
        other=0,
    ).to(tl.float32)

    if USE_INITIAL_STATE:
        if seq_len < BW:
            o_c = W - (BW - seq_len) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)
            b_cache = tl.load(
                initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                mask=m_d[None, :] & m_c[:, None],
                other=0,
            ).to(tl.float32)
            b_x += b_cache

    p_final = final_state + tl.cast(i_n, tl.int64) * D * W + o_d[:, None] * W + o_w[None, :]
    tl.store(p_final, tl.trans(b_x).to(final_state.dtype.element_ty), mask=m_d[:, None] & m_w[None, :])


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
    }
)
@triton.jit(do_not_specialize=["D", "N_OFFSET", "D_BLOCK_OFFSET"])
def causal_conv1d_update_kernel(
    x,
    cache,
    y,
    weight,
    bias,
    stride_x_n,
    stride_x_d,
    stride_y_n,
    stride_y_d,
    N_OFFSET: tl.int64,
    D_BLOCK_OFFSET: tl.int64,
    D: tl.int64,
    W: tl.constexpr,
    BD: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64)
    i_n = tl.program_id(1).to(tl.int64) + tl.cast(N_OFFSET, tl.int64)

    o_d = (tl.cast(D_BLOCK_OFFSET, tl.int64) + i_d) * BD + tl.arange(0, BD).to(tl.int64)
    m_d = o_d < D

    b_x = tl.load(
        x + i_n * tl.cast(stride_x_n, tl.int64) + o_d * tl.cast(stride_x_d, tl.int64),
        mask=m_d,
        other=0,
    ).to(tl.float32)

    b_y = tl.zeros((BD,), dtype=tl.float32)
    for iw in tl.static_range(0, W):
        if iw < W - 1:
            b_c = tl.load(cache + i_n * D * W + o_d * W + (iw + 1), mask=m_d, other=0).to(tl.float32)
        else:
            b_c = b_x
        tl.store(
            cache + i_n * D * W + o_d * W + iw,
            tl.cast(b_c, dtype=cache.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_d,
        )
        if HAS_WEIGHT:
            b_y += b_c * tl.load(weight + o_d * W + iw, mask=m_d, other=0).to(tl.float32)
        else:
            b_y += b_c

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    tl.store(
        y + i_n * tl.cast(stride_y_n, tl.int64) + o_d * tl.cast(stride_y_d, tl.int64),
        tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_d,
    )


def _postprocess_update(
    y: torch.Tensor,
    residual: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    if activation in ("swish", "silu"):
        y = _launch_silu(y)
    if residual is not None:
        if residual.stride() != y.stride():
            residual = residual.contiguous()
        y = _launch_add(y, residual)
    return y


def _launch_fwd_core(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    B: int,
    T: int,
    D: int,
    W: int,
    BT: int,
    BD: int | None = None,
    num_warps: int | None = None,
    residual: torch.Tensor | None = None,
    activation: str | None = None,
    output_dtype: torch.dtype | None = None,
    poison: bool = False,
) -> torch.Tensor:
    if BD is None or num_warps is None:
        BD, BT, num_warps = _npu_tile_config(T, BT, D, x.dtype, initial_state)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    BW = triton.next_power_of_2(W)

    stride_x_n, stride_x_t, stride_x_d = x.stride()
    fill = float("nan") if poison else 0.0
    y = torch.full_like(x, fill, dtype=output_dtype, memory_format=torch.contiguous_format)
    stride_y_n, stride_y_t, stride_y_d = y.stride()
    if residual is not None and residual.stride() != y.stride():
        residual = residual.contiguous()
    stride_residual_n = stride_residual_t = stride_residual_d = 0
    if residual is not None:
        stride_residual_n, stride_residual_t, stride_residual_d = residual.stride()

    kernel_kwargs = dict(
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        BD=BD,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_t=stride_y_t,
        stride_y_d=stride_y_d,
        stride_residual_n=stride_residual_n,
        stride_residual_t=stride_residual_t,
        stride_residual_d=stride_residual_d,
        ACTIVATION=activation in ("swish", "silu"),
        num_warps=num_warps,
    )
    for b_off, b_len, nt_off, nt_len, d_off, d_len in _iter_3d_grid_splits(B, NT, D, BD):
        grid = (d_len, nt_len, b_len)
        kernel_kwargs["chunk_indices"] = chunk_indices
        kernel_kwargs["B_OFFSET"] = b_off
        kernel_kwargs["NT_OFFSET"] = nt_off
        kernel_kwargs["D_BLOCK_OFFSET"] = d_off
        causal_conv1d_fwd_kernel[grid](**kernel_kwargs)
    return y


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_fwd_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
    layout_fallback: bool = False,
):
    del layout_fallback
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, "b t ... -> b t (...)")
    B, T, D = x.shape[0], x.shape[1], weight.shape[0]
    W = weight.shape[1]

    if _is_dense_single_sequence(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    ):
        y = _launch_fwd_dense(x, weight, bias, residual, initial_state, activation)
    else:
        BD, BT, num_warps = _npu_tile_config(T, BT, D, x.dtype, initial_state)
        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)

        y = _launch_fwd_core(
            x,
            weight,
            bias,
            initial_state,
            cu_seqlens,
            chunk_indices,
            B,
            T,
            D,
            W,
            BT,
            BD,
            num_warps,
            residual,
            activation,
        )

    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states_npu(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    return y.view(shape), final_state


def _reduce_dw_general_npu(
    x: torch.Tensor,
    dy: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    B: int,
    T: int,
    D: int,
    W: int,
) -> torch.Tensor:
    """Reduce weight gradients directly in FP32 without a token workspace."""
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B
    BD = 16
    BW = triton.next_power_of_2(W)
    dw = torch.empty((D, W), dtype=torch.float32, device=x.device)
    stride_x_n, stride_x_t, stride_x_d = x.stride()
    stride_dy_n, stride_dy_t, stride_dy_d = dy.stride()
    d_blocks = triton.cdiv(D, BD)
    for d_off in range(0, d_blocks, _get_npu_max_grid()):
        d_len = min(_get_npu_max_grid(), d_blocks - d_off)
        causal_conv1d_dw_reduce_kernel[(d_len,)](
            x=x,
            dy=dy,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            dw=dw,
            N=N,
            T=T,
            D_BLOCK_OFFSET=d_off,
            stride_x_n=stride_x_n,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
            stride_dy_n=stride_dy_n,
            stride_dy_t=stride_dy_t,
            stride_dy_d=stride_dy_d,
            D=D,
            W=W,
            BW=BW,
            BD=BD,
            num_warps=STATIC_WARPS,
            multibuffer=False,
        )
    return dw


def causal_conv1d_bwd_npu(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
    layout_fallback: bool = False,
):
    del layout_fallback
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, "b t ... -> b t (...)")
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None

    if _is_dense_backward(
        x=x,
        dy=dy,
        weight=weight,
        bias=bias,
        initial_state=initial_state,
        dht=dht,
        activation=activation,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    ):
        dx, dw, db, dh0, _ = _launch_bwd_dense(
            x=x,
            dy=dy,
            weight=weight,
            bias=bias,
            initial_state=initial_state,
            activation=activation,
        )
        dr = dy if residual is not None else None
        return dx.view(shape), dw, db, dr, dh0

    BD, BT, num_warps = _npu_bwd_tile_config(T, BT, D, x.dtype, initial_state)
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    BW = triton.next_power_of_2(W)

    dr = dy if residual is not None else None
    dy_conv = dy

    y_pre = None
    if activation in ("swish", "silu"):
        BD_f, BT_f, nw_f = _npu_tile_config(T, BT, D, x.dtype, initial_state)
        chunk_indices_f = chunk_indices
        if cu_seqlens is not None:
            chunk_indices_f = prepare_chunk_indices(cu_seqlens, BT_f, cu_seqlens_cpu=cu_seqlens_cpu)
        y_pre = _launch_fwd_core(
            x,
            weight,
            bias,
            initial_state,
            cu_seqlens,
            chunk_indices_f,
            B,
            T,
            D,
            W,
            BT_f,
            BD_f,
            nw_f,
            output_dtype=torch.float32,
        )
        dy_conv = _launch_silu_bwd(y_pre, dy)

    stride_x_n, stride_x_t, stride_x_d = x.stride()
    use_seq = _use_seq_bwd(B, T, D, x.dtype, initial_state, dht, cu_seqlens)
    stride_dy_n, stride_dy_t, stride_dy_d = dy_conv.stride()

    dx = torch.zeros_like(x)
    stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()

    if use_seq:
        block = 1024
        dw = weight.new_empty(B * T, *weight.shape, dtype=torch.float) if weight is not None else None
        db = None
        grid = (triton.cdiv(B * T * D, block),)
        causal_conv1d_bwd_seq_kernel[grid](
            x=x,
            weight=weight,
            dy=dy_conv,
            dx=dx,
            dw=dw,
            db=db,
            stride_x_n=stride_x_n,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
            stride_dx_n=stride_dx_n,
            stride_dx_t=stride_dx_t,
            stride_dx_d=stride_dx_d,
            stride_dy_n=stride_dy_n,
            stride_dy_t=stride_dy_t,
            stride_dy_d=stride_dy_d,
            B=B,
            TC=T,
            D=D,
            W=W,
            BLOCK=block,
            num_warps=STATIC_WARPS,
        )
    else:
        if not dy_conv.is_contiguous():
            dy_conv = dy_conv.contiguous()
        stride_dy_n, stride_dy_t, stride_dy_d = dy_conv.stride()
        dw = None
        db = None
        kernel_kwargs = dict(
            x=x,
            weight=weight,
            initial_state=initial_state,
            dht=dht,
            dy=dy_conv,
            dx=dx,
            cu_seqlens=cu_seqlens,
            T=T,
            D=D,
            W=W,
            BT=BT,
            BW=BW,
            BD=BD,
            stride_x_n=stride_x_n,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
            stride_dx_n=stride_dx_n,
            stride_dx_t=stride_dx_t,
            stride_dx_d=stride_dx_d,
            stride_dy_n=stride_dy_n,
            stride_dy_t=stride_dy_t,
            stride_dy_d=stride_dy_d,
            num_warps=num_warps,
            NT_TOTAL=NT,
        )
        for b_off, b_len, nt_off, nt_len, d_off, d_len in _iter_3d_grid_splits(B, NT, D, BD):
            grid = (d_len, nt_len, b_len)
            kernel_kwargs["chunk_indices"] = chunk_indices
            kernel_kwargs["dw"] = dw
            kernel_kwargs["db"] = db
            kernel_kwargs["B_OFFSET"] = b_off
            kernel_kwargs["NT_OFFSET"] = nt_off
            kernel_kwargs["D_BLOCK_OFFSET"] = d_off
            causal_conv1d_bwd_kernel[grid](**kernel_kwargs)
    if weight is not None:
        if use_seq:
            dw = dw.sum(0).to(weight)
        else:
            dw = _reduce_dw_general_npu(
                x=x,
                dy=dy_conv,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                B=B,
                T=T,
                D=D,
                W=W,
            ).to(weight)
    if bias is not None:
        db = dy_conv.float().sum(dim=(0, 1)).to(bias)

    dh0 = None
    if initial_state is not None:
        dh0 = compute_dh0_npu(
            dy=dy,
            y=y_pre,
            weight=weight,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )

    return dx.view(shape), dw, db, dr, dh0


def compute_dh0_npu(
    dy: torch.Tensor,
    y: torch.Tensor | None,
    weight: torch.Tensor,
    initial_state: torch.Tensor,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    D, W = weight.shape
    N = initial_state.shape[0]
    T = dy.shape[1]

    BD = 8 if dy.dtype == torch.float16 and activation in ("swish", "silu") else 16
    dh0 = torch.zeros_like(initial_state)

    stride_dy_n = dy.stride(0)
    stride_dy_t = dy.stride(1)
    stride_dy_d = dy.stride(2) if dy.dim() == 3 else dy.stride(-1)
    stride_y_n = stride_y_t = stride_y_d = 0
    if y is not None:
        stride_y_n = y.stride(0)
        stride_y_t = y.stride(1)
        stride_y_d = y.stride(2) if y.dim() == 3 else y.stride(-1)

    kernel_kwargs = dict(
        dy=dy,
        y=y if activation in ("swish", "silu") else None,
        weight=weight,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        stride_dy_n=stride_dy_n,
        stride_dy_t=stride_dy_t,
        stride_dy_d=stride_dy_d,
        stride_y_n=stride_y_n,
        stride_y_t=stride_y_t,
        stride_y_d=stride_y_d,
        T=T,
        D=D,
        W=W,
        BD=BD,
        num_warps=STATIC_WARPS,
    )
    for n_off, n_len, _, _, d_off, d_len in _iter_3d_grid_splits(N, 1, D, BD):
        kernel_kwargs["N_OFFSET"] = n_off
        kernel_kwargs["D_BLOCK_OFFSET"] = d_off
        compute_dh0_kernel[(d_len, n_len)](**kernel_kwargs)
    return dh0


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update_states_npu(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    layout_fallback: bool = False,
    poison: bool = False,
) -> torch.Tensor:
    del layout_fallback
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if x.dim() == 2:
            stride_x_n = 0
            stride_x_t, stride_x_d = x.stride()
            T = x.shape[0]
        else:
            stride_x_n = x.stride(0)
            stride_x_t, stride_x_d = x.stride(1), x.stride(2)
            T = x.shape[1]
        D = x.shape[-1]
    else:
        B, T, D = x.shape
        N = B
        stride_x_n, stride_x_t, stride_x_d = x.stride()

    W = state_len
    fill = float("nan") if poison else 0.0
    final_state = torch.full((N, D, W), fill, dtype=x.dtype, device=x.device)
    BD = min(triton.next_power_of_2(D), 16)
    BW = triton.next_power_of_2(W)
    kernel_kwargs = dict(
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        BW=BW,
        BD=BD,
        num_warps=STATIC_WARPS,
    )
    for n_off, n_len, _, _, d_off, d_len in _iter_3d_grid_splits(N, 1, D, BD):
        kernel_kwargs["N_OFFSET"] = n_off
        kernel_kwargs["D_BLOCK_OFFSET"] = d_off
        causal_conv1d_states_fwd_kernel[(d_len, n_len)](**kernel_kwargs)
    return final_state


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update_npu(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    if weight is not None and x.shape[-1] != weight.shape[0]:
        x = rearrange(x, "b t ... -> b t (...)")

    D = x.shape[-1]
    N = x.numel() // D
    W = weight.shape[1] if weight is not None else None
    BD = min(triton.next_power_of_2(D), 16)

    if x.dim() == 2:
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(1)
    elif x.dim() == 3 and x.shape[0] == 1:
        stride_x_n = x.stride(1)
        stride_x_d = x.stride(2)
    elif x.dim() == 3:
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(2)
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")

    y = torch.zeros_like(x, memory_format=torch.contiguous_format)

    if y.dim() == 2:
        stride_y_n, stride_y_d = y.stride(0), y.stride(1)
    elif y.dim() == 3 and y.shape[0] == 1:
        stride_y_n, stride_y_d = y.stride(1), y.stride(2)
    elif y.dim() == 3:
        stride_y_n, stride_y_d = y.stride(0), y.stride(2)

    kernel_kwargs = dict(
        x=x,
        cache=cache,
        y=y,
        weight=weight,
        bias=bias,
        stride_x_n=stride_x_n,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_d=stride_y_d,
        D=D,
        W=W,
        BD=BD,
        num_warps=STATIC_WARPS,
    )
    for n_off, n_len, _, _, d_off, d_len in _iter_3d_grid_splits(N, 1, D, BD):
        kernel_kwargs["N_OFFSET"] = n_off
        kernel_kwargs["D_BLOCK_OFFSET"] = d_off
        causal_conv1d_update_kernel[(d_len, n_len)](**kernel_kwargs)
    y = _postprocess_update(y, residual, activation)
    return y.view(shape), cache
