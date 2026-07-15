# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend kernels for GDN context-parallel state preprocessing."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.utils.op import exp2
from fla.utils import device_torch_lib
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM, iter_axis_launch_chunks

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext


_NUM_WARPS = 4
_SUMMARY_STREAMS: dict[int, object] = {}
_GDN_PRECISION_ENV = 'FLA_ASCEND_CP_GDN_PRECISION'
_GDN_PRECISION_MODES = ('high', 'a800')


def _gdn_precision_mode() -> str:
    mode = os.environ.get(_GDN_PRECISION_ENV, 'high').lower()
    if mode not in _GDN_PRECISION_MODES:
        choices = ', '.join(_GDN_PRECISION_MODES)
        raise ValueError(f'{_GDN_PRECISION_ENV} must be one of {choices}, but got {mode!r}')
    return mode


def _use_a800_transition_precision(
    *,
    precision_mode: str,
    dtype: torch.dtype,
    K: int,
    V: int,
    segment_t: int,
) -> bool:
    return precision_mode == 'a800' and dtype == torch.bfloat16 and K == V == 128 and segment_t == 2048


def _summary_stream(tensor: torch.Tensor):
    device_index = tensor.device.index
    if device_index is None:
        device_index = device_torch_lib.current_device()
    stream = _SUMMARY_STREAMS.get(device_index)
    if stream is None:
        stream = device_torch_lib.Stream(device=device_index)
        _SUMMARY_STREAMS[device_index] = stream
    return stream


def _value_tile_size(K: int, V: int) -> int:
    if K <= 64:
        return min(64, triton.next_power_of_2(V))
    if K <= 128:
        return min(128, triton.next_power_of_2(V))
    return min(16, triton.next_power_of_2(V))


def _backward_value_tile_size(K: int, V: int) -> int:
    if K <= 128:
        return min(64, triton.next_power_of_2(V))
    return min(16, triton.next_power_of_2(V))


def _matrix_tile_size(K: int) -> int:
    return 128 if K <= 128 else 16


def _launch_flat(kernel, total_tasks: int, **kwargs) -> None:
    for task_offset, task_count in iter_axis_launch_chunks(
        total_tasks,
        other_grid_product=1,
        max_grid=ASCEND_MAX_GRID_DIM,
    ):
        kernel[(task_count,)](
            TASK_OFFSET=task_offset,
            num_warps=_NUM_WARPS,
            multibuffer=False,
            **kwargs,
        )


@triton.jit
def _dot_fp32_low_rhs(lhs, rhs):
    """Preserve the FP32 left operand with a low-precision residual pair."""
    lhs_hi = lhs.to(rhs.dtype)
    lhs_lo = (lhs - lhs_hi.to(tl.float32)).to(rhs.dtype)
    return tl.dot(lhs_hi, rhs, allow_tf32=False) + tl.dot(lhs_lo, rhs, allow_tf32=False)


@triton.jit(do_not_specialize=['BOS', 'SEGMENT_T'])
def _cp_gdn_gate_factors_kernel(
    g,
    gate_rel,
    gate_decay,
    BOS,
    SEGMENT_T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    task = tl.program_id(0) + TASK_OFFSET
    i_h = task // NT
    i_t = task - i_h * NT
    o_t = tl.arange(0, BT)
    rel_t = i_t * BT + o_t
    m_t = rel_t < SEGMENT_T
    token = (BOS + rel_t).to(tl.int64)
    last_rel = tl.minimum((i_t + 1) * BT, SEGMENT_T) - 1
    last_token = (BOS + last_rel).to(tl.int64)
    b_g_last = tl.load(g + last_token * HV + i_h).to(tl.float32)
    b_g = tl.load(g + token * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_rel = tl.where(m_t, exp2(b_g_last - b_g), 0.0)
    tl.store(gate_rel + rel_t * HV + i_h, b_rel, mask=m_t)
    tl.store(gate_decay + i_t * HV + i_h, exp2(b_g_last))


@triton.jit(do_not_specialize=['BOS', 'SEGMENT_T'])
def _cp_gdn_bwd_gate_factors_kernel(
    g,
    gate_rel,
    gate_abs,
    gate_decay,
    BOS,
    SEGMENT_T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    task = tl.program_id(0) + TASK_OFFSET
    i_h = task // NT
    i_t = task - i_h * NT
    o_t = tl.arange(0, BT)
    rel_t = i_t * BT + o_t
    m_t = rel_t < SEGMENT_T
    token = (BOS + rel_t).to(tl.int64)
    last_rel = tl.minimum((i_t + 1) * BT, SEGMENT_T) - 1
    last_token = (BOS + last_rel).to(tl.int64)
    b_g_last = tl.load(g + last_token * HV + i_h).to(tl.float32)
    b_g = tl.load(g + token * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
    gate_base = i_h.to(tl.int64) * SEGMENT_T
    tl.store(gate_rel + gate_base + rel_t, exp2(b_g_last - b_g), mask=m_t)
    tl.store(gate_abs + gate_base + rel_t, exp2(b_g), mask=m_t)
    tl.store(gate_decay + i_h * NT + i_t, exp2(b_g_last))


@triton.jit(do_not_specialize=['BOS', 'SEGMENT_T', 'NT'])
def _cp_gdn_fwd_h_kernel(
    k,
    w,
    u,
    g,
    gate_rel,
    gate_decay,
    hm,
    BOS,
    SEGMENT_T,
    NT,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    PRECOMPUTED_GATE: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    task = tl.program_id(0) + TASK_OFFSET
    i_h = task // NV
    i_v = task - i_h * NV
    i_kh = i_h // (HV // H)

    o_t = tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, 64)
    m_v = o_v < V
    k1 = o_k
    if K > 64:
        k2 = 64 + o_k
    if K > 128:
        k3 = 128 + o_k
    if K > 192:
        k4 = 192 + o_k

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    for i_t in tl.range(
        0,
        NT,
        loop_unroll_factor=1,
        disallow_acc_multi_buffer=True,
    ):
        rel_t = i_t * BT + o_t
        m_t = rel_t < SEGMENT_T
        token = (BOS + rel_t).to(tl.int64)

        p_w = w + token[:, None] * (HV * K) + i_h * K + k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & (k1 < K)[None, :], other=0.0)
        b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype), allow_tf32=False)
        if K > 64:
            p_w = w + token[:, None] * (HV * K) + i_h * K + k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (k2 < K)[None, :], other=0.0)
            b_v_decay += tl.dot(b_w, b_h2.to(b_w.dtype), allow_tf32=False)
        if K > 128:
            p_w = w + token[:, None] * (HV * K) + i_h * K + k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (k3 < K)[None, :], other=0.0)
            b_v_decay += tl.dot(b_w, b_h3.to(b_w.dtype), allow_tf32=False)
        if K > 192:
            p_w = w + token[:, None] * (HV * K) + i_h * K + k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (k4 < K)[None, :], other=0.0)
            b_v_decay += tl.dot(b_w, b_h4.to(b_w.dtype), allow_tf32=False)

        p_u = u + token[:, None] * (HV * V) + i_h * V + o_v[None, :]
        b_v = tl.load(p_u, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v_decay

        if PRECOMPUTED_GATE:
            b_rel = tl.load(gate_rel + rel_t * HV + i_h, mask=m_t, other=0.0)
            b_decay = tl.load(gate_decay + i_t * HV + i_h)
        else:
            last_rel = tl.minimum((i_t + 1) * BT, SEGMENT_T) - 1
            last_token = (BOS + last_rel).to(tl.int64)
            b_g_last = tl.load(g + last_token * HV + i_h).to(tl.float32)
            b_g = tl.load(g + token * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
            b_rel = tl.where(m_t, exp2(b_g_last - b_g), 0.0)
            b_decay = exp2(b_g_last)
        b_v *= b_rel[:, None]
        b_h1 *= b_decay
        if K > 64:
            b_h2 *= b_decay
        if K > 128:
            b_h3 *= b_decay
        if K > 192:
            b_h4 *= b_decay
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + token[None, :] * (H * K) + i_kh * K + k1[:, None]
        b_k = tl.load(p_k, mask=(k1 < K)[:, None] & m_t[None, :], other=0.0)
        b_h1 += tl.dot(b_k, b_v, allow_tf32=False)
        if K > 64:
            p_k = k + token[None, :] * (H * K) + i_kh * K + k2[:, None]
            b_k = tl.load(p_k, mask=(k2 < K)[:, None] & m_t[None, :], other=0.0)
            b_h2 += tl.dot(b_k, b_v, allow_tf32=False)
        if K > 128:
            p_k = k + token[None, :] * (H * K) + i_kh * K + k3[:, None]
            b_k = tl.load(p_k, mask=(k3 < K)[:, None] & m_t[None, :], other=0.0)
            b_h3 += tl.dot(b_k, b_v, allow_tf32=False)
        if K > 192:
            p_k = k + token[None, :] * (H * K) + i_kh * K + k4[:, None]
            b_k = tl.load(p_k, mask=(k4 < K)[:, None] & m_t[None, :], other=0.0)
            b_h4 += tl.dot(b_k, b_v, allow_tf32=False)

    hm_base = (i_h * K * (V + K)).to(tl.int64)
    p_h = hm + hm_base + k1[:, None] * (V + K) + o_v[None, :]
    tl.store(p_h, b_h1, mask=(k1 < K)[:, None] & m_v[None, :])
    if K > 64:
        p_h = hm + hm_base + k2[:, None] * (V + K) + o_v[None, :]
        tl.store(p_h, b_h2, mask=(k2 < K)[:, None] & m_v[None, :])
    if K > 128:
        p_h = hm + hm_base + k3[:, None] * (V + K) + o_v[None, :]
        tl.store(p_h, b_h3, mask=(k3 < K)[:, None] & m_v[None, :])
    if K > 192:
        p_h = hm + hm_base + k4[:, None] * (V + K) + o_v[None, :]
        tl.store(p_h, b_h4, mask=(k4 < K)[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=['BOS', 'SEGMENT_T', 'NT'])
def _cp_gdn_fwd_m_kernel(
    k,
    w,
    g,
    gate_rel,
    gate_decay,
    hm,
    BOS,
    SEGMENT_T,
    NT,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    NM: tl.constexpr,
    PRECOMPUTED_GATE: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    task = tl.program_id(0) + TASK_OFFSET
    i_h = task // NM
    i_m = task - i_h * NM
    i_kh = i_h // (HV // H)

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, 64)
    o_m = i_m * BM + tl.arange(0, BM)
    m_m = o_m < K

    k1 = o_k
    b_m1 = tl.where(k1[:, None] == o_m[None, :], 1.0, 0.0)
    if K > 64:
        k2 = 64 + o_k
        b_m2 = tl.where(k2[:, None] == o_m[None, :], 1.0, 0.0)
    if K > 128:
        k3 = 128 + o_k
        b_m3 = tl.where(k3[:, None] == o_m[None, :], 1.0, 0.0)
    if K > 192:
        k4 = 192 + o_k
        b_m4 = tl.where(k4[:, None] == o_m[None, :], 1.0, 0.0)

    for i_t in tl.range(
        0,
        NT,
        loop_unroll_factor=1,
        disallow_acc_multi_buffer=True,
    ):
        rel_t = i_t * BT + o_t
        m_t = rel_t < SEGMENT_T
        token = (BOS + rel_t).to(tl.int64)

        p_w = w + token[:, None] * (HV * K) + i_h * K + k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & (k1 < K)[None, :], other=0.0)
        b_tmp = tl.dot(b_w.to(tl.float32), b_m1, allow_tf32=False)
        if K > 64:
            p_w = w + token[:, None] * (HV * K) + i_h * K + k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (k2 < K)[None, :], other=0.0)
            b_tmp += tl.dot(b_w.to(tl.float32), b_m2, allow_tf32=False)
        if K > 128:
            p_w = w + token[:, None] * (HV * K) + i_h * K + k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (k3 < K)[None, :], other=0.0)
            b_tmp += tl.dot(b_w.to(tl.float32), b_m3, allow_tf32=False)
        if K > 192:
            p_w = w + token[:, None] * (HV * K) + i_h * K + k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (k4 < K)[None, :], other=0.0)
            b_tmp += tl.dot(b_w.to(tl.float32), b_m4, allow_tf32=False)

        if PRECOMPUTED_GATE:
            b_rel = tl.load(gate_rel + rel_t * HV + i_h, mask=m_t, other=0.0)
            b_decay = tl.load(gate_decay + i_t * HV + i_h)
        else:
            last_rel = tl.minimum((i_t + 1) * BT, SEGMENT_T) - 1
            last_token = (BOS + last_rel).to(tl.int64)
            b_g_last = tl.load(g + last_token * HV + i_h).to(tl.float32)
            b_g = tl.load(g + token * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
            b_rel = tl.where(m_t, exp2(b_g_last - b_g), 0.0)
            b_decay = exp2(b_g_last)

        p_k = k + token[:, None] * (H * K) + i_kh * K + k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (k1 < K)[None, :], other=0.0)
        b_kg = (b_k.to(tl.float32) * b_rel[:, None]).to(b_k.dtype)
        b_m1 = b_decay * b_m1 - tl.dot(tl.trans(b_kg.to(tl.float32)), b_tmp, allow_tf32=False)
        if K > 64:
            p_k = k + token[:, None] * (H * K) + i_kh * K + k2[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (k2 < K)[None, :], other=0.0)
            b_kg = (b_k.to(tl.float32) * b_rel[:, None]).to(b_k.dtype)
            b_m2 = b_decay * b_m2 - tl.dot(tl.trans(b_kg.to(tl.float32)), b_tmp, allow_tf32=False)
        if K > 128:
            p_k = k + token[:, None] * (H * K) + i_kh * K + k3[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (k3 < K)[None, :], other=0.0)
            b_kg = (b_k.to(tl.float32) * b_rel[:, None]).to(b_k.dtype)
            b_m3 = b_decay * b_m3 - tl.dot(tl.trans(b_kg.to(tl.float32)), b_tmp, allow_tf32=False)
        if K > 192:
            p_k = k + token[:, None] * (H * K) + i_kh * K + k4[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (k4 < K)[None, :], other=0.0)
            b_kg = (b_k.to(tl.float32) * b_rel[:, None]).to(b_k.dtype)
            b_m4 = b_decay * b_m4 - tl.dot(tl.trans(b_kg.to(tl.float32)), b_tmp, allow_tf32=False)

    hm_base = (i_h * K * (V + K) + V).to(tl.int64)
    p_m = hm + hm_base + k1[:, None] * (V + K) + o_m[None, :]
    tl.store(p_m, b_m1, mask=(k1 < K)[:, None] & m_m[None, :])
    if K > 64:
        p_m = hm + hm_base + k2[:, None] * (V + K) + o_m[None, :]
        tl.store(p_m, b_m2, mask=(k2 < K)[:, None] & m_m[None, :])
    if K > 128:
        p_m = hm + hm_base + k3[:, None] * (V + K) + o_m[None, :]
        tl.store(p_m, b_m3, mask=(k3 < K)[:, None] & m_m[None, :])
    if K > 192:
        p_m = hm + hm_base + k4[:, None] * (V + K) + o_m[None, :]
        tl.store(p_m, b_m4, mask=(k4 < K)[:, None] & m_m[None, :])


@triton.jit(do_not_specialize=['BOS', 'SEGMENT_T', 'NT', 'scale'])
def _cp_gdn_bwd_dh_kernel(
    q,
    k,
    w,
    do,
    dv,
    g,
    dhm,
    BOS,
    SEGMENT_T,
    NT,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    task = tl.program_id(0) + TASK_OFFSET
    i_h = task // NV
    i_v = task - i_h * NV
    i_qh = i_h // (HV // H)

    o_t = tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, 64)
    m_v = o_v < V
    k1 = o_k
    if K > 64:
        k2 = 64 + o_k
    if K > 128:
        k3 = 128 + o_k
    if K > 192:
        k4 = 192 + o_k

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    for step in tl.range(
        0,
        NT,
        loop_unroll_factor=1,
        disallow_acc_multi_buffer=True,
    ):
        i_t = NT - 1 - step
        rel_t = i_t * BT + o_t
        m_t = rel_t < SEGMENT_T
        token = (BOS + rel_t).to(tl.int64)
        last_rel = tl.minimum((i_t + 1) * BT, SEGMENT_T) - 1
        last_token = (BOS + last_rel).to(tl.int64)
        b_g_last = tl.load(g + last_token * HV + i_h).to(tl.float32)
        b_g = tl.load(g + token * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
        b_rel = tl.where(m_t, exp2(b_g_last - b_g), 0.0)
        b_gate = tl.where(m_t, exp2(b_g), 0.0)

        p_k = k + token[:, None] * (H * K) + i_qh * K + k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (k1 < K)[None, :], other=0.0)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype), allow_tf32=False)
        if K > 64:
            p_k = k + token[:, None] * (H * K) + i_qh * K + k2[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (k2 < K)[None, :], other=0.0)
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype), allow_tf32=False)
        if K > 128:
            p_k = k + token[:, None] * (H * K) + i_qh * K + k3[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (k3 < K)[None, :], other=0.0)
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype), allow_tf32=False)
        if K > 192:
            p_k = k + token[:, None] * (H * K) + i_qh * K + k4[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (k4 < K)[None, :], other=0.0)
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype), allow_tf32=False)
        b_dv *= b_rel[:, None]
        p_dv = dv + token[:, None] * (HV * V) + i_h * V + o_v[None, :]
        b_dv += tl.load(p_dv, mask=m_t[:, None] & m_v[None, :], other=0.0)

        p_do = do + token[:, None] * (HV * V) + i_h * V + o_v[None, :]
        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_decay = exp2(b_g_last)

        p_q = q + token[None, :] * (H * K) + i_qh * K + k1[:, None]
        b_q = tl.load(p_q, mask=(k1 < K)[:, None] & m_t[None, :], other=0.0)
        b_qg = b_q.to(tl.float32) * b_gate[None, :]
        p_w = w + token[None, :] * (HV * K) + i_h * K + k1[:, None]
        b_w = tl.load(p_w, mask=(k1 < K)[:, None] & m_t[None, :], other=0.0)
        b_dh1 *= b_decay
        b_dh1 += _dot_fp32_low_rhs(b_qg, b_do) * scale
        b_dh1 -= tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)
        if K > 64:
            p_q = q + token[None, :] * (H * K) + i_qh * K + k2[:, None]
            b_q = tl.load(p_q, mask=(k2 < K)[:, None] & m_t[None, :], other=0.0)
            b_qg = b_q.to(tl.float32) * b_gate[None, :]
            p_w = w + token[None, :] * (HV * K) + i_h * K + k2[:, None]
            b_w = tl.load(p_w, mask=(k2 < K)[:, None] & m_t[None, :], other=0.0)
            b_dh2 *= b_decay
            b_dh2 += _dot_fp32_low_rhs(b_qg, b_do) * scale
            b_dh2 -= tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)
        if K > 128:
            p_q = q + token[None, :] * (H * K) + i_qh * K + k3[:, None]
            b_q = tl.load(p_q, mask=(k3 < K)[:, None] & m_t[None, :], other=0.0)
            b_qg = b_q.to(tl.float32) * b_gate[None, :]
            p_w = w + token[None, :] * (HV * K) + i_h * K + k3[:, None]
            b_w = tl.load(p_w, mask=(k3 < K)[:, None] & m_t[None, :], other=0.0)
            b_dh3 *= b_decay
            b_dh3 += _dot_fp32_low_rhs(b_qg, b_do) * scale
            b_dh3 -= tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)
        if K > 192:
            p_q = q + token[None, :] * (H * K) + i_qh * K + k4[:, None]
            b_q = tl.load(p_q, mask=(k4 < K)[:, None] & m_t[None, :], other=0.0)
            b_qg = b_q.to(tl.float32) * b_gate[None, :]
            p_w = w + token[None, :] * (HV * K) + i_h * K + k4[:, None]
            b_w = tl.load(p_w, mask=(k4 < K)[:, None] & m_t[None, :], other=0.0)
            b_dh4 *= b_decay
            b_dh4 += _dot_fp32_low_rhs(b_qg, b_do) * scale
            b_dh4 -= tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=False)

    dhm_base = (i_h * K * (V + K)).to(tl.int64)
    p_dh = dhm + dhm_base + k1[:, None] * (V + K) + o_v[None, :]
    tl.store(p_dh, b_dh1, mask=(k1 < K)[:, None] & m_v[None, :])
    if K > 64:
        p_dh = dhm + dhm_base + k2[:, None] * (V + K) + o_v[None, :]
        tl.store(p_dh, b_dh2, mask=(k2 < K)[:, None] & m_v[None, :])
    if K > 128:
        p_dh = dhm + dhm_base + k3[:, None] * (V + K) + o_v[None, :]
        tl.store(p_dh, b_dh3, mask=(k3 < K)[:, None] & m_v[None, :])
    if K > 192:
        p_dh = dhm + dhm_base + k4[:, None] * (V + K) + o_v[None, :]
        tl.store(p_dh, b_dh4, mask=(k4 < K)[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=['BOS', 'SEGMENT_T', 'NT'])
def _cp_gdn_bwd_m_kernel(
    k,
    w,
    g,
    dhm,
    BOS,
    SEGMENT_T,
    NT,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    NM: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    task = tl.program_id(0) + TASK_OFFSET
    i_h = task // NM
    i_m = task - i_h * NM
    i_kh = i_h // (HV // H)

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, 64)
    o_m = i_m * BM + tl.arange(0, BM)
    m_m = o_m < K
    k1 = o_k
    b_m1 = tl.where(k1[:, None] == o_m[None, :], 1.0, 0.0)
    if K > 64:
        k2 = 64 + o_k
        b_m2 = tl.where(k2[:, None] == o_m[None, :], 1.0, 0.0)
    if K > 128:
        k3 = 128 + o_k
        b_m3 = tl.where(k3[:, None] == o_m[None, :], 1.0, 0.0)
    if K > 192:
        k4 = 192 + o_k
        b_m4 = tl.where(k4[:, None] == o_m[None, :], 1.0, 0.0)

    for step in tl.range(
        0,
        NT,
        loop_unroll_factor=1,
        disallow_acc_multi_buffer=True,
    ):
        i_t = NT - 1 - step
        rel_t = i_t * BT + o_t
        m_t = rel_t < SEGMENT_T
        token = (BOS + rel_t).to(tl.int64)
        last_rel = tl.minimum((i_t + 1) * BT, SEGMENT_T) - 1
        last_token = (BOS + last_rel).to(tl.int64)
        b_g_last = tl.load(g + last_token * HV + i_h).to(tl.float32)
        b_g = tl.load(g + token * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
        b_rel = tl.where(m_t, exp2(b_g_last - b_g), 0.0)
        b_decay = exp2(b_g_last)

        p_k = k + token[:, None] * (H * K) + i_kh * K + k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (k1 < K)[None, :], other=0.0)
        b_kg = (b_k.to(tl.float32) * b_rel[:, None]).to(b_k.dtype)
        b_tmp = tl.dot(b_kg.to(tl.float32), b_m1, allow_tf32=False)
        if K > 64:
            p_k = k + token[:, None] * (H * K) + i_kh * K + k2[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (k2 < K)[None, :], other=0.0)
            b_kg = (b_k.to(tl.float32) * b_rel[:, None]).to(b_k.dtype)
            b_tmp += tl.dot(b_kg.to(tl.float32), b_m2, allow_tf32=False)
        if K > 128:
            p_k = k + token[:, None] * (H * K) + i_kh * K + k3[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (k3 < K)[None, :], other=0.0)
            b_kg = (b_k.to(tl.float32) * b_rel[:, None]).to(b_k.dtype)
            b_tmp += tl.dot(b_kg.to(tl.float32), b_m3, allow_tf32=False)
        if K > 192:
            p_k = k + token[:, None] * (H * K) + i_kh * K + k4[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (k4 < K)[None, :], other=0.0)
            b_kg = (b_k.to(tl.float32) * b_rel[:, None]).to(b_k.dtype)
            b_tmp += tl.dot(b_kg.to(tl.float32), b_m4, allow_tf32=False)

        p_w = w + token[:, None] * (HV * K) + i_h * K + k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & (k1 < K)[None, :], other=0.0)
        b_m1 = b_decay * b_m1 - tl.dot(tl.trans(b_w.to(tl.float32)), b_tmp, allow_tf32=False)
        if K > 64:
            p_w = w + token[:, None] * (HV * K) + i_h * K + k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (k2 < K)[None, :], other=0.0)
            b_m2 = b_decay * b_m2 - tl.dot(tl.trans(b_w.to(tl.float32)), b_tmp, allow_tf32=False)
        if K > 128:
            p_w = w + token[:, None] * (HV * K) + i_h * K + k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (k3 < K)[None, :], other=0.0)
            b_m3 = b_decay * b_m3 - tl.dot(tl.trans(b_w.to(tl.float32)), b_tmp, allow_tf32=False)
        if K > 192:
            p_w = w + token[:, None] * (HV * K) + i_h * K + k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (k4 < K)[None, :], other=0.0)
            b_m4 = b_decay * b_m4 - tl.dot(tl.trans(b_w.to(tl.float32)), b_tmp, allow_tf32=False)

    dhm_base = (i_h * K * (V + K) + V).to(tl.int64)
    p_m = dhm + dhm_base + k1[:, None] * (V + K) + o_m[None, :]
    tl.store(p_m, b_m1, mask=(k1 < K)[:, None] & m_m[None, :])
    if K > 64:
        p_m = dhm + dhm_base + k2[:, None] * (V + K) + o_m[None, :]
        tl.store(p_m, b_m2, mask=(k2 < K)[:, None] & m_m[None, :])
    if K > 128:
        p_m = dhm + dhm_base + k3[:, None] * (V + K) + o_m[None, :]
        tl.store(p_m, b_m3, mask=(k3 < K)[:, None] & m_m[None, :])
    if K > 192:
        p_m = dhm + dhm_base + k4[:, None] * (V + K) + o_m[None, :]
        tl.store(p_m, b_m4, mask=(k4 < K)[:, None] & m_m[None, :])


@triton.jit(do_not_specialize=['BOS', 'SEGMENT_T', 'NT', 'scale'])
def _cp_gdn_bwd_fused_128_kernel(
    q,
    k,
    w,
    do,
    dv,
    g,
    gate_rel,
    gate_abs,
    gate_decay,
    dhm,
    BOS,
    SEGMENT_T,
    NT,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    BT: tl.constexpr,
    PRECOMPUTED_GATE: tl.constexpr,
    A800_PRECISION: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    """Compute dH and dM together for the K=V=128 critical path."""
    task = tl.program_id(0) + TASK_OFFSET
    i_h = task // 2
    i_c = task - i_h * 2
    i_qh = i_h // (HV // H)

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, 64)
    k1 = o_k
    k2 = 64 + o_k
    o_c = i_c * 64 + o_k

    b_dh1 = tl.zeros([64, 64], dtype=tl.float32)
    b_dh2 = tl.zeros([64, 64], dtype=tl.float32)
    b_m1 = tl.where(k1[:, None] == o_c[None, :], 1.0, 0.0)
    b_m2 = tl.where(k2[:, None] == o_c[None, :], 1.0, 0.0)

    for step in tl.range(
        0,
        NT,
        loop_unroll_factor=1,
        disallow_acc_multi_buffer=True,
    ):
        i_t = NT - 1 - step
        rel_t = i_t * BT + o_t
        m_t = rel_t < SEGMENT_T
        token = (BOS + rel_t).to(tl.int64)
        if PRECOMPUTED_GATE:
            gate_base = i_h.to(tl.int64) * SEGMENT_T
            b_rel = tl.load(gate_rel + gate_base + rel_t, mask=m_t, other=0.0)
            b_gate = tl.load(gate_abs + gate_base + rel_t, mask=m_t, other=0.0)
            b_decay = tl.load(gate_decay + i_h * NT + i_t)
        else:
            last_rel = tl.minimum((i_t + 1) * BT, SEGMENT_T) - 1
            last_token = (BOS + last_rel).to(tl.int64)
            b_g_last = tl.load(g + last_token * HV + i_h).to(tl.float32)
            b_g = tl.load(g + token * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
            b_rel = tl.where(m_t, exp2(b_g_last - b_g), 0.0)
            b_gate = tl.where(m_t, exp2(b_g), 0.0)
            b_decay = exp2(b_g_last)

        p_k = k + token[:, None] * (H * 128) + i_qh * 128 + k1[None, :]
        b_k1 = tl.load(p_k, mask=m_t[:, None], other=0.0)
        p_k = k + token[:, None] * (H * 128) + i_qh * 128 + k2[None, :]
        b_k2 = tl.load(p_k, mask=m_t[:, None], other=0.0)

        b_dv = tl.dot(b_k1, b_dh1.to(b_k1.dtype), allow_tf32=False)
        b_dv += tl.dot(b_k2, b_dh2.to(b_k2.dtype), allow_tf32=False)
        b_dv *= b_rel[:, None]
        p_dv = dv + token[:, None] * (HV * 128) + i_h * 128 + o_c[None, :]
        b_dv += tl.load(p_dv, mask=m_t[:, None], other=0.0)

        b_kg1 = (b_k1.to(tl.float32) * b_rel[:, None]).to(b_k1.dtype)
        b_kg2 = (b_k2.to(tl.float32) * b_rel[:, None]).to(b_k2.dtype)
        if A800_PRECISION:
            b_tmp = tl.dot(b_kg1, b_m1.to(b_kg1.dtype), allow_tf32=False)
            b_tmp += tl.dot(b_kg2, b_m2.to(b_kg2.dtype), allow_tf32=False)
        else:
            b_tmp = tl.dot(b_kg1.to(tl.float32), b_m1, allow_tf32=False)
            b_tmp += tl.dot(b_kg2.to(tl.float32), b_m2, allow_tf32=False)

        p_do = do + token[:, None] * (HV * 128) + i_h * 128 + o_c[None, :]
        b_do = tl.load(p_do, mask=m_t[:, None], other=0.0)

        p_q = q + token[None, :] * (H * 128) + i_qh * 128 + k1[:, None]
        b_q1 = tl.load(p_q, mask=m_t[None, :], other=0.0)
        p_q = q + token[None, :] * (H * 128) + i_qh * 128 + k2[:, None]
        b_q2 = tl.load(p_q, mask=m_t[None, :], other=0.0)
        b_qg1 = b_q1.to(tl.float32) * b_gate[None, :]
        b_qg2 = b_q2.to(tl.float32) * b_gate[None, :]

        p_w = w + token[None, :] * (HV * 128) + i_h * 128 + k1[:, None]
        b_w1 = tl.load(p_w, mask=m_t[None, :], other=0.0)
        p_w = w + token[None, :] * (HV * 128) + i_h * 128 + k2[:, None]
        b_w2 = tl.load(p_w, mask=m_t[None, :], other=0.0)

        b_dh1 *= b_decay
        b_dh1 += _dot_fp32_low_rhs(b_qg1, b_do) * scale
        b_dh1 -= tl.dot(b_w1, b_dv.to(b_w1.dtype), allow_tf32=False)
        b_dh2 *= b_decay
        b_dh2 += _dot_fp32_low_rhs(b_qg2, b_do) * scale
        b_dh2 -= tl.dot(b_w2, b_dv.to(b_w2.dtype), allow_tf32=False)

        if A800_PRECISION:
            b_m1 = b_decay * b_m1 - tl.dot(b_w1, b_tmp.to(b_w1.dtype), allow_tf32=False)
            b_m2 = b_decay * b_m2 - tl.dot(b_w2, b_tmp.to(b_w2.dtype), allow_tf32=False)
        else:
            b_m1 = b_decay * b_m1 - tl.dot(b_w1.to(tl.float32), b_tmp, allow_tf32=False)
            b_m2 = b_decay * b_m2 - tl.dot(b_w2.to(tl.float32), b_tmp, allow_tf32=False)

    dhm_base = (i_h * 128 * 256).to(tl.int64)
    p_dh = dhm + dhm_base + k1[:, None] * 256 + o_c[None, :]
    tl.store(p_dh, b_dh1)
    p_dh = dhm + dhm_base + k2[:, None] * 256 + o_c[None, :]
    tl.store(p_dh, b_dh2)
    p_m = dhm + dhm_base + k1[:, None] * 256 + 128 + o_c[None, :]
    tl.store(p_m, b_m1)
    p_m = dhm + dhm_base + k2[:, None] * 256 + 128 + o_c[None, :]
    tl.store(p_m, b_m2)


def _launch_gdn_transition(
    *,
    summary: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    bos: int,
    segment_t: int,
    nt: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    chunk_size: int,
    forward: bool,
    gate_rel: torch.Tensor | None = None,
    gate_decay: torch.Tensor | None = None,
) -> None:
    BM = _matrix_tile_size(K)
    NM = triton.cdiv(K, BM)
    kernel = _cp_gdn_fwd_m_kernel if forward else _cp_gdn_bwd_m_kernel
    output_arg = {'hm': summary} if forward else {'dhm': summary}
    gate_args = {}
    if forward:
        precomputed_gate = gate_rel is not None and gate_decay is not None
        gate_args = {
            'gate_rel': g if gate_rel is None else gate_rel,
            'gate_decay': g if gate_decay is None else gate_decay,
            'PRECOMPUTED_GATE': precomputed_gate,
        }
    _launch_flat(
        kernel,
        HV * NM,
        k=k,
        w=w,
        g=g,
        **gate_args,
        **output_arg,
        BOS=bos,
        SEGMENT_T=segment_t,
        NT=nt,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BM=BM,
        NM=NM,
    )


@triton.jit(do_not_specialize=['SOURCE_RANK'])
def _cp_merge_one_rank_kernel(
    state_in,
    ag_hm,
    state_out,
    SOURCE_RANK,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
    NR: tl.constexpr,
    NV: tl.constexpr,
    ZERO_INPUT: tl.constexpr,
    OUTPUT_V_FIRST: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    task = tl.program_id(0) + TASK_OFFSET
    tiles_per_head = NR * NV
    i_h = task // tiles_per_head
    tile = task - i_h * tiles_per_head
    i_r = tile // NV
    i_v = tile - i_r * NV

    o_r = i_r * BR + tl.arange(0, BR)
    o_v = i_v * BV + tl.arange(0, BV)
    m_r = o_r < K
    m_v = o_v < V
    rank_base = (SOURCE_RANK.to(tl.int64) * HV * K * (V + K) + i_h * K * (V + K))
    p_he = ag_hm + rank_base + o_r[:, None] * (V + K) + o_v[None, :]
    b_out = tl.load(p_he, mask=m_r[:, None] & m_v[None, :], other=0.0).to(tl.float32)

    if not ZERO_INPUT:
        o_k = tl.arange(0, 32)
        b_acc = tl.zeros([BR, BV], dtype=tl.float32)
        for k_start in range(0, K, 32):
            k_idx = k_start + o_k
            m_k = k_idx < K
            p_m = ag_hm + rank_base + o_r[:, None] * (V + K) + V + k_idx[None, :]
            b_m = tl.load(p_m, mask=m_r[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            state_base = (i_h * K * V).to(tl.int64)
            p_state = state_in + state_base + k_idx[:, None] * V + o_v[None, :]
            b_state = tl.load(p_state, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            b_acc += tl.dot(b_m, b_state, allow_tf32=False)
        b_out += b_acc

    if OUTPUT_V_FIRST:
        out_base = (i_h * V * K).to(tl.int64)
        p_out = state_out + out_base + o_v[None, :] * K + o_r[:, None]
    else:
        out_base = (i_h * K * V).to(tl.int64)
        p_out = state_out + out_base + o_r[:, None] * V + o_v[None, :]
    tl.store(p_out, b_out, mask=m_r[:, None] & m_v[None, :])


@triton.jit(do_not_specialize=['SOURCE_START', 'SOURCE_STEP'])
def _cp_merge_rank_chain_kernel(
    ag_hm,
    state_out,
    SOURCE_START,
    SOURCE_STEP,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    NUM_RANKS: tl.constexpr,
    OUTPUT_V_FIRST: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    """Compose the ordered rank chain without intermediate HBM states."""
    task = tl.program_id(0) + TASK_OFFSET
    i_h = task // NV
    i_v = task - i_h * NV

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k = tl.arange(0, 64)
    k1 = o_k
    if K > 64:
        k2 = 64 + o_k

    source_rank = SOURCE_START
    rank_base = (source_rank.to(tl.int64) * HV * K * (V + K) + i_h * K * (V + K))
    p_h = ag_hm + rank_base + k1[:, None] * (V + K) + o_v[None, :]
    b_state1 = tl.load(p_h, mask=(k1 < K)[:, None] & m_v[None, :], other=0.0).to(tl.float32)
    if K > 64:
        p_h = ag_hm + rank_base + k2[:, None] * (V + K) + o_v[None, :]
        b_state2 = tl.load(p_h, mask=(k2 < K)[:, None] & m_v[None, :], other=0.0).to(tl.float32)

    for i_rank in tl.range(
        1,
        NUM_RANKS,
        loop_unroll_factor=1,
        disallow_acc_multi_buffer=True,
    ):
        source_rank = SOURCE_START + i_rank * SOURCE_STEP
        rank_base = (source_rank.to(tl.int64) * HV * K * (V + K) + i_h * K * (V + K))

        p_m = ag_hm + rank_base + k1[:, None] * (V + K) + V + k1[None, :]
        b_m = tl.load(p_m, mask=(k1 < K)[:, None] & (k1 < K)[None, :], other=0.0).to(tl.float32)
        b_next1 = tl.dot(b_m, b_state1, allow_tf32=False)
        if K > 64:
            p_m = ag_hm + rank_base + k1[:, None] * (V + K) + V + k2[None, :]
            b_m = tl.load(p_m, mask=(k1 < K)[:, None] & (k2 < K)[None, :], other=0.0).to(tl.float32)
            b_next1 += tl.dot(b_m, b_state2, allow_tf32=False)
        p_h = ag_hm + rank_base + k1[:, None] * (V + K) + o_v[None, :]
        b_next1 += tl.load(p_h, mask=(k1 < K)[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        if K > 64:
            p_m = ag_hm + rank_base + k2[:, None] * (V + K) + V + k1[None, :]
            b_m = tl.load(p_m, mask=(k2 < K)[:, None] & (k1 < K)[None, :], other=0.0).to(tl.float32)
            b_next2 = tl.dot(b_m, b_state1, allow_tf32=False)
            p_m = ag_hm + rank_base + k2[:, None] * (V + K) + V + k2[None, :]
            b_m = tl.load(p_m, mask=(k2 < K)[:, None] & (k2 < K)[None, :], other=0.0).to(tl.float32)
            b_next2 += tl.dot(b_m, b_state2, allow_tf32=False)
            p_h = ag_hm + rank_base + k2[:, None] * (V + K) + o_v[None, :]
            b_next2 += tl.load(p_h, mask=(k2 < K)[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        b_state1 = b_next1
        if K > 64:
            b_state2 = b_next2

    if OUTPUT_V_FIRST:
        out_base = (i_h * V * K).to(tl.int64)
        p_out = state_out + out_base + o_v[:, None] * K + k1[None, :]
        tl.store(p_out, tl.trans(b_state1), mask=m_v[:, None] & (k1 < K)[None, :])
    else:
        out_base = (i_h * K * V).to(tl.int64)
        p_out = state_out + out_base + k1[:, None] * V + o_v[None, :]
        tl.store(p_out, b_state1, mask=(k1 < K)[:, None] & m_v[None, :])
    if K > 64:
        if OUTPUT_V_FIRST:
            p_out = state_out + out_base + o_v[:, None] * K + k2[None, :]
            tl.store(p_out, tl.trans(b_state2), mask=m_v[:, None] & (k2 < K)[None, :])
        else:
            p_out = state_out + out_base + k2[:, None] * V + o_v[None, :]
            tl.store(p_out, b_state2, mask=(k2 < K)[:, None] & m_v[None, :])


@triton.jit
def _cp_transpose_state_kernel(
    state_in,
    state_out,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    TASK_OFFSET: tl.constexpr,
):
    """Transpose a contiguous ``[HV, K, V]`` state using explicit strides."""
    task = tl.program_id(0) + TASK_OFFSET
    tiles_per_head = NK * NV
    i_h = task // tiles_per_head
    tile = task - i_h * tiles_per_head
    i_k = tile // NV
    i_v = tile - i_k * NV

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask = (o_v < V)[:, None] & (o_k < K)[None, :]
    input_base = (i_h * K * V).to(tl.int64)
    p_in = state_in + input_base + o_k[None, :] * V + o_v[:, None]
    b_state = tl.load(p_in, mask=mask, other=0.0)
    output_base = (i_h * V * K).to(tl.int64)
    p_out = state_out + output_base + o_v[:, None] * K + o_k[None, :]
    tl.store(p_out, b_state, mask=mask)


def _segment_bounds(
    cu_seqlens: torch.LongTensor | None,
    context: FLACPContext,
    *,
    forward: bool,
    fallback_t: int,
) -> tuple[int, int]:
    cu_cpu = context.cu_seqlens_cpu
    if cu_cpu is None and cu_seqlens is not None:
        cu_cpu = cu_seqlens.cpu()
    if cu_cpu is None:
        return 0, fallback_t
    if forward:
        return int(cu_cpu[-2]), int(cu_cpu[-1])
    return int(cu_cpu[0]), int(cu_cpu[1])


@torch.no_grad()
def _local_fwd_summary_torch(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    bg: torch.Tensor | None,
    v: torch.Tensor | None,
    bos: int,
    eos: int,
    chunk_size: int,
) -> torch.Tensor:
    """Correctness fallback for non-GDN modes; GDN never enters here."""
    _, _, H, K = k.shape
    HV, V = u.shape[2], u.shape[-1]
    hm = k.new_zeros(HV, K, V + K, dtype=torch.float32)
    is_dplr = bg is not None
    for i_h in range(HV):
        i_kh = i_h // (HV // H)
        state = torch.zeros(K, V, dtype=torch.float32, device=k.device)
        matrix = torch.eye(K, dtype=torch.float32, device=k.device)
        for start in range(bos, eos, chunk_size):
            stop = min(start + chunk_size, eos)
            kg = k[0, start:stop, i_kh]
            decay = torch.exp2(gk[0, stop - 1, i_h].float())
            if is_dplr:
                w_chunk = w[0, start:stop, i_kh]
                bg_chunk = bg[0, start:stop, i_kh]
                value = u[0, start:stop, i_h].float()
                value += w_chunk.float() @ state.to(w_chunk.dtype).float()
                state *= decay[:, None]
                state += kg.float().T @ v[0, start:stop, i_h].to(kg.dtype).float()
                state += bg_chunk.float().T @ value.to(bg_chunk.dtype).float()
                tmp = w_chunk.float() @ matrix
                matrix = decay[:, None] * matrix + bg_chunk.float().T @ tmp
            else:
                w_chunk = w[0, start:stop, i_h]
                value = u[0, start:stop, i_h].float()
                value -= w_chunk.float() @ state.to(w_chunk.dtype).float()
                state *= decay[:, None]
                state += kg.float().T @ value.to(kg.dtype).float()
                tmp = w_chunk.float() @ matrix
                matrix = decay[:, None] * matrix - kg.float().T @ tmp
        hm[i_h, :, :V] = state
        hm[i_h, :, V:] = matrix
    return hm


@torch.no_grad()
def _local_bwd_summary_torch(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    gk: torch.Tensor,
    bg: torch.Tensor | None,
    scale: float,
    bos: int,
    eos: int,
    chunk_size: int,
) -> torch.Tensor:
    """Correctness fallback for KDA/DPLR backward CP summaries."""
    _, _, H, K = q.shape
    HV, V = do.shape[2], do.shape[-1]
    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    is_dplr = bg is not None
    for i_h in range(HV):
        i_qh = i_h // (HV // H)
        state = torch.zeros(K, V, dtype=torch.float32, device=q.device)
        matrix = torch.eye(K, dtype=torch.float32, device=q.device)
        starts = range(bos, eos, chunk_size)
        for start in reversed(tuple(starts)):
            stop = min(start + chunk_size, eos)
            q_chunk = q[0, start:stop, i_qh]
            w_chunk = w[0, start:stop, i_qh if is_dplr else i_h]
            decay = torch.exp2(gk[0, stop - 1, i_h].float())
            do_chunk = do[0, start:stop, i_h]
            if is_dplr:
                bg_chunk = bg[0, start:stop, i_qh]
                value = bg_chunk.float() @ state.to(bg_chunk.dtype).float()
                value += dv[0, start:stop, i_h].float()
                state *= decay[:, None]
                state += q_chunk.float().T @ do_chunk.float()
                state += w_chunk.float().T @ value.to(w_chunk.dtype).float()
                tmp = bg_chunk.float() @ matrix
                matrix = decay[:, None] * matrix + w_chunk.float().T @ tmp
            else:
                kg = k[0, start:stop, i_qh]
                value = kg.float() @ state.to(kg.dtype).float()
                value += dv[0, start:stop, i_h].float()
                state *= decay[:, None]
                state += q_chunk.float().T @ do_chunk.float() * scale
                state -= w_chunk.float().T @ value.to(w_chunk.dtype).float()
                tmp = kg.float() @ matrix
                matrix = decay[:, None] * matrix - w_chunk.float().T @ tmp
        dhm[i_h, :, :V] = state
        dhm[i_h, :, V:] = matrix
    return dhm


def _merge_rank_chain(
    ag_hm: torch.Tensor,
    output: torch.Tensor,
    ranks: tuple[int, ...],
    *,
    state_v_first: bool,
    HV: int,
    K: int,
    V: int,
) -> None:
    if not ranks:
        return
    if K <= 128:
        BV = 32
        NV = triton.cdiv(V, BV)
        source_step = 1 if len(ranks) == 1 else ranks[1] - ranks[0]
        assert all(right - left == source_step for left, right in zip(ranks[:-1], ranks[1:], strict=True))
        merge_output = output
        if state_v_first:
            merge_output = torch.empty((HV, K, V), device=ag_hm.device, dtype=torch.float32)
        _launch_flat(
            _cp_merge_rank_chain_kernel,
            HV * NV,
            ag_hm=ag_hm,
            state_out=merge_output,
            SOURCE_START=ranks[0],
            SOURCE_STEP=source_step,
            HV=HV,
            K=K,
            V=V,
            BV=BV,
            NV=NV,
            NUM_RANKS=len(ranks),
            OUTPUT_V_FIRST=False,
        )
        if state_v_first:
            BK = 32
            NK = triton.cdiv(K, BK)
            _launch_flat(
                _cp_transpose_state_kernel,
                HV * NK * NV,
                state_in=merge_output,
                state_out=output,
                HV=HV,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                NK=NK,
                NV=NV,
            )
        return
    BR = 16
    BV = 16
    NR = triton.cdiv(K, BR)
    NV = triton.cdiv(V, BV)
    scratch_a = torch.empty((HV, K, V), device=ag_hm.device, dtype=torch.float32)
    scratch_b = torch.empty_like(scratch_a) if len(ranks) > 2 else None
    state_in = ag_hm

    for index, source_rank in enumerate(ranks):
        is_last = index == len(ranks) - 1
        if is_last:
            state_out = output
        elif index % 2 == 0:
            state_out = scratch_a
        else:
            state_out = scratch_b
        _launch_flat(
            _cp_merge_one_rank_kernel,
            HV * NR * NV,
            state_in=state_in,
            ag_hm=ag_hm,
            state_out=state_out,
            SOURCE_RANK=source_rank,
            HV=HV,
            K=K,
            V=V,
            BR=BR,
            BV=BV,
            NR=NR,
            NV=NV,
            ZERO_INPUT=index == 0,
            OUTPUT_V_FIRST=is_last and state_v_first,
        )
        state_in = state_out


def chunk_gated_delta_rule_fwd_h_pre_process_npu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
) -> torch.Tensor | None:
    if context is None or context.group is None:
        return initial_state
    is_gdn = g is not None and gk is None and bg is None
    is_kda = g is None and gk is not None and bg is None
    is_dplr = g is None and gk is not None and bg is not None and v is not None
    if not (is_gdn or is_kda or is_dplr):
        raise ValueError('Unsupported Ascend CP gate combination.')
    assert initial_state is None, 'When enable CP, the provided initial_state must be None.'
    if not dist.is_initialized():
        raise RuntimeError('CP requires an initialized process group')

    rank = dist.get_rank(group=context.group)
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    del B
    assert K <= 256, 'current kernel does not support head dimension larger than 256.'
    N = 1 if cu_seqlens is None else len(cu_seqlens) - 1
    if context.is_last_rank:
        hm = k.new_zeros(HV, K, V + K, dtype=torch.float32)
    else:
        hm = k.new_empty(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        output_shape = (N, HV, V, K)
    else:
        output_shape = (N, HV, K, V)
    if context.is_first_rank or N != 1:
        output = k.new_zeros(output_shape, dtype=torch.float32)
    else:
        output = k.new_empty(output_shape, dtype=torch.float32)

    if not context.is_last_rank:
        bos, eos = _segment_bounds(cu_seqlens, context, forward=True, fallback_t=T)
        segment_t = eos - bos
        nt = triton.cdiv(segment_t, chunk_size)
        if is_gdn:
            current_stream = device_torch_lib.current_stream(k.device)
            transition_stream = _summary_stream(k)
            precomputed_gate = K == 128 and V == 128 and k.dtype in (torch.bfloat16, torch.float16)
            if precomputed_gate:
                gate_rel = torch.empty((segment_t, HV), device=k.device, dtype=torch.float32)
                gate_decay = torch.empty((nt, HV), device=k.device, dtype=torch.float32)
                _launch_flat(
                    _cp_gdn_gate_factors_kernel,
                    HV * nt,
                    g=g,
                    gate_rel=gate_rel,
                    gate_decay=gate_decay,
                    BOS=bos,
                    SEGMENT_T=segment_t,
                    HV=HV,
                    BT=chunk_size,
                    NT=nt,
                )
            else:
                gate_rel = g
                gate_decay = g
            transition_stream.wait_stream(current_stream)
            BV = _value_tile_size(K, V)
            NV = triton.cdiv(V, BV)
            _launch_flat(
                _cp_gdn_fwd_h_kernel,
                HV * NV,
                k=k,
                w=w,
                u=u,
                g=g,
                gate_rel=gate_rel,
                gate_decay=gate_decay,
                hm=hm,
                BOS=bos,
                SEGMENT_T=segment_t,
                NT=nt,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=chunk_size,
                BV=BV,
                NV=NV,
                PRECOMPUTED_GATE=precomputed_gate,
            )
            with device_torch_lib.stream(transition_stream):
                _launch_gdn_transition(
                    summary=hm,
                    k=k,
                    w=w,
                    g=g,
                    bos=bos,
                    segment_t=segment_t,
                    nt=nt,
                    H=H,
                    HV=HV,
                    K=K,
                    V=V,
                    chunk_size=chunk_size,
                    forward=True,
                    gate_rel=gate_rel if precomputed_gate else None,
                    gate_decay=gate_decay if precomputed_gate else None,
                )
            current_stream.wait_stream(transition_stream)
        else:
            hm = _local_fwd_summary_torch(
                k=k,
                w=w,
                u=u,
                gk=gk,
                bg=bg,
                v=v,
                bos=bos,
                eos=eos,
                chunk_size=chunk_size,
            )

    ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
    if not context.is_first_rank:
        ranks = tuple(range(rank - context.pre_num_ranks, rank))
        _merge_rank_chain(
            ag_hm,
            output[0],
            ranks,
            state_v_first=state_v_first,
            HV=HV,
            K=K,
            V=V,
        )
    return output


def chunk_gated_delta_rule_bwd_dhu_pre_process_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor | None, None]:
    del initial_state
    if context is None or context.group is None:
        return dht, None
    is_gdn = g is not None and gk is None and bg is None
    is_kda = g is None and gk is not None and bg is None
    is_dplr = g is None and gk is not None and bg is not None
    if not (is_gdn or is_kda or is_dplr):
        raise ValueError('Unsupported Ascend CP gate combination.')
    precision_mode = _gdn_precision_mode() if is_gdn else 'high'
    assert dht is None, 'When enable CP, the provided dht must be None.'
    if not dist.is_initialized():
        raise RuntimeError('CP requires an initialized process group')

    rank = dist.get_rank(group=context.group)
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    del B
    assert K <= 256, 'current kernel does not support head dimension being larger than 256.'
    N = 1 if cu_seqlens is None else len(cu_seqlens) - 1
    if context.is_first_rank:
        dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    else:
        dhm = q.new_empty(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        output_shape = (N, HV, V, K)
    else:
        output_shape = (N, HV, K, V)
    if context.is_last_rank or N != 1:
        output = q.new_zeros(output_shape, dtype=torch.float32)
    else:
        output = q.new_empty(output_shape, dtype=torch.float32)

    if not context.is_first_rank:
        bos, eos = _segment_bounds(cu_seqlens, context, forward=False, fallback_t=T)
        segment_t = eos - bos
        nt = triton.cdiv(segment_t, chunk_size)
        if is_gdn:
            if K == 128 and V == 128 and q.dtype in (torch.bfloat16, torch.float16):
                gate_rel = torch.empty((HV, segment_t), device=q.device, dtype=torch.float32)
                gate_abs = torch.empty((HV, segment_t), device=q.device, dtype=torch.float32)
                gate_decay = torch.empty((HV, nt), device=q.device, dtype=torch.float32)
                _launch_flat(
                    _cp_gdn_bwd_gate_factors_kernel,
                    HV * nt,
                    g=g,
                    gate_rel=gate_rel,
                    gate_abs=gate_abs,
                    gate_decay=gate_decay,
                    BOS=bos,
                    SEGMENT_T=segment_t,
                    HV=HV,
                    BT=chunk_size,
                    NT=nt,
                )
                _launch_flat(
                    _cp_gdn_bwd_fused_128_kernel,
                    HV * 2,
                    q=q,
                    k=k,
                    w=w,
                    do=do,
                    dv=dv,
                    g=g,
                    gate_rel=gate_rel,
                    gate_abs=gate_abs,
                    gate_decay=gate_decay,
                    dhm=dhm,
                    BOS=bos,
                    SEGMENT_T=segment_t,
                    NT=nt,
                    scale=scale,
                    H=H,
                    HV=HV,
                    BT=chunk_size,
                    PRECOMPUTED_GATE=True,
                    A800_PRECISION=_use_a800_transition_precision(
                        precision_mode=precision_mode,
                        dtype=q.dtype,
                        K=K,
                        V=V,
                        segment_t=segment_t,
                    ),
                )
            else:
                current_stream = device_torch_lib.current_stream(q.device)
                transition_stream = _summary_stream(q)
                transition_stream.wait_stream(current_stream)
                BV = _backward_value_tile_size(K, V)
                NV = triton.cdiv(V, BV)
                _launch_flat(
                    _cp_gdn_bwd_dh_kernel,
                    HV * NV,
                    q=q,
                    k=k,
                    w=w,
                    do=do,
                    dv=dv,
                    g=g,
                    dhm=dhm,
                    BOS=bos,
                    SEGMENT_T=segment_t,
                    NT=nt,
                    scale=scale,
                    H=H,
                    HV=HV,
                    K=K,
                    V=V,
                    BT=chunk_size,
                    BV=BV,
                    NV=NV,
                )
                with device_torch_lib.stream(transition_stream):
                    _launch_gdn_transition(
                        summary=dhm,
                        k=k,
                        w=w,
                        g=g,
                        bos=bos,
                        segment_t=segment_t,
                        nt=nt,
                        H=H,
                        HV=HV,
                        K=K,
                        V=V,
                        chunk_size=chunk_size,
                        forward=False,
                    )
                current_stream.wait_stream(transition_stream)
        else:
            dhm = _local_bwd_summary_torch(
                q=q,
                k=k,
                w=w,
                do=do,
                dv=dv,
                gk=gk,
                bg=bg,
                scale=scale,
                bos=bos,
                eos=eos,
                chunk_size=chunk_size,
            )

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)
    if not context.is_last_rank:
        ranks = tuple(range(rank + context.post_num_ranks, rank, -1))
        _merge_rank_chain(
            ag_dhm,
            output[-1],
            ranks,
            state_v_first=state_v_first,
            HV=HV,
            K=K,
            V=V,
        )
    return output, None
