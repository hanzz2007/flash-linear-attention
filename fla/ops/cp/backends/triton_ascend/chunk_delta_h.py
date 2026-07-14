# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend kernels for GDN context-parallel state preprocessing."""

from __future__ import annotations

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


@triton.jit(do_not_specialize=['BOS', 'SEGMENT_T', 'NT'])
def _cp_gdn_fwd_h_kernel(
    k,
    w,
    u,
    g,
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

        last_rel = tl.minimum((i_t + 1) * BT, SEGMENT_T) - 1
        last_token = (BOS + last_rel).to(tl.int64)
        b_g_last = tl.load(g + last_token * HV + i_h).to(tl.float32)
        b_g = tl.load(g + token * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
        b_rel = tl.where(m_t, exp2(b_g_last - b_g), 0.0)
        b_v *= b_rel[:, None]
        b_decay = exp2(b_g_last)
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
) -> None:
    BM = _matrix_tile_size(K)
    NM = triton.cdiv(K, BM)
    kernel = _cp_gdn_fwd_m_kernel if forward else _cp_gdn_bwd_m_kernel
    output_arg = {'hm': summary} if forward else {'dhm': summary}
    _launch_flat(
        kernel,
        HV * NM,
        k=k,
        w=w,
        g=g,
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
    if K <= 128 and not state_v_first:
        BV = 128
        NV = triton.cdiv(V, BV)
        source_step = 1 if len(ranks) == 1 else ranks[1] - ranks[0]
        assert all(right - left == source_step for left, right in zip(ranks[:-1], ranks[1:], strict=True))
        _launch_flat(
            _cp_merge_rank_chain_kernel,
            HV * NV,
            ag_hm=ag_hm,
            state_out=output,
            SOURCE_START=ranks[0],
            SOURCE_STEP=source_step,
            HV=HV,
            K=K,
            V=V,
            BV=BV,
            NV=NV,
            NUM_RANKS=len(ranks),
            OUTPUT_V_FIRST=state_v_first,
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
    hm = k.new_zeros(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        output = k.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        output = k.new_zeros(N, HV, K, V, dtype=torch.float32)

    if not context.is_last_rank:
        bos, eos = _segment_bounds(cu_seqlens, context, forward=True, fallback_t=T)
        segment_t = eos - bos
        nt = triton.cdiv(segment_t, chunk_size)
        if is_gdn:
            current_stream = device_torch_lib.current_stream(k.device)
            transition_stream = _summary_stream(k)
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
    assert dht is None, 'When enable CP, the provided dht must be None.'
    if not dist.is_initialized():
        raise RuntimeError('CP requires an initialized process group')

    rank = dist.get_rank(group=context.group)
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    del B
    assert K <= 256, 'current kernel does not support head dimension being larger than 256.'
    N = 1 if cu_seqlens is None else len(cu_seqlens) - 1
    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        output = q.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        output = q.new_zeros(N, HV, K, V, dtype=torch.float32)

    if not context.is_first_rank:
        bos, eos = _segment_bounds(cu_seqlens, context, forward=False, fallback_t=T)
        segment_t = eos - bos
        nt = triton.cdiv(segment_t, chunk_size)
        if is_gdn:
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
