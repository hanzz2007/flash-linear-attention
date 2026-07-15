# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Independent correctness tests for GDN CP state preprocessing."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import fla.ops.cp.backends.triton_ascend.chunk_delta_h as cp_backend
from fla.ops.cp import FLACPContext, build_cp_context
from fla.ops.cp.backends.triton_ascend.chunk_delta_h import (
    _backward_value_tile_size,
    _cp_gdn_bwd_dh_kernel,
    _cp_gdn_bwd_fused_128_kernel,
    _cp_gdn_bwd_gate_factors_kernel,
    _cp_gdn_fwd_h_kernel,
    _cp_gdn_gate_factors_kernel,
    _gdn_precision_mode,
    _launch_flat,
    _launch_gdn_transition,
    _merge_rank_chain,
    _use_a800_transition_precision,
    _value_tile_size,
)
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    chunk_gated_delta_rule_fwd_h_pre_process,
)
from fla.utils import IS_NPU, device, device_torch_lib


@dataclass(frozen=True)
class GDNPrimitiveCase:
    name: str
    K: int
    V: int
    chunk_size: int
    segment_t: int
    bos: int
    H: int
    HV: int
    dtype: torch.dtype
    input_scale: float = 0.05
    gate_scale: float = 0.02
    check_states: bool = True
    precision: str = "high"
    force_split: bool = False
    nightly: bool = False
    tags: frozenset[str] = frozenset()


GDN_PRIMITIVE_CASES = (
    GDNPrimitiveCase("minimum", 1, 1, 16, 1, 0, 1, 1, torch.bfloat16, gate_scale=0.0, tags=frozenset({"minimum"})),
    GDNPrimitiveCase("bt-minus-one", 32, 17, 16, 15, 7, 1, 1, torch.bfloat16, tags=frozenset({"tail", "k-lt64"})),
    GDNPrimitiveCase("bt-plus-one-gva", 64, 65, 32, 33, 7, 1, 2, torch.bfloat16, tags=frozenset({"gva", "k64"})),
    GDNPrimitiveCase("fp16-non-power-two", 96, 80, 32, 70, 7, 1, 2, torch.float16, tags=frozenset({"fp16"})),
    GDNPrimitiveCase(
        "fused-bwd-bf16",
        128,
        128,
        64,
        130,
        7,
        2,
        8,
        torch.bfloat16,
        force_split=True,
        tags=frozenset({"k128", "precomputed-gate", "gva", "forced-grid"}),
    ),
    GDNPrimitiveCase(
        "fused-bwd-fp16",
        128,
        128,
        64,
        130,
        7,
        1,
        2,
        torch.float16,
        tags=frozenset({"k128", "precomputed-gate", "fp16"}),
    ),
    GDNPrimitiveCase("k193-branch", 193, 33, 32, 33, 65, 2, 2, torch.bfloat16, tags=frozenset({"k-gt192"})),
    GDNPrimitiveCase("k256-maximum", 256, 64, 16, 38, 7, 1, 2, torch.bfloat16, tags=frozenset({"maximum"})),
    GDNPrimitiveCase(
        "long-serial-scan",
        32,
        48,
        32,
        378,
        7,
        1,
        2,
        torch.bfloat16,
        nightly=True,
        tags=frozenset({"long-scan"}),
    ),
    GDNPrimitiveCase(
        "transition-dot-range",
        128,
        64,
        32,
        70,
        7,
        1,
        2,
        torch.bfloat16,
        input_scale=0.2,
        check_states=False,
        nightly=True,
        tags=frozenset({"range-stress"}),
    ),
    GDNPrimitiveCase("k127-v129", 127, 129, 64, 63, 7, 1, 2, torch.bfloat16, nightly=True, tags=frozenset({"k127"})),
    GDNPrimitiveCase("k129-v17", 129, 17, 64, 64, 7, 1, 2, torch.bfloat16, nightly=True, tags=frozenset({"k129"})),
    GDNPrimitiveCase("k192-v127", 192, 127, 64, 65, 7, 1, 2, torch.bfloat16, nightly=True, tags=frozenset({"k192"})),
    GDNPrimitiveCase("k255-v16", 255, 16, 16, 17, 7, 1, 2, torch.bfloat16, nightly=True, tags=frozenset({"k255"})),
)


def _primitive_case_param(case: GDNPrimitiveCase):
    marks = (pytest.mark.nightly,) if case.nightly else ()
    return pytest.param(case, marks=marks, id=case.name)


def test_gdn_precision_mode_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLA_ASCEND_CP_GDN_PRECISION", raising=False)
    assert _gdn_precision_mode() == "high"
    monkeypatch.setenv("FLA_ASCEND_CP_GDN_PRECISION", "A800")
    assert _gdn_precision_mode() == "a800"
    assert _use_a800_transition_precision(
        precision_mode="a800",
        dtype=torch.bfloat16,
        K=128,
        V=128,
        segment_t=2048,
    )
    assert not _use_a800_transition_precision(
        precision_mode="a800",
        dtype=torch.bfloat16,
        K=128,
        V=128,
        segment_t=130,
    )
    assert not _use_a800_transition_precision(
        precision_mode="a800",
        dtype=torch.float16,
        K=128,
        V=128,
        segment_t=2048,
    )
    monkeypatch.setenv("FLA_ASCEND_CP_GDN_PRECISION", "invalid")
    with pytest.raises(ValueError, match="FLA_ASCEND_CP_GDN_PRECISION"):
        _gdn_precision_mode()


def _reference_scan(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    *,
    end: int,
    chunk_size: int,
) -> torch.Tensor:
    H, HV, K, V = k.shape[2], u.shape[2], k.shape[-1], u.shape[-1]
    state = torch.zeros(HV, K, V, dtype=torch.float32, device=k.device)
    for start in range(0, end, chunk_size):
        stop = min(start + chunk_size, end)
        for i_h in range(HV):
            i_kh = i_h // (HV // H)
            k_chunk = k[0, start:stop, i_kh]
            w_chunk = w[0, start:stop, i_h]
            u_chunk = u[0, start:stop, i_h]
            g_chunk = g[0, start:stop, i_h].float()
            value_decay = w_chunk.float() @ state[i_h].to(w.dtype).float()
            value = u_chunk.float() - value_decay
            value *= torch.exp2(g_chunk[-1] - g_chunk)[:, None]
            state[i_h] *= torch.exp2(g_chunk[-1])
            state[i_h] += k_chunk.float().T @ value.to(k.dtype).float()
    return state


def _reference_local_forward_state(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    *,
    bos: int,
    segment_t: int,
    chunk_size: int,
) -> torch.Tensor:
    H, HV, K, V = k.shape[2], u.shape[2], k.shape[-1], u.shape[-1]
    state = torch.zeros(HV, K, V, dtype=torch.float32, device=k.device)
    for start in range(0, segment_t, chunk_size):
        stop = min(start + chunk_size, segment_t)
        token_slice = slice(bos + start, bos + stop)
        for i_h in range(HV):
            i_kh = i_h // (HV // H)
            k_chunk = k[0, token_slice, i_kh]
            w_chunk = w[0, token_slice, i_h]
            u_chunk = u[0, token_slice, i_h]
            g_chunk = g[0, token_slice, i_h].float()
            value = u_chunk.float() - w_chunk.float() @ state[i_h].to(w.dtype).float()
            value *= torch.exp2(g_chunk[-1] - g_chunk)[:, None]
            state[i_h] *= torch.exp2(g_chunk[-1])
            state[i_h] += k_chunk.float().T @ value.to(k.dtype).float()
    return state


def _strict_ratio(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    reference = reference.float()
    actual = actual.float()
    max_abs = (reference - actual).abs().max().item()
    rms = (reference - actual).square().mean().sqrt()
    base = reference.square().mean().sqrt()
    return max_abs, (rms / (base + 1e-8)).item()


def _reference_transition(
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    *,
    bos: int,
    segment_t: int,
    chunk_size: int,
    backward: bool,
) -> torch.Tensor:
    H, HV, K = k.shape[2], w.shape[2], k.shape[-1]
    result = torch.eye(K, dtype=torch.float32, device=k.device).expand(HV, K, K).clone()
    starts = list(range(0, segment_t, chunk_size))
    if backward:
        starts.reverse()
    for start in starts:
        stop = min(start + chunk_size, segment_t)
        token_slice = slice(bos + start, bos + stop)
        for i_h in range(HV):
            i_kh = i_h // (HV // H)
            k_chunk = k[0, token_slice, i_kh]
            w_chunk = w[0, token_slice, i_h]
            g_chunk = g[0, token_slice, i_h].float()
            relative = torch.exp2(g_chunk[-1] - g_chunk)[:, None]
            k_gated = (k_chunk.float() * relative).to(k.dtype).float()
            if backward:
                tmp = k_gated @ result[i_h]
                update = w_chunk.float().T @ tmp
            else:
                tmp = w_chunk.float() @ result[i_h]
                update = k_gated.T @ tmp
            result[i_h] = torch.exp2(g_chunk[-1]) * result[i_h] - update
    return result


def _reference_backward_state(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor,
    *,
    bos: int,
    segment_t: int,
    chunk_size: int,
    scale: float,
) -> torch.Tensor:
    H, HV, K, V = q.shape[2], do.shape[2], q.shape[-1], do.shape[-1]
    state = torch.zeros(HV, K, V, dtype=torch.float32, device=q.device)
    for start in reversed(range(0, segment_t, chunk_size)):
        stop = min(start + chunk_size, segment_t)
        token_slice = slice(bos + start, bos + stop)
        for i_h in range(HV):
            i_qh = i_h // (HV // H)
            q_chunk = q[0, token_slice, i_qh]
            k_chunk = k[0, token_slice, i_qh]
            w_chunk = w[0, token_slice, i_h]
            do_chunk = do[0, token_slice, i_h]
            dv_chunk = dv[0, token_slice, i_h]
            g_chunk = g[0, token_slice, i_h].float()
            value = k_chunk.float() @ state[i_h].to(k.dtype).float()
            value *= torch.exp2(g_chunk[-1] - g_chunk)[:, None]
            value += dv_chunk.float()
            state[i_h] *= torch.exp2(g_chunk[-1])
            q_gated = q_chunk.float() * torch.exp2(g_chunk)[:, None]
            state[i_h] += scale * q_gated.T @ do_chunk.float()
            state[i_h] -= w_chunk.float().T @ value.to(w.dtype).float()
    return state


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _forward_worker(rank: int, world_size: int, state_v_first: bool, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    visible_var = "ASCEND_RT_VISIBLE_DEVICES" if IS_NPU else "CUDA_VISIBLE_DEVICES"
    visible_devices = os.environ.get(visible_var, "").split(",")
    device_key = visible_devices[rank].strip() if len(visible_devices) > rank else str(rank)
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/fla-triton-cache-{device}-{device_key}"
    if IS_NPU:
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    device_torch_lib.set_device(rank)
    backend = "hccl" if IS_NPU else "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        dtype = torch.bfloat16
        chunk_size = 32
        local_t = 64
        total_t = local_t * world_size
        H, HV, K, V = 1, 2, 32, 48
        device_obj = torch.device(device, rank)

        generator = torch.Generator().manual_seed(20260714)
        k_cpu = torch.randn(1, total_t, H, K, generator=generator, dtype=torch.float32) * 0.05
        w_cpu = torch.randn(1, total_t, HV, K, generator=generator, dtype=torch.float32) * 0.05
        u_cpu = torch.randn(1, total_t, HV, V, generator=generator, dtype=torch.float32) * 0.05
        raw_g = -torch.rand(1, total_t, HV, generator=generator, dtype=torch.float32) * 0.02
        g_cpu = raw_g.view(1, total_t // chunk_size, chunk_size, HV).cumsum(dim=2).view_as(raw_g)

        k_global = k_cpu.to(device_obj, dtype=dtype)
        w_global = w_cpu.to(device_obj, dtype=dtype)
        u_global = u_cpu.to(device_obj, dtype=dtype)
        g_global = g_cpu.to(device_obj, dtype=dtype)
        start, stop = rank * local_t, (rank + 1) * local_t
        cu_cpu = torch.tensor([0, total_t], dtype=torch.long)
        cu_global = cu_cpu.to(device_obj)
        context = build_cp_context(cu_global, dist.group.WORLD, cu_seqlens_cpu=cu_cpu)

        actual = chunk_gated_delta_rule_fwd_h_pre_process(
            k=k_global[:, start:stop],
            w=w_global[:, start:stop],
            u=u_global[:, start:stop],
            g=g_global[:, start:stop],
            chunk_size=chunk_size,
            state_v_first=state_v_first,
            cu_seqlens=context.cu_seqlens,
            context=context,
        )[0]
        reference = _reference_scan(
            k_global,
            w_global,
            u_global,
            g_global,
            end=start,
            chunk_size=chunk_size,
        )
        if state_v_first:
            reference = reference.transpose(-1, -2).contiguous()

        assert torch.isfinite(actual).all().item()
        max_abs, ratio = _strict_ratio(reference, actual)
        # The public recurrence casts the state at every chunk, whereas CP
        # composes its FP32 transition matrices before applying the state.
        assert max_abs <= 1e-6 or ratio < 3e-3, f"rank={rank}: max_abs={max_abs:.6g}, ratio={ratio:.6g}"
    finally:
        dist.destroy_process_group()


@pytest.mark.cp_distributed
@pytest.mark.parametrize("state_v_first", [False, True])
def test_gdn_cp4_forward_preprocess(state_v_first: bool) -> None:
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least four accelerator devices are required")
    mp.start_processes(
        _forward_worker,
        args=(4, state_v_first, _free_port()),
        nprocs=4,
        join=True,
        start_method="spawn",
    )


@pytest.mark.skipif(not IS_NPU, reason="Triton-Ascend primitive test")
@pytest.mark.ascend_npu
@pytest.mark.parametrize("case", [_primitive_case_param(case) for case in GDN_PRIMITIVE_CASES])
def test_gdn_local_summaries_match_independent_reference(
    case: GDNPrimitiveCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover non-zero BOS, tail chunks, GVA, K != V, and poisoned outputs."""
    K, V = case.K, case.V
    chunk_size, segment_t = case.chunk_size, case.segment_t
    input_scale, dtype = case.input_scale, case.dtype
    bos, H, HV = case.bos, case.H, case.HV
    if case.force_split:
        monkeypatch.setattr(cp_backend, "ASCEND_MAX_GRID_DIM", 4)
    total_t = bos + segment_t + 6
    scale = K**-0.5
    device_obj = torch.device(device)

    generator = torch.Generator().manual_seed(20260715)
    q_cpu = torch.randn(1, total_t, H, K, generator=generator) * input_scale
    k_cpu = torch.randn(1, total_t, H, K, generator=generator) * input_scale
    w_cpu = torch.randn(1, total_t, HV, K, generator=generator) * input_scale
    u_cpu = torch.randn(1, total_t, HV, V, generator=generator) * input_scale
    do_cpu = torch.randn(1, total_t, HV, V, generator=generator) * input_scale
    dv_cpu = torch.randn(1, total_t, HV, V, generator=generator) * input_scale
    for i_h in range(H):
        q_cpu[:, :, i_h].add_(i_h * 0.01)
        k_cpu[:, :, i_h].add_(i_h * 0.015)
    for i_h in range(HV):
        w_cpu[:, :, i_h].add_(i_h * 0.005)
        u_cpu[:, :, i_h].add_(i_h * 0.007)
    g_cpu = torch.zeros(1, total_t, HV)
    for start in range(0, segment_t, chunk_size):
        stop = min(start + chunk_size, segment_t)
        values = -torch.rand(1, stop - start, HV, generator=generator) * case.gate_scale
        g_cpu[:, bos + start : bos + stop] = values.cumsum(dim=1)

    q = q_cpu.to(device_obj, dtype=dtype)
    k = k_cpu.to(device_obj, dtype=dtype)
    w = w_cpu.to(device_obj, dtype=dtype)
    u = u_cpu.to(device_obj, dtype=dtype)
    do = do_cpu.to(device_obj, dtype=dtype)
    dv = dv_cpu.to(device_obj, dtype=dtype)
    g = g_cpu.to(device_obj, dtype=dtype)
    nt = (segment_t + chunk_size - 1) // chunk_size

    hm = torch.full((HV, K, V + K), float("nan"), dtype=torch.float32, device=device_obj)
    bv = _value_tile_size(K, V)
    nv = (V + bv - 1) // bv
    _launch_flat(
        _cp_gdn_fwd_h_kernel,
        HV * nv,
        k=k,
        w=w,
        u=u,
        g=g,
        gate_rel=g,
        gate_decay=g,
        hm=hm,
        BOS=bos,
        SEGMENT_T=segment_t,
        NT=nt,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BV=bv,
        NV=nv,
        PRECOMPUTED_GATE=False,
    )
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

    precomputed_hm = None
    if K == 128 and V == 128:
        gate_rel = torch.full((segment_t, HV), float("nan"), dtype=torch.float32, device=device_obj)
        gate_decay = torch.full((nt, HV), float("nan"), dtype=torch.float32, device=device_obj)
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
        assert torch.isfinite(gate_rel).all().item()
        assert torch.isfinite(gate_decay).all().item()
        precomputed_hm = torch.full_like(hm, float("nan"))
        _launch_flat(
            _cp_gdn_fwd_h_kernel,
            HV * nv,
            k=k,
            w=w,
            u=u,
            g=g,
            gate_rel=gate_rel,
            gate_decay=gate_decay,
            hm=precomputed_hm,
            BOS=bos,
            SEGMENT_T=segment_t,
            NT=nt,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=chunk_size,
            BV=bv,
            NV=nv,
            PRECOMPUTED_GATE=True,
        )
        _launch_gdn_transition(
            summary=precomputed_hm,
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
            gate_rel=gate_rel,
            gate_decay=gate_decay,
        )

    ref_h = _reference_local_forward_state(
        k,
        w,
        u,
        g,
        bos=bos,
        segment_t=segment_t,
        chunk_size=chunk_size,
    )
    ref_m = _reference_transition(
        k,
        w,
        g,
        bos=bos,
        segment_t=segment_t,
        chunk_size=chunk_size,
        backward=False,
    )

    dhm = torch.full_like(hm, float("nan"))
    if K == 128 and V == 128:
        _launch_flat(
            _cp_gdn_bwd_fused_128_kernel,
            HV * 2,
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gate_rel=g,
            gate_abs=g,
            gate_decay=g,
            dhm=dhm,
            BOS=bos,
            SEGMENT_T=segment_t,
            NT=nt,
            scale=scale,
            H=H,
            HV=HV,
            BT=chunk_size,
            PRECOMPUTED_GATE=False,
            A800_PRECISION=False,
        )
    else:
        bwd_bv = _backward_value_tile_size(K, V)
        bwd_nv = (V + bwd_bv - 1) // bwd_bv
        _launch_flat(
            _cp_gdn_bwd_dh_kernel,
            HV * bwd_nv,
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
            BV=bwd_bv,
            NV=bwd_nv,
        )
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

    precomputed_dhm = None
    if K == 128 and V == 128:
        bwd_gate_rel = torch.full((HV, segment_t), float("nan"), dtype=torch.float32, device=device_obj)
        gate_abs = torch.full((HV, segment_t), float("nan"), dtype=torch.float32, device=device_obj)
        bwd_gate_decay = torch.full((HV, nt), float("nan"), dtype=torch.float32, device=device_obj)
        _launch_flat(
            _cp_gdn_bwd_gate_factors_kernel,
            HV * nt,
            g=g,
            gate_rel=bwd_gate_rel,
            gate_abs=gate_abs,
            gate_decay=bwd_gate_decay,
            BOS=bos,
            SEGMENT_T=segment_t,
            HV=HV,
            BT=chunk_size,
            NT=nt,
        )
        assert torch.isfinite(bwd_gate_rel).all().item()
        assert torch.isfinite(gate_abs).all().item()
        assert torch.isfinite(bwd_gate_decay).all().item()
        precomputed_dhm = torch.full_like(dhm, float("nan"))
        _launch_flat(
            _cp_gdn_bwd_fused_128_kernel,
            HV * 2,
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gate_rel=bwd_gate_rel,
            gate_abs=gate_abs,
            gate_decay=bwd_gate_decay,
            dhm=precomputed_dhm,
            BOS=bos,
            SEGMENT_T=segment_t,
            NT=nt,
            scale=scale,
            H=H,
            HV=HV,
            BT=chunk_size,
            PRECOMPUTED_GATE=True,
            A800_PRECISION=case.precision == "a800",
        )
    ref_dh = _reference_backward_state(
        q,
        k,
        w,
        do,
        dv,
        g,
        bos=bos,
        segment_t=segment_t,
        chunk_size=chunk_size,
        scale=scale,
    )
    ref_dm = _reference_transition(
        k,
        w,
        g,
        bos=bos,
        segment_t=segment_t,
        chunk_size=chunk_size,
        backward=True,
    )

    assert torch.isfinite(hm).all().item()
    assert torch.isfinite(dhm).all().item()
    checked = (
        ("M", ref_m, hm[:, :, V:]),
        ("dM", ref_dm, dhm[:, :, V:]),
    )
    if precomputed_hm is not None:
        assert torch.isfinite(precomputed_hm).all().item()
        checked += (
            ("precomputed H", ref_h, precomputed_hm[:, :, :V]),
            ("precomputed M", ref_m, precomputed_hm[:, :, V:]),
        )
    if precomputed_dhm is not None:
        assert torch.isfinite(precomputed_dhm).all().item()
        checked += (
            ("precomputed dH", ref_dh, precomputed_dhm[:, :, :V]),
            ("precomputed dM", ref_dm, precomputed_dhm[:, :, V:]),
        )
    if case.check_states:
        checked += (
            ("H", ref_h, hm[:, :, :V]),
            ("dH", ref_dh, dhm[:, :, :V]),
        )
    if case.precision == "a800":
        checked += (
            ("gate parity H", hm[:, :, :V], precomputed_hm[:, :, :V]),
            ("gate parity M", hm[:, :, V:], precomputed_hm[:, :, V:]),
        )
    for name, reference, actual in checked:
        max_abs, ratio = _strict_ratio(reference, actual)
        if case.precision == "a800" and name == "precomputed dM":
            limit = 2.353e-2
        elif case.precision == "a800" and not name.startswith("gate parity"):
            limit = 3e-3
        else:
            limit = 1e-4
        assert max_abs <= 1e-6 or ratio < limit, f"{name}: max_abs={max_abs:.6g}, ratio={ratio:.6g}, limit={limit:.6g}"


def test_gdn_primitive_case_matrix_covers_contract() -> None:
    tags = frozenset().union(*(case.tags for case in GDN_PRIMITIVE_CASES))
    required = {
        "minimum",
        "maximum",
        "tail",
        "gva",
        "fp16",
        "k64",
        "k127",
        "k128",
        "k129",
        "k192",
        "k-gt192",
        "k255",
        "precomputed-gate",
        "forced-grid",
        "range-stress",
    }
    assert required <= tags, f"missing GDN primitive coverage tags: {sorted(required - tags)}"
    assert min(case.K for case in GDN_PRIMITIVE_CASES) == 1
    assert max(case.K for case in GDN_PRIMITIVE_CASES) == 256
    assert any(case.K != case.V for case in GDN_PRIMITIVE_CASES)


@dataclass(frozen=True)
class MergeCase:
    name: str
    ranks: tuple[int, ...]
    HV: int
    K: int
    V: int
    state_v_first: bool
    nightly: bool = False


MERGE_CASES = (
    MergeCase("one-rank", (0,), 1, 32, 17, False),
    MergeCase("fused-vtail-transposed", (0, 1, 2), 2, 128, 129, True),
    MergeCase("fused-reverse", (6, 5, 4, 3, 2, 1, 0), 2, 128, 128, False),
    MergeCase("ping-pong", (0, 1, 2), 2, 129, 65, False),
    MergeCase("maximum-transposed", (0, 1, 2, 3, 4, 5, 6), 1, 256, 64, True, nightly=True),
)


def _merge_case_param(case: MergeCase):
    marks = (pytest.mark.nightly,) if case.nightly else ()
    return pytest.param(case, marks=marks, id=case.name)


def _make_merge_input(case: MergeCase) -> tuple[torch.Tensor, torch.Tensor]:
    world_size = max(case.ranks) + 1
    generator = torch.Generator().manual_seed(20260715 + case.K + case.V)
    summaries = torch.zeros(world_size, case.HV, case.K, case.V + case.K, dtype=torch.float32)
    identity = torch.eye(case.K, dtype=torch.float32)
    for rank in range(world_size):
        summaries[rank, :, :, : case.V] = (
            torch.randn(
                case.HV,
                case.K,
                case.V,
                generator=generator,
            )
            * 0.01
        )
        summaries[rank, :, :, case.V :] = identity + torch.randn(
            case.HV,
            case.K,
            case.K,
            generator=generator,
        ) * (0.0005 * (rank + 1))

    if world_size > 1:
        left = summaries[0, 0, :, case.V :]
        right = summaries[1, 0, :, case.V :]
        assert not torch.allclose(left @ right, right @ left)

    state = torch.zeros(case.HV, case.K, case.V, dtype=torch.float32)
    for rank in case.ranks:
        state = summaries[rank, :, :, case.V :] @ state + summaries[rank, :, :, : case.V]
    return summaries, state


def _run_merge_case(case: MergeCase) -> tuple[torch.Tensor, torch.Tensor]:
    summaries_cpu, reference = _make_merge_input(case)
    summaries = summaries_cpu.to(device)
    output_shape = (case.HV, case.V, case.K) if case.state_v_first else (case.HV, case.K, case.V)
    guard = 37
    sentinel = 9876.0
    storage = torch.full((guard + case.HV * case.K * case.V + guard,), sentinel, dtype=torch.float32, device=device)
    output = storage[guard:-guard].view(output_shape)
    output.fill_(float("nan"))

    _merge_rank_chain(
        summaries,
        output,
        case.ranks,
        state_v_first=case.state_v_first,
        HV=case.HV,
        K=case.K,
        V=case.V,
    )

    assert torch.isfinite(output).all().item()
    assert torch.equal(storage[:guard], torch.full_like(storage[:guard], sentinel))
    assert torch.equal(storage[-guard:], torch.full_like(storage[-guard:], sentinel))
    actual = output.transpose(-1, -2) if case.state_v_first else output
    max_abs, ratio = _strict_ratio(reference.to(device), actual)
    assert max_abs <= 1e-6 or ratio < 1e-4, f"merge: max_abs={max_abs:.6g}, ratio={ratio:.6g}"
    return summaries, output.detach().clone()


@pytest.mark.skipif(not IS_NPU, reason="Triton-Ascend merge test")
@pytest.mark.ascend_npu
@pytest.mark.parametrize("case", [_merge_case_param(case) for case in MERGE_CASES])
def test_gdn_merge_rank_chain_matches_reference(case: MergeCase) -> None:
    _run_merge_case(case)


@pytest.mark.skipif(not IS_NPU, reason="Triton-Ascend grid split test")
@pytest.mark.ascend_npu
def test_gdn_merge_forced_grid_matches_unsplit(monkeypatch: pytest.MonkeyPatch) -> None:
    case = MERGE_CASES[1]
    _, unsplit = _run_merge_case(case)
    monkeypatch.setattr(cp_backend, "ASCEND_MAX_GRID_DIM", 4)
    _, split = _run_merge_case(case)
    max_abs, ratio = _strict_ratio(unsplit, split)
    assert max_abs <= 1e-7 or ratio < 1e-7, f"split merge: max_abs={max_abs:.6g}, ratio={ratio:.6g}"


def test_gdn_merge_empty_rank_chain_keeps_output() -> None:
    summaries = torch.empty(0, 1, 1, 2)
    output = torch.tensor([[[3.0]]])
    _merge_rank_chain(summaries, output, (), state_v_first=False, HV=1, K=1, V=1)
    assert output.item() == 3.0


@pytest.mark.parametrize(
    ("dimensions", "error", "match"),
    [
        ({"H": 1, "HV": 1, "K": 0, "V": 1}, ValueError, "must be positive"),
        ({"H": 1, "HV": 1, "K": 1, "V": 0}, ValueError, "must be positive"),
        ({"H": 2, "HV": 3, "K": 64, "V": 64}, ValueError, "HV to be divisible"),
        ({"H": 1, "HV": 1, "K": 257, "V": 64}, AssertionError, "larger than 256"),
    ],
)
def test_gdn_cp_dimension_validation(dimensions, error, match) -> None:
    with pytest.raises(error, match=match):
        cp_backend._validate_cp_dimensions(**dimensions)


def test_gdn_npu_preprocess_rejects_invalid_gate_combinations() -> None:
    context = FLACPContext(group=object())
    k = torch.empty(1, 1, 1, 1)
    w = torch.empty_like(k)
    u = torch.empty_like(k)
    with pytest.raises(ValueError, match="gate combination"):
        cp_backend.chunk_gated_delta_rule_fwd_h_pre_process_npu(k, w, u, g=None, context=context)
    with pytest.raises(ValueError, match="gate combination"):
        cp_backend.chunk_gated_delta_rule_fwd_h_pre_process_npu(
            k,
            w,
            u,
            g=torch.empty(1, 1, 1),
            gk=torch.empty_like(k),
            context=context,
        )


def test_gdn_npu_preprocess_rejects_external_states() -> None:
    context = FLACPContext(group=object())
    q = k = w = do = dv = torch.empty(1, 1, 1, 1)
    g = torch.empty(1, 1, 1)
    with pytest.raises(AssertionError, match="provided initial_state must be None"):
        cp_backend.chunk_gated_delta_rule_fwd_h_pre_process_npu(
            k,
            w,
            do,
            g=g,
            initial_state=torch.empty(1),
            context=context,
        )
    with pytest.raises(AssertionError, match="provided dht must be None"):
        cp_backend.chunk_gated_delta_rule_bwd_dhu_pre_process_npu(
            q,
            k,
            w,
            do,
            dv,
            g=g,
            dht=torch.empty(1),
            context=context,
        )


def _non_gdn_fwd_reference(
    mode: str,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    bg: torch.Tensor,
    v: torch.Tensor,
    *,
    end: int,
    chunk_size: int,
) -> torch.Tensor:
    H, HV, K, V = k.shape[2], u.shape[2], k.shape[-1], u.shape[-1]
    state = torch.zeros(HV, K, V, dtype=torch.float32, device=k.device)
    for start in range(0, end, chunk_size):
        stop = min(start + chunk_size, end)
        for i_h in range(HV):
            i_kh = i_h // (HV // H)
            kg = k[0, start:stop, i_kh]
            decay = torch.exp2(gk[0, stop - 1, i_h].float())
            if mode == "kda":
                w_chunk = w[0, start:stop, i_h]
                value = u[0, start:stop, i_h].float()
                value -= w_chunk.float() @ state[i_h].to(w_chunk.dtype).float()
                state[i_h] *= decay[:, None]
                state[i_h] += kg.float().T @ value.to(kg.dtype).float()
            else:
                w_chunk = w[0, start:stop, i_kh]
                bg_chunk = bg[0, start:stop, i_kh]
                value = u[0, start:stop, i_h].float()
                value += w_chunk.float() @ state[i_h].to(w_chunk.dtype).float()
                state[i_h] *= decay[:, None]
                state[i_h] += kg.float().T @ v[0, start:stop, i_h].to(kg.dtype).float()
                state[i_h] += bg_chunk.float().T @ value.to(bg_chunk.dtype).float()
    return state


def _non_gdn_bwd_reference(
    mode: str,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    gk: torch.Tensor,
    bg: torch.Tensor,
    *,
    begin: int,
    chunk_size: int,
    scale: float,
) -> torch.Tensor:
    H, HV, K, V = q.shape[2], do.shape[2], q.shape[-1], do.shape[-1]
    state = torch.zeros(HV, K, V, dtype=torch.float32, device=q.device)
    for start in reversed(tuple(range(begin, q.shape[1], chunk_size))):
        stop = min(start + chunk_size, q.shape[1])
        for i_h in range(HV):
            i_qh = i_h // (HV // H)
            q_chunk = q[0, start:stop, i_qh]
            decay = torch.exp2(gk[0, stop - 1, i_h].float())
            do_chunk = do[0, start:stop, i_h]
            if mode == "kda":
                kg = k[0, start:stop, i_qh]
                w_chunk = w[0, start:stop, i_h]
                value = kg.float() @ state[i_h].to(kg.dtype).float()
                value += dv[0, start:stop, i_h].float()
                state[i_h] *= decay[:, None]
                state[i_h] += q_chunk.float().T @ do_chunk.float() * scale
                state[i_h] -= w_chunk.float().T @ value.to(w_chunk.dtype).float()
            else:
                bg_chunk = bg[0, start:stop, i_qh]
                w_chunk = w[0, start:stop, i_qh]
                value = bg_chunk.float() @ state[i_h].to(bg_chunk.dtype).float()
                value += dv[0, start:stop, i_h].float()
                state[i_h] *= decay[:, None]
                state[i_h] += q_chunk.float().T @ do_chunk.float()
                state[i_h] += w_chunk.float().T @ value.to(w_chunk.dtype).float()
    return state


def _non_gdn_worker(rank: int, world_size: int, mode: str, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    visible_var = "ASCEND_RT_VISIBLE_DEVICES" if IS_NPU else "CUDA_VISIBLE_DEVICES"
    visible_devices = os.environ.get(visible_var, "").split(",")
    device_key = visible_devices[rank].strip() if len(visible_devices) > rank else str(rank)
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/fla-triton-cache-{device}-{device_key}"
    if IS_NPU:
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    device_torch_lib.set_device(rank)
    dist.init_process_group("hccl" if IS_NPU else "nccl", rank=rank, world_size=world_size)
    try:
        dtype = torch.bfloat16
        local_t, chunk_size = 64, 32
        total_t = local_t * world_size
        H, HV, K, V = 1, 2, 32, 48
        scale = 0.5
        device_obj = torch.device(device, rank)
        generator = torch.Generator().manual_seed(20260715)

        def randn(*shape):
            return (torch.randn(*shape, generator=generator) * 0.04).to(device_obj, dtype=dtype)

        q = randn(1, total_t, H, K)
        k = randn(1, total_t, H, K)
        u = randn(1, total_t, HV, V)
        do = randn(1, total_t, HV, V)
        dv = randn(1, total_t, HV, V)
        v = randn(1, total_t, HV, V)
        w_heads = HV if mode == "kda" else H
        w = randn(1, total_t, w_heads, K)
        bg = randn(1, total_t, H, K)
        raw_gk = -torch.rand(1, total_t, HV, K, generator=generator) * 0.01
        gk_cpu = raw_gk.view(1, total_t // chunk_size, chunk_size, HV, K).cumsum(dim=2).view_as(raw_gk)
        gk = gk_cpu.to(device_obj, dtype=dtype)
        start, stop = rank * local_t, (rank + 1) * local_t
        cu_cpu = torch.tensor([0, total_t], dtype=torch.long)
        context = build_cp_context(cu_cpu.to(device_obj), dist.group.WORLD, cu_seqlens_cpu=cu_cpu)

        common = {
            "gk": gk[:, start:stop],
            "bg": bg[:, start:stop] if mode == "dplr" else None,
            "cu_seqlens": context.cu_seqlens,
            "context": context,
            "chunk_size": chunk_size,
        }
        actual_fwd = chunk_gated_delta_rule_fwd_h_pre_process(
            k=k[:, start:stop],
            w=w[:, start:stop],
            u=u[:, start:stop],
            v=v[:, start:stop] if mode == "dplr" else None,
            **common,
        )[0]
        reference_fwd = _non_gdn_fwd_reference(
            mode,
            k,
            w,
            u,
            gk,
            bg,
            v,
            end=start,
            chunk_size=chunk_size,
        )
        max_abs, ratio = _strict_ratio(reference_fwd, actual_fwd)
        assert max_abs <= 1e-6 or ratio < 8e-3, f"{mode} fwd rank={rank}: max_abs={max_abs:.6g}, ratio={ratio:.6g}"

        actual_bwd = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q[:, start:stop],
            k=k[:, start:stop],
            w=w[:, start:stop],
            do=do[:, start:stop],
            dv=dv[:, start:stop],
            scale=scale,
            **common,
        )[0][-1]
        reference_bwd = _non_gdn_bwd_reference(
            mode,
            q,
            k,
            w,
            do,
            dv,
            gk,
            bg,
            begin=stop,
            chunk_size=chunk_size,
            scale=scale,
        )
        max_abs, ratio = _strict_ratio(reference_bwd, actual_bwd)
        assert max_abs <= 1e-6 or ratio < 8e-3, f"{mode} bwd rank={rank}: max_abs={max_abs:.6g}, ratio={ratio:.6g}"
    finally:
        dist.destroy_process_group()


@pytest.mark.cp_distributed
@pytest.mark.parametrize("mode", ["kda", "dplr"])
def test_non_gdn_cp4_primitive(mode: str) -> None:
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least four accelerator devices are required")
    mp.start_processes(
        _non_gdn_worker,
        args=(4, mode, _free_port()),
        nprocs=4,
        join=True,
        start_method="spawn",
    )


@pytest.mark.cp_distributed
@pytest.mark.cp8
@pytest.mark.nightly
@pytest.mark.parametrize("mode", ["kda", "dplr"])
def test_non_gdn_cp8_primitive(mode: str) -> None:
    """Smoke the shared KDA/DPLR CP summaries across a seven-rank merge chain."""
    if device_torch_lib.device_count() < 8:
        pytest.skip("At least eight accelerator devices are required")
    mp.start_processes(
        _non_gdn_worker,
        args=(8, mode, _free_port()),
        nprocs=8,
        join=True,
        start_method="spawn",
    )
