# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Independent correctness tests for GDN CP state preprocessing."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fla.ops.cp import build_cp_context
from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_fwd_h_pre_process
from fla.utils import IS_NPU, device, device_torch_lib


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


def _strict_ratio(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    reference = reference.float()
    actual = actual.float()
    max_abs = (reference - actual).abs().max().item()
    rms = (reference - actual).square().mean().sqrt()
    base = reference.square().mean().sqrt()
    return max_abs, (rms / (base + 1e-8)).item()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _forward_worker(rank: int, world_size: int, state_v_first: bool, port: int) -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    if IS_NPU:
        os.environ.setdefault('HCCL_NPU_SOCKET_PORT_RANGE', 'auto')
    device_torch_lib.set_device(rank)
    backend = 'hccl' if IS_NPU else 'nccl'
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
        assert max_abs <= 1e-6 or ratio < 3e-3, (
            f'rank={rank}: max_abs={max_abs:.6g}, ratio={ratio:.6g}'
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('state_v_first', [False, True])
def test_gdn_cp4_forward_preprocess(state_v_first: bool) -> None:
    if device_torch_lib.device_count() < 4:
        pytest.skip('At least four accelerator devices are required')
    mp.start_processes(
        _forward_worker,
        args=(4, state_v_first, _free_port()),
        nprocs=4,
        join=True,
        start_method='spawn',
    )
