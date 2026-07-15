# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark only the GDN context-parallel state preprocessing path."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import suppress
from datetime import timedelta

import torch
import torch.distributed as dist

with suppress(ImportError):
    import torch_npu  # noqa: F401

from fla.ops.cp import build_cp_context
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    chunk_gated_delta_rule_fwd_h_pre_process,
)
from fla.utils import IS_NPU, device, device_torch_lib


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-seq-len', type=int, default=16384)
    parser.add_argument('--q-heads', type=int, default=8)
    parser.add_argument('--v-heads', type=int, default=8)
    parser.add_argument('--key-dim', type=int, default=128)
    parser.add_argument('--value-dim', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, choices=(16, 32, 64), default=64)
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    parser.add_argument('--direction', choices=('fwd', 'bwd'), default='fwd')
    parser.add_argument(
        '--component',
        choices=('full', 'local', 'gate', 'h', 'm', 'h-precomputed', 'm-precomputed'),
        default='full',
        help='Use non-full components only for single-device Triton-Ascend diagnostics.',
    )
    parser.add_argument('--state-v-first', action='store_true')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--samples', type=int, default=10)
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _make_inputs(args: argparse.Namespace, local_seq_len: int, local_rank: int):
    dtype = getattr(torch, args.dtype)
    device_obj = torch.device(device, local_rank)
    torch.manual_seed(20260714 + local_rank)

    def randn(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, device=device_obj, dtype=torch.float32) * 0.04).to(dtype)

    q = randn(1, local_seq_len, args.q_heads, args.key_dim)
    k = randn(1, local_seq_len, args.q_heads, args.key_dim)
    w = randn(1, local_seq_len, args.v_heads, args.key_dim)
    u = randn(1, local_seq_len, args.v_heads, args.value_dim)
    do = randn(1, local_seq_len, args.v_heads, args.value_dim)
    dv = randn(1, local_seq_len, args.v_heads, args.value_dim)
    raw_g = -torch.rand(
        1,
        local_seq_len,
        args.v_heads,
        device=device_obj,
        dtype=torch.float32,
    ) * 0.02
    padded_t = ((local_seq_len + args.chunk_size - 1) // args.chunk_size) * args.chunk_size
    if padded_t != local_seq_len:
        raw_g = torch.nn.functional.pad(raw_g, (0, 0, 0, padded_t - local_seq_len))
    g = raw_g.view(1, -1, args.chunk_size, args.v_heads).cumsum(dim=2)
    g = g.view(1, padded_t, args.v_heads)[:, :local_seq_len].to(dtype)
    return q, k, w, u, do, dv, g


@torch.no_grad()
def _run_local_kernel(args: argparse.Namespace, inputs, scratch: torch.Tensor) -> torch.Tensor:
    if not IS_NPU:
        raise RuntimeError('The component diagnostic is Triton-Ascend only')
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
        _summary_stream,
        _use_a800_transition_precision,
        _value_tile_size,
    )

    q, k, w, u, do, dv, g = inputs
    _, local_seq_len, H, K = k.shape
    HV, V = u.shape[2], u.shape[-1]
    nt = (local_seq_len + args.chunk_size - 1) // args.chunk_size
    if args.component in ('local', 'gate', 'h-precomputed', 'm-precomputed'):
        if K != 128 or V != 128 or k.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError('The local component currently covers the K=V=128 fused production path')
        if args.direction == 'fwd':
            current_stream = device_torch_lib.current_stream(k.device)
            transition_stream = _summary_stream(k)
            gate_rel = torch.empty((local_seq_len, HV), device=k.device, dtype=torch.float32)
            gate_decay = torch.empty((nt, HV), device=k.device, dtype=torch.float32)
            _launch_flat(
                _cp_gdn_gate_factors_kernel,
                HV * nt,
                g=g,
                gate_rel=gate_rel,
                gate_decay=gate_decay,
                BOS=0,
                SEGMENT_T=local_seq_len,
                HV=HV,
                BT=args.chunk_size,
                NT=nt,
            )
            if args.component == 'gate':
                return scratch
            transition_stream.wait_stream(current_stream)
            if args.component != 'm-precomputed':
                bv = _value_tile_size(K, V)
                nv = (V + bv - 1) // bv
                _launch_flat(
                    _cp_gdn_fwd_h_kernel,
                    HV * nv,
                    k=k,
                    w=w,
                    u=u,
                    g=g,
                    gate_rel=gate_rel,
                    gate_decay=gate_decay,
                    hm=scratch,
                    BOS=0,
                    SEGMENT_T=local_seq_len,
                    NT=nt,
                    H=H,
                    HV=HV,
                    K=K,
                    V=V,
                    BT=args.chunk_size,
                    BV=bv,
                    NV=nv,
                    PRECOMPUTED_GATE=True,
                )
            if args.component != 'h-precomputed':
                with device_torch_lib.stream(transition_stream):
                    _launch_gdn_transition(
                        summary=scratch,
                        k=k,
                        w=w,
                        g=g,
                        bos=0,
                        segment_t=local_seq_len,
                        nt=nt,
                        H=H,
                        HV=HV,
                        K=K,
                        V=V,
                        chunk_size=args.chunk_size,
                        forward=True,
                        gate_rel=gate_rel,
                        gate_decay=gate_decay,
                    )
                current_stream.wait_stream(transition_stream)
        else:
            if args.component not in ('local', 'gate'):
                raise ValueError('Precomputed H/M component diagnostics are forward-only')
            gate_rel = torch.empty((HV, local_seq_len), device=q.device, dtype=torch.float32)
            gate_abs = torch.empty((HV, local_seq_len), device=q.device, dtype=torch.float32)
            gate_decay = torch.empty((HV, nt), device=q.device, dtype=torch.float32)
            _launch_flat(
                _cp_gdn_bwd_gate_factors_kernel,
                HV * nt,
                g=g,
                gate_rel=gate_rel,
                gate_abs=gate_abs,
                gate_decay=gate_decay,
                BOS=0,
                SEGMENT_T=local_seq_len,
                HV=HV,
                BT=args.chunk_size,
                NT=nt,
            )
            if args.component == 'gate':
                return scratch
            precision_mode = _gdn_precision_mode()
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
                dhm=scratch,
                BOS=0,
                SEGMENT_T=local_seq_len,
                NT=nt,
                scale=K**-0.5,
                H=H,
                HV=HV,
                BT=args.chunk_size,
                PRECOMPUTED_GATE=True,
                A800_PRECISION=_use_a800_transition_precision(
                    precision_mode=precision_mode,
                    dtype=q.dtype,
                    K=K,
                    V=V,
                    segment_t=local_seq_len,
                ),
            )
    elif args.component == 'h':
        bv = (
            _value_tile_size(K, V)
            if args.direction == 'fwd'
            else _backward_value_tile_size(K, V)
        )
        nv = (V + bv - 1) // bv
        if args.direction == 'fwd':
            _launch_flat(
                _cp_gdn_fwd_h_kernel,
                HV * nv,
                k=k,
                w=w,
                u=u,
                g=g,
                gate_rel=g,
                gate_decay=g,
                hm=scratch,
                BOS=0,
                SEGMENT_T=local_seq_len,
                NT=nt,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=args.chunk_size,
                BV=bv,
                NV=nv,
                PRECOMPUTED_GATE=False,
            )
        else:
            _launch_flat(
                _cp_gdn_bwd_dh_kernel,
                HV * nv,
                q=q,
                k=k,
                w=w,
                do=do,
                dv=dv,
                g=g,
                dhm=scratch,
                BOS=0,
                SEGMENT_T=local_seq_len,
                NT=nt,
                scale=K**-0.5,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=args.chunk_size,
                BV=bv,
                NV=nv,
            )
    else:
        _launch_gdn_transition(
            summary=scratch,
            k=k,
            w=w,
            g=g,
            bos=0,
            segment_t=local_seq_len,
            nt=nt,
            H=H,
            HV=HV,
            K=K,
            V=V,
            chunk_size=args.chunk_size,
            forward=args.direction == 'fwd',
        )
    return scratch


@torch.no_grad()
def _run_once(args: argparse.Namespace, inputs, context, scratch: torch.Tensor) -> torch.Tensor:
    if args.component != 'full':
        return _run_local_kernel(args, inputs, scratch)
    q, k, w, u, do, dv, g = inputs
    if args.direction == 'fwd':
        return chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            w=w,
            u=u,
            g=g,
            chunk_size=args.chunk_size,
            state_v_first=args.state_v_first,
            cu_seqlens=context.cu_seqlens,
            context=context,
        )
    else:
        return chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            scale=args.key_dim**-0.5,
            state_v_first=args.state_v_first,
            cu_seqlens=context.cu_seqlens,
            context=context,
            chunk_size=args.chunk_size,
        )[0]


def _reset_peak_memory() -> None:
    if hasattr(device_torch_lib, 'reset_peak_memory_stats'):
        device_torch_lib.reset_peak_memory_stats()


def _peak_memory() -> int:
    if hasattr(device_torch_lib, 'max_memory_allocated'):
        return int(device_torch_lib.max_memory_allocated())
    return 0


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    visible_var = 'ASCEND_RT_VISIBLE_DEVICES' if IS_NPU else 'CUDA_VISIBLE_DEVICES'
    visible_devices = os.environ.get(visible_var, '').split(',')
    device_key = visible_devices[local_rank].strip() if len(visible_devices) > local_rank else str(local_rank)
    os.environ.setdefault('TRITON_CACHE_DIR', f'/tmp/fla-triton-cache-{device}-{device_key}')
    if IS_NPU:
        os.environ.setdefault('HCCL_NPU_SOCKET_PORT_RANGE', 'auto')
        from fla.ops.cp.backends.triton_ascend.chunk_delta_h import _gdn_precision_mode

        precision = _gdn_precision_mode()
    else:
        precision = 'cuda'
    device_torch_lib.set_device(local_rank)
    dist.init_process_group(backend='hccl' if IS_NPU else 'nccl', timeout=timedelta(minutes=20))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.total_seq_len % world_size:
        raise ValueError('total_seq_len must be divisible by world_size')
    if args.v_heads % args.q_heads:
        raise ValueError('v_heads must be divisible by q_heads')

    local_seq_len = args.total_seq_len // world_size
    device_obj = torch.device(device, local_rank)
    cu_cpu = torch.tensor([0, args.total_seq_len], dtype=torch.long)
    context = build_cp_context(cu_cpu.to(device_obj), dist.group.WORLD, cu_seqlens_cpu=cu_cpu)
    inputs = _make_inputs(args, local_seq_len, local_rank)
    scratch = torch.zeros(
        args.v_heads,
        args.key_dim,
        args.value_dim + args.key_dim,
        dtype=torch.float32,
        device=device_obj,
    )

    result = _run_once(args, inputs, context, scratch)
    device_torch_lib.synchronize()
    finite = torch.tensor(float(torch.isfinite(result).all().item()), device=device_obj)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise AssertionError('GDN CP preprocessing produced a non-finite result')
    dist.barrier()
    for _ in range(args.warmup):
        _run_once(args, inputs, context, scratch)
    device_torch_lib.synchronize()
    dist.barrier()

    _reset_peak_memory()
    samples = []
    for _ in range(args.samples):
        dist.barrier()
        device_torch_lib.synchronize()
        started = time.perf_counter_ns()
        _run_once(args, inputs, context, scratch)
        device_torch_lib.synchronize()
        local_ms = (time.perf_counter_ns() - started) / 1e6
        critical_ms = torch.tensor(local_ms, dtype=torch.float32, device=device_obj)
        dist.all_reduce(critical_ms, op=dist.ReduceOp.MAX)
        samples.append(critical_ms.item())

    peak_memory = torch.tensor(_peak_memory(), dtype=torch.int64, device=device_obj)
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)

    if rank == 0:
        mean_ms = statistics.fmean(samples)
        median_ms = statistics.median(samples)
        std_ms = statistics.pstdev(samples)
        result = {
            'backend': 'hccl' if IS_NPU else 'nccl',
            'device': device_torch_lib.get_device_name(local_rank),
            'world_size': world_size,
            'direction': args.direction,
            'component': args.component,
            'precision': precision,
            'dtype': args.dtype,
            'total_seq_len': args.total_seq_len,
            'local_seq_len': local_seq_len,
            'q_heads': args.q_heads,
            'v_heads': args.v_heads,
            'key_dim': args.key_dim,
            'value_dim': args.value_dim,
            'chunk_size': args.chunk_size,
            'state_v_first': args.state_v_first,
            'warmup': args.warmup,
            'samples': args.samples,
            'tokens_per_second': args.total_seq_len / (median_ms / 1e3),
            'peak_memory_bytes': int(peak_memory.item()),
            'latency_ms': {
                'min': min(samples),
                'p10': _percentile(samples, 10),
                'median': median_ms,
                'mean': mean_ms,
                'std': std_ms,
                'p90': _percentile(samples, 90),
                'max': max(samples),
                'cv': std_ms / mean_ms if mean_ms else 0.0,
            },
        }
        print(f'GDN_CP_PREPROCESS_RESULT={json.dumps(result, sort_keys=True)}', flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
