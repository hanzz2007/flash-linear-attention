# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Platform-neutral kernel and distributed benchmark for causal Conv1d CP."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import statistics
import time
from contextlib import suppress
from datetime import timedelta

import torch
import torch.distributed as dist

with suppress(ImportError):
    importlib.import_module("torch_npu")

import triton

from fla.modules.convolution import causal_conv1d
from fla.ops.cp import build_cp_context, conv_cp_send_recv_bwd, conv_cp_send_recv_fwd
from fla.utils import IS_NPU, device, device_torch_lib


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("kernel", "cp", "comm"), default="cp")
    parser.add_argument("--mode", choices=("fwd", "fwd_bwd"), default="fwd_bwd")
    parser.add_argument("--total-seq-len", type=int, default=16384)
    parser.add_argument("--lengths", type=str, default=None)
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument("--width", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--activation", choices=("none", "silu"), default="silu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--confirm-samples", type=int, default=20)
    parser.add_argument("--cv-threshold", type=float, default=0.10)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples <= 0 or args.confirm_samples < args.samples or args.cv_threshold <= 0:
        parser.error("require warmup >= 0, samples > 0, confirm-samples >= samples, and cv-threshold > 0")
    return args


def _package_version(name: str) -> str | None:
    with suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(name)
    return None


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _memory_allocated() -> int:
    if hasattr(device_torch_lib, "memory_allocated"):
        return int(device_torch_lib.memory_allocated())
    return 0


def _build_lengths(args: argparse.Namespace) -> list[int]:
    if args.lengths is None:
        return [args.total_seq_len]
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError(f"Invalid lengths: {args.lengths}")
    if sum(lengths) != args.total_seq_len:
        raise ValueError(f"The lengths sum to {sum(lengths)}, expected {args.total_seq_len}")
    return lengths


def _make_inputs(args: argparse.Namespace, local_rank: int, world_size: int, lengths: list[int]):
    if args.total_seq_len % world_size:
        raise ValueError(f"total_seq_len={args.total_seq_len} must be divisible by world_size={world_size}")
    local_seq_len = args.total_seq_len // world_size
    worker_device = torch.device(device, local_rank)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed + local_rank)
    x = torch.randn(1, local_seq_len, args.dim, device=worker_device, dtype=dtype, requires_grad=True)
    weight = torch.randn(args.dim, args.width, device=worker_device, dtype=dtype, requires_grad=True)
    bias = torch.randn(args.dim, device=worker_device, dtype=dtype, requires_grad=True)
    do = torch.randn_like(x)
    initial_state = torch.randn(1, args.dim, args.width, device=worker_device, dtype=dtype, requires_grad=True)
    cu_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long)
    cu_global = cu_cpu.to(worker_device)
    cp_context = build_cp_context(
        cu_global,
        group=dist.group.WORLD,
        conv1d_kernel_size=args.width,
        cu_seqlens_cpu=cu_cpu,
    )
    return (x, weight, bias, initial_state), do, cp_context


def _run_once(args: argparse.Namespace, inputs, do: torch.Tensor | None, cp_context) -> torch.Tensor:
    if args.kind == "comm":
        halo = inputs[0]
        output = conv_cp_send_recv_fwd(halo, dist.group.WORLD)
        if args.mode == "fwd_bwd":
            output = output + conv_cp_send_recv_bwd(halo, dist.group.WORLD)
        return output

    x, weight, bias, initial_state = inputs
    for tensor in inputs:
        tensor.grad = None
    kwargs = {
        "x": x,
        "weight": weight,
        "bias": bias,
        "activation": None if args.activation == "none" else args.activation,
    }
    if args.kind == "cp":
        kwargs["cp_context"] = cp_context
    else:
        kwargs["initial_state"] = initial_state
    output, _ = causal_conv1d(**kwargs)
    if args.mode == "fwd_bwd":
        output.backward(do)
    return output


def _measure(args: argparse.Namespace, inputs, do: torch.Tensor | None, cp_context, worker_device: torch.device):
    device_torch_lib.synchronize()
    dist.barrier()
    compile_start = time.perf_counter_ns()
    output = _run_once(args, inputs, do, cp_context)
    device_torch_lib.synchronize()
    compile_ms = (time.perf_counter_ns() - compile_start) / 1e6
    tensors = [output]
    if args.mode == "fwd_bwd" and args.kind != "comm":
        tensors.extend(tensor.grad for tensor in inputs[:3])
    finite = torch.tensor(
        float(all(tensor is not None and torch.isfinite(tensor).all().item() for tensor in tensors)),
        device=worker_device,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise AssertionError("Conv1d benchmark produced a non-finite output or gradient")

    for _ in range(args.warmup):
        _run_once(args, inputs, do, cp_context)
    device_torch_lib.synchronize()
    dist.barrier()

    if hasattr(device_torch_lib, "reset_peak_memory_stats"):
        device_torch_lib.reset_peak_memory_stats()
    allocated_before = _memory_allocated()
    samples: list[float] = []
    last_output = output

    def take_sample() -> None:
        nonlocal last_output
        dist.barrier()
        device_torch_lib.synchronize()
        started = time.perf_counter_ns()
        last_output = _run_once(args, inputs, do, cp_context)
        device_torch_lib.synchronize()
        local_ms = (time.perf_counter_ns() - started) / 1e6
        critical_ms = torch.tensor(local_ms, dtype=torch.float32, device=worker_device)
        dist.all_reduce(critical_ms, op=dist.ReduceOp.MAX)
        samples.append(critical_ms.item())

    for _ in range(args.samples):
        take_sample()
    mean_ms = statistics.fmean(samples)
    cv = statistics.pstdev(samples) / mean_ms if mean_ms else 0.0
    if cv > args.cv_threshold and len(samples) < args.confirm_samples:
        for _ in range(args.confirm_samples - len(samples)):
            take_sample()

    tensors = [last_output]
    if args.mode == "fwd_bwd" and args.kind != "comm":
        tensors.extend(tensor.grad for tensor in inputs[:3])
    finite = torch.tensor(
        float(all(tensor is not None and torch.isfinite(tensor).all().item() for tensor in tensors)),
        device=worker_device,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise AssertionError("Conv1d benchmark became non-finite during measured iterations")

    peak_memory = 0
    if hasattr(device_torch_lib, "max_memory_allocated"):
        peak_memory = int(device_torch_lib.max_memory_allocated())
    peak = torch.tensor(peak_memory, dtype=torch.int64, device=worker_device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    memory_growth = torch.tensor(max(0, _memory_allocated() - allocated_before), dtype=torch.int64, device=worker_device)
    dist.all_reduce(memory_growth, op=dist.ReduceOp.MAX)
    compile_time = torch.tensor(compile_ms, dtype=torch.float32, device=worker_device)
    dist.all_reduce(compile_time, op=dist.ReduceOp.MAX)
    return samples, int(peak.item()), compile_time.item(), int(memory_growth.item())


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    visible_var = "ASCEND_RT_VISIBLE_DEVICES" if IS_NPU else "CUDA_VISIBLE_DEVICES"
    visible_devices = os.environ.get(visible_var, "").split(",")
    device_key = visible_devices[local_rank].strip() if len(visible_devices) > local_rank else str(local_rank)
    cache_root = os.environ.get("FLA_BENCH_CACHE_ROOT", "/tmp")
    os.environ.setdefault("TRITON_CACHE_DIR", f"{cache_root}/fla-triton-cache-{device}-{device_key}")
    if IS_NPU:
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "600")
    backend = "hccl" if IS_NPU else "nccl"
    device_torch_lib.set_device(local_rank)
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=20))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    worker_device = torch.device(device, local_rank)

    lengths = _build_lengths(args)
    if args.kind == "comm":
        dtype = getattr(torch, args.dtype)
        halo = torch.randn(args.width - 1, args.dim, device=worker_device, dtype=dtype)
        inputs, do, cp_context = (halo,), None, None
    else:
        inputs, do, cp_context = _make_inputs(args, local_rank, world_size, lengths)
    samples, peak_memory, compile_ms, memory_growth = _measure(args, inputs, do, cp_context, worker_device)

    if rank == 0:
        mean_ms = statistics.fmean(samples)
        std_ms = statistics.pstdev(samples)
        median_ms = statistics.median(samples)
        result = {
            "backend": backend,
            "device": device_torch_lib.get_device_name(local_rank),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "triton_ascend": _package_version("triton-ascend"),
            "world_size": world_size,
            "kind": args.kind,
            "mode": args.mode,
            "dtype": args.dtype,
            "dim": args.dim,
            "width": args.width,
            "activation": args.activation,
            "total_seq_len": args.total_seq_len,
            "local_seq_len": args.total_seq_len // world_size,
            "lengths": lengths,
            "seed": args.seed,
            "warmup": args.warmup,
            "requested_samples": args.samples,
            "samples": len(samples),
            "cv_threshold": args.cv_threshold,
            "compile_ms": compile_ms,
            "latency_ms": {
                "min": min(samples),
                "p20": _percentile(samples, 20),
                "median": median_ms,
                "mean": mean_ms,
                "std": std_ms,
                "p80": _percentile(samples, 80),
                "max": max(samples),
                "cv": std_ms / mean_ms if mean_ms else 0.0,
            },
            "global_tokens_per_second": args.total_seq_len / (median_ms / 1e3),
            "peak_memory_bytes": peak_memory,
            "memory_growth_bytes": memory_growth,
        }
        print(f"CONV_CP_BENCH_RESULT={json.dumps(result, sort_keys=True)}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
