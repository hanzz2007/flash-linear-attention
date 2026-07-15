# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Test for Context Parallel (CP) Gated Delta Rule (GDN)

Implementation Hierarchy and Relationships:
==========================================

1. chunk_gated_delta_rule (fla/ops/gated_delta_rule/chunk.py):
   - Production Triton kernel for GDN
   - Input g is per-token log-space decay, shape [B, T, H] (scalar per head, NOT per-dim)
   - Internally does chunk_local_cumsum on g
   - Recurrence (delta rule with gating):
       S_t = S_{t-1} * exp(g_t) + beta_t * k_t (x) (v_t - S_{t-1} @ k_t)
       o_t = q_t^T @ S_t
     where (x) denotes outer product, S is [K, V] state matrix
   - No fused gate activation (unlike KDA), no lowerbound
   - Supports variable-length sequences via cu_seqlens
   - Supports context parallel via cp_context

2. Context Parallel (CP) chunk_gated_delta_rule:
   - Extension for multi-GPU distributed training
   - Sequence is partitioned across ranks, with state communication between ranks
   - Uses build_cp_context() to manage cross-rank dependencies
   - Forward: Non-first ranks receive initial_state from previous rank
   - Backward: Gradient dht flows back across rank boundaries

Test Architecture Notes:
========================
- Reference: single-GPU chunk_gated_delta_rule with cu_seqlens (varlen, no CP)
- CP path: same function with cp_context, sequence split across ranks
- Both should produce identical results

Differences from KDA test:
- No use_gate_in_kernel / safe_gate / lower_bound / A_log / dt_bias
- No L2 normalization (q/k are pre-normalized before input)
- g shape is [B, T, H] not [B, T, H, D]

Context Parallel Principle:
===========================

With Context Parallel:
1. Sequence Partitioning: input sequence split across ranks along sequence dim
   - Rank i: tokens [i*T/N, (i+1)*T/N)

2. Forward: each rank computes local chunk; non-first ranks receive state from prev rank

3. Backward: gradients flow back through recurrent state across ranks

Test Scenarios:
===============
1. CP2 with sequence cut in the middle
2. CP2 with sequence boundary aligned
3. CP4 with complex sequence distribution
4. CP4 with single long sequence
"""

import logging
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import IS_NPU, device, device_torch_lib

# Configure logging to see assert_close messages
logging.basicConfig(level=logging.INFO, format='%(message)s')
pytestmark = pytest.mark.cp_distributed


def assert_strict_close(name: str, reference: torch.Tensor, actual: torch.Tensor, ratio: float) -> None:
    """Assert finite results without the CI warning downgrade in ``assert_close``."""
    assert torch.isfinite(reference).all().item(), f'{name}: non-finite reference'
    assert torch.isfinite(actual).all().item(), f'{name}: non-finite result'
    reference = reference.detach().float()
    actual = actual.detach().float()
    max_abs = (reference - actual).abs().max().item()
    rms = (reference - actual).square().mean().sqrt()
    base = reference.square().mean().sqrt()
    error_ratio = (rms / (base + 1e-8)).item()
    logging.info(f'{name:>16} diff: {max_abs:.6f} ratio: {error_ratio:.6f}')
    assert max_abs <= 1e-6 or error_ratio < ratio, (
        f'{name}: max_abs={max_abs:.6g}, ratio={error_ratio:.6g}, limit={ratio:.6g}'
    )


def init_distributed(rank, world_size, port):
    """Initialize distributed environment for a single process."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    # Triton-Ascend 3.2 can race while spawned ranks publish one cache entry.
    # A persistent directory per physical device is race-free and preserves
    # compiled artifacts across parametrized tests.
    visible_var = 'ASCEND_RT_VISIBLE_DEVICES' if IS_NPU else 'CUDA_VISIBLE_DEVICES'
    visible_devices = os.environ.get(visible_var, '').split(',')
    device_key = visible_devices[rank].strip() if len(visible_devices) > rank else str(rank)
    os.environ['TRITON_CACHE_DIR'] = f'/tmp/fla-triton-cache-{device}-{device_key}'
    if IS_NPU:
        os.environ.setdefault('HCCL_NPU_SOCKET_PORT_RANGE', 'auto')
        # Rank 0 compiles and runs the full-sequence reference before the
        # other ranks reach their first post-reference collective. Keep the
        # documented connect timeout above that one-time compile window.
        os.environ.setdefault('HCCL_CONNECT_TIMEOUT', '600')

    device_torch_lib.set_device(rank)
    dist.init_process_group(
        backend='hccl' if IS_NPU else 'nccl',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=20),
    )


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_cp_gdn_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    T: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype,
    state_v_first: bool = False,
    Hq: int | None = None,
    Dv: int | None = None,
    op_chunk_size: int = 64,
    fused_inputs: bool = False,
    port: int = 29502,
):
    """
    Worker function for CP GDN test.
    Runs in a spawned process with the given rank.
    """
    try:
        # Distributed correctness accepts only the high-precision path. Keep
        # compatibility selectors out of device execution and timing.
        os.environ["FLA_ASCEND_CP_GDN_PRECISION"] = "high"
        init_distributed(rank, world_size, port)
        worker_device = torch.device(device, rank)
        if IS_NPU:
            from fla.ops.cp.backends.triton_ascend.chunk_delta_h import _gdn_precision_mode

            assert _gdn_precision_mode() == "high"

        assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
        assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Test: {test_name}")
            print(f"Config: T={T}, H={H}, D={D}, Hq={Hq}, world_size={world_size}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'='*60}")

        # Step 1: Prepare Global Data (all generated on rank 0, broadcast to all)
        B = 1
        Hq_actual = Hq if Hq is not None else H
        Dv_actual = Dv if Dv is not None else D
        q_global = torch.empty(B, T, Hq_actual, D, device=worker_device, dtype=dtype)
        k_global = torch.empty(B, T, Hq_actual, D, device=worker_device, dtype=dtype)
        v_global = torch.empty(B, T, H, Dv_actual, device=worker_device, dtype=dtype)
        g_global = torch.empty(B, T, H, device=worker_device, dtype=dtype)
        beta_global = torch.empty(B, T, H, device=worker_device, dtype=torch.float32)
        do_global = torch.empty(B, T, H, Dv_actual, device=worker_device, dtype=dtype)
        A_log = torch.empty(H, device=worker_device, dtype=torch.float32)
        dt_bias = torch.empty(H, device=worker_device, dtype=torch.float32)

        if rank == 0:
            torch.manual_seed(42)
            q_data = torch.randn(B, T, Hq_actual, D, device=worker_device, dtype=torch.float32)
            k_data = torch.randn(B, T, Hq_actual, D, device=worker_device, dtype=torch.float32)
            if not fused_inputs:
                q_data = F.normalize(q_data, p=2, dim=-1)
                k_data = F.normalize(k_data, p=2, dim=-1)
            q_global.copy_(q_data.to(dtype))
            k_global.copy_(k_data.to(dtype))
            v_global.copy_(torch.randn(B, T, H, Dv_actual, device=worker_device, dtype=dtype))
            g_data = torch.randn(B, T, H, device=worker_device, dtype=dtype)
            beta_data = torch.randn(B, T, H, device=worker_device, dtype=torch.float32)
            g_global.copy_(g_data if fused_inputs else F.logsigmoid(g_data))
            beta_global.copy_(beta_data if fused_inputs else beta_data.sigmoid())
            do_global.copy_(torch.randn(B, T, H, Dv_actual, device=worker_device, dtype=dtype))
            A_log.copy_(torch.linspace(-0.3, 0.2, H, device=worker_device))
            dt_bias.copy_(torch.linspace(-0.2, 0.1, H, device=worker_device))

        # Broadcast to ensure all ranks have same data
        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)
        dist.broadcast(g_global, src=0)
        dist.broadcast(beta_global, src=0)
        dist.broadcast(do_global, src=0)
        dist.broadcast(A_log, src=0)
        dist.broadcast(dt_bias, src=0)

        # Prepare cu_seqlens
        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=worker_device, dtype=torch.long)

        # Step 2: Reference Run (single GPU, varlen, no CP)
        # Run the reference on every device. Besides reducing wall time through
        # parallel compilation, this prevents nonzero ranks from entering an
        # HCCL collective minutes before rank 0 on cold-cache configurations.
        q_ref = q_global.clone().detach().requires_grad_(True)
        k_ref = k_global.clone().detach().requires_grad_(True)
        v_ref = v_global.clone().detach().requires_grad_(True)
        g_ref = g_global.clone().detach().requires_grad_(True)
        beta_ref = beta_global.clone().detach().requires_grad_(True)

        o_ref, _ = chunk_gated_delta_rule(
            q=q_ref,
            k=k_ref,
            v=v_ref,
            g=g_ref,
            beta=beta_ref,
            cu_seqlens=cu_seqlens_global,
            state_v_first=state_v_first,
            chunk_size=op_chunk_size,
            use_qk_l2norm_in_kernel=fused_inputs,
            use_gate_in_kernel=fused_inputs,
            A_log=A_log if fused_inputs else None,
            dt_bias=dt_bias if fused_inputs else None,
            use_beta_sigmoid_in_kernel=fused_inputs,
        )

        o_ref.backward(do_global)

        ref_out = o_ref.detach()
        ref_dq = q_ref.grad.detach()
        ref_dk = k_ref.grad.detach()
        ref_dv = v_ref.grad.detach()
        ref_dg = g_ref.grad.detach()
        ref_db = beta_ref.grad.detach()

        # Step 3: Context Parallel Run
        dist.barrier()

        context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD)

        rank_seq_len = T // world_size
        start_idx = rank * rank_seq_len
        end_idx = (rank + 1) * rank_seq_len

        # Get local slices - note: g is [B, T, H], beta is [B, T, H]
        q_local = q_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        k_local = k_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        v_local = v_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        g_local = g_global[:, start_idx:end_idx].clone().detach().requires_grad_(True)
        beta_local = beta_global[:, start_idx:end_idx].clone().detach().requires_grad_(True)
        do_local = do_global[:, start_idx:end_idx, :].clone()

        print(f"[Rank {rank}] chunk: [{start_idx}, {end_idx}), "
              f"cu_seqlens: {context.cu_seqlens.tolist()}, "
              f"pre_num_ranks: {context.pre_num_ranks}")
        dist.barrier()

        # CP Forward
        o_local, _ = chunk_gated_delta_rule(
            q=q_local,
            k=k_local,
            v=v_local,
            g=g_local,
            beta=beta_local,
            cp_context=context,
            state_v_first=state_v_first,
            chunk_size=op_chunk_size,
            use_qk_l2norm_in_kernel=fused_inputs,
            use_gate_in_kernel=fused_inputs,
            A_log=A_log if fused_inputs else None,
            dt_bias=dt_bias if fused_inputs else None,
            use_beta_sigmoid_in_kernel=fused_inputs,
        )

        # CP Backward
        o_local.backward(do_local)

        # Step 4: Result Aggregation and Verification
        o_gathered = [torch.zeros_like(o_local) for _ in range(world_size)]
        dist.all_gather(o_gathered, o_local)
        o_cp_global = torch.cat(o_gathered, dim=1)

        dq_gathered = [torch.zeros_like(q_local.grad) for _ in range(world_size)]
        dist.all_gather(dq_gathered, q_local.grad)
        dq_cp_global = torch.cat(dq_gathered, dim=1)

        dk_gathered = [torch.zeros_like(k_local.grad) for _ in range(world_size)]
        dist.all_gather(dk_gathered, k_local.grad)
        dk_cp_global = torch.cat(dk_gathered, dim=1)

        dv_gathered = [torch.zeros_like(v_local.grad) for _ in range(world_size)]
        dist.all_gather(dv_gathered, v_local.grad)
        dv_cp_global = torch.cat(dv_gathered, dim=1)

        dg_gathered = [torch.zeros_like(g_local.grad) for _ in range(world_size)]
        dist.all_gather(dg_gathered, g_local.grad)
        dg_cp_global = torch.cat(dg_gathered, dim=1)

        db_gathered = [torch.zeros_like(beta_local.grad) for _ in range(world_size)]
        dist.all_gather(db_gathered, beta_local.grad)
        db_cp_global = torch.cat(db_gathered, dim=1)

        test_passed = True
        if rank == 0:
            print(f"\n[{test_name}] Verifying results...")

            tensors_to_verify = [
                ("Output", ref_out, o_cp_global),
                ("dq", ref_dq, dq_cp_global),
                ("dk", ref_dk, dk_cp_global),
                ("dv", ref_dv, dv_cp_global),
                ("dg", ref_dg, dg_cp_global),
                ("db", ref_db, db_cp_global),
            ]

            try:
                for name, ref, cp in tensors_to_verify:
                    assert_strict_close(name, ref, cp, ratio=3e-3)
                print(f"[{test_name}] Test Passed!\n")
            except AssertionError as e:
                print(f"[{test_name}] Test Failed: {e}\n")
                test_passed = False

        status = torch.tensor(int(test_passed), dtype=torch.int32, device=worker_device)
        dist.broadcast(status, src=0)
        cleanup_distributed()

        if not status.item():
            raise AssertionError(f"Test {test_name} failed on rank {rank}")

    except Exception as e:
        cleanup_distributed()
        raise e


def run_cp_test_with_spawn(
    world_size: int,
    test_name: str,
    T: int,
    H: int,
    D: int,
    lengths: list[int],
    dtype=torch.bfloat16,
    state_v_first: bool = False,
    Hq: int | None = None,
    Dv: int | None = None,
    op_chunk_size: int = 64,
    fused_inputs: bool = False,
):
    """
    Run CP test using torch.multiprocessing.spawn.
    This allows running the test directly with pytest.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.start_processes(
        run_cp_gdn_test_worker,
        args=(
            world_size, test_name, T, H, D, lengths, dtype, state_v_first,
            Hq, Dv, op_chunk_size, fused_inputs, port,
        ),
        nprocs=world_size,
        join=True,
        start_method='spawn',
    )


# ============================================================
# Test Scenario Definitions
# ============================================================

def test_cp2_sequence_cut():
    """CP2: sequences cut across rank boundary."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_SequenceCut",
        T=10240, H=4, D=64,
        lengths=[3000, 4000, 3240],
        dtype=torch.bfloat16,
    )


def test_cp2_boundary_aligned():
    """CP2: sequence boundaries aligned with rank boundaries."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_BoundaryAligned",
        T=10240, H=4, D=128,
        lengths=[5120, 5120],
        dtype=torch.bfloat16,
    )


def test_cp4_complex():
    """CP4: complex sequence distribution, first sequence spans 3 ranks."""
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_Complex",
        T=10240, H=4, D=128,
        lengths=[7000, 3240],
        dtype=torch.bfloat16,
    )


def test_cp4_single_sequence():
    """CP4: single long sequence spanning all ranks."""
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_SingleSequence",
        T=10240, H=4, D=128,
        lengths=[10240],
        dtype=torch.bfloat16,
    )


@pytest.mark.cp8
def test_cp8_single_sequence():
    """CP8 high-precision production target spanning all ranks."""
    if device_torch_lib.device_count() < 8:
        pytest.skip("At least 8 GPUs required")

    run_cp_test_with_spawn(
        world_size=8,
        test_name="CP8_SingleSequence",
        T=16384, H=8, D=128,
        lengths=[16384],
        dtype=torch.bfloat16,
    )


def test_cp2_many_short_sequences():
    """CP2: many short sequences."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ManyShortSequences",
        T=10240, H=4, D=128,
        lengths=[1000, 1500, 2000, 2500, 1240, 1000, 1000],
        dtype=torch.bfloat16,
    )


def test_cp2_gqa_sequence_cut():
    """CP2 GQA: sequences cut across rank boundary, Hq < H."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_GQA_SequenceCut",
        T=10240, H=4, D=64, Hq=2,
        lengths=[3000, 4000, 3240],
        dtype=torch.bfloat16,
    )


def test_cp2_gqa_single_sequence():
    """CP2 GQA: single long sequence with Hq < H."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_GQA_SingleSequence",
        T=10240, H=8, D=64, Hq=2,
        lengths=[10240],
        dtype=torch.bfloat16,
    )


@pytest.mark.skipif(not IS_NPU, reason='Ascend K=256 CP coverage')
def test_cp2_k256_value64_tail():
    """CP2: K=256, K != V, and partial boundary chunks."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_K256_Value64_Tail",
        T=512, H=2, D=256, Dv=64,
        lengths=[301, 211],
        dtype=torch.bfloat16,
    )


def test_cp2_fp16_chunk16_gva_tail():
    """CP2: FP16, BT=16, GVA, K != V, and two partial sequences."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_FP16_Chunk16_GVA_Tail",
        T=1088, H=2, D=96, Hq=1, Dv=80,
        lengths=[701, 387],
        dtype=torch.float16,
        op_chunk_size=16,
    )


def test_cp2_fused_gate_beta_qk_norm():
    """CP2: fused raw gate, beta sigmoid, and q/k normalization paths."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_Fused_Gate_Beta_QKNorm",
        T=2048, H=2, D=64,
        lengths=[1301, 747],
        dtype=torch.bfloat16,
        op_chunk_size=32,
        fused_inputs=True,
    )


# ============================================================
# Transpose State Layout Tests
# ============================================================

def test_cp2_state_v_first():
    """CP2: state_v_first=True with sequence cut."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_TransposeState",
        T=10240, H=4, D=128,
        lengths=[3000, 4000, 3240],
        dtype=torch.bfloat16,
        state_v_first=True,
    )


def test_cp4_state_v_first():
    """CP4: state_v_first=True with single long sequence."""
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_TransposeState",
        T=10240, H=4, D=128,
        lengths=[10240],
        dtype=torch.bfloat16,
        state_v_first=True,
    )


# ============================================================
# Main Entry Point (for torchrun)
# ============================================================

def setup_distributed_torchrun():
    """Initialize distributed environment for torchrun."""
    if 'RANK' not in os.environ:
        return False

    dist.init_process_group(backend='hccl' if IS_NPU else 'nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    device_torch_lib.set_device(local_rank)
    return True
