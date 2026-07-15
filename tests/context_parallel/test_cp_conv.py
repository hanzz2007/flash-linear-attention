# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Test for Context Parallel (CP) Causal Convolution 1D

Context Parallel Principle for Causal Conv1d:
=============================================

Causal convolution has a dependency on previous tokens due to the sliding window.
In a standard implementation, each rank processes the full sequence sequentially.

With Context Parallel:
1. Sequence Partitioning: The input sequence is split across ranks along the sequence dimension.
   - Rank 0: tokens [0, T/N)
   - Rank 1: tokens [T/N, 2T/N)
   - Rank 2: tokens [2T/N, 3T/N)
   - ...

2. Forward Pass:
   - Each rank processes its local chunk independently
   - Non-first ranks need the last (W-1) tokens from the previous rank as initial_state
   - Communication: Previous rank sends its tail tokens (last W-1 tokens) to current rank
   - Current rank receives and constructs initial_state from previous rank's tail
   - This allows parallel computation while maintaining causal dependencies

3. Backward Pass:
   - Gradients need to be corrected because tokens used as initial_state by next rank
     also contribute to gradients
   - Communication: Current rank sends d_initial_state to previous rank
   - Previous rank adds received gradients to its tail tokens (last W-1 tokens)
   - This ensures gradient correctness across rank boundaries

Key Insight:
- The last (W-1) tokens of each rank are used as initial_state by the next rank
- These tokens need gradient contributions from both local computation and next rank
- Communication overhead is minimal: only (W-1) tokens per rank boundary

Test Scenarios:
===============
1. CP2 with sequence cut in the middle (sequences span across rank boundary)
2. CP2 with sequence boundary aligned (no sequence is cut)
3. CP4 with one sequence spanning 3 ranks, another sequence also cut
4. Single long sequence spanning all ranks
"""

import logging
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fla.modules.convolution import causal_conv1d
from fla.ops.cp import build_cp_context
from fla.utils import IS_NPU, device, device_torch_lib

# Configure logging to see assert_close messages
logging.basicConfig(level=logging.INFO, format="%(message)s")


def assert_strict_close(name: str, reference: torch.Tensor, actual: torch.Tensor, ratio: float = 1e-3) -> None:
    """Assert finite relative-RMS agreement without CI warning downgrade."""
    assert torch.isfinite(reference).all().item(), f"{name}: non-finite reference"
    assert torch.isfinite(actual).all().item(), f"{name}: non-finite result"
    reference = reference.detach().float()
    actual = actual.detach().float()
    max_abs = (reference - actual).abs().max().item()
    rms = (reference - actual).square().mean().sqrt()
    base = reference.square().mean().sqrt()
    error_ratio = (rms / (base + 1e-8)).item()
    assert max_abs <= 1e-6 or error_ratio < ratio, f"{name}: max_abs={max_abs:.6g}, ratio={error_ratio:.6g}, limit={ratio:.6g}"


def causal_conv1d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Independent FP32 causal depthwise convolution for packed sequences."""
    _, _, D = x.shape
    W = weight.shape[1]
    boundaries = cu_seqlens.detach().cpu().tolist()
    outputs = []
    for bos, eos in zip(boundaries[:-1], boundaries[1:]):
        sequence = x[:, bos:eos].float()
        length = eos - bos
        output = torch.zeros(1, length, D, dtype=torch.float32, device=x.device)
        for tap in range(W):
            shift = W - 1 - tap
            source = (
                sequence if shift == 0 else torch.cat((torch.zeros_like(sequence[:, :shift]), sequence), dim=1)[:, :length]
            )
            output = output + source * weight[:, tap].float()[None, None, :]
        if bias is not None:
            output = output + bias.float()[None, None, :]
        if activation in ("silu", "swish"):
            output = output * torch.sigmoid(output)
        outputs.append(output)
    return torch.cat(outputs, dim=1).to(x.dtype)


def init_distributed(rank, world_size, port):
    """Initialize distributed environment for a single process."""
    # Configure logging in worker process
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    visible_var = "ASCEND_RT_VISIBLE_DEVICES" if IS_NPU else "CUDA_VISIBLE_DEVICES"
    visible_devices = os.environ.get(visible_var, "").split(",")
    device_key = visible_devices[rank].strip() if len(visible_devices) > rank else str(rank)
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/fla-triton-cache-{device}-{device_key}"
    if IS_NPU:
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "600")
    device_torch_lib.set_device(rank)
    dist.init_process_group(
        backend="hccl" if IS_NPU else "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=20),
    )


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_cp_conv_test_worker(
    rank: int,
    world_size: int,
    test_name: str,
    T: int,
    D: int,
    W: int,
    lengths: list[int],
    dtype,
    port: int,
):
    """
    Worker function for CP convolution test.
    Runs in a spawned process with the given rank.
    """
    try:
        init_distributed(rank, world_size, port)
        worker_device = torch.device(device, rank)

        assert T % world_size == 0, f"T={T} must be divisible by world_size={world_size}"
        assert sum(lengths) == T, f"Sum of lengths {sum(lengths)} must equal T={T}"

        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Test: {test_name}")
            print(f"Config: T={T}, D={D}, W={W}, world_size={world_size}")
            print(f"Sequence lengths: {lengths}")
            print(f"{'=' * 60}")

        # Step 1: Prepare Global Data
        torch.manual_seed(42)
        B = 1

        x_global = torch.randn(B, T, D, device=worker_device, dtype=dtype) * 10
        dy_global = torch.randn(B, T, D, device=worker_device, dtype=dtype)

        weight = torch.randn(D, W, device=worker_device, dtype=dtype)
        bias = torch.randn(D, device=worker_device, dtype=dtype)

        dist.broadcast(weight, src=0)
        dist.broadcast(bias, src=0)

        cu_seqlens_list = [0] + torch.cumsum(torch.tensor(lengths), 0).tolist()
        cu_seqlens_global = torch.tensor(cu_seqlens_list, device=worker_device, dtype=torch.int32)

        activation = "swish"

        # Step 2: Reference Run
        # Run the same reference on every device to avoid cold-cache skew before
        # the first HCCL collective.
        x_ref = x_global.clone().detach().requires_grad_(True)
        weight_ref = weight.clone().detach().requires_grad_(True)
        bias_ref = bias.clone().detach().requires_grad_(True)

        y_ref = causal_conv1d_reference(x_ref, weight_ref, bias_ref, activation, cu_seqlens_global)
        y_ref.backward(dy_global)

        ref_out = y_ref.detach()
        ref_dx = x_ref.grad.detach()
        ref_dw = weight_ref.grad.detach()
        ref_db = bias_ref.grad.detach()

        # Step 3: Context Parallel Run
        dist.barrier()

        context = build_cp_context(cu_seqlens_global, group=dist.group.WORLD, conv1d_kernel_size=W)

        chunk_size = T // world_size
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size

        x_local = x_global[:, start_idx:end_idx, :].clone().detach().requires_grad_(True)
        dy_local = dy_global[:, start_idx:end_idx, :].clone()
        weight_local = weight.clone().detach().requires_grad_(True)
        bias_local = bias.clone().detach().requires_grad_(True)

        print(
            f"[Rank {rank}] chunk: [{start_idx}, {end_idx}), "
            f"cu_seqlens: {context.cu_seqlens.tolist()}, "
            f"pre_num_ranks: {context.pre_num_ranks}, "
            f"pre_num_conv_tokens: {context.pre_num_conv_tokens}"
        )
        dist.barrier()

        # CP Forward
        y_local, _ = causal_conv1d(
            x=x_local,
            weight=weight_local,
            bias=bias_local,
            activation=activation,
            cp_context=context,
        )

        # CP Backward
        y_local.backward(dy_local)

        # Step 4: Result Aggregation and Verification
        y_gathered = [torch.zeros_like(y_local) for _ in range(world_size)]
        dist.all_gather(y_gathered, y_local)
        y_cp_global = torch.cat(y_gathered, dim=1)

        dx_gathered = [torch.zeros_like(x_local.grad) for _ in range(world_size)]
        dist.all_gather(dx_gathered, x_local.grad)
        dx_cp_global = torch.cat(dx_gathered, dim=1)

        dw_cp = weight_local.grad.clone()
        db_cp = bias_local.grad.clone()
        dist.all_reduce(dw_cp, op=dist.ReduceOp.SUM)
        dist.all_reduce(db_cp, op=dist.ReduceOp.SUM)

        test_passed = True
        if rank == 0:
            print(f"\n[{test_name}] Verification Results:")
            try:
                assert_strict_close("Output", ref_out, y_cp_global)
                assert_strict_close("dx", ref_dx, dx_cp_global)
                # Each rank's parameter gradient is cast to the BF16 weight
                # dtype before CP reduction. The frozen A800 RMS ratios are
                # 2.978e-3 at CP2 and 4.335e-3 at CP8, so retain their 1.10x
                # world-size-specific numerical envelopes.
                parameter_ratio = 1e-3
                if dtype == torch.bfloat16:
                    parameter_ratio = 4.8e-3 if world_size == 8 else 3.3e-3
                assert_strict_close("dw", ref_dw, dw_cp, ratio=parameter_ratio)
                assert_strict_close("db", ref_db, db_cp, ratio=parameter_ratio)
                print(f"✅ [{test_name}] Test Passed!\n")
            except AssertionError as e:
                print(f"❌ [{test_name}] Test Failed: {e}\n")
                test_passed = False

        dist.barrier()
        cleanup_distributed()

        if not test_passed:
            raise AssertionError(f"Test {test_name} failed on rank {rank}")

    except Exception as e:
        cleanup_distributed()
        raise e


def run_cp_test_with_spawn(
    world_size: int,
    test_name: str,
    T: int,
    D: int,
    W: int,
    lengths: list[int],
    dtype=torch.float32,
):
    """
    Run CP test using torch.multiprocessing.spawn.
    This allows running the test directly with pytest.
    """
    # Use start_processes with spawn to avoid fork/spawn conflicts
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.start_processes(
        run_cp_conv_test_worker,
        args=(world_size, test_name, T, D, W, lengths, dtype, port),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )


# ============================================================
# Test Scenario Definitions
# ============================================================


def test_cp2_sequence_cut():
    """
    Test Case 1: CP2 with sequences cut in the middle.

    Scenario:
    - world_size=2, T=1024, chunk_size=512
    - lengths=[300, 400, 324] -> sequences span across rank boundary
    - Rank 0: tokens [0, 512) contains seq0 (300) + part of seq1 (212)
    - Rank 1: tokens [512, 1024) contains rest of seq1 (188) + seq2 (324)
    """
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_SequenceCut",
        T=1024,
        D=128,
        W=4,
        lengths=[300, 400, 324],
        dtype=torch.float32,
    )


def test_cp2_boundary_aligned():
    """
    Test Case 2: CP2 with sequence boundaries aligned with rank boundaries.

    Scenario:
    - world_size=2, T=1024, chunk_size=512
    - lengths=[512, 512] -> sequence boundary exactly at rank boundary
    - Rank 0: tokens [0, 512) contains exactly seq0
    - Rank 1: tokens [512, 1024) contains exactly seq1
    - No sequence is split across ranks
    """
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_BoundaryAligned",
        T=1024,
        D=128,
        W=4,
        lengths=[512, 512],
        dtype=torch.float32,
    )


def test_cp4_complex():
    """
    Test Case 3: CP4 with complex sequence distribution.

    Scenario:
    - world_size=4, T=1024, chunk_size=256
    - lengths=[700, 324] -> first sequence spans 3 ranks
    - Rank 0: [0, 256) all seq0
    - Rank 1: [256, 512) all seq0
    - Rank 2: [512, 768) - 188 tokens of seq0 + 68 tokens of seq1
    - Rank 3: [768, 1024) - 256 tokens of seq1
    """
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_Complex",
        T=1024,
        D=128,
        W=4,
        lengths=[700, 324],
        dtype=torch.float32,
    )


def test_cp4_single_sequence():
    """
    Test Case 4: CP4 with a single long sequence spanning all ranks.

    Scenario:
    - world_size=4, T=1024, chunk_size=256
    - lengths=[1024] -> single sequence spans all 4 ranks
    - Each rank processes 256 tokens of the same sequence
    """
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_SingleSequence",
        T=1024,
        D=128,
        W=4,
        lengths=[1024],
        dtype=torch.float32,
    )


def test_cp2_many_short_sequences():
    """
    Test Case 5: CP2 with many short sequences.

    Scenario:
    - world_size=2, T=1024, chunk_size=512
    - lengths=[100, 150, 200, 250, 124, 100, 100] -> many short sequences
    - Some sequences are entirely in one rank, some span across
    """
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ManyShortSequences",
        T=1024,
        D=128,
        W=4,
        lengths=[100, 150, 200, 250, 124, 100, 100],
        dtype=torch.float32,
    )


# ============================================================
# Extreme Edge Cases: Short Local Sequences (T_local < W)
#
# These tests target a specific bug in causal_conv1d_bwd_kernel:
# when CP splits a sequence so that a rank gets a very short local
# segment (T_local < W), the backward kernel's mask_head_rows was
# missing the `& (o_t < T)` bound check, causing out-of-bounds
# reads from dy and producing NaN in dw.
# ============================================================


@pytest.mark.parametrize("backend", ["triton"])
def test_cp2_short_tail_len1(backend):
    """
    CP2: seq0 has 513 tokens, so rank 1 gets a length-1 tail (T=1 < W=4).

    Rank 0: [0, 512)  → 512 tokens of seq0
    Rank 1: [512, 1024) → 1 token of seq0 (T_local=1!) + 511 tokens of seq1
    """
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name=f"CP2_ShortTail_Len1_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[513, 511],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp2_short_tail_len2(backend):
    """
    CP2: seq0 has 514 tokens, so rank 1 gets a length-2 tail (T=2 < W=4).

    Rank 1: [512, 1024) → 2 tokens of seq0 (T_local=2!) + 510 tokens of seq1
    """
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name=f"CP2_ShortTail_Len2_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[514, 510],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp4_every_rank_gets_short_tail(backend):
    """
    CP4: every non-first rank gets a length-1 local sequence tail.

    lengths=[257, 255, 257, 255], T=1024, chunk_size=256
    Rank 0: [0, 256)   → 256 tokens of seq0
    Rank 1: [256, 512)  → 1 token of seq0 (T=1!) + 255 tokens of seq1
    Rank 2: [512, 768)  → 256 tokens of seq1(rest) + start of seq2
                          actually seq1=[257,512) so all 255 on rank1, then
                          seq2=[512,769) → rank2 gets 256, rank3 gets 1
    Rank 3: [768, 1024) → 1 token of seq2 (T=1!) + 255 tokens of seq3
    """
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name=f"CP4_EveryRankShortTail_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[257, 255, 257, 255],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp2_multiple_short_tails(backend):
    """
    CP2: multiple sequences each end 1 token into rank 1.

    lengths=[200, 313, 511] → seq0 ends at 200 (rank 0), seq1 ends at 513 (rank 1 gets 1 token)
    Rank 0: [0, 512)   → seq0(200) + 312 tokens of seq1
    Rank 1: [512, 1024) → 1 token of seq1 (T=1!) + 511 tokens of seq2
    """
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name=f"CP2_MultipleShortTails_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[200, 313, 511],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp2_global_len1_sequence(backend):
    """
    CP2: a globally length-1 sequence sits right at the rank boundary.

    lengths=[512, 1, 511] → seq1 is globally length 1, entirely on rank 1
    Rank 0: [0, 512)   → seq0(512)
    Rank 1: [512, 1024) → seq1(1, T=1!) + seq2(511)
    seq1 has no initial_state from prev rank (it starts fresh), but the
    initial_state tensor is still allocated for all seqs on non-first ranks.
    """
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 GPUs required")

    run_cp_test_with_spawn(
        world_size=2,
        test_name=f"CP2_GlobalLen1Sequence_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[512, 1, 511],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp4_multiple_short_tails(backend):
    """
    CP4: multiple sequences each end 1-2 tokens past a rank boundary,
    creating short local segments on multiple ranks.

    lengths=[257, 253, 257, 257], T=1024, chunk_size=256
    Rank 0: [0, 256)   → seq0[0:256]  (256 tokens)
    Rank 1: [256, 512)  → seq0 tail (1 tok, T=1!) + seq1 (253 tok) + seq2 start (2 tok, T=2!)
    Rank 2: [512, 768)  → seq2 cont (255 tok) + seq3 start (1 tok, T=1!)
    Rank 3: [768, 1024) → seq3 cont (256 tok)

    Ranks 1 and 2 each have local sequences with T < W=4.
    """
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name=f"CP4_MultipleShortTails_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[257, 253, 257, 257],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("backend", ["triton"])
def test_cp4_worst_case_many_len1(backend):
    """
    CP4: globally length-1 sequences + short tails across multiple ranks.
    Stress test for cross-sequence gradient isolation in CP backward.

    lengths=[257, 1, 253, 257, 1, 253, 2], T=1024, chunk_size=256
    Rank 1: cu_seqlens=[0,1,2,255,256] → 4 local seqs, lengths [1,1,253,1]
    Rank 2: cu_seqlens=[0,256], pre_num_conv_tokens=1 → gradient must be
            masked to only 1 valid position, not all W-1=3.
    """
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least 4 GPUs required")

    run_cp_test_with_spawn(
        world_size=4,
        test_name=f"CP4_WorstCaseManyLen1_{backend}",
        T=1024,
        D=128,
        W=4,
        lengths=[257, 1, 253, 257, 1, 253, 2],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("comm", ["all_gather", "p2p"])
def test_cp2_bfloat16_target_smoke(comm, monkeypatch):
    """CP2 BF16 smoke for the production W=4, D=1024 path."""
    if device_torch_lib.device_count() < 2:
        pytest.skip("At least 2 accelerators required")
    monkeypatch.setenv("FLA_CP_CONV_COMM", comm)

    run_cp_test_with_spawn(
        world_size=2,
        test_name=f"CP2_BF16_TargetSmoke_{comm}",
        T=512,
        D=1024,
        W=4,
        lengths=[512],
        dtype=torch.bfloat16,
    )


def run_conv_comm_subgroup_worker(rank: int, world_size: int, method: str, port: int):
    """Verify local group-peer routing for a non-contiguous subgroup."""
    from fla.ops.cp import conv_cp_send_recv_bwd, conv_cp_send_recv_fwd

    try:
        os.environ["FLA_CP_CONV_COMM"] = method
        init_distributed(rank, world_size, port)
        group = dist.new_group(ranks=[0, 2])
        if rank in (0, 2):
            group_rank = dist.get_rank(group)
            worker_device = torch.device(device, rank)
            send = torch.full((3, 16), group_rank + 1, dtype=torch.float32, device=worker_device)
            recv_fwd = conv_cp_send_recv_fwd(send, group)
            recv_bwd = conv_cp_send_recv_bwd(send, group)
            expected_fwd = torch.zeros_like(send) if group_rank == 0 else torch.ones_like(send)
            expected_bwd = torch.zeros_like(send) if group_rank == 1 else torch.full_like(send, 2)
            torch.testing.assert_close(recv_fwd, expected_fwd, rtol=0, atol=0)
            torch.testing.assert_close(recv_bwd, expected_bwd, rtol=0, atol=0)
        dist.barrier()
        cleanup_distributed()
    except Exception:
        cleanup_distributed()
        raise


@pytest.mark.parametrize("comm", ["all_gather", "p2p"])
def test_cp_noncontiguous_subgroup(comm):
    """Communication must interpret neighbors as ranks local to the CP group."""
    if device_torch_lib.device_count() < 4:
        pytest.skip("At least 4 accelerators required")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.start_processes(
        run_conv_comm_subgroup_worker,
        args=(4, comm, port),
        nprocs=4,
        join=True,
        start_method="spawn",
    )


def test_cp_halo_left_padding():
    """Short local chunks are right-aligned in the fixed W-1 wire shape."""
    from fla.modules.conv.cp.ops import CausalConv1dFunctionCP

    if device_torch_lib.device_count() < 1:
        pytest.skip("At least 1 accelerator required")
    x = torch.arange(8, dtype=torch.float32, device=device).reshape(1, 8)
    halo = CausalConv1dFunctionCP._right_aligned_halo(x, 3)
    expected = torch.cat((torch.zeros(2, 8, device=x.device), x), dim=0)
    torch.testing.assert_close(halo, expected, rtol=0, atol=0)


def test_cp_invalid_comm_method(monkeypatch):
    """Invalid communication selectors fail before entering a collective."""
    from fla.ops.cp.comm import _resolve_conv_comm_method

    monkeypatch.setenv("FLA_CP_CONV_COMM", "invalid")
    with pytest.raises(ValueError, match="FLA_CP_CONV_COMM"):
        _resolve_conv_comm_method(None)


@pytest.mark.parametrize("D", [1024, 3072])
def test_cp8_target(D):
    """CP8 BF16 production target with Tglobal=16384 and Tlocal=2048."""
    if device_torch_lib.device_count() < 8:
        pytest.skip("At least 8 accelerators required")

    run_cp_test_with_spawn(
        world_size=8,
        test_name=f"CP8_Ascend_Target_D{D}",
        T=16384,
        D=D,
        W=4,
        lengths=[16384],
        dtype=torch.bfloat16,
    )


# ============================================================
# Main Entry Point (for torchrun)
# ============================================================


def setup_distributed_torchrun():
    """Initialize distributed environment for torchrun."""
    if "RANK" not in os.environ:
        return False

    if IS_NPU:
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    dist.init_process_group(backend="hccl" if IS_NPU else "nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_torch_lib.set_device(local_rank)
    return True
