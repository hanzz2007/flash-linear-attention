# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from fla.ops.cp.context import FLACPContext, build_cp_context, get_cp_cu_seqlens
from fla.utils import IS_NPU, device


@dataclass(frozen=True)
class RankExpectation:
    local_cu_seqlens: tuple[int, ...]
    is_first_rank: bool
    is_last_rank: bool
    pre_num_ranks: int
    post_num_ranks: int
    pre_num_conv_tokens: int


@dataclass(frozen=True)
class ContextCase:
    name: str
    global_cu_seqlens: tuple[int, ...]
    world_size: int
    ranks: tuple[RankExpectation, ...]
    tags: frozenset[str]


CONTEXT_CASES = (
    ContextCase(
        name="world1",
        global_cu_seqlens=(0, 8),
        world_size=1,
        ranks=(RankExpectation((0, 8), True, True, 0, 0, 0),),
        tags=frozenset({"world1", "single_sequence", "aligned"}),
    ),
    ContextCase(
        name="cp2-aligned",
        global_cu_seqlens=(0, 4, 8),
        world_size=2,
        ranks=(
            RankExpectation((0, 4), True, True, 0, 0, 0),
            RankExpectation((0, 4), True, True, 0, 0, 0),
        ),
        tags=frozenset({"world2", "multiple_sequences", "aligned"}),
    ),
    ContextCase(
        name="cp2-cut",
        global_cu_seqlens=(0, 3, 7, 8),
        world_size=2,
        ranks=(
            RankExpectation((0, 3, 4), True, False, 0, 1, 0),
            RankExpectation((0, 3, 4), False, True, 1, 0, 1),
        ),
        tags=frozenset({"world2", "multiple_sequences", "sequence_cut"}),
    ),
    ContextCase(
        name="cp4-many-short",
        global_cu_seqlens=(0, 1, 2, 7, 8, 9, 15, 16),
        world_size=4,
        ranks=(
            RankExpectation((0, 1, 2, 4), True, False, 0, 1, 0),
            RankExpectation((0, 3, 4), False, True, 1, 0, 2),
            RankExpectation((0, 1, 4), True, False, 0, 1, 0),
            RankExpectation((0, 3, 4), False, True, 1, 0, 3),
        ),
        tags=frozenset({"world4", "multiple_sequences", "sequence_cut", "length1"}),
    ),
    ContextCase(
        name="cp8-single-sequence",
        global_cu_seqlens=(0, 16),
        world_size=8,
        ranks=tuple(
            RankExpectation(
                local_cu_seqlens=(0, 2),
                is_first_rank=rank == 0,
                is_last_rank=rank == 7,
                pre_num_ranks=rank,
                post_num_ranks=7 - rank,
                pre_num_conv_tokens=2 * rank,
            )
            for rank in range(8)
        ),
        tags=frozenset({"world8", "single_sequence", "sequence_cut"}),
    ),
)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("case", CONTEXT_CASES, ids=lambda case: case.name)
def test_cp_context_partition_contract(case: ContextCase, dtype: torch.dtype) -> None:
    global_cu_seqlens = torch.tensor(case.global_cu_seqlens, dtype=dtype)

    for rank, expected in enumerate(case.ranks):
        context = get_cp_cu_seqlens(
            global_cu_seqlens,
            world_size=case.world_size,
            rank=rank,
            conv1d_kernel_size=4,
        )

        assert context.cu_seqlens.dtype == torch.int32
        assert context.cu_seqlens_cpu.dtype == torch.int32
        assert context.cu_seqlens.device == global_cu_seqlens.device
        assert tuple(context.cu_seqlens_cpu.tolist()) == expected.local_cu_seqlens
        assert torch.equal(context.cu_seqlens.cpu(), context.cu_seqlens_cpu)
        assert context.is_first_rank is expected.is_first_rank
        assert context.is_last_rank is expected.is_last_rank
        assert context.pre_num_ranks == expected.pre_num_ranks
        assert context.post_num_ranks == expected.post_num_ranks
        assert context.pre_num_conv_tokens == expected.pre_num_conv_tokens
        assert context.conv1d_kernel_size == 4
        assert context.num_seqs == len(expected.local_cu_seqlens) - 1
        assert not context.is_cp_enabled


def test_cp_context_case_matrix_covers_contract() -> None:
    tags = frozenset().union(*(case.tags for case in CONTEXT_CASES))
    required = {
        "world1",
        "world2",
        "world4",
        "world8",
        "single_sequence",
        "multiple_sequences",
        "aligned",
        "sequence_cut",
        "length1",
    }
    assert required <= tags, f"missing CP context coverage tags: {sorted(required - tags)}"


def test_cp_context_copy_for_backward_is_independent() -> None:
    context = FLACPContext(
        group=None,
        cu_seqlens=torch.tensor([0, 3, 8], dtype=torch.int32),
        cu_seqlens_cpu=torch.tensor([0, 3, 8], dtype=torch.int32),
        is_last_rank=False,
        pre_num_ranks=2,
        is_first_rank=False,
        post_num_ranks=1,
        conv1d_kernel_size=4,
        pre_num_conv_tokens=5,
    )

    copied = context.copy_for_backward()
    copied.cu_seqlens[1] = 4
    copied.cu_seqlens_cpu[1] = 4

    assert context.cu_seqlens.tolist() == [0, 3, 8]
    assert context.cu_seqlens_cpu.tolist() == [0, 3, 8]
    assert copied.group is context.group
    assert copied.is_last_rank == context.is_last_rank
    assert copied.pre_num_ranks == context.pre_num_ranks
    assert copied.is_first_rank == context.is_first_rank
    assert copied.post_num_ranks == context.post_num_ranks
    assert copied.conv1d_kernel_size == context.conv1d_kernel_size
    assert copied.pre_num_conv_tokens == context.pre_num_conv_tokens


def test_build_cp_context_uses_process_group_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    group = object()
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 1)
    global_cu_seqlens = torch.tensor([0, 3, 7, 8], dtype=torch.int64)

    context = build_cp_context(
        global_cu_seqlens,
        group=group,
        conv1d_kernel_size=4,
        cu_seqlens_cpu=global_cu_seqlens.clone(),
    )

    assert context.group is group
    assert context.cu_seqlens.tolist() == [0, 3, 4]
    assert context.is_first_rank is False
    assert context.is_last_rank is True
    assert context.pre_num_ranks == 1
    assert context.post_num_ranks == 0
    assert context.pre_num_conv_tokens == 1


@pytest.mark.parametrize(
    ("cu_seqlens", "kwargs", "error", "match"),
    [
        (torch.tensor([0.0, 4.0]), {"world_size": 1, "rank": 0}, TypeError, "dtype int32 or int64"),
        (torch.tensor([[0, 4]]), {"world_size": 1, "rank": 0}, ValueError, "one-dimensional"),
        (torch.tensor([0]), {"world_size": 1, "rank": 0}, ValueError, "at least two"),
        (torch.tensor([1, 4]), {"world_size": 1, "rank": 0}, ValueError, "start at 0"),
        (torch.tensor([0, 0]), {"world_size": 1, "rank": 0}, ValueError, "total tokens must be positive"),
        (torch.tensor([0, 3, 2, 4]), {"world_size": 1, "rank": 0}, ValueError, "strictly increasing"),
        (torch.tensor([0, 5]), {"world_size": 2, "rank": 0}, ValueError, "divisible by world_size"),
        (torch.tensor([0, 2]), {"world_size": 4, "rank": 0}, ValueError, "at least world_size"),
        (torch.tensor([0, 4]), {"world_size": 0, "rank": 0}, ValueError, "positive integer"),
        (torch.tensor([0, 4]), {"world_size": 2, "rank": -1}, ValueError, "rank must be in"),
        (torch.tensor([0, 4]), {"world_size": 2, "rank": 2}, ValueError, "rank must be in"),
        (torch.tensor([0, 4]), {"world_size": 2}, ValueError, "rank is required"),
        (torch.tensor([0, 4]), {}, ValueError, "group is required"),
        (torch.tensor([0, 4]), {"world_size": 1, "rank": 0, "conv1d_kernel_size": 0}, ValueError, "one of"),
        (torch.tensor([0, 4]), {"world_size": 1, "rank": 0, "conv1d_kernel_size": 1}, ValueError, "one of"),
        (torch.tensor([0, 4]), {"world_size": 1, "rank": 0, "conv1d_kernel_size": 5}, ValueError, "one of"),
    ],
    ids=[
        "wrong-dtype",
        "wrong-rank",
        "too-short",
        "nonzero-start",
        "zero-total",
        "not-increasing",
        "not-divisible",
        "world-larger-than-tokens",
        "invalid-world",
        "negative-rank",
        "rank-out-of-range",
        "missing-rank",
        "missing-group",
        "invalid-conv-width",
        "conv-width-one",
        "conv-width-five",
    ],
)
def test_cp_context_rejects_invalid_inputs(cu_seqlens, kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        get_cp_cu_seqlens(cu_seqlens, **kwargs)


def test_cp_context_rejects_mismatched_cpu_metadata() -> None:
    with pytest.raises(ValueError, match="identical cumulative lengths"):
        get_cp_cu_seqlens(
            torch.tensor([0, 4], dtype=torch.int32),
            cu_seqlens_cpu=torch.tensor([0, 3], dtype=torch.int64),
            world_size=1,
            rank=0,
        )


@pytest.mark.skipif(not IS_NPU, reason="NPU device metadata coverage")
def test_cp_context_preserves_accelerator_device() -> None:
    cu_seqlens_cpu = torch.tensor([0, 3, 7, 8], dtype=torch.int64)
    cu_seqlens = cu_seqlens_cpu.to(device)
    context = get_cp_cu_seqlens(
        cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        world_size=2,
        rank=1,
    )

    assert context.cu_seqlens.device.type == device
    assert context.cu_seqlens.dtype == torch.int32
    assert context.cu_seqlens.tolist() == [0, 3, 4]
