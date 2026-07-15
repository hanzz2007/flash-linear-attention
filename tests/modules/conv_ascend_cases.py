# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative pairwise coverage catalog for Ascend causal Conv1d tests."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConvDenseCase:
    name: str
    B: int
    T: int
    D: int
    W: int
    dtype: torch.dtype
    activation: str | None
    has_bias: bool
    has_state: bool
    has_residual: bool
    nightly: bool = False
    tags: frozenset[str] = frozenset()


CONV_DENSE_CASES = (
    ConvDenseCase("minimum", 1, 1, 1, 2, torch.bfloat16, None, False, False, False, tags=frozenset({"minimum"})),
    ConvDenseCase(
        "batch5-t3-d17",
        5,
        3,
        17,
        4,
        torch.bfloat16,
        "swish",
        True,
        False,
        True,
        tags=frozenset({"batch5", "t3", "d17", "swish", "residual"}),
    ),
    ConvDenseCase(
        "batch2-t64-d129-fp16",
        2,
        64,
        129,
        3,
        torch.float16,
        None,
        False,
        True,
        False,
        tags=frozenset({"batch2", "t64", "d129", "fp16", "state"}),
    ),
    ConvDenseCase(
        "t65-d1001",
        1,
        65,
        1001,
        3,
        torch.bfloat16,
        "silu",
        True,
        True,
        True,
        tags=frozenset({"t65", "d-tail", "silu", "state", "residual"}),
    ),
    ConvDenseCase(
        "t257-d1024-fp16",
        1,
        257,
        1024,
        4,
        torch.float16,
        None,
        False,
        True,
        False,
        tags=frozenset({"t257", "d1024", "fp16", "state"}),
    ),
    ConvDenseCase(
        "target-d3072",
        1,
        2048,
        3072,
        4,
        torch.bfloat16,
        "silu",
        True,
        True,
        False,
        tags=frozenset({"target", "d3072", "silu", "state"}),
    ),
    ConvDenseCase(
        "fp32-fallback",
        1,
        4,
        128,
        2,
        torch.float32,
        None,
        True,
        False,
        False,
        tags=frozenset({"fp32", "fallback", "t4"}),
    ),
    ConvDenseCase("t511-d511", 1, 511, 511, 4, torch.bfloat16, None, False, False, False, True, frozenset({"t511", "d511"})),
    ConvDenseCase("t512-d512", 1, 512, 512, 3, torch.float16, "silu", True, False, False, True, frozenset({"t512", "d512"})),
    ConvDenseCase("t513-d513", 1, 513, 513, 2, torch.bfloat16, None, False, True, False, True, frozenset({"t513", "d513"})),
    ConvDenseCase("t63-d1023", 1, 63, 1023, 4, torch.bfloat16, None, True, False, False, True, frozenset({"t63", "d1023"})),
    ConvDenseCase("t64-d1025", 1, 64, 1025, 3, torch.float16, None, False, True, False, True, frozenset({"t64", "d1025"})),
    ConvDenseCase(
        "t65-d2047", 1, 65, 2047, 2, torch.bfloat16, "swish", False, False, False, True, frozenset({"t65", "d2047"})
    ),
    ConvDenseCase("t63-d2048", 1, 63, 2048, 4, torch.float16, None, True, False, False, True, frozenset({"t63", "d2048"})),
    ConvDenseCase("t64-d2049", 1, 64, 2049, 3, torch.bfloat16, None, False, True, False, True, frozenset({"t64", "d2049"})),
    ConvDenseCase("t4-d4096", 1, 4, 4096, 4, torch.bfloat16, None, True, False, False, True, frozenset({"t4", "d4096"})),
    ConvDenseCase("t1-d8191", 1, 1, 8191, 2, torch.bfloat16, None, False, False, False, True, frozenset({"d8191"})),
    ConvDenseCase("t2-d8192", 1, 2, 8192, 3, torch.float16, None, True, False, False, True, frozenset({"d8192"})),
    ConvDenseCase(
        "t3-d8193", 1, 3, 8193, 4, torch.bfloat16, None, False, True, False, True, frozenset({"d8193", "maximum-d"})
    ),
)


@dataclass(frozen=True)
class ConvVarlenCase:
    name: str
    lengths: tuple[int, ...]
    D: int
    W: int
    dtype: torch.dtype
    activation: str | None
    has_bias: bool
    has_state: bool
    has_residual: bool
    noncontiguous_x: bool = False
    noncontiguous_dy: bool = False
    nightly: bool = False
    tags: frozenset[str] = frozenset()


CONV_VARLEN_CASES = (
    ConvVarlenCase(
        "many-short-bf16",
        (1, 2, 3, 4),
        129,
        4,
        torch.bfloat16,
        None,
        True,
        True,
        True,
        tags=frozenset({"bf16", "many-short", "d-tail", "state", "residual"}),
    ),
    ConvVarlenCase(
        "tile-boundaries-fp16",
        (63, 64, 65),
        127,
        3,
        torch.float16,
        "swish",
        False,
        True,
        False,
        noncontiguous_dy=True,
        tags=frozenset({"fp16", "t-boundary", "noncontiguous-dy", "swish", "w3"}),
    ),
    ConvVarlenCase(
        "total-257-noncontiguous",
        (1, 63, 64, 129),
        513,
        2,
        torch.bfloat16,
        None,
        True,
        False,
        False,
        noncontiguous_x=True,
        tags=frozenset({"total-257", "noncontiguous-x", "w2", "d-tail"}),
    ),
    ConvVarlenCase(
        "packed-target-d1024",
        (257, 511, 1280),
        1024,
        4,
        torch.bfloat16,
        "silu",
        True,
        True,
        False,
        nightly=True,
        tags=frozenset({"target", "d1024", "packed", "state", "silu"}),
    ),
    ConvVarlenCase(
        "packed-target-d3072",
        (32, 32, 32, 32, 1920),
        3072,
        4,
        torch.bfloat16,
        "silu",
        True,
        False,
        False,
        nightly=True,
        tags=frozenset({"target", "d3072", "packed", "many-short"}),
    ),
    ConvVarlenCase(
        "maximum-d8193",
        (1, 2, 3),
        8193,
        4,
        torch.bfloat16,
        None,
        False,
        False,
        False,
        nightly=True,
        tags=frozenset({"d8193", "maximum-d", "many-short"}),
    ),
)
