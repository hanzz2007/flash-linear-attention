# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only guards for the declarative Ascend Conv1d coverage catalog."""

import torch

from tests.modules.conv_ascend_cases import CONV_DENSE_CASES, CONV_VARLEN_CASES


def test_conv_varlen_case_matrix_covers_contract() -> None:
    tags = frozenset().union(*(case.tags for case in CONV_VARLEN_CASES))
    required = {
        "bf16",
        "fp16",
        "many-short",
        "t-boundary",
        "d-tail",
        "noncontiguous-x",
        "noncontiguous-dy",
        "state",
        "residual",
        "silu",
        "swish",
        "w2",
        "w3",
        "target",
        "d1024",
        "d3072",
        "maximum-d",
    }
    assert required <= tags, f"missing Conv coverage tags: {sorted(required - tags)}"
    assert {case.dtype for case in CONV_VARLEN_CASES} == {torch.bfloat16, torch.float16}
    assert {case.W for case in CONV_VARLEN_CASES} == {2, 3, 4}
    assert any(min(case.lengths) == 1 for case in CONV_VARLEN_CASES)
    assert any(sum(case.lengths) == 257 for case in CONV_VARLEN_CASES)
    assert {(sum(case.lengths), case.D) for case in CONV_VARLEN_CASES if "target" in case.tags} == {
        (2048, 1024),
        (2048, 3072),
    }
    assert max(case.D for case in CONV_VARLEN_CASES) == 8193

    pr_names = {case.name for case in CONV_VARLEN_CASES if not case.nightly}
    nightly_names = {case.name for case in CONV_VARLEN_CASES if case.nightly}
    assert pr_names
    assert nightly_names
    assert pr_names.isdisjoint(nightly_names)


def test_conv_dense_case_matrix_covers_contract() -> None:
    tags = frozenset().union(*(case.tags for case in CONV_DENSE_CASES))
    required = {
        "minimum",
        "batch2",
        "batch5",
        "fp16",
        "fp32",
        "fallback",
        "t3",
        "t4",
        "t63",
        "t64",
        "t65",
        "t257",
        "t511",
        "t512",
        "t513",
        "d17",
        "d511",
        "d512",
        "d513",
        "d1023",
        "d1024",
        "d1025",
        "d2047",
        "d2048",
        "d2049",
        "d3072",
        "d4096",
        "d8191",
        "d8192",
        "d8193",
        "maximum-d",
        "target",
    }
    assert required <= tags, f"missing dense Conv coverage tags: {sorted(required - tags)}"
    assert {case.W for case in CONV_DENSE_CASES} == {2, 3, 4}
    assert {case.B for case in CONV_DENSE_CASES} >= {1, 2, 5}
    assert {case.dtype for case in CONV_DENSE_CASES} == {torch.bfloat16, torch.float16, torch.float32}
    assert min(case.T for case in CONV_DENSE_CASES) == 1
    assert max(case.T for case in CONV_DENSE_CASES) == 2048
    assert min(case.D for case in CONV_DENSE_CASES) == 1
    assert max(case.D for case in CONV_DENSE_CASES) == 8193
