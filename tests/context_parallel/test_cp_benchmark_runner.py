# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Pure-Python contract tests for the paired CP performance runner."""

import pytest

from benchmarks.cp.run_ascend_cp_regression import _cases, _compare_pair, _visible_devices


def _result(*, backend: str = "hccl", latency: float = 1.0, memory: int = 1000):
    return {
        "backend": backend,
        "device": "test-device",
        "torch": "2.7.1",
        "triton": "3.2.0",
        "triton_ascend": "3.2.1" if backend == "hccl" else None,
        "world_size": 8,
        "dtype": "bfloat16",
        "precision": "high" if backend == "hccl" else "cuda",
        "latency_ms": {"median": latency},
        "peak_memory_bytes": memory,
    }


def test_cp_performance_case_matrix_is_high_only() -> None:
    cases = _cases("full")
    names = {case.name for case in cases}
    assert {"gdn_cp2_fwd_bwd", "gdn_cp4_fwd_bwd", "gdn_cp8_fwd_bwd"} <= names
    assert {"conv_cp8_d1024_single", "conv_cp8_d1024_packed", "conv_cp8_d3072_packed"} <= names
    assert {"conv_kernel_d1024", "conv_kernel_d3072", "conv_comm_cp8"} <= names
    serialized = " ".join(argument for case in cases for argument in case.args)
    assert "a800" not in serialized.lower()
    assert _visible_devices(1) == "2"
    assert _visible_devices(4) == "2,3,4,5"
    assert _visible_devices(8) == "0,1,2,3,4,5,6,7"


def test_cp_performance_thresholds_depend_on_platform() -> None:
    case = _cases("pr")[0]
    npu = _compare_pair(case, _result(), _result(latency=1.11, memory=1040))
    assert not npu["latency_regressed"]
    assert not npu["memory_regressed"]
    assert _compare_pair(case, _result(), _result(latency=1.13))["latency_regressed"]

    cuda = _compare_pair(
        case,
        _result(backend="nccl"),
        _result(backend="nccl", latency=1.06),
    )
    assert cuda["latency_threshold"] == 0.05
    assert cuda["latency_regressed"]


def test_cp_performance_refuses_mismatched_stack() -> None:
    case = _cases("pr")[0]
    candidate = _result()
    candidate["torch"] = "2.8.0"
    with pytest.raises(ValueError, match="mismatched metadata"):
        _compare_pair(case, _result(), candidate)
