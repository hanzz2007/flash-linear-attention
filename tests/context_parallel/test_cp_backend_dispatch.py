# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.cp.backends.triton_ascend import TritonAscendCPBackend
from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_fwd_h_pre_process
from fla.ops.cp.context import FLACPContext


def test_cp_backend_verifier_rejects_calls_without_context() -> None:
    backend = TritonAscendCPBackend()
    tensor = torch.empty(1, 1, 1, 1)
    accepted, reason = backend.chunk_gated_delta_rule_fwd_h_pre_process_verifier(
        tensor,
        tensor,
        tensor,
        g=torch.empty(1),
    )
    assert not accepted
    assert reason == 'context parallel is not enabled'


def test_cp_backend_verifier_accepts_gdn_and_rejects_other_modes(monkeypatch) -> None:
    monkeypatch.setattr('fla.utils.device', 'cpu')
    backend = TritonAscendCPBackend()
    tensor = torch.empty(1, 1, 1, 1)
    context = FLACPContext(group=object())

    accepted, reason = backend.chunk_gated_delta_rule_fwd_h_pre_process_verifier(
        tensor,
        tensor,
        tensor,
        g=torch.empty(1),
        context=context,
    )
    assert accepted, reason

    for gate_args in ({'gk': torch.empty(1)}, {'gk': torch.empty(1), 'bg': torch.empty(1)}):
        accepted, reason = backend.chunk_gated_delta_rule_fwd_h_pre_process_verifier(
            tensor,
            tensor,
            tensor,
            context=context,
            **gate_args,
        )
        assert not accepted
        assert reason == 'the Ascend CP backend currently supports GDN only'


def test_cp_public_fallback_preserves_no_context_semantics() -> None:
    tensor = torch.empty(1, 1, 1, 1)
    initial_state = torch.randn(1, 1, 1, 1)
    actual = chunk_gated_delta_rule_fwd_h_pre_process(
        k=tensor,
        w=tensor,
        u=tensor,
        g=torch.empty(1),
        initial_state=initial_state,
        context=None,
    )
    assert actual is initial_state
