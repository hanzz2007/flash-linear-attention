# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules.conv.cp.ops import CausalConv1dFunctionCP


def test_right_aligned_halo_pads_short_input() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    halo = CausalConv1dFunctionCP._right_aligned_halo(x, 3)
    expected = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    torch.testing.assert_close(halo, expected, rtol=0, atol=0)


def test_multi_rank_halo_assembly_is_right_aligned() -> None:
    gathered = torch.tensor([[[0.0], [0.0], [10.0]], [[0.0], [0.0], [20.0]], [[0.0], [0.0], [30.0]],
                             [[0.0], [0.0], [40.0]]])

    actual = CausalConv1dFunctionCP._assemble_previous_halo(gathered, rank=3, local_t=1, needed=3)
    torch.testing.assert_close(actual, torch.tensor([[10.0], [20.0], [30.0]]), rtol=0, atol=0)

    actual = CausalConv1dFunctionCP._assemble_previous_halo(gathered, rank=3, local_t=1, needed=2)
    torch.testing.assert_close(actual, torch.tensor([[0.0], [20.0], [30.0]]), rtol=0, atol=0)


@pytest.mark.parametrize(
    ('rank', 'expected'),
    [(0, 6.0), (1, 50.0), (2, 300.0), (3, 0.0)],
)
def test_multi_rank_halo_gradient_maps_to_owner(rank: int, expected: float) -> None:
    gathered = torch.tensor(
        [
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [1.0]],
            [[0.0], [2.0], [20.0]],
            [[3.0], [30.0], [300.0]],
        ]
    )
    dx = torch.zeros(1, 1, 1)
    CausalConv1dFunctionCP._accumulate_multi_rank_halo_gradients(dx, gathered, rank=rank)
    assert dx.item() == expected
