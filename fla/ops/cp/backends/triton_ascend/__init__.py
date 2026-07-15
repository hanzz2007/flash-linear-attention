# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend backend for context-parallel state preprocessing."""

from fla.ops.backends import BaseBackend


class TritonAscendCPBackend(BaseBackend):
    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    default_enable = True
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    @staticmethod
    def _verify(primary, g, gk, bg, context):
        from fla.utils import device

        if context is None or context.group is None:
            return False, 'context parallel is not enabled'
        if primary.device.type != device:
            return False, f'input must be on {device}, got {primary.device}'
        if g is None or gk is not None or bg is not None:
            return False, 'the Ascend CP backend currently supports GDN only'
        return True, None

    def chunk_gated_delta_rule_fwd_h_pre_process_verifier(
        self,
        k,
        w,
        u,
        g=None,
        gk=None,
        bg=None,
        v=None,
        chunk_size=64,
        state_v_first=False,
        cu_seqlens=None,
        initial_state=None,
        context=None,
    ):
        return self._verify(k, g, gk, bg, context)

    def chunk_gated_delta_rule_fwd_h_pre_process(self, *args, **kwargs):
        from .chunk_delta_h import chunk_gated_delta_rule_fwd_h_pre_process_npu
        return chunk_gated_delta_rule_fwd_h_pre_process_npu(*args, **kwargs)

    def chunk_gated_delta_rule_bwd_dhu_pre_process_verifier(
        self,
        q,
        k,
        w,
        do,
        dv,
        g=None,
        gk=None,
        bg=None,
        scale=None,
        state_v_first=False,
        cu_seqlens=None,
        dht=None,
        initial_state=None,
        context=None,
        chunk_size=64,
    ):
        return self._verify(q, g, gk, bg, context)

    def chunk_gated_delta_rule_bwd_dhu_pre_process(self, *args, **kwargs):
        from .chunk_delta_h import chunk_gated_delta_rule_bwd_dhu_pre_process_npu
        return chunk_gated_delta_rule_bwd_dhu_pre_process_npu(*args, **kwargs)


__all__ = ['TritonAscendCPBackend']
