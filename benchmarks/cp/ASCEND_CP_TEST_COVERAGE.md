# Ascend CP industrial test coverage

This record tracks the correctness, distributed, stability, and performance gates for the Triton-Ascend FLA CP and causal Conv1d CP paths. Public signatures and numerical tolerances remain frozen while coverage is expanded.

## Frozen numerical gates

| Path | Tensor | RMS-ratio limit |
| --- | --- | ---: |
| GDN high primitive | H, M, dH, dM, merge | `1e-4` |
| GDN public | output and every gradient | `3e-3` |
| GDN A800-parity target | local dM | `2.353e-2` |
| KDA/DPLR/RWKV7 smoke | output and gradients | `8e-3` |
| Conv public | output, dx, dw, db | `1e-3` |
| Conv BF16 | dh0 | `3.1e-3` |
| Conv CP2/4 BF16 | dw, db | `3.3e-3` |
| Conv CP8 BF16 | dw, db | `4.8e-3` |

Every numerical check also requires finite reference and actual tensors. Tests use an absolute-error escape only for near-zero references; `FLA_CI_ENV` never downgrades these gates to warnings.

## Phase results

| Phase | Scope | Environment | Result | Commit |
| --- | --- | --- | --- | --- |
| 1 | CP context validation, exact rank metadata, int32/int64 global inputs, copy semantics, coverage tags | 910B physical device 2; CP2 on devices 2–3; CANN 9.0.0, PyTorch 2.7.1, torch-npu 2.7.1.post6, Triton-Ascend 3.2.0 | 29 protocol tests and the existing CP2 Conv sequence-cut public test passed | Backfilled in the final phase |

## Reproduction

```bash
source /data/Ascend/9.0.0/cann-9.0.0/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=2 \
ASCEND_UB_CAPACITY_BITS=1572864 \
PYTHONPATH=. \
/data/qiuhan/3.conda/swift-mmu-beta-v0.2-cann900_triton321/bin/python \
  -m pytest tests/context_parallel/test_cp_context.py -v
```
