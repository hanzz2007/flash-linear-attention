# Ascend CP industrial test coverage

This record tracks the correctness, distributed, stability, and performance gates for the Triton-Ascend FLA CP and causal Conv1d CP paths. Public signatures and numerical tolerances remain frozen while coverage is expanded.

## Frozen numerical gates

| Path | Tensor | RMS-ratio limit |
| --- | --- | ---: |
| GDN high primitive | H, M, dH, dM, merge | `1e-4` |
| GDN public | output and every gradient | `3e-3` |
| KDA/DPLR/RWKV7 smoke | output and gradients | `8e-3` |
| Conv public | output, dx, dw, db | `1e-3` |
| Conv BF16 | dh0 | `3.1e-3` |
| Conv CP2/4 BF16 | dw, db | `3.3e-3` |
| Conv CP8 BF16 | dw, db | `4.8e-3` |

Every numerical check also requires finite reference and actual tensors. Tests use an absolute-error escape only for near-zero references; `FLA_CI_ENV` never downgrades these gates to warnings.

## Phase results

| Phase | Scope | Environment | Result | Commit |
| --- | --- | --- | --- | --- |
| 1 | CP context validation, exact rank metadata, int32/int64 global inputs, copy semantics, coverage tags | 910B physical device 2; CP2 on devices 2–3; CANN 9.0.0, PyTorch 2.7.1, torch-npu 2.7.1.post6, Triton-Ascend 3.2.0 | 29 protocol tests and the existing CP2 Conv sequence-cut public test passed | `2f5d65c6` |
| 2 | GDN primitive dimensions, GVA, non-power-of-two/tail tiles, K=1/193/256, precomputed gates, merge ordering, canaries, forced grid splitting | 910B physical device 2; same stack; isolated Triton caches | 8 PR primitives, 5 merge/grid cases, and 9 host gates passed. Cold-cache PR primitives took 295.53 s. | `41e3ae4e` |

The compatibility selector still accepts `FLA_ASCEND_CP_GDN_PRECISION=a800`, but A800-parity numerical and performance gates are excluded from the active test matrix. High precision is the only acceptance mode.

## Reproduction

```bash
source /data/Ascend/9.0.0/cann-9.0.0/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=2 \
ASCEND_UB_CAPACITY_BITS=1572864 \
PYTHONPATH=. \
/data/qiuhan/3.conda/swift-mmu-beta-v0.2-cann900_triton321/bin/python \
  -m pytest tests/context_parallel/test_cp_context.py -v
```
