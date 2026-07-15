# Ascend Triton specialization optimization log

This record tracks compile-specialization changes for the Ascend GDN CP and causal Conv1d kernels. Latency results collected while the machines are shared are intentionally excluded.

## Specialization contract

Triton-Ascend 3.2.0 implicitly classifies ordinary integer and pointer arguments as divisible by 16, equal to one, or neither. Runtime shape, count, rank, and launch-offset arguments therefore require both an explicit scalar type and `do_not_specialize`; removing `tl.constexpr` alone is insufficient. Pointer and stride specialization remains intentional because alignment and unit-stride information can affect code generation.

## Iterations

| Phase | Change | Correctness and compile evidence | Status |
| --- | --- | --- | --- |
| 1 | Add a machine-checkable specialization manifest and an NPU capability test covering values 1, 16, 17, and greater than int32 | 910B: 6 tests passed; all four runtime values reused one in-memory cache entry and produced no additional disk artifacts after the first compile | Accepted |

Performance measurements remain pending until the machines are exclusive.
