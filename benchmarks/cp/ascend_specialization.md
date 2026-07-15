# Ascend Triton specialization optimization log

This record tracks compile-specialization changes for the Ascend GDN CP and causal Conv1d kernels. Latency results collected while the machines are shared are intentionally excluded.

## Specialization contract

Triton-Ascend 3.2.0 implicitly classifies ordinary integer and pointer arguments as divisible by 16, equal to one, or neither. Runtime shape, count, rank, and launch-offset arguments therefore require both an explicit scalar type and `do_not_specialize`; removing `tl.constexpr` alone is insufficient. Pointer and stride specialization remains intentional because alignment and unit-stride information can affect code generation.

Logical shape and base-pointer arithmetic use `tl.int64`. Triton block pointers are a documented implementation exception in this stack: their shape and strides may be int64, but offsets must be int32. Kernels using `tl.make_block_ptr` therefore keep the dynamic launch or loop value despecialized while narrowing only the derived block offset to `tl.int32`; the base pointer remains int64-addressed.

## Iterations

| Phase | Change | Correctness and compile evidence | Status |
| --- | --- | --- | --- |
| 1 | Add a machine-checkable specialization manifest and an NPU capability test covering values 1, 16, 17, and greater than int32 | Commit `d90b2cc9`; 910B: 6 tests passed; all four runtime values reused one in-memory cache entry and produced no additional disk artifacts after the first compile | Accepted |
| 2 | Despecialize GDN CP and shared GDN logical shapes, counts, ranks, and split offsets; remove unused cumsum parameters | 910B: specialization manifest 43 passed; shared GDN kernel matrix 41 passed; CP preprocessing 21 passed and 2 distributed cases skipped in the single-device gate | Accepted |

The initial Phase 2 attempt made every dynamic offset int64. It was rejected because Triton-Ascend 3.2.0 block pointers accept only int32 offsets. The accepted implementation fixes the scalar type at int32 only where a value directly enters a block-pointer offset and keeps it in `do_not_specialize`; all explicit pointer arithmetic and logical values remain int64. This preserves compile reuse without changing the block-pointer ABI.

Performance measurements remain pending until the machines are exclusive.
