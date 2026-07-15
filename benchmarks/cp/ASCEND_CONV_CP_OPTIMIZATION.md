# Ascend 910B CP Causal Conv1d Optimization Record

## Contract

The target is BF16 causal depthwise Conv1d with `W=4`, `Tglobal=16384`, CP2/4/8, and `D=1024/3072`. Public APIs, return shapes and dtypes, existing CUDA kernels, and strict correctness tolerances remain unchanged. The default precision path remains high precision; an A800-relative path may be promoted only within the measured A800 error envelope.

## Environment

| Platform | Hardware | Software |
| -------- | -------- | -------- |
| A800 | 8x NVIDIA A800-SXM4-80GB | PyTorch 2.11.0+cu129, Triton 3.6.0 |
| 910B | 8x Ascend 910B | CANN 9.0.0, PyTorch 2.7.1, torch_npu 2.7.1.post6, Triton 3.2.0 |

The comparison is therefore the efficiency of each device together with its pinned production stack, not a pure silicon peak comparison.

## Stage 1 — frozen baseline and gates

- The existing CP wrapper already reaches the module-level Triton-Ascend backend.
- A finite single-card initial-state and SiLU forward/backward smoke passes on physical NPU 2.
- Exploratory `Tlocal=2048,D=3072,W=4,BF16` latency is `20.028/90.653 ms` on 910B and `0.111/0.879 ms` on A800.
- The initial 910B/A800 gaps are approximately `180.6x` forward and `103.1x` backward.
- The BF16 CP2 target smoke measures `dw` RMS ratios of `2.978e-3` on A800 and `2.937e-3` on 910B, with `db` at `2.904e-3/2.728e-3`. Because each rank's parameter gradients are rounded to BF16 before the test reduction, `dw/db` use the frozen A800 `1.10x` envelope (`3.3e-3`); output and `dx` retain the strict `1e-3` gate.
- The existing Ascend forward uses `BD=8,BT=32`; backward uses `BD=8,BT=8` at the target shape and materializes FP32 partial `dw/db` workspaces before `torch.sum`.
- `tests/context_parallel/test_cp_conv.py` is platform neutral for NCCL/HCCL, uses strict finite RMS assertions, dynamic rendezvous ports, and contains BF16 target-shape gates.
- `benchmarks/cp/benchmark_conv_cp.py` separates compilation from timing and reports max-rank latency, percentiles, CV, throughput, and peak memory.

The frozen single-rank `fwd_bwd` benchmark at `Tlocal=2048,D=3072,W=4,BF16,SiLU` produced:

| Platform | Samples | Compile | p20 | p50 | p80 | CV | Peak HBM |
| -------- | ------: | ------: | --: | --: | --: | -: | -------: |
| A800 | 50 | 0.500 s (warm cache) | 1.151 ms | 1.165 ms | 1.189 ms | 6.42% | 86.0 MiB |
| 910B | 10 | 18.934 s (cold cache) | 110.825 ms | 111.294 ms | 111.698 ms | 0.61% | 111.2 MiB |

The aggregate median gap is `95.6x`. The A800 10-sample run exceeded the 5% CV gate, so the prescribed single 50-sample confirmation was run; its remaining 6.42% variation is reported rather than hidden by further repetitions. The earlier synchronized component measurements (`0.111/0.879 ms` versus `20.028/90.653 ms`) remain the frozen forward/backward decomposition.

Subsequent stages append every kept or rejected candidate, correctness result, latency, memory result, and commit SHA here.
