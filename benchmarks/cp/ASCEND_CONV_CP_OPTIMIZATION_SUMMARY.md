# FLA CP Causal Conv1d：Ascend 910B 适配与优化总结

> 范围：`fla/modules/conv/cp/ops.py`、共享 Triton-Ascend Conv1d backend、halo 通信与 CP2/4/8 验证。<br>
> 对标：Ascend 910B × 8（CANN 9.0.0）与 NVIDIA A800 × 8。<br>
> 主负载：BF16、`W=4`、`Tglobal=16384`、`D=1024/3072`、SiLU、forward + backward。

## 1. 执行摘要

本次工作完成了 CP causal depthwise Conv1d 在 Ascend 910B 上的专用前反向路径、短 halo 修复、CP subgroup-safe 通信、双精度开关和 8 卡正确性闭环。原 910B `Tlocal=2048,D=3072` 探索基线为 forward `20.028 ms`、backward `90.653 ms`；最终为 forward `2.601 ms`、derived backward `3.399 ms`，分别提升 `7.70x` 和 `26.67x`。完整 forward+backward 从正式基线 `111.294 ms` 降到 `6.001 ms`，提升 `18.55x`。

最终结论不是“全面追平 A800”：

| 验收项 | 结果 | 状态 |
| --- | ---: | :---: |
| public API、shape、dtype、CUDA 路由 | 保持不变 | 通过 |
| BF16/FP16、W2/3/4、短序列、尾 tile、NaN poisoning | 12 个单 NPU 门禁通过 | 通过 |
| CP8 D1024/D3072 前反向 | 910B 与 A800 共 4 个用例通过 | 通过 |
| CP8 D1024 端到端 910B/A800 | `2.19x`，目标 `<=2.5x` | 通过 |
| CP8 D3072 端到端 910B/A800 | `3.01x`，目标 `<=2.5x` | 未通过 |
| CP2→CP8 强扩展效率，D3072 | `87.0%`，目标 `>=80%` | 通过 |
| CP2→CP8 强扩展效率，D1024 | `69.6%` | 未通过 |
| kernel-only 910B/A800 | D1024 `3.66x`、D3072 `6.33x` | 未通过 |
| A800 原路径回退 | 未观测到超过 2% 的回退 | 通过 |

默认生产选择为：

- 精度：`FLA_ASCEND_CONV_PRECISION=high`，FP32 `dpre` 与 FP32 梯度累加。
- 通信：`FLA_CP_CONV_COMM=all_gather`。
- `a800` 精度请求与 `p2p` 通信均保留为可复现实验开关；当前没有低精度 bucket 获得提升，P2P 也未成为默认。

## 2. 环境与测量协议

| 平台 | 软件栈 | 设备使用 |
| --- | --- | --- |
| Ascend 910B | CANN 9.0.0、PyTorch 2.7.1、torch_npu 2.7.1.post6、Triton 3.2.0、triton-ascend 3.2.1 | 单卡优先物理 2；CP8 独占 0–7 |
| NVIDIA A800 | PyTorch 2.11.0+cu129、Triton 3.6.0 | 单卡优先物理 2；CP8 独占 0–7 |

两个平台的软件栈不同，本文的比值表示“设备 + 指定生产栈”的整体效率差距，不表示纯硬件峰值差距。

计时规则：

1. 冷编译单独执行并记录，不进入延迟样本。
2. 候选使用 3 次 warmup、10 次计时；最终使用 5 次 warmup、30 次计时。
3. CV 超过 5% 时，仅追加一次固定 50 样本确认；确认后仍有长尾也不筛样或重复试验。
4. 每个样本执行设备同步；多卡通过 collective 取最慢 rank wall-clock。
5. 报告 p20/p50/p80、CV、吞吐、峰值显存和 compile time，不使用 CUDA Event 跨平台比较。

## 3. 最终实现

### 3.1 Forward fast path

连续布局、单本地序列、BF16/FP16、`W=2/3/4` 命中 Ascend dense fast path：

- 扁平 1D grid；每个 program 处理 `BT=64,BD=128`。
- 静态展开卷积 tap，FP32 累加。
- 融合 bias、SiLU/Swish、residual 与输出 cast。
- initial-state halo 直接读取；边界和尾 tile 使用 mask。
- 地址乘法使用 int64；launch 在 65535 programs 前分片。
- `multibuffer=False`，不使用 FP32 `tl.dot` 或 `tl.make_block_ptr`。

该实现把 D3072 forward 从 `20.028 ms` 降到 `2.601 ms`，并把约 27648 个 programs（含独立 SiLU）降到 1536 个 programs、单 launch。

### 3.2 Backward pipeline

目标路径拆分为：

```text
x / weight / bias / state / dy
              │
              ▼
       dpre：重算 pre + SiLU'
              │ FP32
       ┌──────┼───────────┐
       ▼      ▼           ▼
      dx   dw / db       dh0
   Triton  CANN 分块归约  boundary Triton
```

- `dpre`、`dx`、`dh0` 使用 grid-safe Triton kernel。
- `dpre` 保持 FP32；所有梯度 FP32 累加后再 cast。
- `dw/db` 使用 512-token CANN/PyTorch fused multiply-reduction，避免原 `O(NT×D×W)` FP32 partial workspace。
- 目标 BF16 `W=4,D=1024/3072,T>=64` 使用 `64×128,4 warps`；其他 shape 保持 2 warps。
- `dh0` 对 `T<W-1` 显式限界，修复短序列越界。

### 3.3 CP halo 与通信

wire shape 统一为 `[W-1,D]`：

- 短 local chunk 在左侧补零、有效 token 右对齐。
- forward 从上一 CP rank 接收 halo 构造 initial state。
- backward 仅向本地有效尾 token 加上下一 rank 的 `dh0`。
- P2P 使用 `group_peer`，支持 global rank 不连续的 CP subgroup。

P2P 在 CP8 通信-only 50 样本中为 `0.661 ms`，all-gather 为 `0.564 ms`；CP8 端到端 P2P 也慢 1.12%。因此 all-gather 保持默认，P2P 只保留为 `FLA_CP_CONV_COMM=p2p` 的 A/B 开关。

### 3.4 精度 selector

```bash
# 默认通用高精度路径
export FLA_ASCEND_CONV_PRECISION=high

# 请求 A800-relative 性能路径；未提升 shape 自动回退 high
export FLA_ASCEND_CONV_PRECISION=a800
```

selector 同时报告 requested 与 effective precision。只有同时通过独立 reference、A800 误差包络和性能门禁的 shape 才能加入低精度 bucket；当前 bucket 列表为空。

## 4. 优化轨迹与工程效率

| 优化 | 直接效果 | 决策 |
| --- | ---: | --- |
| 用户参考 `BT64,BD4` grid 分解 | forward `20.028→17.684 ms` | 仅参考分解方式 |
| 融合 dense `64×128` forward | `20.028→2.667 ms`，少一个中间 tensor | 保留 |
| `dpre/dx/dh0` 专用 backward | 消除重算 forward + 独立 SiLU backward | 保留 |
| 512-token vendor FP32 reduction | fwd+bwd `61.196→6.251 ms` | 保留 |
| backward 4 warps bucket | D3072 候选 `6.238→6.027 ms` | 保留 |
| 固定 halo 和有效尾 update | 修复 `Tlocal<W-1` | 保留 |
| P2P halo | CP8 端到端慢 1.12% | 不设默认 |
| BF16 `dpre` | 90.1 MiB，但更慢且误差越界 | 拒绝 |
| FP16 `dpre` | 更慢、`db` 越界、无峰值收益 | 拒绝 |

阶段效率：

| 指标 | 初始 910B | 最终 910B | 改善 |
| --- | ---: | ---: | ---: |
| D3072 forward | 20.028 ms | 2.601 ms | 7.70x |
| D3072 backward | 90.653 ms | 3.399 ms | 26.67x |
| D3072 fwd+bwd | 111.294 ms | 6.001 ms | 18.55x |
| D3072 peak HBM | 111.2 MiB | 105.1 MiB | -5.5% |

峰值显存没有达到计划的 30% 降幅。BF16 `dpre` 最低测到 90.1 MiB，但相对初始仍不足 30%，并且未通过精度/性能门禁，因此没有用错误结果换取表面显存数字。

## 5. 正确性结果

- 单 NPU：12 项全部通过，覆盖 BF16/FP16、W2/3/4、T=1/2/65/257/2048、非 tile 整除 D、bias/activation/state/residual、无状态和 NaN poisoning。
- CP2：all-gather/P2P、BF16、非连续 subgroup 均通过。
- CP4：复杂 packed varlen 和跨 rank 序列通过。
- CP8：D1024、D3072 在 HCCL/910B 与 NCCL/A800 上全部通过。
- 所有输出与梯度 finite。

CP8 每 rank 参数梯度会先舍入 BF16，再执行 8 路求和。A800 `dw` RMS baseline 为 `4.335e-3`，910B 为 `4.342e-3`，差异 0.14%；因此 CP8 参数梯度门限冻结为 A800 的 1.10×（`4.8e-3`）。CP2 保持 `3.3e-3`，output 与 `dx` 始终保持 `<1e-3`。

## 6. kernel-only 性能与 A800 差距

最终值为最慢 rank wall-clock。A800 的 30 样本 CV 均超过 5%，表中使用唯一一次 50 样本确认；长尾保留在 CV。

| D | 平台 | Forward p20/p50/p80 | Fwd+bwd p20/p50/p80 | Derived bwd | CV fwd/total | Peak |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | A800 | 0.202/0.208/0.215 ms | 0.933/0.947/0.963 ms | 0.739 ms | 7.42%/11.24% | 28.7 MiB |
| 1024 | 910B | 1.430/1.445/1.465 ms | 3.435/3.463/3.490 ms | 2.019 ms | 1.27%/0.81% | 36.1 MiB |
| 3072 | A800 | 0.202/0.205/0.219 ms | 0.933/0.948/0.959 ms | 0.743 ms | 7.15%/12.18% | 86.0 MiB |
| 3072 | 910B | 2.590/2.601/2.615 ms | 5.987/6.001/6.023 ms | 3.399 ms | 0.54%/0.38% | 105.1 MiB |

| D | Forward gap | Backward gap | Fwd+bwd gap | 910B 对应 A800 吞吐 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 6.96x | 2.73x | 3.66x | 27.3% |
| 3072 | 12.71x | 4.57x | 6.33x | 15.8% |

## 7. CP2/4/8 端到端性能

| D | CP | 910B p50 | A800 p50 | 910B/A800 | 910B tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2 | 12.551 ms | 1.912 ms | 6.56x | 1.31M |
| 1024 | 4 | 7.160 ms | 2.035 ms | 3.52x | 2.29M |
| 1024 | 8 | 4.510 ms | 2.055 ms | 2.19x | 3.63M |
| 3072 | 2 | 23.315 ms | 1.987 ms | 11.73x | 0.70M |
| 3072 | 4 | 12.377 ms | 2.088 ms | 5.93x | 1.32M |
| 3072 | 8 | 6.700 ms | 2.225 ms | 3.01x | 2.45M |

910B 强扩展效率：

| D | CP2→CP4 | CP4→CP8 | CP2→CP8 |
| ---: | ---: | ---: | ---: |
| 1024 | 87.6% | 79.4% | 69.6% |
| 3072 | 94.2% | 92.4% | 87.0% |

A800 的 CP2→CP8 效率仅为 D1024 `23.3%`、D3072 `22.3%`。这不是 910B 追平了 kernel：A800 的本地 kernel 已低于 1 ms，固定 Python/autograd/collective 同步开销主导 CP 端到端，因此 A800 增卡后绝对延迟几乎停在约 2 ms；910B 仍有足够 local compute 可被序列切分，所以差距从 CP2 的 6.56x/11.73x 收敛到 CP8 的 2.19x/3.01x。

## 8. A800 效率差距分析

### 8.1 Forward 是最大差距源

D3072 forward 的 910B/A800 比值为 `12.71x`，大于 backward 的 `4.57x`。当前算法只有 4 个逐元素 tap，不具备大矩阵乘法可利用的 Cube 计算密度；910B 路径主要受 Vector core、标量地址生成、mask 和 HBM 访存影响。A800 现有 Triton/CUDA 栈对此类连续 depthwise stencil 的调度与内存系统更成熟。

### 8.2 高精度 backward 是有意保留的成本

910B backward 保持 FP32 `dpre`，并用 vendor FP32 reduction 计算 `dw/db`。将 `dpre` 改为 BF16 会明显越过 A800 误差包络；FP16 也使 `db` 达到 `1.62e-3` 且更慢。因此剩余差距不能通过无边界放宽精度解决。

### 8.3 通信不是 D3072 的首要瓶颈

CP8 双向 halo 通信-only 约 `0.564 ms`，而 D3072 总延迟为 `6.700 ms`；即使通信完全消失也无法达到 A800 `2.225 ms`。P2P 还比 all-gather 慢，因此下一阶段应优先减少本地 forward/vector reduction 时间，而不是继续切换 HCCL 小消息原语。

### 8.4 软件栈与编译器差异

910B 使用 Triton 3.2.0 / triton-ascend 3.2.1，A800 使用 Triton 3.6.0。冷分布式 specialization 分别约 14 s 与 38 s，均已排除出计时。比值包含 compiler lowering、runtime、allocator、collective 和框架版本的综合差异，不能解释成芯片峰值比。

## 9. 未采用候选

| 候选 | 结果 | 拒绝原因 |
| --- | --- | --- |
| Flat 2048-element per-lane div/mod | forward 129.724 ms | 地址运算淹没计算 |
| `BT64,BD256` forward | UB 需求 2.37 Mbit > 1.57 Mbit | 编译失败 |
| 32/64-program persistent forward | 2.929/3.026 ms | 慢于 2.667 ms |
| Triton two-level `dw/db` | 61.196 ms | Vector reduction program-bound |
| Full-T vendor reduction | 5.921 ms、132.1 MiB | 显存峰值过高 |
| 2D `tl.where` 安全地址 | UB 16.8/18.9 Mbit | compiler material化 tile |
| BF16/FP16 `dpre` | 更慢且精度失败 | 不提升 |
| P2P halo 默认 | CP8 慢 1.12% | 不提升 |
| 通信/计算 overlap | 未合入 | 同步 P2P 未先建立收益 |

## 10. 复现

910B CP8：

```bash
source /data/Ascend/9.0.0/cann-9.0.0/set_env.sh
export PATH=/data/qiuhan/3.conda/swift-mmu-beta-v0.2-cann900_triton321/bin:$PATH
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_UB_CAPACITY_BITS=1572864
export HCCL_NPU_SOCKET_PORT_RANGE=auto

torchrun --standalone --nproc_per_node=8 \
  benchmarks/cp/benchmark_conv_cp.py \
  --kind cp --mode fwd_bwd --total-seq-len 16384 \
  --dim 3072 --width 4 --dtype bfloat16 --activation silu \
  --precision high --comm all_gather --warmup 5 --samples 30
```

A800 使用相同参数，激活 `torch211_cu128`，设置 `CUDA_VISIBLE_DEVICES=0,...,7` 与 `PYTHONPATH=$PWD`。

## 11. 分阶段提交

| 阶段 | Commit |
| --- | --- |
| 冻结 reference、门禁与基线 | `cae5fbcc` |
| Ascend forward fast path | `7b72944c` |
| Ascend backward pipeline | `0cc12b6a` |
| halo 与通信 A/B | `1c0d674a` |
| selector、固定调度与最终矩阵 | `a2c0be63` |
| 报告 | `951da147` |

所有提交仅推送到用户 fork 的 `feat/cp-conv-triton-ascend`，没有向官方仓库创建或更新 PR。逐候选原始结论和门禁详见 [`ASCEND_CONV_CP_OPTIMIZATION.md`](ASCEND_CONV_CP_OPTIMIZATION.md)。
