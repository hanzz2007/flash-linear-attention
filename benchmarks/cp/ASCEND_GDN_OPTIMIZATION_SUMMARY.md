# FLA GDN Context Parallel：Ascend 910B 优化总结报告

> 报告范围：`fla/ops/cp` 的 GDN Triton-Ascend 迁移、正确性闭环与 CP2/CP4/CP8 性能优化。
> 对标平台：Ascend 910B × 8 与 NVIDIA A800 × 8。
> 最终结论：实现、正确性、双精度控制和 8 卡可复现性已完成；910B 绝对性能仍未达到 A800。

## 1. 执行摘要

本工作完成了 GDN Context Parallel（CP）预处理在 Ascend 910B 上的专用 Triton 实现，包括本地前向状态 `H`、前向转移 `M`、本地反向状态 `dH`、反向转移 `dM`、HCCL 汇聚后的有序 rank merge，以及 CUDA 路由保持。KDA 与 DPLR/RWKV7 完成了本任务范围内的 CP correctness fallback，但没有进入性能优化阶段。

工程上采用了业界常见的“先冻结契约、再建立基线、每次只改一个变量、正确性先行、端到端决定是否提升、失败候选立即回退”的内核优化闭环。最终保留两个运行模式：

- `high`：默认高精度路径，H/dH/M/dM 状态、HCCL payload 与 rank merge 保持 FP32 精度语义。
- `a800`：显式 opt-in，仅对 BF16、`K=V=128,Tlocal=2048` 的 backward dM 两次 contraction 使用原生 BF16 Cube；其他形状自动回退 `high`。

主要结果：

| 类别 | 结果 |
| --- | --- |
| 功能 | GDN CP2/CP4/CP8 前反向完成；KDA、DPLR/RWKV7 correctness-only 完成 |
| 公共精度 | CP8 `a800` 最差 output/gradient RMS ratio 为 `1e-5`，门限 `<3e-3` |
| 高精度路径 | 独立 primitive、NaN poisoning、尾块、K256、FP16/BF16 门禁通过 |
| `a800` dM 精度 | CP8 dM RMS ratio `3.436e-3`，约为 A800 `2.139e-2` 的 16.1% |
| `a800` 局部收益 | CP8 backward 配对测量改善 8.4%–8.6% |
| 最终 910B 性能 | CP2/4/8 forward `1.828/1.363/1.237 ms`；backward `2.293/1.477/1.052 ms` |
| A800 对标效率 | 910B 吞吐为 A800 的 31.9%–43.4%；绝对性能门槛未通过 |
| 扩展效率 | CP4→CP8 backward 为 70.2%，优于 A800 的 67.2%；forward 为 55.1%，落后 10.8 pp |

## 2. 目标、范围与冻结契约

### 2.1 主性能形状

```text
dtype       = BF16
Tglobal     = 16384
H = HV      = 8
K = V       = 128
chunk size  = 64
CP size     = 2 / 4 / 8
```

CP2/CP4 优先从物理 2 号卡开始；CP8 独占物理 0–7 共 8 张卡。

### 2.2 验收门槛

| 门槛 | 约束 |
| --- | --- |
| 公共 GDN output/全部 gradients | RMS ratio `<3e-3`，结果必须有限 |
| `high` H/dH/M/dM 与 merge | 独立 reference RMS ratio `<1e-4` |
| `a800` 内部路径 | 同形状误差不超过实测 A800 的 1.10 倍，并通过公共门禁 |
| 910B latency | 同 CP、同形状 `<=1.10x` A800 |
| 910B throughput | `>=90%` A800 |
| A800 CUDA regression | `<=2%` |
| CP4→CP8 scaling efficiency | 相比 A800 落后不超过 10 个百分点 |

### 2.3 不变量

- public API、返回 shape/dtype、rank 顺序不变。
- HCCL payload、wire shape、collective 数量不变。
- CUDA 默认路径和 intracard 行为不变。
- 编译和 autotune 必须在计时前完成。
- 候选使用 5 次预热、20 次采样；最终证据使用 5 次预热、30 次采样。
- 分布式延迟按同步 wall-clock 测量，并以最慢 rank 的 MAX 为准。

## 3. 运行环境与测量方法

| 平台 | 运行栈 | 用途 |
| --- | --- | --- |
| A800 × 8 | `torch211_cu128`，实际 PyTorch 2.11 CUDA 栈 | CUDA 正确性与性能基线 |
| Ascend 910B3 × 8 | CANN 9.0.0、torch_npu 2.7.1.post6、triton-ascend 3.2.1 包 | Triton-Ascend 实现与 CP2/4/8 验证 |

测量遵循以下原则：

1. 首次编译、cache 构建和 reference 编译不进入性能样本。
2. 每个样本前后执行设备同步；多卡结果通过 collective 取最慢 rank。
3. correctness 与 performance 使用相同目标 shape，但 correctness 额外覆盖 varlen、GVA、V-first、尾块、非零 BOS、K256 和 FP16。
4. 每个候选先跑独立 reference 和 poisoning gate；数值失败则不计时。
5. microbenchmark 只用于定位，只有端到端改善才能进入生产路径。

## 4. 最终方案

### 4.1 数据流

```text
                         ┌───────────────────────────┐
local q/k/w/u/g/dv/do ──►│ gate factor precomputation│
                         └─────────────┬─────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     │                                   │
                     ▼                                   ▼
        persistent H / fused dH scan      factorized persistent M / dM
        FP32 recurrent state              high: FP32 contraction
        BF16/FP16 input casts              a800 dM: BF16 Cube contraction
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
                           hm / dhm [HV,K,V+K], FP32
                                       │
                                       ▼
                      HCCL all_gather_into_tensor, FP32
                                       │
                                       ▼
                     persistent ordered rank-chain merge, FP32
                                       │
                                       ▼
                            initial_state / dht boundary
```

### 4.2 UB 感知的 kernel 分解

910B UB 为 192 KiB，单个 FP32 `256×256` 矩阵已经占用 256 KiB，不能照搬 GPU 上同时驻留多个 K×K tile 的实现。最终实现采用：

- H/dH 按 value tile 处理，状态保留 FP32。
- M/dM 使用 `B @ M`、再 `Aᵀ @ TMP` 的因式分解，避免完整 K×K 中间量驻留 UB。
- K≤128 使用方向相关的 tile 选择；K256 使用安全 fallback。
- host 侧平铺一维 grid，并对 program 数做分片，避免设备 grid 上限和地址溢出。
- pointer offset 在形成全局地址前转为 int64；循环索引保持 int32。

### 4.3 并发与融合

- 前向 H 与 M 放入独立 stream，并通过显式依赖保持消费顺序。
- backward 将 K=V=128 的 dH/dM 融合为共享输入加载的 16-block kernel。
- gate factor 独立预计算，避免每个 value tile 重复 `exp2`。
- 对 backward gate grid 做合并，减少 strided 小 kernel 开销。
- fully-overwritten summary 和单序列最终输出跳过冗余 zero fill；多序列与 boundary neutral state 仍保持正确初始化。

### 4.4 有序 rank merge

原逐 rank 多次 kernel 被替换为 persistent ordered rank-chain merge：

```text
state_out = rank_M @ state_in + rank_H_ext
```

normal layout 使用 BV32；V-first 保留经过验证的专用路径。merge 累加、跨 rank payload 和最终状态都保持 FP32，避免 rank 链上的误差放大。

### 4.5 双精度模式

```bash
# 默认、通用正确性路径
export FLA_ASCEND_CP_GDN_PRECISION=high

# 仅命中已验证的 BF16 K=V=128,Tlocal=2048 backward dM
export FLA_ASCEND_CP_GDN_PRECISION=a800
```

精度选择由 host 解析并作为 `tl.constexpr` 编译期特化传入。没有在 loop-carried dM 状态上使用运行时算术分支；不支持的 dtype、维度或序列长度自动回退 `high`。

## 5. 已提升优化及效果

以下数字来自同一阶段内的配对测量或当时冻结的相同 shape checkpoint。不同阶段的采样数可能不同，因此各项收益不可相加。

| 优化 | 作用位置 | 实测效果 | 决策 |
| --- | --- | ---: | --- |
| 64-column tile | H/M/dH/dM | CP2 forward -39.3%，backward -40.5% | 保留并继续方向化调优 |
| 方向相关 tile | H=128、dH=64、M/dM=128 | 兼顾前反向，避免单一 tile 的互相回退 | 保留 |
| H/M 双 stream | local forward/backward | CP2 forward -29.7%，backward -3.3% | 保留 |
| persistent rank-chain merge | CP8 merge | forward -14.3%，backward -11.4% | 保留 |
| merge BV16→BV32 | CP8 merge | 追加 forward -3.4%，backward -4.1% | 保留 |
| fused dH/dM | backward local summary | CP2/CP4/CP8 backward 约 -21%/-22%/-15% | 保留 |
| gate factor precompute/coalesce | forward/backward | 消除 tile 间重复 `exp2`；重算候选慢 86% | 保留预计算 |
| 跳过被覆盖的 zero fill | summary/output | CP2 -4.6%/-4.1%；CP8 backward -1.9% 至 -2.7% | 保留 |
| `a800` BF16 dM 双 contraction | CP8 backward | 配对改善 8.4%–8.6% | 仅目标 shape 提升 |

作为优化轨迹参考，早期相同 CP2 主形状的可运行 serial checkpoint 为 forward/backward `8.966/10.976 ms`，最终 5/30 矩阵为 `1.828/2.293 ms`，分别约为 4.91×/4.79× 的延迟缩短。该对比用于展示迭代量级；由于采样协议不同，不作为最终 A800 对标值。

## 6. 最终性能与效率

### 6.1 延迟

单位为毫秒，均为最慢 rank wall-clock median；最终矩阵为 5 次预热、30 次采样。

| CP | A800 fwd | 910B fwd | 910B/A800 | A800 bwd | 910B bwd | 910B/A800 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.793 | 1.828 | 2.306× | 0.861 | 2.293 | 2.664× |
| 4 | 0.520 | 1.363 | 2.623× | 0.559 | 1.477 | 2.645× |
| 8 | 0.394 | 1.237 | 3.139× | 0.416 | 1.052 | 2.531× |

### 6.2 吞吐保留率

同 token 数下，吞吐保留率等价于 `A800 latency / 910B latency`。下表使用未截断的原始样本统计；若用上一节仅显示三位小数的 median 反算，个别结果可能相差 0.1 个百分点。

| CP | Forward | Backward |
| ---: | ---: | ---: |
| 2 | 43.4% | 37.5% |
| 4 | 38.1% | 37.8% |
| 8 | 31.9% | 39.5% |

目标为 `>=90%`，因此全部绝对吞吐门槛均未通过。

### 6.3 CP4→CP8 强扩展效率

定义：

```text
efficiency(CP4→CP8) = latency(CP4) / (2 × latency(CP8))
```

| 平台 | Forward | Backward |
| --- | ---: | ---: |
| A800 | 65.9% | 67.2% |
| 910B | 55.1% | 70.2% |
| 910B - A800 | -10.8 pp | +3.0 pp |

backward 扩展效率通过门槛；forward 比允许的 10 pp 差距多 0.8 pp。

### 6.4 A800 效率差距拆解

本节的“效率”特指同工作负载下的相对吞吐保留率与分布式强扩展效率，不代表理论 FLOPS 利用率、HBM 带宽利用率或能效；当前没有在两种硬件上采集可直接对比的峰值计数器。

#### 6.4.1 绝对效率差距

延迟差值按表格中展示的三位小数 median 计算；latency ratio 和 throughput retention 使用最终未截断样本的冻结统计。

| CP | Fwd 延迟差 | Fwd ratio | Fwd 保留率 | Fwd 效率损失 | Bwd 延迟差 | Bwd ratio | Bwd 保留率 | Bwd 效率损失 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | +1.035 ms | 2.306× | 43.4% | 56.6 pp | +1.432 ms | 2.664× | 37.5% | 62.5 pp |
| 4 | +0.843 ms | 2.623× | 38.1% | 61.9 pp | +0.918 ms | 2.645× | 37.8% | 62.2 pp |
| 8 | +0.843 ms | 3.139× | 31.9% | 68.1 pp | +0.636 ms | 2.531× | 39.5% | 60.5 pp |

相对 `>=90%` 的验收线，forward 仍差 `46.6 / 51.9 / 58.1 pp`，backward 仍差 `52.5 / 52.2 / 50.5 pp`。因此 backward 扩展效率通过并不等价于 backward 已接近 A800：它只是随 CP 增大缩短差距，绝对吞吐仍只有 A800 的 39.5%。

#### 6.4.2 扩展效率差距

CP2→CP4 使用展示 median 推导，标记为近似值；CP4→CP8 使用最终冻结统计。

| 扩展区间 | A800 Fwd | 910B Fwd | 差距 | A800 Bwd | 910B Bwd | 差距 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CP2→CP4 | ≈76.3% | ≈67.1% | ≈-9.2 pp | ≈77.0% | ≈77.6% | ≈+0.6 pp |
| CP4→CP8 | 65.9% | 55.1% | -10.8 pp | 67.2% | 70.2% | +3.0 pp |

从 CP2 到 CP8，A800/910B forward 总加速约为 `2.01× / 1.48×`，相对理想 4× 的效率约为 `50.3% / 36.9%`；backward 总加速约为 `2.07× / 2.18×`，对应 `51.7% / 54.5%`。这解释了两个方向不同的 ratio 趋势：

- forward：910B/A800 latency ratio 从 CP2 的 `2.306×` 扩大到 CP8 的 `3.139×`。除了局部算子更慢，fixed HCCL/merge 开销在 local T 下降时占比上升，且 910B forward 两段扩展效率都低于 A800。
- backward：ratio 从 `2.664×` 缩小到 `2.531×`。fused dH/dM 与 CP8 `a800` dM 特化生效，且 910B 两段扩展效率均略好于 A800；剩余问题主要是绝对 local latency，而不是多卡 scaling。

#### 6.4.3 CP8 目标闭合量

| 指标 | Forward | Backward |
| --- | ---: | ---: |
| 当前 910B CP8 | 1.237 ms | 1.052 ms |
| A800+10% 预算 | 0.433 ms | 0.457 ms |
| 仍需减少 | 0.804 ms / 65.0% | 0.595 ms / 56.6% |
| local summary / 总预算 | 1.96× | 1.62× |
| communication+merge / 总预算 | 1.11× | 1.08× |
| local 单项压到预算所需降幅 | 49.0% | 38.4% |
| communication+merge 压到预算所需降幅 | 10.2% | 7.7% |

这些 component median 属于不同 rank/阶段，不能相加；但 local 与 communication+merge 各自都已经超过总预算，足以得出一个稳健结论：单独优化任一侧都不能达到 A800+10%。forward 需要同时改善递推 local kernel 与分布式固定开销；backward 虽然 scaling 健康，也仍需要显著降低 local absolute latency，并继续削减通信/merge 尾部。

## 7. 精度与正确性

### 7.1 最终 CP8 `a800` 公共门禁

目标 shape 在全部 8 张 910B 上执行，每个 rank 都确认读取 `a800` selector：

| Tensor | RMS ratio | 相对 `<3e-3` 门限占用 |
| --- | ---: | ---: |
| output | 0 | 0% |
| dq | 0 | 0% |
| dk | 5e-6 | 0.17% |
| dv | 3e-6 | 0.10% |
| dg | 1e-5 | 0.33% |
| dbeta | 4e-6 | 0.13% |

### 7.2 覆盖矩阵

- GDN：CP2/CP4/CP8、dense/varlen、GVA、normal/V-first、rank boundary、tail chunk。
- dtype：BF16、FP16；FP32 小形状 vector fallback。
- shape：K≠V、K96/V80、K256/V64、BT16/32/64。
- 输入模式：预融合 gate、raw gate、beta sigmoid、q/k normalization。
- 安全性：NaN poisoning、有限值断言、大地址、non-zero BOS。
- KDA、DPLR/RWKV7：H/M/dH/dM/merge 与 CP4 rank chain correctness-only。
- CUDA：precision selector 与 CP4 normal/V-first 路由回归通过；最终 A800 六个性能点没有超过冻结基线 2%。

## 8. 关键瓶颈与性能上限

CP8、`Tlocal=2048` 的 production-equivalent 诊断：

| 组件 | Forward | Backward | 说明 |
| --- | ---: | ---: | --- |
| local summary | 0.849 ms | 0.742 ms | 包含实际 stream/fusion 调度 |
| FP32 all-gather | 0.353 ms | 0.353 ms | 约 1 MiB/rank |
| rank-chain merge | 0.284 ms | 0.286 ms | 独立组件测量 |
| communication + merge | 0.482 ms | 0.495 ms | 实际组合；不能与 local 最大值直接相加 |
| A800+10% 总预算 | 0.433 ms | 0.457 ms | 由最终 A800 CP8 延迟推导 |

需要特别注意：stage maxima 不可简单相加，因为 boundary rank 可能跳过 local summary，而承担最长 merge 的 rank 也不同。但仅 FP32 all-gather 已消耗 forward 预算的 81%、backward 预算的 77%；同时 local summary 单项也高于总预算。因此只优化 H/dH 或只调整 HCCL 环境变量都不足以达到 A800。

HCCL 实测：

| CANN 9.0 mode | CP8 all-gather | 结论 |
| --- | ---: | --- |
| HOST（默认） | 0.353 ms | 最快的训练支持模式 |
| AIV | 0.315 ms | 更快，但官方限定推理且有拓扑/多 communicator 约束 |
| AI_CPU | 0.427 ms | 回退 |
| HOST_TS | 1.139 ms | 显著回退 |

`HCCL_BUFFSIZE` 默认 200 MiB，已远大于 payload；`HCCL_ALGO` 不提供单机 level-0 算法选择，因此没有可提升的训练态环境变量配置。

## 9. 被否决方案与经验

| 候选 | 结果 | 否决原因 |
| --- | --- | --- |
| forward FP16/BF16 Cube transition | isolated M 快 4.4%/5.7% | 与 Cube-heavy H 竞争，CP8 端到端慢 1.2%/7.5% |
| 四段 H/dH associative scan | CP8 快约 6%/11% | H/dH `2.47e-3/2.44e-3`，超过 A800+10% envelope |
| 两段 H/dH scan | M/dM 约 `4e-7` | H/dH `1.771e-3/1.810e-3` 仍超门限 |
| segmented M only | isolated M 快 26.4% | CP8 端到端慢 15.8%，增加资源竞争 |
| fused forward H/M | 小 tail 通过 | T=2048 H ratio `1.92e-4`，超过 high `<1e-4` |
| num_warps 2/8、multibuffer、unroll | arithmetic 不变 | 无端到端收益或 backward 回退 |
| H-only BV64 | arithmetic 不变 | local forward 慢 6.4% |
| backward kernel 内重算 gate | arithmetic 不变 | 16 个 output tile 重复 exp2，慢 86% |
| async all-gather + immediate wait | 结果有限 | 没有可重叠工作，慢 4.7% |
| 预分配 all-gather output | 协议不变 | 0.404 ms vs 0.353 ms，尾延迟更差 |
| FP16/BF16 merge | FP16 精度可接受 | 都比 FP32 merge 慢；BF16 还超精度 cap |
| AIV HCCL mode | 0.315 ms | 仅推理支持，不进入训练默认路径 |

核心经验：

1. Ascend 上 isolated Cube microbenchmark 的胜利可能破坏 Vector/Cube overlap，必须以多 stream 端到端结果为准。
2. 递推 kernel 的并行分段会改变低精度 state cast 顺序；代数等价不代表浮点等价。
3. loop-carried state 上的运行时算术分支在当前编译器上出现过 silent corruption，精度模式必须用编译期特化。
4. 降低 wire/merge dtype 不一定更快；转换、低 program count 和执行单元匹配同样重要。
5. 负结果必须记录并回退，不能用放宽公共门限掩盖内部误差。

## 10. 工程产出与阶段提交

相对基线 `ebf3a0c`，分支共修改 10 个 tracked 文件，新增约 3,466 行、删除 71 行。

| 阶段 | 代表提交 | 产出 |
| --- | --- | --- |
| Ascend 前向与路由 | `65aa7c16` | GDN H/M 与 merge 初版 |
| Device-neutral tests | `f342ada1`, `492083b1` | NCCL/HCCL 公共测试与覆盖扩展 |
| non-GDN fallback | `1598620b` | KDA/DPLR correctness-only |
| 可复现 benchmark | `dd25ad84` | 平台无关 CP preprocess 基准 |
| 核心性能迭代 | `3633558c` | tile、stream 与 persistent transition |
| merge 优化 | `c2b4bba9`, `60b473de`, `d3df8cc5` | persistent chain、BV32、V-first |
| backward 融合 | `8e1e7eeb` | fused dH/dM |
| gate 与 memset | `8080b182`, `7ddd5a84`, `5413f17c`, `b8ac64d5` | gate 预计算/合并、zero-fill 消除 |
| 双精度模式 | `419e244b` | `high/a800` selector 与 BF16 dM 特化 |
| 矩阵、profiling 与审计 | `6a11f11d`, `a1e53be1`, `1a6788a8`, `ceefc1ea` | 最终数据、瓶颈、负结果和 CP8 gate |

主要文件：

- `fla/ops/cp/backends/triton_ascend/chunk_delta_h.py`
- `tests/context_parallel/test_cp_gdn.py`
- `tests/context_parallel/test_cp_gdn_preprocess.py`
- `benchmarks/cp/benchmark_gdn_cp_preprocess.py`
- `benchmarks/cp/ASCEND_GDN_OPTIMIZATION.md`

## 11. 结论与后续建议

### 已完成

- 完整 GDN CP Triton-Ascend 数据通路。
- 高精度默认分支和受控 `a800` 性能分支。
- 8 卡 CP2/4/8 correctness、性能和扩展效率证据。
- A800 CUDA 路由和性能回归保护。
- KDA/DPLR/RWKV7 correctness fallback。
- 失败候选、数值边界和 HCCL 限制的可审计记录。

### 未完成

- 910B latency `<=1.10x` A800。
- 910B throughput `>=90%` A800。
- CP4→CP8 forward efficiency 与 A800 的差距仍为 10.8 pp，超门槛 0.8 pp。

### 推荐下一阶段

在不改变当前冻结协议的条件下，已有证据不支持继续通过小幅 tile 或精度调节达到 A800。若允许扩大范围，建议按以下顺序开展独立阶段，并继续保留 `high`：

1. 仅对 opt-in `a800` 模式研究压缩 summary wire、FP32 解包与 merge；先做 A800-relative numerical gate。
2. 评估 hierarchical/pipelined scan 或自定义 fused collective，减少 all-gather 后的重复 rank-chain work。
3. 在更新的 CANN/triton-ascend 编译器上重测 persistent scan、FP32 dot 和 stream overlap。
4. 任何协议变更都必须重新跑 CP2/4/8 公共前反向、NaN poisoning、A800 regression 与 5/30 最终矩阵。

不建议把两段/四段 H/dH scan 仅凭公共 `<3e-3` 门限直接提升，因为其内部状态误差已经明显超过“略低于 A800”的约束。

## 12. 复现命令

### 12.1 910B CP8 performance

```bash
ssh -p 2225 qiuhan@localhost
source /data/Ascend/9.0.0/cann-9.0.0/set_env.sh
export PATH=/data/qiuhan/3.conda/swift-mmu-beta-v0.2-cann900_triton321/bin:$PATH
cd /data/qiuhan/2.project/fla_cp
export PYTHONPATH=$PWD
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_UB_CAPACITY_BITS=1572864
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export FLA_ASCEND_CP_GDN_PRECISION=a800

torchrun --standalone --nproc-per-node=8 \
  benchmarks/cp/benchmark_gdn_cp_preprocess.py \
  --total-seq-len 16384 --q-heads 8 --v-heads 8 \
  --key-dim 128 --value-dim 128 --chunk-size 64 \
  --dtype bfloat16 --direction bwd --warmup 5 --samples 30
```

forward 使用相同命令并改为 `--direction fwd`。

### 12.2 910B CP8 `a800` correctness

```bash
python -m pytest -q -s \
  tests/context_parallel/test_cp_gdn.py::test_cp8_a800_precision_path
```

### 12.3 A800 baseline

```bash
ssh -p 2223 qiuhan@localhost
source /home/qiuhan/miniforge3/etc/profile.d/conda.sh
conda activate torch211_cu128
cd /data/qiuhan/2.project/fla_cp
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nproc-per-node=8 \
  benchmarks/cp/benchmark_gdn_cp_preprocess.py \
  --total-seq-len 16384 --q-heads 8 --v-heads 8 \
  --key-dim 128 --value-dim 128 --chunk-size 64 \
  --dtype bfloat16 --direction bwd --warmup 5 --samples 30
```

## 13. 参考资料

- 详细逐阶段证据：[`ASCEND_GDN_OPTIMIZATION.md`](ASCEND_GDN_OPTIMIZATION.md)
- CP 数学与 precision mode：[`../../fla/ops/cp/README.md`](../../fla/ops/cp/README.md)
- CANN 9.0 `HCCL_OP_EXPANSION_MODE`：[官方文档](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0096.html)
- CANN 9.0 `HCCL_BUFFSIZE`：[官方文档](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0080.html)
- CANN 9.0 `HCCL_ALGO`：[官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0079.html)
