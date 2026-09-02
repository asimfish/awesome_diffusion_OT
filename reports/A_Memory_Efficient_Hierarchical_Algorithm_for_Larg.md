# A Memory-Efficient Hierarchical Algorithm for Large-scale OT (HALO)

> ⚠ 未读全文，依据摘要
> 作者（前 3 位 + et al.） · ICLR Poster 2026 · [OpenReview](https://openreview.net/forum?id=CkOBcyntGd) · 证据级 [A] · 课题 T29 高性能 OT 求解器与训练基础设施
> **一句话**：HALO 用层次多尺度 warm-start、active support 剪枝与一阶 LP 求解器组合，报告 O(n) 内存与 1024² 图像上 8.9× 加速、省 70.5% 显存。

## 1. 问题

离散 OT 在大规模点云/图像上的精确求解长期受制于内存与时间：朴素 LP 或匈牙利法代价为 $O(n^3\log n)$，Sinkhorn 每迭代一次矩阵-向量积需要 $O(n^2)$ 时间与内存。这使 OT 在深度学习里只能以「小 batch 配对」或「低维近似」形态出现。HALO 针对的是大规模 OT 的内存瓶颈：如何在保持求解精度的前提下，把内存从 $O(n^2)$ 降到 $O(n)$，同时不牺牲速度。

## 2. 方法

调研 agent 给出的一句话贡献概括为：**层次多尺度 warm-start + active support 剪枝 + 因子化-free 一阶 LP 求解器（默认 cuPDLPx），O(n) 内存**。摘要未提供公式或算法步骤细节，原文截断，未见。

## 3. 理论结果

摘要未给出定理、引理或复杂度保证的正式陈述。调研 agent 报告「O(n) 内存」，但摘要原文未提供该复杂度声明的出处或证明条件，原文截断，未见。

## 4. 实验与数字

摘要未提供数据集、基线或数值。调研 agent 报告以下数字，但摘要原文未包含，原文截断，未见：

| 指标 | 数值 | 出处 |
|---|---|---|
| 1024² 像素图像加速比 | 8.9× | 调研 agent 报告，摘要未见 |
| 显存节省 | 70.5% | 调研 agent 报告，摘要未见 |

## 5. 在 OT×扩散地图中的位置

HALO 属于第三代 OT 求解器（2024–2026）中「低秩/层次与一阶 LP 两条线汇流」的代表：它把层次 warm-start、active 剪枝与 cuPDLPx 组合成 O(n) 内存的精确求解管线。与 HiRef（ICML 2025 Oral）同属「低秩从近似变构件」路线，但 HALO 明确引入一阶 LP 求解器（cuPDLPx）作为底层引擎，与 PDOT/cuPDLP 系一阶 LP 复兴直接衔接。在扩散/流匹配训练管线中，HALO 面向的是超大 batch 下 OT 配对成为真实瓶颈的场景（对应 Immiscible Diffusion、Haxholli、Boïté 量化的成本前提），但摘要未说明其是否可微或是否已接入训练管线。

## 6. 局限与批评

作者承认的局限：摘要未提供，原文截断，未见。

读出来的局限（基于摘要缺失与调研 agent 信息）：
1. 摘要未给出任何数值、基线或理论保证，无法独立核验「8.9× 加速、省 70.5% 显存」的测量条件（硬件、dtype、容差、对比基线）。
2. 调研 agent 报告默认求解器为 cuPDLPx，暗示性能依赖特定一阶 LP 求解器；摘要未说明该依赖对精度或收敛的影响。
3. 摘要未提及可微性。若 HALO 输出 plan 对输入点云/cost 的导数未实现，则无法进入需要梯度的 guidance 或训练场景（对应开放问题 #5）。

## 7. 对我们的启发

1. **大 batch OT 配对的成本账**：若 HALO 的 O(n) 内存与加速数字成立，可将其纳入「OT 配对在扩散训练里花多少钱」的统一 benchmark（开放问题 #3），与 Immiscible（22.8ms）、Haxholli（3.4–12%）、Boïté（3–55%）在同一 harness 下对比。
2. **异步重叠调度的候选求解器**：HALO 若为单卡求解器，可作为开放问题 #4 中「配对在独立 GPU stream 前瞻执行」的底层引擎候选，测 k=8192 时 overlap 前后的端到端吞吐差。
3. **可微性缺口**：若 HALO 未提供隐式微分，则「大规模 + 可微」需求（方向二）仍未被覆盖；可尝试推导其层次 warm-start + active 剪枝管线的隐函数定理形式，并入 OTT-JAX。

## 8. 资源

代码链接：未公开（摘要未提供）。OpenReview 页面：https://openreview.net/forum?id=CkOBcyntGd

相关论文互链：
- HiRef（ICML 2025 Oral）：低秩因子与 Monge map 共聚类不变量，递归细化出全秩双射。
- PDOT（2024）：restarted PDHG 做 matrix-free 精确 OT，GPU 上高精度区间反超 Sinkhorn。
- cuPDLPx：HALO 默认一阶 LP 求解器（调研 agent 报告）。
- FlashSinkhorn（ICML 2026 Oral）：IO-aware Sinkhorn，前向 9–32×、端到端最高 161×。
- FastSinkhorn（2026 预印本）：warp-level 归约原生 CUDA log-domain Sinkhorn。
