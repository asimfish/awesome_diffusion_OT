# Efficient Algorithms for Robust and Partial Semi-Discrete OT

> Agarwal, Raghvendra, Shirzadian, Yao · NeurIPS 2025 · [arXiv](https://arxiv.org/abs/None) · 证据级 [P] · 课题 T25 非平衡/部分 OT 在生成建模中的应用
> **一句话**：给出 α-partial 与 λ-TV-robust 半离散 OT 的 restricted Laguerre 刻画、互归约与精确/近似算法。

⚠ 未读全文，依据摘要

## 1. 问题

本文处理半离散最优传输（semi-discrete OT）的两个松弛变体：α-partial OT 与 λ-TV-robust OT。经典半离散 OT 要求源测度与目标测度质量守恒且全量匹配；partial OT 只运输 α 比例的质量，robust OT 用 TV 罚松弛边缘约束以容忍离群点。摘要未给出此前方法的具体不足，仅以「Efficient Algorithms」为题，指向这两个变体在算法效率上的缺口。

## 2. 方法

摘要未展开算法细节。调研 agent 给出的一句话贡献提到两个核心方法要素：

- **restricted Laguerre 刻画**：对 α-partial 与 λ-TV-robust 半离散 OT 给出受限 Laguerre 胞腔（restricted Laguerre cells）形式的解结构刻画。
- **互归约**：两个问题之间可互相归约，即 α-partial 与 λ-TV-robust 半离散 OT 在算法上可互相转化。

具体公式、算法步骤与复杂度在摘要中未见，原文截断，未见。

## 3. 理论结果

摘要未列出具体定理编号、假设或结论。仅能确认作者报告了 restricted Laguerre 刻画与互归约这两个理论性质；精确/近似算法的复杂度保证在摘要中未见，原文截断，未见。

## 4. 实验与数字

摘要未提供数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文属于半离散 OT 数值算法线，与 AAAI 2024 及 NeurIPS 2025 的 partial/robust 半离散算法同属底层数值工具供给。在课题 T25 的脉络中，其 α-partial 与 λ-TV-robust 互归约被背景材料点名为「尤其重要」：该等价关系可直接支撑污染率驱动的 τ 自适应调度（切入点 #2），即把估计污染率 ε 映射为 UOT 的 τ(t) 调度。同时，α-partial 半离散算法可作为 web 规模错配数据清洗-加权训练（切入点 #4）的 GPU 批量化基础。本文不直接涉及生成模型训练目标或动态几何，属于静态 OT 求解器层。

## 6. 局限与批评

作者承认的局限：摘要未提及，原文截断，未见。

读出来的局限：摘要未给出任何数值实验或复杂度数字，无法评估「Efficient」的实际含义；互归约的具体方向与代价（是否保复杂度、常数因子如何）在摘要中不可见。venue 为主会论文，但本报告仅依据摘要，结论强度受限。

## 7. 对我们的启发

1. **切入点 #2（污染率驱动的 τ 自适应调度）**：若 λ-TV-robust ≡ α-partial 的互归约成立，可把数据审计得到的污染率 ε 直接换算为 partial 质量比例 α，再映射为 UOTM/UOT-FM 的 τ(t) 调度，替代 UOTM-SD 的经验调度。
2. **切入点 #4（错配数据清洗-加权训练）**：α-partial 半离散算法若可 GPU 批量化，可用于图文对质量预算式训练，只保留可信质量做训练对加权。
3. **底层求解器选型**：在需要半离散 partial/robust OT 耦合的生成建模管线中，优先评估本文算法与 AAAI 2024 同类算法的实际吞吐与精度，再决定是否替换现有平衡 Sinkhorn 耦合。

## 8. 资源

代码链接：未公开（摘要未提及）。相关论文：AAAI 2024 partial/robust 半离散算法（具体 arXiv id 未提供）；UOTM-SD（ICLR 2024）；UOT-FM（ICLR 2024）。
