# Joint Velocity-Growth Flow Matching

> Wang et al. · NeurIPS 2025 · [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eb1bad7a84ef68a64f1afd6577725d45-Abstract-Conference.html) · 证据级 [P] · 课题 T25 非平衡/部分 OT 在生成建模中的应用
> **一句话**：给静态 semi-relaxed OT 一个"先长质量后运输"的两段式动态解释，联合速度+增长的 simulation-free FM

⚠ 未读全文，依据摘要

## 1. 问题

本文处理的是 semi-relaxed optimal transport（semi-relaxed OT）在生成建模中的动态化问题。semi-relaxed OT 只松弛单侧边缘约束，适用于源域与目标域质量不等的场景（如类别比例失衡、质量生灭）。此前将这类静态松弛问题接入生成模型的路线，多依赖离散预处理或重加权后再做平衡耦合（如 UOT-FM 的重缩放思路），缺乏一个直接以"质量增长 + 运输"联合动态为训练目标的 simulation-free 框架。作者提出的问题是：能否把静态 semi-relaxed OT 解释为两段式动态过程，并据此设计一个同时回归速度场与增长场的 flow matching 目标。

## 2. 方法

核心思想是给静态 semi-relaxed OT 一个"先长质量后运输"的两段式动态解释：第一阶段质量增长，第二阶段运输。基于这一动态解释，作者提出联合速度（velocity）与增长（growth）的 simulation-free flow matching 训练目标，即 Joint Velocity-Growth Flow Matching（VGFM）。摘要未给出具体公式编号与算法细节，原文截断，未见。

## 3. 理论结果

摘要未列出定理、引理或保证的具体内容。原文截断，未见。

## 4. 实验与数字

摘要未给出数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文位于非平衡 OT 动态化与 simulation-free 生成建模的交汇处。与 UOT-FM（ICLR 2024）的"先离散 UOT 重加权、再平衡耦合"路线不同，VGFM 从 semi-relaxed OT 出发直接得到两段式动态，并做成 flow matching 目标。与 WFR-FM（ICLR 2026）同属"速度 + 增长"联合回归的 simulation-free 路线，但 WFR-FM 的几何基础是 Wasserstein–Fisher–Rao（运输 + 生灭代价），VGFM 的出发点则是 semi-relaxed OT 的两段式分解。两者构成同一谱系内不同静态松弛来源的平行方案。

## 6. 局限与批评

作者承认的局限：摘要未提及。原文截断，未见。

读出来的局限：摘要未给出任何实验数字或理论保证，无法评估两段式动态解释对实际生成质量的增益；"先长质量后运输"的阶段划分是否唯一、两段之间的切换点如何确定，摘要未说明。原文截断，未见。

## 7. 对我们的启发

1. 可接切入点 #1：将 VGFM 的联合速度-增长 FM 目标与 minibatch semi-relaxed 耦合结合，在类别失衡的 CIFAR-10/ImageNet 子集上量化"边缘违反度 vs 轨迹直线度/推理步数"的权衡；VGFM 的两段式结构提供了一个可对照的动态基线。
2. 可接切入点 #3：VGFM 的增长率场可作为推理期第二控制量的候选来源，与 WFR 的 birth-death 粒子重生/降权方案对比，检验 semi-relaxed 来源的增长场在免重训模式再平衡中的表现。
3. 若后续读到全文中的两段式切换条件，可将其作为 OT-aware 采样调度的阶段划分依据，与现有 UOTM-SD 的散度调度做对照。

## 8. 资源

代码未公开。相关论文：UOT-FM（arXiv:2404.07407）、WFR-FM（ICLR 2026，arXiv id 待补）、UOTM（NeurIPS 2023，arXiv:2303.00555）。
