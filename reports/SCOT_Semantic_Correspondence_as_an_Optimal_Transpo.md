# SCOT: Semantic Correspondence as an Optimal Transport Problem

> Liu et al. · CVPR 2020 · [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Semantic_Correspondence_as_an_Optimal_Transport_Problem_CVPR_2020_paper.html) · 证据级 [P] · 课题 T16 OT 代价先验引导的跨域语义对应
> **一句话**：首次把语义对应表述为 OT 问题，用显著性做边际、Sinkhorn 求全局 plan，抑制最近邻的 many-to-one 错配

⚠ 未读全文，依据摘要

## 1. 问题

语义对应（semantic correspondence）任务需要在两幅图像之间建立语义上一致的匹配。此前方法依赖逐点最近邻检索或局部贪心匹配，每个 query 独立选择最相似的 key，没有全局约束，导致多个源位置挤到同一目标语义上，即 many-to-one 错配。摘要指出，本文把语义对应重新表述为最优传输（optimal transport）问题，以引入全局匹配约束。

## 2. 方法

核心思想是把语义对应写成 OT 问题：用显著性（saliency）调制边际分布，用 Sinkhorn 算法求全局 transport plan，从而在质量守恒约束下分配语义对应。摘要未给出具体公式编号或算法步骤细节；原文截断，未见。

## 3. 理论结果

摘要未报告定理、引理或理论保证。原文截断，未见。

## 4. 实验与数字

摘要未报告数据集、基线或具体数值。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文是线 A「判别式语义对应的代价设计」的起点：首次把语义对应写成 OT 问题，用显著性调制边际、Sinkhorn 求全局 plan，解决最近邻的 many-to-one 错配。后续 UNITE（CVPR 2021）将其推进到跨域生成场景并引入不平衡 OT；GWOT-SC（BMVC 2025）与 Shape-of-You（CVPR 2026）沿代价设计主线加入几何结构（GW/FGW）与外部 3D 先验。本文与扩散模型无直接接口，但其「OT 全局耦合替代局部贪心匹配」的视角为后续 attention 即 OT 的接口化（Sinkformer、OTSeg、STORM 等）提供了判别式对应侧的源头。

## 6. 局限与批评

摘要未报告作者承认的局限。从摘要可读出的限制：仅依据摘要，无法确认显著性边际的估计方式、Sinkhorn 的熵正则强度选择、以及方法在跨域（外观差异大）场景下的表现；这些均需全文验证。原文截断，未见。

## 7. 对我们的启发

1. 显著性调制边际的做法可迁移到扩散 attention 的 OT 干预：用当前去噪 latent 的显著性图作为边际先验，构造 mass-aware Sinkhorn attention guidance，对应切入点 #2（不平衡/partial OT 的 guidance 化）。
2. 「全局 plan 替代逐点最近邻」的动机可直接用于分析 softmax attention 的 many-to-one 错配：把 attention 矩阵与熵正则 OT plan 的偏差作为语义错乱的度量，接切入点 #3 的定量理论方向。
3. 若后续读到全文中的显著性估计与 Sinkhorn 实现细节，可评估其作为免训练 batch 级保边缘噪声指派（切入点 #1 MPNA）的边际构造参考。

## 8. 资源

代码未公开（摘要未提及）。相关论文：UNITE（CVPR 2021）、GWOT-SC（BMVC 2025）、Shape-of-You（CVPR 2026）、Sinkformer（AISTATS 2022）、OTSeg（ECCV 2024）、STORM（CVPR 2025）。
