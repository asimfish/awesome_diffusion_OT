# Finding the Center of a Wasserstein Ball

> Wang et al. · ICML 2025 · [PMLR](https://proceedings.mlr.press/v267/wang25be.html) · 证据级 [P] · 课题 T27 多边际 OT 与 Wasserstein 重心的生成应用
> **一句话**：Wasserstein ball 中心是一种 min-max 鲁棒聚合，与 barycenter 互补的"最坏情况平均"视角

⚠ 未读全文，依据摘要

## 1. 问题

给定一组概率分布，Wasserstein barycenter 回答"如何取几何平均"。但 barycenter 是最小化到各输入分布距离之和的中心，对离群分布敏感。本文提出另一种聚合对象：Wasserstein ball 的中心（center of a Wasserstein ball），即在给定半径约束下、使到球内任意分布的最坏情况距离最小的分布。这是一个 min-max 鲁棒聚合问题，与 barycenter 的 min-sum 视角互补。

摘要未提供此前方法不足的具体论述，原文截断，未见。

## 2. 方法

核心思想是把"中心"定义为 Wasserstein 球上的最坏情况优化：在 Wasserstein 度量空间中，给定一个以某参考点为中心、半径 $r$ 的球，求球内使最坏情况 $W_p$ 距离最小的分布。摘要未给出具体公式编号或算法步骤，原文截断，未见。

## 3. 理论结果

摘要未列出定理、引理或保证的具体内容，原文截断，未见。

## 4. 实验与数字

摘要未提供数据集、基线或数值结果，原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文属于 barycenter 计算线的鲁棒变体分支，与 Procrustes-WB（ICML 2025 [P]）并列，补齐"对齐+平均"与"最坏情况平均"两种聚合视角。在 OT×扩散地图中，它对应多分布聚合的鲁棒性环节：当输入分布集合存在噪声或离群点时，barycenter 的 min-sum 目标会被拉偏，而 Wasserstein ball 中心的 min-max 目标提供另一种聚合准则。该视角与生成建模中的多风格/多条件融合需求相关，但摘要未说明其与扩散或流模型的直接联系，原文截断，未见。

## 6. 局限与批评

摘要未包含作者承认的局限。从摘要可读出的潜在问题：min-max 目标的计算复杂度通常高于 min-sum（barycenter），摘要未说明可扩展性；"最坏情况平均"的几何意义与 barycenter 的差异在摘要中未量化。以上为基于摘要的推断，原文截断，未见。

## 7. 对我们的启发

1. 在模型融合或多条件融合场景中，若输入分布集合可能含离群模型/风格，可对比 Wasserstein ball 中心与 barycenter 的鲁棒性差异，作为融合准则的替代选项。
2. 可探索将 min-max 鲁棒聚合引入扩散模型的条件分布融合，检验"最坏情况平均"是否比 min-sum barycenter 更抗风格支配。
3. 若后续读到全文中的算法细节，可评估其是否适合作为 batch 级保边缘噪声指派（MPNA）的鲁棒化变体。

## 8. 资源

代码未公开。相关论文：Procrustes-WB（ICML 2025 [P]）；Wasserstein barycenter 计算线见 SGA（ICLR 2026 [A]）、FRBary（[R]）。
