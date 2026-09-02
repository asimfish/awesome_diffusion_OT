# Stochastic Interpolants for Revealing Stylistic Flows across the History of Art

> Ma et al. · ICCV 2025 · [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Stochastic_Interpolants_for_Revealing_Stylistic_Flows_across_the_History_of_ICCV_2025_paper.pdf) · 证据级 [P] · 课题 T17 风格迁移与域自适应中的 OT×扩散
> **一句话**：把艺术风格历史演化建模为风格空间 OT 分布匹配，用 stochastic interpolants+DDIB 无配对对齐跨世纪艺术分布，并发布 65 万艺术品数据集。

⚠ 未读全文，依据摘要

## 1. 问题

艺术风格的历史演化通常被当作离散的风格类别或线性时间序列来处理，难以刻画风格之间连续、可穿越的分布变化。本文要解决的问题是：如何把跨越数世纪的艺术风格演化建模为风格空间中的连续分布流，并在无配对数据条件下对齐不同时期的艺术分布。摘要未给出此前方法的具体缺陷描述，原文截断，未见。

## 2. 方法

核心思想是把艺术风格的历史演化建模为风格空间中的 OT 分布匹配问题，用 stochastic interpolants 构造跨世纪艺术分布之间的连续传输路径，并结合 DDIB 实现无配对对齐。摘要未给出具体公式与算法步骤，原文截断，未见。

## 3. 理论结果

摘要未报告定理、引理或理论保证。无理论结果（依据摘要）。

## 4. 实验与数字

摘要未报告具体实验设置、基线或数值结果。唯一可确认的资源规模是：发布了一个包含 65 万件艺术品的数据集（依据摘要）。其余数字原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文位于「扩散机制层（2025）」：与 SW-Guidance（NeurIPS 2025 spotlight）、ModFlows（AAAI 2025）同期，把 OT 与扩散/流生成机制融合。具体而言，它继承 DDIB 的熵正则 OT / Schrödinger bridge 解释，把 DDIB 从图像对之间的 latent 对齐推广到跨世纪艺术分布的整体对齐；与 ModFlows 同属「用 OT plan 学分布间传输」的路线，但对象是历史风格分布而非单张图像对。在课题地图中，它对应「扩散机制层」的分布匹配管线环节，而非损失层或模型迁移层。

## 6. 局限与批评

作者承认的局限：摘要未报告，原文截断，未见。

读出来的局限（依据摘要，受限）：
1. 摘要未给出任何定量结果，无法判断跨世纪风格对齐的质量或与基线的差距。
2. 摘要未说明 stochastic interpolants 与 DDIB 的具体结合方式，无法评估其相对纯 DDIB 或纯 interpolants 的增益。
3. 摘要未提及风格演化路径的可解释性或历史学验证，仅凭分布匹配难以确认「风格流」是否对应真实艺术史演化。

## 7. 对我们的启发

1. 可接切入点 #2（DDIB latent 对齐的更优传输）：本文用 stochastic interpolants 构造连续 OT 路径，提示可以把 OT-ALD 的离散 OT map 修正替换为 interpolants/flow matching 学到的连续传输，摊销每批次求解成本。
2. 可接切入点 #5（SW 风格损失的理论刻画）：若本文的风格空间分布匹配在特征空间进行，可考察其与 sliced Wasserstein 风格损失的连续性，作为风格分布流理论刻画的一个实例。
3. 数据集资源（65 万艺术品）可作为风格迁移/域自适应任务的跨世纪分布漂移基准，用于检验 OT 几何控制方法在真实长时程分布漂移下的表现。

## 8. 资源

代码链接：未公开（依据摘要，未见）。相关论文互链：DDIB（arXiv:2109.xxxx，原文未给 id，未见）；ModFlows（AAAI 2025，原文未给 id，未见）；SW-Guidance（NeurIPS 2025 spotlight，原文未给 id，未见）。
