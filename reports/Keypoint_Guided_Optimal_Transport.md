# Keypoint-Guided Optimal Transport

> KPG-RL et al. · NeurIPS 2022 · [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6091c5644d73637e3cccdcab52a7031f-Abstract-Conference.html) · 证据级 [P] · 课题 T16 OT 代价先验引导的跨域语义对应
> **一句话**：用 mask 约束 plan 加关系保持，把少量标注 keypoint 语义先验注入 OT，支持异构空间与 partial 设定。

⚠ 未读全文，依据摘要

## 1. 问题

跨域语义对应中，逐点最近邻检索和 softmax attention 是局部贪心匹配，每个 query 独立选最相似的 key，没有全局约束，导致多个源位置挤到同一目标语义上（many-to-one 错配）。OT 把匹配升格为带边际约束的全局最小代价耦合，质量守恒强制「语义预算」分配，熵正则 Sinkhorn 给出可微的软对应。但纯外观代价的 OT 在语义歧义或外观变化大时仍可能错配；本工作处理的是如何把少量人工标注的 keypoint 语义先验注入 OT 的 plan 求解，使对应在语义上更可靠，同时支持源域与目标域特征空间异构（heterogeneous spaces）以及 partial（非全量对应）设定。摘要原文未给出此前方法的具体名称与失败数字，仅以「keypoint-guided optimal transport」为题，定位为 OT 对应中先验注入的一种形态。

## 2. 方法

核心思想：用 mask 约束 transport plan 的可行域，把少量标注 keypoint 的语义先验注入 OT；同时用关系保持（relation preservation）把 keypoint 的引导传播到非标注位置。摘要原文未给出公式编号与具体符号定义，以下为摘要可读出的方法要素：

- **mask 约束 plan**：对标注 keypoint 对应的位置施加 mask，限制 transport plan 在这些位置上的可行取值，使求解出的耦合与人工语义标注一致。
- **关系保持传播引导**：利用 keypoint 之间的结构关系，把先验从少量标注点传播到其余位置，缓解标注稀疏问题。
- **支持异构空间与 partial 设定**：方法不要求源域与目标域特征维度相同（heterogeneous spaces），也不要求所有点都有对应（partial）。

摘要未给出算法步骤、损失函数或训练/求解流程的具体细节，原文截断，未见。

## 3. 理论结果

摘要未报告定理、引理或理论保证。原文截断，未见。

## 4. 实验与数字

摘要未报告数据集、基线、数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本工作属于线 A（判别式语义对应的代价设计）中「先验注入」的一环：在 SCOT（CVPR 2020）用显著性调制边际、UNITE（CVPR 2021）用不平衡 OT 与自适应质量学习之后，KPG-RL 示范了另一种先验形态——少量 keypoint 通过 mask 约束 plan 可行域，并以关系保持传播引导。它处理的是判别式对应阶段的先验注入，不直接涉及扩散采样或训练目标；在 OT×扩散地图中，它对应「代价/plan 先验设计」这一环节，为后续把先验注入扩散 attention 的采样期干预（如 STORM、ASAG）提供了判别式侧的参照。与线 B（attention 即 OT 接口化）和线 C（采样/训练级 OT 耦合先验）无直接方法继承关系，但共享「用 OT 全局耦合替代局部贪心匹配」的动机。

## 6. 局限与批评

作者承认的局限：摘要未报告。原文截断，未见。

读出来的局限（基于摘要信息，非原文断言）：
- 摘要未给出任何实验数字，无法判断 keypoint 先验注入在何种规模标注下带来多少收益；「少量标注」的具体数量未知。
- mask 约束 plan 的可行域方式在 keypoint 标注错误或语义歧义时可能把错误先验强制写入耦合，摘要未提及对标注噪声的处理。
- 关系保持传播的具体形式（是几何距离保持、图结构保持还是其他）摘要未说明，无法评估其在非刚性形变下的适用性。

## 7. 对我们的启发

1. **可接切入点 #2（不平衡/partial OT 的 guidance 化）**：本工作明确支持 partial 设定，与「跨域语义不守恒时 balanced OT 会强制错配」的问题直接相关。可考虑把 KPG-RL 的 mask 约束 plan 思路搬进采样期 attention 干预，做 mass-aware 的 keypoint 引导 Sinkhorn attention，在多物体编辑场景测属性泄漏率与编辑成功率。
2. **可接切入点 #4（代价函数学习）**：KPG-RL 用人工 keypoint 作为先验，本质是手工代价/约束设计。若用 inverse OT 从少量标注对应反推 cost functional，可把这种 keypoint 先验替换为数据驱动先验，目标是在未见类别上泛化对应。
3. **可接切入点 #5（token 级 OT 一致性 metric）**：KPG-RL 的 keypoint 引导 plan 可作为「语义正确对应」的参照，用来度量编辑前后 cross-attention plan 偏离最优耦合的程度；若把 keypoint 标注作为 ground-truth 对应，可验证 OT 一致性 metric 与人工评分、CLIP-score 的相关性。

## 8. 资源

代码链接：未公开（摘要与元数据未提供）。  
相关论文 arXiv id 互链：SCOT（CVPR 2020）、UNITE（CVPR 2021）、Sinkformer（AISTATS 2022）、PLOT（ICLR 2023）、OTCS（NeurIPS 2023）、OTSeg（ECCV 2024）、STORM（CVPR 2025）、GWOT-SC（BMVC 2025）、ASAG（AAAI 2026）、Shape-of-You（CVPR 2026）、OTComp（ICML 2026）；具体 arXiv id 原文未提供，未见。
