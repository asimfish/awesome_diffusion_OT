# OTComp: Dual Optimal Transport for Multi-Concept Composition

> fuhao7i et al. · ICML 2026 · [ICML](https://icml.cc/virtual/2026/poster/63327) · 证据级 [A] · 课题 T16 OT 代价先验引导的跨域语义对应
> **一句话**：双 OT training-free 引导，质量守恒 OT 对齐结构草图，几何引导 OT 传输高频纹理残差，多概念组合无属性串扰。
> ⚠ 未读全文，依据摘要

## 1. 问题

多概念组合生成（multi-concept composition）中的语义错乱——属性泄漏（attribute leakage）、物体错位（mislocated objects）、many-to-one 错配——源于逐点最近邻检索和 softmax attention 的局部贪心匹配：每个 query 独立选择最相似的 key，没有全局约束，多个源位置会挤到同一目标语义上。OTComp 针对的是这一根源，把匹配升格为带边际约束的全局最小代价耦合，用质量守恒强制「语义预算」分配。摘要未给出此前具体基线方法及其不足的量化对比，原文截断，未见。

## 2. 方法

OTComp 提出 training-free 的双 OT 分解，把「结构/纹理」两类语义错乱分开治理：

1. **质量守恒 OT 做结构草图对齐**：用带边际约束的 OT 耦合对齐结构草图（structural sketch），强制语义预算的全局分配，避免 many-to-one 错配。
2. **几何引导 OT 做高频纹理残差传输**：在结构对齐之后，用几何引导的 OT 传输高频纹理残差（high-frequency texture residual），处理纹理层面的错乱。

摘要未给出具体公式编号、代价函数形式、Sinkhorn 熵正则系数或算法伪代码。原文截断，未见。

## 3. 理论结果

摘要未报告任何定理、引理或理论保证。无理论结果（依据摘要）。

## 4. 实验与数字

摘要未给出数据集、基线、指标或任何数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

OTComp 属于线 C（采样/训练级 OT 耦合先验）中 training-free 采样期引导的一支，与 STORM（CVPR 2025，带空间代价 OT 重定位 attention map）、ASAG（AAAI 2026，对抗代价 Sinkhorn 构造劣化 attention 分支）并列。其独特贡献是把「结构对齐」与「纹理传输」拆成两个 OT 子问题，分别用质量守恒 OT 与几何引导 OT 处理，对应线 A 中代价设计从纯外观到几何结构（GW/FGW）的演化在生成式引导侧的落地。与 OTCS（NeurIPS 2023）的训练级耦合估计不同，OTComp 是 training-free 的采样期干预。摘要未说明其与 TP-Blend（2026, [R]）在 cross-attention 特征重分配上的具体差异，原文截断，未见。

## 6. 局限与批评

作者承认的局限：摘要未提及。原文截断，未见。

读出来的局限（依据摘要）：
1. 摘要未报告任何实验数字，无法判断「无属性串扰」这一论断的强度与泛化条件。
2. 双 OT 分解引入两个代价函数与两个求解过程，计算开销与超参敏感性在摘要中不可见。
3. 质量守恒 OT 假设语义预算守恒，在物体增删、部件缺失等跨域语义不守恒场景下可能强制错配（对应课题开放问题 #2 的不平衡/partial OT 方向）。

## 7. 对我们的启发

1. **结构/纹理分离的 guidance 接口**：OTComp 的「先结构草图、后纹理残差」两阶段传输，可接切入点 #2——把质量守恒 OT 替换为 mass-aware Sinkhorn attention guidance，在多物体编辑 benchmark 上测属性泄漏率。
2. **几何引导 OT 的代价设计**：摘要中「几何引导 OT」的具体代价形式未知，但可对照 Shape-of-You 的 FGW 外观+结构代价，探索把判别式对应（线 A）的几何代价搬进采样期纹理传输（线 C），即切入点 #1 的 FGW 对应闭环。
3. **training-free 双 OT 作为基线**：若 OTComp 代码公开，可作为 ASAG/STORM 类采样期 OT 干预的对比基线，验证「分解治理」是否优于「单一 OT 干预」。

## 8. 资源

代码：https://github.com/fuhao7i/OTComp （公开）
相关论文：SCOT（arXiv:2004.11714）、OTCS（arXiv:2303.06802）、STORM（arXiv:2412.05223）、ASAG（AAAI 2026，arXiv id 未见）、GWOT-SC（BMVC 2025，arXiv id 未见）、Shape-of-You（CVPR 2026，arXiv id 未见）。
