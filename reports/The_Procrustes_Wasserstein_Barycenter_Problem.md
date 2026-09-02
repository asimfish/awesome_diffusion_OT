# The Procrustes-Wasserstein Barycenter Problem

> Adamo et al. · ICML 2025 · [PMLR](https://proceedings.mlr.press/v267/adamo25a.html) · 证据级 [P] · 课题 T27 多边际 OT 与 Wasserstein 重心的生成应用
> **一句话**：联合优化正交/刚体对齐与 Wasserstein barycenter，解决输入分布姿态不对齐时平均失真的问题。

⚠ 未读全文，依据摘要

## 1. 问题

本文处理 Wasserstein barycenter 问题的一个前提性缺陷：标准 barycenter 假设输入的多个分布已经在同一个坐标/姿态框架下对齐。当输入分布之间存在正交变换（旋转/反射）或刚体变换（旋转+平移）差异时，直接求 barycenter 会把姿态差异误当作分布形状差异，导致平均结果失真。作者将这种「对齐 + 平均」的联合问题形式化为 Procrustes-Wasserstein barycenter 问题。摘要未给出此前方法的具体名称或缺陷的定量描述。

## 2. 方法

核心思想是把 Procrustes 对齐（正交/刚体变换）与 Wasserstein barycenter 放进同一个优化问题中联合求解，而不是先对齐再平均的两阶段流程。摘要未给出具体的目标函数形式、算法步骤或公式编号，原文截断，未见。

## 3. 理论结果

摘要未提及定理、引理或理论保证。原文截断，未见。

## 4. 实验与数字

摘要未给出数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文位于 barycenter 计算线中的「鲁棒变体」分支，与 Wasserstein ball center（ICML 2025 [P]）并列，补齐「对齐 + 平均」这一场景。它处理的是 barycenter 输入预处理阶段的姿态对齐问题，与生成应用线中的 barycenter 融合/插值（如潜空间 barycenter、模型融合）相关：当多个风格/条件分布或网络层神经元分布存在姿态差异时，直接平均会引入失真，本文的对齐联合优化可视为这类应用的前置修正。与固定支撑熵正则 barycenter（Janati 2020）、精确解（SGA, ICLR 2026 [A]）、神经/连续解（Kolesov et al. ICML 2024）等计算线工作构成互补而非竞争关系。

## 6. 局限与批评

作者承认的局限：摘要未提及。原文截断，未见。

读出来的局限：摘要未给出算法复杂度、局部最优性处理或对齐变换集合的约束条件，无法判断其可扩展性；「联合优化」的具体实现方式（交替优化/同时优化）未知，可能影响收敛性质。以上均为基于摘要信息缺失的推断，非原文结论。

## 7. 对我们的启发

1. 在模型融合（OTFusion 线）中，逐层 barycenter 融合前可加入正交/刚体对齐步骤，消除不同训练网络神经元排列的姿态差异，可能改善融合后模型的性能保持。
2. 多风格/多条件生成融合时，若不同条件分布嵌入存在旋转/平移差异，可借鉴本文的联合对齐-平均思路，替代简单的 embedding 线性插值。
3. 若后续读到全文中的算法细节，可评估其对齐步骤能否作为免训练预处理模块嵌入现有 barycenter 生成管线（如 Wukong 的 free-support barycenter）。

## 8. 资源

代码未公开（摘要未提及）。相关论文：Wasserstein ball center（ICML 2025 [P]）；Janati et al. 2020（熵正则 barycenter 去偏）；OTFusion（NeurIPS 2020）。
