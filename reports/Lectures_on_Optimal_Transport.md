# Lectures on Optimal Transport

> Ambrosio, Brué & Semola · Springer UNITEXT 169 2024 · [Springer](https://link.springer.com/book/10.1007/978-3-031-76834-7) · 证据级 [B] · 课题 T01 OT 数学基础（面向生成模型研究者的最小必要集）
> **一句话**：SNS 二十年课程沉淀；给出两种自包含的 Kantorovich 对偶证明，通向几何/泛函不等式与 PDE

⚠ 未读全文，依据摘要

## 1. 问题

本书处理最优传输（optimal transport）的数学基础：以最小代价把一个概率分布搬运成另一个。对扩散/流生成模型研究者，这套语言不可绕过，因为生成模型本质上就是"把噪声分布运到数据分布"。此前学习材料中，Villani（2003/2009）是理论百科但体量庞大，Santambrogio（2015）面向应用数学，Peyré–Cuturi（2019）面向计算与数据科学；本书定位为课程化精炼，源自 SNS（Scuola Normale Superiore）二十年的课程沉淀，目标是给出自包含的入门路径。

## 2. 方法

核心内容围绕最优传输的经典理论展开。调研 agent 给出的一句话贡献指出：本书给出两种自包含的 Kantorovich 对偶证明。Kantorovich 对偶是 OT 理论的核心工具，把原始耦合线性规划问题转化为对偶形式：

$$\min_{\pi \in \Pi(\mu,\nu)} \int c\,d\pi = \sup_{(\varphi,\psi)} \left\{ \int \varphi\,d\mu + \int \psi\,d\nu : \varphi(x)+\psi(y) \leq c(x,y) \right\}$$

（公式为课题背景中的标准形式，非本书原文编号；原文截断，未见具体编号）。书中内容通向几何/泛函不等式与 PDE，覆盖从基础定义到 Wasserstein 距离、对偶理论、以及向分析应用的延伸。

## 3. 理论结果

本书是教材，理论结果以系统讲授的形式呈现。摘要未给出具体定理编号或结论清单。调研 agent 指出书中包含两种自包含的 Kantorovich 对偶证明，这是本书在方法论上的核心贡献之一。具体定理的假设与结论需读全文确认；原文截断，未见。

## 4. 实验与数字

本书为数学教材，无实验部分。无数据集、基线或数值结果。

## 5. 在 OT×扩散地图中的位置

本书属于 OT 数学基础层，与 Villani（2003/2009）、Santambrogio（2015）、Peyré–Cuturi（2019）、Figalli–Glaudo（2021/2023）构成教材演进链。在课题 T01 的"最小必要集"框架中，本书对应 Kantorovich 对偶与 Wasserstein 距离的严格基础，是理解 Brenier 定理、Benamou–Brenier 动态形式、半离散 OT 的前置装备。对扩散×OT 研究者，本书提供的对偶理论是阅读后续文献（如半离散 OT 的对偶势方法、flow matching 的 OT 分析）所需的最小数学语言。

## 6. 局限与批评

作者承认的局限：原文截断，未见。读出来的局限：作为教材，本书不覆盖计算侧内容（Sinkhorn、半离散数值方法等），也不直接涉及生成模型应用；对需要快速上手代码的研究者，需配合 Peyré–Cuturi（2019）或 Peyré（2025）等计算导向材料。此外，本书定位为数学基础，不包含实验验证或算法实现。

## 7. 对我们的启发

1. Kantorovich 对偶的两种自包含证明可作为课题组内部研讨的基准材料，帮助成员建立对偶势与耦合之间的严格对应，这是理解半离散 OT 对偶势做 training-free guidance（切入点 #1）的前提。
2. 本书通向几何/泛函不等式与 PDE 的路径，为理解 Benamou–Brenier 动能形式与 probability-flow ODE 的连续性方程约束提供数学语言，可支撑切入点 #2（BB 动能作为轨迹直度账本）的理论分析。
3. 作为课程化教材，本书的结构可作为课题组"OT 最小必要集"学习路线的骨架，按章节映射到扩散×OT 文献中的具体使用场景。

## 8. 资源

代码：未公开（教材无代码）。相关论文 arXiv id 互链：无直接 arXiv id；相关教材见 Villani（2003/2009）、Santambrogio（2015）、Peyré–Cuturi（2019，arXiv:1803.00567）、Figalli–Glaudo（2021/2023）。
