# Optimizing Functionals on the Space of Probabilities with ICNNs

> Alvarez-Melis, Schiff, Mroueh · TMLR 2022 · [OpenReview](https://openreview.net/forum?id=dpOYN7o8Jm) · 证据级 [P] · 课题 T05 Wasserstein 梯度流与 JKO 格式生成模型
> **一句话**：用 ICNN 参数化凸势求解 JKO 步，把 Wasserstein 梯度流推到高维并给出收敛保证。

⚠ 未读全文，依据摘要

## 1. 问题

在概率测度空间上优化泛函（如 KL 散度、相互作用能）是 Wasserstein 梯度流与 JKO 格式的核心任务。JKO 格式的每一步本身是一个最优传输问题：$\rho_{k+1}=\arg\min_\rho \frac{1}{2\tau}W_2^2(\rho,\rho_k)+\mathcal{F}(\rho)$。此前方法在数值求解上受限于低维网格或参数化表达力不足，难以处理高维分布与一般泛函。本文要解决的是：如何用神经网络参数化手段高效求解 JKO 步，并保证所得离散流的收敛性。

## 2. 方法

作者提出 JKO-ICNN 框架，核心是用 Input Convex Neural Network（ICNN）逼近凸函数空间，以参数化 JKO 步中的 Brenier 凸势。ICNN 的结构保证其输出关于输入是凸函数，从而与最优传输映射的凸势结构对齐。框架包含针对所优化泛函的专门设计，使 JKO 步的变分问题可通过神经网络训练求解。摘要未给出具体损失函数或算法伪代码的公式编号。

## 3. 理论结果

摘要报告该框架对 JKO 离散化给出收敛保证。具体假设、收敛阶数、所依赖的泛函凸性条件等细节在摘要中未展开，原文截断，未见。

## 4. 实验与数字

摘要未列出具体数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文与 Mokrov 等（2021）几乎同时用 ICNN 参数化 JKO 步的 Brenier 凸势，属于「神经化第一波」中免网格求解高维 Wasserstein 梯度流的代表工作。它继承 JKO（1998）与 Otto calculus（2001）的理论框架，把度量空间梯度流的 proximal 离散从分析工具变成可训练的神经网络算法。后续 S-JKO（2023）通过半对偶目标将训练复杂度从 $O(K^2)$ 降到 $O(K)$，部分取代了逐步嵌套的 ICNN 方案；JKOnet 系则沿反问题方向从快照学习能量。本文在推理管线中对应「用 JKO 步构造生成模型」的求解器环节。

## 6. 局限与批评

作者承认的局限：摘要未提及。读出来的局限：ICNN 的表达力受限于凸架构，可能不足以逼近任意 Brenier 势；逐步嵌套训练导致复杂度随 JKO 步数增长（$O(K^2)$ 量级，来自课题背景，非本文摘要原文）；摘要未报告任何实验数字，无法评估实际高维表现。

## 7. 对我们的启发

1. ICNN 参数化凸势的思路可复用于免训练 guidance：冻结预训练扩散模型后，在线解一个轻量凸势优化问题，把势的梯度作为漂移项注入 PF-ODE（接切入点 #1）。
2. JKO 步的收敛保证提示：若把 JKO 离散解释为隐式 Euler，步长与能量凸性常数的关系可作为 OT-aware 采样调度的理论依据（接切入点 #2）。
3. 本文的泛函设计视角与 JKOnet 系互补，可探索「从快照学能量 + ICNN 求解」的联合管线用于医学图像生成（接切入点 #7）。

## 8. 资源

代码未公开（摘要与元数据均未提供代码链接）。相关论文：Mokrov et al. 2021（ICNN 求解 JKO 的同期工作，arXiv id 未见）；Bunne et al. 2021（JKOnet，arXiv id 未见）；S-JKO（2023，arXiv id 未见）。
