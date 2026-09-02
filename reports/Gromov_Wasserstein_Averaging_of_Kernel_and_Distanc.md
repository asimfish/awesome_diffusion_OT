# Gromov-Wasserstein Averaging of Kernel and Distance Matrices

> Peyré, Cuturi, Solomon et al. · ICML 2016 · [PMLR](https://proceedings.mlr.press/v48/peyre16.html) · 证据级 [P] · 课题 T26 Gromov-Wasserstein 与跨空间生成对齐
> **一句话**：提出 entropic GW 的投影镜像下降求解器与 GW barycenter，确立现代计算 GW 的基本范式。

⚠ 未读全文，依据摘要

## 1. 问题

Gromov-Wasserstein（GW）距离在 metric measure space 之间比较内部距离结构，不要求两个空间共享坐标系，因此适合跨空间对齐。但 GW 距离的求解是一个非凸二次指派问题（QAP），NP-hard，且朴素求解的计算与内存开销大。本文要解决的是：如何为 GW 距离及其 barycenter 提供一个可实际计算的求解框架。

## 2. 方法

作者提出 entropic GW 的投影镜像下降（projected mirror descent）求解器。核心思路是在 entropic 正则化下迭代求解 GW 耦合，并利用投影步骤保持约束。在此基础上定义并计算 GW barycenter，即在多个 metric measure space 之间求一个「平均」空间，使到各输入空间的 GW 距离之和最小。摘要未给出具体公式编号与算法细节。

## 3. 理论结果

摘要未报告定理、引理或收敛性保证。原文截断，未见。

## 4. 实验与数字

摘要未报告数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文是计算 GW 的奠基性工作，确立了 entropic GW 镜像下降的基本范式，后续低秩 GW（Scetbon et al., ICML 2022）、SDP 认证（NeurIPS 2024）、GW-at-Scale（ICML 2026）等计算线工作均在其上扩展。在生成对齐线上，GENOT（NeurIPS 2024）直接使用 entropic (F)GW 耦合作为流匹配骨架的静态端点耦合，其耦合求解可追溯到本文的 entropic GW 框架。本文对应「计算线」的起点，解决的是 GW 从理论定义到可计算求解的瓶颈。

## 6. 局限与批评

作者承认的局限：摘要未报告。读出来的局限：entropic 正则化引入偏差，耦合不再精确等于原始 GW 问题的解；投影镜像下降的收敛速度与正则化参数的关系在摘要中未给出；本文方法针对离散度量空间，连续空间的 GW 求解仍需后续工作（如 NeuralGW 等）处理。

## 7. 对我们的启发

1. entropic GW 耦合可作为跨空间生成对齐的静态端点先验，直接嵌入流匹配或 Schrödinger bridge 框架（对应切入点 #1：跨维 Gromov-Schrödinger bridge）。
2. GW barycenter 的求解框架可迁移到「多模态潜空间缝合」场景：在多个冻结生成模型的潜空间之间求结构平均，作为免训练对齐的初始化（对应切入点 #2）。
3. 投影镜像下降的迭代结构提示：在扩散/流生成中，可考虑将 GW 耦合的迭代求解与去噪步骤交替进行，形成 OT-aware 的采样调度（对应启发 #2）。

## 8. 资源

代码未公开。相关论文：Scetbon et al. 2022（低秩 GW，[arXiv:2202.02123](https://arxiv.org/abs/2202.02123)）；GENOT（NeurIPS 2024，[arXiv:2410.01217](https://arxiv.org/abs/2410.01217)）；Zhang–Goldfeld et al.（AoS 2024，GW 对偶与样本复杂度）。
