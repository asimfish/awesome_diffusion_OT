# The Variational Formulation of the Fokker–Planck Equation

> Jordan, Kinderlehrer, Otto · SIAM J. Math. Anal. 1998 · [DOI](https://doi.org/10.1137/S0036141096303359) · 证据级 [P] · 课题 T05 Wasserstein 梯度流与 JKO 格式生成模型
> **一句话**：把 Fokker–Planck 方程写成 KL 散度在 Wasserstein-2 度量下的梯度流，并给出 JKO 隐式离散格式。

⚠ 未读全文，依据摘要

## 1. 问题

Fokker–Planck 方程（FPE）描述扩散过程下概率密度的演化，是扩散模型 forward SDE 的密度演化方程。在本文之前，FPE 通常被当作一个偏微分方程来研究，缺乏一个把「密度演化」与「能量下降」联系起来的变分视角。本文要解决的问题是：能否把 FPE 解释为某个能量泛函在概率分布空间中的最速下降（梯度流），从而为后续的数值离散与优化算法提供理论基础。

## 2. 方法

核心思想是把 FPE 重新表述为 KL 散度（相对熵）泛函在 Wasserstein-2 度量下的梯度流。具体地，概率密度 $\rho$ 的演化被看作在 $\mathcal{P}_2(\mathbb{R}^d)$ 空间中沿 KL 能量下降最快的方向移动，而「下降最快」的度量由 Wasserstein-2 距离 $W_2$ 定义。

作者提出 JKO（minimizing movement）格式作为该梯度流的隐式 Euler 离散：每一步从当前密度 $\rho_k$ 出发，通过求解

$$\rho_{k+1} = \arg\min_\rho \left\{ \frac{1}{2\tau} W_2^2(\rho, \rho_k) + \mathcal{F}(\rho) \right\}$$

得到下一时刻的密度，其中 $\mathcal{F}$ 是 KL 散度泛函，$\tau$ 是步长。该格式把连续的梯度流转化为一系列在分布空间中的近端（proximal）优化问题。

## 3. 理论结果

摘要未给出具体定理编号与完整假设。作者报告的核心理论结果是：Fokker–Planck 方程等价于 KL 散度泛函在 Wasserstein-2 度量下的梯度流，且 JKO 格式是该流的隐式离散。具体假设、收敛阶数、正则性条件等细节在摘要中未展开，原文截断，未见。

## 4. 实验与数字

本文为纯理论论文，摘要未报告任何数值实验或数据集结果。无实验数字可列。

## 5. 在 OT×扩散地图中的位置

本文是 Wasserstein 梯度流与 JKO 格式生成模型这条线的理论奠基工作。它建立了「FPE = KL 的 $W_2$ 梯度流」这一等价关系，并给出 JKO 离散格式，为后续 Otto (2001) 的黎曼几何语言（Otto calculus）提供了出发点。在扩散模型语境下，它提供了不同于 score matching 的第二套构造原理：不直接估计 score，而是通过逐步求解 Wasserstein 近端问题来演化密度。后续的 JKO-ICNN、S-JKO、JKOnet 等工作都直接建立在本文的 JKO 格式之上。

## 6. 局限与批评

作者承认的局限：摘要未提及，原文未读，未见。

读出来的局限：本文是纯理论分析，未涉及任何数值实现；JKO 格式的每一步本身是一个最优传输问题，在高维情形下求解代价极高，这限制了它作为数值算法的直接可用性——这一瓶颈直到 2021 年前后才被 ICNN 参数化等方法部分缓解。

## 7. 对我们的启发

1. JKO 格式把密度演化写成「$W_2$ 距离项 + 能量泛函」的逐步最小化，这为设计免训练或轻量训练的生成推理流程提供了理论模板：可以在推理期针对目标域小样本在线解一个轻量半对偶问题，把所得势的梯度作为漂移项注入 PF-ODE（对应切入点 #1）。
2. 本文确立的「能量泛函 + Wasserstein 度量」框架是后续所有 WGF 蒸馏与一步生成器工作的共同语言；在分析一步生成器蒸馏误差时，可以沿用「能量泛函沿广义测地线的凸性/光滑常数 × 离散步长」这一思路（对应切入点 #4）。
3. 作为 JKO 格式的源头，本文提醒我们：任何基于 JKO 的生成模型，其每一步的 $W_2$ 求解质量直接决定密度演化的精度；在评估 S-JKO、JKO-iFlow 等方法时，应把「单步 OT 求解误差」与「整体生成质量」分开度量。

## 8. 资源

代码：未公开（纯理论论文）。

相关论文互链：Otto (2001) 的 Otto calculus 形式化；Santambrogio 综述《Optimal Transport for Applied Mathematicians》；Mokrov et al. 2021（JKO-ICNN）；Alvarez-Melis et al. 2021；Bunne et al. 2021（JKOnet）；Xu et al. 2023（S-JKO）。
