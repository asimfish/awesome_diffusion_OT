# Diffusion Bridge Mixture Transports (IDBM), Peluchetti

> Peluchetti · JMLR 24(374) 2023 · [JMLR](https://www.jmlr.org/papers/v24/23-0527.html) · 证据级 [P] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：提出迭代扩散桥混合（IDBM），每步迭代保持两端边缘合法 transport，并给出收敛性初步分析。

⚠ 未读全文，依据摘要

## 1. 问题

本文处理 Schrödinger Bridge（SB）问题的求解。SB 问题要求在所有满足两端边缘约束 $p_0=\mu, p_1=\nu$ 的路径测度中，找到与参考过程 $Q$（通常为 Brownian/OU 过程）KL 散度最小的那个。对生成建模而言，SB 能在有限时间内把任意先验精确传输到数据分布，且两端都可以是数据分布，是 unpaired translation 的天然框架；同时 SB 是熵正则 OT（EOT）的动态实现，给出带最优性保证的随机映射。

此前求解 SB 的方法（如深度 IPF / DSB 一类）通过迭代比例拟合（IPF）交替训练两个网络，痛点在于：交替训练成本高、轨迹仿真昂贵，且 IPF 迭代会破坏一端边缘并造成误差累积。本文提出一种不同的迭代方案，试图在每次迭代中都保持两端边缘的合法 transport。

## 2. 方法

核心思想是迭代扩散桥混合（Iterative Diffusion Bridge Mixture, IDBM）。其关键洞见（据调研 agent 概括）：SB 是唯一既 Markov 又属于参考桥 reciprocal 类的过程，因此可以交替做 reciprocal 投影（重采样端点、插参考桥）与 Markov 投影（一次 bridge-matching 回归）来逼近 SB。与 IPF 不同，IDBM 的每次迭代都同时保持两端边缘约束，即每步迭代都构成一个合法的 transport。

摘要未给出具体公式编号与算法伪代码，原文截断，未见。

## 3. 理论结果

摘要未给出具体定理编号、假设与结论的完整表述。调研 agent 概括为「给出收敛性初步分析」，即作者对 IDBM 迭代过程的收敛性做了初步理论分析。具体收敛率、假设条件与证明细节原文未读，未见。

## 4. 实验与数字

摘要未提供数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文是 IMF / bridge matching 思想的独立源头之一，与 Shi et al.（NeurIPS 2023, DSBM）独立提出同一关键洞见：交替做 reciprocal 投影与 Markov 投影即可收敛，且每步迭代同时保持两端边缘。在方法演进脉络中，本文属于第二代（IMF / bridge matching，2023），针对第一代深度 IPF（DSB, NeurIPS 2021；SB-FBSDE, ICLR 2022）的交替训练昂贵、边缘破坏与误差累积问题给出替代方案。后续工作如 α-DSBM（NeurIPS 2024）将 IMF 连续化为 SB Flow，ASBM / CSBM 建立离散时间/离散空间的 D-IMF 理论，均以本文的 IDBM 迭代框架为基础。

## 6. 局限与批评

作者承认的局限：原文未读，未见。读出来的局限：摘要未提供实验验证，无法判断该方法在高维生成任务上的实际表现；「收敛性初步分析」表明理论保证尚不完整，收敛率与误差界未在摘要中给出；IDBM 仍属于迭代式方法，每步迭代的计算成本与所需迭代次数在摘要中不可见。

## 7. 对我们的启发

1. IDBM 每步保持两端边缘的合法 transport 特性，可对接「免训练 batch 级保边缘噪声指派 MPNA」切入点：在 batch 级噪声指派时显式保持两端边缘，避免 IPF 式边缘破坏。
2. 交替 reciprocal 投影与 Markov 投影的框架，可作为「OT-aware 采样调度」的理论参照：采样调度若能在每步保持 transport 合法性，可减少误差累积。
3. 若后续读到全文中的收敛性初步分析，可将其与 NeurIPS 2025 的 IMF 非渐近指数收敛率对照，定位「学习误差下 IMF 收敛理论」切入点的起点。

## 8. 资源

代码链接：未公开。相关论文：DSB（NeurIPS 2021）、SB-FBSDE（ICLR 2022）、DSBM（NeurIPS 2023）、α-DSBM（NeurIPS 2024）、ASBM（NeurIPS 2024）、CSBM（ICML 2025）。
