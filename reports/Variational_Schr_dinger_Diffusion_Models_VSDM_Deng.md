# Variational Schrödinger Diffusion Models (VSDM), Deng et al.

> ⚠ 未读全文，依据摘要
> Deng et al. · ICML (PMLR v235) 2024 · [PMLR](https://proceedings.mlr.press/v235/deng24c.html) · 证据级 [P] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：变分推断线性化 SB 前向 score，恢复后向 score 的免仿真训练；随机逼近证明收敛、无需 warm-up

## 1. 问题

Schrödinger Bridge（SB）在高维生成建模中的求解代价高：深度 IPF 类方法需要交替训练两个网络、轨迹仿真昂贵，且迭代破坏一端边缘并累积误差。VSDM 针对的是 SB 前向 score 的估计问题——若能免仿真地恢复前向 score，即可线性化 SB 并恢复后向 score，从而降低训练成本。摘要未给出此前方法的具体数字对比，仅以「免仿真训练」和「无需 warm-up」作为方法卖点。

## 2. 方法

核心思想：用变分推断（variational inference）线性化 SB 的前向 score，进而恢复后向 score，实现免仿真训练。摘要未给出具体公式编号或算法步骤细节；「变分推断」「线性化」「前向 score」「后向 score」为摘要原词。随机逼近（stochastic approximation）用于证明收敛，且无需 warm-up。

## 3. 理论结果

摘要报告：随机逼近证明收敛，且无需 warm-up。具体假设、收敛率、误差界等数字在摘要中未给出，原文截断，未见。

## 4. 实验与数字

摘要未提供数据集、基线或任何数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

VSDM 属于第三代「轻量化、变分化、广义化（2024）」：与 LightSB（ICLR 2024）、LightSB-M（ICML 2024）同期，用变分手段替代仿真式交替优化。它继承 DSB（NeurIPS 2021）与 SB-FBSDE（ICLR 2022）的 SB 生成建模叙事，但针对其「轨迹仿真昂贵、交替训练」痛点，走免仿真路线。与 [SF]²M（AISTATS 2024）的静态 minibatch Sinkhorn 捷径、LightSB-M 的高斯混合势闭式解并列，构成「免仿真/轻量 SB」方向。摘要未说明 VSDM 与这些工作的直接对比关系。

## 6. 局限与批评

作者承认的局限：摘要未提及。读出来的局限：摘要未给出任何实验数字，无法判断免仿真训练在高维真实数据上的实际表现；「线性化前向 score」的近似误差对后向 score 恢复精度的影响在摘要中不可见；随机逼近的收敛证明是否覆盖有限样本与神经回归误差，摘要未说明。

## 7. 对我们的启发

1. 免仿真 SB 训练路线可接入「#1 免训练 batch 级保边缘噪声指派 MPNA」：若 VSDM 的变分前向 score 估计能替代仿真式 IPF 半步，则 MPNA 的 batch 级噪声指派可省去轨迹仿真，降低训练成本。
2. 随机逼近收敛且无需 warm-up 的特性，对「#2 OT-aware 采样调度」有参考价值：若前向 score 可在线更新，采样调度可避免 warm-up 阶段，直接进入 OT-aware 采样。
3. 摘要未提供实验，暂无法支撑「#7 医学 SB 刷 SynthRAD」的具体迁移建议；需先读全文确认 VSDM 在高维图像数据上的可行性。

## 8. 资源

代码链接：未公开（摘要未提及）。相关论文：DSB（arXiv:2011.10636）、SB-FBSDE（arXiv:2111.08015）、LightSB（arXiv:2402.03259）、LightSB-M（ICML 2024）、[SF]²M（AISTATS 2024）。
