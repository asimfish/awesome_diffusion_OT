# Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow

> Wang et al. · ICLR 2025 · [OpenReview](https://openreview.net/forum?id=nEDToD1R8M) · 证据级 [P] · 课题 T09 Rectified Flow 与轨迹拉直
> **一句话**：论证 rectification 的本质是「预训练模型配对 + 重训」，而非直线度、流匹配形式或 v-预测；推广为一般扩散的一阶 ODE 目标。

⚠ 未读全文，依据摘要

## 1. 问题

Rectified Flow（RF）的 reflow 操作被广泛理解为一种「拉直轨迹」的手段：通过用当前模型自身生成的 $(Z_0, Z_1)$ 配对重训，迭代降低轨迹曲率，最终使概率流 ODE 可被单步 Euler 精确模拟。但这一几何叙事把「直线度」当成了 rectification 的核心目标。

本文要解决的问题是：rectification 真正起作用的条件和机制到底是什么。作者主张，直线度不是本质——本质是「预训练模型配对 + 重训」这一操作本身，它能在更一般的扩散模型形式下（包括 DDPM 形式）得到一阶 ODE 路径，即使轨迹天然弯曲也成立。

## 2. 方法

核心思想是把 rectification 从 RF 的特定公式中解耦出来。作者论证：rectification 的关键不在于流匹配（flow matching）形式、不在于 v-预测（v-prediction）参数化、也不在于直线度，而在于用预训练模型生成配对数据后重训这一过程。

摘要未给出具体公式或算法步骤。原文截断，未见。

## 3. 理论结果

摘要未列出具体定理或引理。作者报告的核心理论主张是：rectification 可推广为一般扩散模型的一阶 ODE 目标，且该推广不依赖直线度假设。具体假设与结论形式原文截断，未见。

## 4. 实验与数字

摘要未给出数据集、基线或具体数值。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文与 rfpp（Lee et al., NeurIPS 2024）同属对 reflow 机制的反思线：rfpp 实证「一轮 reflow 就够直」，把瓶颈归于时间步分布与损失度量；本文更进一步，主张直线度本身就不是 rectification 的本质。在理论张力上，本文与 Bansal et al. 的 W2 收敛界（误差 ∝ 直线度参数/步数²）形成对照——若直线度非本质，则「用直线度预测 few-step 质量」的度量路线需要重新审视。同时，本文与 Hertrich–Chambolle–Delon（NeurIPS 2025）的反例工作方向一致：都在解耦「直线化」与「最优性/有效性」。

## 6. 局限与批评

- 证据级为 [P]，仅依据摘要，无法核实其论证的完整性与实验支撑。
- 摘要未给出任何定量实验，无法判断「直线度非本质」这一主张在 few-step 采样质量上的实际影响幅度。
- 若直线度确实非本质，则本文与 Bansal et al. 的 W2 界之间的张力需要进一步厘清：直线度在什么条件下仍是一个有用的代理量，摘要未涉及。

## 7. 对我们的启发

1. 若 rectification 的本质是「配对重训」而非直线度，则切入点 #1（免训练 batch 级保边缘噪声指派 MPNA）的动机被加强：配对质量才是核心，直线度只是 RF 特定公式下的副产品。
2. 可设计对照实验：固定配对质量、扫描直线度 $S(Z)$，检验 few-step FID 是否随 $S(Z)$ 单调变化——这正是 §5 开放问题 2 的「直线度-质量解耦」方向，本文为其提供了理论动机。
3. 若一般扩散的一阶 ODE 目标成立，则 DDPM 形式下的 reflow 变体可纳入统一框架，为切入点 #3（保耦合蒸馏）在非 RF 公式的扩散骨干上推广提供依据。

## 8. 资源

代码：未公开（摘要未提及）。  
相关论文：rfpp（Lee et al., NeurIPS 2024）；Bansal et al.（W2 收敛界）；Hertrich–Chambolle–Delon（NeurIPS 2025，reflow 反例）；Liu et al. 2022（RF 奠基）。
