# Riemannian Flow Matching Policy (RFMP), Braun et al.

> Braun et al. · IROS 2024 · [IEEE](https://doi.org/10.1109/iros58592.2024.10801521) · 证据级 [P] · 课题 T28 黎曼流形上的流匹配与 OT
> **一句话**：把黎曼流匹配用于机器人视觉运动策略，作者报告比 Diffusion Policy 推理更快、轨迹更平滑。

⚠ 未读全文，依据摘要

## 1. 问题

机器人视觉运动策略（visuomotor policy）需要从观测生成动作序列。动作常包含姿态分量，其取值在 SO(3)/SE(3) 等黎曼流形上，而非欧氏空间。此前基于扩散的策略（如 Diffusion Policy）在欧氏空间建模动作，直接套用到流形值动作上会产生离开流形的样本；同时扩散采样步数多，推理慢。本文要解决的是：如何在流形值动作空间上定义并训练一个生成式策略，使其采样快、轨迹平滑。

## 2. 方法

核心思想是把 Riemannian Flow Matching（Chen & Lipman, ICLR 2024）用作策略的生成模型：在流形上定义条件概率路径与向量场，回归该向量场，推理时沿学到的流形向量场积分生成动作序列。摘要未给出具体公式、流形选择（SO(3) 还是 SE(3)）、网络结构或训练/采样流程的细节；原文截断，未见。

## 3. 理论结果

摘要未报告定理、引理或理论保证。无理论结果（依据摘要）。

## 4. 实验与数字

摘要未给出数据集名称、基线配置、数值指标或消融数字。作者报告两点定性/相对结论：相比 Diffusion Policy，RFMP 推理更快、生成轨迹更平滑。具体加速倍数、平滑度度量、成功率等数字原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文属于 T28 中「生成层」的流形 FM 分支，直接继承 Chen & Lipman 的 Riemannian Flow Matching（ICLR 2024 Oral），把该框架从生成任务迁移到机器人策略学习。与 Diffusion Policy 构成竞争关系：后者是欧氏扩散策略的代表，本文用流形原生建模替代欧氏建模。与 TDM（李群平凡化动量）、SE(3) 蛋白骨架系（FrameDiff/FoldFlow）同属「流形生成 × 应用」谱系，但本文落在机器人动作生成，不涉及 OT 耦合层或 Wasserstein 空间层。摘要未提 OT 配对或测地成本最优性，因此与 Riemannian minibatch OT、RNOT 等理论线无直接关系。

## 6. 局限与批评

作者承认的局限：摘要未列出。读出的局限：摘要未报告任何数值实验证据，无法判断「更快」「更平滑」的幅度与统计显著性；未说明流形选择是 SO(3) 还是 SE(3)，也未说明如何处理位置分量的欧氏部分与姿态分量的流形部分的混合动作空间；未提与欧氏 Diffusion Policy 在相同流形投影/参数化下的公平对比。

## 7. 对我们的启发

1. 若后续读到全文，可检查其流形 FM 策略是否用了 OT 配对训练；若没有，可接切入点 #1（免训练 batch 级保边缘噪声指派 MPNA）或流形 minibatch 测地 OT 配对，直接改进训练耦合。
2. 其「推理更快」的卖点与 RCM（流形一致性蒸馏）正交：可在 RFMP 之上做流形少步蒸馏，进一步压 NFE，对应切入点 #3 保耦合蒸馏的流形版。
3. 若其动作空间含 SE(3)，可检验「单切空间谬误」（Jaquier et al.）在该策略中的近似误差，并考虑用测地 OT 对齐仿真→真机姿态分布（切入点 #5），作为 training-free guidance 层。

## 8. 资源

代码未公开（摘要未提供链接）。相关论文：Riemannian Flow Matching（Chen & Lipman, ICLR 2024, arXiv:2402.03647）；Diffusion Policy（Chi et al., RSS 2023, arXiv:2303.04137）；Riemannian Consistency Models（arXiv:2503.16087）。
