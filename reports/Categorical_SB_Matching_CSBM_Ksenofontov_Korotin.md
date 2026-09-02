# Categorical SB Matching (CSBM), Ksenofontov & Korotin

> Ksenofontov & Korotin · ICML (PMLR v267) 2025 · [PMLR](https://proceedings.mlr.press/v267/ksenofontov25a.html) · 证据级 [P] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：证明离散有限状态空间上 D-IMF 收敛到 SB，把 SB matching 推广到 VQ token/文本/分子等离散数据

⚠ 未读全文，依据摘要

## 1. 问题

Schrödinger Bridge（SB）在连续状态空间上的求解已有 IMF / bridge matching 类方法，但离散（有限）状态空间——如 VQ token、文本、分子等——缺少对应的 SB matching 理论。此前方法依赖连续状态假设，无法直接用于离散数据。

## 2. 方法

作者提出 Categorical SB Matching（CSBM），在离散有限状态空间上建立 D-IMF（Discrete Iterative Markovian Fitting）框架，证明其收敛到 SB。核心思想是把 SB matching 推广到离散状态空间，用离散时间/离散状态的迭代投影逼近 SB。摘要未给出具体公式与算法步骤，原文截断，未见。

## 3. 理论结果

摘要报告：证明离散（有限）状态空间上 D-IMF 收敛到 SB。具体假设、收敛率、误差界等细节原文截断，未见。

## 4. 实验与数字

摘要未给出数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本工作属于 SB 求解的「离散化」分支：ASBM（NeurIPS 2024）建立离散时间 D-IMF 理论并用 GAN 实现少步生成；CSBM 进一步把状态空间离散化，覆盖 VQ token/文本/分子等离散数据。与连续状态空间的 IMF（DSBM, IDBM）、α-DSBM（在线单网络更新）形成互补。对应课题背景中「离散空间的在线 SB（α-D-IMF）」开放问题的前置工作：CSBM 先建立离散空间 D-IMF 的交替收敛，α-D-IMF 的在线版本尚未出现。

## 6. 局限与批评

- 摘要未报告实验数字，无法评估实际效果（原文截断，未见）。
- 摘要未给出收敛率或误差界，理论保证的强度未知（原文截断，未见）。
- 未读全文，无法确认离散状态空间的具体假设（如状态数、转移核结构）是否限制适用范围。

## 7. 对我们的启发

1. 若后续读到全文，可检查 CSBM 的离散 D-IMF 是否保留两端边缘，作为离散 OT-aware 采样调度的理论依据（对应切入点 #2）。
2. 离散 SB matching 与 discrete flow matching 的关系值得追踪，可对接 VQ latent 生成任务（对应切入点 #3）。
3. 若 CSBM 的迭代投影在离散空间有闭式或轻量实现，可考虑用于 token 级保耦合蒸馏（对应切入点 #3 的少步生成方向）。

## 8. 资源

代码链接：未公开。相关论文：ASBM（NeurIPS 2024，离散时间 D-IMF）；DSBM（NeurIPS 2023）；IDBM（JMLR 2023）；α-DSBM（NeurIPS 2024）。
