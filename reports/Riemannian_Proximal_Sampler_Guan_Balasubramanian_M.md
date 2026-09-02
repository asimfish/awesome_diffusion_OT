# Riemannian Proximal Sampler, Guan, Balasubramanian & Ma

⚠ 未读全文，依据摘要

> Guan, Balasubramanian & Ma · NeurIPS 2025 · [proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/8e185f16e458ef5e666901260079cd42-Abstract-Conference.html) · 证据级 [P] · 课题 T28 黎曼流形上的流匹配与 OT
> **一句话**：用 MBI+热核双 oracle 在流形上高精度采样，$O(\log(1/\varepsilon))$ 迭代，可视为 Wasserstein 空间上熵正则黎曼 proximal point 的离散化。

## 1. 问题

本文处理黎曼流形上的采样问题。调研 agent 给出的一句话贡献指出，此前流形采样方法（如 RSGM 依赖热核的谱分解或渐近近似）在高维一般流形上难以扩展；本文提出的方法以 MBI（原文截断，未见全称）与热核双 oracle 为基础，目标是实现流形上的高精度采样，并给出 $O(\log(1/\varepsilon))$ 的迭代复杂度。摘要未提供更多关于此前方法不足的细节，上述背景来自课题材料，非本文摘要原文。

## 2. 方法

核心思想是把流形采样解释为 Wasserstein 空间上的熵正则黎曼 proximal point 的离散化。方法依赖两个 oracle：MBI 与热核。摘要未给出具体公式、算法步骤或训练/采样流程，原文截断，未见。

## 3. 理论结果

调研 agent 报告迭代复杂度为 $O(\log(1/\varepsilon))$，但摘要未给出定理编号、假设条件或完整结论陈述。原文截断，未见。

## 4. 实验与数字

摘要未提供数据集、基线或实验数字。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文位于课题 T28 的「空间层」：把 Wasserstein 空间自身当作无穷维黎曼流形做生成与优化。课题背景材料将其与 WFM/BW-FM、BW 流形上的 SVR-VI、ITSPACE 型 BW 单调更新归为同一板块，即「在分布空间上做几何优化/生成」。其 proximal point 视角与 Wasserstein proximal point 线直接相关；热核 oracle 技术被课题材料列为流形上 simulation-free SB matching（切入点 #4）的候选工具。与 RCM（流形一致性蒸馏）的少步推理线正交，本文侧重采样精度与迭代复杂度，而非少步生成。

## 6. 局限与批评

作者承认的局限：摘要未提供，原文截断，未见。

读出来的局限：方法依赖热核 oracle，而热核在一般高维流形上不可解析，课题背景材料明确指出这是 RSGM 的扩展瓶颈；本文是否绕开该瓶颈，摘要未说明。MBI 的具体含义与实现代价未在摘要中出现，无法评估其适用范围。

## 7. 对我们的启发

1. 热核 oracle 技术可接入切入点 #4（流形上的 simulation-free SB matching）：用热核截断或 Varadhan 渐近写出流形桥漂移的可回归形式，在球面气候插值上验证。
2. 熵正则黎曼 proximal point 的离散化视角，可为切入点 #2（曲率感知的 minibatch 测地 OT 偏差理论）提供 Wasserstein 空间上的优化解释，帮助设计曲率修正权重。
3. 若 MBI 在一般流形上可高效实现，可考虑与 RCM 蒸馏组合，检验高精度采样与少步推理的 NFE-质量 Pareto 曲线（对应切入点 #1 的实验设计）。

## 8. 资源

代码链接：未公开。相关论文：RSGM（NeurIPS 2022）、RDSB、Chen & Lipman RFM（ICLR 2024）、RCM、WFM/BW-FM（arXiv id 原文未提供，未见）。
