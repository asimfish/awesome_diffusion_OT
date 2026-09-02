# A Combinatorial Algorithm for Semi-Discrete Optimal Transport

> 作者（前 3 位 + et al.）· NeurIPS 2024 · [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2d950a2cfd8a75124c178a89545b97fd-Abstract-Conference.html) · 证据级 [P] · 课题 T01 OT 数学基础（面向生成模型研究者的最小必要集）
> **一句话**：提出半离散 OT 的组合算法路径，区别于主流 smooth dual/Newton 方法。

⚠ 未读全文，依据摘要

## 1. 问题

半离散最优传输（semi-discrete optimal transport）处理「连续源分布 → 离散目标分布」的运输问题。该设定在生成模型中对应「连续先验 → 有限样本数据集」的真实场景。此前主流求解路径是 smooth dual / Newton 类方法（如 Kitagawa–Mérigot–Thibert 2019 的阻尼牛顿法）。本文提出一条不同的组合算法（combinatorial algorithm）路径。摘要未给出此前方法的具体不足或本文针对的失败模式，原文截断，未见。

## 2. 方法

核心思想是组合算法路径，与主流的 smooth dual / Newton 方法形成区分。摘要未给出具体算法步骤、关键公式或伪代码，原文截断，未见。

## 3. 理论结果

摘要未给出定理、引理或收敛性保证的具体内容，原文截断，未见。

## 4. 实验与数字

摘要未给出数据集、基线或关键数值，原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文属于半离散 OT 的计算方法线，与 Kitagawa–Mérigot–Thibert（2019）的阻尼牛顿全局线性收敛、Mérigot（2011）多尺度方法、Aurenhammer 等（1998）的 power/Laguerre diagram 方法同处一条脉络。其「组合算法」定位与主流 smooth dual/Newton 形成方法学上的分叉。在扩散×OT 地图中，半离散 OT 精确对应「连续先验 → 有限样本数据集」的设定，因此本文若提供更快的求解器，可能服务于对偶势计算、training-free guidance 等下游环节；但摘要未给出复杂度或实验证据，无法判断其实际替代能力。

## 6. 局限与批评

作者承认的局限：摘要未提及，原文截断，未见。

读出来的局限：摘要未给出任何理论保证或实验数字，无法评估该组合算法相对 damped Newton（KMT 2019）在收敛速度、数值稳定性或实现复杂度上的优劣；「组合算法」的具体含义在摘要层面不可验证。

## 7. 对我们的启发

1. 若该组合算法在求解半离散 OT 对偶势上比 damped Newton 更快或更稳，可替换切入点 #1（training-free guidance）中的对偶势求解器，降低 batch 级 guidance 的计算开销。
2. 半离散 OT 求解器是「连续先验 → 有限样本」设定的基础组件；关注该算法是否提供对偶势梯度（即分段 Brenier map 的近似），这决定它能否直接嵌入扩散采样器的 drift 注入。
3. 在未读到全文与实验数字前，暂不将其纳入任何依赖半离散 OT 求解速度的工程方案；先等待复现或与 KMT 2019 的对比数据。

## 8. 资源

代码：未公开（摘要未提及）。  
相关论文：Kitagawa–Mérigot–Thibert 2019（damped Newton 全局线性收敛）；Mérigot 2011（多尺度半离散 OT）；Aurenhammer et al. 1998（power diagram 与半离散 OT）。
