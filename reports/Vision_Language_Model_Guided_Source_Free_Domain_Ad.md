# Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport

> Han et al. · CVPR 2026 · [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Han_Vision-Language_Model_Guided_Source-Free_Domain_Adaptation_via_Optimal_Transport_CVPR_2026_paper.html) · 证据级 [P] · 课题 T17 风格迁移与域自适应中的 OT×扩散
> **一句话**：用 VLM 语义先验引导源原型与目标特征的 OT 对齐，做 source-free DA。

⚠ 未读全文，依据摘要

## 1. 问题

本文处理 source-free domain adaptation（SFDA）：源域数据在适配阶段不可用，只能拿到源模型与目标域无标注数据。此前 SFDA 方法在缺乏源数据时难以可靠地估计源域特征分布，导致源-目标对齐退化。本文提出用 vision-language model（VLM）的语义先验来补偿源数据缺失，并以最优传输（OT）完成源原型与目标特征的对齐。摘要未给出此前方法的具体名称或量化缺陷。

## 2. 方法

核心思想：用 VLM 语义先验引导源原型与目标特征的 OT 对齐。摘要未给出公式、算法步骤或训练流程细节；原文截断，未见。

## 3. 理论结果

摘要未报告任何定理、引理或理论保证。无理论结果（依据摘要）。

## 4. 实验与数字

摘要未给出数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文位于模型迁移层的 SFDA×OT 分支，与课题背景中提到的 semi-dual OT 逐步域迁移（ICML 2026）及 SFDA×OT（CVPR 2025/2026）同属一类。其差异点在于引入 VLM 语义先验作为源域分布的替代信息源，对应「预训练大模型如何低成本适配新域」这一扩散/DA 交叉问题。与扩散机制层（SW-Guidance、OT-ALD）无直接继承关系，摘要未提及扩散模型。

## 6. 局限与批评

作者承认的局限：摘要未报告。读出来的局限：摘要未给出任何实验数字或理论结果，无法评估对齐质量或计算成本；VLM 语义先验的质量依赖预训练 VLM 与目标域语义的匹配程度，摘要未说明失效条件。

## 7. 对我们的启发

1. 若 VLM 先验能稳定替代源数据，可尝试把同样的「语义先验 + OT 对齐」思路用于扩散增广式 DG/DA，给伪域生成补上 OT 几何控制（对应切入点 #3）。
2. 关注其源原型构造方式是否可迁移到 latent 空间，与 OT-ALD 的 latent 起点修正结合（对应切入点 #2）。
3. 若后续全文给出类级对齐细节，可对照 WAT 的类级 OT 匹配，检验 VLM 先验在不平衡/开集场景下的表现（对应切入点 #4）。

## 8. 资源

代码链接：未公开。相关论文：semi-dual OT 逐步域迁移（ICML 2026，原文未给 arXiv id）；SFDA×OT（CVPR 2025/2026，原文未给 arXiv id）。
