# Learning Single-Cell Perturbation Responses using Neural Optimal Transport

> Bunne et al. · Nature Methods 2023 · [DOI](https://doi.org/10.1038/s41592-023-01969-x) · 证据级 [P] · 课题 T24 单细胞与生物轨迹推断中的 OT×流
> **一句话**：用 ICNN 对偶势学 control→perturbed 的 Monge map，预测未见药物与病人的单细胞扰动响应。

⚠ 未读全文，依据摘要

## 1. 问题

单细胞扰动响应预测的核心困难：时序 scRNA-seq 只给出 control 与 perturbed 两组独立细胞群体快照，细胞间无对应关系。传统方法无法在单细胞分辨率下预测「同一个细胞在药物处理后会发生什么变化」，更无法外推到训练中未见过的药物、剂量或病人。作者提出用神经最优传输学习 control→perturbed 的传输映射，把扰动响应建模为测度间的 Monge 问题。

## 2. 方法

核心思想：用输入凸神经网络（ICNN）参数化对偶势，学习 control 分布到 perturbed 分布的 Monge map。ICNN 保证映射是某个凸函数的梯度，从而在理论上对应最优传输映射。训练后，该映射可作用于任意 control 细胞，输出其扰动后的对应状态。摘要未给出具体公式编号与训练流程细节，原文截断，未见。

## 3. 理论结果

摘要未提及定理或理论保证。ICNN 的凸性结构隐含 Monge map 的数学性质，但摘要未给出形式化结论。原文未读，未见。

## 4. 实验与数字

摘要未给出具体数据集、基线或数值结果。调研 agent 标注涉及 4i 与 scRNA 数据，预测未见病人药物响应，但摘要本身未提供可引用的数字。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文是扰动线的奠基工作，位于「静态耦合时代」向「连续动力学化」过渡的关键节点：它不输出离散耦合矩阵，而是学一个可外推的神经传输映射，直接作用于新细胞。后续工作沿此线展开——Monge Gap 与 UOT-Monge 修正其 unbalanced 缺陷，MMFM 与 MFM 将其 FM 化并引入条件嵌入以泛化到未见条件，CellFlow 工程化为通用表型引擎。在理论张力上，它对应「离散耦合 → 连续映射 → 可外推条件生成」的管线升级。

## 6. 局限与批评

作者承认的局限：摘要未提及。读出来的局限：ICNN 对偶势方法通常受限于标准 OT 的保质量假设，无法直接处理扰动过程中的细胞增殖/凋亡（unbalanced 情形）；外推到未见药物/病人的泛化能力取决于条件信息的注入方式，而本文方法本身未在摘要中体现条件化机制。原文未读，未见。

## 7. 对我们的启发

1. 可接切入点 #3：将 CellOT 的 ICNN Monge map 作为 baseline，与 MFM（GNN 群体嵌入）、MMFM（classifier-free guidance）、CellFlow（条件编码器）做 head-to-head 对比，建立扰动外推的统一基准。
2. 可接切入点 #2：系统研究「谱系条码/RNA velocity」等生物先验注入 OT 成本或 unbalanced 松弛权重时，对 CellOT 型映射外推能力的影响。
3. 推理期重校准（切入点 #5）：对训练好的 CellOT 映射，用一次离散 OT 耦合构造 posterior steering，免重训地适配新批次或新病人快照。

## 8. 资源

代码链接：摘要未给出，原文未读，未见。相关论文：Monge Gap、UOT-Monge、MMFM、MFM、CellFlow（arXiv id 未提供，需后续补全）。
