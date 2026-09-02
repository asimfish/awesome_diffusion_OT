# InfoCD: A Contrastive Chamfer Distance Loss for Point Cloud Completion

> Lin et al. · NeurIPS 2023 · [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2ea1943896474b7cd9796b93e526f6f-Abstract.html) · 证据级 [P] · 课题 T20 3D/点云/几何生成中的 OT 与流
> **一句话**：用对比正则改进 Chamfer Distance，作者报告其等价于最大化底层曲面互信息下界，并被 UOT-UPC 选为最优 cost。

⚠ 未读全文，依据摘要

## 1. 问题

点云补全任务中，Chamfer Distance（CD）是常用损失，但存在对密度失衡与离群点不敏感的问题。本文提出 InfoCD，用对比学习正则化 CD，目标是让匹配点在特征空间中「摊开」以对齐分布。摘要未给出此前方法的具体缺陷描述或定量对比，仅能确认作者针对 CD 的上述不足提出改进。

## 2. 方法

核心思想：在 CD 基础上引入对比学习正则项，形成 InfoCD 损失。作者报告该损失等价于最大化底层曲面互信息的下界。摘要未给出具体公式、网络结构或训练流程细节，原文截断，未见。

## 3. 理论结果

摘要仅报告「等价于最大化底层曲面互信息下界」这一论断，未给出定理编号、假设条件或证明概要。原文截断，未见。

## 4. 实验与数字

摘要未包含任何数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

InfoCD 属于「修 CD 使其逼近 OT 性质」的度量线谱系，位于 DCD（NeurIPS 2021）与 HyperCD 之后。其与本课题的关联在于：UOT-UPC 将 InfoCD 选为 neural OT 的最优 cost，说明补全度量研究与传输研究在此合流——度量线产出被反向消费为传输问题的代价函数。这对应课题背景中「损失层面」与「耦合层面」的交叉环节。

## 6. 局限与批评

作者承认的局限：摘要未提及。读出来的局限：摘要未给出任何实验数字，无法评估其相对 DCD/HyperCD 的实际增益；「等价于互信息下界」的论断在摘要层面无法验证其假设范围与适用条件；对比正则的具体形式（温度、负样本策略、特征空间选择）均不可见，复现与比较受限。

## 7. 对我们的启发

1. InfoCD 被 UOT-UPC 选为最优 cost 这一事实提示：在补全/重建任务中，度量损失本身的质量会直接影响后续 OT 耦合求解的上限。可考虑在 NSOT 或 WFM 的点云生成管线中，将 InfoCD 作为评测或辅助损失，检验其是否比 CD/EMD 更利于下游 flow matching 的轨迹质量。
2. 对比正则与互信息下界的联系提示：若在 flow matching 的噪声-数据配对中引入类似「摊开」机制，可能降低轨迹交叉。可探索将 InfoCD 的对比项作为耦合正则，与 minibatch OT-CFM 结合，验证是否减少采样步数需求。
3. 摘要未给数字，建议在后续调研中获取全文，重点核对：InfoCD 在 ShapeNet 补全上的 CD/EMD 数值、与 DCD/HyperCD 的对比、以及 UOT-UPC 选用 InfoCD 的具体实验依据。

## 8. 资源

代码未公开（摘要未提及）。相关论文：DCD（NeurIPS 2021）、HyperCD、UOT-UPC（课题背景提及，arXiv id 未提供）。
