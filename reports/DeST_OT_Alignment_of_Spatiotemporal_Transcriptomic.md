# DeST-OT: Alignment of Spatiotemporal Transcriptomics Data

> ⚠ 未读全文，依据摘要
> Halmos et al. · Cell Systems 2025 · [Cell Systems](https://doi.org/10.1016/j.cels.2024.12.001) · 证据级 [P] · 课题 T24 单细胞与生物轨迹推断中的 OT×流
> **一句话**：用 semi-relaxed FGW 对齐时空转录组切片，建模生长/凋亡/分化并给出 growth-distortion 与 migration 度量。

## 1. 问题

时空转录组学数据来自发育或扰动过程中不同时间点、不同空间位置的切片。每个切片是带空间坐标的基因表达测度，切片之间没有细胞级对应关系。要对齐这些切片，需要同时处理三类生物过程：生长（growth）、凋亡（apoptosis）与分化（differentiation）。此前方法在跨切片对齐时，难以在统一的 OT 框架下同时建模这些过程并给出可解释的定量度量。摘要未给出此前方法的具体名称与不足细节，原文截断，未见。

## 2. 方法

作者提出 DeST-OT，核心是 semi-relaxed fused Gromov-Wasserstein（semi-relaxed FGW）框架。摘要未给出具体公式与算法步骤，原文截断，未见。从一句话贡献可知，方法用 semi-relaxed FGW 建模发育组织切片间的生长、凋亡与分化，并定义了两个度量：growth-distortion 与 migration。growth-distortion 度量生长与形变，migration 度量细胞迁移。具体数学形式、松弛方式、求解算法均未在摘要中给出。

## 3. 理论结果

摘要未提及任何定理、引理或理论保证。无理论结果（依据摘要）。

## 4. 实验与数字

摘要未给出数据集、基线方法或任何数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

DeST-OT 属于空间转录组配准这一任务线，是 PASTE 之后、moscot.spatiotemporal 同期的静态耦合方法。它用 semi-relaxed FGW 处理发育切片的生长与迁移，对应课题背景中「空间侧 PASTE→DeST-OT 用 semi-relaxed FGW 处理发育切片的生长与迁移」这一环节。与 WFM（flow matching 生成组织 niche）相比，DeST-OT 仍是静态耦合，无时间维动力学；与 moscot.spatiotemporal 同属空间时序静态对齐路线。在 OT×扩散地图中，它位于「静态耦合时代」的 GW/FGW 分支向空间时序任务的延伸，尚未进入 simulation-free 或连续动力学阶段。

## 6. 局限与批评

作者承认的局限：摘要未提及，原文截断，未见。

读出来的局限（依据摘要与课题背景）：
1. 静态耦合：DeST-OT 输出的是切片间的对齐/耦合，不是连续时间动力学，无法外推到未观测时间点或新切片。
2. 摘要未报告任何数值验证，无法判断 growth-distortion 与 migration 度量在真实数据上的可解释性与稳定性。
3. semi-relaxed FGW 的具体松弛方式（哪一侧松弛、松弛程度如何控制）未在摘要中说明，无法评估其对生长/凋亡建模的充分性。

## 7. 对我们的启发

1. 空间时序联合动力学（spatial SB）切入点直接相关：DeST-OT 是静态 FGW 耦合，可将其 growth-distortion 与 migration 度量作为监督信号或正则项，引入 FGW 型 bridge matching，在 (expression, location) 联合空间上学习连续迁移场与增殖场。
2. 若后续读到全文中的 growth-distortion 定义，可考虑将其作为 OT-aware 采样调度中的几何惩罚项，用于约束生成路径的空间形变程度。
3. DeST-OT 的 semi-relaxed 结构提示：在保耦合蒸馏或免训练 batch 级噪声指派中，对生长/凋亡主导的边缘可考虑非对称松弛，而非统一 unbalanced 权重。

## 8. 资源

代码链接：摘要未给出，原文截断，未见。相关论文：PASTE（空间对齐前作，arXiv id 未在摘要中给出）；moscot（Nature 2025，空间时序静态对齐工程化）；WFM（flow matching 生成组织 niche，arXiv id 未在摘要中给出）。
