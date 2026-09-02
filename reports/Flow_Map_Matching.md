# Flow Map Matching

> FMM · TMLR 2025 · [OpenReview](https://openreview.net/forum?id=cqDH0e6ak2) · 证据级 [P] · 课题 T10 一致性模型与少步蒸馏的 OT 视角
> **一句话**：两时间 flow map 统一框架，证明 Lagrangian/Eulerian 蒸馏损失上界控制教师-学生 W2 距离，Eulerian 损失是一致性蒸馏的连续时间极限。

⚠ 未读全文，依据摘要

## 1. 问题

扩散/流模型采样需要数十到数百次网络评估（NFE），少步蒸馏的目标是把 PF-ODE 的解算子——即两时间 flow map——压缩进单个网络，实现 1–4 步生成。此前的一致性模型（CM）、一致性轨迹模型（CTM）、渐进蒸馏（Progressive Distillation）等方法各自提出不同的训练目标与采样方案，但缺乏一个统一的理论框架来说明：这些蒸馏损失究竟在什么度量意义下控制学生模型与教师模型之间的偏差。

## 2. 方法

作者提出以「两时间 flow map」为统一对象，将 CM、CTM、渐进蒸馏等方法组织为同一框架下的不同学习方案。核心区分是 Lagrangian 与 Eulerian 两种蒸馏损失形式：Lagrangian 损失沿轨迹采样点对，Eulerian 损失在固定时间网格上施加自一致约束。摘要指出，Eulerian 损失是一致性蒸馏（consistency distillation）的连续时间极限。具体公式与算法步骤在摘要中未给出，原文截断，未见。

## 3. 理论结果

摘要报告的核心理论结果：Lagrangian 与 Eulerian 蒸馏损失均给出教师模型与学生模型之间 Wasserstein-2（W2）距离的上界控制。即，蒸馏损失的减小直接约束教师-学生分布间的 W2 距离。定理的精确假设、常数因子与证明结构在摘要中未展开，原文截断，未见。

## 4. 实验与数字

摘要未包含任何实验数字、数据集或基线对比结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

该工作位于本课题「理论层」：蒸馏/一致性损失如何控制学生与教师分布之间的 Wasserstein 距离。它把 CM、CTM、渐进蒸馏统一为两时间 flow map 的学习方案，并给出 W2 上界，直接对应课题背景中「FMM 证明 Lagrangian/Eulerian 蒸馏损失给出 W2 上界」这一理论环节。与 Dou、Li 等人的 Wasserstein 统计估计率与离散化步数下界工作同属理论层，但 FMM 侧重蒸馏损失本身的上界性质，而非采样离散化的统计收敛。在方法演进脉络中，FMM 是一致性线（轨迹压缩）的理论统一者：CM/CTM/PD 均被纳入同一框架，其 Lagrangian/Eulerian 区分也为后续 NeurIPS 2025 将自蒸馏方案组织为 Eulerian/Lagrangian/Progressive 三族提供了基础。

## 6. 局限与批评

作者承认的局限：摘要未提及，原文截断，未见。

读出来的局限：摘要未报告任何实验数字，无法判断理论界在实践中的松紧程度；W2 上界控制的是教师-学生分布距离，但少步生成的实际评测指标（如 FID-NFE 曲线）与 W2 之间的关系未在摘要中建立；「Eulerian 损失是一致性蒸馏的连续时间极限」这一论断的离散化误差量级未在摘要中给出。

## 7. 对我们的启发

1. **保耦合蒸馏的理论接口**：FMM 的 W2 上界框架为切入点 #3（保耦合蒸馏）提供了可直接引用的理论工具——若在蒸馏目标中引入 OT 耦合替换独立耦合，可检验 W2 上界是否收紧，以及上界中的常数是否随耦合曲率降低而改善。
2. **OT-aware 采样调度的理论依据**：Lagrangian/Eulerian 两种损失对 W2 的控制方式不同，可为切入点 #2（OT-aware 采样调度）提供设计原则：在哪些时间点施加 Eulerian 约束、哪些区段用 Lagrangian 轨迹约束，以最小化 W2 上界。
3. **统一框架下的方法选型**：将 CM/CTM/渐进蒸馏视为同一 flow map 学习问题的不同损失选择，提示我们在做少步蒸馏实验时，应把损失形式（Lagrangian vs. Eulerian）作为独立变量消融，而非绑定具体方法名。

## 8. 资源

代码链接：未公开（摘要未提及）。相关论文互链：Consistency Models（ICML 2023）、Consistency Trajectory Models（ICLR 2024）、Progressive Distillation（Salimans & Ho, ICLR 2022）。
