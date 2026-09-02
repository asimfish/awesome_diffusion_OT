# An Invitation to Optimal Transport, Wasserstein Distances, and Gradient Flows

> Figalli & Glaudo · EMS Textbooks 2023 · [EMS](https://ems.press/books/etb/258) · 证据级 [B] · 课题 T01 OT 数学基础（面向生成模型研究者的最小必要集）
> **一句话**：146 页最短严格入门，覆盖对偶、Brenier、W 距离、JKO/Otto 微积分，含带解答习题。

⚠ 未读全文，依据摘要

## 1. 问题

本文是一本教材，不是研究论文。它要解决的问题是：给需要最优传输（OT）与 Wasserstein 距离严格数学基础的研究者提供一条最短的入门路径。调研 agent 给出的一句话贡献称其为「146 页最短严格入门」，覆盖对偶理论、Brenier 定理、Wasserstein 距离、JKO 格式与 Otto 微积分，并含带解答习题，体量为一学期课程。原文摘要未提供，以上信息来自调研 agent 的一句话贡献与元数据，非原文逐字证据。

## 2. 方法

作为教材，其「方法」是课程化的内容组织：把 OT 的核心数学装备压缩到 146 页，并配带解答习题。调研 agent 列出的覆盖内容包括：对偶理论、Brenier 定理、Wasserstein 距离、JKO 格式与 Otto 微积分。原文摘要未提供，无法给出具体公式编号或章节结构。

## 3. 理论结果

原文摘要未提供，无法列出具体定理、假设或结论。调研 agent 的一句话贡献提到覆盖「对偶、Brenier、W 距离、JKO/Otto 微积分」，但这些是教材主题而非可引用的定理陈述。按简报卡规则，本节只能写摘要里有的内容；摘要为空，故无理论结果可写。

## 4. 实验与数字

无实验。唯一数字是调研 agent 给出的「146 页」，出处为调研 agent 一句话贡献，非原文摘要。原文摘要未提供，未见其他数字。

## 5. 在 OT×扩散地图中的位置

本教材属于 T01「OT 数学基础」课题，定位是「面向生成模型研究者的最小必要集」。在方法演进脉络中，它处于教材谱系的课程化精炼阶段：Villani（2003/2009）理论百科 → Santambrogio（2015）应用数学 → Peyré–Cuturi（2019）计算与数据科学 → Figalli–Glaudo（2021/2023）、Ambrosio–Brué–Semola（2024）课程化精炼 → Chewi–Niles-Weed–Rigollet（2024）统计化 → Peyré（2025 两部）直接面向生成模型。它继承 Villani 与 Santambrogio 的理论框架，但压缩为更短的教学单元；与 Peyré–Cuturi（2019）的计算取向互补——后者侧重 Sinkhorn 与算法，本书侧重严格数学基础。对扩散×OT 研究者，它提供的是阅读 Brenier 定理、Benamou–Brenier 动态形式、半离散 OT 文献所需的底层语言。

## 6. 局限与批评

- 原文摘要未提供，作者自认的局限无法读取。
- 从元数据看，本书是 EMS Textbooks 教材而非预印本，定位为教学而非前沿研究，不包含新定理或新算法。
- 调研 agent 称其为「最短严格入门」，暗示其覆盖范围是经过取舍的：146 页体量下，熵正则/Sinkhorn、半离散 OT 的计算实现、统计估计等主题可能不深入或未覆盖（此为基于体量与定位的推断，原文未读，未见）。

## 7. 对我们的启发

1. 作为 T01 的基准教材，可将其对偶理论与 Brenier 定理部分作为阅读半离散 OT 对偶势 guidance（切入点 #1）的前置材料：半离散 OT 的对偶势梯度逼近分段 Brenier map，需要先掌握 Brenier 定理的凸势梯度刻画。
2. JKO 格式与 Otto 微积分部分可直接支撑对 Benamou–Brenier 动能账本（切入点 #2）的理解：BB 动态形式与 JKO 离散化是同一 \(W_2\) 几何的两种表述，教材若覆盖 Otto 微积分，则提供了把「轨迹直度」形式化的语言。
3. 带解答习题使其适合作为课题组短期读书会材料：按一学期课体量拆分，可在数周内完成对偶、Brenier、W 距离三块，再进入 Peyré–Cuturi（2019）的计算部分。

## 8. 资源

- 代码：未公开（教材，无代码）。
- 相关论文/教材 arXiv id：
  - Villani, *Topics in Optimal Transportation* (2003)；*Optimal Transport: Old and New* (2009)
  - Santambrogio, *Optimal Transport for Applied Mathematicians* (2015)
  - Peyré & Cuturi, *Computational Optimal Transport* (2019)，arXiv:1803.00567
  - Ambrosio, Brué & Semola, *Lectures on Optimal Transport* (2024)
  - Chewi, Niles-Weed & Rigollet, *Statistical Optimal Transport* (2024)，arXiv:2407.18163
  - 同作者免费 ETH 讲义：[An introduction to optimal transport and Wasserstein gradient flows](https://people.math.ethz.ch/~afigalli/lecture-notes-pdf/An-introduction-to-optimal-transport-and-Wasserstein-gradient-flows.pdf)
