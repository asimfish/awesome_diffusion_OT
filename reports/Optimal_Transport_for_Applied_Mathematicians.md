# Optimal Transport for Applied Mathematicians

> Santambrogio · Birkhäuser 2015 · 证据级 [B] · 课题 T01 OT 数学基础（面向生成模型研究者的最小必要集）
> **一句话**：应用数学侧 OT 标准参考书，系统给出 Kantorovich 对偶、Brenier 定理、Benamou–Brenier 动态形式与 Wasserstein 空间几何的严格证明。

⚠ 未读全文，依据摘要

## 1. 问题

本书面向应用数学研究者，解决「如何以严格但可读的方式掌握最优传输核心理论」的问题。OT 研究以最小代价把一个概率分布搬运成另一个：Monge 形式直接求映射 $T$，但该问题非凸且可能无解；Kantorovich 松弛为耦合 $\pi$ 上的线性规划后，OT 才成为可分析的对象。对扩散/流匹配研究者，这套语言不可绕过，因为生成模型本质上就是「把噪声分布运到数据分布」。此前 Villani（2003/2009）的理论百科虽全面，但重心偏分析理论；本书把重心移到应用数学侧，提供从对偶理论到动态形式的连贯证明链。

## 2. 方法

本书以教材形式组织 OT 核心理论，覆盖五块内容（据调研 agent 一句话贡献与课题背景）：

- **Kantorovich 对偶**：把 Monge 问题松弛为耦合上的线性规划，建立对偶理论，给出存在性。
- **Brenier 定理**：二次代价下最优映射唯一且为凸势梯度 $T=\nabla\varphi$，联结 Monge–Ampère 方程与凸分析。
- **Benamou–Brenier 动态形式**：把 $W_2^2$ 写成连续性方程约束下的最小动能，即静态耦合问题的时间流改写。
- **Wasserstein 空间几何**：$W_2$ 诱导的度量与测地结构，位移插值 $((1-t)\mathrm{Id}+tT)_\#\mu$ 作为分布形变的标准语言。
- **半离散 OT**：连续源到离散目标设定的理论与计算基础。

原文摘要未提供具体公式编号，无法标注 Eq. 编号。

## 3. 理论结果

原文摘要未提供具体定理陈述。据课题背景，本书覆盖的理论主线包括：Kantorovich 对偶理论与存在性；Brenier 极分解定理（二次代价下最优映射唯一且为凸势梯度）；Gangbo–McCann 在 twist 条件下向一般代价的推广；McCann 位移插值；Benamou–Brenier 动态形式。具体定理的假设与结论需读全文确认，此处不列。

## 4. 实验与数字

本书为理论教材，无实验与数值结果。原文摘要未提供任何数字。

## 5. 在 OT×扩散地图中的位置

本书处于 OT 理论教材演进线的「应用数学」节点：Villani（2003/2009）奠定理论百科 → **Santambrogio（2015）面向应用数学** → Peyré–Cuturi（2019）面向计算与数据科学 → Figalli–Glaudo（2021/2023）、Ambrosio–Brué–Semola（2024）课程化精炼 → Chewi–Niles-Weed–Rigollet（2024）统计化 → Peyré（2025 两部）直接面向生成模型研究者。对扩散×OT 研究者，本书提供的是「最小充分装备」中的理论证明层：Brenier 定理支撑「DDPM encoder ≈ OT map」类命题的判定（对应切入点 #3）；Benamou–Brenier 动态形式是 probability-flow ODE / flow matching 分析的同一语法（对应切入点 #2 的动能账本）；半离散 OT 对应「连续先验 → 有限样本数据集」的真实设定（对应切入点 #1 的 training-free guidance）。

## 6. 局限与批评

- 本书出版于 2015 年，早于熵正则/Sinkhorn（Cuturi 2013 刚出现）成为主流计算工具，也早于 flow matching 与扩散模型的 OT 化浪潮，计算与生成模型侧内容必然缺位（据课题背景教材演进线推断）。
- 作为教材，证明严格但面向应用数学读者，对 ML 背景读者存在分析语言门槛；Peyré（2025）等后续教材明确以「ML 最小数学装备 + 可运行代码」为替代定位。
- 原文摘要未提供作者自认局限，以上为基于出版时间与教材定位的推断。

## 7. 对我们的启发

- **作为理论查证底本**：当需要严格证明 Brenier 定理、Kantorovich 对偶或 Benamou–Brenier 等价性时，优先查本书对应章节，而非从 ML 论文的二手转述中拼凑。
- **支撑切入点 #2**：Benamou–Brenier 动能作为轨迹直度账本的严格定义与性质，应从本书获取完整证明链，再用于 rectified-flow 类迭代的单调性分析。
- **支撑切入点 #3**：Brenier 定理与凸势梯度的刻画，是判定 PF-ODE time-map 何时偏离 OT map 的理论基准；cyclical monotonicity 判据的严格形式需以本书为底。

## 8. 资源

- 作者稿 PDF：https://math.univ-lyon1.fr/~santambrogio/OTAM-cvgmt.pdf
- 相关论文互链：Villani《Topics in Optimal Transportation》(2003)、Villani《Optimal Transport: Old and New》(2009)、Peyré–Cuturi《Computational Optimal Transport》(2019, arXiv:1803.00567)、Cuturi《Sinkhorn Distances》(2013, arXiv:1306.0895)、Benamou–Brenier (2000, Numer. Math.)、Mérigot (2011)、Kitagawa–Mérigot–Thibert (2019, arXiv:1710.10075)。
