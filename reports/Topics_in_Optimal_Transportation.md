# Optimal Transport: Old and New

> Cédric Villani · Springer Grundlehren 338, 2009（本地全文为 2006-09-27 Saint-Flour 讲义预印，547 页）· 证据级 [B] · 课题 T01 OT 数学基础
> **一句话**：OT 理论百科：30 章覆盖对偶、位移插值、Monge 问题解、正则性、Ricci 曲率与梯度流；查证明时用，不作第一本。
> 导读说明：manifest 的 report_id 为 `Topics_in_Optimal_Transportation`（Villani 2003 年 AMS 教材名），但清单标题与本地 PDF 均为 *Optimal Transport: Old and New*（2009）的作者预印；本报告按后者写。`data/text` 只含前 90k 字符（前言、约定、导论第 1–3 章、第 4 章开头，约 pp. 1–44）。§2–§3 中第 5 章以后只据目录与前言定位，定理编号不引。

## 1. 这本书解决读者什么问题
它是对 OT 现代理论的第二次系统重写：作者在前言中说，这不是《Topics in Optimal Transportation》（2003）的缩写或扩写，而是「从不同视角、用不同证明、更概率化地重写整个理论，并纳入近期进展」——「更多概率、更多几何、更多动力系统；更少分析、更少物理」（Preface, p. 2）。两个新支柱：Mather 极小测度与 OT 可嵌入同一框架（动力系统）；OT 给出 Ricci 曲率下界的稳健综合定义（几何）。

对读者的承诺是「对最重要结果给出完整自包含证明」，并承认篇幅膨胀到原计划的五倍，建议非专家「首读跳过长证明，专注解释、陈述与例子」（Preface, p. 2）。写法上的核心选择（Preface, p. 2–3）：McCann 位移插值在任何 Monge 问题可解性定理之前引入，用抽象「Lagrangian 作用量」统一长度空间与黎曼流形上的光滑代价；「从中间时刻而非初始时刻看 OT」是贯穿全书的想法，用 Mather 估计证中间传输映射的 Lipschitz 正则性，从而免费得到位移插值的绝对连续性。

## 2. 章节结构与推荐阅读路线
目录（pp. IX–X）：导论 3 章 + 三部分 27 章。

| 部分 | 章 | 内容 |
|---|---|---|
| 导论 | 1 耦合与变量替换 · 2 三个耦合技术例子（Langevin 收敛、Knothe–Rosenblatt 证几何不等式、Moser/Caffarelli 型收缩）· 3 OT 的奠基者 | 概率化预备与历史 |
| I 定性描述 | 4 基本性质 · 5 循环单调与 Kantorovich 对偶 · 6 Wasserstein 距离 · 7 位移插值 · 8 Monge–Mather 缩短原理 · 9/10 Monge 问题的解（整体/局部）· 11 Jacobian 方程 · 12 光滑性 · 13 定性全景 | **T01 核心**：对偶、$W_p$、位移插值、Brenier–McCann、正则性 |
| II OT 与黎曼几何 | 14 Ricci 曲率 · 15 Otto 微积分 · 16/17 位移凸性 · 18 体积控制 · 19 密度控制与局部正则 · 20 无穷小位移凸性 · 21 等周型不等式 · 22 集中不等式 · 23–25 梯度流 I/II/III | T05 的理论源头；§15 是 Benamou–Brenier 的黎曼几何形式 |
| III Ricci 曲率的综合处理 | 26–30 解析/综合观点、度量测度空间收敛、OT 稳定性、弱 Ricci 界 | 与扩散研究基本无关 |

面向扩散/流研究者的最小必要集（本书只作「查证明」用，路线按需跳读）：
1. **Monge/Kantorovich**：第 4 章 Thm 4.1（Polish 空间、下半连续有下界代价 ⇒ 最优耦合存在）、Lemma 4.2/4.3、Remark 4.4（代价可能无穷，$c\le c_X+c_Y$ 条件）、限制性质。
2. **对偶**：第 5 章——全书强调「$c$-循环单调映射」既在陈述也在证明中（Part I 引言，p. 41）；这是最一般（Polish 空间、下半连续代价）的 Kantorovich 对偶证明出处。
3. **Wasserstein 距离**：第 6 章（$W_p$ 度量弱收敛、$\mathcal P_p$ 的拓扑）——Peyré–Cuturi Remark 2.18 转引的正是此章。
4. **Brenier**：第 9–10 章（Monge 问题的整体/局部解法，涵盖二次代价欧氏情形与黎曼流形上的 McCann 推广）、第 11 章 Jacobian（Monge–Ampère）方程、第 12 章光滑性（Caffarelli；作者自承正则性是「表现最差的主题」）。
5. **Benamou–Brenier / 位移插值**：第 7 章（位移插值作为 Lagrangian 作用量极小、在任何可解性定理之前引入）、第 15 章 Otto 微积分（$W_2$ 的形式黎曼结构）。
6. **梯度流/JKO**：第 23–25 章（定义与收敛、定性性质、泛函不等式）；作者自评「精确但不穷尽」，并推 AGS 为完整参考（Preface, p. 3）。
7. **熵正则/Sinkhorn、半离散**：本书**不含**数值方法（Preface 明言「numerical simulation ... not addressed at all」，p. 3）。

第 2 章值得扩散研究者单独读：Langevin 过程收敛用同步耦合（同一 Brownian 运动）几行证出——这是 T02/T06 所有「$W_2$ 收缩 ⇒ 采样收敛」论证的原型；Ex. 1.5 把热方程解写成 Brownian 耦合（Ch. 1）。

## 3. 关键定理清单（定理名 + 所在章节）
已读部分（逐条核对）：
- Def. 1.1 耦合、Def. 1.2 确定性耦合（$T_\#\mu=\nu$）；Remark 1.3–1.4 Jacobian 与近似可微；Ex. 1.5 热方程解 = Brownian 运动的律（第 1 章）。
- 第 2 章三例：(i) 过阻尼 Langevin (2.2) 与副本 (2.3) 共用同一 Brownian 运动 ⇒ $V$ 一致凸时指数收敛；(ii) Knothe–Rosenblatt 耦合证几何不等式；(iii) Moser 传输的收缩性 $|T(x)|\le|x|$（第 2 章）。
- 第 3 章历史：Monge 1781 的「déblais/remblais」（代价 = 质量 × 距离）；Kantorovich 1942 对偶与距离（Kantorovich–Rubinstein），且「几年后才联系到 Monge」；80 年代末三条独立路线——Mather 极小测度、Brenier 极分解（起于不可压流体的测度保持映射投影，揭示 Monge–Ampère 联系）、Cullen 半地转气象方程；Otto 引入微分观点连接扩散方程（第 3 章）。
- **Thm 4.1（最优耦合存在）**：$(\mathcal X,\mu),(\mathcal Y,\nu)$ Polish、$c$ 下半连续有下界 ⇒ 存在极小化 $\mathbb E\,c(X,Y)$ 的耦合。证明 = Lemma 4.2（代价泛函弱下半连续）+ Lemma 4.3（边缘紧 ⇒ 方案集紧，Prokhorov）（第 4 章）。
- Remark 4.4：存在性不保证有限代价；$\int c\,d\mu\,d\nu<\infty$ 或 $c\le c_X+c_Y$ 排除之（第 4 章）。

未读部分（据目录与 Part I 引言）：Kantorovich 对偶与 $c$-循环单调（第 5 章）；$W_p$ 度量化弱拓扑（第 6 章）；位移插值与 Lagrangian 作用量（第 7 章）；Mather 缩短原理 ⇒ 中间映射 Lipschitz（第 8 章）；Brenier–McCann 型解与 $c$-凸势（第 9–10 章）；Jacobian/Monge–Ampère 方程在非光滑映射下仍成立（第 11 章）；Caffarelli 正则性（第 12 章）；Otto 微积分（第 15 章）；McCann 位移凸性（第 16–17 章）；Talagrand/HWI 型集中不等式（第 22 章）；JKO 与梯度流（第 23–25 章）。

## 4. 与代码/软件的对应
本书无任何算法与代码（Preface, p. 3）。可对应的只有「理论 → 可计算特例」：

| 章节 | 可计算对应（我的整理） |
|---|---|
| 第 2 章 Langevin 同步耦合 | 任何 SDE 采样器的 $W_2$ 收缩验证；扩散模型收敛分析（T06）的证明模板 |
| 第 6 章 $W_p$ | POT `ot.emd2` / `ot.wasserstein_1d`；OTT-JAX `Sinkhorn` 的 $\varepsilon\to0$ 极限 |
| 第 7 章位移插值 | POT `ot.emd` 得耦合后按 $(1-t)x+ty$ 推前；torchcfm `ExactOptimalTransportConditionalFlowMatcher` 的条件路径即 minibatch 版位移插值 |
| 第 9–10 章 Brenier–McCann | 高斯闭式 `ot.gaussian.bures_wasserstein_mapping`；一般情形只能经半离散（pysdot）或神经近似（OTT-JAX `ott.neural`） |
| 第 15 章 Otto 微积分 | 无直接实现；Benamou–Brenier 离散化见 1803.00567 第 7 章 |

## 5. 在 OT×扩散地图中的位置
- T01 内：教材演进的起点（Villani 2003/2009 → Santambrogio 2015 → Peyré–Cuturi 2019 → Figalli–Glaudo / ABS → Chewi 等 → Peyré 2025）。其他教材的证明多转引本书：Peyré–Cuturi Prop. 2.2/Remark 2.18 引其 gluing 与弱收敛定理，2505.06589 Remark 2.26/Prop. 2.31 引其一般代价与正则性结果，1603.05579 引其第 12 章的 MTW 条件。
- 对扩散×OT：第 2 章的同步耦合是「$W_2$ 收缩」论证的原型；第 7 章「从中间时刻看 OT」与位移插值的 Lagrangian 观点，正是 flow matching 把生成建模写成 $t\mapsto\alpha_t$ 的语法源头（2512.06797 §2 的 Eulerian/Lagrangian 二分即此）；第 16–17 章位移凸性是 T05 收敛分析（McCann 条件）的出处；第 22 章 Talagrand 不等式连接 T06 的 KL–$W_2$ 界。
- 与 T01 其他书的分工：本书给最一般的存在性/对偶（Polish 空间、下半连续代价），Santambrogio 给可读证明，Peyré 两书给计算与 ML 接口——扩散研究者应「先 Peyré/Santambrogio，遇到假设边界再回本书」。

## 6. 局限与批评
作者承认（Preface, pp. 2–3）：
1. 正则性理论「表现最差」，因为「故事太长且本书目的不需要」；期待专著。
2. 数值方法「完全不涉及」；Monge–Mather–Mañé 问题「发展不足」。
3. 梯度流章（23–25）「不穷尽」，且本预印版明言第 22–25 章「仍需重做/调试」（p. VII）。

我读出的：
1. 本地全文是 2006 年预印，与 2009 出版版的章节编号、定理编号有差异，引用定理号必须核对出版版。
2. 全书几乎不谈熵正则、半离散、统计率与生成模型——对本课题「五块装备」只覆盖对偶/Brenier/BB 三块的理论极限版，且以黎曼流形为默认场景，对欧氏 $\mathbb R^d$ 读者有额外负担。
3. 「更多概率」的承诺主要体现在耦合语言与第 2 章，Part II–III 仍是几何分析；ML 读者大约只需第 4–7、9–10、15–17 章。

## 7. 对我们的启发
1. 写 T02/T06 类收敛证明时，第 2 章同步耦合 + 第 6 章 $W_p$ 弱收敛是最短的引用链；比转引二手综述更稳。
2. Top-10 #2（OT-aware 调度）：第 7 章「从中间时刻 $t_0$ 出发的 OT」提示可把调度比较改成「任意中间边缘 $\alpha_{t_0}\to\alpha_1$ 的 OT 代价」曲线，而非只看端点。
3. 凡宣称「学到的映射是 OT 映射」的论文，其假设边界（源测度不给小集合赋质量、代价 twist、目标凸性）都应回本书第 9–10、12 章核对，而不是引 Brenier 1991 原文的欧氏特例。

## 8. 资源
- 全文：[作者免费预印（2006）](https://www.ceremade.dauphine.fr/~mischler/articles/VBook-O&N.pdf)；[Springer 出版版](https://link.springer.com/book/10.1007/978-3-540-71050-9)。无代码。
- 互链：Optimal_Transport_for_Applied_Mathematicians（同类理论书的可读版）、Lectures_on_Optimal_Transport（两种自包含对偶证明）、An_Invitation_to_Optimal_Transport_Wasserstein_Dis（最短严格入门）、2505.06589 / 1803.00567（转引本书定理的计算教材）、Polar_factorization_and_monotone_rearrangement_of（第 3 章历史中的 Brenier 路线）、1603.05579（引本书 MTW 条件）、The_Variational_Formulation_of_the_Fokker_Planck_E（第 23–25 章梯度流的源头论文）。
