# TIGON

> Sha, Qiu, Zhou & Nie · Nat. Mach. Intell. 2024 · [NMI](https://www.nature.com/articles/s42256-023-00763-w) · 证据级 [P] · 课题 T24 单细胞与生物轨迹推断中的 OT×流
> **一句话**：用 Wasserstein–Fisher–Rao 动态 unbalanced OT 的 neural ODE 同时重建轨迹、增殖率与基因调控网络。

⚠ 未读全文，依据摘要

## 1. 问题

时序 scRNA-seq 在若干时间点给出独立细胞群体快照，细胞间无对应关系。要从这些快照恢复发育或扰动过程的连续动力学，需要同时处理两个耦合量：驱动细胞状态变化的速度场，以及由增殖/凋亡引起的质量变化。此前方法要么只输出离散耦合矩阵、无法外推到未观测时间或新细胞，要么把连续动力学与质量变化分开处理，难以在统一框架内同时重建轨迹、增殖率与基因调控网络。

## 2. 方法

作者报告 TIGON 用 Wasserstein–Fisher–Rao（WFR）距离的 neural ODE 求解动态 unbalanced OT。WFR 距离在 Wasserstein 传输项之外引入 Fisher–Rao 项，允许质量在演化中增减，从而把增殖/凋亡纳入测度演化的成本结构。TIGON 以 neural ODE 参数化连续动力学，同时学习速度场与增殖率，并据摘要反推时序基因调控网络。具体网络结构、训练目标与求解流程在摘要中未给出，原文截断，未见。

## 3. 理论结果

摘要未报告定理、引理或收敛保证。无理论结果（依据摘要）。

## 4. 实验与数字

摘要未给出数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

TIGON 处于「连续动力学化」阶段：它把 Waddington-OT 式的静态 unbalanced OT 耦合升级为 WFR 距离下的连续 neural ODE 动力学，与 TrajectoryNet（CNF + 动态 OT 罚项）、MIOFlow（流形约束流）同代。相对 TrajectoryNet 的纯 Wasserstein 动态，TIGON 把 unbalanced 质量变化作为第一等公民纳入 WFR 成本，直接输出增殖率。后续 DeepRUOT 在 RUOT 框架下用 Fisher 正则消掉 SDE 仿真、无先验学增殖，可视为对同一「unbalanced 连续动力学」问题的 simulation-free 推进；TIGON 的 neural ODE 仿真训练路线在维度与规模上受限于 ODE/PDE 求解，属于被后续免仿真方法部分取代的路线。

## 6. 局限与批评

作者承认的局限：摘要未提及，原文截断，未见。

读出来的局限：neural ODE 训练需要仿真 ODE/PDE，维度与规模受限，这是该路线相对后续 simulation-free 方法的共同短板；摘要未报告任何数值结果，无法评估其在真实 scRNA-seq 数据上的轨迹、增殖率与基因调控网络重建质量。

## 7. 对我们的启发

1. WFR 距离把「传输成本 + 质量变化成本」写进同一个测度演化目标，可作为我们设计 unbalanced 流模型训练目标时的成本结构参考，尤其当数据存在增殖/凋亡类质量不守恒时。
2. TIGON 同时输出速度场与增殖率并反推基因调控网络，提示「动力学参数 → 调控关系」的读出路径值得关注；若我们做医学 SB 或扰动外推，可考虑在学到的速度场上附加可解释读出层。
3. 其 neural ODE 仿真训练是明确瓶颈，提醒我们在新方法中优先采用免仿真训练（如 flow matching / Fisher 正则消仿真），避免重走 TIGON 的规模受限路线。

## 8. 资源

代码链接：未公开（依据摘要与元数据）。相关论文：TrajectoryNet（CNF + 动态 OT，连续动力学化早期工作）、DeepRUOT（RUOT 框架，Fisher 正则免仿真，unbalanced 连续动力学的后续推进）。
