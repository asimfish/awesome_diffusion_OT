# Léonard, A Survey of the Schrödinger Problem

> ⚠ 未读全文，依据摘要
> Léonard · DCDS 2014 · [DOI](https://doi.org/10.3934/dcds.2014.34.1533) · 证据级 [B] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：系统梳理 Schrödinger 问题，建立静态/动态等价、熵正则 OT 关系与大偏差极限。

## 1. 问题

Schrödinger 问题（Schrödinger Problem, SP）源于 1932 年 Schrödinger 提出的一个关于扩散过程的大偏差问题：给定参考过程（通常为 Brownian 或 OU 过程），在所有满足两端边缘约束 $p_0=\mu$、$p_1=\nu$ 的路径测度中，寻找与参考过程 KL 散度最小的那个。该问题在数学上长期缺乏统一表述，静态形式（耦合上的熵正则最优传输）与动态形式（路径空间相对熵最小化）之间的关系、以及它与大偏差理论的联系，散见于不同文献中。

此前的工作没有提供一份从静态到动态、从变分表述到大偏差极限的完整梳理。本文作为综述，目标是把这些分散的结果整合为一个自洽的理论框架。

## 2. 方法

本文是综述性工作，不提出新算法。其核心组织逻辑是：

1. **静态与动态的等价性**：Schrödinger 问题的静态投影形式为熵正则最优传输（EOT），即在耦合上最小化 $\min_\pi KL(\pi \| \pi_Q)$，其中 $\pi_Q$ 是参考过程的联合分布；动态形式则是在路径测度空间上最小化与参考过程 $Q$ 的相对熵。
2. **大偏差极限**：Schrödinger 问题可被理解为大量独立 Brownian 粒子在给定两端经验分布条件下的大偏差极限，这为静态/动态等价提供了概率论解释。
3. **与熵正则 OT 的关系**：静态 Schrödinger 问题恰是熵正则最优传输，正则化强度由参考过程的扩散系数决定。

由于未读全文，具体公式编号与推导细节无法给出。

## 3. 理论结果

摘要未列出具体定理编号或数字。本文作为综述，其理论贡献在于**整合既有结果**：静态 Schrödinger 问题与熵正则 OT 的等价性、动态形式与静态形式的等价性、以及大偏差解释。具体定理的假设与结论需阅读全文确认。

## 4. 实验与数字

本文为纯理论综述，无实验部分，无数据集、基线或数值结果。

## 5. 在 OT×扩散地图中的位置

本文是 Schrödinger Bridge 理论线的**奠基性综述**。在课题 T03 的脉络中，它处于「理论奠基（1932→2021）」阶段的核心位置：把 Schrödinger 1932 年的大偏差问题系统化为「路径空间相对熵最小化 ⇔ 静态熵正则 OT + 参考桥」这一框架。后续 Chen-Georgiou-Pavon (2021) 补上随机控制视角，DSB (NeurIPS 2021)、SB-FBSDE (ICLR 2022) 等算法工作均以此框架为理论出发点。本文对应的是「静态 EOT 与动态 SB 的等价性」这一理论张力，为后续所有基于 IPF/Sinkhorn 或 bridge matching 的算法提供了对偶路线的共同语言。

## 6. 局限与批评

- **作者承认的**：未读全文，无法确认作者自述局限。
- **读出来的**：作为 2014 年的综述，它不涉及任何深度学习方法，也不讨论高维求解的计算可行性；其理论框架止于连续时间、参考过程为 Brownian/OU 的经典设定，对后续算法关心的有限样本、神经网络参数化、离散化误差等问题没有覆盖。此外，综述的覆盖面取决于作者选材，可能未纳入当时刚起步的计算最优传输视角。

## 7. 对我们的启发

1. **静态-动态等价是免训练方法的理论根基**：本文确立的「SB 静态投影 = EOT」关系，直接支持用静态 minibatch Sinkhorn 耦合替代动态迭代的免仿真路线（如 [SF]²M、LightSB-M）。在推进「免训练 batch 级保边缘噪声指派 MPNA」时，可引用此等价性作为理论依据：只要静态耦合是 EOT 解，动态桥的构造就有最优性保证。
2. **大偏差视角为误差分析提供概率论工具**：SB 的大偏差解释意味着有限样本下的边缘约束偏差可被理解为大偏差速率的有限 $N$ 修正。这为分析「不精确投影版 IMF」的误差传播（切入点 #1）提供了一个不同于泛化误差的视角：用大偏差速率函数刻画有限 batch 下 Sinkhorn 耦合偏离真实 EOT 的程度。
3. **参考过程的选择对应熵正则强度**：本文框架中参考过程的扩散系数直接决定 EOT 的正则化强度。这为切入点 #4（参考过程的学习与设计）提供了理论锚点：调参考过程等价于调熵正则系数，二者的稳定性关系可从 KL 扰动界角度分析。

## 8. 资源

代码：未公开（纯理论综述，无代码）。

相关论文 arXiv id 互链：
- Chen, Georgiou, Pavon, "Stochastic control liaisons: Richard Sinkhorn meets Gaspard Monge on a Schrödinger bridge" (2021), arXiv:2005.10963
- De Bortoli et al., "Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling" (NeurIPS 2021), arXiv:2106.01357
- Chen et al., "Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory" (ICLR 2022), arXiv:2110.11291
- Léonard, "From the Schrödinger problem to the Monge-Kantorovich problem" (J. Funct. Anal. 2012), arXiv:1011.2564
