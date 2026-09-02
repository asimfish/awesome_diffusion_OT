# Score-Based Generative Modeling through SDEs

> Song et al. · ICLR (Oral) 2021 · [OpenReview](https://openreview.net/forum?id=PxTIG12RRHS) · 证据级 [P] · 课题 T02 扩散模型与 OT 的理论联系
> **一句话**：用 SDE 统一扩散模型，提出逆向 SDE 与 PF-ODE 两种采样路径。

⚠ 未读全文，依据摘要

## 1. 问题

此前基于分数的生成模型（score-based generative models）主要依赖离散时间步的噪声扰动与去噪过程，缺乏统一的连续时间描述。不同模型（如 NCSN、DDPM）各自使用不同的噪声调度与采样算法，难以在统一框架下比较和扩展。作者提出用随机微分方程（SDE）来描述噪声扰动过程，将此前方法视为其离散化特例。

## 2. 方法

核心思想是把数据分布到噪声分布的演化写成一个前向 SDE，然后通过时间反转得到逆向 SDE 用于采样。摘要中给出的关键对象包括：

- 前向 SDE：$dx = f(x,t)dt + g(t)dw$，将数据逐渐扰动为噪声；
- 逆向 SDE：通过估计 score function $\nabla_x \log p_t(x)$ 来反转前向过程；
- probability flow ODE（PF-ODE）：与逆向 SDE 具有相同边缘分布的确定性常微分方程。

作者报告该框架统一了此前基于分数的生成建模与扩散概率模型，并允许新的采样方法与新的模型设计。

## 3. 理论结果

摘要未给出具体定理或证明。仅报告该框架提供了统一的连续时间表述，并指出逆向 SDE 与 PF-ODE 在边缘分布层面等价。具体假设与结论需读全文确认。

## 4. 实验与数字

摘要未给出具体实验数字、数据集或基线结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文是 T02 课题的起点性工作：它定义的 PF-ODE 给出了确定性的 encoder map（数据→噪声）及其逆（噪声→数据），使「扩散 encoder 是否为 OT map」成为可精确表述的问题。后续 Khrulkov et al.（ICLR 2023）的猜想与 Lavenant–Santambrogio（2022）的反例都直接以 PF-ODE 为对象。本文本身不讨论最优传输，但为整个辩论提供了映射对象与连续时间语言。

## 6. 局限与批评

- 摘要未报告理论保证（如采样误差界、score 估计误差传播），需读全文确认。
- 摘要未给出实验数字，无法评估其经验主张的强度。
- PF-ODE 的确定性映射性质（是否 OT、是否 Lipschitz 等）在本文中未被讨论，后续工作表明该映射一般不是全局 OT map。

## 7. 对我们的启发

- 本文的 PF-ODE 是「扩散 encoder/flow map」的标准定义来源，后续若做 OT-aware 采样调度或保耦合蒸馏，应以 PF-ODE 轨迹为基准对象。
- 逆向 SDE 与 PF-ODE 的边缘分布等价性提示：在讨论映射层 OT 性质时，需区分随机映射（逆向 SDE）与确定性映射（PF-ODE），两者几何性质不同。
- 若做少步采样误差认证，PF-ODE 的离散化误差是核心对象，本文框架提供了连续时间参照。

## 8. 资源

代码未公开（摘要未提及）。相关论文：Khrulkov et al.（ICLR 2023）、Lavenant–Santambrogio（2022）、Song et al.（NeurIPS 2020, DDPM 前身 NCSN 系列）。
