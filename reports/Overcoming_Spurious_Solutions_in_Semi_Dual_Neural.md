# Overcoming Spurious Solutions in Semi-Dual Neural Optimal Transport

> OTP · ICML 2025 · [PMLR](https://proceedings.mlr.press/v267/choi25a.html) · 证据级 [P] · 课题 T13 神经 OT 映射与无配对图像翻译
> **一句话**：给出 semi-dual max-min 恢复真 OT map 的充分条件；源分布平滑化 + 渐退火学 OT plan，可学随机映射（one-to-many 上色）

⚠ 未读全文，依据摘要

## 1. 问题

神经最优传输（neural OT）在无配对图像翻译中把任务形式化为 Monge/Kantorovich 问题：在所有把源分布推前到目标分布的映射中，选传输成本最小者。主流求解路线是 semi-dual max-min——位势 $f$ 与映射 $T$ 对抗优化。但这条路线存在一个核心缺陷：max-min 的解不一定是真 OT map，可能出现 fake/spurious solution。本文要回答的是：在什么条件下 semi-dual max-min 能恢复真 OT map；条件不满足时如何修正训练过程。

## 2. 方法

作者给出 semi-dual max-min 恢复真 OT map 的充分条件（原文具体条件未见，摘要未展开）。当条件不满足时，采用两条修正策略：对源分布做平滑化（source distribution smoothing），以及渐退火（annealing）来学 OT plan。该方法可学随机映射（stochastic map），即一对多翻译场景下的 plan 而非单值 map。

## 3. 理论结果

摘要报告：给出 semi-dual max-min 恢复真 OT map 的充分条件；在条件不满足时，源分布平滑化 + 渐退火可确保收敛（子序列意义）。具体定理编号、假设与结论的完整表述原文截断，未见。

## 4. 实验与数字

摘要未提供数据集、基线或数值结果。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

本文属于 semi-dual max-min 主线的「发现缺陷→修复」环节，直接继承 OTM（ICLR 2022）的 max-min 框架与 NOT（ICLR 2023）的 weak OT 随机 plan 路线，并回应 Kernel NOT（ICLR 2023）指出的 fake solutions 问题。与 DIOTM（ICLR 2025）的动态 OT 反哺静态 map 学习、UOTM 系列的 unbalanced 分支构成并行关系。其「源分布平滑化 + 退火」策略与 UOTM-SD（ICLR 2024）的 divergence 调度在思路上有交集，但针对的是 balanced 半对偶的 spurious solution 问题。

## 6. 局限与批评

- 收敛保证是子序列意义（摘要原文），不是全序列收敛。
- 摘要未给出实验验证的规模与基线，无法判断方法在高维图像上的实际表现。
- 充分条件的具体适用范围（哪些 cost、哪些分布类）在摘要中未展开，原文截断，未见。

## 7. 对我们的启发

- 若我们做 OT-aware 采样调度（切入点 #2），需警惕 semi-dual max-min 的 spurious solution：训练中应监控 map 是否真正推前源分布到目标，而非只看 duality gap 下降。
- 源分布平滑化 + 退火可作为我们 pipeline 中 OT map 学习阶段的稳定性增强手段，尤其在源分布有离散结构或低支撑时。
- 本文的随机 plan 学习能力直接对应 one-to-many 翻译（如上色），可作为我们保耦合蒸馏（切入点 #3）中教师 plan 的候选构造方式。

## 8. 资源

代码未公开（摘要未提及）。相关论文：OTM（ICLR 2022）、NOT（ICLR 2023）、Kernel NOT（ICLR 2023）、DIOTM（ICLR 2025）、UOTM-SD（ICLR 2024）。
