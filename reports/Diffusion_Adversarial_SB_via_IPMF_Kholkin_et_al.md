# Diffusion & Adversarial Schrödinger Bridges via Iterative Proportional Markovian Fitting (IPMF)

> Kholkin et al. · ICLR 2026 (Poster) · [OpenReview](https://openreview.net/forum?id=38fGCBhFF5) · 证据级 [A] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：证明实践中的「双向交替 IMF」启发式等于 IMF 与 IPF 的组合（IPMF），在多种设定下收敛，并给出相似度-质量旋钮。
> ⚠ 未读全文，依据摘要（清单一句话贡献与 KB 笔记；无 abstract 文本）。

## 1. 问题
DSBM 的 Algorithm 1 交替做后向与前向 Markov 投影以抵消单向投影的边缘偏差（见 `Diffusion_Schr_dinger_Bridge_Matching_DSBM_Shi_et` §2），但这一双向交替在理论上并不是纯 IMF——纯 IMF 的 Markov 投影只有一个方向。KB 笔记概括本文问题：实践中的双向交替启发式到底在求什么、是否收敛。

## 2. 方法
据清单一句话贡献：把双向交替解释为 IMF（Markov/reciprocal 投影）与 IPF（半桥边缘投影）的组合，命名 Iterative Proportional Markovian Fitting（IPMF）；「Diffusion & Adversarial」指同时覆盖扩散式（DSBM 类）与对抗式（ASBM 类 D-IMF）实现。具体算法未读全文，未见。

## 3. 理论结果
据清单一句话贡献：IPMF 在多种设定下收敛。KB 笔记称其还给出 IMF/IPF 混合比例作为「相似度（与输入的对齐度）vs 生成质量」的显式旋钮。定理陈述、假设与是否有速率均未见。

## 4. 实验与数字
未读全文，无数字。

## 5. 在 OT×扩散地图中的位置
T03 第五代「理论收口」中面向算法结构的一篇：它为 DSBM（双向 Markov 投影）、α-DSBM（单网络双向在线）与 ASBM（D-IMF + GAN）共同使用的双向交替给出统一名称与收敛论证，把第一代 IPF 与第二代 IMF 重新拼回同一框架（对照 DSBM Table 1 对 IPF/IMF 的二分）。KB 把其混合比例视为方向一「对齐强度 vs 生成质量」的旋钮，也与 `2607.03626` 观察到的「参考过程改变端点对齐度」形成互补。证据级 [A]（已接收、正式版未核对）。

## 6. 局限与批评
未读全文，作者自述未见。可推断：(1) 若收敛结果仍假设精确投影，则与 IMF/α-IMF 理论一样不覆盖网络近似误差；(2) 「相似度-质量」旋钮的量化效果需看全文实验。

## 7. 对我们的启发
1. 方向一：如果 IPMF 的混合比例确能调节对齐度，它比改 σ（DSBM CelebA 实验中 FID 与 LPIPS 反向）多一个正交旋钮，值得在蒸馏/对齐管线中测试。
2. 优先获取全文与 OpenReview 评审，核对收敛设定（连续/离散时间、是否含 GAN 实现）。

## 8. 资源
- 代码：未见
- 相关报告：`Diffusion_Schr_dinger_Bridge_Matching_DSBM_Shi_et`、`Schr_dinger_Bridge_Flow_IMF_DSBM_De_Bortoli_et_al`、`Adversarial_SB_Matching_ASBM_Gushchin_et_al`、`Diffusion_Schr_dinger_Bridge_DSB_De_Bortoli_et_al`
