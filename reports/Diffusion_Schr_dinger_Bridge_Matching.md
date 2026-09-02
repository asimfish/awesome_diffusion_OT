# Diffusion Schrödinger Bridge Matching

> Yuyang Shi, Valentin De Bortoli, Andrew Campbell et al.（作者取自 DDBM/CDBM 参考文献）· NeurIPS 2023 · [arXiv](https://arxiv.org/abs/2303.16852) · 证据级 [P] · 课题 T14 扩散桥 / Schrödinger 桥的图像到图像翻译
> **一句话**：IMF + bridge matching 的通用 SB 求解器，是翻译类桥方法的算法底座；理论细节归 T03。

⚠ 未读全文，依据摘要。本条目摘要为空，以下依据课题清单的一句话贡献，并补充本课题已读全文（DDBM、CDBM）中对它的描述；§3/§4 不含任何数字。

## 1. 问题

SB 是两个任意分布间的熵正则 OT；经典求解器 IPF（DSB）迭代做半桥投影，误差随迭代累积、在高维上代价大（DDBM Sec. 1 与 Sec. 6 对 SB 类方法的评价：依赖昂贵迭代近似；I2SB Figure 2 给出 SB 求解器比 SGM 慢 6×、显存 3× 的量级，属对 FB-SDE 类的测量）。DSBM 要给出不累积误差、可与桥匹配训练兼容的 SB 求解流程（课题笔记 Sec. 3 的转述）。

## 2. 方法

课题清单/笔记转述：IMF 在"马尔可夫过程"与"给定端点耦合的桥混合"两类路径测度之间交替投影；每一步用 bridge matching（学参考桥的漂移）实现，最终收敛到 SB。DDBM Sec. 6 的描述："扩展 SB 与 IPF，Bridge-Matching 提出用 Iterative Markovian Fitting 求解 SB 问题"，并指出它"同样需要昂贵的迭代计算"。CDBM Sec. 5 的描述：bridge matching 是 flow matching 的随机对应物，假设可获得联合分布与端点间的插值/前向过程，再学另一 SDE/ODE 逼近其动力学；DDBM 已被证明等价于"保持初始联合分布的条件桥匹配"。

## 3. 理论结果

原文未读，未见。课题清单将理论细节归入 T03。

## 4. 实验与数字

原文未读，未见。课题清单未给数字。

## 5. 在 OT×扩散地图中的位置

DSBM 是 T14 的算法底座与谱系锚点：配对赛道的 I2SB/DDBM/LBM 都可视为"只做一次桥匹配、耦合由配对数据给定"的 IMF 单步；非配对赛道的 ASBM 是其离散时间对抗化版本，UNSB 用对抗序列替代 IMF 投影。DDBM 与 CDBM 都以"IMF 迭代代价高"作为自身 simulation-free 路线的动机，这一对比是综合报告"学耦合 vs 用配对"张力的来源。理论与 benchmark 归 T03；T13 的 neural OT 是它在无熵正则极限下的对照。

## 6. 局限与批评

作者承认的：原文未读，未见。

我读出来的（基于他人描述）：
- DDBM/CDBM 对它的批评集中在迭代代价与高维扩展性；这些是 2023 年的评价，后续 ASBM 等已部分缓解，引用时注明时间。
- 作为"底座"，其在自然图像 I2I 上的直接结果（分辨率、FID）在本课题内没有读到，不能据此比较。

## 7. 对我们的启发

1. 课题笔记开放问题 1 的 ground truth 工具：在低维合成任务上用 DSBM 求得近似 EOT 耦合，作为 UNSB/ASBM/LSB 耦合漂移的参照。
2. **#3 保耦合蒸馏**：IMF 的马尔可夫投影本身保持端点耦合，是设计"保耦合"损失的理论出发点；对照 CDBM 只保 PF-ODE 解映射的差别。
3. 读全文请转 T03 的报告，本卡不重复。

## 8. 资源

- 代码：课题笔记未列；详见 T03
- 互链：2309.16948（DDBM，以其迭代代价为动机）、2410.22637（CDBM，桥匹配统一视角）、ASBM_Adversarial_Schr_dinger_Bridge_Matching（离散时间 IMF）、UNSB_Unpaired_Neural_Schr_dinger_Bridge、I2SB_Image_to_Image_Schr_dinger_Bridge、2203.08382（DDIB）
