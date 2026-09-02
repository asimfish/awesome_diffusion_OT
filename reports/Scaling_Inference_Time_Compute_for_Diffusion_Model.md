# Scaling Inference Time Compute for Diffusion Models (Inference-Time Scaling beyond Denoising Steps)

> Nanye Ma, Shangyuan Tong, Haolin Jia et al. · CVPR 2025 · [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.html) · 证据级 [P] · 课题 T12 推理阶段的 OT 对齐与噪声-样本耦合
> **一句话**：把「找好噪声」形式化为 verifier × 搜索算法的设计空间（random / zero-order / search-over-paths），确立噪声搜索为扩散 test-time scaling 的第二轴。

> ⚠ 未读全文，依据摘要。清单中 `text` 为空且 `abstract` 字段为空，本卡仅依据 `contribution_zh` 与课题背景笔记（`source/kb/t12_inference_time_ot_alignment.md`）撰写；**不含任何实验数字**，所有数字请以原文为准。

## 1. 问题

扩散模型的推理期算力此前只沿一个轴扩展：增加去噪步数，而步数增加到一定程度后收益饱和。本文提出第二轴：把算力花在**搜索更好的采样噪声**上。前提是本课题反复确认的事实——初始噪声（以及 SDE 采样中途注入的噪声）决定了大量生成质量，因而「哪一个噪声」是一个可以用算力去优化的自由度。

## 2. 方法

据贡献摘要，本文把噪声搜索拆成两个正交设计维度：
- **verifier**：对候选样本（或中间状态）打分的反馈函数，用来判断哪条采样路径更好；
- **搜索算法**：如何用 verifier 的反馈在噪声空间里找候选。贡献摘要列出三类——**random search**（独立采 $k$ 个噪声、取 verifier 最高者，即 top-1-of-$k$）、**zero-order search**（在当前最优噪声邻域内做无梯度的局部搜索）、**search over paths**（不只搜初始噪声，而是在采样轨迹中间时刻重注噪声并继续搜索，把选择插入轨迹中段）。

具体 verifier 的选择、各算法的超参与预算分配方式：未读全文。

## 3. 理论结果

未读全文。据贡献摘要，本文定位为经验性的设计空间研究，未见理论结果的记载。

## 4. 实验与数字

未读全文；清单 abstract 为空，无可引用的数字。据贡献摘要，本文的结论性主张是：在固定去噪步数下增加搜索算力可持续改善样本质量，且不同 verifier–算法组合对不同任务/指标的适配性不同。数字请以 CVF 原文为准。

## 5. 在 OT×扩散地图中的位置

- T12 环节 **② ODE 初值选择（离散耦合）**的搜索线代表，也是唯一明确触及环节 **⑤ 轨迹中段**的工作（search over paths 在中间时刻重注噪声再搜索）。
- 与检索线 NoiseQuery（2412.05101）同为离散选择：搜索在线生成候选并打分，检索把候选与打分离线化；与优化线 ReNO（2406.04312）共享 verifier 但用零阶而非梯度；与学习式映射 NPNet（2411.09502）、NoiseRefine（2412.03895）互补。
- 对张力「改耦合 vs 保边缘」：random search 的 top-1-of-$k$ 使有效初始分布变成次序统计量的混合，$\pi_k\notin\Pi(\mu,\nu)$；kb 笔记 §5.1 指出「目前无人刻画」这一漂移。本文是该漂移最直接的载体，也是 verifier hacking 讨论的来源。

## 6. 局限与批评

作者承认的：未读全文，无法转述。

我读出的（基于框架本身）：
1. 所有三类搜索都是实例级选择，verifier 偏好的「通用好噪声」会被所有提示共同选中，人口级分布漂移与同提示同质化随 $k$ 增长；据贡献摘要与 kb 笔记，本文未以边缘漂移/多样性作为主评测维度。
2. 搜索预算按每样本计（$k$ 次生成 + $k$ 次 verifier），与蒸馏/少步方法比较时需统一 NFE 口径。
3. verifier 与评测指标是否同源（如用 CLIP 做 verifier 又用 CLIP 系指标评测）：未读全文，无法判断。

## 7. 对我们的启发

1. **MPNA（#1）的 B2 基线族直接取自本文的 random search**：同一 verifier 下 top-1-of-$k$（$k\in\{2,4,8\}$）是 PROPOSAL §6.3 基线 2 的定义来源；比较必须同 verifier、同预算（PROPOSAL §8 红线 4）。PROPOSAL §3 Prop. 2(ii) 的闭式（$\mathbb E\langle a,z^\star\rangle=m_k/\sqrt{1+\lambda^2}$、范数漂移 $v_k+m_k^2-1$）给出了 random search 漂移的理论预测，可作为本文框架缺失的「漂移轴」补充。
2. **search over paths 指向 MPNA 的环节 ⑤ 扩展**：若在中间时刻 $t$ 对 batch 内 $(x_t^{(i)}, c_i)$ 做置换重排而非逐样本重注噪声搜索，可把保边缘性质推广到轨迹中段（PROPOSAL §7 的转向选项之一）。
3. **verifier 的 g–h 分解可用作本文设计空间的第三维**：对任何 verifier 计算列均值 $g$ 与残差 $h$，$\|g\|/\|h\|$ 预测该 verifier 在 random search 下的漂移风险，从而为「选哪个 verifier」提供与任务无关的先验判据。

## 8. 资源

- 项目页：[inference-scale-diffusion.github.io](https://inference-scale-diffusion.github.io/)（kb 笔记 §6）；代码是否公开：未核验。
- 相关报告：2412.05101（NoiseQuery，离线检索版）、2406.04312（ReNO，同 verifier 的梯度版）、2411.09502（NPNet）、2412.03895（NoiseRefine）、2404.04650（InitNO，含拒绝采样成分）。
