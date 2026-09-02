# 扩散 × 最优传输：2026 Q3 增量趋势扫描

> 扫描日期 2026-09-01 ｜ 覆盖 arXiv 提交日期 2026-08-01 至 2026-09-01 ｜ 方法：6 组 arXiv API 查询（`scripts/scan_arxiv_recent.py`）得 122 篇候选 → DeepSeek 按「相关度 0–3 / 归属课题 / 一句话贡献」分类（`scripts/classify_candidates.py`）→ 保留相关度 ≥2 的 62 篇（核心 24 篇），逐条人工复核标题与摘要 ｜ 上一次扫描截止 2026-08-14（8/25 小增量已收录 c-RF 2608.02487、BTM 2608.01692、PMOT 2608.05666）
> 机器可读清单：`new_papers_2026Q3.jsonl`（62 条）、`classified_all.jsonl`（122 条含相关度 0/1）

## 1. 一页结论

八月的 62 篇相关新工作里，**理论收口继续、耦合工程降本、桥模型细化**是三条主线，没有出现推翻综合报告四大张力的结果。对我们最重要的三篇：

1. **《Limit Points of Reflow with Minibatch OT》（2608.07042）**：证明 reflow 与固定 batch 的 minibatch OT 交替迭代的极限点是 N-循环单调耦合，在梯度场条件下收敛到 OT 映射——把 T09「直 ≠ 最优」的反例线（Hertrich NeurIPS 2025）与 c-RF（2608.02487）的修复线在 minibatch 设置下接了起来。**Top-10 #4 的转向建议 (b)「c-RF 投影的实用化」有了第二个竞争者**。
2. **《One-Sided Quantile Coupling for Flow Matching》（2608.00978）**：用单侧分位数耦合以 O(n) 代价替代 minibatch OT 并保持直线流——耦合工程线从「更大 batch 的 Sinkhorn」（Apple 2506.05526）转向「更便宜的结构化耦合」。对 MPNA（推理期 batch 置换指派）是直接相关的竞争/互补工作：它在训练侧、我们在推理侧，但都在回答「不解完整 OT 能拿到多少耦合收益」。
3. **《A Lagrangian View of Flow Matching》（2609.00198）**：把 FM 轨迹曲率归因到去噪器的雅可比——给 Top-10 #2「OT-aware 调度」提供了一个比 KLUB 更接近第一性原理的曲率来源刻画。

## 2. 顶会信号（均已核实 URL）

| 会议 | 状态（2026-09-01） | 出处 |
|---|---|---|
| ECCV 2026 | 9 月 8–12 日 Malmö；Springer 论文集 LNCS 17001–17083 已上线（Part XIII 页面可见）；8 月 arXiv 已见 3 篇 ECCV 2026 的 OT×扩散工作：ReFP-AD 2608.01793、EmbodiedVAE 2608.02990、DDB 离散扩散桥 2608.29997 | https://eccv.ecva.net/Conferences/2026 · https://link.springer.com/book/9783032372703 |
| NeurIPS 2026 | 作者通知 **9 月 24 日**（AoE）；主会 12 月 6–12 日悉尼；工作坊决定 9 月 29 日 | https://neurips.cc/Conferences/2026/Dates |
| ICML 2026 → PMLR | FlashSinkhorn（Oral，7 月 9 日）仍为 [A]：proceedings.mlr.press 最新卷 v303（2026-01），ICML 2026 卷未发布；repo 停在 v0.3.3（2026-04），**多 GPU / 非欧 cost 空位仍在** | https://icml.cc/virtual/2026/oral/71180 · https://github.com/ot-triton-lab/flash-sinkhorn |
| MICCAI 2026 / SIGGRAPH Asia 2026 / ISMIR 2026 | 8 月 arXiv 已见接收稿：KANResDiff（局部残差 SB 分割，MICCAI）、SketchFlow（OT-CFM 矢量草图，SIGGRAPH Asia）、音频带宽扩展几何（ISMIR） | 见 §4 清单 |
| ICLR 2027 | 投稿窗口按惯例 9 月底–10 月初；本季度 [R] 预印本中相当比例应为 ICLR 2027 在审稿 | — |

## 3. 趋势判断（先说结论，再给证据）

**T1 · 「reflow / 拉直」的理论边界在一个月内被两次推进，且方向一致：加投影约束就能到 OT。** c-RF（2608.02487）证明速度场投影到梯度类后 rectification 恒收敛到 OT 耦合并给出 minimax 率；2608.07042 证明 reflow×minibatch-OT 的极限是 N-循环单调耦合、加梯度场条件即到 OT 映射。两篇都把「直 vs 最优」的张力（综合报告 §2.8 第 3 条）从叙事之争变成了「选哪个投影算子、付多少计算」的工程问题。**空位收窄**：Top-10 #4 原始表述已被解决 ≥ 70%，剩余的是高斯混合/流形数据的非渐近刻画与大规模投影算子实现。

**T2 · 耦合工程从「算得更大」转向「算得更巧」。** 2506.05526（Apple，n≈10⁶ Sinkhorn）证明大 batch 才见收益之后，八月出现两条降本路线：QC-FM（2608.00978）的单侧分位数耦合 O(n)；Gromov-Monge FM（2608.26961）用 GW 型松弛构造图生成中的等变耦合。加上 5 月的半离散耦合（2509.25519）与 LOOM-CFM，「不解完整 OT、只取结构化近似」正在成为主流。**对 MPNA 的含义**：推理期 batch 置换是同一思路在采样侧的镜像，写作时必须把 QC-FM 列为最近邻并说明训练侧/推理侧的分工。

**T3 · Schrödinger 桥理论进入「参考过程与端点约束」的精细化阶段。** PRISM（2608.06893）问 SB 的参考过程该怎么设计；SDDBMs（2608.08594）把桥的 Dirac 端点松弛为非退化高斯并统一多种桥模型；《On Bridging Mixture Distributions》（2608.13383）给高斯混合桥的 Wasserstein 连续性界；《Noising-Denoising by Large Temperature SB》（2608.25094）在过程层面把去噪扩散与高温 SB 接起来；时间序列 SB（2608.13968）给出完整分布收敛分析。五代求解器史（综合报告 §2.4）之后，SB 线的论文题目从「怎么解」变成「解什么」。

**T4 · 桥模型在应用侧全面铺开，医学与图像复原是主战场。** 乳腺 DCE-MRI 潜在桥（2608.10000）、MRI 超分桥 10 步（2608.08819）、质子剂量预测 DoseBridge（2608.10173，直接以剂量为目标——与 SynthRAD2025 的教训一致）、法医组织病理 SB benchmark（2608.21813）、EditBridge 4K 编辑（2608.18063）、ReBridge-Flow 后验桥重耦合（2609.00811）、夜间增强循环 SB（2608.29043）、离散扩散桥（2608.29997，ECCV）。**Top-10 #7 的判断被强化**：以剂量学为目标的桥模型已经有人在做（DoseBridge），SynthRAD 榜上的 SB 空位窗口在收窄，应在 MICCAI 2027 周期内动手。

**T5 · Wasserstein 梯度流成为「微调」工具，而不只是「训练」工具。** 奖励引导微调一步生成器（2608.29647，免奖励梯度）、CVaR 惩罚的极端事件微调（2608.11544）、Langevin 正则化 SVGD 的定量收敛（2608.28827）。这与综合报告 §2.7「reward 微调 = 熵正则 OT」的判断一致，并把 T05 从理论课题推向了对齐/微调的实用课题。

**T6 · 一步/少步生成的理论叙事在向「自治流 / 拉格朗日视角」迁移。** Beckmann 传输模型（2608.01692）用时间无关速度场做一步映射；《A Lagrangian View of FM》（2609.00198）把曲率归因到去噪器雅可比；MeanFlow 的领域扩展（SE(3) 抓取 2608.03295、李群约束 2608.26076）说明平均速度范式已经成为默认少步方案。**对 Top-10 #2 的含义**：调度理论应以雅可比诱导曲率为变量，而不是 KLUB。

**T7 · 评测与蒸馏开始借用 OT 的语言。** 《The Distributional View of Knowledge Distillation》（2608.15215）把蒸馏写成熵正则 Wasserstein 重心 / Sinkhorn 散度目标；Fused GW 被用作 3D 场景图的结构评测（2608.28733）。这是 Top-10 #10「OT 系评测指标」空位正在被外围填充的信号——FID 替代品的窗口不会一直开着。

## 4. Top-10 切入点状态（相对 8/25 版）

| # | 切入点 | Q3 状态 | 证据 |
|---|---|---|---|
| 1 | 免训练 batch 级保边缘噪声指派（MPNA） | **空位仍在**，但最近邻变多：QC-FM（训练侧结构化耦合）、免训练隐藏状态细化（2608.29160，推理期免训练但不碰耦合） | 2608.00978, 2608.29160 |
| 2 | OT-aware 采样调度 | 空位仍在；曲率来源有了新刻画（去噪器雅可比） | 2609.00198 |
| 3 | 保耦合的桥蒸馏 + 漂移度量 | 空位仍在；T14 深读证实 DDBM/DBIM/CDBM 均无终端耦合保持陈述 | reports/2309.16948, 2405.15885, 2410.22637 |
| 4 | reflow 正定理 | **进一步被占**：c-RF + reflow×minibatch-OT 极限点定理；剩余 GMM/流形非渐近刻画（2608.13383 给了高斯混合桥的连续性界，可借工具） | 2608.02487, 2608.07042, 2608.13383 |
| 5 | OT-CFM 端到端统计率 | 空位仍在；函数空间 FM 的离散化一致性（2608.04531）是可借的技术路线 | 2608.04531 |
| 6 | PF-ODE 次优度量化 | 空位仍在 | — |
| 7 | 医学 SB 刷 SynthRAD（以剂量学为目标） | **窗口收窄**：DoseBridge 已以剂量为目标；SB 端仍无人参赛 | 2608.10173, 2608.10000, 2608.08819 |
| 8 | FGW 语义对应进采样 | 空位仍在；Gromov-Monge FM 在图生成侧用了 GW 松弛，可借 | 2608.26961 |
| 9 | 视频侧时空分解 reflow / 帧间 Sinkhorn 耦合 | 空位仍在；EmbodiedVAE 用 OT 一致性模块做视频潜变量时序一致（ECCV 2026） | 2608.02990 |
| 10 | OT 系评测指标 | **外围开始填充**（蒸馏的 Sinkhorn 散度目标、FGW 结构评测） | 2608.15215, 2608.28733 |

## 5. 新论文清单（相关度 3 = 核心，2 = 相关；按日期）

### 核心（24 篇）

| arXiv | 日期 | 课题 | 证据 | 题目 | 一句话 |
|---|---|---|---|---|---|
| 2608.00978 | 08-02 | T08 | R | One-Sided Quantile Coupling for Flow Matching | 单侧分位数耦合 O(n) 替代 minibatch OT，保持直线流 |
| 2608.02487 | 08-03 | T09 | R | Computational and Statistical Guarantees of the c-Rectified Flow | c-RF 迭代恒收敛到 OT 耦合，给计算与统计保证 |
| 2608.01692 | 08-03 | T01 | R | Beckmann Transport Models | 自治流实现分布间精确映射与一步生成 |
| 2608.05557 | 08-06 | T20 | R | Hierarchical Flow Matching for 3D Point Cloud Generation | 双层 OT 路径 FM，15 步生成点云 |
| 2608.07042 | 08-07 | T09 | R | Limit Points of Reflow with Minibatch Optimal Transport | reflow×minibatch-OT 极限 = N-循环单调耦合；梯度场条件下到 OT 映射 |
| 2608.06784 | 08-07 | T09 | R | UniCycleFlow | 共享整流流场做双向无配对翻译，单步 Euler |
| 2608.10000 | 08-07 | T14 | R | Pre- to Post-Contrast Synthesis of Breast DCE-MRI via Latent Bridge Matching | 潜在桥匹配合成对比增强 MRI |
| 2608.06893 | 08-07 | T03 | R | PRISM: Principled Reference Identification for SB Models | SB 参考过程的最优设计理论 |
| 2608.08594 | 08-09 | T03 | R | SDDBMs: Soft Denoising Diffusion Bridge Models | 非退化高斯终端边际松弛端点约束，统一多种桥 |
| 2608.11617 | 08-12 | T14 | A:MICCAI 2026 | KANResDiff | 局部残差 SB 做医学模糊分割 |
| 2608.11544 | 08-12 | T05 | R | Fine-Tuning Generative Models for Extreme Events via CVaR-Penalized WGF | CVaR 惩罚 WGF 微调捕获重尾 |
| 2608.12715 | 08-13 | T23 | R | HybridSB-MoE | 双域 SB 语音增强 + Wasserstein 采样误差界 |
| 2608.13383 | 08-13 | T03 | R | On Bridging Mixture Distributions | 高斯混合 SB 的 Wasserstein 连续性界 |
| 2608.13968 | 08-14 | T03 | R | Nonparametric SB Time Series Generator | SB 时间序列生成器的分布收敛分析 |
| 2608.15215 | 08-15 | T04 | R | The Distributional View of Knowledge Distillation | 蒸馏 = 熵正则 Wasserstein 重心 / Sinkhorn 散度目标 |
| 2608.21659 | 08-21 | T08 | A:SIGGRAPH Asia 2026 | SketchFlow | OT-CFM 在 CLIP 空间做零样本矢量草图 |
| 2608.21070 | 08-21 | T24 | R | TracingFlow | 二阶动力学 FM 做单细胞轨迹推断 |
| 2608.25094 | 08-25 | T03 | R | Noising-Denoising by Large Temperature Schrödinger Bridges | 去噪扩散 = 高温动态 SB 的过程级联系 |
| 2608.25838 | 08-26 | T28 | R | Hard-Constrained Sampling on Embedded Riemannian Manifolds via Adjoint SB | 伴随 SB 在嵌入流形上硬约束采样 |
| 2608.26961 | 08-27 | T26 | R | Gromov-Monge Flow Matching for Equivariant Graph Generation | GW 型松弛构造等变耦合 |
| 2608.27885 | 08-28 | T03 | R | There and Back Again: Bidirectional Diffusion Bridges | 文本–图像双向扩散桥统一生成与反演 |
| 2608.29647 | 08-30 | T05 | R | Reward-guided Fine-Tuning of One-Step Generative Models via WGF | WGF 做一步模型的奖励微调，免奖励梯度 |
| 2609.00198 | 08-31 | T09 | R | A Lagrangian View of Flow Matching | 去噪器雅可比是 FM 轨迹曲率主因 |
| 2609.00811 | 09-01 | T14 | R | ReBridge-Flow | 修复 FM 图像复原中的后验桥失配 |

### 相关（38 篇）

见 `new_papers_2026Q3.jsonl` 中 relevance=2 的条目；README §G 已自动列出全部 62 篇。要点：ECCV 2026 三篇（ReFP-AD 2608.01793、EmbodiedVAE 2608.02990、DDB 2608.29997）；医学四篇（全容积多任务翻译 2608.08135、MRI 超分桥 2608.08819、DoseBridge 2608.10173、法医病理 SB 2608.21813）；理论两篇（函数 FM 离散化一致性 2608.04531、GW 梯度流收敛 2608.19198）；语音五篇（GROW 2608.03215 等）。

## 6. 方法与限制

- 查询词：optimal transport / Wasserstein / Sinkhorn / Schrödinger bridge × diffusion / flow matching / rectified flow / consistency / generative；bridge matching / diffusion bridge；initial noise / noise selection / inference-time scaling；Gromov-Wasserstein / unbalanced / semi-discrete / Wasserstein gradient flow × generative；mean flow / flow map / one-step × transport / coupling。每组取前 100 条，按提交日期降序。
- 覆盖盲区：只搜 arXiv 摘要，未覆盖 OpenReview 在审稿与会议官网；标题/摘要不含关键词的工作会漏掉（例如只在正文用 OT 做分析的论文）；相关度分级由 DeepSeek 给出、经人工复核标题与摘要，但未读全文。
- 证据级：本清单全部为 [R] 预印本，除 comment 明确写明接收的 7 篇（ECCV 2026 ×3、MICCAI 2026、SIGGRAPH Asia 2026、ISMIR 2026、Sashimi 2026）标 [A]。
- 下一次触发点：NeurIPS 2026 放榜（9/24）后重跑本脚本并核对接收名单；ICML 2026 PMLR 卷上线后把 FlashSinkhorn 等 [A] 升 [P]。
