# T27 多边际 OT 与 Wasserstein 重心的生成应用

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题覆盖「扩散×OT」全景中从两边缘走向 k 边缘的分支：多边际 OT（MMOT）的理论与复杂度、Wasserstein barycenter 的计算前沿，以及二者进入生成建模的三条通道——多时间边缘的 multimarginal flow matching / SB matching、barycenter 驱动的多风格融合与多条件插值、模型融合（model merging）的 OT 视角。两边缘 Schrödinger bridge 本体见 T03，Gromov-Wasserstein 见 T26。

## 1. 核心问题与背景

两边缘 OT 回答"如何把一个分布搬到另一个分布"；MMOT 与 barycenter 回答"如何让 k 个分布**共同**耦合、如何对 k 个分布取**几何平均**"。这在生成建模里对应三类真实需求：(i) 时序快照生成——单细胞、气象、视频只提供多个时间点的静态边缘，生成模型必须同时贴合所有中间边缘而非逐段两两插值；(ii) 多风格/多条件融合——把多个概念、风格或条件分布融成一个新分布，线性插值会产生模糊与风格支配问题，Wasserstein 几何给出有原则的替代；(iii) 模型融合——多个独立训练网络的逐层平均可解释为对神经元分布求 barycenter。难点在于：MMOT 是 \(n^k\) 变量的指数规模 LP，一般代价下近似求解 NP-hard，barycenter 在维度上有"维数灾难"式的硬度下界；因此 2024–2026 的主线是**利用结构**（树代价、动力学形式、势函数软约束、神经对偶）把多边缘问题变成可扩展的生成式算法。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Multimarginal Flow Matching with Optimal Transport Potentials (OTP-FM) | 2026·ICML | [A] | 把逐段 CFM 重写为带硬约束的动态 OT，再把中间边缘约束松弛为动态 OT 作用量中的**势能项**，得到 simulation-free 的多边缘 FM，并给出势强度—Wasserstein 偏差界；单细胞/海洋/气象 SOTA | [arXiv](https://arxiv.org/abs/2606.05327) · [ICML页](https://icml.cc/virtual/2026/poster/62108) |
| ⭐ Momentum Multi-Marginal Schrödinger Bridge Matching (3MSBM) | 2025·NeurIPS | [P] | 相空间提升+多点条件化随机桥，学习满足多个位置约束的测度值样条；matching 迭代中保持中间边缘不变，解决两两插值丢失长程时序依赖的问题 | [官方](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7c3875b86bd2b0639ab1e858c678af40-Abstract-Conference.html) |
| ⭐ Modeling Complex System Dynamics with Flow Matching Across Time and Conditions (MMFM) | 2025·ICLR Spotlight | [P] | 首批"多边缘 FM"：三次样条跨时间点构造条件路径 + classifier-free guidance 跨条件共享动力学，能对缺失时间点×条件组合做插补 | [OpenReview](https://openreview.net/forum?id=hwnObmOTrV) |
| Multi-Marginal Temporal Schrödinger Bridge Matching (MMtSBM) | 2026·ICML | [A] | 把 DSBM 的 Iterative Markovian Fitting 以 factorized 方式推广到多边缘，在 100 维转录组与高维图像上首次恢复多时间耦合与动力学 | [arXiv](https://arxiv.org/abs/2510.01894) |
| ⭐ Estimating Barycenters of Distributions with Neural OT | 2024·ICML | [P] | 基于 Neural OT 对偶的双层对抗目标求连续 barycenter，支持一般代价（对比既有三层 min-max 且限于二次代价），含误差界；实验含 StyleGAN 潜空间 | [PMLR](https://proceedings.mlr.press/v235/kolesov24a.html) |
| Energy-Guided Continuous Entropic Barycenter Estimation for General Costs | 2024·NeurIPS Spotlight | [P] | 弱 OT 对偶+EBM 学连续熵正则 barycenter，免 min-max，带质量界；直接在预训练生成模型的图像流形上学 barycenter | [官方](https://papers.nips.cc/paper_files/paper/2024/hash/c2bd9242609219deb380f161682f4568-Abstract-Conference.html) |
| ⭐ Sobolev Gradient Ascent for Optimal Transport: Barycenter Optimization and Convergence Analysis | 2026·ICLR | [A] | 精确（非熵正则）barycenter 的无约束凹对偶 + \(\dot H^1\) Sobolev 几何梯度上升；证明可去掉昂贵的 c-concave 投影仍有全局 \(O(T^{-1/2})\) 收敛率，每步 \(O(mn\log n)\) | [OpenReview](https://openreview.net/forum?id=IjL1xEoxXi) |
| Wasserstein Gradient Flows for Scalable and Regularized Barycenter Computation | 2026·UAI (PMLR 337) | [P] | 把 barycenter 问题重写为 Wasserstein 空间中的梯度流：mini-batch OT 实现可扩展、支持模块化正则泛函（内能/势能/交互能）与监督化 ground cost，PL 条件下收敛保证；注：任务书中记为"NeurIPS 2025"，官方归属实为 UAI 2026 | [PMLR](https://proceedings.mlr.press/v337/fernandes-montesuma26a.html) |
| Tree-Based Diffusion Schrödinger Bridge (TreeDSB) | 2023·NeurIPS Spotlight | [P] | 树结构代价的熵正则 MMOT 的动态连续版本（多边缘 Sinkhorn 的 DSB 对应物）；星形树即 barycenter，可在高维做图像插值与贝叶斯融合——「用扩散模型算 barycenter」的奠基工作 | [官方](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ad08767706825033b99122332293033d-Abstract.html) |
| Finding the Center of a Wasserstein Ball | 2025·ICML | [P] | Wasserstein ball 中心=一种 min-max 鲁棒聚合，与 barycenter 互补的"最坏情况平均"视角 | [PMLR](https://proceedings.mlr.press/v267/wang25be.html) |
| The Procrustes-Wasserstein Barycenter Problem | 2025·ICML | [P] | barycenter 与正交/刚体对齐联合优化，解决输入分布姿态不对齐时平均失真的问题 | [PMLR](https://proceedings.mlr.press/v267/adamo25a.html) |
| Wukong's 72 Transformations: High-fidelity Textured 3D Morphing via Flow Models | 2025·arXiv | [R] | 在预训练 3D flow transformer 的**条件 token 空间**解 free-support barycenter 得到插值条件，实现 training-free 高保真 3D 形变+纹理渐变——barycenter×预训练流模型的代表性应用 | [arXiv](https://arxiv.org/abs/2511.22425) |
| Model Fusion via Optimal Transport (OTFusion) | 2020·NeurIPS | [P] | 逐层用 OT 对齐神经元再平均，显式解释为逐层 Wasserstein barycenter；一次性(one-shot)、无需训练数据的模型融合奠基 | [arXiv](https://arxiv.org/abs/1910.05653) · [官方](https://proceedings.neurips.cc/paper/2020/hash/fb2697869f56484404c8ceee2985b01d-Abstract.html) |
| Partial Fusion of Neural Networks via Partial Optimal Transport | 2026·ICML | [A] | 用 partial OT 处理不同模型间神经元只有部分对应的情形，放松 OTFusion 的完全匹配假设 | [OpenReview](https://openreview.net/forum?id=lvRLG6C0zZ) |
| A Dynamical Formulation of Multi-Marginal Optimal Transport | 2025·arXiv | [R] | 首个一般 (semi-)convex 代价的 MMOT 原-对偶**动力学**形式（耦合流而非边缘流），凸优化可解并给出 quasi-Monge 解；为"多边缘 Benamou-Brenier"补上缺失的一块 | [arXiv](https://arxiv.org/abs/2509.22494) |

理论底座（表外必引）：MMOT 综述 Pass 2015 [B]（[ESAIM:COCV](https://doi.org/10.1051/cocv/2014010)）；barycenter 定义与 \(W_2\) 几何 Agueh & Carlier 2011 [P]（[SIAM](https://doi.org/10.1137/100805741)）；复杂度双壁垒——固定维数多项式可解 Altschuler & Boix-Adserà, JMLR 2021 [P]（[JMLR](https://www.jmlr.org/papers/v22/20-588.html)）与维度上 NP-hard, SIMODS 2022 [P]（[arXiv](https://arxiv.org/abs/2101.01100)）；免费支撑近似算法 von Lindheim, COAP 2023 [P]（[DOI](https://doi.org/10.1007/s10589-023-00458-3)，N−1 次两边缘 OT 换 N 倍相对误差界）；熵正则 barycenter 去偏 Janati et al., ICML 2020 [P]（[PMLR](https://proceedings.mlr.press/v119/janati20a.html)）；数值 MMOT 教材 Nenna 博士论文 [B]（本地 `ot_variants_survey/book/Multi_OT_book.pdf`）。

## 3. 方法演进脉络

**MMOT 理论线：** Agueh–Carlier (2011) 确立 barycenter 即一个特殊 MMOT；Pass (2015) 系统化 MMOT 的 Monge 解结构。复杂度在 2021–2022 被 Altschuler–Boix-Adserà 钉死：固定维数多项式可解（JMLR 2021），但对维度的指数依赖不可去除（NP-hard, SIMODS 2022）——这解释了此后所有工作都在"利用结构"：树代价（TreeDSB）、层次结构（AAAI 2024 [Hierarchical MMOT for Network Alignment](https://ojs.aaai.org/index.php/AAAI/article/download/29605/31022) [P]）、稀疏近似（von Lindheim 2023）、以及 2025 年 Pass–Shenfeld 的动力学化（DMMOT [R]）——后者把静态高维耦合变成"耦合的流"，为多边缘问题打开 Benamou–Brenier 式的凸优化与数值管线。

**Barycenter 计算线：** 从 fixed-support 熵正则（IBP/Sinkhorn，Janati 2020 去偏）走向三支：(a) **精确解**——SGA (ICLR 2026 [A]) 用无约束凹对偶+Sobolev 几何免去 c-concave 投影，配合 FRBary（[A Unified Approach for Computing Wasserstein Barycenters](https://arxiv.org/abs/2605.11270) [R]）构成 exact barycenter 前沿；(b) **神经/连续解**——Kolesov 等 ICML 2024 双层对抗 Neural OT、NeurIPS 2024 能量引导 EBM 版（可在预训练生成流形上学 barycenter），摆脱离散支撑；(c) **梯度流解**——UAI 2026 WGF 正则化 barycenter 把问题变成测度空间梯度流，mini-batch OT + 可插拔正则泛函，兼容监督信息。鲁棒变体（Wasserstein ball center、Procrustes-WB，均 ICML 2025 [P]）补齐"对齐+平均"与"最坏情况平均"。

**生成应用线：** 三条通道汇流。其一，**多时间边缘生成**：TreeDSB (2023) 证明扩散式算法能解树形 MMOT/barycenter；MMFM (ICLR 2025) 用样条+CFG 做多时间×多条件 FM 但平滑是"规定式"的；3MSBM (NeurIPS 2025) 以相空间动量桥给出有控制论依据的平滑；OTP-FM (ICML 2026) 最终把两者统一进动态 OT 作用量的势函数框架——硬约束是奇异势极限，势强度可调且有 Wasserstein 偏差界；MMtSBM (ICML 2026) 则把 IMF 因子化推到多边缘高维图像。其二，**barycenter 融合/插值**：从 Gaussian OT 闭式风格混合（[Wasserstein Style Transfer, AISTATS 2020](https://proceedings.mlr.press/v108/mroueh20a.html) [P]）到潜空间 barycenter（NOTBarycenters 的 StyleGAN 实验、能量引导版的图像流形 barycenter），再到 Wukong (2025 [R]) 在预训练 flow transformer 条件 token 上 training-free 解 free-support barycenter 做 3D 形变；[Wasserstein Flow Matching (ICML 2025)](https://proceedings.mlr.press/v267/haviv25a.html) [P] 更进一步把"分布族上的生成"直接建立在 Wasserstein 测地线上。其三，**模型融合**：OTFusion (NeurIPS 2020) 的逐层 barycenter 解释 → Transformer 化（[ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/7e0af0d1bc0ec2a90fc294be2e00447e-Paper-Conference.pdf) [P]，软对齐+异构压缩）→ 与 GW-barycenter 融合 RNN 的变体（[arXiv 2210.06671](https://arxiv.org/abs/2210.06671) [R]）→ 安全应用（[AAAI 2025 融合+后门缓解](https://ojs.aaai.org/index.php/AAAI/article/view/34828) [P]）→ 部分匹配（ICML 2026 Partial Fusion [A]）。生物方向另有 [CellStream (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/download/37041/41003) [P] 用 MMOT 做细胞轨迹推断。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **直接相关**。(i) OTP-FM 的核心机制——把中间边缘化成动态 OT 作用量里的势——天然可移植为推理期引导：把学到的 OT 势转成对预训练扩散/FM 采样轨迹的 guidance 项，即可在不重训的情况下迫使轨迹经过指定的中间分布，其论文已给出"势强度→边缘偏差"的 Wasserstein 界，可直接改造成引导误差预算。(ii) Wukong 展示了完全 training-free 的范式：冻结预训练 flow 模型，只在条件 token 空间解 free-support barycenter 即可获得平滑语义插值轨迹。(iii) OT 模型融合线（OTFusion→Partial Fusion）本质是"无须重训地对齐多个预训练模型的参数轨迹终点"，与轨迹对齐共享"先对齐再平均"的模板。
- 方向二（OT 引导跨域生成）: **直接相关**。barycenter 就是"多个域的公共中间域"：能量引导 EOT barycenter 在预训练生成流形上学 barycenter，本身就是一种跨域 OT 引导采样；WGF 正则化 barycenter 已在视觉/神经科学/化工的域适应 benchmark 上验证"多源域→barycenter 域"的聚合式迁移；MMtSBM/3MSBM 把两域桥推广为 k 域联合桥，为"多源→单目标"的跨域生成提供了比逐对 SB 更全局的耦合结构。

## 5. 开放问题与可发论文的切入点

1. **扩散/FM 模型的 barycentric merging**：OT 融合线止步于判别网络。开放问题：逐层 \(W_2\) barycenter 融合两个微调扩散模型（如两个 LoRA 风格模型）时，生成分布是否近似输出分布的 Wasserstein barycenter？可做：(a) 证明速度场线性结构下"权重 barycenter ⇒ 分布 barycenter"的充分条件（如速度场对参数仿射+位移凸性）；(b) 实验对比逐层 OT-barycenter merge vs task-arithmetic vs 直接平均，在 FID/风格保真上验证。
2. **OT 势即插即用引导（衔接方向一）**：把 OTP-FM 的势项改写为对预训练 FM/扩散模型的 inference-time guidance——给定少量中间域样本，拟合 OT 势并注入 probability-flow ODE，使采样轨迹无须重训地经过中间分布；理论上把 OTP-FM 的势强度–Wasserstein 偏差界推广为"引导误差沿轨迹的传播界"。这是空白：现有 guidance 均为终端条件式，没有"中间边缘约束"式 guidance。
3. **多条件组合生成的 free-support barycenter 理论**：Wukong 只给了 3D 形变的工程配方。可证：预训练确定性流映射 \(\Phi\) 与条件空间 barycentric 插值何时"可交换"（\(\Phi(\mathrm{bary}(c_i))\approx\mathrm{bary}(\Phi(c_i))\)），给出以 \(\Phi\) 的 Lipschitz 常数/单调性刻画的误差界；实验推到 T2I 多概念组合，替代 embedding 线性插值。
4. **结构化 MMOT 作为多模态联合生成的训练目标**：利用 Altschuler–Boix-Adserà 的"结构化代价多项式可解"判据（树/低秩/图形代价），设计 k 模态联合生成器：以树形 MMOT 耦合为配对目标训练 multimarginal FM，绕开一般 MMOT 的 NP-hard；对比逐对耦合训练在跨模态一致性上的增益。DMMOT 的动力学形式可为其提供连续时间训练目标。
5. **熵偏差×生成插值的系统研究**：熵正则 barycenter 的 blur/shrinkage（Janati 2020）在潜空间插值里如何表现为生成质量退化？做一个 exact(SGA)/entropic/debiased/neural(NOT) 四方对照的生成插值 benchmark（同一 StyleGAN/SD 潜空间），并设计随去噪时间表衰减 \(\varepsilon\) 的 debiasing schedule——目前无任何工作量化这一点。

## 6. 代码与资源

- [Bexorg-Inc/OTP-FM](https://github.com/Bexorg-Inc/OTP-FM) — OTP-FM 官方 PyTorch 实现
- [panostheo98/3MSBM](https://github.com/panostheo98/3MSBM)（PyTorch Lightning）；[Genentech/MMFM](https://github.com/Genentech/MMFM)；[tgravier/MMDSBM-pytorch](https://github.com/tgravier/MMDSBM-pytorch)（MMtSBM，附 [实验站点](https://mmdsbm.notion.site)）
- [justkolesov/NOTBarycenters](https://github.com/justkolesov/NOTBarycenters)、[justkolesov/EnergyGuidedBarycenters](https://github.com/justkolesov/EnergyGuidedBarycenters) — 神经/能量引导 barycenter（含 StyleGAN2 潜空间与 Ave, Celeba! 数据集 notebook）
- [SigmaNova/barycentric-gradient-flows](https://github.com/SigmaNova/barycentric-gradient-flows) — UAI 2026 WGF 正则化 barycenter
- [jvlindheim/free-support-barycenters](https://github.com/jvlindheim/free-support-barycenters)、[jvlindheim/mot](https://github.com/jvlindheim/mot) — 免费支撑近似算法（POT 后端）
- [yairshenfeld/DMMOT](https://github.com/yairshenfeld/DMMOT) — 动力学 MMOT 的 Julia 原-对偶近端分裂
- [graldij/transformer-fusion](https://github.com/graldij/transformer-fusion) — Transformer OT 融合
- 库：[POT](https://pythonot.github.io/)（`ot.lp.free_support_barycenter`、`ot.bregman.barycenter`、MMOT 模块）；[OTT-JAX](https://ott-jax.readthedocs.io/)（GPU Sinkhorn barycenter）
- 常用数据/benchmark：单细胞时序（EB/CITE-seq、100 维转录组）、Ave, Celeba!（barycenter 图像基准）、气象/海洋浮标轨迹（OTP-FM 实验）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2026_Kansal_multimarginal_fm_ot_potentials.pdf | Multimarginal Flow Matching with Optimal Transport Potentials | 成功（11.9MB, arXiv） |
| 2025_Theodoropoulos_3MSBM_momentum_mmsb_matching.pdf | Momentum Multi-Marginal Schrödinger Bridge Matching | 成功（8.4MB, arXiv） |
| 2025_Rohbeck_MMFM_flow_matching_time_conditions.pdf | Modeling Complex System Dynamics with Flow Matching Across Time and Conditions | 成功（8.5MB, ICLR 官方 OA） |
| 2026_Gravier_MMtSBM_temporal_sb_matching.pdf | Multi-marginal Temporal Schrödinger Bridge Matching from Unpaired Data | 成功（8.0MB, arXiv） |
| 2024_Kolesov_neural_ot_barycenters.pdf | Estimating Barycenters of Distributions with Neural Optimal Transport | 成功（19.2MB, arXiv） |
| 2026_Kim_sobolev_gradient_ascent_barycenter.pdf | Sobolev Gradient Ascent for Optimal Transport: Barycenter Optimization and Convergence Analysis | 成功（5.5MB, arXiv） |
| 2026_Montesuma_wgf_regularized_barycenter.pdf | Wasserstein Gradient Flows for Scalable and Regularized Barycenter Computation | 成功（10.7MB, arXiv） |
| 2023_Noble_treedsb_wasserstein_barycenters.pdf | Tree-Based Diffusion Schrödinger Bridge with Applications to Wasserstein Barycenters | 成功（6.5MB, arXiv） |
