# T21 分子与科学计算中的 OT 流生成

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」全景在 AI4Science 的落地面——把 flow matching 的「直路径耦合」直接实例化为分子/蛋白/晶体这类**带对称性与流形结构**数据上的最优传输问题。OT 在这里有两副面孔：训练期作为 minibatch 直路径耦合（在 SE(3)^N、环面、置换群 quotient 上做 Riemannian/equivariant OT），以及作为**跨生物学分布对齐**工具（apo→holo、序列→结构、LLM 分布→晶体数据）。与 T24（单细胞轨迹）、T28（黎曼流形一般理论）、T22（离散序列扩散）互补而不重叠。

## 1. 核心问题与背景

分子科学的生成任务天然是「把一个分布传输到另一个分布」：从噪声/先验到平衡构象（Boltzmann 采样）、从 apo 到 holo（对接）、从序列到骨架（结构预测）、从组成到晶胞（材料）。这些数据都带强几何约束——旋转/平移/置换不变性、SE(3) 刚体、周期边界、离散原子种类——直接套用欧氏直线插值会穿出数据流形、产生非物理中间态、并使 ODE 积分步数（NFE）暴涨。Flow matching 的两个自由度恰好对症：(i) **概率路径的几何**要放到正确流形上（Riemannian FM）；(ii) **source–target 耦合**用 OT 拉直可显著缩短积分路径。分子界因此把「OT 耦合」做成两类核心工具：等变/黎曼 OT（在对称群 quotient 上求 Monge/minibatch OT，得到近最优直路径，加速采样、稳定训练），以及不平衡/条件 OT（松弛质量守恒或注入 side-information，对齐跨域分布）。由此催生了 equivariant Boltzmann generator、SE(3) 蛋白骨架流、等变构象流、晶体黎曼流、以及基于（不平衡）OT 的柔性对接一整条谱系，成为「扩散×OT」在真实科学问题上最活跃的验证场。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Equivariant Flow Matching (Klein, Krämer, Noé) | 2023·NeurIPS | [P] | 提出 equivariant OT-FM：先 Hungarian 排列对齐、再 Kabsch 旋转对齐，对 invariant 密度得近最优直路径，首个 Cartesian 坐标高效 Boltzmann generator（丙氨酸二肽） | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bc827452450356f9f558f4e4568d553b-Abstract-Conference.html) |
| ⭐ FoldFlow: SE(3)-Stochastic Flow Matching for Protein Backbone Generation (Bose, Akhound-Sadegh et al.) | 2024·ICLR | [P] | 蛋白 OT-flow 范式：证明 SE(3)^N 上 Monge map 存在，构造 FoldFlow-OT（更直更稳）与 FoldFlow-SFM（SE(3) 随机桥），可任意 invariant source→target | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html) |
| FoldFlow-2: Sequence-Augmented SE(3)-Flow Matching (Huguet, Vuckovic et al.) | 2024·NeurIPS | [P] | 序列条件化 SE(3)-FM：pLM 编码序列 + minibatch Riemannian OT 耦合 + ReFT 强化微调，规模化到 ~21M 合成结构，超 RFdiffusion | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/39ca8893ea38905a9d2ffe786e85af0f-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2405.20313) |
| ⭐ Proteina: Scaling Flow-based Protein Structure Generative Models (Geffner et al., NVIDIA) | 2025·ICLR (Oral) | [P] | 把蛋白骨架 FM 规模化：非等变大 transformer（~400M）+ 层级 fold 条件 + autoguidance，800 残基仍可设计，并引入分布相似度指标 | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f4e9121ad30cd4e5528042fbfd835b3f-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2503.00710) |
| Improved motif-scaffolding with SE(3) flow matching (FrameFlow; Yim, Campbell et al.) | 2024·TMLR | [P] | FrameFlow（SE(3)-FM，采样步数少 5×）的 scaffolding 扩展：motif amortization 与**无须重训**的 motif guidance，可设计性/多样性大幅提升 | [OpenReview/TMLR](https://openreview.net/forum?id=fa1ne8xDGn) |
| ⭐ ET-Flow: Equivariant Flow-Matching for Molecular Conformer Generation (Hassan et al.) | 2024·NeurIPS | [P] | 等变 FM + harmonic prior + Kabsch 对齐直接在全原子坐标上做构象生成，轻量、少 NFE，GEOM 上刷新精度/物理有效性 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e8bd617e7dd0394ceadf37b4a7773179-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2410.22388) |
| Transferable Boltzmann Generators (Klein & Noé) | 2024·NeurIPS | [P] | 等变 CNF+FM 做跨化学空间零样本平衡采样（二肽），并实证：可区分粒子多时 OT-FM 相对普通 FM 增益变小（重要 caveat） | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5035a409f5798e188079e236f437e522-Abstract.html) · [arXiv](https://arxiv.org/abs/2406.14426) |
| Generative Flows on Discrete State-Spaces / MultiFlow (Campbell, Yim et al.) | 2024·ICML | [P] | 用 CTMC 实现离散 FM，并与 FrameFlow 连续结构流组合成 序列-结构 co-design（离散部分见 T22） | [PMLR v235](https://proceedings.mlr.press/v235/campbell24a.html) · [arXiv](https://arxiv.org/abs/2402.04997) |
| ⭐ FlowMM: Generating Materials with Riemannian Flow Matching (Miller, Chen et al.) | 2024·ICML | [P] | 晶体黎曼 FM：在分数坐标环面 + 晶格 + 原子种类的联合流形上做几何约束生成（CSP/DNG） | [PMLR v235](https://proceedings.mlr.press/v235/miller24a.html) |
| FlowLLM: Flow Matching for Material Generation with LLMs as Base Distributions (Sriram, Miller, Chen, Wood) | 2024·NeurIPS | [P] | 把微调 LLM 的分布当作 RFM 的 base，实现「文本晶体分布→图数据分布」跨域传输，稳定率×3、S.U.N.×~1.5 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/51d317df78eded9eb3c9d3fb1091c279-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2410.23405) |
| Open Materials Generation with Stochastic Interpolants / OMatG (Hollmer, Fuemmeler et al.) | 2025·ICML | [P] | 用 stochastic interpolants 统一 diffusion/FM，对晶格/坐标各自调插值 + 种类用 discrete FM，CSP/DNG 超 FlowMM/DiffCSP/MatterGen | [PMLR v267](https://proceedings.mlr.press/v267/hollmer25a.html) · [arXiv](https://arxiv.org/abs/2502.02582) |
| ⭐ Composing Unbalanced Flows for Flexible Docking and Relaxation / FlexDock (Corso, Somnath et al.) | 2025·ICLR | [P] | 提出 **Unbalanced Flow Matching**（松弛边缘约束→更易学耦合），链式 apo→holo 流形对接 + 全原子松弛，PoseBusters 合格率 30%→73% | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/451dbb8f4fca0327ac4e6782786673bf-Abstract-Conference.html) |
| FlowDock: Geometric Flow Matching for Protein-Ligand Docking and Affinity Prediction (Morehead & Cheng) | 2025·Bioinformatics | [P] | CFM 直接把 apo 映到 holo（多配体）并预测亲和力；耦合用 apo–holo 结构过滤定义 + harmonic ligand prior，盲对接超单序列 AF3 | [DOI](https://doi.org/10.1093/bioinformatics/btaf187) · [arXiv](https://arxiv.org/abs/2412.10966) |

补充（正文引用，跨课题/上游/基线，不计入上表统计）：Fast protein backbone generation with SE(3) flow matching（FrameFlow 原始 workshop 版，[arXiv 2310.05297](https://arxiv.org/abs/2310.05297)）[R]；Riemannian Flow Matching on General Geometries（Chen & Lipman, ICLR 2024，流形 FM 底座，一般理论归 T28）[P]；Metric Flow Matching（Kapusniak et al., NeurIPS 2024，[arXiv 2405.14780](https://arxiv.org/abs/2405.14780)，数据流形测地插值，主战场单细胞归 T24）[P]；Wasserstein Flow Matching（Haviv et al., ICML 2025，[PMLR v267](https://proceedings.mlr.press/v267/haviv25a.html)，「分布的分布」生成含点云，OT 一般化见 T05/T08）[P]；扩散基线 FrameDiff / RFdiffusion / DiffDock / DiffCSP / MatterGen（分别为这些 FM 工作的前身或对照，多为 [P]）。

## 3. 方法演进脉络

**2023 起源。** 底座是 Riemannian FM（Chen & Lipman）与 OT-CFM/minibatch OT（Tong et al.）。分子界最早把二者缝合的是 **Equivariant Flow Matching**（Klein et al., NeurIPS 2023）：注意到 invariant 密度下欧氏 minibatch OT 需要极大 batch 才近似真 OT，于是在对称群上做「先 Hungarian 排列、再 Kabsch 旋转」的等变对齐，得到近最优直路径，首次在 Cartesian 坐标下训出高效 Boltzmann generator。几乎同期 **FoldFlow**（Bose et al., ICLR 2024）把 Riemannian OT 搬上 SE(3)^N：证明 SE(3) 上 Monge map 存在，派生 FoldFlow-OT（更直更稳）与 FoldFlow-SFM（SE(3) 随机桥），确立蛋白骨架 OT-flow 范式；**FrameFlow**（Yim et al.）则走效率路线（少 5× 步数），其无条件模型经 motif guidance 可**无重训**做 scaffolding（TMLR 2024）。

**2024 扩张。** 蛋白侧 **FoldFlow-2**（NeurIPS 2024）加入 pLM 序列编码、minibatch Riemannian OT 耦合与 ReFT 奖励对齐，规模化到 ~21M 合成结构；**MultiFlow**（ICML 2024）用 CTMC 离散 FM 组合 FrameFlow 结构流做序列-结构 co-design（离散部分属 T22）。构象侧 **ET-Flow**（NeurIPS 2024）用 harmonic prior + Kabsch 对齐在全原子坐标上把直路径做到 GEOM SOTA 且更轻量。采样侧 **Transferable Boltzmann Generators**（NeurIPS 2024）实现跨化学空间零样本采样，并给出关键 caveat：当可区分粒子多时，OT-FM 相对普通 FM 的增益会缩小——提醒 OT 在 AI4Science 并非无脑增益。材料侧 **FlowMM**（ICML 2024）在「分数坐标环面 + 晶格 + 种类」联合流形上做黎曼 FM；**FlowLLM**（NeurIPS 2024）把微调 LLM 的输出分布当 RFM 的 base，做「文本晶体→图数据」跨域传输，稳定率×3。

**2025 深化与统一。** **Proteina**（ICLR 2025 Oral）证明非等变大 transformer + FM + autoguidance 能在 800 残基尺度刷新可设计性，并用分布相似度指标跳出单点 designability 的评价局限。对接被正式改写成分布传输：**FlexDock**（ICLR 2025）提出 Unbalanced Flow Matching（松弛边缘约束换取更易学的耦合），链式「流形对接 + 全原子松弛」，PoseBusters 合格率 30%→73%；**FlowDock**（Bioinformatics 2025）用 CFM 直接 apo→holo 并预测亲和力，盲对接超单序列 AF3。材料侧 **OMatG**（ICML 2025）以 stochastic interpolants 统一 diffusion/FM，对不同自由度分别调插值并联合离散 FM 生成种类，全面超 FlowMM/DiffCSP/MatterGen。总脉络：OT 的角色从「训练期直路径耦合」（Klein / FoldFlow / ET-Flow）→「跨域分布对齐工具」（FlowLLM / FlowDock / FoldFlow-2）→「可调不平衡 / 周期几何传输」（FlexDock / OMatG）。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **中等相关**。两条线索：(i) 等变/黎曼 OT 耦合把 source–target 拉直，等价于在训练阶段就「对齐」了积分轨迹，使采样 NFE 大降（FrameFlow 少 5× 步、ET-Flow 更少步、FoldFlow-OT 更稳）——这是「更好轨迹」的耦合级实现；(ii) 推理期的 **motif guidance**（FrameFlow）与 **autoguidance/CFG**（Proteina）能在**不改权重**的前提下把已训好的无条件流场对齐到条件/目标 motif，正是「无须重训的轨迹对齐」在结构生物学的直接实例。可复用点：把这些 training-free 引导视为对预训练分子流的轨迹算子。
- 方向二（OT 引导跨域生成）: **强相关，且是本子课题主线**。FoldFlow / FoldFlow-2 用 Riemannian OT 耦合实现任意 invariant source→target；FlowLLM 用 RFM 把「LLM 分布→晶体数据分布」这对异构域对齐；FlowDock / FlexDock 把 apo→holo 显式写成（不平衡）OT 传输，FlexDock 的 Unbalanced FM 更是 UOT 思想在生成中的落地；Transferable BG 则把 OT/等变对齐用作跨分子的分布迁移。可以说，「OT 作为分布对齐工具」在分子科学已被反复验证为跨域生成的骨架。

## 5. 开放问题与可发论文的切入点

1. **可扩展的等变/群-quotient OT 耦合**：现有 equivariant OT（Hungarian+Kabsch）在 minibatch 内近 O(n^3)、且需大 batch 才逼近真 OT，对大蛋白/多链不可扩展。做什么：在 SE(3)^N/S_N quotient 上设计 entropic/Sinkhorn 型等变 OT 或 amortized 排列-对齐网络，证明其对 quotient OT plan 的收敛性，并在 FoldFlow-2 / ET-Flow 上量化 NFE–designability 权衡（可对接种子库 ICML 2026「Optimal Transport with Symmetry Groups」的群对称 OT 目标）。
2. **不平衡 OT 用于构象态重加权**：分子有多个 metastable 态且采样存在质量偏置。做什么：把 FlexDock 的 Unbalanced FM 推广到 Boltzmann reweighting，证明「松弛边缘 = 对目标能量的温度/质量重标定」，在 alanine dipeptide / 二肽上比较 free-energy 估计的方差与偏差，给出 UOT 松弛强度→重加权有效样本量的界。
3. **OT 耦合强度对 designability–diversity–novelty 的定量刻画**：FoldFlow-OT 更稳但可能压低多样性，目前无理论。做什么：在 Proteina / FoldFlow-2 上把「OT 耦合温度」设为可调超参，度量三角指标随耦合强度的 Pareto 前沿，并从耦合诱导的生成分布熵推导单调关系，指导「多稳但仍可设计」的采样配置。
4. **跨域生物学传输的统一条件-Riemannian-OT 框架**：docking（apo→holo）、seq→structure（FoldFlow-2）、LLM→crystal（FlowLLM）都在做条件分布传输，但耦合定义各异（RMSD 过滤 / masked / minibatch ROT）。做什么：提出以 pocket/sequence/组成为 cost side-information 的统一条件 Riemannian OT，给出对应 c-transform 与可 simulation-free 训练的目标，在 docking + co-design 上做统一 benchmark。
5. **周期几何上的联合（连续×离散）最优传输**：晶体同时含平移群（周期环面）、离散种类、连续坐标/晶格，OMatG/FlowMM 分而治之却缺「环面 OT + 种类 discrete OT 联合插值」的最优性理论。做什么：在分数坐标环面上定义 Riemannian OT 并与种类的 discrete OT 耦合，证明联合插值的动能最优条件（Benamou–Brenier 型），实验刷新 MP-20 的 CSP/DNG。

## 6. 代码与资源

**官方代码库**
- FoldFlow（Dreamfold）: https://github.com/DreamFold/FoldFlow
- FrameFlow / Improved motif-scaffolding（Microsoft）: https://github.com/microsoft/protein-frame-flow
- Proteina（NVIDIA-BioNeMo）: https://github.com/NVIDIA-BioNeMo/proteina
- ET-Flow: https://github.com/shenoynikhil/ETFlow ; 检查点 https://zenodo.org/records/14226681
- FlowMM / FlowLLM（FAIR）: https://github.com/facebookresearch/flowmm
- OMatG（FERMat-ML）: https://github.com/FERMat-ML/OMatG
- FlowDock（BioinfoMachineLearning）: https://github.com/BioinfoMachineLearning/FlowDock
- TorchCFM（含 OT-CFM / minibatch OT 耦合，通用底座；注意其 minibatch-OT 非总体精确 OT）: https://github.com/atong01/conditional-flow-matching
- Meta 官方 flow_matching 库（连续/离散/Riemannian）: https://github.com/facebookresearch/flow_matching

**数据集 / benchmark**
- 蛋白骨架: PDB、SCOPe；RFdiffusion 24-motif scaffolding benchmark；可设计性=ProteinMPNN(8 序列)→AlphaFold2 自洽 scRMSD<2Å；多样性/新颖性用 TM-score / Foldseek；Proteina 新增分布相似度指标（FPSD/fJSD 类）。
- 分子构象: GEOM-QM9、GEOM-DRUGS；指标 COV/MAT-Recall/Precision（RMSD 阈值）。
- 材料/晶体: Materials Project（MP-20）、Alex-MP；任务 CSP（给定组成）与 DNG（de novo）；用 MLIP 结构弛豫 + e-above-hull 判 S.U.N.（stable/unique/novel）。
- 对接: PDBBind（含柔性 pocket split）、PoseBusters、DockGen(-E)、CASP16 亲和力赛道。
- Boltzmann 采样: alanine dipeptide、dipeptides（二肽转移集）。

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2023_Klein_equivariant_flow_matching.pdf | Equivariant Flow Matching (NeurIPS 2023) | 成功 |
| 2024_Bose_foldflow_se3_stochastic.pdf | SE(3)-Stochastic Flow Matching for Protein Backbone Generation / FoldFlow (ICLR 2024) | 成功 |
| 2024_Huguet_foldflow2_seq_augmented.pdf | Sequence-Augmented SE(3)-Flow Matching / FoldFlow-2 (NeurIPS 2024) | 成功 |
| 2025_Geffner_proteina_scaling.pdf | Proteina: Scaling Flow-based Protein Structure Generative Models (ICLR 2025 Oral) | 成功 |
| 2024_Hassan_etflow_conformer.pdf | ET-Flow: Equivariant Flow-Matching for Molecular Conformer Generation (NeurIPS 2024) | 成功 |
| 2024_Miller_flowmm_materials.pdf | FlowMM: Generating Materials with Riemannian Flow Matching (ICML 2024) | 成功 |
| 2025_Corso_flexdock_unbalanced_flows.pdf | Composing Unbalanced Flows for Flexible Docking and Relaxation / FlexDock (ICLR 2025) | 成功 |
| 2025_Hollmer_omatg_stochastic_interpolants.pdf | Open Materials Generation with Stochastic Interpolants / OMatG (ICML 2025) | 成功 |
