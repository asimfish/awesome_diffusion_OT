# T28 黎曼流形上的流匹配与 OT

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题覆盖「扩散×OT」全景中的非欧几何分支：当数据活在球面、双曲空间、李群 SO(3)/SE(3)、SPD 矩阵流形或统计流形上时，流匹配/扩散的概率路径与 OT 耦合都要换成测地线成本与内蕴几何。它同时是理论支点（流形 OT、Wasserstein-over-manifold 几何）与应用出口（机器人姿态、地球科学、脑影像）。欧氏 FM 基础归 T07，分子/蛋白应用归 T21，此处只保留其方法谱系引用。

## 1. 核心问题与背景

许多科学与工程数据天然生活在黎曼流形上：地震/气候事件在球面 \(S^2\)、机器人姿态在 SO(3)/SE(3)、脑功能连接是 SPD/相关矩阵、离散类别分布在带 Fisher-Rao 度量的单纯形上。直接套用欧氏扩散/FM 会产生离开流形的样本、有偏的似然，以及"切空间近似 ≠ 内蕴 OT"的系统误差（曲率大或跨 cut locus 时误差放大）。本子方向要解决三层问题：(i) 生成层——如何在流形上定义 simulation-free 的概率路径与目标向量场（RFM 的 premetric/谱距离、桥过程混合、平凡化动量）；(ii) 耦合层——如何用测地成本 \(c=d_g^2/2\) 的 OT/熵 OT 给训练配对与跨域映射提供最优性（Riemannian minibatch OT、神经 c-凹势、流形 Sinkhorn/SB）；(iii) 空间层——把 Wasserstein 空间自身当作无穷维黎曼流形做生成与优化（Bures-Wasserstein 子流形上的 FM 与变分推断、Wasserstein proximal point）。这条线在 2024-2026 从"能在流形上生成"快速走向"少步化、可扩展、带 OT 最优性保证"，可发论文空间集中在三层的交叉处。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Flow Matching on General Geometries (RFM), Chen & Lipman | 2024·ICLR Oral | [P] | 用 premetric（测地/谱距离）闭式构造流形条件向量场，简单几何上完全 simulation-free，奠定黎曼 FM 范式 | [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d1f9936d3be6997ffffab692977eebe6-Abstract-Conference.html) |
| ⭐ Riemannian Score-Based Generative Modelling (RSGM), De Bortoli et al. | 2022·NeurIPS | [P] | 把 SGM 的前向/反向 SDE 定义到紧流形（测地随机游走+热核），开创流形扩散并给出地球科学球面基准 | [proceedings](https://papers.nips.cc/paper_files/paper/2022/hash/105112d52254f86d5854f3da734a52b4-Abstract-Conference.html) |
| Riemannian Diffusion Schrödinger Bridge (RDSB), Thornton et al. | 2022·arXiv | [R] | 把 DSB/IPF 推广到紧流形，做流形上两分布间 SB 插值（地球气候数据），是流形熵 OT 动态解法源头 | [arXiv](https://arxiv.org/abs/2207.03024) |
| Riemannian Diffusion Mixture, Jo & Hwang | 2024·ICML | [P] | 用桥过程混合直接构造生成扩散（漂移=数据方向切向量加权平均），绕开热核估计与散度计算，一般流形可扩展 | [PMLR](https://proceedings.mlr.press/v235/jo24a.html) |
| ⭐ Metric Flow Matching (MFM), Kapusniak et al. | 2024·NeurIPS | [P] | 在数据诱导度量下学最小动能插值（近似测地线）替代直线插值，OT-MFM 在单细胞轨迹上 SOTA；"数据流形"版黎曼 FM | [OpenReview](https://openreview.net/forum?id=fE3RqiF4Nx) |
| Fisher-Flow, Davis et al. | 2024·NeurIPS | [P] | 把离散数据重参数化到 \(S^d_+\) 球面正象限沿 Fisher-Rao 测地做 FM，并用黎曼 OT 配对改善训练动力学，证明 KL 最速下降 | [arXiv](https://arxiv.org/abs/2405.14664) |
| Statistical/Categorical Flow Matching (SFM), Cheng et al. | 2024·NeurIPS | [P] | 统计流形（Fisher 信息度量）上的 FM：测地最短路+自然梯度解释+精确似然，训练中可加 OT | [OpenReview](https://openreview.net/forum?id=5fybcQZ0g4) |
| Trivialized Momentum Diffusion (TDM), Zhu et al. | 2025·ICLR | [P] | 李群上引入平凡化动量：score 在固定李代数（平坦空间）学习，无投影/切空间近似，首次做高维 SO(n)/U(n) 生成 | [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b3cff9947cde08b4fa70652cb8ac9209-Abstract-Conference.html) |
| ⭐ Wasserstein Flow Matching (WFM/BW-FM), Haviv et al. | 2025·ICML | [P] | 把 FM 提升到"分布的分布"：证明 Wasserstein 测地是合法条件流，高斯族用闭式 Bures-Wasserstein 路径、点云用熵 OT 估计 | [PMLR](https://proceedings.mlr.press/v267/haviv25a.html) |
| Stochastic Variance-Reduced VI on the Bures-Wasserstein Manifold | 2025·ICLR | [A] | BW 流形（高斯族 Wasserstein 几何）上的方差缩减变分推断，完善 BW 空间一阶优化工具箱 | [OpenReview](https://openreview.net/forum?id=iMJpmcYucq) |
| Riemannian Proximal Sampler, Guan, Balasubramanian & Ma | 2025·NeurIPS | [P] | MBI+热核双 oracle 的流形高精度采样，\(O(\log(1/\varepsilon))\) 迭代；可解释为 Wasserstein 空间上熵正则黎曼 proximal point 的离散化 | [proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/8e185f16e458ef5e666901260079cd42-Abstract-Conference.html) |
| DiffeoCFM: RFM for Brain Connectivity via Pullback Geometry, Collas et al. | 2025·NeurIPS | [P] | 全局微分同胚 pullback 度量下的黎曼 CFM 等价于"变换后做欧氏 CFM"：SPD 用矩阵对数、相关矩阵用归一化 Cholesky，fMRI/EEG 大规模验证 | [proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/5616112a0120c15bf7d47a6bccc21bc3-Abstract-Conference.html) |
| Riemannian Consistency Model (RCM), Cheng et al. | 2025·NeurIPS | [P] | 用协变导数+指数映射参数化把一致性模型推广到流形，蒸馏(RCD)与从头训练(RCT)理论等价，球面/环面/SO(3) 少步生成 | [NeurIPS 页](https://neurips.cc/virtual/2025/poster/117955) |
| Riemannian Flow Matching Policy (RFMP), Braun et al. | 2024·IROS | [P] | 把黎曼 FM 用于机器人视觉运动策略（状态含姿态流形），比 Diffusion Policy 更平滑、推理更快 | [IEEE](https://doi.org/10.1109/iros58592.2024.10801521) |
| ⭐ Riemannian Neural Optimal Transport (RNOT), Micheli et al. | 2026·arXiv | [R] | 证明离散化流形 OT 必有维数灾难；用 c-凹神经势 \(T=\exp_x(-\nabla\phi)\) 学连续流形 OT map，次指数复杂度保证 | [arXiv](https://arxiv.org/abs/2602.03566) |

补充条目（正文引用）：Spherical DYffusion（NeurIPS 2024 Spotlight [P]，[proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e6a11b618402617342f38f5b49430937-Abstract-Conference.html)，SFNO×扩散做百年气候集合模拟）；FlowMM（ICML 2024 [P]，[PMLR](https://proceedings.mlr.press/v235/miller24a.html)，晶体材料的黎曼 FM，分子类应用详见 T21）；Flow Matching on Lie Groups（[R]，[arXiv 2504.00494](https://arxiv.org/abs/2504.00494)，用指数曲线替代测地线的内蕴李群 FM）；Entropic Riemannian Neural OT（[R]，[arXiv 2605.04255](https://arxiv.org/abs/2605.04255)，内蕴熵 OT 目标+神经求解器）；From Schrödinger Bridge to OT over Sub-Riemannian Manifolds（[R]，[arXiv 2605.11429](https://arxiv.org/abs/2605.11429)，非完整约束几何上的 Sinkhorn 型算法与零噪声极限）；Diffusion Approximations to Schrödinger Bridges on Manifolds（[R]，[arXiv 2512.18867](https://arxiv.org/abs/2512.18867)，小温度极限下 SB 势梯度收敛到流形 score）；Riemannian Barycentric Projections（[R]，[arXiv 2606.07926](https://arxiv.org/abs/2606.07926)）；Spherical Tree-Sliced Wasserstein Distance（ICLR 2025 [A]，[OpenReview](https://openreview.net/forum?id=FPQzXME9NK)）；The Riemannian Geometry of Sinkhorn Divergences（[R]，[arXiv 2405.04987](https://arxiv.org/abs/2405.04987)）。

计数：正表 [P]×12、[A]×1、[R]×2；含补充共 [P]×14、[A]×2、[R]×8，合计 24 篇。

## 3. 方法演进脉络

**第一代（2020-2022）：把扩散搬上流形。** Riemannian CNF（Mathieu & Nickel 2020）与 Moser Flow（2021）需要昂贵模拟或散度计算；RSGM（NeurIPS 2022）确立"流形前向 SDE + 时间反演 + 测地随机游走"的标准框架，但依赖热核（谱分解/渐近近似），高维一般流形难以扩展。同期 RDSB 把 Schrödinger bridge 的 IPF 迭代推广到紧流形，第一次把"流形上的熵 OT 动态解"用于生成与插值。

**第二代（2023-2024）：simulation-free 化与几何特化。** Chen & Lipman 的 RFM（ICLR 2024 Oral）是分水岭：只需一个 premetric（简单几何用测地距离、一般几何用谱距离），条件向量场闭式可得，简单流形上训练完全 simulation-free、无散度项。此后按几何类型特化：统计流形上 Fisher-Flow 与 SFM 借 Fisher-Rao 度量把离散生成变成球面正象限上的黎曼 FM（并显式引入黎曼 OT 配对）；李群上 TDM 用平凡化动量把 score 学习搬回固定李代数，绕开 RFM 需要的对数映射与 RSGM 需要的热核特征函数，首次扩展到高维 SO(n)/U(n)；蛋白骨架的 SE(3) 系（FrameDiff、FoldFlow 等）属于同一谱系（应用细节归 T21）。另一条支线不假设已知解析流形，而是"学度量"：MFM 用数据诱导度量（LAND/RBF）学最小动能插值，把"贴着数据流形走"变成 FM 的一部分，OT-MFM 表明 OT 配对与学到的几何可以叠加增益。Jo & Hwang 的桥混合模型则从过程构造角度绕开时间反演本身。

**第三代（2025-2026）：少步化、平坦化与 OT 理论深化。** 工程侧：DiffeoCFM 指出若流形存在全局微分同胚到欧氏空间（SPD 的 log、相关矩阵的 Cholesky），pullback 度量下的黎曼 CFM 严格等价于"变换后欧氏 CFM"，训练/采样都用标准 ODE 求解器；RCM 把一致性蒸馏推广到流形（协变导数+exp 参数化），补齐流形生成的少步推理短板。理论侧：RNOT 证明任何离散化流形 OT 都逃不掉维数灾难，并用 c-凹神经势给出多项式复杂度的连续替代——这是流形版 neural OT（对应欧氏 Makkuva/Korotin 线）的奠基；Entropic RNOT 补上内蕴熵正则版本；sub-Riemannian SB 给非完整约束系统（欠驱动机器人）提供 Sinkhorn 型算法与 Γ-收敛到 Benamou-Brenier 的保证；Riemannian Proximal Sampler 把采样解释为 Wasserstein 空间上的熵正则 proximal point，将"流形上的采样/生成"与"测度空间上的优化"正式缝合。空间层面，WFM/BW-FM 把 FM 的基空间从流形推到 Wasserstein 空间本身（高斯族=BW 子流形有闭式测地线），与 BW 流形上的 SVR-VI、ITSPACE 型 BW 单调更新共同构成"在分布空间上做几何优化/生成"的新板块。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）：**部分直接、整体是空白区**。RCM 是"流形版蒸馏"，说明少步化需求已经蔓延到非欧几何；但欧氏世界成熟的 reflow/直化、OT 耦合替换、training-free 轨迹拼接在流形上几乎没有对应物——RFM 的条件路径虽是测地线（逐条"直"），其边际耦合仍是独立耦合而非测地 OT 耦合，因此"Riemannian rectified flow / 测地 reflow"是天然缺口。TDM 与 DiffeoCFM 提供的平坦化坐标（李代数、log/Cholesky 空间）恰好是把欧氏免重训技巧零成本迁移到流形的桥梁。
- 方向二（OT 引导跨域生成）：**直接相关**。RNOT/ERNOT 就是"跨域 map"的流形版（测地成本 Monge map，支持 out-of-sample）；RDSB、sub-Riemannian SB 与 Diffusion Approximations to SB on Manifolds 给流形上两分布间的桥/插值提供动态 OT 工具（气候两时刻分布、机器人姿态分布对齐）；WFM/BW-FM 把"跨域"抬到分布族层面（源域协方差族→目标域协方差族），对 EEG 跨被试/跨设备增广（DiffeoCFM 的数据设定）是现成的引导结构。

## 5. 开放问题与可发论文的切入点

1. **黎曼 Rectified Flow / 测地 reflow**：把 reflow 的"重耦合+直化"搬到流形——耦合改用测地成本 minibatch OT（Fisher-Flow 已有实现可复用），"直"的目标改为测地线（加速度=0 的协变导数判据）。理论上证每轮 reflow 使内蕴传输成本单调不增、并刻画 cut locus 附近的退化；实验在 RSGM 地球科学数据 + SO(3) 姿态上画 NFE-质量 Pareto 曲线。与 RCM 蒸馏正交，可组合。
2. **曲率感知的 minibatch 测地 OT 偏差理论**：欧氏 minibatch OT 的偏差已有刻画（T08），流形上完全缺失。证明 minibatch 测地 OT 耦合到总体 OT 的收敛率如何依赖截面曲率上界/单射半径，并设计曲率修正权重；直接改进 Fisher-Flow、OT-MFM、SFM 的训练配对，是"小定理+即插即用模块"型论文。
3. **BW 空间上的 OT 引导跨域生成**：结合 WFM（BW 测地闭式）与熵 BW 耦合，做"协方差=风格"的条件生成：EEG/fMRI 跨被试、跨设备的连接矩阵增广（DiffeoCFM 的五个公开数据集现成），对比 pullback-欧氏基线，验证内蕴 BW 耦合在小样本医学数据上的增益。
4. **流形上的 simulation-free SB matching**：RDSB 依赖 IPF 反复模拟；把 Light-SB / IMF（T-系列已覆盖欧氏版）推广到紧流形——用热核截断或 Varadhan 渐近（Riemannian Proximal Sampler 的 oracle 技术）写出流形桥漂移的可回归形式，在球面气候插值上验证；难点与卖点都在"热核不可解析时如何保持 simulation-free"。
5. **SE(3) 上的 training-free OT guidance 用于机器人跨域适配**：在 RFMP/抓取姿态生成之上加测地 OT 对齐层（仿真姿态分布→真机姿态分布），推理时用 RNOT 势的梯度做 guidance 而不重训策略；同时检验"单切空间谬误"（Jaquier et al.）警告的近似误差在 guidance 场景是否致命。

## 6. 代码与资源

- [facebookresearch/riemannian-fm](https://github.com/facebookresearch/riemannian-fm)：RFM 官方（球面/环面/双曲/SPD/网格；注意 CC BY-NC 许可）
- [ccr-cheng/riemannian-consistency-model](https://github.com/ccr-cheng/riemannian-consistency-model)：RCM 官方（含 RFM teacher 训练）
- [antoinecollas/DiffeoCFM](https://github.com/antoinecollas/DiffeoCFM)：SPD/相关矩阵 FM + ADNI/ABIDE/OASIS-3、BNCI EEG 数据加载器
- [olsdavis/fisher-flow](https://github.com/olsdavis/fisher-flow)：Fisher-Flow 官方（含黎曼 OT 配对开关）
- [yuchen-zhu-zyc/TDM](https://github.com/yuchen-zhu-zyc/TDM)：李群平凡化扩散官方
- [WassersteinFlowMatching](https://github.com/WassersteinFlowMatching/WassersteinFlowMatching)：WFM/BW-FM 官方
- [Rose-STL-Lab/spherical-dyffusion](https://github.com/Rose-STL-Lab/spherical-dyffusion)：球面气候模拟（权重在 HuggingFace）
- 通用库：[Geomstats](https://geomstats.github.io/)（流形原语）、[geoopt](https://github.com/geoopt/geoopt)（黎曼优化）、[OTT-JAX](https://ott-jax.readthedocs.io/)（含低秩/几何 Sinkhorn）、[TorchCFM](https://github.com/atong01/conditional-flow-matching)（OT-CFM 基线）
- 基准数据：RSGM 地球科学四件套（火山/地震/洪水/野火，球面）；蛋白/RNA 扭转角（环面，方法基准）；LASA 手写（机器人策略）；BNCI2014-002/BNCI2015-001（EEG SPD）；ADNI/ABIDE/OASIS-3（fMRI 相关矩阵）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2024_Chen_riemannian_flow_matching.pdf | Flow Matching on General Geometries | 成功 |
| 2022_DeBortoli_riemannian_score_based_gen.pdf | Riemannian Score-Based Generative Modelling | 成功 |
| 2024_Kapusniak_metric_flow_matching.pdf | Metric Flow Matching for Smooth Interpolations on the Data Manifold | 成功 |
| 2025_Haviv_wasserstein_flow_matching.pdf | Wasserstein Flow Matching: Generative Modeling Over Families of Distributions | 成功 |
| 2026_Micheli_riemannian_neural_ot.pdf | Riemannian Neural Optimal Transport | 成功 |
| 2025_Cheng_riemannian_consistency_model.pdf | Riemannian Consistency Model | 成功 |
| 2025_Collas_diffeocfm_brain_connectivity.pdf | Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry | 成功 |
| 2025_Zhu_trivialized_momentum_lie_groups.pdf | Trivialized Momentum Facilitates Diffusion Generative Modeling on Lie Groups | 成功 |
