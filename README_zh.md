# Awesome Diffusion × Optimal Transport

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Papers](https://img.shields.io/badge/papers-446-orange.svg)](#content) [![Reports](https://img.shields.io/badge/deep--dive%20reports-438-green.svg)](#reports) [![zh-PDF](https://img.shields.io/badge/translated%20PDFs-126-red.svg)](#reports)

[English](README.md) | [中文](README_zh.md)

**扩散/流生成模型 × 最优传输**的证据优先阅读清单：30 个子课题、六大板块，覆盖理论（扩散≟OT、Schrödinger 桥、收敛率）、
流匹配与轨迹拉直、跨域翻译、多模态、OT 变体与系统基建。每篇论文附：

- 中文**深读报告**（`reports/`，8 节模板：问题 / 方法 / 理论 / 实验数字 / 地图位置 / 局限 / 启发 / 资源，数字带出处）；
- **原文 PDF**（`papers/`）与 [SuperTranslate](https://github.com/asimfish/super_translate) **保版式中文译文**（`papers_zh/`，附对象级 QA `*.inspect.json`）；
- 机器可读元数据（`data/`），本 README 由 `src/generator.py` 生成。

总量：446 篇（⭐ 核心 139）| 深读报告 438 | 原文 PDF 363 | 中文译文 126。证据级：[P] 论文集 / [A] 官方已接收 / [R] 预印本 / [B] 教材综述——主会、期刊、workshop、预印本永远分开标。

综合分析见 [`report/`](report/)（问题→理论→经典→前沿→我们能做什么，中英 PDF）与 [`slides/`](slides/)（HTML PPT + Beamer PDF）；2026 Q3 增量趋势见 [`trends/`](trends/)。

*维护：[asimfish](https://github.com/asimfish)。结构参考 [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co)。*

## 目录

A. [理论基础](#sec-a)
&emsp;T01 [OT 数学基础（面向生成模型研究者的最小必要集）](#t01)
&emsp;T02 [扩散模型与 OT 的理论联系](#t02)
&emsp;T03 [Schrödinger Bridge 与扩散生成](#t03)
&emsp;T04 [熵正则 OT 与 Sinkhorn 在生成建模中的角色](#t04)
&emsp;T05 [Wasserstein 梯度流与 JKO 格式生成模型](#t05)
&emsp;T06 [扩散/流生成模型的收敛性与统计理论](#t06)
B. [流匹配与轨迹拉直](#sec-b)
&emsp;T07 [Flow Matching 基础谱系](#t07)
&emsp;T08 [OT-CFM 与 minibatch OT 耦合](#t08)
&emsp;T09 [Rectified Flow 与轨迹拉直](#t09)
&emsp;T10 [一致性模型与少步蒸馏的 OT 视角](#t10)
&emsp;T11 [免训练采样器与 ODE 求解器](#t11)
&emsp;T12 [推理阶段的 OT 对齐与噪声-样本耦合](#t12)
C. [跨域生成与翻译](#sec-c)
&emsp;T13 [神经 OT 映射与无配对图像翻译](#t13)
&emsp;T14 [扩散桥 / Schrödinger 桥的图像到图像翻译](#t14)
&emsp;T15 [医学影像模态转换与 OT/SB/扩散](#t15)
&emsp;T16 [OT 代价先验引导的跨域语义对应](#t16)
&emsp;T17 [风格迁移与域自适应中的 OT×扩散](#t17)
&emsp;T18 [条件生成与 guidance 的 OT 形式化](#t18)
D. [模态扩展](#sec-d)
&emsp;T19 [视频生成与时序一致性中的 OT/流](#t19)
&emsp;T20 [3D/点云/几何生成中的 OT 与流](#t20)
&emsp;T21 [分子与科学计算中的 OT 流生成](#t21)
&emsp;T22 [离散数据与文本中的扩散/流与最优传输](#t22)
&emsp;T23 [语音与音频中的流匹配与 Schrödinger 桥](#t23)
&emsp;T24 [单细胞与生物轨迹推断中的 OT×流](#t24)
E. [OT 变体前沿](#sec-e)
&emsp;T25 [非平衡/部分 OT 在生成建模中的应用](#t25)
&emsp;T26 [Gromov-Wasserstein 与跨空间生成对齐](#t26)
&emsp;T27 [多边际 OT 与 Wasserstein 重心的生成应用](#t27)
&emsp;T28 [黎曼流形上的流匹配与 OT](#t28)
F. [系统、评测与趋势](#sec-f)
&emsp;T29 [高性能 OT 求解器与训练基础设施](#t29)
&emsp;T30 [端侧部署、benchmark 与顶会趋势（博客落地场景：端侧图像生成）](#t30)
G. [2026 Q3 增量与趋势](#trends)
H. [深读报告、译文与综合报告](#reports)
I. [贡献与引用](#contributing)

<a id="sec-a"></a>
## A. 理论基础

<a id="t01"></a>
### T01. OT 数学基础（面向生成模型研究者的最小必要集）

课题综合：[`topics/t01.md`](topics/t01.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t01_ot_foundations.md`](source/kb/t01_ot_foundations.md)

1. ⭐ **Optimal Transport for Machine Learners.** arXiv 课程讲义, 2025. [B] [paper](https://arxiv.org/abs/2505.06589) [report](reports/2505.06589.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.06589.pdf)

    *Gabriel Peyré*

    面向 ML 的现代精简版：Monge/Kantorovich、Brenier、对偶、动态形式、Bures 度量、梯度流，并直连 GAN/扩散/transformer，配可运行 notebook

2. ⭐ **Computational Optimal Transport.** Foundations and Trends in ML, 2019. [B] [paper](https://arxiv.org/abs/1803.00567) [report](reports/1803.00567.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1803.00567.pdf)

    *Gabriel Peyré, Marco Cuturi*

    计算 OT 标准教材：离散 OT、对偶、动态形式、barycenter 全覆盖，"从零到能跑代码"的主线读物

3. ⭐ **Optimal Transport for Applied Mathematicians.** Birkhäuser, 2015. [B] [paper](https://math.univ-lyon1.fr/~santambrogio/OTAM-cvgmt.pdf) [report](reports/Optimal_Transport_for_Applied_Mathematicians.md)

    *Santambrogio*

    应用数学侧标准参考：Kantorovich 对偶、Brenier、Benamou–Brenier、W 空间几何的严格但可读证明

4. ⭐ **A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem.** Numerische Mathematik 84, 2000. [P] [paper](https://doi.org/10.1007/s002110050002) [report](reports/A_computational_fluid_mechanics_solution_to_the_Mo.md)

    *Benamou & Brenier*

    \(W_2^2\) = 连续性方程约束下的最小动能；扩散/流匹配轨迹分析的通用语言与数值入口

5. ⭐ **Polar factorization and monotone rearrangement of vector-valued functions.** Comm. Pure Appl. Math. 44(4), 1991. [P] [paper](https://doi.org/10.1002/cpa.3160440402) [report](reports/Polar_factorization_and_monotone_rearrangement_of.md)

    *Brenier*

    二次代价下最优映射存在唯一且 = 凸势梯度；一切"扩散潜码 ≈ OT 映射"讨论的理论根基

6. **A Survey on Optimal Transport for Machine Learning: Theory and Applications.** IEEE Access 13, 2025. [B] [paper](https://arxiv.org/abs/2106.01963) [report](reports/2106.01963.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2106.01963.pdf)

    *Luis Caicedo Torres, Luiz Manella Pereira, M. Hadi Amini*

    入门友好的 2025 应用综述，含历史脉络与对偶/熵正则数学预备

7. **Optimal and Diffusion Transports in Machine Learning.** arXiv 综述, 2025. [B] [paper](https://arxiv.org/abs/2512.06797) [report](reports/2512.06797.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2512.06797.pdf)

    *Gabriel Peyré*

    直接把扩散与 OT 两条主线串成统一框架（Eulerian/Lagrangian、BB、梯度流、transformer token 流）；本项目的桥梁综述

8. **Recent Advances in Optimal Transport for Machine Learning.** IEEE TPAMI 47(2), 2025. [B] [paper](https://arxiv.org/abs/2306.16156) [report](reports/2306.16156.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2306.16156.pdf)

    *Eduardo Fernandes Montesuma, Fred Ngolè Mboula, Antoine Souloumiac*

    2012–2023 OT×ML 全景应用综述（生成、迁移、RL 与计算 OT 扩展）；查应用先查它

9. **A Combinatorial Algorithm for Semi-Discrete Optimal Transport.** NeurIPS, 2024. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2d950a2cfd8a75124c178a89545b97fd-Abstract-Conference.html) [report](reports/A_Combinatorial_Algorithm_for_Semi_Discrete_Optima.md)

    与主流 smooth dual/Newton 不同的组合算法路径，半离散 OT 的最新进展

10. **Lectures on Optimal Transport.** Springer UNITEXT 169, 2024. [B] [paper](https://link.springer.com/book/10.1007/978-3-031-76834-7) [report](reports/Lectures_on_Optimal_Transport.md)

    *Ambrosio, Brué & Semola*

    SNS 二十年课程沉淀；给出两种自包含的 Kantorovich 对偶证明，通向几何/泛函不等式与 PDE

11. **Statistical optimal transport.** arXiv 讲义（Saint-Flour）, 2024. [B] [paper](https://arxiv.org/abs/2407.18163) [report](reports/2407.18163.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2407.18163.pdf)

    *Sinho Chewi, Jonathan Niles-Weed, Philippe Rigollet*

    统计侧系统讲义（经验测度收敛、估计率）；T01 只需其数学预备章，深入归 T06

12. **An Invitation to Optimal Transport, Wasserstein Distances, and Gradient Flows.** EMS Textbooks, 2023. [B] [paper](https://ems.press/books/etb/258) [report](reports/An_Invitation_to_Optimal_Transport_Wasserstein_Dis.md)

    *Figalli & Glaudo*

    146 页最短严格入门：对偶、Brenier、W 距离、JKO/Otto 微积分，含带解答习题，一学期课体量

13. **Optimal transport: discretization and algorithms.** Handbook of Numerical Analysis 22, 2021. [B] [paper](https://arxiv.org/abs/2003.00855) [report](reports/2003.00855.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2003.00855.pdf)

    *Quentin Merigot, Boris Thibert*

    半离散 OT 最佳系统讲义：Laguerre cell、damped Newton、离散化误差分析

14. **Convergence of a Newton algorithm for semi-discrete optimal transport.** J. Eur. Math. Soc. 21(9), 2019. [P] [paper](https://arxiv.org/abs/1603.05579) [report](reports/1603.05579.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1603.05579.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1603.05579.zh.pdf)

    *Jun Kitagawa, Quentin Mérigot, Boris Thibert*

    半离散 OT 阻尼牛顿法的全局线性收敛，半离散数值求解的理论支柱

15. **Optimal Transport: Old and New.** Springer Grundlehren 338, 2009. [B] [paper](https://link.springer.com/book/10.1007/978-3-540-71050-9) [report](reports/Topics_in_Optimal_Transportation.md)

    *Villani*

    理论百科全书：正则性、几何（Ricci 曲率）方向的终极参考；不建议作为第一本，查证明时用

<a id="t02"></a>
### T02. 扩散模型与 OT 的理论联系

课题综合：[`topics/t02.md`](topics/t02.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t02_diffusion_ot_theory.md`](source/kb/t02_diffusion_ot_theory.md)

1. ⭐ **Diffusion models for Gaussian distributions: Exact solutions and Wasserstein errors.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2405.14250) [report](reports/2405.14250.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.14250.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2405.14250.zh.pdf)

    *Emile Pierret, Bruno Galerne*

    高斯数据下逆向 SDE 与 PF-ODE 的解析解；对任意数值格式给出初始化/截断/离散化误差的**精确** W2 表达；证实 Heun 格式最优、SDE 采样器对误差更鲁棒

2. ⭐ **An optimal control perspective on diffusion-based generative modeling.** TMLR, 2024. [P] [paper](https://arxiv.org/abs/2211.01364) [report](reports/2211.01364.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2211.01364.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2211.01364.zh.pdf)

    *Julius Berner, Lorenz Richter, Karen Ullrich*

    Hopf–Cole 变换导出时间反转 log 密度的 HJB 方程；ELBO = 控制论 verification theorem 的直接推论；给出路径空间 KL 表述与 DIS 采样器

3. ⭐ **On the Trajectory Regularity of ODE-based Diffusion Sampling.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2405.11326) [report](reports/2405.11326.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.11326.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2405.11326.zh.pdf)

    *Defang Chen, Zhenyu Zhou, Can Wang, Chunhua Shen, Siwei Lyu*

    发现 PF-ODE 采样轨迹与内容无关的"线性–非线性–线性"强形状正则性（近似落在低维平面），据此提出 GITS 时间表，5–10 NFE 显著提效

4. ⭐ **Understanding DDPM Latent Codes Through Optimal Transport.** ICLR, 2023. [P] [paper](https://arxiv.org/abs/2202.07477) [report](reports/2202.07477.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2202.07477.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2202.07477.zh.pdf)

    *Valentin Khrulkov, Gleb Ryzhakov, Andrei Chertkov, Ivan Oseledets*

    猜想 DDPM encoder map = Monge OT map；证明多元高斯情形，并用 tensor-train Fokker–Planck 数值求解器支持一般情形

5. ⭐ **The flow map of the Fokker–Planck equation does not provide optimal transport.** Appl. Math. Lett. 133:108225, 2022. [P] [paper](https://www.sciencedirect.com/science/article/abs/pii/S089396592200180X) [report](reports/The_flow_map_of_the_Fokker_Planck_equation_does_no.md)

    *Lavenant & Santambrogio*

    反例证伪上述猜想：流映射一般不是凸函数梯度，障碍是一个 Hessian 非交换项；同时指出数值上"近乎最优"、量化次优度是开放问题

6. **Learning Monge maps with constrained drifting models.** arXiv 2603.25182, 2026. [R] [paper](https://arxiv.org/abs/2603.25182) [report](reports/2603.25182.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2603.25182.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2603.25182.zh.pdf)

    *Théo Dumont, Théo Lacombe, François-Xavier Vialard*

    辩论线最新延续：在"传输映射空间"内做约束梯度流（约束到 OT 映射凸集），证明长时存在性并收敛到 Monge map——把 drift 型生成动力学"修正"为真 OT

7. **Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control.** ICLR (Spotlight), 2025. [P] [paper](https://arxiv.org/abs/2409.08861) [report](reports/2409.08861.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2409.08861.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2409.08861.zh.pdf)

    *Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, Ricky T. Q. Chen*

    把 reward 微调严格表述为 SOC 问题；证明必须用 memoryless 噪声调度 σ(t)=√(2η_t) 消除初值-value-function 偏差；lean adjoint 回归算法

8. **Stochastic Optimal Control Matching.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2312.02027) [report](reports/2312.02027.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2312.02027.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2312.02027.zh.pdf)

    *Carles Domingo-Enrich, Jiequn Han, Brandon Amos, Joan Bruna, Ricky T. Q. Chen*

    把 SOC 求解转化为最小二乘回归（借鉴条件 score matching 哲学），path-wise reparameterization trick 降方差

9. **The Brownian transport map.** Probab. Theory Relat. Fields, 2024. [P] [paper](https://arxiv.org/abs/2111.11521) [report](reports/2111.11521.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2111.11521.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2111.11521.zh.pdf)

    *Dan Mikulincer, Yair Shenfeld*

    用 Föllmer 过程（即扩散去噪漂移 ∇log P_{1−t}f）构造 Wiener 测度→目标测度的传输映射，在 OT 尚属开放的情形证明 Lipschitz 收缩，导出新泛函不等式

10. **Understanding Diffusion Models by Feynman's Path Integral.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2403.11262) [report](reports/2403.11262.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.11262.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2403.11262.zh.pdf)

    *Yuji Hirono, Akinori Tanaka, Kenji Fukushima*

    路径积分（Onsager–Machlup 作用量）表述扩散模型；ODE↔SDE 插值参数 h 类比普朗克常数，用 WKB 展开计算 NLL 并以 W2 度量评估随机性收益

11. **Wasserstein proximal operators describe score-based generative models and resolve memorization.** arXiv 2402.06162, 2024. [R] [paper](https://arxiv.org/abs/2402.06162) [report](reports/2402.06162.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.06162.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2402.06162.zh.pdf)

    *Benjamin J. Zhang, Siting Liu, Wuchen Li, Markos A. Katsoulakis, Stanley J. Osher*

    SGM = 交叉熵的（正则化）Wasserstein proximal 算子；MFG 最优性条件 = 前向受控 FP + 后向 HJB；核公式解释流形学习与记忆化

12. **Formulating Discrete Probability Flow Through Optimal Transport.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2311.03886) [report](reports/2311.03886.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2311.03886.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2311.03886.zh.pdf)

    *Pengze Zhang, Hubery Yin, Chen Li, Xiaohua Xie*

    给出连续 PF 在特定条件下于任意有限时间区间上是 Monge OT map 的证明，并据此定义离散扩散的 probability flow，提升采样确定性

13. **Score-based Generative Modeling Secretly Minimizes the Wasserstein Distance.** NeurIPS, 2022. [P] [paper](https://arxiv.org/abs/2212.06359) [report](reports/2212.06359.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2212.06359.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2212.06359.zh.pdf)

    *Dohyun Kwon, Ying Fan, Kangwook Lee*

    分布层结论：W2(数据,生成) ≤ √(score matching 损失)×常数+偏移，证明用连续性方程估计 W2 的时间导数

14. **Comparison of transport map generated by heat flow interpolation and the optimal transport Brenier map.** Comm. Contemp. Math. 23(6), 2021. [P] [paper](https://arxiv.org/abs/1709.06464) [report](reports/1709.06464.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1709.06464.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1709.06464.zh.pdf)

    *Anastasiya Tanana*

    更早的反例前驱：Kim–Milman 热流插值映射对高斯测度（drift −Ax）一般 ≠ Brenier map；一维或径向对称时二者重合

15. **Score-Based Generative Modeling through SDEs.** ICLR (Oral), 2021. [P] [paper](https://openreview.net/forum?id=PxTIG12RRHS) [report](reports/Score_Based_Generative_Modeling_through_SDEs.md)

    *Song et al.*

    提出 VP/VE-SDE 与 PF-ODE 统一框架，定义了本课题讨论的 encoder/flow map 对象

<a id="t03"></a>
### T03. Schrödinger Bridge 与扩散生成

课题综合：[`topics/t03.md`](topics/t03.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t03_schrodinger_bridge.md`](source/kb/t03_schrodinger_bridge.md)

1. ⭐ **Light and Optimal Schrödinger Bridge Matching.** ICML, 2024. [P] [paper](https://proceedings.mlr.press/v235/gushchin24a.html) [report](reports/Light_and_Optimal_SB_Matching_LightSB_M_Gushchin_e.md)

    「最优 SB matching」：任意输入耦合、单次 matching 即可证明恢复 SB（免迭代误差累积），并统一 matching 与 EBM 目标

2. ⭐ **Schrödinger Bridge Flow for Unpaired Data Translation (α-DSBM).** NeurIPS, 2024. [P] [paper](https://papers.nips.cc/paper_files/paper/2024/hash/bb3cfcb0284642a973dd631ec9184f2f-Abstract-Conference.html) [report](reports/Schr_dinger_Bridge_Flow_IMF_DSBM_De_Bortoli_et_al.md)

    定义路径测度流「SB Flow」，离散化得 α-IMF（α=1 退化为 IMF）；α<1 时在线更新单一网络，免多轮重训，∀α∈(0,1] 收敛到 SB

3. ⭐ **Diffusion Schrödinger Bridge Matching.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2303.16852) [report](reports/2303.16852.md)

    *Yuyang Shi, Valentin De Bortoli, Andrew Campbell, Arnaud Doucet*

    提出 IMF：交替 Markov 投影与 reciprocal 投影，配 bridge-matching 回归实现；解决 DSB 的误差累积与"遗忘"问题

4. ⭐ **Likelihood Training of Schrödinger Bridge using FBSDEs Theory.** ICLR, 2022. [P] [paper](https://openreview.net/forum?id=nioAdKCEdXB) [report](reports/Likelihood_Training_of_SB_using_FBSDEs_SB_FBSDE_Ch.md)

    用前向-后向 SDE 理论把 SB 最优性条件变为可训练的对数似然目标，严格泛化 SGM 的训练目标

5. ⭐ **Diffusion Schrödinger Bridge (DSB), De Bortoli et al.** NeurIPS (Spotlight), 2021. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html) [report](reports/Diffusion_Schr_dinger_Bridge_DSB_De_Bortoli_et_al.md)

    深度网络实现 IPF 迭代求解 SB：有限时间生成、连续状态空间的 Sinkhorn 类比，SGM 恰为第一次 IPF 迭代

6. **Diffusion & Adversarial SB via IPMF, Kholkin et al.** ICLR (Poster), 2026. [A] [paper](https://openreview.net/forum?id=38fGCBhFF5) [report](reports/Diffusion_Adversarial_SB_via_IPMF_Kholkin_et_al.md)

    证明实践中"双向交替 IMF"启发式 = IMF+IPF 的组合（IPMF），多设定下收敛，并给出相似度-质量 trade-off 旋钮

7. **Reflected Schrödinger Bridge Matching.** arXiv 2607.03626, 2026. [R] [paper](https://arxiv.org/abs/2607.03626) [report](reports/2607.03626.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2607.03626.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2607.03626.zh.pdf)

    *Marcus Häggbom, Viktor Nilsson, Pierre Nyquist, Joakim andén*

    把 IMF/α-DSBM 推广到反射 SDE（有界域约束生成），保持收敛论证

8. **From Schrodinger Bridge to Optimal Transport over Sub-Riemannian Manifolds.** arXiv 2605.11429, 2026. [R] [paper](https://arxiv.org/abs/2605.11429) [report](reports/2605.11429.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2605.11429.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2605.11429.zh.pdf)

    *Daniel Owusu Adu, Karthik Elamvazhuthi, Bahman Gharesifard*

    非完整约束几何（sub-Riemannian）上的 SB 与 OT 理论

9. **Foundations of Schrödinger Bridges for Generative Modeling.** arXiv 2603.18992, 2026. [B] [paper](https://arxiv.org/abs/2603.18992) [report](reports/2603.18992.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2603.18992.pdf)

    *Sophia Tang*

    220 页专著式教程：从 EOT/路径空间优化/随机控制第一性原理统一 diffusion、score、flow matching 与 SB

10. **Categorical SB Matching (CSBM), Ksenofontov & Korotin.** ICML, 2025. [P] [paper](https://proceedings.mlr.press/v267/ksenofontov25a.html) [report](reports/Categorical_SB_Matching_CSBM_Ksenofontov_Korotin.md)

    证明离散(有限)状态空间上 D-IMF 收敛到 SB，把 SB matching 推广到 VQ token/文本/分子等离散数据

11. **Exponential Convergence Guarantees for Iterative Markovian Fitting.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2510.20871) [report](reports/2510.20871.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2510.20871.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2510.20871.zh.pdf)

    *Marta Gentiloni Silveri, Giovanni Conforti, Alain Durmus*

    首个 IMF 非渐近指数收敛率（KL）：基于 Markovian 投影的新收缩估计，覆盖(强/弱)对数凹两个 regime，为 DSBM 铺路

12. **Feedback SB Matching (FSBM), Theodoropoulos et al.** ICLR (Oral), 2025. [P] [paper](https://openreview.net/forum?id=k3tbMMW8rH) [report](reports/Feedback_SB_Matching_FSBM_Theodoropoulos_et_al.md)

    半监督 SB：<8% 预配对样本作为 state feedback 嵌入广义 EOT→动态匹配，显著加速训练并提升泛化

13. **Momentum Multi-Marginal Schrödinger Bridge Matching.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2506.10168) [report](reports/2506.10168.md)

    *Panagiotis Theodoropoulos, Augustinos D. Saravanos, Evangelos A. Theodorou, Guan-Horng Liu*

    相空间提升 + 多点条件随机桥：多边缘条件最优控制的 matching 解法，训练中保持全部中间边缘，捕捉长程时间依赖

14. **Multi-marginal temporal Schrödinger Bridge Matching from unpaired data.** arXiv 2510.01894, 2025. [R] [paper](https://arxiv.org/abs/2510.01894) [report](reports/2510.01894.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2510.01894.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2510.01894.zh.pdf)

    *Thomas Gravier, Thomas Boyer, Auguste Genovesio*

    非配对多时刻快照的多边缘 SB matching，factorized 拟合支撑高维视频/生物动力学

15. **Statistical Analysis of the Sinkhorn Iterations for Two-Sample Schrödinger Bridge Estimation.** arXiv 2510.22560, 2025. [R] [paper](https://arxiv.org/abs/2510.22560) [report](reports/2510.22560.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2510.22560.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2510.22560.zh.pdf)

    *Ibuki Maeda, Rentian Yao, Atsushi Nitanda*

    「Sinkhorn bridge」统计分析：证明 [SF]²M/DSBM-IMF/BM²/LightSB(-M) 的最优估计量一致，泛化误差分析对全家族生效

16. **Adversarial SB Matching (ASBM), Gushchin et al.** NeurIPS, 2024. [P] [paper](https://openreview.net/forum?id=L3Knnigicu) [report](reports/Adversarial_SB_Matching_ASBM_Gushchin_et_al.md)

    离散时间 IMF（D-IMF）理论 + DD-GAN 实现：只学几个离散转移核，推断从数百步降到几步

17. **BM$^2$: Coupled Schrödinger Bridge Matching.** arXiv 2409.09376, 2024. [R] [paper](https://arxiv.org/abs/2409.09376) [report](reports/2409.09376.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2409.09376.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2409.09376.zh.pdf)

    *Stefano Peluchetti*

    耦合双向 bridge matching，无需交替优化的 SB 逼近

18. **Generalized SB Matching (GSBM), Liu et al.** ICLR, 2024. [P] [paper](https://openreview.net/forum?id=SoismgeX7z) [report](reports/Generalized_SB_Matching_GSBM_Liu_et_al.md)

    把任务特定 state cost 纳入匹配框架 = 条件随机最优控制求解广义 SB，训练全程保持可行 transport

19. **Light Schrödinger Bridge.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2310.01174) [report](reports/2310.01174.md)

    *Alexander Korotin, Nikita Gushchin, Evgeny Burnaev*

    Schrödinger 势的高斯混合参数化：归一化常数闭式、免仿真、CPU 分钟级求解，并证 SB 万能逼近性

20. **Simulation-free Score & Flow Matching ([SF]²M), Tong et al.** AISTATS, 2024. [P] [paper](https://proceedings.mlr.press/v238/tong24a.html) [report](reports/Simulation_free_Score_Flow_Matching_SF_M_Tong_et_a.md)

    用静态 (minibatch) Sinkhorn 耦合 + score/flow matching 免仿真近似 SB；首个高维单细胞动力学建模

21. **Variational Schrödinger Diffusion Models (VSDM), Deng et al.** ICML, 2024. [P] [paper](https://proceedings.mlr.press/v235/deng24c.html) [report](reports/Variational_Schr_dinger_Diffusion_Models_VSDM_Deng.md)

    变分推断线性化 SB 前向 score，恢复后向 score 的免仿真训练；随机逼近证明收敛、无需 warm-up

22. **Deep Momentum Multi-Marginal SB (DMSB), Chen et al.** NeurIPS, 2023. [P] [paper](https://openreview.net/forum?id=ykvvv0gc4R) [report](reports/Deep_Momentum_Multi_Marginal_SB_DMSB_Chen_et_al.md)

    相空间 Bregman-IPF 解多边缘 SB，从位置快照重建速度分布（3MSBM 的前身）

23. **Diffusion Bridge Mixture Transports (IDBM), Peluchetti.** JMLR 24(374), 2023. [P] [paper](https://www.jmlr.org/papers/v24/23-0527.html) [report](reports/Diffusion_Bridge_Mixture_Transports_IDBM_Peluchett.md)

    迭代扩散桥混合（IMF 思想的独立源头）：每次迭代都保持两端边缘的合法 transport，并给出收敛性初步分析

24. **Stochastic control liaisons: Richard Sinkhorn meets Gaspard Monge on a Schroedinger bridge.** SIAM Review, 2021. [B] [paper](https://arxiv.org/abs/2005.10963) [report](reports/2005.10963.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2005.10963.pdf)

    *Yongxin Chen, Tryphon T. Georgiou, Michele Pavon*

    从随机控制视角统一 Sinkhorn/IPF 与 SB，扩散生成前的经典理论坐标系

25. **Léonard, A Survey of the Schrödinger Problem.** DCDS, 2014. [B] [paper](https://doi.org/10.3934/dcds.2014.34.1533) [report](reports/L_onard_A_Survey_of_the_Schr_dinger_Problem.md)

    SB 问题标准综述：静态/动态等价、与熵正则 OT 的关系、large deviation 极限

<a id="t04"></a>
### T04. 熵正则 OT 与 Sinkhorn 在生成建模中的角色

课题综合：[`topics/t04.md`](topics/t04.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t04_entropic_sinkhorn_gen.md`](source/kb/t04_entropic_sinkhorn_gen.md)

1. ⭐ **Progressive Entropic Optimal Transport Solvers.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2406.05061) [report](reports/2406.05061.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.05061.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2406.05061.zh.pdf)

    *Parnian Kassraie, Aram-Alexandre Pooladian, Michal Klein, James Thornton, Jonathan Niles-Weed, Marco Cuturi*

    把 ε 调度嵌入分步(时间离散化)求解：逐段解 EOT 并收缩正则，估计耦合与 map 更稳、可证一致性，是"ε 不再是单一超参"的代表

2. ⭐ **Entropic estimation of optimal transport maps.** arXiv:2109.12004, 2021. [R] [paper](https://arxiv.org/abs/2109.12004) [report](reports/2109.12004.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2109.12004.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2109.12004.zh.pdf)

    *Aram-Alexandre Pooladian, Jonathan Niles-Weed*

    entropic map = 熵耦合的 barycentric projection = 熵对偶势的梯度（熵版 Brenier 定理），O(n²) 可算且带有限样本率

3. ⭐ **Interpolating between Optimal Transport and MMD using Sinkhorn Divergences.** AISTATS, 2019. [P] [paper](https://arxiv.org/abs/1810.08278) [report](reports/1810.08278.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1810.08278.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1810.08278.zh.pdf)

    *Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, Gabriel Peyré*

    给出 debiased Sinkhorn divergence 的标准定义并证正定性、凸性、度量化极限（ε→0 得 OT，ε→∞ 得 MMD）

4. ⭐ **Learning Generative Models with Sinkhorn Divergences.** AISTATS, 2018. [P] [paper](https://arxiv.org/abs/1706.00292) [report](reports/1706.00292.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1706.00292.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1706.00292.zh.pdf)

    *Aude Genevay, Gabriel Peyré, Marco Cuturi*

    首个用 Sinkhorn loss（unrolled 自动微分 + entropic 平滑）大规模训练生成模型的可行方案

5. ⭐ **Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances.** NeurIPS, 2013. [P] [paper](https://arxiv.org/abs/1306.0895) [report](reports/1306.0895.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1306.0895.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1306.0895.zh.pdf)

    *Marco Cuturi*

    把熵正则+Sinkhorn 矩阵缩放引入 ML，开创可扩展正则化 OT 这一领域

6. **One-Step Generative Modeling via Wasserstein Gradient Flows.** arXiv:2605.11755, 2026. [R] [paper](https://arxiv.org/abs/2605.11755) [report](reports/2605.11755.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2605.11755.pdf)

    *Jiaqi Han, Puheng Li, Qiushan Guo, Renyuan Xu, Stefano Ermon, Emmanuel J. Candès*

    用 debiased Sinkhorn divergence 作为 WGF 能量泛函，把多步分布演化压缩进一步生成器；ImageNet-256 一步 FID 1.29

7. **Energy-guided Entropic Neural Optimal Transport.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2304.06094) [report](reports/2304.06094.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2304.06094.pdf)

    *Petr Mokrov, Alexander Korotin, Alexander Kolesov, Nikita Gushchin, Evgeny Burnaev*

    把 EBM 与 EOT 对偶结合：能量函数参数化熵对偶势，学到的随机 plan 直接用于 unpaired 图像域迁移

8. **ENOT: Expectile Regularization for Fast and Accurate Training of Neural Optimal Transport.** NeurIPS spotlight, 2024. [P] [paper](https://arxiv.org/abs/2403.03777) [report](reports/2403.03777.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.03777.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2403.03777.zh.pdf)

    *Nazar Buzun, Maksim Bobrin, Dmitry V. Dylov*

    用 expectile 回归正则约束对偶 Kantorovich 势，替代昂贵的 c-transform 内层优化，W₂ benchmark 上质量 3×、速度 10×

9. **Sinkhorn Flow as Mirror Flow: A Continuous-Time Framework for Generalizing the Sinkhorn Algorithm.** AISTATS, 2024. [P] [paper](https://proceedings.mlr.press/v238/reza-karimi24a.html) [report](reports/Sinkhorn_Flow_as_Mirror_Flow_A_Continuous_Time_Fra.md)

    Sinkhorn 的连续时间极限是测度空间 mirror flow，导出对噪声/偏差鲁棒的新变体，统一 Wasserstein mirror flow 等动力学

10. **A Unified Framework for Implicit Sinkhorn Differentiation.** CVPR, 2022. [P] [paper](https://arxiv.org/abs/2205.06688) [report](reports/2205.06688.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2205.06688.pdf)

    *Marvin Eisenberger, Aysim Toker, Laura Leal-Taixé, Florian Bernard, Daniel Cremers*

    用隐式函数定理统一各种 Sinkhorn 层梯度（学习 cost 与 marginal 皆可），比 unrolling 更省内存更稳

11. **Debiaser Beware: Pitfalls of Centering Regularized Transport Maps.** ICML, 2022. [P] [paper](https://arxiv.org/abs/2202.08919) [report](reports/2202.08919.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2202.08919.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2202.08919.zh.pdf)

    *Aram-Alexandre Pooladian, Marco Cuturi, Jonathan Niles-Weed*

    证明对 map 估计 debiasing 并非总有益：ε 大或样本少时反而更差，动摇"一律 debias"的信条

12. **Faster Wasserstein Distance Estimation with the Sinkhorn Divergence.** NeurIPS, 2020. [P] [paper](https://arxiv.org/abs/2006.08172) [report](reports/2006.08172.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2006.08172.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2006.08172.zh.pdf)

    *Lenaic Chizat, Pierre Roussillon, Flavien Léger, François-Xavier Vialard, Gabriel Peyré*

    证明 debiased Sinkhorn divergence 估计平方 W₂ 距离在更大 ε 下仍达到近最优误差，计算-统计两头受益

13. **Sinkhorn AutoEncoders.** UAI, 2019. [P] [paper](https://arxiv.org/abs/1810.01118) [report](reports/1810.01118.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1810.01118.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1810.01118.zh.pdf)

    *Giorgio Patrini, Rianne van den Berg, Patrick Forré, Marcello Carioni, Samarth Bhargav, Max Welling et al.*

    在 WAE 框架内用 Sinkhorn 距离对齐 latent aggregated posterior 与先验，确定性编解码器即可生成

14. **Statistical bounds for entropic optimal transport: sample complexity and the central limit theorem.** NeurIPS, 2019. [P] [paper](https://arxiv.org/abs/1905.11882) [report](reports/1905.11882.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1905.11882.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1905.11882.zh.pdf)

    *Gonzalo Mena, Jonathan Weed*

    EOT 经验估计的 O(1/√n) 样本复杂度与 CLT，是"熵正则修复维数灾难"的理论支柱

15. **Improving GANs Using Optimal Transport.** ICLR, 2018. [P] [paper](https://arxiv.org/abs/1803.05573) [report](reports/1803.05573.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1803.05573.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1803.05573.zh.pdf)

    *Tim Salimans, Han Zhang, Alec Radford, Dimitris Metaxas*

    minibatch energy distance：primal OT 与对抗特征空间 energy distance 结合，梯度无偏、大 batch 下训练稳定

<a id="t05"></a>
### T05. Wasserstein 梯度流与 JKO 格式生成模型

课题综合：[`topics/t05.md`](topics/t05.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t05_wasserstein_gradient_flow.md`](source/kb/t05_wasserstein_gradient_flow.md)

1. ⭐ **Hessian-guided Perturbed Wasserstein Gradient Flows for Escaping Saddle Points.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2509.16974) [report](reports/2509.16974.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2509.16974.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2509.16974.zh.pdf)

    *Naoya Yamamoto, Juno Kim, Taiji Suzuki*

    PWGF：沿 Wasserstein Hessian 最小特征方向注入 GP 扰动逃离鞍点，测度空间非凸优化首个二阶最优性+多项式时间保证

2. ⭐ **Scalable Wasserstein Gradient Flow for Generative Modeling through Unbalanced Optimal Transport.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2402.05443) [report](reports/2402.05443.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.05443.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2402.05443.zh.pdf)

    *Jaemoo Choi, Jaewoong Choi, Myungjoo Kang*

    S-JKO：JKO 步 ↔ UOT 等价 → 半对偶形式把训练复杂度 \(O(K^2)\to O(K)\)，CIFAR-10 FID 2.62，WGF 生成模型首次逼近 SOTA

3. ⭐ **Normalizing flow neural networks by JKO scheme.** NeurIPS (Spotlight), 2023. [P] [paper](https://arxiv.org/abs/2212.14424) [report](reports/2212.14424.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2212.14424.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2212.14424.zh.pdf)

    *Chen Xu, Xiuyuan Cheng, Yao Xie*

    JKO-iFlow：每个残差块=一个 JKO 步，逐块训练 CNF，免 score matching / 端到端反传，省显存

4. ⭐ **Large-Scale Wasserstein Gradient Flows.** NeurIPS, 2021. [P] [paper](https://arxiv.org/abs/2106.00736) [report](reports/2106.00736.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2106.00736.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2106.00736.zh.pdf)

    *Petr Mokrov, Alexander Korotin, Lingxiao Li, Aude Genevay, Justin Solomon, Evgeny Burnaev*

    神经化 JKO 开山：Brenier 定理 + ICNN 参数化每个 JKO 步的凸势，SGD 免网格/免粒子求解

5. **Flowing Datasets with Wasserstein over Wasserstein Gradient Flows.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2506.07534) [report](reports/2506.07534.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.07534.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2506.07534.zh.pdf)

    *Clément Bonet, Christophe Vauthier, Anna Korba*

    把梯度流升到"测度的测度"（WoW）空间，对整个带标签数据集做流动（数据集级迁移/蒸馏）

6. **Importance Corrected Neural JKO Sampling.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2407.20444) [report](reports/2407.20444.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2407.20444.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2407.20444.zh.pdf)

    *Johannes Hertrich, Robert Gruhlke*

    CNF 实现的 JKO 局部步 + 重要性拒绝重采样非局部步，克服 WGF 采样的多峰质量错配，可产 iid 样本并评估密度

7. **Learning diffusion at lightspeed.** NeurIPS (Oral), 2024. [P] [paper](https://arxiv.org/abs/2406.12616) [report](reports/2406.12616.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.12616.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2406.12616.zh.pdf)

    *Antonio Terpin, Nicolas Lanzetti, Martin Gadea, Florian Dörfler*

    JKOnet*：用 JKO 步的一阶最优性条件替代双层优化，二次损失学 potential/interaction/internal 三种能量，线性参数化有闭式解

8. **Mirror and Preconditioned Gradient Descent in Wasserstein Space.** NeurIPS (Spotlight), 2024. [P] [paper](https://arxiv.org/abs/2406.08938) [report](reports/2406.08938.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.08938.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2406.08938.zh.pdf)

    *Clément Bonet, Théo Uscidda, Adam David, Pierre-Cyril Aubin-Frankowski, Anna Korba*

    把镜像下降/预条件梯度下降提升到 \(\mathcal{P}_2\)：相对光滑/凸下的收敛保证，病态目标与单细胞对齐实验

9. **Optimizing Functionals on the Space of Probabilities with ICNNs.** TMLR, 2022. [P] [paper](https://openreview.net/forum?id=dpOYN7o8Jm) [report](reports/Optimizing_Functionals_on_the_Space_of_Probabiliti.md)

    *Alvarez-Melis, Schiff, Mroueh*

    JKO-ICNN 框架：ICNN 逼近凸函数空间做 JKO，含收敛保证的泛函设计与分子受控生成

10. **Proximal Optimal Transport Modeling of Population Dynamics.** AISTATS, 2022. [P] [paper](https://arxiv.org/abs/2106.06345) [report](reports/2106.06345.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2106.06345.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2106.06345.zh.pdf)

    *Charlotte Bunne, Laetitia Meng-Papaxanthos, Andreas Krause, Marco Cuturi*

    JKOnet：反问题视角——从时序快照端到端学习驱动种群演化的能量泛函（单细胞应用）

11. **Variational inference via Wasserstein gradient flows.** NeurIPS, 2022. [P] [paper](https://arxiv.org/abs/2205.15902) [report](reports/2205.15902.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2205.15902.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2205.15902.zh.pdf)

    *Marc Lambert, Sinho Chewi, Francis Bach, Silvère Bonnabel, Philippe Rigollet*

    把 VI 写成 Bures–Wasserstein 子流形上的 WGF/JKO（Gaussian 与混合 Gaussian），log-concave 下有保证

12. **Variational Wasserstein gradient flow.** ICML, 2022. [P] [paper](https://arxiv.org/abs/2112.02424) [report](reports/2112.02424.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2112.02424.pdf)

    *Jiaojiao Fan, Qinsheng Zhang, Amirhossein Taghvaei, Yongxin Chen*

    用 f-divergence 的变分（对偶）形式替代显式密度项，primal-dual 求 JKO 步，可扩展到高维

13. **The Variational Formulation of the Fokker–Planck Equation.** SIAM J. Math. Anal., 1998. [P] [paper](https://doi.org/10.1137/S0036141096303359) [report](reports/The_Variational_Formulation_of_the_Fokker_Planck_E.md)

    *Jordan, Kinderlehrer, Otto*

    奠基：FPE = KL 的 \(W_2\) 梯度流，提出 JKO（minimizing movement）格式

另见（跨课题重复）：Wasserstein proximal operators describe score-based generative models and resolve memorization → T02; One-Step Generative Modeling via Wasserstein Gradient Flows → T04

<a id="t06"></a>
### T06. 扩散/流生成模型的收敛性与统计理论

课题综合：[`topics/t06.md`](topics/t06.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t06_convergence_statistics.md`](source/kb/t06_convergence_statistics.md)

1. ⭐ **Flow matching achieves almost minimax optimal convergence.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2405.20879) [report](reports/2405.20879.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.20879.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2405.20879.zh.pdf)

    *Kenji Fukumizu, Taiji Suzuki, Noboru Isobe, Kazusato Oko, Masanori Koyama*

    FM 在 p-Wasserstein (1≤p≤2) 下几乎 minimax，统计上与扩散等价；σ_t≍√t 的方差衰减是达到最优率的关键

2. ⭐ **O(d/T) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions.** ICLR；扩展版 JMLR 26(292), 2025. [P] [paper](https://arxiv.org/abs/2409.18959) [report](reports/2409.18959.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2409.18959.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2409.18959.zh.pdf)

    *Gen Li, Yuling Yan*

    仅需一阶矩有限 + L² score：DDPM 的 TV 收敛率 O(d/T)；系数设计得当可改进为 O(k/T)（k=内在维数）

3. ⭐ **Diffusion Models are Minimax Optimal Distribution Estimators.** ICML, 2023. [P] [paper](https://proceedings.mlr.press/v202/oko23a.html) [report](reports/Diffusion_Models_are_Minimax_Optimal_Distribution.md)

    *Oko, Akiyama, Suzuki*

    Besov 密度 + 经验 score matching：TV/W1 下近 minimax 最优，端到端统计理论开山之作

4. ⭐ **Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions.** ICLR oral, 2023. [P] [paper](https://arxiv.org/abs/2209.11215) [report](reports/2209.11215.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2209.11215.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2209.11215.zh.pdf)

    *Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, Anru R. Zhang*

    首个在 L² score 误差 + 任意非 log-concave 数据下的多项式收敛保证，奠定 Girsanov 分析范式

5. ⭐ **Minimax estimation of smooth optimal transport maps.** Ann. Statist. 49(2), 2021. [P] [paper](https://arxiv.org/abs/1905.05828) [report](reports/1905.05828.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1905.05828.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1905.05828.zh.pdf)

    *Jan-Christian Hütter, Philippe Rigollet*

    首个一般维度 OT map 的 minimax 率 n^{-2α/(2α-2+d)}（log 因子内），半对偶+小波估计器

6. **Denoising Diffusion Probabilistic Models Are Optimally Adaptive to Unknown Low Dimensionality.** Math. Oper. Res., 2026. [P] [paper](https://doi.org/10.1287/moor.2024.0769) [report](reports/Denoising_Diffusion_Probabilistic_Models_Are_Optim.md)

    *Huang, Wei, Chen*

    DDPM 无需知道 k 即自动以近 k-线性迭代复杂度收敛，且 KL 度量下最优

7. **A Sharp KL-Convergence Analysis for Diffusion Models under Minimal Assumptions.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2508.16306) [report](reports/2508.16306.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2508.16306.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2508.16306.zh.pdf)

    *Nishant Jain, Tong Zhang*

    「ODE 步+小加噪步」复合分析把 KL 迭代复杂度提到 Õ(d/ε)，无光滑假设下当前最优

8. **Linear Convergence of Diffusion Models Under the Manifold Hypothesis.** COLT, 2025. [P] [paper](https://arxiv.org/abs/2410.09046) [report](reports/2410.09046.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.09046.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2410.09046.zh.pdf)

    *Peter Potaptchik, Iskander Azangulov, George Deligiannidis*

    流形支撑数据下 KL 收敛步数对内在维数线性（log 因子内），且证明线性依赖 sharp

9. **Minimax Optimality of the Probability Flow ODE for Diffusion Models.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2503.09583) [report](reports/2503.09583.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.09583.pdf)

    *Changxiao Cai, Gen Li*

    首个确定性 ODE 采样器的端到端近 minimax 框架：光滑正则化 score 估计器同时控 L² 误差与 Jacobian 误差，绕开 Girsanov

10. **Optimal transport map estimation in general function spaces.** Ann. Statist. 53(3), 2025. [P] [paper](https://arxiv.org/abs/2212.03722) [report](reports/2212.03722.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2212.03722.pdf)

    *Vincent Divol, Jonathan Niles-Weed, Aram-Alexandre Pooladian*

    Poincaré 不等式+度量熵的统一估计框架，覆盖无限宽浅层网络 map 的首个统计率

11. **Error Bounds for Flow Matching Methods.** TMLR, 2024. [P] [paper](https://arxiv.org/abs/2305.16860) [report](reports/2305.16860.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2305.16860.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2305.16860.zh.pdf)

    *Joe Benton, George Deligiannidis, Arnaud Doucet*

    首批 FM 的 W2 误差界：L² 速度场误差 + 流的正则性假设（连续时间、不含离散化）

12. **Minimax Optimality of Score-based Diffusion Models: Beyond the Density Lower Bound Assumptions.** ICML spotlight, 2024. [P] [paper](https://arxiv.org/abs/2402.15602) [report](reports/2402.15602.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.15602.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2402.15602.zh.pdf)

    *Kaihong Zhang, Caitlyn H. Yin, Feng Liang, Jingbo Liu*

    截断核 score 估计器达最优 MSE，去掉密度下界假设后 β≤2 Sobolev 类仍 minimax

13. **Nearly $d$-Linear Convergence Bounds for Diffusion Models via Stochastic Localization.** ICLR spotlight, 2024. [P] [paper](https://arxiv.org/abs/2308.03686) [report](reports/2308.03686.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2308.03686.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2308.03686.zh.pdf)

    *Joe Benton, Valentin De Bortoli, Arnaud Doucet, George Deligiannidis*

    用随机局部化技巧把 KL 迭代复杂度降至 Õ(d/ε²)，仅需数据二阶矩有限

14. **Optimal score estimation via empirical Bayes smoothing.** COLT, 2024. [P] [paper](https://arxiv.org/abs/2402.07747) [report](reports/2402.07747.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.07747.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2402.07747.zh.pdf)

    *Andre Wibisono, Yihong Wu, Kaylee Yingxi Yang*

    确立 score 估计本身的 minimax 率 Θ̃(n^{-2/(d+4)})，正式坐实 score 学习的维数灾难

15. **Plugin Estimation of Smooth Optimal Transport Maps.** Ann. Statist. 52(3), 2024. [P] [paper](https://arxiv.org/abs/2107.12364) [report](reports/2107.12364.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2107.12364.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2107.12364.zh.pdf)

    *Tudor Manole, Sivaraman Balakrishnan, Jonathan Niles-Weed, Larry Wasserman*

    可计算的 plug-in 估计器（经验耦合+线性平滑/密度估计）同样 minimax 最优，并给出 W2² 的 CLT 推断

<a id="sec-b"></a>
## B. 流匹配与轨迹拉直

<a id="t07"></a>
### T07. Flow Matching 基础谱系

课题综合：[`topics/t07.md`](topics/t07.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t07_flow_matching_foundations.md`](source/kb/t07_flow_matching_foundations.md)

1. ⭐ **Generator Matching: Generative modeling with arbitrary Markov processes.** ICLR Oral, 2025. [P] [paper](https://arxiv.org/abs/2410.20587) [report](reports/2410.20587.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.20587.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2410.20587.zh.pdf)

    *Peter Holderrieth, Marton Havasi, Jason Yim, Neta Shaul, Itai Gat, Tommi Jaakkola et al.*

    最大一般化：用 Markov 生成元统一 FM/diffusion/离散 diffusion/jump 过程，支持模型叠加与多模态组合

2. ⭐ **Mean Flows for One-step Generative Modeling.** NeurIPS Oral, 2025. [P] [paper](https://arxiv.org/abs/2505.13447) [report](reports/2505.13447.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.13447.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2505.13447.zh.pdf)

    *Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He*

    用"平均速度场"替代瞬时速度，MeanFlow identity 直接从头训练一步生成，ImageNet-256 1-NFE FID 3.43

3. ⭐ **Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.** JMLR 26(209)（arXiv 2023）, 2025. [P] [paper](https://arxiv.org/abs/2303.08797) [report](reports/2303.08797.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2303.08797.pdf)

    *Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden*

    统一 flows/diffusions/SB：任意两分布的随机插值 + 可调扩散系数，ODE/SDE 采样二选一，含 likelihood 控制理论

4. ⭐ **Flow Matching Guide and Code.** arXiv 2412.06264, 2024. [B] [paper](https://arxiv.org/abs/2412.06264) [report](reports/2412.06264.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2412.06264.pdf)

    *Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer et al.*

    官方教科书级综述+PyTorch 库：统一记号覆盖连续/离散/流形/generator matching，训练与调度实践大全

5. ⭐ **Flow Matching for Generative Modeling.** ICLR, 2023. [P] [paper](https://arxiv.org/abs/2210.02747) [report](reports/2210.02747.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2210.02747.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2210.02747.zh.pdf)

    *Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le*

    奠基：conditional probability path + CFM 目标，simulation-free 训练 CNF，提出 Cond-OT 路径优于 diffusion 路径

6. **On the Guidance of Flow Matching.** ICML spotlight, 2025. [P] [paper](https://arxiv.org/abs/2502.02150) [report](reports/2502.02150.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2502.02150.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2502.02150.zh.pdf)

    *Ruiqi Feng, Chenglei Yu, Wenhao Deng, Peiyan Hu, Tailin Wu*

    首个通用 FM 引导框架：导出 training-free 渐近精确引导、训练式引导损失，经典梯度引导为特例

7. **Bespoke Solvers for Generative Flow Models.** ICLR spotlight, 2024. [P] [paper](https://arxiv.org/abs/2310.19075) [report](reports/2310.19075.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.19075.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2310.19075.zh.pdf)

    *Neta Shaul, Juan Perez, Ricky T. Q. Chen, Ali Thabet, Albert Pumarola, Yaron Lipman*

    为给定预训练流模型定制 ODE 求解器（约 80 个参数、1% 训练开销），低 NFE 采样大幅提质；后续 Non-Stationary 版本发表于 ICML 2024

8. **SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers.** ECCV, 2024. [P] [paper](https://arxiv.org/abs/2401.08740) [report](reports/2401.08740.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2401.08740.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2401.08740.zh.pdf)

    *Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, Saining Xie*

    用 DiT 骨干系统消融 interpolant/连续时间/速度参数化/采样器四个设计轴，同架构同算力全面超越 DiT，ImageNet-256 FID 2.06

9. **Stochastic interpolants with data-dependent couplings.** ICML spotlight, 2024. [P] [paper](https://arxiv.org/abs/2310.03725) [report](reports/2310.03725.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.03725.pdf)

    *Michael S. Albergo, Mark Goldstein, Nicholas M. Boffi, Rajesh Ranganath, Eric Vanden-Eijnden*

    把 base 分布条件化于目标数据（非 minibatch-OT 的耦合方式），一样的平方损失训练，用于超分/补全等条件生成

10. **Action Matching: Learning Stochastic Dynamics from Samples.** ICML, 2023. [P] [paper](https://arxiv.org/abs/2210.06662) [report](reports/2210.06662.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2210.06662.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2210.06662.zh.pdf)

    *Kirill Neklyudov, Rob Brekelmans, Daniel Severo, Alireza Makhzani*

    只用时间边缘快照学动力学：从 Benamou-Brenier 最小作用量出发，无需耦合样本或 OT 求解器，含 entropic/unbalanced 扩展

11. **Building Normalizing Flows with Stochastic Interpolants.** ICLR, 2023. [P] [paper](https://arxiv.org/abs/2209.15571) [report](reports/2209.15571.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2209.15571.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2209.15571.zh.pdf)

    *Michael S. Albergo, Eric Vanden-Eijnden*

    与 FM 同期独立提出插值式 simulation-free 训练（InterFlow），并给出最小化路径长度→OT map 的视角

12. **On Kinetic Optimal Probability Paths for Generative Models.** ICML, 2023. [P] [paper](https://arxiv.org/abs/2306.06626) [report](reports/2306.06626.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2306.06626.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2306.06626.zh.pdf)

    *Neta Shaul, Ricky T. Q. Chen, Maximilian Nickel, Matt Le, Yaron Lipman*

    噪声调度理论：在 Gaussian 路径族中求动能最优路径，证明 n/√d→0 时 Cond-OT 路径动能最优

13. **Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2303.00848) [report](reports/2303.00848.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2303.00848.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2303.00848.zh.pdf)

    *Diederik P. Kingma, Ruiqi Gao*

    统一视角：一切常用 diffusion/FM 加权目标 = 不同噪声级 ELBO 的加权积分，单调加权时等价于加噪数据增广下的 ELBO

另见（跨课题重复）：Error Bounds for Flow Matching Methods → T06; Flow matching achieves almost minimax optimal convergence → T06

<a id="t08"></a>
### T08. OT-CFM 与 minibatch OT 耦合

课题综合：[`topics/t08.md`](topics/t08.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t08_ot_cfm_minibatch.md`](source/kb/t08_ot_cfm_minibatch.md)

1. ⭐ **Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2603.15279) [report](reports/2603.15279.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2603.15279.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2603.15279.zh.pdf)

    *Aram Davtyan, Leello Tadesse Dadi, Volkan Cevher, Paolo Favaro*

    跨 minibatch 存储并交换局部最优配对、多噪声缓存防过拟合，以近零开销逼近全局 OT plan

2. ⭐ **On Fitting Flow Models with Large Sinkhorn Couplings.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2506.05526) [report](reports/2506.05526.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.05526.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2506.05526.zh.pdf)

    *Stephen Zhang, Alireza Mousavi-Hosseini, Michal Klein, Marco Cuturi*

    把 Sinkhorn 耦合分片扩到 n≈10⁶ 并系统消融 ε：n≈256 的 OT-FM 增益微弱是小样本诅咒，大 n+低 ε 才显著收益

3. ⭐ **The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation.** ICCV, 2025. [P] [paper](https://arxiv.org/abs/2503.10636) [report](reports/2503.10636.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.10636.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2503.10636.zh.pdf)

    *Ho Kei Cheng, Alexander Schwing*

    揭示无条件 OT 耦合在条件生成中反而有害（条件偏斜先验造成 train-test gap），在成本矩阵加条件加权项修复

4. ⭐ **Improving and generalizing flow-based generative models with minibatch optimal transport.** TMLR, 2024. [P] [paper](https://arxiv.org/abs/2302.00482) [report](reports/2302.00482.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2302.00482.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2302.00482.zh.pdf)

    *Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks et al.*

    提出 CFM 统一框架与 OT-CFM：batch 内 OT 重配对得到更直、更稳、可近似动态 OT 的流，且 source 不必是高斯

5. ⭐ **Multisample Flow Matching: Straightening Flows with Minibatch Couplings.** ICML, 2023. [P] [paper](https://arxiv.org/abs/2304.14772) [report](reports/2304.14772.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2304.14772.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2304.14772.zh.pdf)

    *Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, Ricky T. Q. Chen*

    形式化 batch 耦合族（BatchOT/BatchEOT/StableCoupling），证明 k→∞ 路径直线化、梯度方差降低，ImageNet 上省 30–60% NFE

6. **Expected Batch Optimal Transport Plans and Consequences for Flow Matching.** arXiv, 2026. [R] [paper](https://arxiv.org/abs/2605.12174) [report](reports/2605.12174.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2605.12174.pdf)

    *Samuel Boïté, Julie Delon, Kimia Nadjahi*

    首个系统理论：期望 batch 耦合 π_k 的大 batch 一致性、半离散情形成本偏差与 plan 收敛速率、FM 流的良定性

7. **Minibatch Optimal Transport and Perplexity Bound Estimation in Discrete Flow Matching.** ICML, 2026. [A] [paper](https://arxiv.org/abs/2411.00759) [report](reports/2411.00759.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.00759.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2411.00759.zh.pdf)

    *Etrit Haxholli, Yeti Z. Gurbuz, Ogul Can, Eli Waxman*

    把 minibatch OT 耦合引入离散 flow matching，分析耦合对离散路径与 perplexity 的影响

8. **Flow Matching with Semidiscrete Couplings.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2509.25519) [report](reports/2509.25519.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2509.25519.pdf)

    *Alireza Mousavi-Hosseini, Stephen Y. Zhang, Michal Klein, Marco Cuturi*

    绕开 minibatch：对整个（离散）数据集预计算半离散 OT 对偶势，训练时按势函数配对，含条件三角映射扩展

9. **Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2406.12303) [report](reports/2406.12303.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.12303.pdf)

    *Yiheng Li, Heyang Jiang, Akio Kodaira, Masayoshi Tomizuka, Kurt Keutzer, Chenfeng Xu*

    扩散侧的 batch 内就近噪声分配（量化线性分配，1024 batch 仅 22.8ms），训练加速最高 3×

10. **Metric Flow Matching for Smooth Interpolations on the Data Manifold.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2405.14780) [report](reports/2405.14780.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.14780.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2405.14780.zh.pdf)

    *Kacper Kapuśniak, Peter Potaptchik, Teodora Reu, Leo Zhang, Alexander Tong, Michael Bronstein et al.*

    非欧成本：插值改为数据依赖黎曼度量下的近似测地线（OT-MFM），路径贴合数据流形，单细胞轨迹 SOTA

11. **Simulation-free Schrödinger bridges via score and flow matching.** AISTATS, 2024. [P] [paper](https://arxiv.org/abs/2307.03672) [report](reports/2307.03672.md)

    *Alexander Tong, Nikolay Malkin, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang, Guillaume Huguet et al.*

    用静态熵正则 OT / minibatch Sinkhorn 耦合 + score+flow 双回归，simulation-free 求解 Schrödinger bridge

12. **Stochastic interpolants with data-dependent couplings.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2310.03725) [report](reports/2310.03725.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.03725.pdf)

    *Michael S. Albergo, Mark Goldstein, Nicholas M. Boffi, Rajesh Ranganath, Eric Vanden-Eijnden*

    把「耦合的选择」形式化为随机插值框架中的建模自由度，给出依赖数据/条件的耦合构造与理论

13. **Equivariant flow matching.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2306.15030) [report](reports/2306.15030.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2306.15030.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2306.15030.zh.pdf)

    *Leon Klein, Andreas Krämer, Frank Noé*

    成本设计开端:对多体系统用旋转（Kabsch）+置换（Hungarian）对齐后的不变成本做 batch OT，得到近似 OT 的等变流

14. **On Transportation of Mini-batches: A Hierarchical Approach.** ICML, 2022. [P] [paper](https://arxiv.org/abs/2102.05912) [report](reports/2102.05912.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2102.05912.pdf)

    *Khai Nguyen, Dang Nguyen, Quoc Nguyen, Tung Pham, Hung Bui, Dinh Phung et al.*

    BoMb-OT 在「batch 之间」再解一层 OT 以修正朴素平均的失真，m-OT 是其熵正则极限

15. **Learning with minibatch Wasserstein : asymptotic and gradient properties.** AISTATS, 2020. [P] [paper](https://arxiv.org/abs/1910.04091) [report](reports/1910.04091.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1910.04091.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/1910.04091.zh.pdf)

    *Kilian Fatras, Younes Zine, Rémi Flamary, Rémi Gribonval, Nicolas Courty*

    minibatch OT 的奠基分析：无偏梯度、维度无关集中界，但等价于隐式正则化、失去距离公理且产生错配

<a id="t09"></a>
### T09. Rectified Flow 与轨迹拉直

课题综合：[`topics/t09.md`](topics/t09.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t09_rectified_flow.md`](source/kb/t09_rectified_flow.md)

1. ⭐ **On the Relation between Rectified Flows and Optimal Transport.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2505.19712) [report](reports/2505.19712.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.19712.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2505.19712.zh.pdf)

    *Johannes Hertrich, Antonin Chambolle, Julie Delon*

    反例定论：迭代 rectification 存在**非最优不动点**、损失趋零不蕴含最优、梯度约束版等价定理（Liu22 Thm 5.6）需强得多的假设——reflow 不是可靠的 OT 求解器

2. ⭐ **Improving the Training of Rectified Flows.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2405.20320) [report](reports/2405.20320.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.20320.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2405.20320.zh.pdf)

    *Sangyun Lee, Zinan Lin, Giulia Fanti*

    实证现实设置下**一轮 reflow 即近乎直**；U 形时间步分布+LPIPS-Huber 前度量，1-NFE FID 最高改善 75%，ImageNet64 上超 CD/PD

3. ⭐ **InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2309.06380) [report](reports/2309.06380.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2309.06380.pdf)

    *Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu*

    首个 SD 级一步文生图：文本条件 reflow 改善噪声-图像耦合后蒸馏，COCO-5k FID 23.3（199 A100 天）

4. ⭐ **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.** ICML Oral, 2024. [P] [paper](https://arxiv.org/abs/2403.03206) [report](reports/2403.03206.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.03206.pdf)

    *Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini et al.*

    大规模对照研究证明 RF+logit-normal 时间步采样优于既有扩散公式；MMDiT 架构；RF 由此进入工业主流

5. ⭐ **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.** ICLR, 2023. [P] [paper](https://arxiv.org/abs/2209.03003) [report](reports/2209.03003.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2209.03003.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2209.03003.zh.pdf)

    *Xingchao Liu, Chengyue Gong, Qiang Liu*

    奠基作：线性插值+因果化定义 RF，reflow 迭代拉直；证明凸代价单调不增、直线度以 O(1/K) 速率下降

6. **FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2506.15742) [report](reports/2506.15742.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.15742.pdf)

    *Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne et al.*

    12B rectified flow transformer 的工业实践：logit-normal shift 调度（μ 随分辨率调整）、双流/单流混合 DiT、少步化走 LADD 对抗蒸馏而非 reflow

7. **Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow.** ICLR, 2025. [P] [paper](https://openreview.net/forum?id=nEDToD1R8M) [report](reports/Rectified_Diffusion_Straightness_Is_Not_Your_Need.md)

    *Wang et al.*

    论证 rectification 的本质是"预训练模型配对+重训"而非直线度/流匹配形式/v-预测；推广为一般扩散的一阶 ODE 目标

8. **Straighten Viscous Rectified Flow via Noise Optimization.** ICCV, 2025. [P] [paper](https://arxiv.org/abs/2507.10218) [report](reports/2507.10218.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2507.10218.pdf)

    *Jimin Dai, Jiexi Yan, Jian Yang, Lei Luo*

    指出 reflow 合成耦合与真实图像存在分布差距；历史速度项+噪声再参数化优化，直接与**真实图像**构造耦合来拉直

9. **Towards Hierarchical Rectified Flow.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2502.17436) [report](reports/2502.17436.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2502.17436.pdf)

    *Yichi Zhang, Yici Yan, Alex Schwing, Zhizhen Zhao*

    层级耦合位置/速度/加速度多条 ODE，建模多模态随机速度场，允许积分路径相交从而更直、更少 NFE

10. **Bellman Optimal Stepsize Straightening of Flow-Matching Models.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2312.16414) [report](reports/2312.16414.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2312.16414.pdf)

    *Bao Nguyen, Binh Nguyen, Viet Anh Nguyen*

    BOSS：动态规划求最优步长序列再按其重训速度场，低资源（可仅 LoRA 2% 参数）下优于标准 reflow

11. **Constant Acceleration Flow.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2411.00322) [report](reports/2411.00322.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.00322.pdf)

    *Dogyun Park, Sojin Lee, Sihyeon Kim, Taehoon Lee, Youngjoon Hong, Hyunwoo J. Kim*

    放弃常速假设改学常加速度方程（初速度条件化+初速度 reflow），一步生成与耦合保持/反演精度双改进

12. **On the Convergence and Straightness of Rectified Flow.** 2024–26 · arXiv, 2024. [R] [paper](https://arxiv.org/abs/2410.14949) [report](reports/2410.14949.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.14949.pdf)

    *Vansh Bansal, Saptarshi Roy, Alessandro Rinaldo, Purnamrita Sarkar*

    正面理论：W2² 误差界由（分段）直线度参数+离散步数刻画，给出 1-RF 唯一且直的充分条件；1D 高斯出发时 RF 即 Monge 映射

13. **PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2405.07510) [report](reports/2405.07510.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.07510.pdf)

    *Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, Jiashi Feng*

    分时间窗做分段 reflow，免去整条 ODE 轨迹仿真、可在线训练；ΔW 即插即用加速整个 SD 生态

14. **SlimFlow: Training Smaller One-Step Diffusion Models with Rectified Flow.** ECCV, 2024. [P] [paper](https://arxiv.org/abs/2407.12718) [report](reports/2407.12718.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2407.12718.pdf)

    *Yuanzhi Zhu, Xingchao Liu, Qiang Liu*

    Annealing Reflow 解决大师小徒初始化失配 + Flow-Guided Distillation，15.7M 参数一步 FID 5.02（CIFAR-10）

15. **Rectified Flow: A Marginal Preserving Approach to Optimal Transport.** arXiv, 2022. [R] [paper](https://arxiv.org/abs/2209.14577) [report](reports/2209.14577.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2209.14577.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2209.14577.zh.pdf)

    *Qiang Liu*

    理论姊妹篇：rectification 同时降低一切凸代价；提出 c-rectified flow 声称可逼近特定代价 OT（后被 Hertrich 等指出需更强假设）

<a id="t10"></a>
### T10. 一致性模型与少步蒸馏的 OT 视角

课题综合：[`topics/t10.md`](topics/t10.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t10_consistency_distillation_ot.md`](source/kb/t10_consistency_distillation_ot.md)

1. ⭐ **VDOT: Efficient Unified Video Creation via Optimal Transport Distillation.** CVPR, 2026. [P] [paper](https://arxiv.org/abs/2512.06802) [report](reports/2512.06802.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2512.06802.pdf)

    *Yutong Wang, Haiyu Zhang, Tianfan Xue, Yu Qiao, Yaohui Wang, Chang Xu et al.*

    首次把熵正则 OT 距离引入 DMD（替代/增强 KL）：OT plan 给分布匹配加几何约束，缓解 few-step 场景的 zero-forcing 与梯度崩塌；4 步统一视频生成匹敌 50-100 步教师，并发布 UVCBench

2. ⭐ **Flow Map Matching.** TMLR, 2025. [P] [paper](https://openreview.net/forum?id=cqDH0e6ak2) [report](reports/Flow_Map_Matching.md)

    *FMM*

    两时间 flow map 统一框架：证明 Lagrangian/Eulerian 蒸馏损失上界控制教师-学生 W2 距离，且 Eulerian 损失是一致性蒸馏的连续时间极限，统一 CM/CTM/渐进蒸馏

3. ⭐ **Improving Consistency Models with Generator-Augmented Flows.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2406.09570) [report](reports/2406.09570.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.09570.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2406.09570.zh.pdf)

    *Thibaut Issenhuth, Sangchul Lee, Ludovic Dos Santos, Jean-Yves Franceschi, Chansoo Kim, Alain Rakotomamonjy*

    证明一致性训练与蒸馏的差异在连续时间极限仍不消失；用 generator 诱导的流/耦合同时降低该差异与噪声-数据传输成本，加速收敛并提升质量

4. ⭐ **One-step Diffusion with Distribution Matching Distillation.** CVPR, 2024. [P] [paper](https://arxiv.org/abs/2311.18828) [report](reports/2311.18828.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2311.18828.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2311.18828.zh.pdf)

    *Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman et al.*

    反向 KL 梯度 = 真/假 score 之差；用教师 ODE 噪声-图像对的回归损失（确定性耦合锚定）防模式坍缩

5. ⭐ **Consistency Models.** ICML, 2023. [P] [paper](https://arxiv.org/abs/2303.01469) [report](reports/2303.01469.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2303.01469.pdf)

    *Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever*

    奠基：学习 PF-ODE 的自一致映射（任意轨迹点→端点），CD 蒸馏 CIFAR-10 一步 FID 3.55、两步 2.93

6. **One Step Diffusion via Shortcut Models.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2410.12557) [report](reports/2410.12557.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.12557.pdf)

    *Kevin Frans, Danijar Hafner, Sergey Levine, Pieter Abbeel*

    把步长 d 作为网络条件输入，自蒸馏只需 log2(T) 次 bootstrap；单网络单阶段支持任意步数预算（ImageNet-256 DiT-XL：1 步 10.6 / 4 步 7.8 / 128 步 3.8）

7. **Simplifying, Stabilizing and Scaling Continuous-time CMs (sCM).** ICLR (Oral), 2025. [P] [paper](https://iclr.cc/virtual/2025/oral/31868) [report](reports/Simplifying_Stabilizing_and_Scaling_Continuous_tim.md)

    TrigFlow 统一参数化 + 连续时间训练稳定化，1.5B 参数两步 FID：CIFAR-10 2.06 / ImageNet-64 1.48 / ImageNet-512 1.88

8. **Towards a mathematical theory for consistency training in diffusion models.** AISTATS, 2025. [P] [paper](https://arxiv.org/abs/2402.07802) [report](reports/2402.07802.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.07802.pdf)

    *Gen Li, Zhihan Huang, Yuting Wei*

    证明一致性训练步数超过 O(d^{5/2}/ε) 即可生成 Wasserstein 意义下 ε-接近目标的样本，给出离散化-精度定量关系

9. **Adversarial Diffusion Distillation.** ECCV (Oral), 2024. [P] [paper](https://arxiv.org/abs/2311.17042) [report](reports/2311.17042.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2311.17042.pdf)

    *Axel Sauer, Dominik Lorenz, Andreas Blattmann, Robin Rombach*

    score distillation + hinge-GAN 判别器，SDXL-Turbo 的基础，1-4 步实时生成、单步胜过 LCM

10. **Adversarial Score Distillation: When score distillation meets GAN.** CVPR, 2024. [P] [paper](https://arxiv.org/abs/2312.00739) [report](reports/2312.00739.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2312.00739.pdf)

    *Min Wei, Jingkai Zhou, Junyao Sun, Xuesong Zhang*

    用 WGAN 范式重推 SDS/VSD：SDS=固定次优判别器、VSD=不完整判别器优化；补全 W1 对偶判别器训练解决 CFG 尺度敏感

11. **Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2310.02279) [report](reports/2310.02279.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.02279.pdf)

    *Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka et al.*

    推广为任意时刻→任意时刻的两时间轨迹映射，可同时输出 score；GAN 辅助下一步 FID 1.73 (CIFAR-10)/1.92 (ImageNet-64)，γ-sampling 提供质量-步数连续旋钮

12. **DMD2: Improved Distribution Matching Distillation.** NeurIPS (Oral), 2024. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html) [report](reports/DMD2_Improved_Distribution_Matching_Distillation.md)

    去掉 ODE 对回归，TTUR + GAN（真数据）稳定纯分布匹配；一步 ImageNet-64 FID 1.28 超越教师

13. **Improved Techniques for Training Consistency Models.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2310.14189) [report](reports/2310.14189.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.14189.pdf)

    *Yang Song, Prafulla Dhariwal*

    去 EMA teacher + Pseudo-Huber 损失，免蒸馏一致性训练一步 FID 2.51 (CIFAR-10)，首次反超蒸馏

14. **Score identity Distillation.** ICML, 2024. [P] [paper](https://proceedings.mlr.press/v235/zhou24x.html) [report](reports/Score_identity_Distillation.md)

    *SiD*

    三个 score 恒等式构造 data-free 蒸馏损失，FID 指数速率下降、逼近甚至超过教师

15. **Theory of Consistency Diffusion Models: Distribution Estimation Meets Fast Sampling.** ICML, 2024. [P] [paper](https://proceedings.mlr.press/v235/dou24a.html) [report](reports/Theory_of_Consistency_Diffusion_Models.md)

    首个 CM 统计理论：把训练形式化为分布差异最小化，给出 Wasserstein 距离下的估计率（与原扩散模型同阶），同时覆盖蒸馏与免蒸馏两种训练

另见（跨课题重复）：Mean Flows for One-step Generative Modeling → T07

<a id="t11"></a>
### T11. 免训练采样器与 ODE 求解器

课题综合：[`topics/t11.md`](topics/t11.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t11_fast_ode_solvers.md`](source/kb/t11_fast_ode_solvers.md)

1. ⭐ **Fast ODE-based Sampling for Diffusion Models in Around 5 Steps.** CVPR (Highlight), 2024. [P] [paper](https://arxiv.org/abs/2312.00094) [report](reports/2312.00094.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2312.00094.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2312.00094.zh.pdf)

    *Zhenyu Zhou, Defang Chen, Can Wang, Chun Chen*

    观察到采样轨迹几乎躺在 2D 子空间 → 由中值定理学习「平均方向」消截断误差，~5 NFE 采样；AMED-Plugin 可插任意求解器（轻量训练，solver 蒸馏边界情形）

2. ⭐ **Align Your Steps: Optimizing Sampling Schedules in Diffusion Models.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2404.14507) [report](reports/2404.14507.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2404.14507.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2404.14507.zh.pdf)

    *Amirmojtaba Sabour, Sanja Fidler, Karsten Kreis*

    首个原理化调度优化框架：用 Girsanov 定理导出真实与线性化生成 SDE 间 KL 上界（KLUB），求 model×solver×dataset 专属最优调度，few-step 域普适增益

3. ⭐ **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.** NeurIPS, 2022. [P] [paper](https://arxiv.org/abs/2206.00927) [report](reports/2206.00927.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2206.00927.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2206.00927.zh.pdf)

    *Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu*

    利用扩散 ODE 半线性结构的定制指数积分器 + log-SNR 换元，线性项解析解出、只近似神经网络项，10–20 NFE 高质量采样并有收敛阶证明

4. ⭐ **Elucidating the Design Space of Diffusion-Based Generative Models.** NeurIPS, 2022. [P] [paper](https://arxiv.org/abs/2206.00364) [report](reports/2206.00364.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2206.00364.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2206.00364.zh.pdf)

    *Tero Karras, Miika Aittala, Timo Aila, Samuli Laine*

    用 σ 参数化统一各家训练/采样设计空间：ρ=7 时间调度、Heun 二阶格式、churn 随机性，至今仍是 few-NFE 采样的默认骨架与基准

5. **TJS: x-Prediction Is All You Need（端点可解码性）.** arXiv, 2026. [R] [report](reports/TJS_x_Prediction_Is_All_You_Need.md)

    形式化「端点可解码性」：中间态+路径速度即 E[x₀\

6. **Analyzing and Improving Fast Sampling of Text-to-Image Diffusion Models.** arXiv, 2026. [R] [paper](https://arxiv.org/abs/2603.00763) [report](reports/2603.00763.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2603.00763.pdf)

    *Zhenyu Zhou, Defang Chen, Siwei Lyu, Chun Chen, Can Wang*

    系统消融 T2I 免训练设计空间，发现**时间调度是最关键因子**；由 Frenet-Serret 公式导出「恒定总旋转」调度，Flux.1/SD3.5 十步高质量

7. **Parallel Diffusion Solver via Residual Dirichlet Policy Optimization.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2512.22796) [report](reports/2512.22796.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2512.22796.pdf)

    *Ruoyu Wang, Ziyu Li, Beier Zhu, Liangyu Yuan, Hanwang Zhang, Xun Yang et al.*

    每步引入多条**并行**梯度方向的加权集成（向量值中值定理），蒸馏 + Dirichlet 策略 RL 学权重，以并行换低延迟压制高曲率截断误差

8. **F-scheduler: illuminating the free-lunch design space for fast sampling of diffusion models.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2510.02390) [report](reports/2510.02390.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2510.02390.pdf)

    *Zilai Li, Lujia Bai*

    免训练插件宣称 6 步采 1024² 图超过蒸馏 SOTA；从信息论角度分析免训练求解器 vs 蒸馏模型的能力边界

9. **Geometric Regularity in Deterministic Sampling Dynamics of Diffusion-based Generative Models.** arXiv（作者称已被 J. Stat. Mech. 接收）, 2025. [R] [paper](https://arxiv.org/abs/2506.10177) [report](reports/2506.10177.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.10177.pdf)

    *Defang Chen, Zhenyu Zhou, Can Wang, Siwei Lyu*

    KDE 视角给出去噪轨迹闭式解（= 时变带宽 mean-shift），解释逐步旋转、似然单调上升与「线性–非线性–线性」全局模式

10. **Learning to Discretize Denoising Diffusion ODEs.** ICLR (Oral), 2025. [P] [paper](https://arxiv.org/abs/2405.15506) [report](reports/2405.15506.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.15506.pdf)

    *Vinh Tong, Hoang Trung-Dung, Anji Liu, Guy Van den Broeck, Mathias Niepert*

    通过学生求解器可微反传端点对齐误差来**学习**离散化，单 GPU 5–10 分钟训练；10 NFE 达 FID 2.38（CIFAR-10），4 NFE 从 35.04 → 9.31

11. **Optimal Stepsize for Diffusion Sampling.** arXiv（ICLR 2026 在审）, 2025. [R] [paper](https://arxiv.org/abs/2503.21774) [report](reports/2503.21774.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.21774.pdf)

    *Jianning Pei, Han Hu, Shuyang Gu*

    步长调度 = 递归误差最小化的最优子结构 → DP 提取全局最优调度，跨架构/求解器/噪声调度鲁棒；T2I 10× 加速保留 99.4% GenEval

12. **PFDiff: Training-Free Acceleration of Diffusion Models Combining Past and Future Scores.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2408.08822) [report](reports/2408.08822.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2408.08822.pdf)

    *Guangyi Wang, Yuren Cai, Lijiang Li, Wei Peng, Songzhi Su*

    完全免训练的跳步策略：复用过去 score 预测「跳板」+ Nesterov 式前瞻更新校正一阶离散误差，正交叠加于现有求解器；DDIM 基线 4 NFE 由 138.81 → 16.46 FID

13. **S4S: Solving for a Fast Diffusion Model Solver.** ICML, 2025. [P] [paper](https://icml.cc/virtual/2025/poster/46229) [report](reports/S4S_Solving_for_a_Fast_Diffusion_Model_Solver.md)

    论证低 NFE 下逐点跟踪真 ODE 轨迹在原理上不可行 → 黑盒学习求解器系数（S4S）与离散化（S4S-Alt）以对齐教师端点；5 NFE 达 CIFAR-10 FID 3.73

14. **Accelerating Diffusion Sampling with Optimized Time Steps.** CVPR, 2024. [P] [paper](https://arxiv.org/abs/2402.17376) [report](reports/2402.17376.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.17376.pdf)

    *Shuchen Xue, Zhaoqiang Liu, Fei Chen, Shifeng Zhang, Tianyang Hu, Enze Xie et al.*

    把调度选取写成最小化全局离散误差的约束优化（信赖域法，<15 秒求解），与 UniPC/DPM-Solver++ 即插即用

15. **Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models.** ICLR spotlight / 2024·ICML, 2024. [P] [paper](https://arxiv.org/abs/2403.01329) [report](reports/2403.01329.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.01329.pdf)

    *Neta Shaul, Uriel Singer, Ricky T. Q. Chen, Matthew Le, Ali Thabet, Albert Pumarola et al.*

    solver 蒸馏路线的奠基：为给定预训练流/扩散模型定制仅 80–200 参数的（非平稳）求解器，证明 NS 族涵盖既有数值格式；16 NFE 达 PSNR 45/FID 1.76（ImageNet-64）

16. **Fast Sampling of Diffusion Models with Exponential Integrator.** ICLR, 2023. [P] [paper](https://arxiv.org/abs/2204.13902) [report](reports/2204.13902.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2204.13902.pdf)

    *Qinsheng Zhang, Yongxin Chen*

    与 DPM-Solver 同期独立提出指数积分器，配 Adams-Bashforth 型多项式外推；其 iPNDM 变体至今仍是低 NFE 强基线

17. **DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2310.13268) [report](reports/2310.13268.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.13268.pdf)

    *Kaiwen Zheng, Cheng Lu, Jianfei Chen, Jun Zhu*

    引入经验模型统计量（EMS）在线选择最优参数化系数，最小化一阶离散误差，5–10 NFE 进一步提升

18. **UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2302.04867) [report](reports/2302.04867.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2302.04867.pdf)

    *Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu*

    统一任意阶预测-校正框架（UniP+UniC），校正器无额外 NFE 即提升低步数精度，diffusers 生态默认求解器之一

19. **DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models.** arXiv → 2025·Mach. Intell. Res. 22(4):730-751, 2022. [P] [paper](https://arxiv.org/abs/2211.01095) [report](reports/2211.01095.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2211.01095.pdf)

    *Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu*

    改用 data-prediction 参数化 + 多步格式 + thresholding，解决大 guidance 尺度下高阶求解器的失稳，成为文生图部署标准

20. **Denoising Diffusion Implicit Models.** ICLR, 2021. [P] [paper](https://arxiv.org/abs/2010.02502) [report](reports/2010.02502.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2010.02502.pdf)

    *Jiaming Song, Chenlin Meng, Stefano Ermon*

    把 DDPM 采样确定化为非马尔可夫隐式过程（PF-ODE 的一阶指数式离散），首个 10–50 步实用采样器，并给出确定性 encode–decode 映射

另见（跨课题重复）：Understanding DDPM Latent Codes Through Optimal Transport → T02; On the Trajectory Regularity of ODE-based Diffusion Sampling → T02

<a id="t12"></a>
### T12. 推理阶段的 OT 对齐与噪声-样本耦合

课题综合：[`topics/t12.md`](topics/t12.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t12_inference_time_ot_alignment.md`](source/kb/t12_inference_time_ot_alignment.md)

1. ⭐ **Golden Noise for Diffusion Models: A Learning Framework.** ICCV, 2025. [P] [paper](https://arxiv.org/abs/2411.09502) [report](reports/2411.09502.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.09502.pdf)

    *Zikai Zhou, Shitong Shao, Lichen Bai, Shufei Zhang, Zhiqiang Xu, Bo Han et al.*

    学一个噪声→"黄金噪声"的传输网络：re-denoise 采样+偏好模型筛选构建 10 万对噪声数据集，SVD 结构先验的小网络即插即用（+3% 开销），跨模型/采样器泛化

2. ⭐ **Scaling Inference Time Compute for Diffusion Models (Inference-Time Scaling beyond Denoising Steps).** CVPR, 2025. [P] [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.html) [report](reports/Scaling_Inference_Time_Compute_for_Diffusion_Model.md)

    *Ma et al.*

    把"找好噪声"形式化为验证器×搜索算法的设计空间（random / zero-order / search-over-paths），确立噪声搜索作为扩散 test-time scaling 的第二轴

3. ⭐ **Solving Prior Distribution Mismatch in Diffusion Models via Optimal Transport.** arXiv, 2024. [R] [paper](https://arxiv.org/abs/2410.13431) [report](reports/2410.13431.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.13431.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2410.13431.zh.pdf)

    *Zhanpeng Wang, Shenghao Li, Jiameng Che, Chen Wang, Shangling Jui, Na Lei et al.*

    证明扩散两阶段本质是计算时变 OT、概率流指数收敛到 Monge–Ampère 解的梯度；据此用**半离散静态 OT（Brenier 势的几何变分解）**桥接 \(p_\infty\to p_{T'}\) 消除 prior error，实现"一步 OT + 短程扩散"加速采样

4. **A Noise is Worth Diffusion Guidance.** ICLR, 2026. [A] [paper](https://arxiv.org/abs/2412.03895) [report](reports/2412.03895.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2412.03895.pdf)

    *Donghoon Ahn, Jiwon Kang, Sanghyun Lee, Jaewon Min, Minjae Kim, Wooseok Jang et al.*

    把 CFG 折叠进初值：学噪声→"guidance-free 噪声"的一次映射，低频小幅分量替代引导，同管线免 CFG 高质量生成、吞吐/显存双省

5. **Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise.** CVPR Oral, 2025. [P] [paper](https://arxiv.org/abs/2501.08331) [report](reports/2501.08331.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2501.08331.pdf)

    *Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He et al.*

    实时噪声 warp 算法（逐帧迭代传输、保空间高斯性），把运动控制变成"换结构化初值"，模型结构与训练管线零改动

6. **Improved Immiscible Diffusion: Accelerate Diffusion Training by Reducing Its Miscibility.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2505.18521) [report](reports/2505.18521.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.18521.pdf)

    *Yiheng Li, Feng Liang, Dan Kondratyuk, Masayoshi Tomizuka, Kurt Keutzer, Chenfeng Xu*

    把概念推广为"任意层的可混性降低"：KNN 噪声选择（O(n)、0.2ms/256batch）、image scaling 等实现族，>4× 加速；证明去噪双射性故不损多样性，给出 OT 助益扩散的新解释

7. **The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation.** ICCV Highlight, 2025. [P] [paper](https://arxiv.org/abs/2412.05101) [report](reports/2412.05101.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2412.05101.pdf)

    *Ruoyu Wang, Huayang Huang, Ye Zhu, Olga Russakovsky, Yu Wu*

    免优化检索式耦合：离线构建 10 万噪声库（键=无条件生成后验的语义/低层特征），推理 0.2ms 查库选起点，跨 T2I 模型零样本迁移

8. **How I Warped Your Noise: a Temporally-Correlated Noise Prior for Diffusion Models.** ICLR Oral, 2024. [P] [paper](https://arxiv.org/abs/2504.03072) [report](reports/2504.03072.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2504.03072.pdf)

    *Pascal Chang, Jingwei Tang, Markus Gross, Vinicius C. Azevedo*

    噪声-噪声耦合的理论工具：∫-noise 积分噪声表示+噪声传输方程，沿光流保分布地 warp 高斯噪声，免训练提升视频时间一致性

9. **Improving Diffusion-Based Generative Models via Approximated Optimal Transport.** arXiv, 2024. [R] [paper](https://arxiv.org/abs/2403.05069) [report](reports/2403.05069.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.05069.pdf)

    *Daegyu Kim, Jooyoung Choi, Chaehun Shin, Uiwon Hwang, Sungroh Yoon*

    EDM 侧等价做法：训练时用近似 OT 为每张图选噪声，ODE 轨迹曲率降低，CIFAR-10 达 FID 1.68/1.58@29NFE（自称 IJCAI 2024，官方 proceedings 未核验）

10. **InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization.** CVPR, 2024. [P] [paper](https://arxiv.org/abs/2404.04650) [report](reports/2404.04650.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2404.04650.pdf)

    *Xiefan Guo, Jinlin Liu, Miaomiao Cui, Jiankai Li, Hongyu Yang, Di Huang*

    起点优化开山：以交叉/自注意力目标在"有效区域"内梯度优化初始噪声（含高斯性约束），治 subject mixing/neglect

11. **ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2406.04312) [report](reports/2406.04312.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.04312.pdf)

    *Luca Eyring, Shyamgopal Karthik, Karsten Roth, Alexey Dosovitskiy, Zeynep Akata*

    对一步模型以人类偏好 reward 梯度上升优化初始噪声（50 步、20–50s），一步模型反超 SDXL、比肩 SD3

12. **The Emergence of Reproducibility and Consistency in Diffusion Models.** ICML, 2024. [P] [paper](https://proceedings.mlr.press/v235/zhang24cn.html) [report](reports/The_Emergence_of_Reproducibility_and_Consistency_i.md)

    *Zhang et al.*

    同一初始噪声+确定性采样器下，不同框架/架构/训练的模型输出几乎相同——噪声↔样本耦合是数据内在的、可跨模型复用的对象

另见（跨课题重复）：Understanding DDPM Latent Codes Through Optimal Transport → T02; The Flow Map of the Fokker–Planck Equation Does Not Provide Optimal Transport → T02; Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment → T08

<a id="sec-c"></a>
## C. 跨域生成与翻译

<a id="t13"></a>
### T13. 神经 OT 映射与无配对图像翻译

课题综合：[`topics/t13.md`](topics/t13.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t13_neural_ot_translation.md`](source/kb/t13_neural_ot_translation.md)

1. ⭐ **ENOT: Expectile Regularization for Fast and Accurate Training of Neural Optimal Transport.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2403.03777) [report](reports/2403.03777.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.03777.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2403.03777.zh.pdf)

    *Nazar Buzun, Maksim Bobrin, Dmitry V. Dylov*

    用 expectile 回归正则近似共轭（c-transform）算子，替代昂贵不稳的内环优化，W2 基准上速度/精度大幅提升

2. ⭐ **Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2305.14777) [report](reports/2305.14777.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2305.14777.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2305.14777.zh.pdf)

    *Jaemoo Choi, Jaewoong Choi, Myungjoo Kang*

    UOT 半对偶的生成模型：φ-divergence 松弛边际约束，outlier 稳健、训练稳、收敛快（CIFAR-10 FID 2.97）

3. ⭐ **Neural Optimal Transport.** ICLR Spotlight, 2023. [P] [paper](https://arxiv.org/abs/2201.12220) [report](reports/2201.12220.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2201.12220.pdf)

    *Alexander Korotin, Daniil Selikhanovych, Evgeny Burnaev*

    统一 strong/weak cost 的 saddle-point 求解器，证明 NN 是 transport plan 的万能逼近器；one-to-one 与 one-to-many unpaired 翻译

4. ⭐ **Generative Modeling with Optimal Transport Maps.** ICLR, 2022. [P] [paper](https://arxiv.org/abs/2110.02999) [report](reports/2110.02999.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2110.02999.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2110.02999.zh.pdf)

    *Litu Rout, Alexander Korotin, Evgeny Burnaev*

    首次在高维 ambient 图像空间把 W2 map 本身当生成器；min-max 算法 + 基于 duality gap 的误差界；unpaired 去噪/上色/补全

5. ⭐ **Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark.** NeurIPS, 2021. [P] [paper](https://arxiv.org/abs/2106.01954) [report](reports/2106.01954.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2106.01954.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2106.01954.zh.pdf)

    *Alexander Korotin, Lingxiao Li, Aude Genevay, Justin Solomon, Alexander Filippov, Evgeny Burnaev*

    用 ICNN 构造有解析 ground-truth OT map 的连续分布对（含图像空间），系统评测 W2 求解器，揭示"下游表现好 ≠ map 准"

6. **Conditional Unbalanced Optimal Transport Maps: An Outlier-Robust Framework for Conditional Generative Modeling.** arXiv (2026-03), 2026. [R] [paper](https://arxiv.org/abs/2603.06972) [report](reports/2603.06972.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2603.06972.pdf)

    *Jiwoo Yoon, Kyumin Choi, Jaewoong Choi*

    条件 UOT 的 dual/semi-dual 形式 + 三角 c-transform 参数化，outlier-robust 的条件生成模型

7. **Improving Neural Optimal Transport via Displacement Interpolation.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2410.03783) [report](reports/2410.03783.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.03783.pdf)

    *Jaemoo Choi, Yongxin Chen, Jaewoong Choi*

    导出 displacement interpolation 逐时刻对偶并证明跨时刻关联，用全轨迹 + HJB 正则稳定 max-min；I2I 翻译 FID 显著改善

8. **Overcoming Spurious Solutions in Semi-Dual Neural Optimal Transport.** ICML, 2025. [P] [paper](https://proceedings.mlr.press/v267/choi25a.html) [report](reports/Overcoming_Spurious_Solutions_in_Semi_Dual_Neural.md)

    *OTP*

    给出 semi-dual max-min 恢复真 OT map 的充分条件；源分布平滑化 + 渐退火学 OT plan，可学随机映射（one-to-many 上色）

9. **Analyzing and Improving Optimal-Transport-based Adversarial Networks.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2310.02611) [report](reports/2310.02611.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.02611.pdf)

    *Jaemoo Choi, Jaewoong Choi, Myungjoo Kang*

    统一 OT-based GAN 框架逐组件分析；divergence 调度（τ 渐增）解决 UOTM 超参敏感，FID 2.51/CIFAR-10

10. **Neural Optimal Transport with General Cost Functionals.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2205.15403) [report](reports/2205.15403.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2205.15403.pdf)

    *Arip Asadulaev, Alexander Korotin, Vage Egiazarian, Petr Mokrov, Evgeny Burnaev*

    cost 从点对点函数推广到一般泛函，支持 class-guided、pair-guided 等任务先验的可控翻译

11. **Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2311.15100) [report](reports/2311.15100.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2311.15100.pdf)

    *Luca Eyring, Dominik Klein, Théo Uscidda, Giovanni Palla, Niki Kilbertus, Zeynep Akata et al.*

    证明 unbalanced Monge map = 两个重缩放测度间的 balanced map，可插入任意估计器（含 OT-FM）；确立 UOT-FM 为 unpaired 翻译的原则性方法

12. **Extremal Domain Translation with Neural Optimal Transport.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2301.12874) [report](reports/2301.12874.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2301.12874.pdf)

    *Milena Gazdieva, Alexander Korotin, Daniil Selikhanovych, Evgeny Burnaev*

    提出 extremal transport：翻译保真度的理论最优形式化，用 incomplete transport（partial OT 特例）的极限逼近 ET map

13. **Kernel Neural Optimal Transport.** ICLR, 2023. [P] [paper](https://arxiv.org/abs/2205.15269) [report](reports/2205.15269.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2205.15269.pdf)

    *Alexander Korotin, Daniil Selikhanovych, Evgeny Burnaev*

    证明 γ-weak quadratic cost 的 NOT 存在 fake solutions；改用 kernel weak cost 修复并改善理论保证与多样性

14. **Neural Monge Map estimation and its applications.** TMLR (Featured), 2023. [P] [paper](https://arxiv.org/abs/2106.03812) [report](reports/2106.03812.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2106.03812.pdf)

    *Jiaojiao Fan, Shu Liu, Shaojun Ma, Haomin Zhou, Yongxin Chen*

    一般 cost、可跨维度的 Monge map 弱式 max-min 求解；用 duality gap 给出严格的后验误差分析；unpaired 文生图/补全

15. **The Monge Gap: A Regularizer to Learn All Transport Maps.** ICML, 2023. [P] [paper](https://arxiv.org/abs/2302.04953) [report](reports/2302.04953.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2302.04953.pdf)

    *Théo Uscidda, Marco Cuturi*

    摆脱 ICNN 与 minimax：Monge gap 正则度量任意映射偏离 c-最优的程度，任意 cost 下单目标回归学 map

<a id="t14"></a>
### T14. 扩散桥 / Schrödinger 桥的图像到图像翻译

课题综合：[`topics/t14.md`](topics/t14.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t14_bridge_i2i.md`](source/kb/t14_bridge_i2i.md)

1. ⭐ **Diffusion Bridge Implicit Models.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2405.15885) [report](reports/2405.15885.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.15885.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2405.15885.zh.pdf)

    *Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, Jun Zhu*

    把 DDBM 推广为非马尔可夫桥（DDIM 的 bridge 对应物），诱导新 ODE 与高阶求解器，免训练加速 25×；booting noise 保翻译多样性与语义插值

2. ⭐ **LBM: Latent Bridge Matching for Fast Image-to-Image Translation.** ICCV (Highlight), 2025. [P] [paper](https://arxiv.org/abs/2503.07535) [report](reports/2503.07535.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.07535.pdf)

    *Clément Chadebec, Onur Tasar, Sanjeev Sreetharan, Benjamin Aubin*

    VAE latent 上的 Brownian bridge matching + 蒸馏，单步（1 NFE）完成重光照/去物体/深度法线估计/阴影生成；消融显示随机桥优于其零噪声极限（流匹配）

3. ⭐ **Denoising Diffusion Bridge Models.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2309.16948) [report](reports/2309.16948.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2309.16948.pdf)

    *Linqi Zhou, Aaron Lou, Samar Khanna, Stefano Ermon*

    一般化 bridge score matching 统一设计空间（VE/VP 桥），退化情形回收标准扩散与 OT-Flow-Matching；配对翻译（edges2handbags、DIODE）显著超基线

4. ⭐ **Unpaired Image-to-Image Translation via Neural Schrödinger Bridge.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2305.15086) [report](reports/2305.15086.md)

    *Beomsu Kim, Gihyun Kwon, Kwanyoung Kim, Jong Chul Ye*

    利用 SB 自相似性将其分解为对抗学习序列（时间条件判别器+正则化），首次在高分辨率非配对 I2I（horse2zebra 等）上成功

5. ⭐ **Dual Diffusion Implicit Bridges for Image-to-Image Translation.** ICLR, 2023. [P] [paper](https://arxiv.org/abs/2203.08382) [report](reports/2203.08382.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2203.08382.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2203.08382.zh.pdf)

    *Xuan Su, Jiaming Song, Chenlin Meng, Stefano Ermon*

    两个独立预训练扩散的 PF-ODE latent 拼接实现零配对/免联合训练翻译，理论上等价于"源→latent→目标"两段 Schrödinger 桥（熵正则 OT）串联，精确循环一致

6. ⭐ **I$^2$SB: Image-to-Image Schrödinger Bridge.** ICML, 2023. [P] [paper](https://arxiv.org/abs/2302.05872) [report](reports/2302.05872.md)

    *Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, Anima Anandkumar*

    边界对给定时 SB 退化为边缘解析可算的 tractable 类，simulation-free 训练；ImageNet-256 修复/超分/去模糊/JPEG 修复超越条件扩散，媲美已知退化算子的逆问题法

7. **UniDB++（UniDB 期刊版）.** IEEE TPAMI, 2026. [P] [report](reports/UniDB_UniDB.md)

    推导 UniDB 逆向 SDE 精确闭式解 + data-prediction 参数化 + SDE-Corrector，免训练加速 5-20×，低步数（5-10）保感知质量，DBIM 为其特例

8. **A Unified and Fast-Sampling Diffusion Bridge Framework via Stochastic Optimal Control.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2505.21528) [report](reports/2505.21528.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.21528.pdf)

    *Mokai Pan, Kaizhen Zhu, Yuexin Ma, Yanwei Fu, Jingyi Yu, Jingya Wang et al.*

    用随机最优控制统一扩散桥：Doob h-transform 是终端惩罚 γ→∞ 的特例，可调 γ 改善细节保真；统一 DDBM/GOUB 等

9. **Adversarial Schrödinger Bridge Matching.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2405.14449) [report](reports/2405.14449.md)

    *Nikita Gushchin, Daniil Selikhanovych, Sergei Kholkin, Evgeny Burnaev, Alexander Korotin*

    离散时间 IMF（D-IMF）只学少数转移概率，用 DD-GAN 实现，几步推理达到连续 IMF 百步的非配对翻译质量（CelebA 128）

10. **Consistency Diffusion Bridge Models.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2410.22637) [report](reports/2410.22637.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.22637.pdf)

    *Guande He, Kaiwen Zheng, Jianfei Chen, Fan Bao, Jun Zhu*

    学习 bridge PF-ODE 的一致性函数，提出 consistency bridge distillation/training 两种范式，采样加速 4-50×，两步生成可用

11. **GOUB: Generalized Ornstein-Uhlenbeck Bridge.** ICML, 2024. [P] [report](reports/GOUB_Generalized_Ornstein_Uhlenbeck_Bridge.md)

    对广义 OU 过程施加 Doob h-transform 消掉稳态方差，实现点对点修复映射并统一多种桥为特例；修复/去雨/超分 SOTA，另给 Mean-ODE 变体

12. **Latent Schrödinger Bridge.** arXiv, 2024. [R] [report](reports/Latent_Schr_dinger_Bridge.md)

    *LSB*

    把 SB PF-ODE 速度场分解为源/目标/噪声三个预测子的线性组合，用预训练 Stable Diffusion + prompt 优化免训练近似，低 NFE 非配对翻译胜过 SDEdit/DDIB

13. **BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models.** CVPR, 2023. [P] [paper](https://arxiv.org/abs/2205.07680) [report](reports/2205.07680.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2205.07680.pdf)

    *Bo Li, Kaitao Xue, Bin Liu, Yu-Kun Lai*

    首个把 I2I 建模为（VQGAN latent 上）Brownian bridge 双向扩散过程而非条件生成的工作

14. **Diffusion Schrödinger Bridge Matching.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2303.16852) [report](reports/2303.16852.md)

    *Yuyang Shi, Valentin De Bortoli, Andrew Campbell, Arnaud Doucet*

    IMF + bridge matching 的通用 SB 求解器，为翻译类 bridge 方法提供算法底座（理论细节见 T03，此处作谱系锚点）

另见（跨课题重复）：Stochastic interpolants with data-dependent couplings → T07

<a id="t15"></a>
### T15. 医学影像模态转换与 OT/SB/扩散

课题综合：[`topics/t15.md`](topics/t15.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t15_medical_modality_transfer.md`](source/kb/t15_medical_modality_transfer.md)

1. ⭐ **Anatomy-Conserving Unpaired CBCT-to-CT Translation via Schrödinger Bridge (MICCAI 2025).** MICCAI, 2025. [P] [paper](https://papers.miccai.org/miccai-2025/paper/5303_paper.pdf) [report](reports/Anatomy_Conserving_Unpaired_CBCT_to_CT_Translation.md)

    *ACSB, Shi et al.*

    熵正则 OT 解耦「模态伪影 vs 解剖」，AC-ViT 多尺度解剖先验 + 频率感知优化，无配对 CBCT→CT 跨部位泛化

2. ⭐ **Diffusion Schrödinger Bridge Models for High-Quality MR-to-CT Synthesis for Head and Neck Proton Treatment Planning.** Medical Physics（arXiv 2024）, 2025. [P] [paper](https://arxiv.org/abs/2404.11741) [report](reports/2404.11741.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2404.11741.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2404.11741.zh.pdf)

    *Muheng Li, Xia Li, Sairos Safai, Damien Weber, Antony Lomax, Ye Zhang*

    首个用 DSBM 做质子放疗 sCT 并做剂量学级验证：46/77 对小数据训练，MAE 与骨 Dice 全面优于条件扩散，1%/1mm gamma 95.9–97.9%，NFE 大幅减少

3. ⭐ **Harmonizing OCT Across Devices with Latent-Metric Schrödinger Bridges (NeurIPS 2025).** NeurIPS, 2025. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/08b60b4af0b8163b18553b15f5ce25d2-Abstract-Conference.html) [report](reports/Harmonizing_Optical_Coherence_Tomography_Across_De.md)

    *LMSB, Wei et al., JHU*

    指出 SB 的欧氏传输成本是医学解剖漂移根源，用可逆网络学 pullback 潜空间度量再训 SB，跨设备 OCT 协调保解剖 SOTA

4. ⭐ **OT-StainNet: Optimal Transport Driven Semantic Matching for Weakly Paired H&E-to-IHC Stain Transfer.** AAAI, 2025. [P] [paper](https://ojs.aaai.org/index.php/AAAI/article/view/32329) [report](reports/OT_StainNet_Optimal_Transport_Driven_Semantic_Matc.md)

    *Guan et al.*

    用 OT 在特征空间为弱配对（相邻切片错位）的 H&E–IHC 建立语义对应，把 OT 匹配变成监督信号驱动预训练扩散 LoRA 微调

5. ⭐ **Self-Consistent Recursive Diffusion Bridge for Medical Image Translation.** Medical Image Analysis, 2025. [P] [paper](https://arxiv.org/abs/2405.06789) [report](reports/2405.06789.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.06789.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2405.06789.zh.pdf)

    *Fuat Arslan, Bilal Kabas, Onat Dalmaz, Muzaffer Ozbey, Tolga Çukur*

    医学定制扩散桥：端点方差单调递增的噪声调度（软先验、抗测量噪声）+ 自洽递归采样，多对比 MRI 与 MRI↔CT SOTA

6. **Heterogeneity-Adaptive Diffusion Schrodinger Bridge for PET-Guided Whole-Body MRI Translation.** arXiv, 2026. [R] [paper](https://arxiv.org/abs/2607.07401) [report](reports/2607.07401.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2607.07401.pdf)

    *Chengbo Wang, Jiacheng Yu, Linjie Bian, Ming Qi, Xiaosheng Liu, Tongtong Che et al.*

    全身 MR 序列转换：VLM 区域上下文嵌入应对全身异质性，PET 代谢先验双阶段（前向噪声调制 + 反向注意力放大）保病灶保真

7. **PASB: Pathology-aware Schrödinger Bridge for Virtual Immunohistochemical Staining.** Medical Image Analysis, 2026. [P] [paper](https://doi.org/10.1016/j.media.2025.103869) [report](reports/PASB_Pathology_aware_Schr_dinger_Bridge_for_Virtua.md)

    *Qiu et al.*

    StainSB 升级：约束驱动对齐学习（高层病理语义监督）+ 相似度动态路径修正，下游诊断任务上接近真实 IHC

8. **Topology-aware Diffusion Schrödinger Bridge for Unpaired H&E-to-IHC Stain Translation.** IEEE JBHI, 2026. [P] [paper](https://doi.org/10.1109/JBHI.2026.3668658) [report](reports/Topology_aware_Diffusion_Schr_dinger_Bridge_for_Un.md)

    *TDSB*

    把 UNSB 引入组织病理并修其二病：拓扑引导模块保腺体/细胞拓扑，双域自适应 patch-NCE 学 IHC 染色表征；7 个转换任务 SOTA + 病理医生评估

9. **Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality.** MICCAI, 2025. [P] [paper](https://arxiv.org/abs/2503.00266) [report](reports/2503.00266.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.00266.pdf)

    *Milad Yazdani, Yasamin Medghalchi, Pooria Ashrafian, Ilker Hacihaliloglu, Dena Shahriari*

    OT flow matching 进医学：直线路径少步采样，2D 超声/3D MRI、类别/掩码条件通吃，10 步优于 50 步 DDPM

10. **Fully Guided Neural Schrödinger bridge for Brain MR image synthesis.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2501.14171) [report](reports/2501.14171.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2501.14171.pdf)

    *Hanyeol Yang, Sunggyu Kim, Mi Kyung Kim, Yongseon Yoo, Yu-Mi Kim, Min-Ho Shin et al.*

    极少配对数据（2 个受试者）下的多序列 MRI 合成：两阶段生成-训练迭代 + 互信息一致性；可注入病灶 mask 先验保病灶

11. **Path and Bone-Contour Regularized Unpaired MRI-to-CT Translation.** Computerized Medical Imaging and Graphics, 2025. [P] [paper](https://arxiv.org/abs/2505.03114) [report](reports/2505.03114.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.03114.pdf)

    *Teng Zhou, Jax Luo, Yuping Sun, Yiheng Tan, Shun Yao, Nazim Haouchine et al.*

    无配对 MRI→CT：潜空间 neural-ODE 流 + 最短传输路径正则（OT 味的路径长度最小化）+ 骨轮廓引导，下游骨分割保真最好

12. **Implicit Image-to-Image Schrodinger Bridge for Image Restoration.** 2024–2025·arXiv（v3 标注刊于 Pattern Recognition，DOI 未核验）, 2024. [R] [paper](https://arxiv.org/abs/2403.06069) [report](reports/2403.06069.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.06069.pdf)

    *Yuang Wang, Siyeop Yoon, Pengfei Jin, Matthew Tivnan, Sifan Song, Zhennong Chen et al.*

    把 I2SB 采样改成非马尔可夫（每步注入初始退化图），免重训复用预训练 I2SB，1/4 剂量腹部 CT 去噪与 4× 胸部 CT 超分少步数纹理更优

13. **PET Image Denoising based on Diffusion Schrödinger Bridge Model.** IEEE NSS/MIC/RTSD, 2024. [P] [paper](https://doi.org/10.1109/NSS/MIC/RTSD57108.2024.10657633) [report](reports/PET_Image_Denoising_based_on_Diffusion_Schr_dinger.md)

    低剂量 PET 去噪：从低剂量 PET（而非高斯）起步的 DSBM + 解剖 MR 先验，避免条件扩散的过度增强 SNR，对未见数据稳健

14. **Weakly Supervised Virtual Immunohistochemistry Staining via Schrödinger Bridge.** IEEE BIBM, 2024. [P] [paper](https://doi.org/10.1109/BIBM62325.2024.10822509) [report](reports/Weakly_Supervised_Virtual_Immunohistochemistry_Sta.md)

    *StainSB, Qiu et al.*

    首批 SB 虚拟染色：区域颜色状态损失把病理相似性注入 H&E→IHC 生成，聚合策略平衡质量与病理一致性

15. **Optimal Transport driven CycleGAN for Unsupervised Learning in Inverse Problems.** SIAM J. Imaging Sciences, 2020. [P] [paper](https://arxiv.org/abs/1909.12116) [report](reports/1909.12116.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1909.12116.pdf)

    *Byeongsu Sim, Gyutaek Oh, Jeongsol Kim, Chanyong Jung, Jong Chul Ye*

    奠基旧文：从 Kantorovich 对偶 + PLS 传输成本严格推导 cycleGAN 家族，前向算子知识可化简架构；统一无监督加速 MRI、低剂量 CT、显微镜超分

<a id="t16"></a>
### T16. OT 代价先验引导的跨域语义对应

课题综合：[`topics/t16.md`](topics/t16.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t16_ot_guided_semantic_correspondence.md`](source/kb/t16_ot_guided_semantic_correspondence.md)

1. ⭐ **Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance.** AAAI, 2026. [P] [paper](https://arxiv.org/abs/2511.07499) [report](reports/2511.07499.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2511.07499.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2511.07499.zh.pdf)

    *Kwanyoung Kim*

    把 self-attention 重释为 OT，用 Sinkhorn 注入对抗代价构造"劣化分支"做 guidance，即插即用提升 T2I/IP-Adapter/ControlNet 保真度

2. ⭐ **Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild.** CVPR, 2026. [P] [paper](https://arxiv.org/abs/2603.11618) [report](reports/2603.11618.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2603.11618.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2603.11618.zh.pdf)

    *Jiin Im, Sisung Liu, Je Hyeong Hong*

    FGW 融合外观代价（W）与结构代价（GW），用 3D 结构先验 + anchor 线性化压低 FGW 计算成本，做 in-the-wild 语义对应

3. ⭐ **Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis.** CVPR, 2025. [P] [paper](https://arxiv.org/abs/2503.22168) [report](reports/2503.22168.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.22168.pdf)

    *Woojung Han, Yeonkyung Lee, Chanyoung Kim, Kwanghyun Park, Seong Jae Hwang*

    training-free：定制空间传输代价的 OT 在早期去噪阶段重定位物体 cross-attention map，同时缓解物体错位/缺失/属性错配

4. ⭐ **OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation.** ECCV, 2024. [P] [paper](https://arxiv.org/abs/2403.14183) [report](reports/2403.14183.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.14183.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2403.14183.zh.pdf)

    *Kwanyoung Kim, Yujin Oh, Jong Chul Ye*

    MPSA 用 Sinkhorn 替换 Transformer 解码器 cross-attention 归一化，多文本 prompt 选择性对齐像素语义，ZS3 三个基准 SOTA

5. ⭐ **Optimal Transport-Guided Conditional Score-Based Diffusion Models.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2311.01226) [report](reports/2311.01226.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2311.01226.pdf)

    *Xiang Gu, Liwei Yang, Jian Sun, Zongben Xu*

    本方向奠基：L2 正则 OT 在非配对/半配对数据上建耦合，"按相容性重采样"引导条件分数模型训练，理论证明其实现 OT 数据传输

6. **OTComp: Dual Optimal Transport for Multi-Concept Composition.** ICML, 2026. [A] [paper](https://icml.cc/virtual/2026/poster/63327) [code](https://github.com/fuhao7i/OTComp) [report](reports/OTComp_Dual_Optimal_Transport_for_Multi_Concept_Co.md)

    双 OT training-free 引导：质量守恒 OT 做结构草图对齐 + 几何引导 OT 做高频纹理残差传输，多概念组合无属性串扰

7. **TP-Blend: Textual-Prompt Attention Pairing for Precise Object-Style Blending in Diffusion Models.** arXiv, 2026. [R] [paper](https://arxiv.org/abs/2601.08011) [report](reports/2601.08011.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2601.08011.pdf)

    *Xin Jin, Yichuan Zhong, Yapeng Tian*

    CAOF 用熵正则 OT 在 cross-attention 中按完整多头维度重分配特征向量，实现对象与风格双 prompt 的精确融合（风格部分归 T17）

8. **Gromov Wasserstein Optimal Transport for Semantic Correspondences.** BMVC, 2025. [P] [paper](https://arxiv.org/abs/2602.03105) [report](reports/2602.03105.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2602.03105.pdf)

    *Francis Snelgar, Stephen Gould, Ming Xu, Liang Zheng, Akshay Asthana*

    用 GW 空间平滑先验的 OT 匹配替代最近邻，DINOv2 单模型即可竞争 SD 特征 ensemble，效率高 5–10 倍

9. **Optimal Transport for Rectified Flow Image Editing: Unifying Inversion-Based and Direct Methods.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2508.02363) [report](reports/2508.02363.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2508.02363.pdf)

    *Marian Lupascu, Mihai-Sorin Stupariu*

    用传输论轨迹校正统一 inversion-based 与 direct 编辑，training-free 大幅提升 FLUX 等模型编辑的重建/一致性

10. **PLOT: Prompt Learning with Optimal Transport for Vision-Language Models.** ICLR (top-25%), 2023. [P] [paper](https://arxiv.org/abs/2210.01253) [report](reports/2210.01253.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2210.01253.pdf)

    *Guangyi Chen, Weiran Yao, Xiangchen Song, Xinyue Li, Yongming Rao, Kun Zhang*

    prompt/token 级 OT 对齐范式：多 prompt 与局部视觉特征集合做 Sinkhorn 内层对齐、外层学 prompt，防止 prompt 坍缩

11. **Keypoint-Guided Optimal Transport.** NeurIPS, 2022. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6091c5644d73637e3cccdcab52a7031f-Abstract-Conference.html) [report](reports/Keypoint_Guided_Optimal_Transport.md)

    *KPG-RL*

    用 mask 约束 plan + 关系保持把少量标注 keypoint 语义先验注入 OT，支持异构空间与 partial 设定

12. **Simultaneous Multiple-Prompt Guided Generation Using Differentiable Optimal Transport.** ICCC, 2022. [P] [paper](https://arxiv.org/abs/2204.08472) [report](reports/2204.08472.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2204.08472.pdf)

    *Yingtao Tian, Marco Cuturi, David Ha*

    早期先驱：图像 patch ↔ 多 prompt 的可微 OT 距离直接作为生成引导损失（VQGAN-CLIP 时代）

13. **Sinkformers: Transformers with Doubly Stochastic Attention.** AISTATS, 2022. [P] [paper](https://arxiv.org/abs/2110.11773) [report](reports/2110.11773.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2110.11773.pdf)

    *Michael E. Sander, Pierre Ablin, Mathieu Blondel, Gabriel Peyré*

    理论接口：softmax attention → Sinkhorn 双随机化，形式化"attention ≈ 熵正则 OT"，是后续所有 attention-OT 工作的依据

14. **Unbalanced Feature Transport for Exemplar-based Image Translation.** CVPR, 2021. [P] [paper](https://arxiv.org/abs/2106.10482) [report](reports/2106.10482.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2106.10482.pdf)

    *Fangneng Zhan, Yingchen Yu, Kaiwen Cui, Gongjie Zhang, Shijian Lu, Jianxiong Pan et al.*

    用不平衡 OT + 自适应质量学习对齐条件输入与 exemplar 特征，解决跨域分布偏差下的稠密对应（完整 I2I 框架部分归 T13/T14）

15. **SCOT: Semantic Correspondence as an Optimal Transport Problem.** CVPR, 2020. [P] [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Semantic_Correspondence_as_an_Optimal_Transport_Problem_CVPR_2020_paper.html) [report](reports/SCOT_Semantic_Correspondence_as_an_Optimal_Transpo.md)

    奠基：首次把语义对应表述为 OT 问题，用显著性做边际、Sinkhorn 求全局 plan，抑制最近邻的 many-to-one 错配

<a id="t17"></a>
### T17. 风格迁移与域自适应中的 OT×扩散

课题综合：[`topics/t17.md`](topics/t17.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t17_style_domain_adaptation.md`](source/kb/t17_style_domain_adaptation.md)

1. ⭐ **OT-ALD: Aligning Latent Distributions with Optimal Transport for Accelerated Image-to-Image Translation.** AAAI, 2026. [P] [paper](https://arxiv.org/abs/2511.11162) [report](reports/2511.11162.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2511.11162.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2511.11162.zh.pdf)

    *Zhanpeng Wang, Shuting Cao, Yuhang Lu, Yuhan Li, Na Lei, Zhongxuan Luo*

    证明 DDIB 在有限 T 下两域 latent 分布错配必然导致翻译轨迹偏差，用显式 OT map 对齐 latent 后再反向去噪，平均提速 20.3%、FID 降 2.6

2. ⭐ **Wasserstein-Aware Transfer: Class-Level Alignment for Robust Diffusion Model Adaptation.** AAAI, 2026. [P] [paper](https://ojs.aaai.org/index.php/AAAI/article/view/39365) [report](reports/Wasserstein_Aware_Transfer_Class_Level_Alignment_f.md)

    *WAT*

    分析扩散轨迹间 W 距离随 t 递减的规律，据此做源↔目标类级 OT 匹配指导扩散模型微调，并线性组合预训练/微调条件分支保知识

3. ⭐ **Color Conditional Generation with Sliced Wasserstein Guidance.** NeurIPS (spotlight), 2025. [P] [paper](https://arxiv.org/abs/2503.19034) [report](reports/2503.19034.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.19034.pdf)

    *Alexander Lobashev, Maria Larchenko, Dmitry Guskov*

    把可微 Sliced-1-Wasserstein 色彩距离塞进扩散采样循环做 training-free 颜色条件生成，胜过「先生成再色彩迁移」流水线

4. ⭐ **Stochastic Interpolants for Revealing Stylistic Flows across the History of Art.** ICCV, 2025. [P] [paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Stochastic_Interpolants_for_Revealing_Stylistic_Flows_across_the_History_of_ICCV_2025_paper.pdf) [report](reports/Stochastic_Interpolants_for_Revealing_Stylistic_Fl.md)

    *Art-FM*

    把艺术风格的历史演化建模为风格空间中的 OT 分布匹配，用 stochastic interpolants+DDIB 无配对对齐跨世纪艺术分布，并发布 65 万艺术品数据集

5. ⭐ **Multiscale Sliced Wasserstein Distances as Perceptual Color Difference Measures.** ECCV, 2024. [P] [paper](https://arxiv.org/abs/2407.10181) [report](reports/2407.10181.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2407.10181.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2407.10181.zh.pdf)

    *Jiaqi He, Zhihua Wang, Leon Wang, Tsein-I Liu, Yuming Fang, Qilin Sun et al.*

    多尺度 SW 距离做 training-free 感知色差度量，对非对齐图像对稳健，实证满足度量公理，可直接当图像/视频颜色迁移损失

6. **Rethinking the Flow-Based Gradual Domain Adaptation: A Semi-Dual Optimal Transport Perspective.** ICML, 2026. [A] [paper](https://arxiv.org/abs/2602.01179) [report](reports/2602.01179.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2602.01179.pdf)

    *Zhichao Chen, Zhan Zhuang, Yunfei Teng, Hao Wang, Fangyikang Wang, Zhengnan Li et al.*

    用 semi-dual OT 重构 flow-based 逐步域自适应的中间域生成路径

7. **Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport.** CVPR, 2026. [P] [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Han_Vision-Language_Model_Guided_Source-Free_Domain_Adaptation_via_Optimal_Transport_CVPR_2026_paper.html) [report](reports/Vision_Language_Model_Guided_Source_Free_Domain_Ad.md)

    用 VLM 语义先验引导源原型与目标特征的 OT 对齐，source-free DA 新范式

8. **Color Transfer with Modulated Flows.** AAAI, 2025. [P] [paper](https://arxiv.org/abs/2503.19062) [report](reports/2503.19062.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.19062.pdf)

    *Maria Larchenko, Alexander Lobashev, Dmitry Guskov, Vladimir Vladimirovich Palyulin*

    基于 rectified flow 的可逆 RGB 颜色迁移：在 OT plan 数据集上训练流+编码器预测流权重，新图像对零微调泛化，可处理 4K

9. **Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing.** CVPR, 2025. [P] [paper](https://arxiv.org/abs/2503.22984) [report](reports/2503.22984.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.22984.pdf)

    *Zhuowei Li, Tianchen Zhao, Xiang Xu, Zheng Zhang, Zhihua Li, Xuanbai Chen et al.*

    source-free 约束下用 OT 引导原型/特征传输，客户端自适应活体检测

10. **Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2504.03188) [report](reports/2504.03188.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2504.03188.pdf)

    *Kotaro Ikeda, Masanori Koyama, Jinzhe Zhang, Kohei Hayashi, Kenji Fukumizu*

    设计一个成本函数同时学所有条件分布对之间的 pairwise OT，支持连续条件的 all-to-all 风格/属性迁移，有无限样本极限收敛保证

11. **GIST: Towards Photorealistic Style Transfer via Multiscale Geometric Representations.** arXiv, 2024. [R] [paper](https://arxiv.org/abs/2412.02214) [report](reports/2412.02214.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2412.02214.pdf)

    *Renan A. Rojas-Gomez, Minh N. Do*

    在小波/Contourlet 子带上用高斯假设下的闭式 W2 匹配做 training-free 照片级风格迁移，替代神经自编码框架

12. **Scalable Motion Style Transfer with Constrained Diffusion Generation.** AAAI, 2024. [P] [paper](https://arxiv.org/abs/2312.07311) [report](reports/2312.07311.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2312.07311.pdf)

    *Wenjie Yin, Yi Yu, Hang Yin, Danica Kragic, Mårten Björkman*

    各风格域独立训练扩散模型，借 DDIB（熵正则 OT/SB 解释）桥接+关键帧流形约束梯度，可扩展到十种舞蹈动作风格

13. **WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians.** ECCV, 2024. [P] [paper](https://arxiv.org/abs/2409.17917) [report](reports/2409.17917.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2409.17917.pdf)

    *Dmytro Kotovenko, Olga Grebenkova, Nikolaos Sarafianos, Avinash Paliwal, Pingchuan Ma, Omid Poursaeed et al.*

    用熵正则 W2/EMD 直接匹配风格与内容场景的 3D 高斯分布，training-free 场景级 3DGS 风格迁移，把风格化从生成问题改写为显式分布匹配

14. **A Sliced Wasserstein Loss for Neural Texture Synthesis.** CVPR（奠基）, 2021. [P] [paper](https://arxiv.org/abs/2006.07229) [report](reports/2006.07229.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2006.07229.pdf)

    *Eric Heitz, Kenneth Vanhoey, Thomas Chambon, Laurent Belcour*

    用 SWD（1D 投影排序闭式解）替代 Gram 矩阵作纹理损失，捕获完整特征分布而非二阶统计量

15. **Style Transfer by Relaxed Optimal Transport and Self-Similarity.** CVPR（奠基）, 2019. [P] [paper](https://arxiv.org/abs/1904.12785) [report](reports/1904.12785.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1904.12785.pdf)

    *Nicholas Kolkin, Jason Salavon, Greg Shakhnarovich*

    用 relaxed EMD 定义风格损失+自相似保内容，OT 风格迁移的开山之作

<a id="t18"></a>
### T18. 条件生成与 guidance 的 OT 形式化

课题综合：[`topics/t18.md`](topics/t18.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t18_conditional_ot_guidance.md`](source/kb/t18_conditional_ot_guidance.md)

1. ⭐ **Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2409.08861) [report](reports/2409.08861.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2409.08861.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2409.08861.zh.pdf)

    *Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, Ricky T. Q. Chen*

    把 reward fine-tune 严格写成 SOC；证明必须用 memoryless 噪声调度才收敛到 KL-tilted 分布；SOC 化为回归（adjoint matching）

2. ⭐ **Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching.** JMLR 26(141), 2025. [P] [paper](https://arxiv.org/abs/2403.18705) [report](reports/2403.18705.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.18705.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2403.18705.zh.pdf)

    *Jannis Chemseddine, Paul Hagemann, Gabriele Steidl, Christian Wald*

    用受限耦合定义条件 Wasserstein 距离 = posterior W2 的期望；刻画测地线/速度场（Y 分量为零）并给出 Bayesian OT-FM

3. ⭐ **Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2503.02819) [report](reports/2503.02819.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.02819.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2503.02819.zh.pdf)

    *Marta Skreta, Tara Akhound-Sadegh, Viktor Ohanesian, Roberto Bondesan, Alán Aspuru-Guzik, Arnaud Doucet et al.*

    用 Feynman-Kac 公式+SMC 加权模拟，从退火/几何平均/乘积分布精确采样，修正 CFG 的中间分布失配

4. ⭐ **The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation.** ICCV, 2025. [P] [paper](https://arxiv.org/abs/2503.10636) [report](reports/2503.10636.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.10636.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2503.10636.zh.pdf)

    *Ho Kei Cheng, Alexander Schwing*

    揭示无条件 minibatch OT 在条件 FM 中造成"条件偏斜先验"的 train-test gap，在 OT 代价矩阵加条件加权项修复

5. ⭐ **Dynamic Conditional Optimal Transport through Simulation-Free Flows.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2404.04240) [report](reports/2404.04240.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2404.04240.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2404.04240.zh.pdf)

    *Gavin Kerrigan, Giosue Migliorini, Padhraic Smyth*

    证明条件 OT 的动态形式（条件版 Benamou-Brenier），用三角 COT 耦合做 simulation-free 条件生成，适用无穷维 Bayesian 逆问题

6. **Hyperparameter Trajectory Modeling via Conditional Lagrangian Optimal Transport.** ICLR (Oral), 2026. [A] [paper](https://openreview.net/forum?id=P5B97gZwRb) [report](reports/Hyperparameter_Trajectory_Modeling_via_Conditional.md)

    把条件 Lagrangian OT 用于建模训练轨迹，是条件 OT 走向新应用域的前沿样本

7. **Conditional Optimal Transport on Function Spaces.** SIAM/ASA JUQ 13(1), 2025. [P] [paper](https://arxiv.org/abs/2311.05672) [report](reports/2311.05672.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2311.05672.pdf)

    *Bamdad Hosseini, Alexander W. Hsu, Amirhossein Taghvaei*

    无穷维函数空间上 block-triangular Monge 映射与 Kantorovich 松弛的系统理论，给 amortized Bayesian 推断正则性估计

8. **Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization.** ICLR, 2025. [A] [paper](https://arxiv.org/abs/2502.06061) [report](reports/2502.06061.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2502.06061.pdf)

    *Jiajun Fan, Shuaike Shen, Chaoran Cheng, Yuxin Chen, Chumeng Liang, Ge Liu*

    RLHF 式在线 fine-tune FM，用可计算的 W2 上界正则防 policy collapse，给出 reward-多样性可控权衡

9. **Classifier-Free Guidance is a Predictor-Corrector.** arXiv（NeurIPS 2024 M3L workshop）, 2024. [R] [paper](https://arxiv.org/abs/2408.09000) [report](reports/2408.09000.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2408.09000.pdf)

    *Arwen Bradley, Preetum Nakkiran*

    证明 CFG≠gamma-powered 分布采样；SDE 极限下等价于"条件 DDIM 预测 + gamma-powered Langevin 校正"

10. **Theoretical Insights for Diffusion Guidance: A Case Study for Gaussian Mixture Models.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2403.01639) [report](reports/2403.01639.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.01639.pdf)

    *Yuchen Wu, Minshuo Chen, Zihao Li, Mengdi Wang, Yuting Wei*

    GMM 下证明 guidance 提升分类置信度同时降低微分熵（多样性），覆盖 DDPM/DDIM

11. **What does guidance do? A fine-grained analysis in a simple setting.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2409.13074) [report](reports/2409.13074.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2409.13074.pdf)

    *Muthu Chidambaram, Khashayar Gatmiry, Sitan Chen, Holden Lee, Jianfeng Lu*

    严格证明 guidance 不采样 tilted 分布；w 增大时样本堆向条件支撑集边界，有 score 误差时甚至逸出支撑集

12. **Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning.** ICML, 2023. [P] [paper](https://arxiv.org/abs/2304.12824) [report](reports/2304.12824.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2304.12824.pdf)

    *Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, Jun Zhu*

    给出中间时刻能量 guidance 的精确形式与对比学习目标，是 energy guidance 精确化的奠基工作

13. **Efficient Neural Network Approaches for Conditional Optimal Transport with Applications in Bayesian Inference.** arXiv（SISC 刊出信息未直接核验）, 2023. [R] [paper](https://arxiv.org/abs/2310.16975) [report](reports/2310.16975.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.16975.pdf)

    *Zheyu Oliver Wang, Ricardo Baptista, Youssef Marzouk, Lars Ruthotto, Deepanshu Verma*

    静态（部分输入凸网络梯度）与动态（正则化 neural ODE）两种神经条件 OT 求解器，likelihood-free 推断基线

14. **Optimal Transport-Guided Conditional Score-Based Diffusion Models.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2311.01226) [report](reports/2311.01226.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2311.01226.pdf)

    *Xiang Gu, Liwei Yang, Jian Sun, Zongben Xu*

    用 L2 正则 OT 耦合为无配对/半配对数据构造条件 score 模型，证明其以理论界实现 OT 数据传输

另见（跨课题重复）：Energy-guided Entropic Neural Optimal Transport → T04

<a id="sec-d"></a>
## D. 模态扩展

<a id="t19"></a>
### T19. 视频生成与时序一致性中的 OT/流

课题综合：[`topics/t19.md`](topics/t19.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t19_video_generation.md`](source/kb/t19_video_generation.md)

1. ⭐ **From Slow Bidirectional to Fast Autoregressive Video Diffusion Models.** CVPR (pp. 22963-22974), 2025. [P] [paper](https://arxiv.org/abs/2412.07772) [report](reports/2412.07772.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2412.07772.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2412.07772.zh.pdf)

    *Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman et al.*

    把 DMD 扩到视频：双向教师**非对称蒸馏**因果自回归学生 + 教师 ODE 轨迹初始化，4 步、KV cache 流式 9.4 FPS，VBench-Long 84.27，零样本流式 V2V/I2V

2. ⭐ **Pyramidal Flow Matching for Efficient Video Generative Modeling.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2410.05954) [report](reports/2410.05954.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.05954.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2410.05954.zh.pdf)

    *Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang et al.*

    视频原生流匹配设计：把去噪轨迹重写为空间金字塔分段流（仅末段全分辨率）+ 时间金字塔压缩历史，单一 DiT 端到端；20.7k A100 时训出 768p·24fps·10s

3. ⭐ **Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion.** NeurIPS Spotlight, 2025. [P] [paper](https://arxiv.org/abs/2506.08009) [report](reports/2506.08009.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.08009.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2506.08009.zh.pdf)

    *Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman*

    训练时自回归 self-rollout（KV 缓存）+ 视频级整体分布匹配损失，消除曝光偏差；随机梯度截断与滚动 KV cache，单 H100 17 FPS 亚秒延迟实时流式生成

4. ⭐ **How I Warped Your Noise: a Temporally-Correlated Noise Prior for Diffusion Models.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2504.03072) [report](reports/2504.03072.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2504.03072.pdf)

    *Pascal Chang, Jingwei Tang, Markus Gross, Vinicius C. Azevedo*

    帧间噪声耦合奠基作：把离散噪声重释为连续积分噪声场（∫-noise），推导**噪声传输方程**做分布保持的跨帧噪声平流，免训练消除闪烁/纹理粘连

5. **Flowception: Temporally Expansive Flow Matching for Video Generation.** CVPR (pp. 16185-16195), 2026. [P] [paper](https://arxiv.org/abs/2512.11438) [report](reports/2512.11438.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2512.11438.pdf)

    *Tariq Berrada Ifriqi, John Nguyen, Karteek Alahari, Jakob Verbeek, Ricky T. Q. Chen*

    概率路径中交错"离散帧插入 + 连续帧去噪"：非自回归、变长视频生成，训练 FLOPs 降 3 倍，缓解 AR 误差累积，统一 I2V 与插帧

6. **StreamDiT: Real-Time Streaming Text-to-Video Generation.** CVPR, 2026. [P] [paper](https://arxiv.org/abs/2507.03745) [report](reports/2507.03745.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2507.03745.pdf)

    *Akio Kodaira, Tingbo Hou, Ji Hou, Markos Georgopoulos, Felix Juefei-Xu, Masayoshi Tomizuka et al.*

    基于流匹配的**移动缓冲**训练（缓冲内帧带时变噪声水平），混合分区方案+少步蒸馏，4B 模型单 GPU 16 FPS 流式 512p 生成

7. **Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2506.09350) [report](reports/2506.09350.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.09350.pdf)

    *Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia et al.*

    把预训练视频扩散改造成 1NFE/帧的自回归实时交互生成器（单 H100 24fps 736×416、8×H100 720p），流式接收用户控制、可至分钟级

8. **Diffusion Adversarial Post-Training for One-Step Video Generation.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2501.08316) [report](reports/2501.08316.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2501.08316.pdf)

    *Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, Lu Jiang*

    扩散预训练后对**真实数据**对抗后训练（近似 R1 正则稳定训练），单次前向实时生成 2s·1280×720·24fps 视频——工业级一步视频的首个公开配方

9. **FrameBridge: Improving Image-to-Video Generation with Bridge Models.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2410.15371) [report](reports/2410.15371.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.15371.pdf)

    *Yuji Wang, Zehua Chen, Xiaoyu Chen, Yixiang Wei, Jun Zhu, Jianfei Chen*

    把 I2V 从 noise-to-data 改写为 data-to-data **桥过程**（图像为先验），提出 SNR 对齐微调（扩散→桥模型迁移）与 neural prior；MSR-VTT 零样本 FVD 95 vs 扩散基线 192

10. **Learning Few-Step Diffusion Models by Trajectory Distribution Matching.** ICCV, 2025. [P] [paper](https://arxiv.org/abs/2503.06674) [report](reports/2503.06674.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.06674.pdf)

    *Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, Jing Tang*

    统一分布匹配与轨迹匹配的**数据自由**少步蒸馏（沿教师 ODE 轨迹逐段分布对齐），图像与视频通吃：CogVideoX-2B 蒸到 4 步且 VBench 超教师

11. **Taming Rectified Flow for Inversion and Editing.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2411.04746) [report](reports/2411.04746.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.04746.pdf)

    *Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang et al.*

    免训练高阶求解器降低 rectified flow ODE 反演误差（精确解+泰勒展开），在 FLUX 与 **OpenSora 视频**上改进反演重建与编辑——视频流模型编辑的求解器基础

12. **LTX-Video: Realtime Video Latent Diffusion.** 2024–25 · arXiv, 2024. [R] [paper](https://arxiv.org/abs/2501.00103) [report](reports/2501.00103.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2501.00103.pdf)

    *Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson et al.*

    实时渲染标杆：1:192 高压缩 Video-VAE（32×32×8/token）与去噪 DiT 整体协同设计 + rectified flow，H100 上 2 秒生成 5 秒 768×512·24fps 视频（快于播放速度）

13. **T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2405.18750) [report](reports/2405.18750.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.18750.pdf)

    *Jiachen Li, Weixi Feng, Tsu-Jui Fu, Xinyi Wang, Sugato Basu, Wenhu Chen et al.*

    视频一致性蒸馏 + 图文/视频文本奖励模型混合反馈，突破 VCM 质量瓶颈，4–8 步 VBench 超当时闭源模型（Gen-2/Pika）

另见（跨课题重复）：VDOT: Efficient Unified Video Creation via Optimal Transport Distillation → T10; Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise → T12

<a id="t20"></a>
### T20. 3D/点云/几何生成中的 OT 与流

课题综合：[`topics/t20.md`](topics/t20.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t20_3d_pointcloud_generation.md`](source/kb/t20_3d_pointcloud_generation.md)

1. ⭐ **Not-So-Optimal Transport Flows for 3D Point Cloud Generation.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2502.12456) [report](reports/2502.12456.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2502.12456.pdf)

    *Ka-Hei Hui, Chao Liu, Xiaohui Zeng, Chi-Wing Fu, Arash Vahdat*

    证明 equivariant/在线 OT 耦合在大点云上失效且完全拉直反而让 t≈0 处的场更难学，提出离线 superset OT 预计算 + 与独立耦合混合的 hybrid coupling，ShapeNet 无条件生成与补全双 SOTA

2. ⭐ **SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis.** CVPR, 2025. [P] [paper](https://arxiv.org/abs/2411.16443) [report](reports/2411.16443.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.16443.pdf)

    *Hyojun Go, Byeongjun Park, Jiho Jang, Jin-Young Kim, Soonwoo Kwon, Changick Kim*

    多视角 rectified flow 在 latent 空间联合生成图像/深度/相机位姿，经前馈 GSDecoder 输出 3DGS；training-free 反演/补绘统一生成与编辑

3. ⭐ **Unsupervised Point Cloud Completion through Unbalanced Optimal Transport.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2410.02671) [report](reports/2410.02671.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.02671.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2410.02671.zh.pdf)

    *Taekyung Lee, Jaemoo Choi, Jaewoong Choi, Myungjoo Kang*

    把无配对补全形式化为 UOT map 学习，marginal 松弛天然吸收类别不平衡；系统分析 cost 选择并论证 InfoCD 最适配

4. ⭐ **Wasserstein Flow Matching: Generative modeling over families of distributions.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2411.00698) [report](reports/2411.00698.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.00698.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2411.00698.zh.pdf)

    *Doron Haviv, Aram-Alexandre Pooladian, Dana Pe'er, Brandon Amos*

    把 FM 提升到测度空间：每个样本本身是一个分布（点云/Gaussian），沿 Wasserstein 测地线定义条件流，Gaussian 族用闭式 Bures-W 路径、点云用 entropic OT 估计；首个高维「分布的分布」生成器

5. ⭐ **GaussianCube: A Structured and Explicit Radiance Representation for 3D Generative Modeling.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2403.19655) [report](reports/2403.19655.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.19655.pdf)

    *Bowen Zhang, Yiji Cheng, Jiaolong Yang, Chunyu Wang, Feng Zhao, Yansong Tang et al.*

    定数化拟合 3DGS 后用 Jonker-Volgenant 线性指派（OT）把 Gaussians 摆进 N³ voxel 网格，使标准 3D U-Net 扩散直接可用；OT 在此充当「表示结构化」角色

6. **Gaussian Herding across Pens: An Optimal Transport Perspective on Global Gaussian Reduction for 3DGS.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2506.09534) [report](reports/2506.09534.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.09534.pdf)

    *Tao Wang, Mengyu Li, Geduo Zeng, Cheng Meng, Qiong Zhang*

    把 3DGS 压缩视为全局 Gaussian 混合约简，最小化 composite transport divergence，10% 图元几乎无损渲染

7. **Neural Geometry Image-Based Representations with Optimal Transport (OT).** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2511.18679) [report](reports/2511.18679.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2511.18679.pdf)

    *Xiang Gao, Yuanpeng Liu, Xinmu Wang, Jiazhi Li, Minghao Guo, Yu Guo et al.*

    Ricci flow 共形参数化后用 OT 校正面积畸变，得到保面积 geometry image（UV 域均匀采样），支持单趟重建与连续 LoD

8. **TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2502.06608) [report](reports/2502.06608.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2502.06608.pdf)

    *Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu et al.*

    1.5B rectified flow transformer + SDF-VAE 的图生 3D 基础模型，2M 高质量样本；RF 线性轨迹成为 3D 资产工业界主干的代表

9. **Improving Dynamic NeRFs with Optimal Transport.** ICLR, 2024. [P] [paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/568b6cc71889ea0b2aa74152ef9c28db-Abstract-Conference.html) [report](reports/Improving_Dynamic_NeRFs_with_Optimal_Transport.md)

    用 OT 约束时变隐式场（dynamic NeRF）的形变一致性，是 W 距离进入隐式表示优化的代表

10. **Integrating Efficient Optimal Transport and Functional Maps For Unsupervised Shape Correspondence Learning.** CVPR, 2024. [P] [paper](https://arxiv.org/abs/2403.01781) [report](reports/2403.01781.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.01781.pdf)

    *Tung Le, Khai Nguyen, Shanlin Sun, Nhat Ho, Xiaohui Xie*

    sliced Wasserstein 与 functional map 结合做无监督形状对应，为几何生成提供跨形状 OT 对齐 anchor

11. **Fast Point Cloud Generation with Straight Flows.** CVPR, 2023. [P] [paper](https://arxiv.org/abs/2212.01747) [report](reports/2212.01747.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2212.01747.pdf)

    *Lemeng Wu, Dilin Wang, Chengyue Gong, Xingchao Liu, Yunyang Xiong, Rakesh Ranjan et al.*

    把 rectified flow 的 reflow+蒸馏引入点云扩散，实现一步生成；3D 域「轨迹拉直」的开山之作

12. **InfoCD: A Contrastive Chamfer Distance Loss for Point Cloud Completion.** NeurIPS, 2023. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2ea1943896474b7cd9796b93e526f6f-Abstract.html) [report](reports/InfoCD_A_Contrastive_Chamfer_Distance_Loss_for_Poi.md)

    *Lin et al.*

    对比学习正则化 CD，等价于最大化底层曲面互信息下界；后被 UOT-UPC 选为最优 cost

13. **Density-aware Chamfer Distance.** NeurIPS, 2021. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/f3bd5ad57c8389a8a1a541a76be463bf-Paper.pdf) [report](reports/Density_aware_Chamfer_Distance.md)

    *DCD, Wu et al.*

    指出 CD 密度盲区与 EMD 全局主导的双重缺陷，提出有界、密度敏感的折中度量并可作训练损失

14. **Texture Mapping via Optimal Mass Transport.** IEEE TVCG, 2010. [P] [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2886313/) [report](reports/Texture_Mapping_via_Optimal_Mass_Transport.md)

    *Dominitz & Tannenbaum*

    纹理/UV×OT 的奠基工作：共形初始化后经 OT 梯度流得到保面积贴图，最小化质量意义下的角度畸变

另见（跨课题重复）：WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians → T17

<a id="t21"></a>
### T21. 分子与科学计算中的 OT 流生成

课题综合：[`topics/t21.md`](topics/t21.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t21_molecules_science.md`](source/kb/t21_molecules_science.md)

1. ⭐ **Composing Unbalanced Flows for Flexible Docking and Relaxation / FlexDock (ICLR 2025).** ICLR, 2025. [P] [paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/451dbb8f4fca0327ac4e6782786673bf-Abstract-Conference.html) [report](reports/Composing_Unbalanced_Flows_for_Flexible_Docking_an.md)

    *Corso, Somnath et al.*

    提出 **Unbalanced Flow Matching**（松弛边缘约束→更易学耦合），链式 apo→holo 流形对接 + 全原子松弛，PoseBusters 合格率 30%→73%

2. ⭐ **Proteina: Scaling Flow-based Protein Structure Generative Models.** ICLR (Oral), 2025. [P] [paper](https://arxiv.org/abs/2503.00710) [report](reports/2503.00710.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.00710.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2503.00710.zh.pdf)

    *Tomas Geffner, Kieran Didi, Zuobai Zhang, Danny Reidenbach, Zhonglin Cao, Jason Yim et al.*

    把蛋白骨架 FM 规模化：非等变大 transformer（~400M）+ 层级 fold 条件 + autoguidance，800 残基仍可设计，并引入分布相似度指标

3. ⭐ **ET-Flow: Equivariant Flow-Matching for Molecular Conformer Generation.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2410.22388) [report](reports/2410.22388.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.22388.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2410.22388.zh.pdf)

    *Majdi Hassan, Nikhil Shenoy, Jungyoon Lee, Hannes Stark, Stephan Thaler, Dominique Beaini*

    等变 FM + harmonic prior + Kabsch 对齐直接在全原子坐标上做构象生成，轻量、少 NFE，GEOM 上刷新精度/物理有效性

4. ⭐ **FlowMM: Generating Materials with Riemannian Flow Matching.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2406.04713) [report](reports/2406.04713.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.04713.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2406.04713.zh.pdf)

    *Benjamin Kurt Miller, Ricky T. Q. Chen, Anuroop Sriram, Brandon M Wood*

    晶体黎曼 FM：在分数坐标环面 + 晶格 + 原子种类的联合流形上做几何约束生成（CSP/DNG）

5. ⭐ **SE(3)-Stochastic Flow Matching for Protein Backbone Generation.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2310.02391) [report](reports/2310.02391.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.02391.pdf)

    *Avishek Joey Bose, Tara Akhound-Sadegh, Guillaume Huguet, Kilian Fatras, Jarrid Rector-Brooks, Cheng-Hao Liu et al.*

    蛋白 OT-flow 范式：证明 SE(3)^N 上 Monge map 存在，构造 FoldFlow-OT（更直更稳）与 FoldFlow-SFM（SE(3) 随机桥），可任意 invariant source→target

6. **FlowDock: Geometric Flow Matching for Generative Protein-Ligand Docking and Affinity Prediction.** Bioinformatics, 2025. [P] [paper](https://arxiv.org/abs/2412.10966) [report](reports/2412.10966.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2412.10966.pdf)

    *Alex Morehead, Jianlin Cheng*

    CFM 直接把 apo 映到 holo（多配体）并预测亲和力；耦合用 apo–holo 结构过滤定义 + harmonic ligand prior，盲对接超单序列 AF3

7. **Open Materials Generation with Stochastic Interpolants.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2502.02582) [report](reports/2502.02582.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2502.02582.pdf)

    *Philipp Hoellmer, Thomas Egg, Maya M. Martirossyan, Eric Fuemmeler, Zeren Shui, Amit Gupta et al.*

    用 stochastic interpolants 统一 diffusion/FM，对晶格/坐标各自调插值 + 种类用 discrete FM，CSP/DNG 超 FlowMM/DiffCSP/MatterGen

8. **FlowLLM: Flow Matching for Material Generation with Large Language Models as Base Distributions.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2410.23405) [report](reports/2410.23405.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.23405.pdf)

    *Anuroop Sriram, Benjamin Kurt Miller, Ricky T. Q. Chen, Brandon M. Wood*

    把微调 LLM 的分布当作 RFM 的 base，实现「文本晶体分布→图数据分布」跨域传输，稳定率×3、S.U.N.×~1.5

9. **Sequence-Augmented SE(3)-Flow Matching For Conditional Protein Backbone Generation.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2405.20313) [report](reports/2405.20313.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.20313.pdf)

    *Guillaume Huguet, James Vuckovic, Kilian Fatras, Eric Thibodeau-Laufer, Pablo Lemos, Riashat Islam et al.*

    序列条件化 SE(3)-FM：pLM 编码序列 + minibatch Riemannian OT 耦合 + ReFT 强化微调，规模化到 ~21M 合成结构，超 RFdiffusion

10. **Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2402.04997) [report](reports/2402.04997.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.04997.pdf)

    *Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, Tommi Jaakkola*

    用 CTMC 实现离散 FM，并与 FrameFlow 连续结构流组合成 序列-结构 co-design（离散部分见 T22）

11. **Improved motif-scaffolding with SE(3) flow matching.** TMLR, 2024. [P] [paper](https://arxiv.org/abs/2401.04082) [report](reports/2401.04082.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2401.04082.pdf)

    *Jason Yim, Andrew Campbell, Emile Mathieu, Andrew Y. K. Foong, Michael Gastegger, José Jiménez-Luna et al.*

    FrameFlow（SE(3)-FM，采样步数少 5×）的 scaffolding 扩展：motif amortization 与**无须重训**的 motif guidance，可设计性/多样性大幅提升

12. **Transferable Boltzmann Generators.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2406.14426) [report](reports/2406.14426.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.14426.pdf)

    *Leon Klein, Frank Noé*

    等变 CNF+FM 做跨化学空间零样本平衡采样（二肽），并实证：可区分粒子多时 OT-FM 相对普通 FM 增益变小（重要 caveat）

另见（跨课题重复）：Equivariant flow matching → T08

<a id="t22"></a>
### T22. 离散数据与文本中的扩散/流与最优传输

课题综合：[`topics/t22.md`](topics/t22.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t22_discrete_text.md`](source/kb/t22_discrete_text.md)

1. ⭐ **Minibatch Optimal Transport and Perplexity Bound Estimation in Discrete Flow Matching.** ICML, 2026. [A] [paper](https://arxiv.org/abs/2411.00759) [report](reports/2411.00759.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.00759.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2411.00759.zh.pdf)

    *Etrit Haxholli, Yeti Z. Gurbuz, Ogul Can, Eli Waxman*

    首个离散流的动态 OT 式目标及其 Kantorovich 形式（成本=状态间相异度），minibatch-OT 耦合把达到同等生成困惑度的转移次数降至 1/32；另给出离散流困惑度上界

2. ⭐ **Flexible-length Text Infilling for Discrete Diffusion Models.** EMNLP main, 2025. [P] [paper](https://arxiv.org/abs/2506.13579) [report](reports/2506.13579.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.13579.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2506.13579.zh.pdf)

    *Andrew Zhang, Anushka Sivakumar, Chiawei Tang, Chris Thomas*

    首个灵活长度/位置文本填充的离散扩散：联合去噪 token 值与位置，用 sample-level OT 耦合保持相对语序、动态调整填充段位置与长度

3. ⭐ **Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.** ICML (Best Paper), 2024. [P] [paper](https://arxiv.org/abs/2310.16834) [report](reports/2310.16834.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.16834.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2310.16834.zh.pdf)

    *Aaron Lou, Chenlin Meng, Stefano Ermon*

    提出 score entropy 把 score matching 推广到离散空间（学概率比值），扩散 LM 首次在困惑度上压过 GPT-2，并可 32× 减少 NFE

4. ⭐ **Discrete Flow Matching.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2407.15595) [report](reports/2407.15595.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2407.15595.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2407.15595.zh.pdf)

    *Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve et al.*

    通用离散概率路径族 + 后验参数化的生成速度公式 + corrector 采样，1.7B 模型显著缩小与 AR 的代码/文本生成差距

5. ⭐ **Fisher Flow Matching for Generative Modeling over Discrete Data.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2405.14664) [report](reports/2405.14664.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.14664.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2405.14664.zh.pdf)

    *Oscar Davis, Samuel Kessler, Mircea Petrache, İsmail İlkan Ceylan, Michael Bronstein, Avishek Joey Bose*

    把类别分布放到 Fisher-Rao 统计流形（球面正象限）上做连续 FM，用黎曼 OT 重耦合改善训练动力学，并证其梯度流最优降低前向 KL

6. **An Optimal Transport View of Activation Steering in Masked Diffusion Models.** ICLR Workshop (TTU), 2026. [A] [paper](https://openreview.net/forum?id=3JM0DTKxgE) [report](reports/An_Optimal_Transport_View_of_Activation_Steering_i.md)

    把 dLLM 激活转向统一为仿射 OT map（矩匹配估计），推理时零开销提升 LLaDA/Dream 指令遵循 +6.5~11.9 分

7. **Consistent Diffusion Language Models.** arXiv, 2026. [R] [paper](https://arxiv.org/abs/2605.00161) [report](reports/2605.00161.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2605.00161.pdf)

    *Hasan Amin, Yuan Gao, Yaser Souri, Subhojit Som, Ming Yin, Rajiv Khanna et al.*

    离散域没有 PF-ODE，提出以精确后验桥为「轨迹」的多路径离散一致性训练（teacher-free），统一 masked diffusion/连续一致性/渐进蒸馏为解析极限

8. **Dimension-Free Convergence of Discrete Diffusion Models: Adjoint Equations Induce the Right Space.** arXiv, 2026. [R] [paper](https://arxiv.org/abs/2605.17232) [report](reports/2605.17232.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2605.17232.pdf)

    *Kelvin Kan, Xingjian Li, Benjamin J. Zhang, Tuhin Sahai, Stanley Osher, Markos A. Katsoulakis*

    伴随方程框架给出任意 IPM 下完全不依赖词表大小 S 的收敛界，首次同时覆盖 masked（奇异先验）与 uniform

9. **Efficient Sampling with Discrete Diffusion Models: Sharp and Adaptive Guarantees.** COLT, 2026. [P] [paper](https://arxiv.org/abs/2602.15008) [report](reports/2602.15008.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2602.15008.pdf)

    *Daniil Dmitriev, Zhihan Huang, Yuting Wei*

    τ-leaping 达 \(\tilde O(d/\varepsilon)\) KL 复杂度（消去词表 S 依赖）+匹配下界；masked 情形由「有效总相关」自适应控制，可对结构化数据亚线性

10. **d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation.** arXiv（repo 自称 ICML 2026，未经官方页核验）, 2026. [R] [paper](https://arxiv.org/abs/2601.07568) [report](reports/2601.07568.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2601.07568.pdf)

    *Yu-Yang Qian, Junda Su, Lanxiang Hu, Peiyuan Zhang, Zhijie Deng, Peng Zhao et al.*

    伪轨迹蒸馏使 LLaDA/Dream 级 dLLM 接近每步多 token 的极限吞吐

11. **Beyond Autoregression: Fast LLMs via Self-Distillation Through Time.** ICLR, 2025. [A] [paper](https://arxiv.org/abs/2410.21035) [report](reports/2410.21035.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.21035.pdf)

    *Justin Deschenaux, Caglar Gulcehre*

    跨时间自蒸馏 masked 扩散 LM，32-64 token/步并行解码仍优于 GPT-2 级 AR（多篇 proceedings 交叉引用确认 ICLR 2025）

12. **Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.** ICLR (Oral), 2025. [P] [paper](https://arxiv.org/abs/2503.09573) [report](reports/2503.09573.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.09573.pdf)

    *Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han et al.*

    块间自回归+块内扩散的插值族，支持 KV cache、任意长度生成与并行采样，扩散 LM 似然新 SOTA

13. **Convergence Analysis of Discrete Diffusion Model: Exact Implementation through Uniformization.** J. Mach. Learn. 4(2), 2025. [P] [paper](https://arxiv.org/abs/2402.08095) [report](reports/2402.08095.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.08095.pdf)

    *Hongrui Chen, Lexing Ying*

    用 CTMC uniformization 精确模拟反向链，给出超立方体上 TV/KL 收敛保证，对齐连续扩散最优结果

14. **Distillation of Discrete Diffusion through Dimensional Correlations.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2410.08709) [report](reports/2410.08709.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.08709.pdf)

    *Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, Yuki Mitsufuji*

    用混合模型显式学习维度间相关性，把多步独立分解的教师蒸馏成 few-step 学生，并给出学生-教师分布距离上界

15. **Dream 7B: Diffusion Large Language Models.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2508.15487) [report](reports/2508.15487.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2508.15487.pdf)

    *Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang et al.*

    AR 初始化+上下文自适应 token 级噪声重调度训练的 7B 扩散 LM，规划类任务显著优于同规模 AR

16. **Edit Flows: Flow Matching with Edit Operations.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2506.09018) [report](reports/2506.09018.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.09018.pdf)

    *Marton Havasi, Brian Karrer, Itai Gat, Ricky T. Q. Chen*

    在整条序列空间上定义 CTMC，转移=插入/删除/替换编辑操作（编辑距离几何），经辅助对齐过程+Bregman 散度损失可训练，原生支持变长生成

17. **$\textit{Jump Your Steps}$: Optimizing Sampling Schedule of Discrete Diffusion Models.** ICLR, 2025. [A] [paper](https://arxiv.org/abs/2410.07761) [report](reports/2410.07761.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.07761.pdf)

    *Yong-Hyun Park, Chieh-Hsin Lai, Satoshi Hayakawa, Yuhta Takida, Yuki Mitsufuji*

    无额外计算下优化离散扩散采样时间表，最小化复合解码误差（Di4C 官方引用确认 ICLR 2025）

18. **Large Language Diffusion Models.** NeurIPS (Oral), 2025. [P] [paper](https://arxiv.org/abs/2502.09992) [report](reports/2502.09992.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2502.09992.pdf)

    *Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu et al.*

    8B 从零预训练+SFT 的 masked 扩散 LM，多基准比肩 LLaMA3-8B，证明扩散范式可规模化并破解 reversal curse

19. **Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models.** AAAI (Oral), 2025. [P] [paper](https://arxiv.org/abs/2412.14528) [report](reports/2412.14528.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2412.14528.pdf)

    *Xiao Cui, Mo Zhu, Yulei Qin, Liang Xie, Wengang Zhou, Houqiang Li*

    token 级+序列级双层 OT（Sinkhorn）对齐不同词表的 logit 分布，实现任意教师→学生的跨 tokenizer LLM 蒸馏

20. **Optimal Transport-Based Token Weighting scheme for Enhanced Preference Optimization.** ACL main, 2025. [P] [paper](https://arxiv.org/abs/2505.18720) [report](reports/2505.18720.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.18720.pdf)

    *Meng Li, Guangda Huzhang, Haibo Zhang, Xiting Wang, Anxiang Zeng*

    用 unbalanced OT 在 chosen/rejected 回复间算语义对齐权重，重加权 DPO 的 token 级损失，统一 SimPO/SamPO 等为特例

21. **The Diffusion Duality.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2506.10892) [report](reports/2506.10892.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.10892.pdf)

    *Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, Volodymyr Kuleshov*

    证明 uniform-state 离散扩散 = 底层高斯扩散经 argmax 投影而来；借对偶把课程学习与一致性蒸馏搬到离散域（DCD），采样加速两个数量级

22. **Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data.** ICLR, 2025. [A] [paper](https://arxiv.org/abs/2406.03736) [report](reports/2406.03736.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.03736.pdf)

    *Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li et al.*

    证明 absorbing 扩散的 concrete score 可分解为条件分布×时间标量，与任意序 AR 等价，支持缓存加速

23. **Simple and Effective Masked Diffusion Language Models.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2406.07524) [report](reports/2406.07524.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.07524.pdf)

    *Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu et al.*

    Rao-Blackwell 化的连续时间 ELBO 证明 masked diffusion 目标 = 加权 MLM 交叉熵混合，给 BERT 式编码器赋予有原则的生成能力

24. **Simplified and Generalized Masked Diffusion for Discrete Data.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2406.04329) [report](reports/2406.04329.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.04329.pdf)

    *Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, Michalis K. Titsias*

    统一简化 masked diffusion 框架，支持状态依赖的 masking schedule；像素级离散建模超过同规模 AR

25. **Sinkhorn Distance Minimization for Knowledge Distillation.** LREC-COLING; 2025·IEEE TNNLS, 2024. [P] [paper](https://arxiv.org/abs/2402.17110) [report](reports/2402.17110.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.17110.pdf)

    *Xiao Cui, Yulei Qin, Yuting Gao, Enwei Zhang, Zihan Xu, Tong Wu et al.*

    用 Sinkhorn 距离替代 KL/RKL/JS 做 LLM logit 蒸馏，批级重构感知分布几何，规避 mode-averaging/collapsing

26. **Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs.** arXiv, 2024. [R] [paper](https://arxiv.org/abs/2402.12030) [report](reports/2402.12030.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.12030.pdf)

    *Nicolas Boizard, Kevin El Haddad, Céline Hudelot, Pierre Colombo*

    最早用 Wasserstein 距离做跨词表 logit 蒸馏的损失（MultiLevelOT 的直接前驱）

27. **Structured Denoising Diffusion Models in Discrete State-Spaces.** NeurIPS, 2021. [P] [paper](https://arxiv.org/abs/2107.03006) [report](reports/2107.03006.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2107.03006.pdf)

    *Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg*

    离散扩散奠基：结构化转移矩阵（uniform/absorbing/离散化高斯）统一离散前向过程设计空间

另见（跨课题重复）：Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design → T21

<a id="t23"></a>
### T23. 语音与音频中的流匹配与 Schrödinger 桥

课题综合：[`topics/t23.md`](topics/t23.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t23_speech_audio.md`](source/kb/t23_speech_audio.md)

1. ⭐ **F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.** ACL, 2025. [P] [paper](https://aclanthology.org/2025.acl-long.313/) [report](reports/F5_TTS_A_Fairytaler_that_Fakes_Fluent_and_Faithful.md)

    *Chen et al.*

    E2 配方的可训练化：ConvNeXt 精炼文本表示 + 推理期 Sway Sampling（免重训的流步重分配，可移植到任意 FM 模型），RTF 0.15，10 万小时全开源

2. ⭐ **Matcha-TTS: A fast TTS architecture with conditional flow matching.** ICASSP, 2024. [P] [paper](https://arxiv.org/abs/2309.03199) [report](reports/2309.03199.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2309.03199.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2309.03199.zh.pdf)

    *Shivam Mehta, Ruibo Tu, Jonas Beskow, Éva Székely, Gustav Eje Henter*

    轻量开源标杆：OT-CFM 训练的 ODE 解码器 + 联合学发音与对齐（无外部对齐器），2-10 步合成、最小内存占用

3. ⭐ **Schrödinger Bridge for Generative Speech Enhancement.** Interspeech, 2024. [P] [paper](https://arxiv.org/abs/2407.16074) [report](reports/2407.16074.md)

    *Ante Jukić, Roman Korostik, Jagadeesh Balam, Boris Ginsburg*

    SB 增强开山：clean-noisy 配对 SB + 数据预测损失 + 时域辅助损失，去噪/去混响相对 WER 降 20%/6%，已入 NeMo

4. ⭐ **Schrodinger Bridges Beat Diffusion Models on Text-to-Speech Synthesis.** arXiv 2312.03491（ICLR 2024 撤稿）, 2023. [R] [paper](https://arxiv.org/abs/2312.03491) [report](reports/2312.03491.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2312.03491.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2312.03491.zh.pdf)

    *Zehua Chen, Guande He, Kaiwen Zheng, Xu Tan, Jun Zhu*

    用文本潜变量替换高斯先验：配对数据间完全可解 SB + bridge SDE/ODE 采样器与指数积分器，2-4 步即超 Grad-TTS 与快速 TTS 基线

5. ⭐ **Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2306.15687) [report](reports/2306.15687.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2306.15687.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2306.15687.zh.pdf)

    *Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz et al.*

    FM 进语音的奠基：5 万小时文本引导语音 infilling 预训练，零样本 TTS/编辑/去噪一模型通吃，比 VALL-E 准且快 20 倍

6. **A2SB: Audio-to-Audio Schrodinger Bridges.** arXiv 2501.11311, 2025. [R] [paper](https://arxiv.org/abs/2501.11311) [report](reports/2501.11311.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2501.11311.pdf)

    *Zhifeng Kong, Kevin J Shih, Weili Nie, Arash Vahdat, Sang-gil Lee, Joao Felipe Santos et al.*

    44.1kHz 高保真音乐修复：单一 SB 模型统一带宽扩展+inpainting，幅度/相位分解表示免声码器端到端，MultiDiffusion 拼接修复小时级长音频

7. **Bridge-SR: Schrödinger Bridge for Efficient SR.** ICASSP, 2025. [P] [paper](https://arxiv.org/abs/2501.07897) [report](reports/2501.07897.md)

    *Chang Li, Zehua Chen, Fan Bao, Jun Zhu*

    波形域 any-to-48kHz 语音超分：低分辨率波形作先验的可解 SB，1.7M 参数骨干 4 步胜 8 步条件扩散

8. **FlowSE: Efficient and High-Quality Speech Enhancement via Flow Matching.** Interspeech, 2025. [P] [paper](https://arxiv.org/abs/2505.19476) [report](reports/2505.19476.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.19476.pdf)

    *Ziqian Wang, Zikai Liu, Xinfa Zhu, Yike Zhu, Mingshuai Liu, Jun Chen et al.*

    FM 语音增强：noisy mel（+可选文本）条件下单程连续变换，延迟远低于扩散 SE 且质量更高

9. **Schrödinger Bridge Consistency Trajectory Models for Speech Enhancement.** arXiv 2507.11925（GitHub 称 WASPAA 2025 接收，未见官方页）, 2025. [R] [paper](https://arxiv.org/abs/2507.11925) [report](reports/2507.11925.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2507.11925.pdf)

    *Shuichiro Nishigori, Koichi Saito, Naoki Murata, Masato Hirano, Shusuke Takahashi, Yuki Mitsufuji*

    把一致性轨迹模型（CTM）嫁接到 SB 增强：一步推理 RTF 提升约 16×，一步不够再多步细化

10. **StableVC: Style Controllable Zero-Shot Voice Conversion with Conditional Flow Matching.** AAAI, 2025. [P] [paper](https://arxiv.org/abs/2412.04724) [report](reports/2412.04724.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2412.04724.pdf)

    *Jixun Yao, Yuguang Yang, Yu Pan, Ziqian Ning, Jiaohao Ye, Hongbin Zhou et al.*

    内容/音色/风格三解耦 + 双注意力自适应门控 CFM 重建：音色与风格可独立迁移到不同 unseen 说话人，比扩散 VC 快 1.65×

11. **CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models.** arXiv 2412.10117, 2024. [R] [paper](https://arxiv.org/abs/2412.10117) [report](reports/2412.10117.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2412.10117.pdf)

    *Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao et al.*

    工业界标准栈：LLM 出语义 token、chunk-aware 因果 FM 出 mel，单模型统一流式/非流式，流式质量近乎无损

12. **E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS.** IEEE SLT, 2024. [P] [paper](https://arxiv.org/abs/2406.18009) [report](reports/2406.18009.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.18009.pdf)

    *Sefik Emre Eskimez, Xiaofei Wang, Manthan Thakker, Canrun Li, Chung-Hsien Tsai, Zhen Xiao et al.*

    极简范式：字符序列补 filler token 到 mel 长度 + FM infilling，砍掉时长模型/G2P/单调对齐，仍达人类级自然度

13. **MusicFlow: Cascaded Flow Matching for Text Guided Music Generation.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2410.20478) [report](reports/2410.20478.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.20478.pdf)

    *K R Prajwal, Bowen Shi, Matthew Lee, Apoorv Vyas, Andros Tjandra, Mahi Luthra et al.*

    级联双 FM（文本→语义→声学）+ masked 预测目标，参数小 2-5 倍、步数少 5 倍，零样本 infilling/续写

14. **Generative Pre-training for Speech with Flow Matching.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2310.16338) [report](reports/2310.16338.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.16338.pdf)

    *Alexander H. Liu, Matt Le, Apoorv Vyas, Bowen Shi, Andros Tjandra, Wei-Ning Hsu*

    FM + masked 条件在 6 万小时无标注语音上预训练的「语音生成基础模型」，微调即匹配增强/分离/合成专家模型

15. **P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting.** NeurIPS, 2023. [P] [paper](https://openreview.net/forum?id=zNA7u7wtIN) [report](reports/P_Flow_A_Fast_and_Data_Efficient_Zero_Shot_TTS_thr.md)

    *Kim et al., NVIDIA*

    speech prompt 文本编码器 + FM 解码器：用比 VALL-E 少两个数量级的数据达到同级说话人相似度、采样快 20 倍

<a id="t24"></a>
### T24. 单细胞与生物轨迹推断中的 OT×流

课题综合：[`topics/t24.md`](topics/t24.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t24_singlecell_trajectory.md`](source/kb/t24_singlecell_trajectory.md)

1. ⭐ **Mapping Cells Through Time and Space with moscot.** Nature, 2025. [P] [paper](https://www.nature.com/articles/s41586-024-08453-2) [report](reports/Mapping_Cells_Through_Time_and_Space_with_moscot.md)

    *Klein, Palla, Lange et al.*

    工程集大成：低秩熵 OT/GW/FGW 统一时序、空间、时空、谱系全部单细胞 OT 应用，多模态、170 万细胞×20 时间点 atlas 规模，实验验证 NEUROD2

2. ⭐ **GENOT: Entropic (Gromov) Wasserstein Flow Matching with Applications to Single-Cell Genomics.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2310.09254) [report](reports/2310.09254.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.09254.pdf)

    *Dominik Klein, Théo Uscidda, Fabian Theis, Marco Cuturi*

    范式切换：用条件 FM 直接建模熵 OT 耦合的条件分布 π_ε(y\

3. ⭐ **Learning Single-Cell Perturbation Responses using Neural Optimal Transport.** Nature Methods, 2023. [P] [paper](https://doi.org/10.1038/s41592-023-01969-x) [report](reports/Learning_Single_Cell_Perturbation_Responses_using.md)

    *CellOT, Bunne et al.*

    扰动线奠基：ICNN 对偶势学 control→perturbed 的 Monge map，预测未见病人药物响应（4i+scRNA）

4. ⭐ **Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming.** Cell, 2019. [P] [paper](https://doi.org/10.1016/j.cell.2019.01.006) [report](reports/Optimal_Transport_Analysis_of_Single_Cell_Gene_Exp.md)

    *Waddington-OT, Schiebinger et al.*

    奠基：把发育建模为测度演化，相邻时间点间解带增殖率的熵正则 unbalanced OT，31.5 万细胞重编程谱系与祖先/命运分析

5. **Branched Schrödinger Bridge Matching.** ICLR, 2026. [A] [paper](https://arxiv.org/abs/2506.09007) [report](reports/2506.09007.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.09007.pdf)

    *Sophia Tang, Yinuo Zhang, Alexander Tong, Pranam Chatterjee*

    分支 SB：把广义 SB 分解为多条带权 unbalanced 随机最优控制分支（每支独立速度+增长网络），建模命运分叉与扰动分歧

6. **DeST-OT: Alignment of Spatiotemporal Transcriptomics Data.** Cell Systems, 2025. [P] [paper](https://doi.org/10.1016/j.cels.2024.12.001) [report](reports/DeST_OT_Alignment_of_Spatiotemporal_Transcriptomic.md)

    *Halmos et al.*

    空间时序配准：semi-relaxed FGW 建模发育组织切片间的生长/凋亡/分化，提出 growth-distortion 与 migration 度量

7. **Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport.** ICLR (Oral), 2025. [P] [paper](https://arxiv.org/abs/2410.00844) [report](reports/2410.00844.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.00844.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2410.00844.zh.pdf)

    *Zhenyi Zhang, Tiejun Li, Peijie Zhou*

    正则化 unbalanced OT（≈unbalanced SB）的深度求解器：Fisher 正则把 SDE 问题化成 ODE 约束，无先验地同时学增殖与转移、重建 Waddington 景观

8. **Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2408.14608) [report](reports/2408.14608.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2408.14608.pdf)

    *Lazar Atanackovic, Xi Zhang, Brandon Amos, Mathieu Blanchette, Leo J. Lee, Yoshua Bengio et al.*

    把「初始群体」用 GNN 嵌入后 amortize 速度场——Wasserstein 流形上的向量场积分，泛化到未见病人的治疗响应

9. **Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2505.11197) [report](reports/2505.11197.md)

    *Zhenyi Zhang, Zihan Wang, Yuhao Sun, Tiejun Li, Peijie Zhou*

    UMFSB：unbalanced SB 加 mean-field 交互项，四网络（速度/增长/对数密度/交互势）从快照学细胞间相互作用

10. **Modeling Complex System Dynamics with Flow Matching Across Time and Conditions.** ICLR (Spotlight), 2025. [P] [paper](https://openreview.net/forum?id=hwnObmOTrV) [report](reports/Modeling_Complex_System_Dynamics_with_Flow_Matchin.md)

    *MMFM, Rohbeck et al.*

    多边缘 FM：样条插值构造跨时间点平滑条件路径 + classifier-free guidance 跨条件共享动力学，补全缺失(时间点×扰动)组合

11. **TIGON.** Nat. Mach. Intell., 2024. [P] [paper](https://www.nature.com/articles/s42256-023-00763-w) [report](reports/TIGON.md)

    *Sha, Qiu, Zhou & Nie*

    Wasserstein–Fisher–Rao 动态 unbalanced OT 的 neural ODE 求解：同时重建轨迹、增殖率与时序基因调控网络

12. **MIOFlow.** NeurIPS, 2022. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bfc03f077688d8885c0a9389d77616d0-Abstract-Conference.html) [report](reports/MIOFlow.md)

    *Huguet et al.*

    测地自编码器潜空间中训练 neural ODE、以流形 ground distance 的 OT 罚项插值快照，处理分叉/汇合

13. **TrajectoryNet.** ICML, 2020. [P] [paper](https://proceedings.mlr.press/v119/tong20a.html) [report](reports/TrajectoryNet.md)

    *Tong et al.*

    首个连续化：CNF + 动态 OT 能量惩罚做快照间连续插值，可加密度/velocity 正则

另见（跨课题重复）：Simulation-free Schrödinger bridges via score and flow matching → T08; Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation → T13

<a id="sec-e"></a>
## E. OT 变体前沿

<a id="t25"></a>
### T25. 非平衡/部分 OT 在生成建模中的应用

课题综合：[`topics/t25.md`](topics/t25.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t25_unbalanced_partial_ot_gen.md`](source/kb/t25_unbalanced_partial_ot_gen.md)

1. ⭐ **WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport.** ICLR, 2026. [A] [paper](https://arxiv.org/abs/2601.06810) [report](reports/2601.06810.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2601.06810.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2601.06810.zh.pdf)

    *Qiangwei Peng, Zihan Wang, Junda Ying, Yuhao Sun, Qing Nie, Lei Zhang et al.*

    flow matching 同时回归速度场+标量增长率，证明最小化损失恰好回收 WFR 测地线；非平衡快照动态的统一范式

2. ⭐ **Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport.** ICLR oral, 2025. [P] [paper](https://arxiv.org/abs/2410.00844) [report](reports/2410.00844.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.00844.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2410.00844.zh.pdf)

    *Zhenyi Zhang, Tiejun Li, Peijie Zhou*

    RUOT 神经求解器：无先验联合学 growth/death 与漂移，Fisher 正则打通 RUOT↔SB

3. ⭐ **Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2505.11823) [report](reports/2505.11823.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.11823.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2505.11823.zh.pdf)

    *Yuhao Sun, Zhenyi Zhang, Zihan Wang, Tiejun Li, Peijie Zhou*

    把 RUOT 一阶最优性条件写进参数化与损失，单个标量场解 RUOT、作用量更小、训练更稳；讨论 WFR 增长罚函数选择

4. **Branched Schrödinger Bridge Matching.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2506.09007) [report](reports/2506.09007.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2506.09007.pdf)

    *Sophia Tang, Yinuo Zhang, Alexander Tong, Pranam Chatterjee*

    分支 GSB：每支一个速度场+增长网络，把"一源多汇"的质量再分配化为可分解的 Unbalanced CondSOC

5. **Efficient Algorithms for Robust and Partial Semi-Discrete OT.** NeurIPS, 2025. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e8f3ca5a82f5511370af7ed0efcad0f-Abstract-Conference.html) [report](reports/Efficient_Algorithms_for_Robust_and_Partial_Semi_D.md)

    *Agarwal, Raghvendra, Shirzadian, Yao*

    α-partial 与 λ-TV-robust 半离散 OT 的 restricted Laguerre 刻画、两问题互相归约与精确/近似算法

6. **Joint Velocity-Growth Flow Matching.** NeurIPS, 2025. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eb1bad7a84ef68a64f1afd6577725d45-Abstract-Conference.html) [report](reports/Joint_Velocity_Growth_Flow_Matching.md)

    *VGFM; Wang et al.*

    给静态 semi-relaxed OT 一个"先长质量后运输"的两段式动态解释，联合速度+增长的 simulation-free FM

7. **Taming Flow Matching with Unbalanced Optimal Transport into Fast Pansharpening.** ICCV, 2025. [P] [paper](https://arxiv.org/abs/2503.14975) [report](reports/2503.14975.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.14975.pdf)

    *Zihan Cao, Yu Zhong, Liang-Jian Deng*

    UOT 对偶 + 任务正则构造一步跨模态融合流，UOT 松弛吸收遥感光谱/空间失配

8. **Light Unbalanced Optimal Transport.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2303.07988) [report](reports/2303.07988.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2303.07988.pdf)

    *Milena Gazdieva, Arip Asadulaev, Alexander Korotin, Evgeny Burnaev*

    非 minimax、Gaussian-mixture 参数化的轻量 UEOT solver，附普适逼近与泛化界

9. **Scalable Wasserstein Gradient Flow via Unbalanced OT.** ICML, 2024. [P] [paper](https://proceedings.mlr.press/v235/choi24a.html) [report](reports/Scalable_Wasserstein_Gradient_Flow_via_Unbalanced.md)

    *S-JKO; Choi, Choi, Kang*

    发现 JKO 步 ≡ UOT 问题，半对偶化把 WGF 生成训练复杂度 O(K²)→O(K)

10. **Unbalanced Diffusion Schrödinger Bridge.** arXiv, 2023. [R] [paper](https://arxiv.org/abs/2306.09099) [report](reports/2306.09099.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2306.09099.pdf)

    *Matteo Pariset, Ya-Ping Hsieh, Charlotte Bunne, Andreas Krause, Valentin De Bortoli*

    推导带 killing/birth 项 SDE 的时间反演，把 DSB 推广到任意有限质量边缘（药物响应、病毒变体）

11. **Optimal Entropy-Transport Problems and a New Hellinger–Kantorovich Distance.** Invent. Math., 2018. [P] [paper](https://doi.org/10.1007/s00222-017-0759-8) [report](reports/Optimal_Entropy_Transport_Problems_and_a_New_Helli.md)

    *Liero, Mielke, Savaré*

    奠基：熵-运输问题（KL 松弛边缘）、conic 提升与 HK 距离，UOT 的静态理论骨架

12. **Unbalanced Optimal Transport: Dynamic and Kantorovich Formulation.** J. Funct. Anal., 2018. [P] [paper](https://arxiv.org/abs/1508.05216) [report](reports/1508.05216.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1508.05216.pdf)

    *Lenaic Chizat, Gabriel Peyré, Bernhard Schmitzer, François-Xavier Vialard*

    奠基：WFR 动态形式 ≡ 静态 conic 形式，配套 generalized Sinkhorn（Math. Comp. 2018）

另见（跨课题重复）：Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport → T13; Analyzing and Improving Optimal-Transport-based Adversarial Networks → T13; Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation → T13

<a id="t26"></a>
### T26. Gromov-Wasserstein 与跨空间生成对齐

课题综合：[`topics/t26.md`](topics/t26.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t26_gromov_wasserstein_gen.md`](source/kb/t26_gromov_wasserstein_gen.md)

1. ⭐ **Gromov-Wasserstein at Scale, Beyond Squared Norms.** ICML（种子库 [A]）, 2026. [A] [paper](https://arxiv.org/abs/2602.06658) [report](reports/2602.06658.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2602.06658.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2602.06658.zh.pdf)

    *Guillaume Houry, Jean Feydy, François-Xavier Vialard*

    识别出条件负定型（CNT）畸变代价大类，使 GW 化为 lifted 特征空间线性对齐 + 标准平方欧氏 OT：线性内存、二次（而非三次）时间、可微、可探索能量景观的 EGW solver，数十万点分钟级

2. ⭐ **LAST: Bridging Vision-Language and Action Manifolds via Gromov-Wasserstein Alignment.** ICML, 2026. [A] [paper](https://arxiv.org/abs/2606.11221) [report](reports/2606.11221.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2606.11221.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2606.11221.zh.pdf)

    *Huaihai Lyu, Chaofan Chen, Yuheng Ji, Xiansheng Chen, Pengwei Wang, Shanghang Zhang et al.*

    把 VLA 学习表述为 GW 对齐问题：Lie 代数 tokenizer 全局线性化动作流形 + 白化局部度量离散化，使动作空间的关系几何与 VL 语义嵌入统计兼容

3. ⭐ **It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data.** CVPR, 2025. [P] [paper](https://arxiv.org/abs/2503.24129) [report](reports/2503.24129.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.24129.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2503.24129.zh.pdf)

    *Dominik Schnaus, Nikita Araslanov, Daniel Cremers*

    把「无任何平行数据的视觉-语言匹配」形式化为 GW 型 QAP，改进 Hahn-Grant 对偶求解器，实证 platonic representation hypothesis 下基础模型嵌入可被无监督结构对齐

4. ⭐ **Gromov-Wasserstein Distances: Entropic Regularization, Duality, and Sample Complexity.** Annals of Statistics 52(4), 2024. [P] [paper](https://arxiv.org/abs/2212.12848) [report](reports/2212.12848.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2212.12848.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2212.12848.zh.pdf)

    *Zhengxin Zhang, Ziv Goldfeld, Youssef Mroueh, Bharath K. Sriperumbudur*

    通过辅助矩阵变量把二次 GW 线性化为 OT/EOT 族的下确界，建立首个对偶理论与尖锐经验收敛率：GW 为 n^{−2/max{min(dx,dy),4}}，EGW 达参数率 n^{−1/2}

5. **MIRROR: Aligning Semantic Relations from Language to Image via Gromov--Wasserstein.** arXiv（自称 ECCV 2026 接收，待论文集核验）, 2026. [R] [paper](https://arxiv.org/abs/2606.29462) [report](reports/2606.29462.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2606.29462.pdf)

    *Hong-Han Wang, Yuntao Wang, Hu Ding*

    用 GW 型正则强制「概念间关系结构」在语言→视觉投影中保持，修复 MLLM 的关系推理盲区

6. **Private Synthetic Graph Generation and Fused Gromov-Wasserstein Distance.** AISTATS, 2026. [A] [paper](https://arxiv.org/abs/2502.11778) [report](reports/2502.11778.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2502.11778.pdf)

    *Leoni Carla Wirth, Gholamali Aminian, Gesine Reinert*

    顶点级 ε-DP 属性图生成器，并用 FGW 距离给出生成分布与真实分布的精度理论保证——FGW 作为图生成「度量+证明工具」

7. **Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild.** CVPR, 2026. [P] [paper](https://arxiv.org/abs/2603.11618) [report](reports/2603.11618.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2603.11618.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2603.11618.zh.pdf)

    *Jiin Im, Sisung Liu, Je Hyeong Hong*

    用 3D 结构先验 + anchor 线性化缓解 FGW 计算成本，做野外语义对应；展示 FGW 在视觉对应任务的工程化路径

8. **Semidefinite Relaxations of the Gromov-Wasserstein Distance.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2312.14572) [report](reports/2312.14572.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2312.14572.pdf)

    *Junyu Chen, Binh T. Nguyen, Shang Hui Koh, Yong Sheng Soh*

    GW 的 SDP 松弛给出可认证的全局下界与最优性证书，是非凸 GW「可认证计算」路线的代表

9. **Gromov-Wasserstein Autoencoders.** ICLR, 2023. [P] [paper](https://arxiv.org/abs/2209.07007) [report](reports/2209.07007.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2209.07007.pdf)

    *Nao Nakagawa, Ren Togo, Takahiro Ogawa, Miki Haseyama*

    抛弃似然目标，直接用 GW 度量匹配（不同维度的）latent 与 data 分布，把 meta-prior 表征学习变成跨空间结构匹配

10. **Linear-Time Gromov Wasserstein Distances using Low Rank Couplings and Costs.** ICML, 2022. [P] [paper](https://arxiv.org/abs/2106.01128) [report](reports/2106.01128.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2106.01128.pdf)

    *Meyer Scetbon, Gabriel Peyré, Marco Cuturi*

    低秩耦合 + 低秩代价分解把 GW 降至线性时间，是 moscot/OTT-JAX 规模化 GW 的算法基石

11. **Optimal Transport for structured data with application on graphs.** ICML, 2019. [P] [paper](https://arxiv.org/abs/1805.09114) [report](reports/1805.09114.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1805.09114.pdf)

    *Titouan Vayer, Laetitia Chapel, Rémi Flamary, Romain Tavenard, Nicolas Courty*

    FGW：特征项（Wasserstein）与结构项（GW）凸组合的联合传输，图比较、barycenter 与结构化生成的标准工具

12. **Gromov-Wasserstein Alignment of Word Embedding Spaces.** EMNLP, 2018. [P] [paper](https://arxiv.org/abs/1809.00013) [report](reports/1809.00013.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1809.00013.pdf)

    *David Alvarez-Melis, Tommi S. Jaakkola*

    无平行语料的跨语言词嵌入 GW 对齐，「嵌入空间结构对齐」整条线的奠基

13. **Gromov-Wasserstein Averaging of Kernel and Distance Matrices.** ICML, 2016. [P] [paper](https://proceedings.mlr.press/v48/peyre16.html) [report](reports/Gromov_Wasserstein_Averaging_of_Kernel_and_Distanc.md)

    *Peyré, Cuturi, Solomon*

    entropic GW 的投影镜像下降求解器与 GW barycenter，现代计算 GW 的起点

另见（跨课题重复）：GENOT: Entropic (Gromov) Wasserstein Flow Matching with Applications to Single-Cell Genomics → T24; Mapping Cells Through Time and Space with moscot → T24

<a id="t27"></a>
### T27. 多边际 OT 与 Wasserstein 重心的生成应用

课题综合：[`topics/t27.md`](topics/t27.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t27_multimarginal_barycenter_gen.md`](source/kb/t27_multimarginal_barycenter_gen.md)

1. ⭐ **Multimarginal flow matching with optimal transport potentials.** ICML, 2026. [A] [paper](https://arxiv.org/abs/2606.05327) [report](reports/2606.05327.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2606.05327.pdf)

    *Raghav Kansal, David Crair, Nghia Nguyen, Scott Pope, Bradley Parry*

    把逐段 CFM 重写为带硬约束的动态 OT，再把中间边缘约束松弛为动态 OT 作用量中的**势能项**，得到 simulation-free 的多边缘 FM，并给出势强度—Wasserstein 偏差界；单细胞/海洋/气象 SOTA

2. ⭐ **Sobolev Gradient Ascent for Optimal Transport: Barycenter Optimization and Convergence Analysis.** ICLR, 2026. [A] [paper](https://arxiv.org/abs/2505.13660) [report](reports/2505.13660.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.13660.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2505.13660.zh.pdf)

    *Kaheon Kim, Bohan Zhou, Changbo Zhu, Xiaohui Chen*

    精确（非熵正则）barycenter 的无约束凹对偶 + \(\dot H^1\) Sobolev 几何梯度上升；证明可去掉昂贵的 c-concave 投影仍有全局 \(O(T^{-1/2})\) 收敛率，每步 \(O(mn\log n)\)

3. ⭐ **Momentum Multi-Marginal Schrödinger Bridge Matching.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2506.10168) [report](reports/2506.10168.md)

    *Panagiotis Theodoropoulos, Augustinos D. Saravanos, Evangelos A. Theodorou, Guan-Horng Liu*

    相空间提升+多点条件化随机桥，学习满足多个位置约束的测度值样条；matching 迭代中保持中间边缘不变，解决两两插值丢失长程时序依赖的问题

4. ⭐ **Estimating Barycenters of Distributions with Neural Optimal Transport.** ICML, 2024. [P] [paper](https://arxiv.org/abs/2402.03828) [report](reports/2402.03828.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2402.03828.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2402.03828.zh.pdf)

    *Alexander Kolesov, Petr Mokrov, Igor Udovichenko, Milena Gazdieva, Gudmund Pammer, Evgeny Burnaev et al.*

    基于 Neural OT 对偶的双层对抗目标求连续 barycenter，支持一般代价（对比既有三层 min-max 且限于二次代价），含误差界；实验含 StyleGAN 潜空间

5. **Partial Fusion of Neural Networks via Partial Optimal Transport.** ICML, 2026. [A] [paper](https://openreview.net/forum?id=lvRLG6C0zZ) [report](reports/Partial_Fusion_of_Neural_Networks_via_Partial_Opti.md)

    用 partial OT 处理不同模型间神经元只有部分对应的情形，放松 OTFusion 的完全匹配假设

6. **Wasserstein Gradient Flows for Scalable and Regularized Barycenter Computation.** UAI, 2026. [P] [paper](https://arxiv.org/abs/2510.04602) [report](reports/2510.04602.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2510.04602.pdf)

    *Eduardo Fernandes Montesuma, Yassir Bendou, Mike Gartrell*

    把 barycenter 问题重写为 Wasserstein 空间中的梯度流：mini-batch OT 实现可扩展、支持模块化正则泛函（内能/势能/交互能）与监督化 ground cost，PL 条件下收敛保证；注：任务书中记为"NeurIPS 2025"，官方归属实为 UAI 2026

7. **A dynamical formulation of multi-marginal optimal transport.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2509.22494) [report](reports/2509.22494.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2509.22494.pdf)

    *Brendan Pass, Yair Shenfeld*

    首个一般 (semi-)convex 代价的 MMOT 原-对偶**动力学**形式（耦合流而非边缘流），凸优化可解并给出 quasi-Monge 解；为"多边缘 Benamou-Brenier"补上缺失的一块

8. **Finding the Center of a Wasserstein Ball.** ICML, 2025. [P] [paper](https://proceedings.mlr.press/v267/wang25be.html) [report](reports/Finding_the_Center_of_a_Wasserstein_Ball.md)

    Wasserstein ball 中心=一种 min-max 鲁棒聚合，与 barycenter 互补的"最坏情况平均"视角

9. **The Procrustes-Wasserstein Barycenter Problem.** ICML, 2025. [P] [paper](https://proceedings.mlr.press/v267/adamo25a.html) [report](reports/The_Procrustes_Wasserstein_Barycenter_Problem.md)

    barycenter 与正交/刚体对齐联合优化，解决输入分布姿态不对齐时平均失真的问题

10. **Wukong's 72 Transformations: High-fidelity Textured 3D Morphing via Flow Models.** arXiv, 2025. [R] [paper](https://arxiv.org/abs/2511.22425) [report](reports/2511.22425.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2511.22425.pdf)

    *Minghao Yin, Yukang Cao, Kai Han*

    在预训练 3D flow transformer 的**条件 token 空间**解 free-support barycenter 得到插值条件，实现 training-free 高保真 3D 形变+纹理渐变——barycenter×预训练流模型的代表性应用

11. **Energy-Guided Continuous Entropic Barycenter Estimation for General Costs.** NeurIPS Spotlight, 2024. [P] [paper](https://arxiv.org/abs/2310.01105) [report](reports/2310.01105.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.01105.pdf)

    *Alexander Kolesov, Petr Mokrov, Igor Udovichenko, Milena Gazdieva, Gudmund Pammer, Anastasis Kratsios et al.*

    弱 OT 对偶+EBM 学连续熵正则 barycenter，免 min-max，带质量界；直接在预训练生成模型的图像流形上学 barycenter

12. **Tree-Based Diffusion Schrödinger Bridge with Applications to Wasserstein Barycenters.** NeurIPS Spotlight, 2023. [P] [paper](https://arxiv.org/abs/2305.16557) [report](reports/2305.16557.md)

    *Maxence Noble, Valentin De Bortoli, Arnaud Doucet, Alain Durmus*

    树结构代价的熵正则 MMOT 的动态连续版本（多边缘 Sinkhorn 的 DSB 对应物）；星形树即 barycenter，可在高维做图像插值与贝叶斯融合——「用扩散模型算 barycenter」的奠基工作

13. **Model Fusion via Optimal Transport.** NeurIPS, 2020. [P] [paper](https://arxiv.org/abs/1910.05653) [report](reports/1910.05653.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1910.05653.pdf)

    *Sidak Pal Singh, Martin Jaggi*

    逐层用 OT 对齐神经元再平均，显式解释为逐层 Wasserstein barycenter；一次性(one-shot)、无需训练数据的模型融合奠基

另见（跨课题重复）：Modeling Complex System Dynamics with Flow Matching Across Time and Conditions → T24; Multi-marginal temporal Schrödinger Bridge Matching from unpaired data → T03

<a id="t28"></a>
### T28. 黎曼流形上的流匹配与 OT

课题综合：[`topics/t28.md`](topics/t28.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t28_riemannian_manifold_fm.md`](source/kb/t28_riemannian_manifold_fm.md)

1. ⭐ **Riemannian Neural Optimal Transport.** arXiv, 2026. [R] [paper](https://arxiv.org/abs/2602.03566) [report](reports/2602.03566.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2602.03566.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2602.03566.zh.pdf)

    *Alessandro Micheli, Yueqi Cao, Anthea Monod, Samir Bhatt*

    证明离散化流形 OT 必有维数灾难；用 c-凹神经势 \(T=\exp_x(-\nabla\phi)\) 学连续流形 OT map，次指数复杂度保证

2. ⭐ **Wasserstein Flow Matching: Generative modeling over families of distributions.** ICML, 2025. [P] [paper](https://arxiv.org/abs/2411.00698) [report](reports/2411.00698.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.00698.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2411.00698.zh.pdf)

    *Doron Haviv, Aram-Alexandre Pooladian, Dana Pe'er, Brandon Amos*

    把 FM 提升到"分布的分布"：证明 Wasserstein 测地是合法条件流，高斯族用闭式 Bures-Wasserstein 路径、点云用熵 OT 估计

3. ⭐ **Flow Matching on General Geometries.** ICLR Oral, 2024. [P] [paper](https://arxiv.org/abs/2302.03660) [report](reports/2302.03660.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2302.03660.pdf)

    *Ricky T. Q. Chen, Yaron Lipman*

    用 premetric（测地/谱距离）闭式构造流形条件向量场，简单几何上完全 simulation-free，奠定黎曼 FM 范式

4. ⭐ **Metric Flow Matching for Smooth Interpolations on the Data Manifold.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2405.14780) [report](reports/2405.14780.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.14780.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2405.14780.zh.pdf)

    *Kacper Kapuśniak, Peter Potaptchik, Teodora Reu, Leo Zhang, Alexander Tong, Michael Bronstein et al.*

    在数据诱导度量下学最小动能插值（近似测地线）替代直线插值，OT-MFM 在单细胞轨迹上 SOTA；"数据流形"版黎曼 FM

5. ⭐ **Riemannian Score-Based Generative Modelling.** NeurIPS, 2022. [P] [paper](https://arxiv.org/abs/2202.02763) [report](reports/2202.02763.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2202.02763.pdf)

    *Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, Arnaud Doucet*

    把 SGM 的前向/反向 SDE 定义到紧流形（测地随机游走+热核），开创流形扩散并给出地球科学球面基准

6. **Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2505.18193) [report](reports/2505.18193.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2505.18193.pdf)

    *Antoine Collas, Ce Ju, Nicolas Salvy, Bertrand Thirion*

    全局微分同胚 pullback 度量下的黎曼 CFM 等价于"变换后做欧氏 CFM"：SPD 用矩阵对数、相关矩阵用归一化 Cholesky，fMRI/EEG 大规模验证

7. **Riemannian Consistency Model.** NeurIPS, 2025. [P] [paper](https://arxiv.org/abs/2510.00983) [report](reports/2510.00983.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2510.00983.pdf)

    *Chaoran Cheng, Yusong Wang, Yuxin Chen, Xiangxin Zhou, Nanning Zheng, Ge Liu*

    用协变导数+指数映射参数化把一致性模型推广到流形，蒸馏(RCD)与从头训练(RCT)理论等价，球面/环面/SO(3) 少步生成

8. **Riemannian Proximal Sampler, Guan, Balasubramanian & Ma.** NeurIPS, 2025. [P] [paper](https://papers.nips.cc/paper_files/paper/2025/hash/8e185f16e458ef5e666901260079cd42-Abstract-Conference.html) [report](reports/Riemannian_Proximal_Sampler_Guan_Balasubramanian_M.md)

    MBI+热核双 oracle 的流形高精度采样，\(O(\log(1/\varepsilon))\) 迭代；可解释为 Wasserstein 空间上熵正则黎曼 proximal point 的离散化

9. **Stochastic variance-reduced Gaussian variational inference on the Bures-Wasserstein manifold.** ICLR, 2025. [A] [paper](https://arxiv.org/abs/2410.02490) [report](reports/2410.02490.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2410.02490.pdf)

    *Hoang Phuc Hau Luu, Hanlin Yu, Bernardo Williams, Marcelo Hartmann, Arto Klami*

    BW 流形（高斯族 Wasserstein 几何）上的方差缩减变分推断，完善 BW 空间一阶优化工具箱

10. **Trivialized Momentum Facilitates Diffusion Generative Modeling on Lie Groups.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2405.16381) [report](reports/2405.16381.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.16381.pdf)

    *Yuchen Zhu, Tianrong Chen, Lingkai Kong, Evangelos A. Theodorou, Molei Tao*

    李群上引入平凡化动量：score 在固定李代数（平坦空间）学习，无投影/切空间近似，首次做高维 SO(n)/U(n) 生成

11. **Riemannian Diffusion Mixture, Jo & Hwang.** ICML, 2024. [P] [paper](https://proceedings.mlr.press/v235/jo24a.html) [report](reports/Riemannian_Diffusion_Mixture_Jo_Hwang.md)

    用桥过程混合直接构造生成扩散（漂移=数据方向切向量加权平均），绕开热核估计与散度计算，一般流形可扩展

12. **Riemannian Flow Matching Policy (RFMP), Braun et al.** IROS, 2024. [P] [paper](https://doi.org/10.1109/iros58592.2024.10801521) [report](reports/Riemannian_Flow_Matching_Policy_RFMP_Braun_et_al.md)

    把黎曼 FM 用于机器人视觉运动策略（状态含姿态流形），比 Diffusion Policy 更平滑、推理更快

13. **Statistical/Categorical Flow Matching (SFM), Cheng et al.** NeurIPS, 2024. [P] [paper](https://openreview.net/forum?id=5fybcQZ0g4) [report](reports/Statistical_Categorical_Flow_Matching_SFM_Cheng_et.md)

    统计流形（Fisher 信息度量）上的 FM：测地最短路+自然梯度解释+精确似然，训练中可加 OT

14. **Riemannian Diffusion Schrödinger Bridge.** arXiv, 2022. [R] [paper](https://arxiv.org/abs/2207.03024) [report](reports/2207.03024.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2207.03024.pdf)

    *James Thornton, Michael Hutchinson, Emile Mathieu, Valentin De Bortoli, Yee Whye Teh, Arnaud Doucet*

    把 DSB/IPF 推广到紧流形，做流形上两分布间 SB 插值（地球气候数据），是流形熵 OT 动态解法源头

另见（跨课题重复）：Fisher Flow Matching for Generative Modeling over Discrete Data → T22

<a id="sec-f"></a>
## F. 系统、评测与趋势

<a id="t29"></a>
### T29. 高性能 OT 求解器与训练基础设施

课题综合：[`topics/t29.md`](topics/t29.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t29_ot_solvers_infra.md`](source/kb/t29_ot_solvers_infra.md)

1. ⭐ **A Memory-Efficient Hierarchical Algorithm for Large-scale OT (HALO).** ICLR Poster, 2026. [A] [paper](https://openreview.net/forum?id=CkOBcyntGd) [report](reports/A_Memory_Efficient_Hierarchical_Algorithm_for_Larg.md)

    层次多尺度 warm-start + active support 剪枝 + 因子化-free 一阶 LP 求解器（默认 cuPDLPx），O(n) 内存；1024² 像素图像 8.9× 加速、省 70.5% 显存

2. ⭐ **FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU.** ICML Oral, 2026. [A] [paper](https://arxiv.org/abs/2602.03067) [report](reports/2602.03067.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2602.03067.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2602.03067.zh.pdf)

    *Felix X. -F. Ye, Xingjie Li, An Yu, Ming-Ching Chang, Linsong Chu, Davis Wertheimer*

    把 log-domain Sinkhorn 更新重写为 attention 同构的 online-LSE，Triton 融合 kernel 流式过 SRAM，O(nd) 显存 + 解析梯度/HVP/半对偶 c-transform；A100 上比 KeOps 前向快 9–32×、端到端最高 161×

3. ⭐ **Hierarchical Refinement: Optimal Transport to Infinity and Beyond.** ICML Oral, 2025. [P] [paper](https://arxiv.org/abs/2503.03025) [report](reports/2503.03025.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2503.03025.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2503.03025.zh.pdf)

    *Peter Halmos, Julian Gold, Xinhao Liu, Benjamin J. Raphael*

    证明低秩耦合因子与 Monge map 共聚类，用低秩 OT 递归构造多尺度划分、log-linear 时间/线性空间恢复**全秩双射**，百万点规模超出 Sinkhorn 可及范围

4. **Fast Log-Domain Sinkhorn Optimal Transport with Warp-Level GPU Reductions.** arXiv, 2026. [R] [paper](https://arxiv.org/abs/2605.00837) [report](reports/2605.00837.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2605.00837.pdf)

    *Hao Xiao*

    原生 CUDA warp-level shuffle 归约 + shared-memory tiling 的 log-domain Sinkhorn；ε 低至 1e-4 仍稳定，n=8192 时比 POT 快 12×、仅 256MB 显存

5. **cuRegOT: A GPU-Accelerated Solver for Entropic-Regularized Optimal Transport.** arXiv, 2026. [R] [paper](https://arxiv.org/abs/2605.08793) [report](reports/2605.08793.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2605.08793.pdf)

    *Yixuan Qiu*

    把 sparse-plus-low-rank 拟牛顿法 GPU 化：摊销稀疏符号分析、CPU/GPU 异步流水、融合梯度 kernel；在难例（小 η=0.001）上显著快于 POT/OTT-JAX/AccSinkhorn

6. **Accelerating Sinkhorn Algorithm with Sparse Newton Iterations.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2401.12253) [report](reports/2401.12253.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2401.12253.pdf)

    *Xun Tang, Michael Shavlovsky, Holakou Rahmanian, Elisa Tardini, Kiran Koshy Thekumparampil, Tesi Xiao et al.*

    Sinkhorn 一阶缩放后接 Hessian 稀疏化的 Newton 迭代，超线性收敛；与硬件加速正交、可叠加

7. **Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2411.10555) [report](reports/2411.10555.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.10555.pdf)

    *Peter Halmos, Xinhao Liu, Julian Gold, Benjamin J Raphael*

    latent coupling 因子化把低秩 OT 解耦成三个子 OT 问题，坐标镜像下降求解；统一 W/GW/FGW × 均衡/非均衡/半松弛，线性空间

8. **PDOT: a Practical Primal-Dual Algorithm and a GPU-Based Solver for Optimal Transport.** arXiv, 2024. [R] [paper](https://arxiv.org/abs/2407.19689) [report](reports/2407.19689.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2407.19689.pdf)

    *Haihao Lu, Jinwen Yang*

    用 restarted PDHG（cuPDLP 血统）做 matrix-free 高精度 OT：数据无关 O(1/ε) 复杂度，GPU 上高精度区间胜过 Sinkhorn 与商用 LP

9. **Quasi-Monte Carlo for 3D Sliced Wasserstein.** ICLR, 2024. [P] [paper](https://arxiv.org/abs/2309.11713) [report](reports/2309.11713.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2309.11713.pdf)

    *Khai Nguyen, Nicola Bariletto, Nhat Ho*

    用球面低差异点集替代 MC 投影方向，系统评测多种 QMC 构造；RQSW 随机化后保无偏可做 SGD——sliced 系的数值工程标杆

10. **Low-Rank Sinkhorn Factorization.** ICML, 2021. [P] [paper](https://arxiv.org/abs/2103.04737) [report](reports/2103.04737.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2103.04737.pdf)

    *Meyer Scetbon, Marco Cuturi, Gabriel Peyré*

    不近似核而直接约束耦合的非负秩，任意 cost 通用；"低秩耦合"路线的源头，OTT-JAX 内置实现

11. **Massively scalable Sinkhorn distances via the Nyström method.** NeurIPS, 2019. [P] [paper](https://arxiv.org/abs/1812.05189) [report](reports/1812.05189.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/1812.05189.pdf)

    *Jason Altschuler, Francis Bach, Alessandro Rudi, Jonathan Niles-Weed*

    Nyström 低秩核近似 + Sinkhorn 缩放的稳定性分析，近线性时间/内存；"核近似"路线的源头

另见（跨课题重复）：Progressive Entropic Optimal Transport Solvers → T04; Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment → T08; Expected Batch Optimal Transport Plans and Consequences for Flow Matching → T08; Minibatch Optimal Transport and Perplexity Bound Estimation in Discrete Flow Matching → T22

<a id="t30"></a>
### T30. 端侧部署、benchmark 与顶会趋势（博客落地场景：端侧图像生成）

课题综合：[`topics/t30.md`](topics/t30.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`source/kb/t30_edge_benchmark_trends.md`](source/kb/t30_edge_benchmark_trends.md)

1. ⭐ **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models.** ICLR, 2025. [P] [paper](https://arxiv.org/abs/2411.05007) [report](reports/2411.05007.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2411.05007.pdf)

    *Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo et al.*

    W4A4 量化范式：smoothing 把激活离群值移到权重，SVD 低秩分支吸收权重离群值；与 Nunchaku 引擎 kernel 融合 co-design，12B FLUX 跑进 16GB 笔记本 4090（3×提速）且免重量化支持 LoRA

2. ⭐ **SnapGen: Taming High-Resolution Text-to-Image Models for Mobile Devices.** CVPR Highlight, 2025. [P] [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_SnapGen_Taming_High-Resolution_Text-to-Image_Models_for_Mobile_Devices_with_Efficient_CVPR_2025_paper.pdf) [report](reports/SnapGen_Taming_High_Resolution_Text_to_Image_Model.md)

    *Chen et al.*

    端侧模型**从头训练**新范式：379M UNet 宏/微架构搜索 + 跨架构多级蒸馏（教师 SD3.5-Large）+ 对抗步蒸馏，iPhone 16 Pro-Max 1024² ~1.4s，GenEval 0.66 超 SDXL（7×大）

3. ⭐ **MobileDiffusion: Instant Text-to-Image Generation on Mobile Devices.** ECCV, 2024. [P] [paper](https://arxiv.org/abs/2311.16567) [report](reports/2311.16567.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2311.16567.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2311.16567.zh.pdf)

    *Yang Zhao, Yanwu Xu, Zhisheng Xiao, Haolin Jia, Tingbo Hou*

    首个系统性的移动端扩散**架构设计空间研究**（<400M UNet）+ diffusion-GAN 一步采样兼容下游任务，iPhone 15 Pro 0.2s/512²

4. ⭐ **Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models.** NeurIPS, 2023. [P] [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0bc795afae289ed465a65a3b4b1f4eb7-Abstract-Conference.html) [report](reports/Exposing_flaws_of_generative_model_evaluation_metr.md)

    *Stein et al.*

    最大规模人评心理物理实验：**没有任何现有指标与人评强相关**；Inception-V3 特征系统性压低扩散模型排名；建议全面换用 DINOv2-ViT-L/14 特征算 FD

5. ⭐ **SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds.** NeurIPS, 2023. [P] [paper](https://arxiv.org/abs/2306.00980) [report](reports/2306.00980.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2306.00980.pdf) [zh-PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-zh-v1/2306.00980.zh.pdf)

    *Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu et al.*

    端侧 T2I 开山：高效 UNet（冗余块识别+数据蒸馏压 VAE decoder）+ CFG 正则化步蒸馏，iPhone 14 Pro 上 8 步 <2s，FID/CLIP 超 SD1.5-50 步

6. **SANA: Efficient High-Resolution Text-to-Image Synthesis with Linear Diffusion Transformers.** ICLR Oral, 2025. [A] [paper](https://openreview.net/forum?id=N8Oj1XhtYZ) [report](reports/SANA_Efficient_High_Resolution_Text_to_Image_Synth.md)

    *Xie et al.*

    可部署 DiT 路线：32× 深压缩 AE + linear attention DiT + 小 LLM 文本编码器 + FM 训练/Flow-DPM-Solver，0.6B 模型笔记本 GPU <1s/1024²，配 SVDQuant 4-bit 跑进 8GB

7. **BitsFusion: 1.99 bits Weight Quantization of Diffusion Model.** NeurIPS, 2024. [P] [paper](https://arxiv.org/abs/2406.04333) [report](reports/2406.04333.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2406.04333.pdf)

    *Yang Sui, Yanyu Li, Anil Kag, Yerlan Idelbayev, Junli Cao, Ju Hu et al.*

    QAT 极限压缩：逐层最优比特分配 + 量化初始化 + 两阶段蒸馏训练，SD1.5 UNet 1.72GB→219MB（1.99 bit）且 TIFA/GenEval/人评反超全精度

8. **EdgeFusion: On-Device Text-to-Image Generation.** arXiv（CVPR-W 系）, 2024. [R] [paper](https://arxiv.org/abs/2404.11925) [report](reports/2404.11925.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2404.11925.pdf)

    *Thibault Castells, Hyoung-Kyu Song, Tairen Piao, Shinkook Choi, Bo-Kyeong Kim, Hanyoung Yim et al.*

    NPU 全栈工程样本：BK-SDM-Tiny + 改进 LCM 蒸馏 + 合成数据，W8/A16(UNet INT8 权重+INT16 激活)混合精度 + model-level tiling + kernel fusion，三星 Exynos 2400 NPU 2 步 <1s

9. **MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization.** ECCV, 2024. [P] [paper](https://arxiv.org/abs/2405.17873) [report](reports/2405.17873.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2405.17873.pdf)

    *Tianchen Zhao, Xuefei Ning, Tongcheng Fang, Enshu Liu, Guyue Huang, Zinan Lin et al.*

    **少步模型专用 PTQ**：发现 1 步 SDXL-Turbo 量化瓶颈在文本嵌入 BOS 离群值；BOS-aware 量化 + 指标解耦敏感度分析 + 整数规划配比特，W4A8 仅 +0.5 FID（基线全崩）

10. **Rethinking FID: Towards a Better Evaluation Metric for Image Generation.** CVPR Highlight, 2024. [P] [paper](https://arxiv.org/abs/2401.09603) [report](reports/2401.09603.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2401.09603.pdf)

    *Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, Sanjiv Kumar*

    指出 FID 三宗罪：正态假设不成立、样本复杂度差、与人评矛盾（无法反映 T2I 迭代改进）；提出 CMMD = CLIP 嵌入 + Gaussian RBF 核 MMD（无分布假设、无偏、样本高效）

11. **SDXS: Real-Time One-Step Latent Diffusion Models with Image Conditions.** arXiv, 2024. [R] [paper](https://arxiv.org/abs/2403.16627) [report](reports/2403.16627.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2403.16627.pdf)

    *Yuda Song, Zehao Sun, Xuanwu Yin*

    UNet+VAE 双小型化 + 特征匹配/分数蒸馏的一步训练（显式做轨迹拉直 straightening），512² 达 100 FPS；ControlNet 蒸馏支持实时 image-to-image

12. **GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment.** NeurIPS D&B, 2023. [P] [paper](https://arxiv.org/abs/2310.11513) [report](reports/2310.11513.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2310.11513.pdf)

    *Dhruba Ghosh, Hanna Hajishirzi, Ludwig Schmidt*

    用目标检测器做组合能力评测（共现/计数/颜色/位置），实例级可解释、与人评强一致；已成端侧论文标配指标（SnapGen/SANA 均报告）

13. **Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis.** arXiv, 2023. [R] [paper](https://arxiv.org/abs/2306.09341) [report](reports/2306.09341.md) [PDF](https://github.com/asimfish/awesome_diffusion_OT/releases/download/pdf-en-v1/2306.09341.pdf)

    *Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao et al.*

    79.8 万人类偏好对训练的偏好模型 + HPD v2 基准，作为可扩展的"人评代理"；MobileDiffusion 等端侧工作用其验证主观质量

<a id="trends"></a>
## G. 2026 Q3 增量与趋势

扫描日期 2026-09-01；完整分析见 [`trends/TRENDS_2026Q3.md`](trends/TRENDS_2026Q3.md)。

1. **ReBridge-Flow: Re-Coupling Posterior Bridges in Flow Matching for Image Restoration.** arXiv 2026-09 [R] → T14. [paper](https://arxiv.org/abs/2609.00811)

    提出后验桥重耦合方法，修复流匹配图像复原中的桥失配。

2. **A Lagrangian View of Flow Matching.** arXiv 2026-08 [R] → T09. [paper](https://arxiv.org/abs/2609.00198)

    从拉格朗日视角推导Flow Matching的直线轨迹，指出去噪器雅可比是曲率主因。

3. **PixelIR: Fidelity-Perception Decoupling via Pixel-Space Image-Residual Flow Matching for Efficient One-Step Real-World Super-Resolution.** arXiv 2026-08 [R] → T07. [paper](https://arxiv.org/abs/2608.30782)

    像素空间图像-残差流匹配实现一步真实世界超分。

4. **Efficient primal--dual splitting methods for a Poisson-constrained JKO scheme for Poisson-Nernst-Planck models.** arXiv 2026-08 [R] → T05. [paper](https://arxiv.org/abs/2608.30693)

    提出Poisson约束JKO格式的高效原始对偶分裂方法求解PNP方程。

5. **Reward-guided Fine-Tuning of One-Step Generative Models via Wasserstein Gradient Flow.** arXiv 2026-08 [R] → T05. [paper](https://arxiv.org/abs/2608.29647)

    用Wasserstein梯度流对单步生成模型做奖励引导微调，无需奖励梯度。

6. **Discrete Diffusion Bridges for Spatiotemporally Aligned Image Translation and Generation.** arXiv 2026-08 [A:ECCV 2026] → T14. [paper](https://arxiv.org/abs/2608.29997)

    提出DDB，通过混合吸收机制和信息引导噪声调度构建离散扩散桥，解决图像翻译的时空错位。

7. **Training-Free Hidden-State Refinement for Flow-Matching Image Generators.** arXiv 2026-08 [R] → T11. [paper](https://arxiv.org/abs/2608.29160)

    免训练隐藏状态细化提升冻结流匹配图像生成器质量。

8. **Di$^2$CycleSB: Towards High-Quality Unsupervised Nighttime Visibility Enhancement via Schrödinger Bridge Transformer.** arXiv 2026-08 [R] → T14. [paper](https://arxiv.org/abs/2608.29043)

    提出Di²CycleSB，用循环Schrödinger桥Transformer实现无监督夜间图像增强。

9. **There and Back Again: Bidirectional Diffusion Bridges for Multimodality Translation.** arXiv 2026-08 [R] → T03. [paper](https://arxiv.org/abs/2608.27885)

    提出BIT，构建文本与图像之间的双向扩散桥，实现统一的生成与反演框架。

10. **Quantitative Target Convergence and Uniform-in-Time Propagation of Chaos for Langevin-Regularized SVGD.** arXiv 2026-08 [R] → T05. [paper](https://arxiv.org/abs/2608.28827)

    证明Langevin正则化SVGD在Wasserstein几何下指数收敛并传播混沌。

11. **Generation of High-Level Concepts in 3D Scene Graphs via Autoregressive Diffusion.** arXiv 2026-08 [R] → T26. [paper](https://arxiv.org/abs/2608.28733)

    用自回归扩散生成3D场景图，并提出Fused Gromov-Wasserstein评估图结构。

12. **Gromov-Monge Flow Matching for Equivariant Graph Generation.** arXiv 2026-08 [R] → T26. [paper](https://arxiv.org/abs/2608.26961)

    提出Gromov-Monge流匹配，用Gromov-Wasserstein型松弛构造图生成中的等变耦合。

13. **Mechanistic Reaction Prediction via Discrete Flow Matching on Graph-Structured Electron Occupation.** arXiv 2026-08 [R] → T21. [paper](https://arxiv.org/abs/2608.27429)

    用图结构电子占据空间上的离散流匹配建模化学反应，以最优传输构造电子移动插值。

14. **Hard-Constrained Sampling on Embedded Riemannian Manifolds via Adjoint Schrödinger Bridges.** arXiv 2026-08 [R] → T28. [paper](https://arxiv.org/abs/2608.25838)

    通过伴随Schrödinger桥在嵌入黎曼流形上实现硬约束采样。

15. **Fast Generative Grasping via Lie Group-Constrained MeanFlow.** arXiv 2026-08 [R] → T28. [paper](https://arxiv.org/abs/2608.26076)

    在SO(3)×R^3李群上用条件流匹配实现快速抓取生成。

16. **Schrödinger Bridges over Kinetic Swarming Models.** arXiv 2026-08 [R] → T03. [paper](https://arxiv.org/abs/2608.25281)

    用Schrödinger桥理论解决惯性群体模型的有限时域最小能量集体转向问题。

17. **Noising-Denoising by Large Temperature Schrödinger Bridges.** arXiv 2026-08 [R] → T03. [paper](https://arxiv.org/abs/2608.25094)

    建立去噪扩散模型与高温动态Schrödinger桥之间的过程级联系。

18. **MoRF-AST: Calibrated Probabilistic Virtual Sensing for Structural Monitoring under Changing Operating Conditions.** arXiv 2026-08 [R] → T17. [paper](https://arxiv.org/abs/2608.24531)

    用条件流匹配与Bures-Wasserstein传输实现结构监测中的校准虚拟传感。

19. **Through the Schrödinger Bridge: Benchmarking Antemortem Image Restoration from Postmortem Autolysis to Enhance Forensic Diagnostics.** arXiv 2026-08 [R] → T15. [paper](https://arxiv.org/abs/2608.21813)

    将法医组织病理学自溶恢复形式化为Schrödinger桥问题，并构建首个同源非配对数据集。

20. **SketchFlow: Zero-Shot Vector Sketch Generation via GMM Prior Flow in CLIP Latent Space.** arXiv 2026-08 [A:SIGGRAPH Asia 2026] → T08. [paper](https://arxiv.org/abs/2608.21659)

    提出SketchFlow，用OT-CFM在CLIP空间做零样本矢量草图生成。

21. **TracingFlow: A Simulation-Free Trajectory Inference Framework Based on Second-Order Dynamics.** arXiv 2026-08 [R] → T24. [paper](https://arxiv.org/abs/2608.21070)

    提出TracingFlow，用二阶动力学流匹配做单细胞轨迹推断。

22. **ReCurveflow: A Flow Matching Framework that Learns Curved Reaction Trajectories to Predict Transition State Geometries.** arXiv 2026-08 [R] → T21. [paper](https://arxiv.org/abs/2608.20869)

    用流匹配学习弯曲反应轨迹预测过渡态几何。

23. **Probabilistic Representation and Convergence of Gromov-Wasserstein Gradient Flows.** arXiv 2026-08 [R] → T26. [paper](https://arxiv.org/abs/2608.19198)

    研究Gromov-Wasserstein梯度流的概率表示与指数收敛。

24. **Pathology Transport: Optimal-Transport Explanations for Clinical Data, and When Their Heatmaps (Fail to) Localize Disease.** arXiv 2026-08 [R] → T09. [paper](https://arxiv.org/abs/2608.17370)

    用OT整流流在临床分布间生成解释热图并检验其定位能力。

25. **EDITBRIDGE: Towards Faithful and Efficient Ultra-High-Resolution Image Editing.** arXiv 2026-08 [R] → T14. [paper](https://arxiv.org/abs/2608.18063)

    提出 EditBridge，用扩散桥实现低分辨率编辑结果到高分辨率的结构化转换，支持 4K 编辑。

26. **Cyclops: LiDAR as a Camera That Dreams in Color.** arXiv 2026-08 [R] → T07. [paper](https://arxiv.org/abs/2608.16264)

    Cyclops 用 Latent Bridge Matching 将稀疏 LiDAR 强度转换为 RGB 视频，支持全天候感知。

27. **The Distributional View of Knowledge Distillation.** arXiv 2026-08 [R] → T04. [paper](https://arxiv.org/abs/2608.15215)

    将知识蒸馏视为分布匹配，提出多温度视角下的熵正则 Wasserstein 重心与 Sinkhorn 散度目标。

28. **Nonparametric Schrödinger Bridge Time Series Generator: Algorithm, Convergence Analysis and Applications.** arXiv 2026-08 [R] → T03. [paper](https://arxiv.org/abs/2608.13968)

    给出 Schrödinger Bridge 时间序列生成器的完整分布收敛分析，并验证 SDE 参考测度的灵活性。

29. **Generation-Powered Inference for Distribution-Valued Outcomes.** arXiv 2026-08 [R] → T27. [paper](https://arxiv.org/abs/2608.14542)

    提出生成驱动推断框架，利用Wasserstein重心桥表示提升分布值参数估计效率。

30. **Information-Calibrated Quantum Diffusion: Aligning Forward Noise with Reverse Recoverability.** arXiv 2026-08 [R] → T02. [paper](https://arxiv.org/abs/2608.14083)

    引入信息校准坐标对齐量子扩散前向噪声与逆向可恢复性，提升生成性能。

31. **HybridSB-MoE: Dual-Domain Schrödinger Bridges with Scene-Adaptive Expert Routing for Speech Enhancement.** arXiv 2026-08 [R] → T23. [paper](https://arxiv.org/abs/2608.12715)

    提出双域Schrödinger桥语音增强框架，结合MoE路由并给出Wasserstein采样误差界。

32. **On Bridging Mixture Distributions.** arXiv 2026-08 [R] → T03. [paper](https://arxiv.org/abs/2608.13383)

    研究混合分布间的 Schrödinger Bridge，给出高斯混合桥的 Wasserstein 连续性界。

33. **Wasserstein Filtering: A Sample Selection Method for Robust Distribution Learning.** arXiv 2026-08 [R] → T25. [paper](https://arxiv.org/abs/2608.13418)

    提出Wasserstein过滤样本选择方法，通过最大化Wasserstein距离剔除异常值。

34. **KANResDiff: Learning Local Residual Diffusion via Kolmogorov-Arnold Network for Ambiguous Medical Image Segmentation.** arXiv 2026-08 [A:MICCAI 2026] → T14. [paper](https://arxiv.org/abs/2608.11617)

    提出KANResDiff，用局部残差Schrödinger桥实现医学图像模糊分割的渐进语义建模。

35. **Fine-Tuning Generative Models for Extreme Events via CVaR-Penalized Wasserstein Gradient Flows.** arXiv 2026-08 [R] → T05. [paper](https://arxiv.org/abs/2608.11544)

    提出CVaR惩罚的Wasserstein梯度流算法，微调生成模型以捕获重尾极端事件。

36. **MiDashengLM-Gen: Unified Audio Scene Generation via LLM-Driven Autoregressive Flow Matching.** arXiv 2026-08 [R] → T23. [paper](https://arxiv.org/abs/2608.11804)

    用LLM驱动逐token条件流匹配，统一生成混合音频场景，语音清晰度大幅提升。

37. **Phoenix TTS: High-Fidelity Synthesis and Voice Conversion via Flow-Matching-Driven Speech Tokenization.** arXiv 2026-08 [R] → T23. [paper](https://arxiv.org/abs/2608.11737)

    语音tokenizer与流匹配声学模型联合训练，弥合离散token与连续空间鸿沟。

38. **Marrying Optimal Transport and ODEs for Unified Continuous-Time 4D Reconstruction and Tracking.** arXiv 2026-08 [R] → T08. [paper](https://arxiv.org/abs/2608.09613)

    提出 Uni4R，用 OT 定义的概率路径与 Flow Matching 引导解码器统一连续时间 4D 重建与跟踪。

39. **DoseBridge: Denoising Diffusion Bridge Model for Dose Prediction in Lung Intensity-Modulated Proton Therapy.** arXiv 2026-08 [R] → T15. [paper](https://arxiv.org/abs/2608.10173)

    DoseBridge 用去噪扩散桥模型预测质子治疗剂量，以 CT 为桥端点并编码束几何。

40. **SDDBMs: Soft Denoising Diffusion Bridge Models.** arXiv 2026-08 [R] → T03. [paper](https://arxiv.org/abs/2608.08594)

    提出软去噪扩散桥模型，用非退化高斯终端边际正则化端点约束，统一多种桥模型。

41. **MRI super-resolution in ten sampling steps using a diffusion bridge model.** arXiv 2026-08 [R] → T15. [paper](https://arxiv.org/abs/2608.08819)

    SR-DBM 用扩散桥模型在十步内完成 MRI 超分辨率，从 LR 直接重建 HR。

42. **Compositional Cross-Modality Translation via Whole-Volume Multitask Latent Flow Matching.** arXiv 2026-08 [A:Sashimi 2026] → T15. [paper](https://arxiv.org/abs/2608.08135)

    用3D VAE潜空间条件流匹配实现全容积多任务医学图像翻译。

43. **Limit Points of Reflow with Minibatch Optimal Transport.** arXiv 2026-08 [R] → T09. [paper](https://arxiv.org/abs/2608.07042)

    证明 Reflow 与固定批量 minibatch OT 交替迭代的极限是 N-循环单调耦合，并在梯度场条件下收敛到 OT 映射。

44. **UniCycleFlow: Bidirectional Unpaired Image Translation with a Shared Rectified Flow.** arXiv 2026-08 [R] → T09. [paper](https://arxiv.org/abs/2608.06784)

    用共享整流流场统一双向无配对图像翻译，单次Euler即可生成。

45. **Pre- to Post-Contrast Synthesis of Breast DCE-MRI using Latent Bridge Matching.** arXiv 2026-08 [R] → T14. [paper](https://arxiv.org/abs/2608.10000)

    提出潜在桥匹配框架，从预对比图像合成乳腺DCE-MRI峰值增强图像。

46. **PRISM: Principled Reference Identification for Schrodinger Bridge Model.** arXiv 2026-08 [R] → T03. [paper](https://arxiv.org/abs/2608.06893)

    提出PRISM理论，刻画Schrödinger桥参考过程的最优设计。

47. **FUSE: Feature-Wise Unified Specialization with Cross-Column Exchange for Mixed-Type Tabular Flow Matching.** arXiv 2026-08 [R] → T07. [paper](https://arxiv.org/abs/2608.07294)

    FUSE 为混合类型表格数据设计特征级专业化与跨列交换的流匹配框架，并给出 Wasserstein 生成误差界。

48. **Hierarchical Flow Matching for 3D Point Cloud Generation.** arXiv 2026-08 [R] → T20. [paper](https://arxiv.org/abs/2608.05557)

    提出双层流匹配，用OT路径在潜空间和点空间分别建模全局形状与局部细节，15步采样即可生成3D点云。

49. **LC-GRPO: Bridging Train-Inference Gap for Flow-Based GRPO with Langevin Correction.** arXiv 2026-08 [R] → T11. [paper](https://arxiv.org/abs/2608.05600)

    LC-GRPO 在流模型 GRPO 训练中用 Langevin 校正桥接 ODE 推理与 SDE 探索的离散化差距。

50. **Discretization and Statistical Consistency of Functional Flow Matching.** arXiv 2026-08 [R] → T06. [paper](https://arxiv.org/abs/2608.04531)

    证明函数流匹配在有限系数/点值离散化下速度目标强L2收敛，并给出端到端Wasserstein误差界。

51. **Coupled Continuous-Discrete Generation for Scene Text Image Super-Resolution.** arXiv 2026-08 [R] → T22. [paper](https://arxiv.org/abs/2608.04525)

    条件流匹配恢复图像潜变量，吸收态离散扩散重建文本token，联合超分。

52. **On the Geometry of Music Bandwidth Extension in Latent Spaces of Audio Codecs.** arXiv 2026-08 [A:ISMIR 2026] → T23. [paper](https://arxiv.org/abs/2608.03721)

    发现音频编解码潜空间中简单质心平移向量即可媲美扩散/SB/流匹配的音乐带宽扩展效果。

53. **GROW: Group-Relative Advantage-Weighted On-Policy Reinforcement Learning of Autoregressive-Diffusion Text-to-Speech model.** arXiv 2026-08 [R] → T23. [paper](https://arxiv.org/abs/2608.03215)

    提出GROW，用组相对优势加权和Wasserstein-2速度惩罚直接强化学习流匹配TTS，降低WER并提升说话人相似度。

54. **EmbodiedVAE: Disentangled Video VAE for Efficient and Controllable Embodied Manipulation.** arXiv 2026-08 [A:ECCV 2026] → T19. [paper](https://arxiv.org/abs/2608.02990)

    EmbodiedVAE用最优传输一致性模块增强视频VAE中机器人运动潜变量的时序一致性，提升操控世界模型重建与控制精度。

55. **GraspMeanFlow: SE(3)-Equivariant MeanFlow for Few-Step 6-DoF Grasp Generation.** arXiv 2026-08 [R] → T09. [paper](https://arxiv.org/abs/2608.03295)

    提出SE(3)等变MeanFlow，用平均速度实现少步6-DoF抓取生成。

56. **Computational and Statistical Guarantees of the \textit{c}-Rectified flow.** arXiv 2026-08 [R] → T09. [paper](https://arxiv.org/abs/2608.02487)

    提出c-rectified flow，证明其迭代总能收敛到最优传输耦合，并给出计算与统计保证。

57. **Beckmann Transport Models: From Autonomous Flows to One-Step Maps.** arXiv 2026-08 [R] → T01. [paper](https://arxiv.org/abs/2608.01692)

    提出Beckmann传输模型，用自治流实现分布间精确映射与一步生成。

58. **ReFP-AD: Rectified Flow Preconditioning for Energy-Based Anomaly Detection.** arXiv 2026-08 [A:ECCV 2026] → T09. [paper](https://arxiv.org/abs/2608.01793)

    ReFP-AD用OT耦合的rectified flow将高维嵌入映射到良态潜空间，稳定EBM的MCMC采样并提升异常检测。

59. **QWRF-Net: A Quantum-Wavelet Framework with Rectified Flow for Short-Term Precipitation Nowcasting.** arXiv 2026-08 [R] → T09. [paper](https://arxiv.org/abs/2608.01626)

    量子小波分解结合整流流，提升短时降水预报的强对流核心保持能力。

60. **One-Sided Quantile Coupling for Flow Matching.** arXiv 2026-08 [R] → T08. [paper](https://arxiv.org/abs/2608.00978)

    提出QC-FM，用单侧分位数耦合替代minibatch OT，以O(n)代价构造源样本并保持直线流。

61. **Wasserstein gradient flows of Maximum Mean Discrepancy with energy kernels.** arXiv 2026-08 [R] → T05. [paper](https://arxiv.org/abs/2608.01182)

    研究能量核 MMD 的 Wasserstein 梯度流，证明全局适定性与粒子系统收敛。

62. **Latent Flow Matching for Arbitrage-Aware Implied Volatility Surface Generation.** arXiv 2026-08 [R] → T07. [paper](https://arxiv.org/abs/2608.00616)

    用潜空间流匹配生成无套利隐含波动率曲面。

<a id="reports"></a>
## H. 深读报告、译文与综合报告

- `reports/`：438 份逐篇深读（文件名 = arXiv id）；`data/meta/`：每篇的 TL;DR / 关键数字 / 关系卡。
- `papers/`：363 份原文；`papers_zh/`：126 份保版式中文译文 + `*.inspect.json` QA 报告。缺失的译文在持续补齐（`scripts/translate_batch.sh`）。
- `report/`：综合分析报告（`AWESOME_DIFFUSION_OT_REPORT_zh.md` / `_en.md` 及 PDF）。
- `slides/`：[HTML PPT](slides/awesome_diffusion_OT_deck.html)（浏览器打开，方向键翻页）、[可编辑 PPTX](slides/awesome_diffusion_OT_deck.pptx) 与 [Beamer PDF](slides/beamer/awesome_diffusion_OT_slides.pdf)；报告 PDF：[中文](report/pdf/awesome_diffusion_ot_report_zh.pdf) / [English](report/pdf/awesome_diffusion_ot_report_en.pdf)。
- 复现整条流水线：`scripts/build_corpus.py → resolve_arxiv.py → fetch_papers.py → extract_text.py → translate_batch.sh → src/generator.py`。

<a id="contributing"></a>
## I. 贡献与引用

欢迎 PR：在 `data/papers.jsonl` 或对应课题笔记加入条目（标题 / venue / 证据级 / 链接），运行 `python3 src/generator.py` 重新生成 README。PDF 与译文仅供个人学习研究，版权归原作者与出版方。

```bibtex
@misc{awesome_diffusion_ot_2026,
  title  = {Awesome Diffusion x Optimal Transport},
  author = {Li, Yufeng},
  year   = {2026},
  url    = {https://github.com/asimfish/awesome_diffusion_OT}
}
```
