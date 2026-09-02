# T05 Wasserstein 梯度流与 JKO 格式生成模型

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」的动力学底座——把扩散/生成过程看成概率测度空间 \((\mathcal{P}_2, W_2)\) 上某个能量泛函的最速下降。JKO 格式是这条最速下降曲线的时间离散化，其神经网络化直接产出一族"逐块可训、几何可解释"的生成模型；Wasserstein proximal 算子则给出 score-based 模型的变分刻画。采样收敛速率的理论界归 T06，barycenter 计算归 T27，本笔记不覆盖。

## 1. 核心问题与背景

Fokker-Planck 方程（即扩散模型 forward SDE 的密度演化）可以等价地写成 KL 泛函在 Wasserstein-2 度量下的梯度流（Jordan–Kinderlehrer–Otto, 1998）；Otto (2001) 进一步把 \(\mathcal{P}_2(\mathbb{R}^d)\) 装备成形式黎曼流形（"Otto calculus"），使"在分布空间上做优化"具有了梯度、Hessian、测地凸性等完整微分几何词汇。JKO 格式 \(\rho_{k+1}=\arg\min_\rho \frac{1}{2\tau}W_2^2(\rho,\rho_k)+\mathcal{F}(\rho)\) 是这条流的隐式 Euler（proximal point）离散。核心问题有三：(i) 每个 JKO 步本身是一个 OT 问题，如何用神经网络高效求解并推到高维（ICNN 参数化、半对偶、逐块 flow）；(ii) 如何反过来从数据学习驱动流的能量泛函（JKOnet 系）；(iii) 非测地凸泛函下测度空间优化的鞍点/二阶理论。这条线为扩散模型提供了不同于 score matching 的第二套构造原理，也是粒子法（SVGD/ParVI）与生成模型之间的桥梁。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| The Variational Formulation of the Fokker–Planck Equation (Jordan, Kinderlehrer, Otto) | 1998 · SIAM J. Math. Anal. | [P] | 奠基：FPE = KL 的 \(W_2\) 梯度流，提出 JKO（minimizing movement）格式 | [DOI](https://doi.org/10.1137/S0036141096303359) |
| ⭐ Large-Scale Wasserstein Gradient Flows (Mokrov, Korotin, Li, Genevay, Solomon, Burnaev) | 2021 · NeurIPS | [P] | 神经化 JKO 开山：Brenier 定理 + ICNN 参数化每个 JKO 步的凸势，SGD 免网格/免粒子求解 | [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/810dfbbebb17302018ae903e9cb7a483-Abstract.html) |
| Optimizing Functionals on the Space of Probabilities with ICNNs (Alvarez-Melis, Schiff, Mroueh) | 2022 · TMLR | [P] | JKO-ICNN 框架：ICNN 逼近凸函数空间做 JKO，含收敛保证的泛函设计与分子受控生成 | [OpenReview](https://openreview.net/forum?id=dpOYN7o8Jm) |
| Variational Wasserstein Gradient Flow (Fan, Zhang, Taghvaei, Chen) | 2022 · ICML | [P] | 用 f-divergence 的变分（对偶）形式替代显式密度项，primal-dual 求 JKO 步，可扩展到高维 | [PMLR](https://proceedings.mlr.press/v162/fan22d.html) |
| Proximal Optimal Transport Modeling of Population Dynamics (Bunne, Meng-Papaxanthos, Krause, Cuturi) | 2022 · AISTATS | [P] | JKOnet：反问题视角——从时序快照端到端学习驱动种群演化的能量泛函（单细胞应用） | [PMLR](https://proceedings.mlr.press/v151/bunne22a.html) |
| Variational Inference via Wasserstein Gradient Flows (Lambert, Chewi, Bach, Bonnabel, Rigollet) | 2022 · NeurIPS | [P] | 把 VI 写成 Bures–Wasserstein 子流形上的 WGF/JKO（Gaussian 与混合 Gaussian），log-concave 下有保证 | [OpenReview](https://openreview.net/forum?id=K2PTuvVTF1L) |
| ⭐ Normalizing Flow Neural Networks by JKO Scheme (Xu, Cheng, Xie) | 2023 · NeurIPS (Spotlight) | [P] | JKO-iFlow：每个残差块=一个 JKO 步，逐块训练 CNF，免 score matching / 端到端反传，省显存 | [OpenReview](https://openreview.net/forum?id=ZQMlfNijY5) |
| ⭐ Scalable Wasserstein Gradient Flow for Generative Modeling through Unbalanced Optimal Transport (Choi, Choi, Kang) | 2024 · ICML | [P] | S-JKO：JKO 步 ↔ UOT 等价 → 半对偶形式把训练复杂度 \(O(K^2)\to O(K)\)，CIFAR-10 FID 2.62，WGF 生成模型首次逼近 SOTA | [PMLR](https://proceedings.mlr.press/v235/choi24a.html) |
| Learning Diffusion at Lightspeed (Terpin, Lanzetti, Gadea, Dörfler) | 2024 · NeurIPS (Oral) | [P] | JKOnet*：用 JKO 步的一阶最优性条件替代双层优化，二次损失学 potential/interaction/internal 三种能量，线性参数化有闭式解 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0ce1eb87dbb03fdfa872a93d15cfe333-Abstract.html) |
| Mirror and Preconditioned Gradient Descent in Wasserstein Space (Bonet, Uscidda, David, Aubin-Frankowski, Korba) | 2024 · NeurIPS (Spotlight) | [P] | 把镜像下降/预条件梯度下降提升到 \(\mathcal{P}_2\)：相对光滑/凸下的收敛保证，病态目标与单细胞对齐实验 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2cf153951b5e9b39564fc4a0ef6adc1a-Abstract-Conference.html) |
| ⭐ Wasserstein Proximal Operators Describe Score-Based Generative Models and Resolve Memorization (Zhang, Liu, Li, Katsoulakis, Osher) | 2026 · SIAM J. Math. Data Sci.（arXiv 2024) | [P] | 证明 SGM 本质上在实现 Wasserstein proximal 算子（经 MFG 的 FP+HJB 对偶），据此构造核模型解释并消除记忆化 | [arXiv](https://arxiv.org/abs/2402.06162) · [DOI](https://doi.org/10.1137/24M1644584) |
| Importance Corrected Neural JKO Sampling (Hertrich, Gruhlke) | 2025 · ICML | [P] | CNF 实现的 JKO 局部步 + 重要性拒绝重采样非局部步，克服 WGF 采样的多峰质量错配，可产 iid 样本并评估密度 | [PMLR](https://proceedings.mlr.press/v267/hertrich25a.html) |
| Flowing Datasets with Wasserstein over Wasserstein Gradient Flows (Bonet 等) | 2025 · ICML | [P] | 把梯度流升到"测度的测度"（WoW）空间，对整个带标签数据集做流动（数据集级迁移/蒸馏） | [PMLR](https://proceedings.mlr.press/v267/bonet25a.html) |
| ⭐ Hessian-Guided Perturbed Wasserstein Gradient Flows for Escaping Saddle Points (Yamamoto, Kim, Suzuki) | 2025 · NeurIPS | [P] | PWGF：沿 Wasserstein Hessian 最小特征方向注入 GP 扰动逃离鞍点，测度空间非凸优化首个二阶最优性+多项式时间保证 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3dca8a6a5422c2d5d22c6e9a26469d7e-Abstract-Conference.html) |
| ⭐ One-Step Generative Modeling via Wasserstein Gradient Flows (Han, Li, Guo, Xu, Ermon, Candès) | 2026 · arXiv 2605.11755 | [R] | W-Flow：用 Sinkhorn 散度的 WGF 定义训练动力学，再把整条演化蒸馏进一步生成器；一步 ImageNet-256 FID 1.29（约百倍加速） | [arXiv](https://arxiv.org/abs/2605.11755) |

正文补充（未入表）：Otto (2001) porous medium 几何化 [B]；Salim, Korba & Luise, The Wasserstein Proximal Gradient Algorithm, NeurIPS 2020 [P]（forward-backward 拆分）；Liu, SVGD as Gradient Flow, NeurIPS 2017 [P]（[arXiv](https://arxiv.org/abs/1704.07520)）；Chewi 等, SVGD as a Kernelized Wasserstein Gradient Flow of the Chi-Squared Divergence, NeurIPS 2020 [P]（[官方页](https://proceedings.neurips.cc/paper/2020/hash/16f8e136ee5693823268874e58795216-Abstract.html)）；Cheng 等, Particle-based VI with Generalized WGF, NeurIPS 2023 [P]（[官方页](https://neurips.cc/virtual/2023/poster/70461)）；Li & Osher 等, A Kernel Formula for Regularized Wasserstein Proximal Operators, 2023 [R]（[arXiv](https://arxiv.org/abs/2301.10301)）；BRWP 收敛分析（JMLR 2026，[官方 PDF](https://www.jmlr.org/papers/volume27/24-1560/24-1560.pdf)）属采样收敛理论 → 详见 T06；Semi-Implicit Functional Gradient Flow, 2024 [R]（[arXiv](https://arxiv.org/abs/2410.17935)）；A Unifying View of Variational Generative Wasserstein Flows, 2026 [R]（[arXiv](https://arxiv.org/abs/2605.31369)）；GenWGP 大偏差作用量路径搜索, 2026 [R]（[arXiv](https://arxiv.org/abs/2604.11519)）；Accelerated Regularized Wasserstein Proximal Sampling, 2026 [R]（[arXiv](https://arxiv.org/abs/2601.09848)）；教材：Ambrosio–Gigli–Savaré《Gradient Flows》[B]、Santambrogio 综述 [B]（[arXiv](https://arxiv.org/abs/1609.03890)）、Figalli 讲义 [B]（[PDF](https://people.math.ethz.ch/~afigalli/lecture-notes-pdf/An-introduction-to-optimal-transport-and-Wasserstein-gradient-flows.pdf)）、Peyré OT4ML [B]（[arXiv](https://arxiv.org/abs/2505.06589)）。

## 3. 方法演进脉络

**理论奠基（1998–2017）**：JKO (1998) 证明 FPE 是 KL 的 \(W_2\) 最速下降并给出 proximal 离散；Otto (2001) 提供黎曼几何语言；AGS 专著与 Santambrogio 综述把度量空间梯度流理论系统化。这一阶段 JKO 只是分析工具，数值上受限于低维网格。

**神经化第一波（2021–2022）**：Mokrov 等与 Alvarez-Melis 等（JKO-ICNN）几乎同时用 ICNN 参数化 JKO 步的 Brenier 凸势，首次免网格求解高维 WGF；Fan 等用 f-divergence 对偶变分式绕开显式密度估计，将 JKO 步变成 primal-dual min-max；Bunne 等（JKOnet）打开反问题方向——从快照学习能量。共同瓶颈：ICNN 表达力不足、逐步嵌套导致 \(O(K^2)\) 复杂度、难以扩展到图像规模。

**走向生成模型（2023–2024）**：JKO-iFlow 把 CNF 的每个残差块解释成一个 JKO 步做逐块训练，免去 SDE 轨迹采样与端到端反传；S-JKO 是规模化转折点——利用 JKO 步与 UOT 的等价性推出半对偶目标，训练复杂度降到 \(O(K)\)，在 CIFAR-10/CelebA-HQ 上首次让 WGF 生成模型与扩散模型 FID 可比；JKOnet* 用一阶最优性条件把"学能量"化为单层二次损失（NeurIPS 2024 Oral）。

**几何与优化理论深化（2024–2025）**：Bonet 等把镜像/预条件下降提升到 \(\mathcal{P}_2\)（改变优化几何）；PWGF（NeurIPS 2025）给出测度空间逃离鞍点的首个二阶最优性框架——沿 Wasserstein Hessian 最小特征方向做 GP 扰动；WoW flows 把流动升到"数据集的空间"。

**Wasserstein proximal 支线**：Salim 等 (2020) 的 Wasserstein proximal gradient → Li–Osher 正则化 WPO 核公式 (2023, 经 Hopf–Cole 变换有闭式) → BRWP 无噪声采样器（收敛理论见 T06）→ WPO-SGM (SIMODS 2026)：score-based 模型 = WPO 的 MFG（FP+HJB）刻画，核公式直接解释 memorization 并给出解决方案。**粒子/SVGD 支线**：Liu (2017) 证 SVGD 是 KL 的 kernelized WGF；Chewi 等 (2020) 给出 χ² 散度视角与 LAWGD；GWG (NeurIPS 2023) 用凸正则化广义化 Wasserstein 度量；SIFG (2024) 引入半隐式分布族+去噪得分匹配。**2026 前沿**：W-Flow 把 Sinkhorn-WGF 动力学蒸馏成一步生成器（ImageNet-256 FID 1.29）；GWF 统一 f-divergence/IPM/MMD 下各类参数化 JKO；GenWGP 用大偏差作用量做全局路径优化替代逐步 JKO。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **强相关**。(i) WPO-SGM 证明预训练 SGM 的 score 隐式实现 Wasserstein proximal 算子，其核公式给出一个**免训练的 score 替代/修正通道**——可在推理期用 RWPO 核平滑 score 场实现轨迹重塑或去记忆化；(ii) Bonet 的镜像/预条件框架说明"换一个 Bregman 势或预条件器"即可改变梯度流轨迹的几何而不动能量泛函本身，为 training-free 轨迹再参数化提供理论词汇；(iii) JKO-iFlow 的逐块结构意味着可以只重排/重标定部分块（时间重参数化）而不整体重训。
- 方向二（OT 引导跨域生成）: **强相关**。(i) S-JKO 的 JKO↔UOT 等价把每个演化步骤本身变成一个（不平衡）跨域传输问题，其半对偶势即"引导场"；(ii) JKOnet/JKOnet* 提供从跨域快照数据**反学习引导能量**的机制——学到的 potential/interaction 能量可直接充当跨域生成的 guidance；(iii) WoW flows 把跨域对齐提升到数据集级（测度的测度），适合多域/多任务迁移；(iv) W-Flow 表明 OT 型能量（Sinkhorn 散度）驱动的 WGF 可蒸馏成一步跨域映射。

## 5. 开放问题与可发论文的切入点

1. **半对偶 JKO 势作为免训练 guidance**：S-JKO 的 UOT 半对偶势只在训练期使用。切入点：冻结预训练扩散模型，推理期针对目标域小样本在线解一个轻量半对偶问题，把所得势的梯度作为 plug-in 漂移项注入 PF-ODE，与 SDEdit/DDIB 在跨域保真-对齐权衡上对比（CIFAR→CelebA、画风迁移）。理论上可证：该修正等价于对原 KL 能量加一个 UOT 罚项的 JKO 步。
2. **可扩展的 Wasserstein 二阶方法**：PWGF 的 Hessian 引导扰动目前只有 GP 采样的概念性实现。切入点：用随机 Lanczos/Hutchinson 在粒子系综上近似 Wasserstein Hessian 最小特征方向（Hessian-vector 积只需 score 的 Jacobian-vector 积），做成可在 mean-field 两层网络训练与多峰采样上运行的算法；实验验证"二阶信息 vs 各向同性噪声"的逃逸速度差距，并给出步长-谱关系的自适应 JKO 步长规则。
3. **RWPO 核公式作为记忆化诊断/修复工具**：WPO-SGM 只在核模型内讨论 memorization。切入点：对任意预训练扩散模型，用 Li–Osher 核公式从训练集构造"完全记忆化 score"，定义逐时刻的 score 偏差谱作为记忆化度量，构建 benchmark；再用核平滑局部替换高偏差区域的 score 实现免重训去记忆化，与 dataset deduplication 基线对比。
4. **WGF 蒸馏的统一理论（连接一步生成）**：W-Flow 蒸馏 Sinkhorn-WGF、consistency model 蒸馏 PF-ODE，两者形式高度平行但无统一分析。切入点：证明"一步生成器蒸馏误差 ≤ 能量泛函沿广义测地线的凸性/光滑常数 × 离散步长"型的界，统一 JKO 步数-蒸馏质量 trade-off；把 W-Flow 的能量换成 UOT/interaction 能量以处理跨域与类不平衡。
5. **从 JKOnet\* 到"能量可迁移性"**：JKOnet* 能从单域快照学能量，但学到的能量能否跨域/跨分辨率泛化未知。切入点：在两个域各学能量后研究能量空间的插值与组合（potential 相加、interaction 混合）产生的新流是否对应有意义的语义混合，为"能量算术"式的可控生成提供第一批证据（单细胞扰动数据 + 图像 toy 集）。

## 6. 代码与资源

- Mokrov 等 ICNN-JKO：https://github.com/PetrMokrov/Large-Scale-Wasserstein-Gradient-Flows
- Fan 等 Variational WGF：https://github.com/sbyebss/variational_wgf
- JKOnet（AISTATS 2022）：https://github.com/bunnech/jkonet
- JKOnet*（NeurIPS 2024 Oral，JAX）：https://github.com/antonioterpin/jkonet-star （项目页含文档）
- S-JKO（ICML 2024）：https://github.com/Jae-Moo/Scalable-JKO
- Neural JKO IC（ICML 2025）：https://github.com/johertrich/neural_JKO_ic
- 通用 OT 库（JKO 步内的 OT 求解）：POT（https://pythonot.github.io/）、OTT-JAX（https://ott-jax.readthedocs.io/）
- 常用 benchmark：合成多峰分布（funnel、8-modes、高维 Gaussian mixture，见 Neural JKO IC 附录）；图像 FID（CIFAR-10、CelebA-HQ-256，S-JKO/W-Flow）；单细胞轨迹（JKOnet/JKOnet* 用的 scRNA 快照数据）
- 教材/讲义：Ambrosio–Gigli–Savaré《Gradient Flows in Metric Spaces and in the Space of Probability Measures》；Santambrogio overview（arXiv:1609.03890）；Figalli 讲义（ETH）；Peyré《OT for Machine Learners》（arXiv:2505.06589）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2021_Mokrov_Large_Scale_Wasserstein_Gradient_Flows.pdf | Large-Scale Wasserstein Gradient Flows | 成功 |
| 2023_Xu_JKO_iFlow.pdf | Normalizing Flow Neural Networks by JKO Scheme | 成功 |
| 2024_Choi_S_JKO_Unbalanced_OT.pdf | Scalable Wasserstein Gradient Flow for Generative Modeling through Unbalanced Optimal Transport | 成功 |
| 2024_Zhang_Wasserstein_Proximal_SGM.pdf | Wasserstein Proximal Operators Describe Score-Based Generative Models and Resolve Memorization | 成功 |
| 2024_Terpin_Learning_Diffusion_Lightspeed.pdf | Learning Diffusion at Lightspeed (JKOnet*) | 成功 |
| 2025_Hertrich_Importance_Corrected_Neural_JKO.pdf | Importance Corrected Neural JKO Sampling | 成功 |
| 2025_Yamamoto_Hessian_Perturbed_WGF.pdf | Hessian-Guided Perturbed Wasserstein Gradient Flows for Escaping Saddle Points | 成功 |
| 2026_Han_W_Flow_One_Step_Generative.pdf | One-Step Generative Modeling via Wasserstein Gradient Flows (W-Flow) | 成功 |
