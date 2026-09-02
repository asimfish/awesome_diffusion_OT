# T04 熵正则 OT 与 Sinkhorn 在生成建模中的角色

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」全景的计算与损失函数基座：熵正则把 OT 变成可微、可 GPU 化、统计上可行的对象，Sinkhorn divergence 及 entropic map 由此成为生成模型的训练损失与耦合/映射估计基元。上承 T01-T03 的 OT 理论，下接 T08（minibatch 耦合用于流匹配）与 T29（GPU 高性能求解器）——这两块本笔记不覆盖。

## 1. 核心问题与背景

精确 OT 距离（Wasserstein）在生成建模中有天然吸引力——能比较支撑不重叠的分布、提供有意义的梯度——但它有三重障碍：求解代价 O(n³ log n)、对样本数的维数灾难统计率 n^{-1/d}、以及作为损失时的不可微与梯度偏差。熵正则 OT（EOT）用 KL 罚项把线性规划变成强凸问题，Sinkhorn 矩阵缩放迭代即可求解，且解对输入光滑可微。由此衍生三条主线：(i) **Sinkhorn divergence 作为训练损失**——去掉熵偏差后插值于 OT 与 MMD 之间，样本复杂度摆脱维数灾难，成为 GAN/VAE/一步生成模型的分布匹配损失；(ii) **可微化机制**——对 Sinkhorn 迭代做 unrolling 自动微分或利用最优性条件做隐式微分，把 EOT 作为可训练层嵌入网络；(iii) **entropic map 估计与 debiasing**——熵耦合的 barycentric projection 给出 O(n²) 可算的 Monge 映射估计，而"减自传输项"的 debiasing 何时有益是有细微条件的。2024-2026 的新进展集中在正则化本身的再设计（ε 调度、expectile 对偶正则、二次/稀疏正则）与 Sinkhorn divergence 梯度流驱动的一步生成。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Sinkhorn Distances: Lightspeed Computation of Optimal Transport | 2013·NeurIPS | [P] | 把熵正则+Sinkhorn 矩阵缩放引入 ML，开创可扩展正则化 OT 这一领域 | [proceedings](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) |
| ⭐ Learning Generative Models with Sinkhorn Divergences | 2018·AISTATS | [P] | 首个用 Sinkhorn loss（unrolled 自动微分 + entropic 平滑）大规模训练生成模型的可行方案 | [PMLR](https://proceedings.mlr.press/v84/genevay18a.html) |
| Improving GANs Using Optimal Transport (OT-GAN) | 2018·ICLR | [P] | minibatch energy distance：primal OT 与对抗特征空间 energy distance 结合，梯度无偏、大 batch 下训练稳定 | [OpenReview](https://openreview.net/forum?id=rkQkBnJAb) |
| ⭐ Interpolating between Optimal Transport and MMD using Sinkhorn Divergences | 2019·AISTATS | [P] | 给出 debiased Sinkhorn divergence 的标准定义并证正定性、凸性、度量化极限（ε→0 得 OT，ε→∞ 得 MMD） | [PMLR](https://proceedings.mlr.press/v89/feydy19a.html) |
| Statistical bounds for entropic optimal transport: sample complexity and the central limit theorem | 2019·NeurIPS | [P] | EOT 经验估计的 O(1/√n) 样本复杂度与 CLT，是"熵正则修复维数灾难"的理论支柱 | [proceedings](https://papers.nips.cc/paper_files/paper/2019/hash/5acdc9ca5d99ae66afdfe1eea0e3b26b-Abstract.html) |
| Sinkhorn AutoEncoders | 2019·UAI (PMLR v115) | [P] | 在 WAE 框架内用 Sinkhorn 距离对齐 latent aggregated posterior 与先验，确定性编解码器即可生成 | [PMLR](https://proceedings.mlr.press/v115/patrini20a.html) |
| Faster Wasserstein Distance Estimation with the Sinkhorn Divergence | 2020·NeurIPS | [P] | 证明 debiased Sinkhorn divergence 估计平方 W₂ 距离在更大 ε 下仍达到近最优误差，计算-统计两头受益 | [proceedings](https://papers.nips.cc/paper_files/paper/2020/hash/17f98ddf040204eda0af36a108cbdea4-Abstract.html) |
| ⭐ Entropic estimation of optimal transport maps | 2021·arXiv:2109.12004 | [R] | entropic map = 熵耦合的 barycentric projection = 熵对偶势的梯度（熵版 Brenier 定理），O(n²) 可算且带有限样本率 | [arXiv](https://arxiv.org/abs/2109.12004) |
| Debiaser Beware: Pitfalls of Centering Regularized Transport Maps | 2022·ICML | [P] | 证明对 map 估计 debiasing 并非总有益：ε 大或样本少时反而更差，动摇"一律 debias"的信条 | [PMLR](https://proceedings.mlr.press/v162/pooladian22a.html) |
| A Unified Framework for Implicit Sinkhorn Differentiation | 2022·CVPR | [P] | 用隐式函数定理统一各种 Sinkhorn 层梯度（学习 cost 与 marginal 皆可），比 unrolling 更省内存更稳 | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Eisenberger_A_Unified_Framework_for_Implicit_Sinkhorn_Differentiation_CVPR_2022_paper.html) |
| Sinkhorn Flow as Mirror Flow: A Continuous-Time Framework for Generalizing the Sinkhorn Algorithm | 2024·AISTATS | [P] | Sinkhorn 的连续时间极限是测度空间 mirror flow，导出对噪声/偏差鲁棒的新变体，统一 Wasserstein mirror flow 等动力学 | [PMLR](https://proceedings.mlr.press/v238/reza-karimi24a.html) |
| ⭐ Progressive Entropic Optimal Transport Solvers (PROGOT) | 2024·NeurIPS | [P] | 把 ε 调度嵌入分步(时间离散化)求解：逐段解 EOT 并收缩正则，估计耦合与 map 更稳、可证一致性，是"ε 不再是单一超参"的代表 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/22b6bc18be9c2bfaa48adc1122f0a971-Abstract-Conference.html) |
| Expectile Regularization for Fast and Accurate Training of Neural Optimal Transport (ENOT) | 2024·NeurIPS spotlight | [P] | 用 expectile 回归正则约束对偶 Kantorovich 势，替代昂贵的 c-transform 内层优化，W₂ benchmark 上质量 3×、速度 10× | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d885c74aa0e00cc07a35346aa7988e34-Abstract-Conference.html) |
| Energy-guided Entropic Neural Optimal Transport | 2024·ICLR | [P] | 把 EBM 与 EOT 对偶结合：能量函数参数化熵对偶势，学到的随机 plan 直接用于 unpaired 图像域迁移 | [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/517eb19e99947f60afff0cf93e451825-Abstract-Conference.html) |
| One-Step Generative Modeling via Wasserstein Gradient Flows (W-Flow) | 2026·arXiv:2605.11755 | [R] | 用 debiased Sinkhorn divergence 作为 WGF 能量泛函，把多步分布演化压缩进一步生成器；ImageNet-256 一步 FID 1.29 | [arXiv](https://arxiv.org/abs/2605.11755) |

**表外补充**（证据分级见括号）：Differential Properties of Sinkhorn Approximation（2018·NeurIPS [P]，隐式微分先驱，[链接](https://papers.nips.cc/paper_files/paper/2018/hash/3fc2c60b5782f641f76bcefc39fb2392-Abstract.html)）；Sample Complexity of Sinkhorn Divergences（2019·AISTATS [P]，ε-维度-样本量折中，[链接](https://proceedings.mlr.press/v89/genevay19a.html)）；Linear Time Sinkhorn Divergences using Positive Features（2020·NeurIPS [P]，[链接](https://papers.nips.cc/paper_files/paper/2020/hash/9bde76f262285bb1eaeb7b40c758b53e-Abstract.html)）；Sinkhorn Natural Gradient for Generative Models（2020·NeurIPS [P]，Sinkhorn 信息几何下的自然梯度，[链接](https://papers.nips.cc/paper_files/paper/2020/hash/122e27d57ae8ecb37f3f1da67abb33cb-Abstract.html)）；DP-Sinkhorn: Don't Generate Me（2021·NeurIPS [P]，差分隐私生成用 Sinkhorn divergence 免对抗训练，[链接](https://papers.nips.cc/paper_files/paper/2021/hash/67ed94744426295f96268f4ac1881b46-Abstract.html)）；Understanding Entropic Regularization in GANs（2024·JMLR [P]，熵正则给 GAN 解带来 nuclear-norm 型 shrinkage 而 Sinkhorn divergence 恢复未正则解、两者皆 O(1/√n)，[链接](https://jmlr.org/papers/volume25/21-1295/21-1295.pdf)）；Annealed Sinkhorn（2024·arXiv:2408.11620 [R]，退火调度收敛充要条件 β_t→∞ 且增量→0，√t 调度是极限，提出 debiased annealed 变体，[链接](https://arxiv.org/abs/2408.11620)）；Sparsity of Quadratically Regularized Optimal Transport: Scalar Case（2024·arXiv:2410.03353 [R]，QOT 支撑以 ε^{1/3} 速率收缩到 Monge 图，[链接](https://arxiv.org/abs/2410.03353)）；Sparsity of QOT: Bounds on Concentration and Bias（2025·SIAM 期刊 [P]，一般维数 ε^{1/(d+2)} 界，[DOI](https://doi.org/10.1137/25m1723633)）；QOT: Localization Bounds and Affine Case Analysis（2026·arXiv:2605.24644 [R]，[链接](https://arxiv.org/abs/2605.24644)）；Sinkhorn-Drifting Generative Models（2026·arXiv:2603.12366 [R]，drifting 动力学 ≈ Sinkhorn divergence 梯度流的单侧归一化近似，解决零漂移可辨识性，[链接](https://arxiv.org/abs/2603.12366)）。

## 3. 方法演进脉络

**第一阶段（2013-2018）：从可算到可当损失。** Cuturi (2013) 用熵罚把 OT 变成 Sinkhorn 缩放可解的强凸问题；Genevay-Peyré-Cuturi (2018) 把 Sinkhorn 迭代 unroll 进自动微分图，第一次把"OT 型损失"塞进大规模生成模型训练，同期 OT-GAN (ICLR 2018) 走 primal minibatch 路线、用对抗学到的特征空间定义 minibatch energy distance；Sinkhorn AE (UAI 2019) 则把同一损失搬到 latent 空间对齐。这一阶段留下两个未决问题：熵偏差（OT_ε(μ,μ)≠0）与 minibatch 梯度的偏差。

**第二阶段（2018-2022）：偏差的形式化与两条可微化路线。** Feydy et al. (2019) 给出 debiased Sinkhorn divergence 的标准形式 S_ε = OT_ε(μ,ν) − ½OT_ε(μ,μ) − ½OT_ε(ν,ν) 并证其正定凸性；统计侧 Genevay et al. (2019)、Mena & Niles-Weed (2019) 建立 O(1/√n) 率（常数随 1/ε 指数增长），Chizat et al. (2020) 证明 debiasing 允许用大 ε 估计 W₂² 而不牺牲精度——"计算换统计"的关键论据。可微化上，unrolling（内存随迭代数线性增长）与隐式微分（Luise 2018 开端，Eisenberger 2022 统一到学习 cost/marginal 的一般 Sinkhorn 层，O(1) 内存）分化成标准工具箱。同期 Pooladian & Niles-Weed (2021) 用熵版 Brenier 定理把 entropic map（熵耦合的 barycentric projection）确立为 O(n²) 的 Monge 映射估计器；Debiaser Beware (ICML 2022) 随即泼冷水：对 map 估计，debias 只在 ε 小、样本多时占优——偏差修正本身成了需要调度的对象。

**第三阶段（2024-2026）：正则化的再设计与回到生成。** 算法理论上，Sinkhorn Flow as Mirror Flow (AISTATS 2024) 把 Sinkhorn 放进测度空间 mirror flow 框架，Annealed Sinkhorn (Chizat 2024) 揭示退火调度的"熵误差 + 松弛误差"分解及 √t 极限并给出 debiased 退火；训练策略上，PROGOT (NeurIPS 2024) 把 ε 调度与分步求解绑定成更稳的 map/耦合估计器，ENOT (NeurIPS 2024) 用 expectile 正则直接约束对偶势替代内层 conjugate 求解（正则对象从 primal 耦合转移到 dual 势）；正则项本身也多样化——二次正则 QOT 的稀疏支撑率（ε^{1/3}（d=1）/ ε^{1/(d+2)}（一般 d））在 2024-2026 被精确刻画，提供"介于全支撑 EOT 与确定性 Monge 图之间"的耦合谱系。生成侧完成闭环：energy-guided entropic NOT (ICLR 2024) 用 EBM 参数化熵对偶做跨域迁移；W-Flow 与 Sinkhorn-Drifting (2026) 把 debiased Sinkhorn divergence 的 Wasserstein 梯度流直接接到一步生成器训练，前者在 ImageNet-256 上把一步 FID 推到 1.29，后者证明 drifting 动力学是 Sinkhorn 梯度流的单侧近似并借 S_ε 正定性补上平衡点可辨识性——Sinkhorn divergence 从"GAN 时代的替代损失"变成"post-diffusion 一步生成的能量泛函"。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **间接但基础**。本课题不直接做轨迹对齐，但提供其两个核心基元：(a) 推理期用 Sinkhorn 求 noise↔sample 的熵耦合做重配对时，ε 的选择与 debiasing 直接决定对齐质量——PROGOT 的 ε 调度、Annealed Sinkhorn 的误差分解、Debiaser Beware 的"何时该 debias"给出了可移植的定量指导；(b) entropic map 的 barycentric projection 是闭式、O(n²)、可外推到新样本的映射估计，可在不动 backbone 的前提下对 latent/噪声做轻量重排。W-Flow/Sinkhorn-Drifting 属于"训练新的一步生成器"，不算免重训，但其 cross-minus-self 耦合结构提示了推理期粒子修正的可能形态。
- 方向二（OT 引导跨域生成）: **直接相关**。Energy-guided entropic neural OT 本身就是 unpaired 跨域生成方法（学 entropic plan 的随机映射）；ENOT 把 neural OT 训练成本降一个量级，使 OT 引导的域迁移更可用；entropic map 估计 + debiasing 理论决定了"plug-in OT 映射做域桥接"的统计可靠性；Sinkhorn divergence 则是跨域分布对齐中最常用的可微损失（含 DP-Sinkhorn 这类隐私约束场景）。

## 5. 开放问题与可发论文的切入点

1. **训练期偏差调度（bias scheduling）**：Debiaser Beware 只在 map 估计中刻画了 debias 的利弊边界；生成训练里默认用 debiased S_ε。可证/可测：训练早期（模型分布远离数据）用带偏 OT_ε 的 shrinkage 当隐式退火、后期切换 S_ε，给出切换点与 ε 的联合调度理论（借 Annealed Sinkhorn 的熵/松弛误差分解），在 toy Gaussian（有闭式）+ CIFAR 一步生成上验证 FID/模式覆盖。
2. **QOT 稀疏耦合用于生成配对**：QOT 支撑收缩理论（ε^{1/(d+2)}）刚建立但从未接入生成建模。具体做法：用 QOT 耦合替代 EOT 做蒸馏/配对数据构造，证 QOT barycentric projection 的 map 估计率（对标 Pooladian-Niles-Weed），实验比较 EOT（blur）vs QOT(稀疏)在配对噪声与样本质量上的 trade-off。注意与 T08 的 minibatch-FM 边界：切入点放在损失/配对估计理论而非 FM pipeline。
3. **expectile 正则与 softmin c-transform 的统一**：ENOT 的 expectile 正则和 Sinkhorn 的 log-sum-exp 平滑都是 c-transform 的松弛。证明二者是同一族"广义平滑 c-transform"的两个端点（τ→1 对应硬 max，ε→0 对应精确 conjugate），给出中间态的 bias-variance 刻画，并测试混合正则在 W₂ benchmark 上是否超过 ENOT。
4. **Sinkhorn divergence WGF 的非高斯收敛理论**：W-Flow 的全局收敛论据依赖高斯假设。可证：log-concave 或有界密度比条件下 S_ε-WGF 的收敛速率、debiased vs biased 能量的临界 ε；实验上把 W-Flow 的 velocity guidance 换成 entropic map 方向场，检验一步生成质量与训练稳定性。
5. **隐式微分 + ε 调度的联合层设计**：Eisenberger 框架假设固定 ε 的最优性条件；PROGOT 的分步 ε 调度目前只做前向。推导"隐式微分穿过 ε 调度链"的梯度（每段一个隐式函数定理 + 链式耦合），做成可端到端训练的 progressive-Sinkhorn 层，用在需要学 cost 的跨域对齐任务上。

## 6. 代码与资源

- **OTT-JAX**（https://github.com/ott-jax/ott，文档 https://ott-jax.readthedocs.io）：官方实现含 Sinkhorn 隐式微分、Sinkhorn divergence、PROGOT（`progot`）、ENOT（`ExpectileNeuralDual`，教程 https://ott-jax.readthedocs.io/tutorials/neural/300_ENOT.html）
- **GeomLoss**（https://www.kernel-operations.io/geomloss/）：Feydy 的 debiased Sinkhorn divergence GPU 实现（PyTorch），生成模型训练损失的事实标准
- **POT**（https://pythonot.github.io/）：通用 OT 库，含 Sinkhorn 各变体与 `stochastic`/`smooth`（二次正则）求解器
- **implicit-sinkhorn**（https://github.com/marvin-eisenberger/implicit-sinkhorn）：CVPR 2022 隐式微分参考实现
- **ENOT 项目页**（https://skylooop.github.io/enot/）；**W-Flow 项目页**（https://hanjq17.github.io/W-Flow/）；**Sinkhorn-Drifting 项目页**（https://mint-vu.github.io/SinkhornDrifting/）；**DP-Sinkhorn 项目页**（https://nv-tlabs.github.io/DP-Sinkhorn/）
- **Wasserstein-2 benchmark**（https://github.com/iamalexkorotin/Wasserstein2Benchmark）：ENOT 等 neural OT/EOT 方法的标准评测
- 教材/讲义 [B]：Peyré & Cuturi《Computational Optimal Transport》Ch.4（https://optimaltransport.github.io/book/）；Peyré《Optimal Transport for Machine Learners》（https://arxiv.org/abs/2505.06589）；Chewi, Niles-Weed & Rigollet《Statistical Optimal Transport》（https://arxiv.org/abs/2407.18163，熵正则统计理论系统讲义）；Léonard《A Survey of the Schrödinger Problem...》（https://doi.org/10.3934/dcds.2014.34.1533，EOT 与 Schrödinger 问题/大偏差极限的经典综述）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2018_Genevay_learning_generative_models_sinkhorn.pdf | Learning Generative Models with Sinkhorn Divergences | 成功 |
| 2019_Feydy_interpolating_ot_mmd_sinkhorn.pdf | Interpolating between Optimal Transport and MMD using Sinkhorn Divergences | 成功 |
| 2021_Pooladian_entropic_map_estimation.pdf | Entropic estimation of optimal transport maps | 成功 |
| 2022_Pooladian_debiaser_beware.pdf | Debiaser Beware: Pitfalls of Centering Regularized Transport Maps | 成功 |
| 2024_Kassraie_progressive_entropic_ot_solvers.pdf | Progressive Entropic Optimal Transport Solvers | 成功 |
| 2024_Buzun_enot_expectile_regularization.pdf | Expectile Regularization for Fast and Accurate Training of Neural Optimal Transport | 成功 |
| 2024_Chizat_annealed_sinkhorn.pdf | Annealed Sinkhorn for Optimal Transport | 成功 |
| 2026_Han_wflow_one_step_sinkhorn_wgf.pdf | One-Step Generative Modeling via Wasserstein Gradient Flows | 成功 |
