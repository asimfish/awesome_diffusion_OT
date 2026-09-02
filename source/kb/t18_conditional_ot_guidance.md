# T18 条件生成与 guidance 的 OT 形式化

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题处在「扩散×OT」全景的"可控性"轴上：一端是把条件 y 显式放进传输耦合/传输距离（conditional OT、条件 Wasserstein 几何），另一端是把 classifier(-free) guidance、energy guidance、reward alignment 统一成随机最优控制（SOC）/h-transform，并用 W2 度量 guidance 造成的分布偏移。它向下衔接 T07/T08（FM 与 minibatch OT 耦合），向上支撑对齐与可控生成应用。
> 边界: 跨域语义对应见 T16；纯采样加速见 T11。

## 1. 核心问题与背景

条件生成的本质是从先验到条件分布 p(x|y) 的传输，但主流做法（CFG）只是推理时对 score 做线性外推，2024 年起一批理论工作证明它并不采样自其动机所声称的 tilted 分布 p(x|y)^w p(x)^{1-w}，且强 guidance 会推高置信度、压缩多样性、造成分布偏移。OT 视角在三个层面给出形式化：(i) 耦合层面——把条件写进耦合或代价矩阵（COT-FM 的三角耦合、C2OT 的条件加权代价），使训练/测试先验一致并保持直线路径；(ii) 距离层面——条件 Wasserstein 距离（restricted couplings）保证 joint 距离控制 posterior 距离，为 Bayesian 逆问题与条件 FM 提供正确的几何；(iii) 控制层面——guidance/reward fine-tune 统一为带 terminal cost 的 SOC 或 Doob h-transform，KL/W2 正则决定极限分布（KL→tilted 分布，W2→防坍缩的近端约束）。该方向直接决定文生图对齐、逆问题后验采样与 RLHF 式对齐的保真-多样性权衡，是 2024-2026 扩散×OT 最活跃的交叉带之一。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Dynamic Conditional Optimal Transport through Simulation-Free Flows (COT-FM) | 2024·NeurIPS | [P] | 证明条件 OT 的动态形式（条件版 Benamou-Brenier），用三角 COT 耦合做 simulation-free 条件生成，适用无穷维 Bayesian 逆问题 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aa1d8cf866c4a684ef2e066dd200f8e4-Abstract-Conference.html) |
| ⭐ Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching | 2025·JMLR 26(141) | [P] | 用受限耦合定义条件 Wasserstein 距离 = posterior W2 的期望；刻画测地线/速度场（Y 分量为零）并给出 Bayesian OT-FM | [JMLR](https://www.jmlr.org/papers/v26/24-0586.html) |
| ⭐ The Curse of Conditions: Analyzing and Improving OT for Conditional Flow-Based Generation (C2OT) | 2025·ICCV | [P] | 揭示无条件 minibatch OT 在条件 FM 中造成"条件偏斜先验"的 train-test gap，在 OT 代价矩阵加条件加权项修复 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_The_Curse_of_Conditions_Analyzing_and_Improving_Optimal_Transport_for_ICCV_2025_paper.html) |
| Conditional Optimal Transport on Function Spaces | 2025·SIAM/ASA JUQ 13(1) | [P] | 无穷维函数空间上 block-triangular Monge 映射与 Kantorovich 松弛的系统理论，给 amortized Bayesian 推断正则性估计 | [SIAM](https://epubs.siam.org/doi/10.1137/23M1618922) |
| Efficient Neural Network Approaches for Conditional Optimal Transport (PCP-Map / COT-Flow) | 2023·arXiv（SISC 刊出信息未直接核验） | [R] | 静态（部分输入凸网络梯度）与动态（正则化 neural ODE）两种神经条件 OT 求解器，likelihood-free 推断基线 | [arXiv](https://arxiv.org/abs/2310.16975) |
| Hyperparameter Trajectory Modeling via Conditional Lagrangian Optimal Transport | 2026·ICLR (Oral) | [A] | 把条件 Lagrangian OT 用于建模训练轨迹，是条件 OT 走向新应用域的前沿样本 | [OpenReview](https://openreview.net/forum?id=P5B97gZwRb) |
| ⭐ Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control | 2025·ICLR | [P] | 把 reward fine-tune 严格写成 SOC；证明必须用 memoryless 噪声调度才收敛到 KL-tilted 分布；SOC 化为回归（adjoint matching） | [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/852f50969a9e523ec41d26f2f68bd456-Abstract-Conference.html) |
| Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization (ORW-CFM-W2) | 2025·ICLR | [A] | RLHF 式在线 fine-tune FM，用可计算的 W2 上界正则防 policy collapse，给出 reward-多样性可控权衡 | [OpenReview](https://openreview.net/forum?id=2IoFFexvuw) |
| What does guidance do? A fine-grained analysis in a simple setting | 2024·NeurIPS | [P] | 严格证明 guidance 不采样 tilted 分布；w 增大时样本堆向条件支撑集边界，有 score 误差时甚至逸出支撑集 | [OpenReview](https://openreview.net/forum?id=AdS3H8SaPi) |
| Theoretical Insights for Diffusion Guidance: A Case Study for Gaussian Mixture Models | 2024·ICML (PMLR 235) | [P] | GMM 下证明 guidance 提升分类置信度同时降低微分熵（多样性），覆盖 DDPM/DDIM | [PMLR](https://proceedings.mlr.press/v235/wu24b.html) |
| Classifier-Free Guidance is a Predictor-Corrector | 2024·arXiv（NeurIPS 2024 M3L workshop） | [R] | 证明 CFG≠gamma-powered 分布采样；SDE 极限下等价于"条件 DDIM 预测 + gamma-powered Langevin 校正" | [arXiv](https://arxiv.org/abs/2408.09000) |
| ⭐ Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts | 2025·ICML (PMLR 267, spotlight) | [P] | 用 Feynman-Kac 公式+SMC 加权模拟，从退火/几何平均/乘积分布精确采样，修正 CFG 的中间分布失配 | [PMLR](https://proceedings.mlr.press/v267/skreta25a.html) |
| Energy-guided Entropic Neural Optimal Transport | 2024·ICLR | [P] | 打通 EBM 与熵正则 OT：用能量模型参数化熵 OT 解并给泛化界，"energy guidance"落到 OT 求解器内部 | [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/517eb19e99947f60afff0cf93e451825-Abstract-Conference.html) |
| Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline RL (CEP/QGPO) | 2023·ICML (PMLR 202) | [P] | 给出中间时刻能量 guidance 的精确形式与对比学习目标，是 energy guidance 精确化的奠基工作 | [PMLR](https://proceedings.mlr.press/v202/lu23d.html) |
| Optimal Transport-Guided Conditional Score-Based Diffusion Model (OTCS) | 2023·NeurIPS | [P] | 用 L2 正则 OT 耦合为无配对/半配对数据构造条件 score 模型，证明其以理论界实现 OT 数据传输 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/72c12e48c6135762f56bf188cd2479d2-Abstract-Conference.html) |

补充条目（边界/支撑，不占主表）：

- DEFT: Efficient Fine-tuning of Diffusion Models by Learning the Generalised h-transform，2024·NeurIPS [P]——Doob h-transform 统一各类条件采样/guidance，SOC 目标可从单个观测学 h。[OpenReview](https://openreview.net/forum?id=AKBTFQhCjm)
- RB-Modulation: Training-Free Stylization using Reference-Based Modulation，2025·ICLR [P]——把风格/内容约束编码进随机最优控制器的 **terminal cost**，训练无关的可控生成，是"约束写成传输/控制代价"的代表。[proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8f3705d6354fcecf48515cc43aa16023-Abstract-Conference.html)
- Applying Guidance in a Limited Interval…，2024·NeurIPS [P]——经验证明 guidance 仅在中噪声区间有益（ImageNet-512 FID 1.81→1.40），为强度调度提供事实依据。[NeurIPS](https://neurips.cc/virtual/2024/poster/93711)
- A Stochastic Analysis Approach to Conditional Diffusion Guidance (Tang & Xu)，2024·working paper [R]——鞅/Malliavin 视角下给出 guided 分布与目标条件分布之间 **TV 与 W2 距离界**，是"guidance 强度→分布偏移"最直接的定量理论。[PDF](https://www.columbia.edu/~wt2319/CDG.pdf)
- Fine-tuning of diffusion models via stochastic control: entropy regularization and beyond (Tang & Zhou)，2024·arXiv [R]——熵正则 fine-tune 的严格控制论处理，推广到一般 f-divergence 正则。[arXiv](https://arxiv.org/abs/2403.06279)
- Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized Control (Uehara et al.)，2024·arXiv [R]——提出熵正则 SOC fine-tune 框架（ELEGANT），是 Adjoint Matching/Tang 系列的直接前驱。[arXiv](https://arxiv.org/abs/2402.15194)
- Understanding RL-Based Fine-Tuning of Diffusion Models: A Tutorial and Review (Uehara et al.)，2024·arXiv [B]——reward alignment 全景综述，含 SOC/guidance/DPO 谱系。[arXiv](https://arxiv.org/abs/2407.13734)
- Understanding Classifier-Free Guidance: High-Dimensional Theory and Non-Linear Generalizations，2025·arXiv [R]——高维极限下解释"CFG 理论上错但实践中好"，并给非线性推广。[arXiv](https://arxiv.org/abs/2502.07849)
- Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models，2025·arXiv [R]——GMM 多模态条件下 CFG 三阶段动力学（方向偏移/模式分离/收缩），解释多样性损失来源。[arXiv](https://arxiv.org/abs/2509.22007)
- Conditional Diffusion Models with Classifier-Free Gibbs-like Guidance (CFGIG)，2025·arXiv [R]——指出 CFG 若要瞄准 tilted 分布须加 Rényi 散度梯度（排斥项），提出 Gibbs 式迭代修正。[arXiv](https://arxiv.org/abs/2505.21101)
- DPAC: Distribution-Preserving Adversarial Control for Diffusion Sampling，2025·arXiv [R]——SOC 视角：path-KL=控制能量，同时上界 W2 与 FID；把 guidance 梯度投影到等密度面切空间以保分布。[arXiv](https://arxiv.org/abs/2512.01153)
- COT-FM: Cluster-wise Optimal Transport Flow Matching，2026·arXiv [R]——按簇（类标签/文本）分配源分布再做簇内 OT 耦合；注意与 Kerrigan COT-FM 重名，且其主要动机是加速（T11 边界）。[arXiv](https://arxiv.org/abs/2603.13395)

## 3. 方法演进脉络

**第一阶段（2022-2023，条件进入 OT）**：Bunne et al. 的 CondOT（NeurIPS 2022，[arXiv](https://arxiv.org/abs/2206.14262)）用部分输入凸网络学以协变量为条件的 Monge 映射族，确立"amortized 条件传输"范式；OTCS（NeurIPS 2023）反向操作——先用 L2 正则 OT 在无配对数据间估计耦合，再拿耦合当条件 score 模型的监督，OT 第一次成为条件扩散的"条件构造器"。同期 CEP（ICML 2023）解决 energy guidance 的根本困难：中间时刻的能量由数据分布与能量函数联合决定、不可直接求，对比学习给出精确估计。

**第二阶段（2023-2025，条件 OT 理论化）**：静态理论上，Hosseini-Hsu-Taghvaei（SIAM/ASA JUQ 2025）把 block-triangular 条件传输推广到无穷维函数空间；数值上 PCP-Map/COT-Flow 给出两类神经求解器。动态理论的突破是 Kerrigan et al. 的 COT-FM（NeurIPS 2024）：条件版 McCann 插值与 Benamou-Brenier 定理，把条件生成写成条件 Wasserstein 空间中的测地流。Chemseddine et al.（JMLR 2025）补上度量端：一般的 joint W2 不控制 posterior W2，必须用受限耦合的条件 Wasserstein 距离，其测地线速度场在条件分量为零——这直接解释了为何 FM 中"条件不该被传输"。工程端 C2OT（ICCV 2025）发现同一问题的实践面：无条件 minibatch OT 令训练先验按条件偏斜、测试时无法复现，在代价矩阵加条件项即可修复。

**第三阶段（2024-2025，guidance 的理论觉醒与控制论统一）**：Wu et al.（ICML 2024）、Chidambaram et al.（NeurIPS 2024）、Bradley-Nakkiran 三线并进，证明 CFG 不采样 tilted 分布：它推高置信度、压缩熵、把样本推向支撑集边界；Kynkäänniemi et al.（NeurIPS 2024）给出限区间调度的经验修正。统一框架随后出现：DEFT 用 Doob h-transform 收编各类条件采样，Berner et al. 的最优控制视角（[arXiv](https://arxiv.org/abs/2211.01364)）与 Uehara、Tang 系列把 fine-tune 写成熵正则 SOC，Adjoint Matching（ICLR 2025）证明 memoryless 噪声调度是收敛到 KL-tilted 分布的必要条件并把 SOC 化为回归；RB-Modulation（ICLR 2025）把风格约束写进 terminal cost 实现 training-free 控制。修正分布偏移方面，FKC（ICML 2025）用 Feynman-Kac/SMC 加权精确采样退火与乘积分布，CFGIG 补出被 CFG 丢掉的 Rényi 排斥项。

**第四阶段（2025-2026，W2 视角登场）**：正则从 KL 走向 W2——ORW-CFM-W2（ICLR 2025）用 FM 速度场差的 W2 上界防 reward 坍缩；Tang-Xu 给出 guided 分布与目标条件分布的 W2 界；DPAC 证明 path-KL 同时上界 W2 与 FID 并投影 guidance 梯度。条件 OT 则扩散到新领域（ICLR 2026 条件 Lagrangian OT Oral）。整体趋势：耦合侧（训练时）与控制侧（推理/微调时）两条线正在 W2 几何下汇合。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: 强关联。guidance/SOC 系是"推理时改 drift 而不动权重"的正统理论：DEFT 的 h-transform、RB-Modulation 的 terminal cost、FKC 的 SMC 加权都属 training-free 干预；Tang-Xu 与 DPAC 的 W2 界给"对齐后分布离目标多远"提供了可检验的度量。若轨迹对齐被写成带 running/terminal cost 的控制问题，Adjoint Matching 的 memoryless 结论提示：对齐时的噪声调度选择会决定最终分布是否正确。
- 方向二（OT 引导跨域生成）: 中等偏强关联。OTCS 是"OT 耦合当条件先验"的原型（其无配对翻译应用与 T16 交界）；C2OT/COT-FM 说明跨域条件生成中耦合必须 condition-aware，否则训练/测试先验失配；EgNOT 则展示反方向——用能量 guidance 求解熵 OT 本身，可作为跨域 OT 映射的生成式求解器。

## 5. 开放问题与可发论文的切入点

1. **CFG 分布偏移的 W2/传输代价刻画**：现有 CFG 理论（predictor-corrector、GMM、支撑集边界）全是密度/KL 视角；Tang-Xu 的 W2 界只覆盖 classifier guidance 的条件化设定。具体可做：把 guided 概率流 ODE 写成带 running cost 的 Benamou-Brenier 问题，证明 W2(p_w, p_target) 关于 guidance 强度 w 的（非）单调界，并用它推导最优 w 调度（对照 Kynkäänniemi 的经验区间）；GMM 下可显式算。
2. **训练侧条件耦合 vs 推理侧 guidance 的等价性**：C2OT（改耦合）与 CFG（改 drift）在何种意义下可互换？可证：条件加权 OT 耦合诱导一个"隐式 guidance 强度"，并在相同 NFE 预算下测 FID-CLIP 前沿（C2OT w=0 vs FM+CFG w>0），若两者可组合则给出联合最优的 (耦合温度, w) 配方。
3. **条件 Wasserstein 距离作为 alignment 正则/评测**：Chemseddine 证明 joint W2 不控制 posterior W2，而 ORW-CFM-W2 的正则恰是 joint/marginal 型上界——理论上无法排除"条件坍缩"（不同 y 的输出互相串扰）。用受限耦合的条件 W2 替换之，证明其防坍缩保证，并给出基于 minibatch 条件分组的可计算估计器。
4. **W2-tilted 极限分布定理**：Adjoint Matching 证明 KL 正则 + memoryless 调度 → tilted 分布 p·e^{r/β}；把 KL 换成 W2 后极限分布是什么？猜想为 reward 的 Moreau-Yosida 包络诱导的 Wasserstein proximal（JKO 一步），可与 T05 的梯度流工具对接；先在 Gaussian/GMM 上闭式验证，再给一般凸性条件。
5. **熵 OT 势函数作为精确 energy guidance**：CEP 用对比学习估计中间能量，误差不可控；EgNOT 反向用 EBM 解熵 OT。可把两者闭环：用 Sinkhorn 势（半对偶）直接构造中间时刻的精确 guidance 场，在 offline RL（D4RL）与分子多目标生成上对比 CEP/FKC 的采样偏差。

## 6. 代码与资源

- [TorchCFM](https://github.com/atong01/conditional-flow-matching)——conditional/OT flow matching 标准库（注意 minibatch OT 非总体精确 OT）
- [C2OT](https://github.com/hkchengrex/C2OT)——条件加权 OT 耦合，含 8gaussians→moons / CIFAR-10 / ImageNet 实验与 Colab
- [Conditional_Wasserstein_Distances](https://github.com/JChemseddine/Conditional_Wasserstein_Distances)——条件 W2 与 Bayesian OT-FM 官方实现
- [fkc-diffusion](https://github.com/martaskrt/fkc-diffusion)——Feynman-Kac correctors（CFG 修正/退火/乘积分布 SMC）
- [Energy-guided-Entropic-OT](https://github.com/PetrMokrov/Energy-guided-Entropic-OT)——EBM 解熵 OT，含 AFHQ 512 无配对翻译
- [CEP-energy-guided-diffusion](https://github.com/thu-ml/CEP-energy-guided-diffusion)——精确 energy guidance + QGPO（D4RL）
- [OTCS](https://github.com/XJTU-XGU/OTCS)——OT 耦合条件 score 模型（无配对超分/半配对翻译）
- [guidance-interval](https://github.com/kynkaat/guidance-interval)——限区间 guidance 官方实现
- [orw-cfm](https://github.com/markerthu/orw-cfm)——W2 正则在线 reward fine-tune（SD3 实验）
- [RB-Modulation 项目页](https://rb-modulation.github.io/)——terminal cost 控制的 training-free 风格化
- 常用基准：8gaussians→moons（条件 2D 合成）、CIFAR-10/ImageNet-32/256（类条件）、Darcy Flow/Lotka-Volterra（Bayesian 逆问题，COT-FM）、D4RL（energy guidance）、GenEval/human preference reward（alignment）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2024_Kerrigan_cot_flow_matching.pdf | Dynamic Conditional Optimal Transport through Simulation-Free Flows | 成功 (1.9MB) |
| 2025_Chemseddine_conditional_wasserstein_distances.pdf | Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching | 成功 (5.6MB, JMLR 官方) |
| 2025_Cheng_c2ot_curse_of_conditions.pdf | The Curse of Conditions: Analyzing and Improving OT for Conditional Flow-Based Generation | 成功 (8.3MB, CVF 官方) |
| 2025_DomingoEnrich_adjoint_matching.pdf | Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless SOC | 成功 (11.9MB) |
| 2025_Skreta_feynman_kac_correctors.pdf | Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts | 成功 (14.2MB) |
| 2025_Fan_orw_cfm_w2.pdf | Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization | 成功 (39.8MB) |
| 2024_Chidambaram_what_does_guidance_do.pdf | What does guidance do? A fine-grained analysis in a simple setting | 成功 (3.4MB) |
| 2024_Mokrov_energy_guided_entropic_ot.pdf | Energy-guided Entropic Neural Optimal Transport | 成功 (32.5MB, ICLR 官方) |
