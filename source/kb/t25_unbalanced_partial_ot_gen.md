# T25 非平衡/部分 OT 在生成建模中的应用

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 平衡 OT 假设两端质量守恒且每个样本必须被匹配，而真实生成场景充满 outlier、类别失衡与"质量生灭"（细胞增殖/死亡、模式消长）。本子课题覆盖 UOT/partial OT/semi-relaxed OT 的松弛边缘思想如何进入生成建模：从 UOTM 系半对偶对抗生成器，到 unbalanced SB/RUOT/WFR 的生灭动态桥，再到 partial OT 的离群点鲁棒工具链。平衡 OT 耦合（OT-CFM/minibatch）归 T08，单细胞应用细节归 T24，此处只写方法学。

## 1. 核心问题与背景

经典 OT 的两个硬约束——质量守恒与全量匹配——在生成建模中产生三类实际痛点：(i) **离群点敏感**：边缘约束强迫模型为每个噪声样本分配质量，训练目标被污染数据主导，OT-GAN/神经 Monge 映射训练不稳定；(ii) **类别/比例失衡**：源域与目标域类比例不同时，平衡耦合强行跨类搬运质量，跨域翻译中破坏输入特征；(iii) **质量生灭**：细胞增殖凋亡、群体分叉、长尾类目生成等场景中两端本就不是等质量的概率测度。非平衡 OT（KL/Csiszár 散度松弛边缘，或 conic/Hellinger–Kantorovich 提升）、partial OT（只运输 α 比例质量）与 semi-relaxed OT（只松弛单侧边缘）为上述问题提供了统一数学语言。2024–2026 的核心进展是把这些松弛从"离散预处理工具"升级为**生成模型本身的训练目标与动态几何**：半对偶 UOT 直接当生成器目标（UOTM 系）、Wasserstein–Fisher–Rao 几何下的生灭流（RUOT/WFR-FM）、以及带 killing/birth 的非平衡 Schrödinger 桥。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| Optimal Entropy-Transport Problems and a New Hellinger–Kantorovich Distance (Liero, Mielke, Savaré) | 2018·Invent. Math. | [P] | 奠基：熵-运输问题（KL 松弛边缘）、conic 提升与 HK 距离，UOT 的静态理论骨架 | [DOI](https://doi.org/10.1007/s00222-017-0759-8) |
| Unbalanced Optimal Transport: Dynamic and Kantorovich Formulations (Chizat, Peyré, Schmitzer, Vialard) | 2018·J. Funct. Anal. | [P] | 奠基：WFR 动态形式 ≡ 静态 conic 形式，配套 generalized Sinkhorn（Math. Comp. 2018） | [DOI](https://doi.org/10.1016/j.jfa.2018.03.008) |
| ⭐ Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport (UOTM; Choi, Choi, Kang) | 2023·NeurIPS | [P] | 半对偶 UOT 直接作生成目标：对 outlier 鲁棒、收敛更快更稳，CIFAR-10 FID 2.97 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/84706cdfc192cd0351daf48f379847e6-Abstract-Conference.html) |
| Analyzing and Improving Optimal-Transport-based Adversarial Networks (UOTM-SD; Choi et al.) | 2024·ICLR | [P] | 统一 OT-GAN 框架；divergence 权重调度让 UOT 计划收敛到 OT 计划，解 τ 敏感性（FID 2.51） | [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1d051fb631f104cb2a621451f37676b9-Abstract-Conference.html) |
| ⭐ Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation (UOT-FM; Eyring, Klein, Uscidda et al.) | 2024·ICLR | [P] | 证明"非平衡 Monge 映射 = 重缩放测度间的平衡映射"，把 UOT 即插即用进任意估计器与 OT-FM；图像翻译+细胞轨迹双赢 | [OpenReview](https://openreview.net/forum?id=2UnCj3jeao) |
| Scalable Wasserstein Gradient Flow via Unbalanced OT (S-JKO; Choi, Choi, Kang) | 2024·ICML | [P] | 发现 JKO 步 ≡ UOT 问题，半对偶化把 WGF 生成训练复杂度 O(K²)→O(K) | [PMLR](https://proceedings.mlr.press/v235/choi24a.html) |
| Light Unbalanced Optimal Transport (Gazdieva et al.) | 2024·NeurIPS | [P] | 非 minimax、Gaussian-mixture 参数化的轻量 UEOT solver，附普适逼近与泛化界 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aa93a55655e49cc8bf8e6e9295d9b295-Abstract-Conference.html) |
| Unbalanced Diffusion Schrödinger Bridge (UDSB; Pariset, Hsieh, Bunne, Krause, De Bortoli) | 2023·arXiv | [R] | 推导带 killing/birth 项 SDE 的时间反演，把 DSB 推广到任意有限质量边缘（药物响应、病毒变体） | [arXiv](https://arxiv.org/abs/2306.09099) |
| ⭐ Learning Stochastic Dynamics from Snapshots through Regularized Unbalanced OT (DeepRUOT; Zhang, Li, Zhou) | 2025·ICLR oral | [P] | RUOT 神经求解器：无先验联合学 growth/death 与漂移，Fisher 正则打通 RUOT↔SB | [OpenReview](https://openreview.net/forum?id=gQlxd3Mtru) |
| ⭐ Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action (Var-RUOT; Sun et al.) | 2025·NeurIPS | [P] | 把 RUOT 一阶最优性条件写进参数化与损失，单个标量场解 RUOT、作用量更小、训练更稳；讨论 WFR 增长罚函数选择 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/18aee41e1bb41bbb8fee53cfff8138b7-Abstract-Conference.html) |
| Joint Velocity-Growth Flow Matching (VGFM; Wang et al.) | 2025·NeurIPS | [P] | 给静态 semi-relaxed OT 一个"先长质量后运输"的两段式动态解释，联合速度+增长的 simulation-free FM | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eb1bad7a84ef68a64f1afd6577725d45-Abstract-Conference.html) |
| ⭐ WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport (Peng et al.) | 2026·ICLR | [A] | flow matching 同时回归速度场+标量增长率，证明最小化损失恰好回收 WFR 测地线；非平衡快照动态的统一范式 | [OpenReview](https://openreview.net/forum?id=1nqu7bK1mm) |
| BranchSBM: Branched Schrödinger Bridge Matching (Tang, Zhang, Tong, Chatterjee) | 2025·arXiv | [R] | 分支 GSB：每支一个速度场+增长网络，把"一源多汇"的质量再分配化为可分解的 Unbalanced CondSOC | [arXiv](https://arxiv.org/abs/2506.09007) |
| Efficient Algorithms for Robust and Partial Semi-Discrete OT (Agarwal, Raghvendra, Shirzadian, Yao) | 2025·NeurIPS | [P] | α-partial 与 λ-TV-robust 半离散 OT 的 restricted Laguerre 刻画、两问题互相归约与精确/近似算法 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e8f3ca5a82f5511370af7ed0efcad0f-Abstract-Conference.html) |
| Taming Flow Matching with Unbalanced OT into Fast Pansharpening (OTFM; Cao, Zhong, Deng) | 2025·ICCV | [P] | UOT 对偶 + 任务正则构造一步跨模态融合流，UOT 松弛吸收遥感光谱/空间失配 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Cao_Taming_Flow_Matching_with_Unbalanced_Optimal_Transport_into_Fast_Pansharpening_ICCV_2025_paper.html) |

**外围与补充条目**（不占主表，供交叉引用）：
- Robust Optimal Transport with Applications in Generative Modeling and Domain Adaptation（Balaji, Chellappa, Feizi）2020·NeurIPS [P]：最早把 TV-鲁棒 OT 对偶做成深度可训练目标，噪声数据上训 GAN 并自动降权 outlier。[proceedings](https://proceedings.neurips.cc/paper/2020/hash/9719a00ed0c5709d80dfef33795dcef3-Abstract.html)
- P2OT: Progressive Partial Optimal Transport for Deep Imbalanced Clustering，2024·ICLR [P]：渐进 α 调度的 partial OT 伪标签分配，类别不平衡表征学习的代表作。[proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3d03791377a31cb3e7357014ba58eb80-Abstract-Conference.html)
- Revisiting Partial Optimal Transport: A Fast, Robust and Numerically Stable Algorithm，2024·AAAI [P]：partial OT 的数值稳定 Sinkhorn 型算法。[AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/28648)
- Bypassing the Transport Plan: Dynamic Reweighting for OOD Detection with Semi-Unbalanced OT，2026·CVPR [P]：单侧松弛（semi-UOT）+ 动态重权，不物化完整计划。[CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_Bypassing_the_Transport_Plan_Dynamic_Reweighting_for_Out-of-Distribution_Detection_with_CVPR_2026_paper.html)
- Partial Fusion of Neural Networks via Partial Optimal Transport，2026·ICML [A]：partial OT 处理网络神经元的不完全对应做模型融合。[OpenReview](https://openreview.net/forum?id=lvRLG6C0zZ)
- Conditional Unbalanced Optimal Transport Maps (CUOTM)，2026·arXiv [R]：首个条件 UOT 形式化（triangular COT 参数化 + 半对偶 UOT），条件生成的 outlier 鲁棒框架。[arXiv](https://arxiv.org/abs/2603.06972)
- A Regime-Switching Approach to the Unbalanced Schrödinger Bridge Problem，2025·arXiv [R]：把带 killing 的 uSBP 统一成 regime-switching 扩散，系统分析四类 killing 约束。[arXiv](https://arxiv.org/abs/2512.12971)
- Multiscale Supervised UOT Flow Matching，2026·arXiv [R]：层级先验引导的多尺度非平衡 FM。[arXiv](https://arxiv.org/abs/2605.16529)
- Unbalanced Optimal Transport, from Theory to Numerics（Séjourné, Vialard, Peyré）2023·arXiv [B]：UOT 理论-数值系统综述，含 translation-invariant Sinkhorn。[arXiv](https://arxiv.org/abs/2211.08775)

## 3. 方法演进脉络

**理论奠基（2010s）**：partial OT 的自由边界理论（Caffarelli–McCann 2010、Figalli 2010）先行；Liero–Mielke–Savaré (2018) 把"KL 松弛边缘"的熵-运输问题与 conic 提升、Hellinger–Kantorovich 距离系统化，Chizat 等 (2018) 证明其动态形式即 Wasserstein–Fisher–Rao（运输+生灭两种代价），并给出 generalized Sinkhorn 数值。这套"静态散度罚 ↔ 动态生灭几何"的双面结构是后续所有生成工作的模板。

**进入深度生成（2020–2023）**：Balaji 等 (NeurIPS 2020) 首先把 TV-鲁棒 OT 写成可训练对偶用于噪声数据 GAN。决定性一步是 UOTM (NeurIPS 2023)：把 UOT 的半对偶形式直接当生成器-判别器目标，软边缘罚带来 outlier 鲁棒与稳定收敛，性能反超平衡 OT 系（OTM）。同期 UDSB (2023) 从桥的角度推导带 killing/birth 的 SDE 时间反演，把 diffusion Schrödinger bridge 推广到有限测度端点。

**2024：松弛作为可调设计维度**。UOTM-SD (ICLR 2024) 发现 UOT 的 τ 敏感性并用散度调度让非平衡计划逐渐逼近平衡计划——"unbalancedness 不是二选一而是一条可退火的路径"。S-JKO (ICML 2024) 发现 JKO 离散步本身就是一个 UOT 问题，用半对偶把 Wasserstein 梯度流生成模型的复杂度降到线性。UOT-FM (ICLR 2024) 给出最重要的工程接口：非平衡 Monge 问题可重写为**重缩放后测度间的平衡问题**，因此任何估计器（ICNN、Monge Gap、OT-FM）都能以"先离散 UOT 重加权、再平衡耦合"的方式获得非平衡性；在类比例失配的图像翻译与细胞轨迹上全面改进。Light UOT (NeurIPS 2024) 补上轻量理论保证 solver。

**2024–2026：向生灭动态与 simulation-free 收敛**。RUOT 线（同一谱系三连）：DeepRUOT (ICLR 2025 oral) 无先验联合学习漂移与增长场并连接 SB；Var-RUOT (NeurIPS 2025) 把一阶最优性条件（HJB 结构）内嵌进参数化，单标量场即可解 RUOT 且作用量更小；WFR-FM (ICLR 2026) 最终把它做成纯 flow matching——同时回归速度与增长率、损失最小化点恰为 WFR 测地线。平行地，VGFM (NeurIPS 2025) 从 semi-relaxed OT 出发得到"两段式"联合速度-增长 FM；BranchSBM (2025) 把非平衡 CondSOC 分解到多分支处理"一源多汇"分叉；理论侧 regime-switching uSBP (2025) 统一各类 killing 约束。应用侧 OTFM (ICCV 2025)、CUOTM (2026)、semi-UOT OOD (CVPR 2026) 表明 UOT 松弛正在成为跨域/条件生成的默认鲁棒化组件，而 AAAI 2024 与 NeurIPS 2025 的 partial/robust 半离散算法持续供给底层数值工具（α-partial 与 λ-TV-robust 的互归约尤其重要）。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **间接相关但提供关键自由度**。UOT-FM 的重缩放定理表明"非平衡性 = 对耦合两端的样本重加权"，这天然是免重训操作：推理/对齐阶段对 minibatch 耦合先做离散 UOT 重加权即可剔除 outlier 质量、缓解类失配，而无需动网络。NeurIPS 2025 的鲁棒半离散算法（连续噪声端 × 离散数据端正是生成器的标准配置）为这种推理期对齐给出带精度保证的工具。更进一步，WFR 几何提示轨迹对齐除了"搬运"还有"生灭"这第二个控制量——沿轨迹调节质量权重（birth-death 重加权）等价于免重训的模式再平衡（见开放问题 3）。
- 方向二（OT 引导跨域生成）: **直接相关，且是本子课题最强的落点**。UOT-FM 用实验证明：跨域类比例失配时平衡 OT 耦合强行跨类搬运质量、破坏输入保真，UOT 松弛后 FID 与特征保持同时改善——这是"OT 引导跨域生成必须考虑非平衡性"的最直接证据。OTFM (ICCV 2025) 把 UOT 对偶+任务正则做成一步遥感跨模态融合；CUOTM (2026) 把半对偶 UOT 推到条件生成并保证对条件数据 outlier 鲁棒。对含噪声、含错配对的真实跨域数据，UOT/partial OT 是引导项的默认鲁棒化选择。

## 5. 开放问题与可发论文的切入点

1. **minibatch UOT 耦合的 flow matching 系统研究**：OT-CFM（T08）默认平衡 Sinkhorn 耦合；把它换成 translation-invariant UOT Sinkhorn（Séjourné）或 semi-relaxed 耦合，在人工注入 outlier / 类别失衡的 CIFAR-10、ImageNet 子集上量化轨迹直线度、FID、少数类召回率随 τ 的相图；理论上结合 UOT-FM 重缩放定理证明"边缘违反度 vs 直线度/推理步数"的权衡界。
2. **污染率驱动的 τ 自适应调度**：UOTM-SD 的调度是经验的；利用 NeurIPS 2025 证明的 λ-TV-robust ≡ α-partial 等价，把估计污染率 ε 直接映射为 τ(t) 调度并给出 robust 统计意义下的 minimax 保证；实验验证 UOTM/UOT-FM 在 ε-污染 CelebA 上的 FID 崩溃阈值被推后。
3. **生灭作为推理期第二控制量（免重训模式再平衡）**：把 WFR 增长率场 g_t 附加到已训练 diffusion/FM 采样器上，用 birth-death 粒子重生/降权在推理期纠正类别比例或补偿低密度模式；先在 2D 多模态 + 预训练 EDM 上验证，再证明其等价于对 score 的加性 h-transform 修正（与 UDSB 的 killing 时间反演公式对接）。
4. **web 规模错配数据的 partial OT 清洗-加权训练**：CUOTM 只处理无配对 outlier；对含错配 caption 的图文对，用 α-partial OT 只保留可信质量做训练对加权（把 NeurIPS 2025 半离散 α-partial 算法 GPU 批量化），考察 CLIP-FID 与 prompt fidelity；α 可由数据审计先验设定，形成"质量预算"式训练。
5. **神经 UOT 的统计理论缺口**：Light UOT 的泛化界只覆盖 Gaussian-mixture 参数化；推广到神经半对偶 UOTM/UOT-FM 的样本复杂度与 map 估计误差，刻画 KL、χ²、TV 罚在收敛速率上的次序差异；并研究 conic lifting 参数化（HK 几何）是否带来更好的对偶正则性与优化景观。

## 6. 代码与资源

- UOTM 官方实现: https://github.com/Jae-Moo/UOTM （UOTM-SD 见同作者仓库/论文附录）
- UOT-FM 官方实现 (JAX): https://github.com/explainableml/uot-fm
- Light UOT: https://github.com/milenagazdieva/LightUnbalancedOptimalTransport
- DeepRUOT: https://github.com/zhenyiizhang/DeepRUOT ｜ Var-RUOT: https://github.com/ZerooVector/VarRUOT ｜ WFR-FM: https://github.com/QiangweiPeng/WFR-FM （同一谱系，含单细胞 benchmark，细节归 T24）
- VGFM: https://github.com/DongyiWang-66/VGFM
- OTFM (pansharpening): https://github.com/294coder/PAN-OTFM
- Robust OT (NeurIPS 2020): https://github.com/yogeshbalaji/robustOT
- 库：POT（`ot.unbalanced`、`ot.partial` 模块，最全 UOT/POT 基线）https://pythonot.github.io/ ；OTT-JAX（Sinkhorn 支持 `tau_a/tau_b` 非平衡参数）https://ott-jax.readthedocs.io/
- 常用 benchmark：CIFAR-10 / CelebA-HQ FID（UOTM 系）；EMNIST→MNIST、CelebA 属性翻译（UOT-FM）；WV3/GF2 遥感融合（OTFM）；单细胞快照数据（EB、CITE-seq，属 T24）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2023_Choi_uotm_semidual_unbalanced_ot.pdf | Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport | 成功 |
| 2024_Eyring_uotfm_unbalanced_monge_maps.pdf | Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation | 成功 |
| 2025_Zhang_deepruot_snapshot_dynamics.pdf | Learning Stochastic Dynamics from Snapshots through Regularized Unbalanced Optimal Transport | 成功 |
| 2025_Sun_var_ruot_least_action.pdf | Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action | 成功 |
| 2026_Peng_wfr_flow_matching.pdf | WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport | 成功 |
| 2025_Tang_branchsbm_branched_bridge.pdf | Branched Schrödinger Bridge Matching | 成功 |
| 2023_Pariset_unbalanced_diffusion_sb.pdf | Unbalanced Diffusion Schrödinger Bridge | 成功 |
| 2024_Gazdieva_light_unbalanced_ot.pdf | Light Unbalanced Optimal Transport | 成功 |
