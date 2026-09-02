# T13 神经 OT 映射与无配对图像翻译

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」全景中**不走扩散桥/SB、直接把 OT map/plan 学成一个前馈网络**的静态映射路线：一步推理完成 unpaired 翻译与生成，是 T14（扩散桥/SB 翻译）的低成本方法学对照组，也是 OT 对偶理论（semi-dual、weak OT、UOT）落地为神经算法最直接的地方。该方向由 Korotin/Skoltech 系与 Choi/首尔大 KIAS 系两条谱系主导。

## 1. 核心问题与背景

无配对图像翻译要在没有 (x, y) 对应样本的条件下学习 X→Y 的映射。CycleGAN 一类方法靠循环一致性约束，但"该保留什么"没有数学定义。神经 OT 路线把任务形式化为 Monge/Kantorovich 问题：在所有把源分布推前到目标分布的映射中，选传输成本最小者，天然给出"保内容"的归纳偏置，且解有唯一性/最优性理论。核心技术问题包括：(i) 连续高维分布间如何用神经网络求 OT map——主流是半对偶 max-min（位势 f 与映射 T 对抗）；(ii) max-min 的解是否真是 OT map——fake/spurious solution 与误差分析（duality gap 界）；(iii) 真实数据两侧质量不平衡、含离群点——UOT map（UOTM 系列）；(iv) 一对多翻译需要随机 plan——weak OT 与 kernel weak cost；(v) 非 L2 成本承载任务先验（保类、跨维度、语义引导）；(vi) 如何客观评测 map 精度——Wasserstein-2 benchmark。2024-2026 的主线是把 max-min 变得更稳（DIOTM、ENOT、OTP）、更一般（general cost、条件 UOT、流形），并与 flow matching 融合（UOT-FM）。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark | 2021·NeurIPS | [P] | 用 ICNN 构造有解析 ground-truth OT map 的连续分布对（含图像空间），系统评测 W2 求解器，揭示"下游表现好 ≠ map 准" | [proceedings](https://proceedings.neurips.cc/paper/2021/hash/7a6a6127ff85640ec69691fb0f7cb1a2-Abstract.html) |
| ⭐ Generative Modeling with Optimal Transport Maps (OTM) | 2022·ICLR | [P] | 首次在高维 ambient 图像空间把 W2 map 本身当生成器；min-max 算法 + 基于 duality gap 的误差界；unpaired 去噪/上色/补全 | [OpenReview](https://openreview.net/forum?id=5JdLZg346Lw) |
| ⭐ Neural Optimal Transport (NOT) | 2023·ICLR Spotlight | [P] | 统一 strong/weak cost 的 saddle-point 求解器，证明 NN 是 transport plan 的万能逼近器；one-to-one 与 one-to-many unpaired 翻译 | [OpenReview](https://openreview.net/forum?id=d8CBRlWNkqH) |
| Kernel Neural Optimal Transport | 2023·ICLR | [P] | 证明 γ-weak quadratic cost 的 NOT 存在 fake solutions；改用 kernel weak cost 修复并改善理论保证与多样性 | [OpenReview](https://openreview.net/forum?id=Zuc_MHtUma4) |
| Neural Monge Map Estimation and Its Applications | 2023·TMLR (Featured) | [P] | 一般 cost、可跨维度的 Monge map 弱式 max-min 求解；用 duality gap 给出严格的后验误差分析；unpaired 文生图/补全 | [OpenReview](https://openreview.net/forum?id=2mZSlQscj3) |
| The Monge Gap: A Regularizer to Learn All Transport Maps | 2023·ICML | [P] | 摆脱 ICNN 与 minimax：Monge gap 正则度量任意映射偏离 c-最优的程度，任意 cost 下单目标回归学 map | [PMLR](https://proceedings.mlr.press/v202/uscidda23a.html) |
| ⭐ Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport (UOTM) | 2023·NeurIPS | [P] | UOT 半对偶的生成模型：φ-divergence 松弛边际约束，outlier 稳健、训练稳、收敛快（CIFAR-10 FID 2.97） | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/84706cdfc192cd0351daf48f379847e6-Abstract-Conference.html) |
| Extremal Domain Translation with Neural Optimal Transport | 2023·NeurIPS | [P] | 提出 extremal transport：翻译保真度的理论最优形式化，用 incomplete transport（partial OT 特例）的极限逼近 ET map | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7eed2822411dc37b3768ae04561caafa-Abstract-Conference.html) |
| Analyzing and Improving Optimal-Transport-based Adversarial Networks (UOTM-SD) | 2024·ICLR | [P] | 统一 OT-based GAN 框架逐组件分析；divergence 调度（τ 渐增）解决 UOTM 超参敏感，FID 2.51/CIFAR-10 | [OpenReview](https://openreview.net/forum?id=jODehvtTDx) |
| Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation (UOT-FM) | 2024·ICLR | [P] | 证明 unbalanced Monge map = 两个重缩放测度间的 balanced map，可插入任意估计器（含 OT-FM）；确立 UOT-FM 为 unpaired 翻译的原则性方法 | [OpenReview](https://openreview.net/forum?id=2UnCj3jeao) |
| Neural Optimal Transport with General Cost Functionals | 2024·ICLR | [P] | cost 从点对点函数推广到一般泛函，支持 class-guided、pair-guided 等任务先验的可控翻译 | [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html) |
| ⭐ ENOT: Expectile Regularization for Fast and Accurate Training of Neural Optimal Transport | 2024·NeurIPS | [P] | 用 expectile 回归正则近似共轭（c-transform）算子，替代昂贵不稳的内环优化，W2 基准上速度/精度大幅提升 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d885c74aa0e00cc07a35346aa7988e34-Abstract-Conference.html) |
| Improving Neural Optimal Transport via Displacement Interpolation (DIOTM) | 2025·ICLR | [P] | 导出 displacement interpolation 逐时刻对偶并证明跨时刻关联，用全轨迹 + HJB 正则稳定 max-min；I2I 翻译 FID 显著改善 | [OpenReview](https://openreview.net/forum?id=CfZPzH7ftt) |
| Overcoming Spurious Solutions in Semi-Dual Neural Optimal Transport (OTP) | 2025·ICML | [P] | 给出 semi-dual max-min 恢复真 OT map 的充分条件；源分布平滑化 + 渐退火学 OT plan，可学随机映射（one-to-many 上色） | [PMLR](https://proceedings.mlr.press/v267/choi25a.html) |
| Conditional Unbalanced Optimal Transport Maps (CUOTM) | 2026·arXiv (2026-03) | [R] | 条件 UOT 的 dual/semi-dual 形式 + 三角 c-transform 参数化，outlier-robust 的条件生成模型 | [arXiv](https://arxiv.org/abs/2603.06972) |

## 3. 方法演进脉络

**起点：ICNN 时代（2018-2021）。** Seguy et al.（ICLR 2018）用熵正则对偶 + barycentric projection 做大规模 map 估计；Taghvaei & Jalali（2019）、Makkuva et al.（ICML 2020，[PMLR](https://proceedings.mlr.press/v119/makkuva20a.html)）利用 Brenier 定理把 W2 求解写成 ICNN 上的 minimax；W2GN（Korotin et al., ICLR 2021）用循环正则替代。但 ICNN 表达力与可优化性差。**W2 benchmark（NeurIPS 2021）**首次给出带解析真值的系统评测：许多求解器下游好但 map 不准，maximin 型（MM:R，非 ICNN 网络）反而最稳——这一实证直接催生了后续"扔掉 ICNN"的主流路线。W1 侧的姊妹工作 Kantorovich Strikes Back!（NeurIPS 2022 Datasets & Benchmarks，[OpenReview](https://openreview.net/forum?id=VtEEpi-dGlt)）进一步表明 WGAN 求解器估不准 W1、只是梯度方向可用，厘清了"WGAN≠OT"的边界。

**主干：semi-dual max-min 在 ambient 空间成型（2021-2023）。** OTM（Rout et al., ICLR 2022）用普通 CNN 直接在图像空间解 min-max，并给出 duality-gap 误差界——"误差分析"这条线由 Fan et al.（TMLR 2023）推广到一般 cost 与跨维度、给出严格后验误差分析。NOT（ICLR 2023 Spotlight）把框架统一到 weak OT：一个网络学确定 map，加噪声输入学随机 plan（one-to-many 翻译），并证万能逼近；Kernel NOT（ICLR 2023）随即发现 weak quadratic cost 有 fake solutions 并用 kernel cost 修复。"发现缺陷→修复"的循环在 OTP（ICML 2025）到达新深度：首次刻画 max-min 恢复真 OT map 的充分条件，条件不满足时用源分布平滑化 + 退火确保收敛（子序列意义），并能学真正的随机 plan。

**非对抗替代与加速（2023-2024）。** Monge Gap（ICML 2023）用"最优性亏损"正则把 map 学习变成单目标回归；ENOT（NeurIPS 2024）用 expectile 回归近似 c-transform，绕开内环优化；Energy-guided entropic NOT（ICLR 2024，[proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/517eb19e99947f60afff0cf93e451825-Abstract-Conference.html)）把 EBM 引入熵正则半对偶。

**unbalanced 分支（2023-2026）。** UOTM（NeurIPS 2023）把 UOT 半对偶做成生成模型，对 outlier 与训练不稳定同时起效；UOTM-SD（ICLR 2024）用 divergence 调度消除 τ 敏感并让 plan 收敛回 OT；UOT-FM（ICLR 2024）从另一侧证明 unbalanced map 可化归为重缩放边际的 balanced 问题，从而与任意估计器（ICNN、Monge gap、flow matching）即插即用；Light UOT（NeurIPS 2024，[proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aa93a55655e49cc8bf8e6e9295d9b295-Abstract-Conference.html)）给出轻量非 minimax UOT 求解器；CUOTM（2026 preprint）把 UOT map 推到条件生成。翻译语义端，Extremal（NeurIPS 2023）用 incomplete transport 回答"理论上最保真的翻译是什么"。

**静态-动态融合与新前沿（2024-2026）。** DIOTM（ICLR 2025）借动态 OT 的 displacement interpolation 与 HJB 结构反哺静态 map 学习，是本子课题与 T14（桥/SB）之间的天然接口；UNOT（ICML 2025，[PMLR](https://proceedings.mlr.press/v267/geuter25a.html)）用 Fourier Neural Operator 跨数据集/分辨率摊销预测熵正则位势（Sinkhorn 初始化提速至 7.4×），代表"amortized neural OT"新范式；Riemannian NOT（[arXiv 2602.03566](https://arxiv.org/abs/2602.03566)）与 Entropic Riemannian NOT（[arXiv 2605.04255](https://arxiv.org/abs/2605.04255)，均 [R]）把 NOT 推广到流形几何。

## 4. 与博客两个方向的关联

- **方向一（无须重训的轨迹对齐）**: 间接相关。NOT 家族是一步静态映射，不触碰扩散采样轨迹；但其价值在于：(a) OT map 可作为**已训练模型之间的事后分布对齐器**（在 latent/噪声空间学一个轻量 map，不重训基模型，OTM 的噪声→数据配对即此思路）；(b) ENOT/Monge Gap/Light UOT 等把"训练一个小 map"的成本压得很低，使事后对齐实际可行；(c) DIOTM 的时间索引位势与轨迹对齐的 HJB/位势视角在形式上相通，可迁移其正则化技巧。
- **方向二（OT 引导跨域生成）**: 正中靶心，本课题就是"OT map 即翻译器"的主战场。NOT/OTM 提供基础算法；Extremal/UOT 系解决保真度与两域质量不平衡；general cost functionals 允许把类别/语义约束写进 cost 实现可控翻译；UOT-FM 进一步说明 unbalanced 化能直接强化 flow matching 类生成器。与 T14 的 SB/扩散桥构成"一步 vs 多步"的方法学对照：本线推理快、结构保持强，但高分辨率纹理质量弱于扩散桥，融合空间大。

## 5. 开放问题与可发论文的切入点

1. **weak/unbalanced 半对偶的可计算误差证书**：duality-gap 后验误差界目前只覆盖强形式 cost（OTM 的 L2、Fan et al. 的一般 cost）；对 weak、kernel、UOT 半对偶均缺失。可做：把 duality-gap 分析推广到 γ-weak 与 φ-divergence 松弛情形，给出可在训练中监控的 a-posteriori 证书，并在 W2 benchmark + 闭式 Gaussian(-mixture) UOT 真值上验证证书与真误差的相关性。
2. **新一代 neural OT benchmark**：2021 W2 benchmark 用 ICNN 造真值、最高只到 CelebA 64px，且对 weak/UOT/条件 cost 无 ground truth，2024-2026 的新方法（UOTM-SD、DIOTM、OTP）实际上缺少 map 级评测。可做：以闭式 Gaussian mixture UOT 解 + 预训练 VAE latent 空间组合构造高维新基准，系统重评一代方法的 map 精度（而非 FID），并复现 OTP 指出的 spurious-solution 发生率。
3. **自适应 unbalancedness**：UOTM-SD 的 τ 调度是全局手工设计。可做：按样本学 per-sample unbalancedness（outlier 分数驱动的 τ(x)），证明其收敛到 balanced OT 的速率，在含污染的 unpaired 翻译与 class-imbalanced 翻译上验证（可与 CVPR 2026 的 semi-UOT OOD 动态重权思路互证）。
4. **一步 OT map + 少步扩散 refinement 的混合 pipeline**：NOT/UOTM 在 ≥256px 纹理掉分，SB/桥推理贵。可做：OT map 先做结构/内容粗对齐，再用冻结的预训练扩散模型 2-4 步细化；量化"传输成本-输入保真-FID"三方 trade-off，并与 latent NOT（在 DINOv2/SD-VAE latent 上学 map）比较——这是连接 T13 与 T14 的空白地带。
5. **weak/kernel cost 的多样性理论**：one-to-many 翻译中 γ 与 kernel 的选择决定多样性-保真 trade-off，目前纯经验调参。可做：证明条件分布多样性（如条件熵/支撑集直径）关于 γ 与 kernel 谱的定量下界，据此设计自动选 γ 准则，在 colorization（OTP 已示 one-to-many）上验证。

## 6. 代码与资源

- NOT 官方实现: https://github.com/iamalexkorotin/NeuralOptimalTransport （含 one-to-one/one-to-many 翻译 notebook）
- Wasserstein-2 benchmark: https://github.com/iamalexkorotin/Wasserstein2Benchmark ；W1 侧: https://github.com/justkolesov/Wasserstein1Benchmark
- UOTM: https://github.com/Jae-Moo/UOTM
- UOT-FM: https://github.com/ExplainableML/uot-fm
- Extremal/Incomplete transport: https://github.com/milenagazdieva/ExtremalNeuralOptimalTransport
- 一般 cost Monge map（Fan et al.）: https://github.com/sbyebss/monge_map_solver
- UNOT: https://github.com/GregorKornhardt/UNOT
- OTT-JAX（含 Monge gap、entropic map 等工具）: https://github.com/ott-jax/ott
- 常用 unpaired 翻译数据对：CelebA(-HQ) male↔female、AFHQ wild→cat、handbags↔shoes、celeba→anime、outdoor→church；map 精度评测用 W2 benchmark 的 ICNN 构造对

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2021_Korotin_wasserstein2_benchmark.pdf | Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark | 成功 |
| 2022_Rout_generative_modeling_ot_maps.pdf | Generative Modeling with Optimal Transport Maps | 成功 |
| 2023_Korotin_neural_optimal_transport.pdf | Neural Optimal Transport | 成功 |
| 2023_Choi_uotm_semidual_unbalanced.pdf | Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport | 成功 |
| 2024_Asadulaev_general_cost_functionals.pdf | Neural Optimal Transport with General Cost Functionals | 成功 |
| 2024_Buzun_enot_expectile.pdf | ENOT: Expectile Regularization for Fast and Accurate Training of Neural Optimal Transport | 成功 |
| 2025_Choi_diotm_displacement_interpolation.pdf | Improving Neural Optimal Transport via Displacement Interpolation | 成功 |
| 2025_Choi_otp_spurious_solutions.pdf | Overcoming Spurious Solutions in Semi-Dual Neural Optimal Transport | 成功 |
