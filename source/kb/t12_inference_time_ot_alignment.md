# T12 推理阶段的 OT 对齐与噪声-样本耦合

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题覆盖「不重训模型、在推理或数据准备阶段引入测度耦合」的全部机制：噪声-样本指派、初始噪声选择/优化/检索/搜索、prior error 的 OT 桥接、跨帧噪声传输。它是博客方向一（无须重训的轨迹对齐）的核心弹药库；训练期 loss 级 minibatch OT 耦合归 T08，ODE/SDE 求解器本身归 T11，本篇只在接口处引用它们。

## 1. 核心问题与背景

预训练扩散/流模型采样时从 \(z\sim\mathcal N(0,I)\) 独立起步，\((z,\text{条件}/\text{样本})\) 的默认耦合是乘积测度。但 PF-ODE 定义了确定性的噪声↔样本映射，其几何性质（是否近似 OT、轨迹直线度）直接决定少步采样质量与语义可控性。两个经验/理论事实让"推理期耦合"变得既可行又有价值：(i) DDPM encoder 映射被证明在高斯情形恰为 OT 映射、一般情形数值上近似 OT（Khrulkov–Oseledets 猜想），且不同架构、不同训练的模型学到几乎同一个噪声↔样本映射（ICML 2024 可复现性现象）——耦合是数据的内在属性，可以离线预计算、跨模型复用；(ii) 初始噪声携带强语义先验（"silent prompt"），换一个起点就能换布局、语义与美学。于是无须重训即可在多个环节插入测度耦合：训练前的数据管线噪声指派、采样起点的选择/优化/检索/搜索、先验→前向终态的一步 OT 桥、batch 内重排、跨帧噪声传输。核心张力是**改耦合 vs 保边缘**：任何实例级挑选都会使有效初始分布偏离 \(\mathcal N(0,I)\)，如何在提升对齐的同时控制生成分布漂移（多样性、FID、reward hacking）是贯穿全线的问题。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Understanding DDPM Latent Codes Through Optimal Transport (Khrulkov & Oseledets) | 2023·ICLR | [P] | 奠基猜想：DDPM encoder（PF-ODE 映射）≈ OT 映射；高斯情形严格证明，一般情形张量列车解 Fokker–Planck 数值验证 | [arXiv](https://arxiv.org/abs/2202.07477) |
| The Flow Map of the Fokker–Planck Equation Does Not Provide Optimal Transport (Lavenant & Santambrogio) | 2022·Applied Math. Letters | [P] | 否定性反例：光滑快衰减初分布下 PF-ODE 流映射一般**不是** OT 映射（但数值上"几乎最优"，量化亏损是开放题） | [期刊页](https://www.sciencedirect.com/science/article/abs/pii/S089396592200180X) |
| The Emergence of Reproducibility and Consistency in Diffusion Models (Zhang et al.) | 2024·ICML | [P] | 同一初始噪声+确定性采样器下，不同框架/架构/训练的模型输出几乎相同——噪声↔样本耦合是数据内在的、可跨模型复用的对象 | [PMLR](https://proceedings.mlr.press/v235/zhang24cn.html) |
| ⭐ Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment (Li et al.) | 2024·NeurIPS | [P] | 数据管线级噪声指派：batch 内量化线性指派把每张图映到就近噪声（1024 batch 仅 22.8ms、一行代码），训练加速至 3× | [NeurIPS 页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a422a2f016c14406a01ddba731c0969a-Abstract-Conference.html) |
| Improved Immiscible Diffusion: Accelerate Diffusion Training by Reducing Its Miscibility (Li et al.) | 2025·arXiv | [R] | 把概念推广为"任意层的可混性降低"：KNN 噪声选择（O(n)、0.2ms/256batch）、image scaling 等实现族，>4× 加速；证明去噪双射性故不损多样性，给出 OT 助益扩散的新解释 | [arXiv](https://arxiv.org/abs/2505.18521) |
| Improving Diffusion-Based Generative Models via Approximated Optimal Transport (Kim et al.) | 2024·arXiv | [R] | EDM 侧等价做法：训练时用近似 OT 为每张图选噪声，ODE 轨迹曲率降低，CIFAR-10 达 FID 1.68/1.58@29NFE（自称 IJCAI 2024，官方 proceedings 未核验） | [arXiv](https://arxiv.org/abs/2403.05069) |
| ⭐ Solving Prior Distribution Mismatch in Diffusion Models via Optimal Transport (Wang et al.) | 2024·arXiv | [R] | 证明扩散两阶段本质是计算时变 OT、概率流指数收敛到 Monge–Ampère 解的梯度；据此用**半离散静态 OT（Brenier 势的几何变分解）**桥接 \(p_\infty\to p_{T'}\) 消除 prior error，实现"一步 OT + 短程扩散"加速采样 | [arXiv](https://arxiv.org/abs/2410.13431) |
| InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization (Guo et al.) | 2024·CVPR | [P] | 起点优化开山：以交叉/自注意力目标在"有效区域"内梯度优化初始噪声（含高斯性约束），治 subject mixing/neglect | [arXiv](https://arxiv.org/abs/2404.04650) |
| ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization (Eyring et al.) | 2024·NeurIPS | [P] | 对一步模型以人类偏好 reward 梯度上升优化初始噪声（50 步、20–50s），一步模型反超 SDXL、比肩 SD3 | [OpenReview](https://openreview.net/forum?id=MXY0qsGgeO) |
| ⭐ Golden Noise for Diffusion Models: A Learning Framework (Zhou et al., NPNet) | 2025·ICCV | [P] | 学一个噪声→"黄金噪声"的传输网络：re-denoise 采样+偏好模型筛选构建 10 万对噪声数据集，SVD 结构先验的小网络即插即用（+3% 开销），跨模型/采样器泛化 | [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Golden_Noise_for_Diffusion_Models_A_Learning_Framework_ICCV_2025_paper.pdf) |
| A Noise is Worth Diffusion Guidance (Ahn et al., NoiseRefine) | 2026·ICLR | [A] | 把 CFG 折叠进初值：学噪声→"guidance-free 噪声"的一次映射，低频小幅分量替代引导，同管线免 CFG 高质量生成、吞吐/显存双省 | [ICLR 页](https://iclr.cc/virtual/2026/poster/10006657) / [arXiv](https://arxiv.org/abs/2412.03895) |
| The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation (Wang et al.) | 2025·ICCV Highlight | [P] | 免优化检索式耦合：离线构建 10 万噪声库（键=无条件生成后验的语义/低层特征），推理 0.2ms 查库选起点，跨 T2I 模型零样本迁移 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_The_Silent_Assistant_NoiseQuery_as_Implicit_Guidance_for_Goal-Driven_Image_ICCV_2025_paper.html) |
| ⭐ Scaling Inference Time Compute for Diffusion Models (Ma et al.) | 2025·CVPR | [P] | 把"找好噪声"形式化为验证器×搜索算法的设计空间（random / zero-order / search-over-paths），确立噪声搜索作为扩散 test-time scaling 的第二轴 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.html) |
| How I Warped Your Noise: a Temporally-Correlated Noise Prior for Diffusion Models (Chang et al.) | 2024·ICLR Oral | [P] | 噪声-噪声耦合的理论工具：∫-noise 积分噪声表示+噪声传输方程，沿光流保分布地 warp 高斯噪声，免训练提升视频时间一致性 | [OpenReview](https://openreview.net/forum?id=pzElnMrgSD) |
| Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise (Burgert et al.) | 2025·CVPR Oral | [P] | 实时噪声 warp 算法（逐帧迭代传输、保空间高斯性），把运动控制变成"换结构化初值"，模型结构与训练管线零改动 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Burgert_Go-with-the-Flow_Motion-Controllable_Video_Diffusion_Models_Using_Real-Time_Warped_Noise_CVPR_2025_paper.html) |

表外相邻工作（正文引用）：DOODL（端到端可微 latent/噪声优化的前史，ICCV 2023，[arXiv](https://arxiv.org/abs/2303.13703)，[P]）、DNO: Optimizing Diffusion Noise Can Serve as Universal Motion Priors（把噪声当通用运动先验做编辑/补全，CVPR 2024，[arXiv](https://arxiv.org/abs/2312.11994)，[P]）、FreeInit（视频初始噪声低频重初始化，ECCV 2024，[arXiv](https://arxiv.org/abs/2312.07537)，[P]）、Tanana 2021（热流插值映射 vs Brenier 映射的更早比较，Comm. Contemp. Math.，[P]）、Noise-Level Diffusion Guidance（免反传的噪声级引导，用模型自身输出扰动噪声，[arXiv 2509.13936](https://arxiv.org/abs/2509.13936)，[R]）、Oracle Noise（指出欧氏梯度噪声优化会推离超球流形并给出球面对齐修正，[arXiv 2604.23540](https://arxiv.org/abs/2604.23540)，[R]）、OTCache（OT 启发的缓存调度预测做免训练加速，偏 T11 外围，[arXiv 2606.31026](https://arxiv.org/abs/2606.31026)，[R]）。

## 3. 方法演进脉络

**理论起点（2021–2024，"encoder 是不是 OT"）**：Khrulkov–Oseledets 猜想 DDPM encoder 即 OT 映射并在高斯情形证明；Lavenant–Santambrogio（承接 Tanana 2021）构造反例证明一般不成立，但指出数值上"几乎最优"、量化亏损是开放题。Wang et al.(2024) 把链条补全：离散初分布下概率流在任意闭区间上确为动态 OT，且随扩散时间增长指数收敛到 Monge–Ampère 解的梯度。ICML 2024 可复现性研究给出经验补钉——噪声↔样本映射跨模型一致，意味着围绕这个"准 OT 耦合"做的任何离线构造（噪声库、golden noise 数据集）都可跨模型复用。

**数据管线侧（2024–2025，指派进管线而非 loss）**：Immiscible Diffusion 与 AOT 同期把 batch 级噪声指派塞进数据准备（量化线性指派/近似 OT 选噪声），不改 loss、不改架构、一行代码，训练加速最高 3–4×、轨迹曲率下降；2025 扩展版把概念抽象为"降低轨迹可混性"，用 KNN 噪声选择把开销降到 O(n)，并证明去噪双射性保多样性。与 T08 的分工：T08 的 OT-CFM 系把耦合写进 FM 训练目标，本线只动数据管线，故同样适用于"已有训练脚本零侵入后装"。

**推理起点侧（2023–2026，从优化到学习到检索到搜索）**：DOODL/InitNO/ReNO/DNO 构成梯度优化线——目标从注意力能量到偏好 reward，代价是反传显存与 reward hacking；NPNet（golden noise）与 NoiseRefine 构成学习映射线——离线蒸馏"随机噪声→好噪声"的小网络，推理只多一次前传，后者更把 CFG 的作用整体折叠进初值；NoiseQuery 构成免优化检索线——噪声库+特征键值查询 0.2ms 选起点；Ma et al. 构成搜索线——verifier×算法的 test-time scaling 第二轴，并从"只搜初值"扩展到"沿路径搜噪声注入"（search over paths，把耦合插入轨迹中段）。2026 的 Oracle Noise、Noise-Level Guidance 开始修正该家族的几何缺陷（保持超球流形、免反传）。

**OT 显式化与噪声传输（2024–2026）**：Wang et al. 把"prior error 校正"做成半离散静态 OT 桥（几何变分求 Brenier 势、奇异集检测处理非凸支撑），采样变为"一步 OT + 短程扩散"；视频侧 HIWYN 推导保分布的噪声传输方程（∫-noise），Go-with-the-Flow 把它做成实时算法并规模化——这是"噪声-噪声耦合"（跨帧、跨视角）的成熟实例。整体趋势：从启发式挑噪声走向有测度论保证的耦合设计，从像素空间走向 latent，从单样本 top-1 检索走向保边缘的 batch 级指派（尚是空位）。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **正中靶心**。可插入测度耦合的环节清单（按管线顺序）：
  1. **训练前数据管线**——batch 内噪声指派/KNN 选择（Immiscible、AOT、Improved Immiscible）：改 \((x_0,z)\) 耦合，唯一涉及训练但零侵入；
  2. **ODE 初值选择（离散耦合）**——噪声库检索（NoiseQuery）、验证器搜索（Ma et al.）：把 \(q(c)q(z)\) 换成对齐的 \(\pi(c,z)\)，top-1-of-k 选择即 order-statistics 耦合，边缘只在期望意义保持；
  3. **ODE 初值变换（连续映射）**——梯度优化（InitNO/ReNO/DNO/DOODL）或学习的小位移传输网络（NPNet、NoiseRefine）：本质是构造 prior-preserving 的映射 \(T:\mathcal N(0,I)\to\)"好噪声"区域，NPNet 的小扰动≈小 \(W_2\) 位移的 Monge 映射；
  4. **先验→前向终态桥接**——半离散静态 OT 替换扩散尾段（Wang et al.）：一步 Brenier 映射消 prior error 并缩短积分区间；
  5. **轨迹中段**——search over paths 在中间时刻重注噪声再搜索（Ma et al.）；batch 内中段重排 \((x_t,c)\) 目前几乎空白；
  6. **跨帧/跨视角噪声-噪声耦合**——∫-noise 传输方程（HIWYN）、实时 warp（Go-with-the-Flow）、低频重初始化（FreeInit）；
  7. **encoder/inversion 侧**——PF-ODE encoder 本身≈OT（Khrulkov/Lavenant 定界），DDIM inversion 产出的 (样本→噪声) 耦合正是 NPNet/NoiseRefine 训练数据的来源（re-denoise/inversion 采集）。
- 方向二（OT 引导跨域生成）: 间接相关但有三个接口——(a) Wang et al. 的半离散 OT 桥不限于高斯先验，可把任意 source 分布 OT 对齐到 \(p_{T'}\) 作为跨域生成的初始化（SDEdit 类方法的 OT 化起点）；(b) NoiseQuery 的特征键值噪声库可按目标域风格/低层属性检索，相当于把跨域先验编码进初值耦合；(c) HIWYN/GwtF 的保分布噪声传输是"跨帧域"耦合的特例，同一机制可推广到跨视角/跨模态的结构化初值构造。

## 5. 开放问题与可发论文的切入点

1. **实例级选择的保边缘理论**：噪声检索/搜索（top-1-of-k）使有效初始分布偏离 \(\mathcal N(0,I)\)，目前无人刻画。可证目标：给定选择强度 k 与验证器，生成分布 =（采样映射）#（order-statistics 耦合）与原分布的 \(W_2\)/KL 偏差上界（Immiscible 的双射性论证可迁移）；实验在 ImageNet/GenEval 上扫 k，画"对齐收益 vs FID/recall 损失"曲线并对照理论界。
2. **免训练 batch 级重排采样（严格保边缘的推理耦合）**：把 NoiseQuery 的 per-prompt top-1 检索升级为 batch 内 \((c_i,z_j)\) 的一次性 Hungarian/Sinkhorn 指派（成本=条件嵌入与噪声生成后验特征的距离，库可离线预计算）——每个噪声恰用一次，batch 边缘严格保持。理论（保边缘性+方差降低）+ 实验（T2I-CompBench/GenEval，对比 top-1 检索与随机耦合）即一篇干净的论文；这是 2 与 5 号环节之间明确的空位。
3. **Monge–Ampère 桥的规模化**：Wang et al. 的几何变分半离散 OT 难上高维 latent。用熵正则半离散对偶（SGD 学势函数）或 ICNN 摊销替换，把"一步 OT + 短程扩散"推到 SDXL/FLUX latent 空间，与蒸馏（LCM/DMD）在同 NFE 下对比 FID 与多样性；顺带回答"OT 桥 vs 蒸馏谁更保多样性"。
4. **噪声优化的流形/OT 正则**：InitNO/ReNO 的欧氏梯度上升把 z 推离高斯 typical set（Oracle Noise 已指出几何退化）。提出以 \(W_2(\mathcal N(0,I),\,T_\#\mathcal N(0,I))\) 或球面测地约束为正则的噪声优化，证明其对 reward hacking 的抑制（分布漂移界→reward 泛化界），并与 NPNet 的 SVD 结构先验统一成"最小位移传输"框架。
5. **推理期"Monge gap"诊断指标**：Lavenant 反例是低维构造、高维亏损未量化。设计可计算的轨迹级指标（速度场沿轨迹的角度漂移、动能与 \(W_2\) 下界之差）作为"该样本离 OT 测地线多远"的免训练诊断，用于自适应决定在哪个时刻插入耦合/重排或切换步长——与 T11 求解器侧形成互补接口。

## 6. 代码与资源

- [Immiscible Diffusion 官方 repo](https://github.com/yhli123/immiscible-diffusion)（含 KNN/线性指派两种实现）
- [EDM-AOT](https://github.com/large-scale-kim/EDM-AOT)（EDM 上的近似 OT 噪声选择）
- [Golden Noise / NPNet](https://github.com/xie-lab-ml/Golden-Noise-for-Diffusion-Models)（含 [HF 权重](https://huggingface.co/Klayand/GoldenNoiseModel)，支持 SDXL/DreamShaper/Hunyuan-DiT）
- [ReNO](https://github.com/ExplainableML/ReNO)（一步模型 reward 噪声优化）
- [Inference-Time Scaling 项目页](https://inference-scale-diffusion.github.io/)（verifier×search 设计空间）
- [NoiseRefine 项目页](https://cvlab-kaist.github.io/NoiseRefine/)
- [How I Warped Your Noise 项目页](https://warpyournoise.github.io/)（∫-noise 噪声传输）
- [Go-with-the-Flow](https://github.com/Eyeline-Research/Go-with-the-Flow)（实时噪声 warp + 权重）
- [OTCache](https://github.com/UnicomAI/OTCache)（OT 启发缓存调度）
- 常用评测：GenEval、T2I-CompBench（组合对齐）、HPSv2/PickScore/ImageReward（偏好 verifier）、FID/recall（分布漂移监控）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2023_Khrulkov_DDPM_Latent_OT.pdf | Understanding DDPM Latent Codes Through Optimal Transport | 成功 |
| 2024_Li_Immiscible_Diffusion.pdf | Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment | 成功 |
| 2024_Wang_Prior_Mismatch_OT.pdf | Solving Prior Distribution Mismatch in Diffusion Models via Optimal Transport | 成功 |
| 2024_Guo_InitNO.pdf | InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization | 成功 |
| 2024_Eyring_ReNO.pdf | ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization | 成功 |
| 2025_Zhou_Golden_Noise_NPNet.pdf | Golden Noise for Diffusion Models: A Learning Framework | 成功 |
| 2025_Ma_Inference_Time_Scaling_Noise_Search.pdf | Scaling Inference Time Compute for Diffusion Models (Inference-Time Scaling beyond Denoising Steps) | 成功 |
| 2025_Wang_NoiseQuery_Silent_Assistant.pdf | The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation | 成功 |
