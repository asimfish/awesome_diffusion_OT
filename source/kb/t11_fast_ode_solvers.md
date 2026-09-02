# T11 免训练采样器与 ODE 求解器

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 扩散/流模型推理加速的「免训练」主线：把采样视为解概率流 ODE（PF-ODE），通过专用数值格式与时间离散化优化，在 5–20 NFE 内逼近千步教师轨迹。在「扩散×OT」全景中，本子课题提供了「采样轨迹几何 = 传输路径几何」的经验与理论支点——轨迹低维、准直线、与 OT 位移插值的偏差可量化——是方向一（无须重训的轨迹对齐）的数值分析基座。

## 1. 核心问题与背景

扩散模型生成等价于从噪声先验沿反向 SDE 或其等价 PF-ODE 积分到数据分布，朴素祖先采样需约 1000 次网络前传（NFE）。免训练加速把问题转化为数值分析：**不改动预训练权重**，只重新设计求解器与调度，使极少步数的离散解逼近真解。三条技术轴：(i) 求解器结构——利用扩散 ODE 的半线性结构做指数积分（DPM-Solver/DEIS）、多步与预测-校正格式（UniPC）、参数化选择（ε-pred vs x0-pred，DPM-Solver++/v3）；(ii) 时间步调度——低 NFE 下截断误差集中于轨迹高曲率区段，手工调度（uniform-t、log-SNR、EDM ρ=7）远非最优，2024 年起被原理化优化取代（AYS/GITS/LD3 等）；(iii) 轨迹几何分析——采样轨迹被证实近似躺在低维子空间、形状高度正则（"回旋镖"形），其曲率即「与直线传输的偏差」。该方向重要性：大模型时代重训与蒸馏成本急剧上升，training-free 是唯一即插即用路径；同时轨迹几何为理解扩散模型隐含的传输结构（是否接近 OT 映射）提供了最直接的窗口。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| DDIM: Denoising Diffusion Implicit Models | 2021·ICLR | [P] | 把 DDPM 采样确定化为非马尔可夫隐式过程（PF-ODE 的一阶指数式离散），首个 10–50 步实用采样器，并给出确定性 encode–decode 映射 | [arXiv](https://arxiv.org/abs/2010.02502) |
| ⭐ EDM: Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al.) | 2022·NeurIPS | [P] | 用 σ 参数化统一各家训练/采样设计空间：ρ=7 时间调度、Heun 二阶格式、churn 随机性，至今仍是 few-NFE 采样的默认骨架与基准 | [arXiv](https://arxiv.org/abs/2206.00364) |
| ⭐ DPM-Solver | 2022·NeurIPS | [P] | 利用扩散 ODE 半线性结构的定制指数积分器 + log-SNR 换元，线性项解析解出、只近似神经网络项，10–20 NFE 高质量采样并有收敛阶证明 | [arXiv](https://arxiv.org/abs/2206.00927) |
| DPM-Solver++ | 2022·arXiv → 2025·Mach. Intell. Res. 22(4):730-751 | [P] | 改用 data-prediction 参数化 + 多步格式 + thresholding，解决大 guidance 尺度下高阶求解器的失稳，成为文生图部署标准 | [arXiv](https://arxiv.org/abs/2211.01095) · [期刊页](https://link.springer.com/article/10.1007/s11633-025-1562-4) |
| DEIS: Fast Sampling with Exponential Integrator | 2023·ICLR | [P] | 与 DPM-Solver 同期独立提出指数积分器，配 Adams-Bashforth 型多项式外推；其 iPNDM 变体至今仍是低 NFE 强基线 | [arXiv](https://arxiv.org/abs/2204.13902) |
| UniPC: Unified Predictor-Corrector Framework | 2023·NeurIPS | [P] | 统一任意阶预测-校正框架（UniP+UniC），校正器无额外 NFE 即提升低步数精度，diffusers 生态默认求解器之一 | [arXiv](https://arxiv.org/abs/2302.04867) |
| DPM-Solver-v3 | 2023·NeurIPS | [P] | 引入经验模型统计量（EMS）在线选择最优参数化系数，最小化一阶离散误差，5–10 NFE 进一步提升 | [arXiv](https://arxiv.org/abs/2310.13268) |
| Understanding DDPM Latent Codes Through Optimal Transport (Khrulkov et al.) | 2023·ICLR | [P] | 猜想并数值验证 DDPM 的 PF-ODE 流映射 ≈ Monge OT 映射（高斯情形严格证明）——本子课题与 OT 之间最直接的理论桥梁 | [arXiv](https://arxiv.org/abs/2202.07477) |
| ⭐ AMED-Solver: Fast ODE-based Sampling in Around 5 Steps | 2024·CVPR (Highlight) | [P] | 观察到采样轨迹几乎躺在 2D 子空间 → 由中值定理学习「平均方向」消截断误差，~5 NFE 采样；AMED-Plugin 可插任意求解器（轻量训练，solver 蒸馏边界情形） | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Fast_ODE-based_Sampling_for_Diffusion_Models_in_Around_5_Steps_CVPR_2024_paper.html) · [arXiv](https://arxiv.org/abs/2312.00094) |
| ⭐ On the Trajectory Regularity of ODE-based Diffusion Sampling (GITS) | 2024·ICML (PMLR 235:7905-7934) | [P] | 刻画「隐式去噪轨迹」与采样轨迹的强形状正则性（低维"回旋镖"形，与内容无关）；以局部截断误差为代价做 DP 搜索几何感知时间调度，5–10 NFE 大幅领先 | [PMLR](https://proceedings.mlr.press/v235/chen24bm.html) · [arXiv](https://arxiv.org/abs/2405.11326) |
| ⭐ Align Your Steps (AYS) | 2024·ICML (PMLR 235:42947-42975) | [P] | 首个原理化调度优化框架：用 Girsanov 定理导出真实与线性化生成 SDE 间 KL 上界（KLUB），求 model×solver×dataset 专属最优调度，few-step 域普适增益 | [PMLR](https://proceedings.mlr.press/v235/sabour24a.html) · [arXiv](https://arxiv.org/abs/2404.14507) |
| Accelerating Diffusion Sampling with Optimized Time Steps (DM-NonUniform) | 2024·CVPR | [P] | 把调度选取写成最小化全局离散误差的约束优化（信赖域法，<15 秒求解），与 UniPC/DPM-Solver++ 即插即用 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xue_Accelerating_Diffusion_Sampling_with_Optimized_Time_Steps_CVPR_2024_paper.pdf) · [arXiv](https://arxiv.org/abs/2402.17376) |
| Bespoke Solvers / Bespoke Non-Stationary (BNS) Solvers | 2024·ICLR spotlight / 2024·ICML | [P] | solver 蒸馏路线的奠基：为给定预训练流/扩散模型定制仅 80–200 参数的（非平稳）求解器，证明 NS 族涵盖既有数值格式；16 NFE 达 PSNR 45/FID 1.76（ImageNet-64） | [OpenReview](https://openreview.net/forum?id=1PXEY7ofFX) · [arXiv-BNS](https://arxiv.org/abs/2403.01329) |
| LD3: Learning to Discretize Denoising Diffusion ODEs | 2025·ICLR (Oral) | [P] | 通过学生求解器可微反传端点对齐误差来**学习**离散化，单 GPU 5–10 分钟训练；10 NFE 达 FID 2.38（CIFAR-10），4 NFE 从 35.04 → 9.31 | [OpenReview](https://openreview.net/forum?id=xDrFWUmCne) · [arXiv](https://arxiv.org/abs/2405.15506) |
| PFDiff: Training-Free Acceleration Combining Past and Future Scores | 2025·ICLR | [P] | 完全免训练的跳步策略：复用过去 score 预测「跳板」+ Nesterov 式前瞻更新校正一阶离散误差，正交叠加于现有求解器；DDIM 基线 4 NFE 由 138.81 → 16.46 FID | [OpenReview](https://openreview.net/forum?id=wmmDvZGFK7) · [arXiv](https://arxiv.org/abs/2408.08822) |
| S4S: Solving for a Fast Diffusion Model Solver | 2025·ICML | [P] | 论证低 NFE 下逐点跟踪真 ODE 轨迹在原理上不可行 → 黑盒学习求解器系数（S4S）与离散化（S4S-Alt）以对齐教师端点；5 NFE 达 CIFAR-10 FID 3.73 | [ICML](https://icml.cc/virtual/2025/poster/46229) |

**2025–2026 预印本雷达（补充，均未见官方接收记录）**

| 论文 | 年份·来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| Optimal Stepsize for Diffusion Sampling (OSS) | 2025·arXiv（ICLR 2026 在审） | [R] | 步长调度 = 递归误差最小化的最优子结构 → DP 提取全局最优调度，跨架构/求解器/噪声调度鲁棒；T2I 10× 加速保留 99.4% GenEval | [arXiv](https://arxiv.org/abs/2503.21774) |
| EPD-Solver: Parallel Diffusion Solver via Residual Dirichlet Policy Optimization | 2025·arXiv | [R] | 每步引入多条**并行**梯度方向的加权集成（向量值中值定理），蒸馏 + Dirichlet 策略 RL 学权重，以并行换低延迟压制高曲率截断误差 | [arXiv](https://arxiv.org/abs/2512.22796) |
| Geometric Regularity in Deterministic Sampling（GITS 期刊扩展版） | 2025·arXiv（作者称已被 J. Stat. Mech. 接收） | [R] | KDE 视角给出去噪轨迹闭式解（= 时变带宽 mean-shift），解释逐步旋转、似然单调上升与「线性–非线性–线性」全局模式 | [arXiv](https://arxiv.org/abs/2506.10177) |
| F-scheduler: Free-Lunch Design Space for Fast Sampling | 2025·arXiv | [R] | 免训练插件宣称 6 步采 1024² 图超过蒸馏 SOTA；从信息论角度分析免训练求解器 vs 蒸馏模型的能力边界 | [arXiv](https://arxiv.org/abs/2510.02390) |
| TORS: Analyzing and Improving Fast Sampling of T2I Diffusion Models | 2026·arXiv | [R] | 系统消融 T2I 免训练设计空间，发现**时间调度是最关键因子**；由 Frenet-Serret 公式导出「恒定总旋转」调度，Flux.1/SD3.5 十步高质量 | [arXiv](https://arxiv.org/abs/2603.00763) |
| TJS: x-Prediction Is All You Need（端点可解码性） | 2026·arXiv | [R] | 形式化「端点可解码性」：中间态+路径速度即 E[x₀\|xₜ] 的最优估计 → 提前退出解码，NFE 降 20–70%，且**无需**轨迹变直 | [arXiv](https://arxiv.org/abs/2607.06114) |

正文另涉及（未列表）：PNDM（ICLR 2022，经典线性多步引入者）、gDDIM（ICLR 2023，DDIM 到一般扩散的推广）、GENIE（NeurIPS 2022，蒸馏二阶导的高阶法）、SA-Solver（NeurIPS 2023，随机 Adams SDE 求解器）、Restart Sampling（NeurIPS 2023，ODE+噪声注入组合）、ParaDiGMS（NeurIPS 2023，Picard 迭代并行采样）、DC-Solver（ECCV 2024，动态补偿预测-校正）；理论侧 "The probability flow ODE is provably fast"（NeurIPS 2023, [arXiv](https://arxiv.org/abs/2305.11798)）与 "Accelerating Convergence of Score-Based Diffusion Models, Provably"（ICML 2024, [arXiv](https://arxiv.org/abs/2403.03852)）；OT 反例 Lavenant & Santambrogio, *The flow map of the Fokker–Planck equation does not provide optimal transport*（Applied Mathematics Letters 133:108225, 2022 [P]，[期刊页](https://www.sciencedirect.com/science/article/abs/pii/S089396592200180X)）。

## 3. 方法演进脉络

**2020–2021 一阶时代。** DDIM 把 DDPM 祖先采样确定化，事后被理解为 PF-ODE 的一阶（指数式）离散，奠定「采样 = 解 ODE」范式；PNDM 首次引入经典线性多步法。

**2022 指数积分器与设计空间。** DPM-Solver 与 DEIS 同期独立发现：扩散 ODE 是半线性的——线性漂移可解析解出，只需对神经网络项做 log-SNR 域的多项式近似，由此得到 2–3 阶收敛的定制指数积分器，把 NFE 从数百拉到 10–20。EDM 则从 σ 参数化出发统一训练/采样设计空间（ρ=7 调度、Heun 格式、churn 随机性），成为此后所有 few-NFE 工作的公共基准。GENIE 用蒸馏出的二阶导数走高阶泰勒路线，是「需训练」边界上最早的探索。

**2022–2023 稳定性与统一。** 大 guidance 尺度下 ε-prediction 高阶法失稳甚至劣于 DDIM → DPM-Solver++ 换 data-prediction 参数化并采用多步+thresholding；UniPC 以统一预测-校正框架免费提升精度；DPM-Solver-v3 用经验模型统计（EMS）自动选最优参数化。SDE 侧出现 SA-Solver 与 Restart（确定性 ODE 与噪声注入的组合收缩误差）。理论侧给出 PF-ODE 离散化的多项式收敛保证（NeurIPS 2023）与 DDIM 型高阶加速的可证明性（ICML 2024），确立「低步数误差主要由曲率与调度失配主导」的共识。

**2023–2024 几何转向。** AMED 发现每条采样轨迹几乎躺在 2D 子空间；GITS 系统刻画轨迹形状正则性——隐式去噪轨迹控制方向、曲率沿时间可预测变化、整体呈与内容无关的"回旋镖"形——并把调度选取形式化为 DP。几何视角一举解释了此前手工调度为何有效，标志着研究重心从「更高阶格式」转向「让离散化贴合轨迹几何」。

**2024–2025 调度优化成为主战场。** 五条独立的原理化路线取代手工调度：AYS（KLUB 上界最小化）、DM-NonUniform（全局离散误差的信赖域优化）、GITS（截断误差 DP）、LD3（可微反传端点误差，ICLR 2025 Oral）、OSS（最优子结构 DP）。同时涌现正交的免训练插件（PFDiff 跳步、DC-Solver 动态补偿）。**solver 蒸馏边界**上，Bespoke → BNS → S4S → EPD 逐步证明：用 <200 个参数、分钟级优化即可对齐教师端点分布，几乎闭合与 Progressive Distillation 的差距——代价是不再「零训练」，且 S4S 论证了低 NFE 下逐点跟踪真轨迹在原理上不可行，只能追求端点/分布层面的对齐。

**2026 前沿。** TORS 用 Frenet-Serret 标架证实「调度 > 求解器阶数」的重要性层级并给出恒定总旋转调度；TJS 提出端点可解码性——无需拉直轨迹即可提前退出解码 E[x₀|xₜ]；F-scheduler 等开始讨论免训练与蒸馏之间的信息论边界。

## 4. 与博客两个方向的关联

- **方向一（无须重训的轨迹对齐）：强关联，本子课题即其数值基座。** (a) 轨迹几何事实——低维性（AMED 的 2D 子空间）、形状正则性与曲率集中（GITS/TORS）——说明预训练模型的传输路径离直线只差一个低维、跨样本一致、可预测的弯曲，为「事后拉直/对齐」提供了可行性依据与操作量（曲率、总旋转、局部截断误差都是现成的对齐目标函数）。(b) 与 OT 的量化桥：Khrulkov 等猜想 PF-ODE 流映射 ≈ Monge OT 映射（高斯情形严格成立），Lavenant–Santambrogio 构造反例证明一般**不精确**成立、但数值上「几乎最优」——因此 DDIM/EDM 轨迹可视为「近 OT」路径，轨迹曲率即「与 OT 位移插值的偏差」的可计算代理。(c) 调度优化（AYS/GITS/LD3/OSS）可解读为在固定边缘分布路径下重分配离散化预算的轨迹对齐；solver 蒸馏（BNS/S4S/EPD）则直接对齐采样映射本身。与 T09（rectified flow 重训拉直）恰好互补：那边改模型，这边只改推理。
- **方向二（OT 引导跨域生成）：间接相关，但是必要基础设施。** 确定性 ODE 映射（DDIM inversion）提供跨域编辑的 encode–decode 骨架；OT 引导推理的实用性直接受 few-NFE 求解器预算约束（每步含 OT 求解时必须压缩 NFE）。此外 TJS 的端点解码 E[x₀|xₜ] 与熵 OT 的 barycentric 投影在形式上同构，是两个方向的潜在交叉点。

## 5. 开放问题与可发论文的切入点

1. **曲率—最优调度的第一性理论（证定理）**：GITS/TORS 的经验规律（曲率集中于中段、总旋转近似恒定）缺乏推导。可在 KDE/高斯混合数据假设下（利用 GITS 期刊版的闭式去噪轨迹）推导 PF-ODE 轨迹曲率的解析式，证明「等旋转/等截断误差分配 = W₂ 意义下最优调度」的定理，并把 AYS 的 KLUB（KL 泛函）换成 Benamou–Brenier 动能泛函，得到 OT-aware 调度，在 5 NFE 与 AYS/GITS/LD3 对比。
2. **「距 OT 的偏差」作为样本级加速上限（做实验+小模块）**：由 Lavenant–Santambrogio 知 PF-ODE 几乎最优但不精确。定义 per-sample 传输次优度指标（轨迹弧长 ÷ ‖x_T−x₀‖ 或局部曲率积分），验证其与「该样本达到目标质量所需最小 NFE」的相关性，进而做 sample-adaptive NFE 分配器——免训练、即插即用，天然的 workshop→顶会递进题目。
3. **免训练 vs solver 蒸馏的信息论下界（证定理）**：S4S 论证低 NFE 逐点跟踪不可行、F-scheduler 有初步信息论讨论，但「给定 NFE 预算下 training-free 求解器可达的最小 W₂/FID 差距」下界从未被形式化。可类比数值分析 Peano 核定理 + score 场 Lipschitz 常数给出下界，并刻画蒸馏突破该下界所需的最小参数量。
4. **端点可解码性 × OT barycentric 投影（改模块）**：把 TJS 的 early-exit 解码（模型自身 posterior mean）替换为基于小批量 Sinkhorn 耦合的 barycentric 投影解码，检验在跨域/条件生成中能否以更少 NFE 保持语义对齐（与 T12 衔接）。
5. **最优调度的跨分布迁移律（做实验+理论）**：AYS/LD3/OSS 的调度均为 model×dataset 专属。用 OT 距离刻画数据集偏移量与最优调度漂移之间的 Lipschitz 型关系，产出「零成本调度迁移」规则（例如从 ImageNet 调度外推到医学影像域）。

## 6. 代码与资源

- **diff-sampler 工具箱**（AMED + GITS 官方，含各求解器统一实现与 FID 参考统计）: https://github.com/zju-pi/diff-sampler
- DPM-Solver/++ 官方: https://github.com/LuChengTHU/dpm-solver ；UniPC 官方: https://github.com/wl-zhao/UniPC
- EDM 官方（模型权重是 few-NFE 研究的标准 checkpoint）: https://github.com/NVlabs/edm
- AYS 项目页（含各模型最优调度数值，已并入 diffusers）: https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/
- LD3 官方: https://github.com/vinhsuhi/LD3 ；PFDiff 官方: https://github.com/onefly123/PFDiff
- DM-NonUniform（CVPR 2024 调度优化）: https://github.com/scxue/DM-NonUniform ；OSS 官方: https://github.com/bebebe666/OptimalSteps
- k-diffusion（EDM 式采样器的事实标准第三方实现）: https://github.com/crowsonkb/k-diffusion
- HuggingFace diffusers schedulers（DDIM/DPM++/UniPC/DEIS/AYS 均内置）: https://github.com/huggingface/diffusers
- 常用 benchmark：CIFAR-10 / FFHQ-64 / AFHQv2-64 / ImageNet-64（EDM checkpoints）、LSUN Bedroom、MS-COCO（SD 文生图 FID）、GenEval（T2I 语义对齐）。

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2020_Song_DDIM.pdf | Denoising Diffusion Implicit Models | 成功（10.9MB，%PDF 校验通过） |
| 2022_Karras_EDM_Design_Space.pdf | Elucidating the Design Space of Diffusion-Based Generative Models | 成功（19.2MB，%PDF 校验通过） |
| 2022_Lu_DPM_Solver.pdf | DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps | 成功（15.9MB，%PDF 校验通过） |
| 2023_Zhao_UniPC.pdf | UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models | 成功（15.3MB，%PDF 校验通过） |
| 2023_Khrulkov_DDPM_Latent_OT.pdf | Understanding DDPM Latent Codes Through Optimal Transport | 成功（10.4MB，%PDF 校验通过） |
| 2024_Zhou_AMED_Solver.pdf | Fast ODE-based Sampling for Diffusion Models in Around 5 Steps | 成功（18.1MB，%PDF 校验通过） |
| 2024_Chen_Trajectory_Regularity_GITS.pdf | On the Trajectory Regularity of ODE-based Diffusion Sampling | 成功（25.3MB，%PDF 校验通过） |
| 2024_Sabour_Align_Your_Steps.pdf | Align Your Steps: Optimizing Sampling Schedules in Diffusion Models | 成功（10.1MB，%PDF 校验通过） |
