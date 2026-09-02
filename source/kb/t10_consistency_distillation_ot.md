# T10 一致性模型与少步蒸馏的 OT 视角

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题覆盖扩散加速中的"少步/一步生成"谱系（一致性模型、轨迹一致性、分布匹配蒸馏、对抗蒸馏、score distillation），专挖 OT/Wasserstein 在其中的三重角色——理论误差度量（W2 上界与统计率）、蒸馏目标/正则（熵 OT 距离、WGAN 对偶）、噪声-数据耦合设计（ODE 对回归、generator-induced coupling）。与 T09（rectified flow 直线化重训）和 T11（免训练快速采样器）互补，共同构成"扩散 × OT 加速"版图。

## 1. 核心问题与背景

扩散/流模型采样需数十到数百次网络评估（NFE），少步蒸馏的目标是把 PF-ODE 的解算子（两时间 flow map）压缩进单个网络，实现 1-4 步生成。OT 视角在该方向自然出现于三个层面：(a) **理论层**——蒸馏/一致性损失如何控制学生与教师（或数据）分布之间的 Wasserstein 距离：FMM 证明 Lagrangian/Eulerian 蒸馏损失给出 W2 上界，Dou 与 Li 分别给出 Wasserstein 统计估计率与离散化步数下界；(b) **目标层**——直接以 OT/W 距离替代或增强 KL 蒸馏目标：VDOT 用熵正则 OT 距离约束 DMD 缓解 zero-forcing 与梯度崩塌，ASD 把 score distillation 完整解读为 WGAN（W1 对偶）问题，ADD/CTM/DMD2 的对抗项本质是 IPM 判别器；(c) **耦合层**——噪声-数据配对的选取决定回归目标方差与传输成本：DMD 用教师 ODE 的确定性耦合做回归锚定，Issenhuth 证明 generator 诱导的耦合能同时降低一致性训练偏差与噪声-数据传输成本。核心度量是质量-步数（FID-NFE）权衡曲线。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Consistency Models | 2023 · ICML (PMLR v202) | [P] | 奠基：学习 PF-ODE 的自一致映射（任意轨迹点→端点），CD 蒸馏 CIFAR-10 一步 FID 3.55、两步 2.93 | [PMLR](https://proceedings.mlr.press/v202/song23a.html) |
| Improved Techniques for Training Consistency Models (iCT) | 2024 · ICLR | [P] | 去 EMA teacher + Pseudo-Huber 损失，免蒸馏一致性训练一步 FID 2.51 (CIFAR-10)，首次反超蒸馏 | [OpenReview](https://openreview.net/forum?id=WNzy9bRDvG) |
| Consistency Trajectory Models (CTM) | 2024 · ICLR | [P] | 推广为任意时刻→任意时刻的两时间轨迹映射，可同时输出 score；GAN 辅助下一步 FID 1.73 (CIFAR-10)/1.92 (ImageNet-64)，γ-sampling 提供质量-步数连续旋钮 | [OpenReview](https://openreview.net/forum?id=ymjI8feDTD) |
| ⭐ One-step Diffusion with Distribution Matching Distillation (DMD) | 2024 · CVPR | [P] | 反向 KL 梯度 = 真/假 score 之差；用教师 ODE 噪声-图像对的回归损失（确定性耦合锚定）防模式坍缩 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html) |
| DMD2: Improved Distribution Matching Distillation | 2024 · NeurIPS (Oral) | [P] | 去掉 ODE 对回归，TTUR + GAN（真数据）稳定纯分布匹配；一步 ImageNet-64 FID 1.28 超越教师 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html) |
| Adversarial Diffusion Distillation (ADD) | 2024 · ECCV (Oral) | [P] | score distillation + hinge-GAN 判别器，SDXL-Turbo 的基础，1-4 步实时生成、单步胜过 LCM | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11557_ECCV_2024_paper.php) |
| Score identity Distillation (SiD) | 2024 · ICML (PMLR v235) | [P] | 三个 score 恒等式构造 data-free 蒸馏损失，FID 指数速率下降、逼近甚至超过教师 | [PMLR](https://proceedings.mlr.press/v235/zhou24x.html) |
| Adversarial Score Distillation (ASD) | 2024 · CVPR | [P] | 用 WGAN 范式重推 SDS/VSD：SDS=固定次优判别器、VSD=不完整判别器优化；补全 W1 对偶判别器训练解决 CFG 尺度敏感 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wei_Adversarial_Score_Distillation_When_score_distillation_meets_GAN_CVPR_2024_paper.html) |
| Theory of Consistency Diffusion Models | 2024 · ICML (PMLR v235) | [P] | 首个 CM 统计理论：把训练形式化为分布差异最小化，给出 Wasserstein 距离下的估计率（与原扩散模型同阶），同时覆盖蒸馏与免蒸馏两种训练 | [PMLR](https://proceedings.mlr.press/v235/dou24a.html) |
| Towards a Mathematical Theory for Consistency Training | 2025 · AISTATS (PMLR v258) | [P] | 证明一致性训练步数超过 O(d^{5/2}/ε) 即可生成 Wasserstein 意义下 ε-接近目标的样本，给出离散化-精度定量关系 | [PMLR](https://proceedings.mlr.press/v258/li25c.html) |
| ⭐ Flow Map Matching (FMM) | 2025 · TMLR | [P] | 两时间 flow map 统一框架：证明 Lagrangian/Eulerian 蒸馏损失上界控制教师-学生 W2 距离，且 Eulerian 损失是一致性蒸馏的连续时间极限，统一 CM/CTM/渐进蒸馏 | [OpenReview](https://openreview.net/forum?id=cqDH0e6ak2) |
| ⭐ Improving Consistency Models with Generator-Augmented Flows | 2025 · ICML (PMLR v267) | [P] | 证明一致性训练与蒸馏的差异在连续时间极限仍不消失；用 generator 诱导的流/耦合同时降低该差异与噪声-数据传输成本，加速收敛并提升质量 | [PMLR](https://proceedings.mlr.press/v267/issenhuth25a.html) |
| Simplifying, Stabilizing and Scaling Continuous-time CMs (sCM) | 2025 · ICLR (Oral) | [P] | TrigFlow 统一参数化 + 连续时间训练稳定化，1.5B 参数两步 FID：CIFAR-10 2.06 / ImageNet-64 1.48 / ImageNet-512 1.88 | [ICLR](https://iclr.cc/virtual/2025/oral/31868) |
| One Step Diffusion via Shortcut Models | 2025 · ICLR | [P] | 把步长 d 作为网络条件输入，自蒸馏只需 log2(T) 次 bootstrap；单网络单阶段支持任意步数预算（ImageNet-256 DiT-XL：1 步 10.6 / 4 步 7.8 / 128 步 3.8） | [ICLR](https://iclr.cc/virtual/2025/poster/29802) |
| Mean Flows for One-step Generative Modeling (MeanFlow) | 2025 · NeurIPS (Oral) | [P] | 平均速度-瞬时速度恒等式指导训练，从零训练 1-NFE ImageNet-256 FID 3.43，当前 from-scratch 一步 SOTA 系 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html) |
| ⭐ VDOT: Efficient Unified Video Creation via Optimal Transport Distillation | 2026 · CVPR | [P] | 首次把熵正则 OT 距离引入 DMD（替代/增强 KL）：OT plan 给分布匹配加几何约束，缓解 few-step 场景的 zero-forcing 与梯度崩塌；4 步统一视频生成匹敌 50-100 步教师，并发布 UVCBench | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_VDOT_Efficient_Unified_Video_Creation_via_Optimal_Transport_Distillation_CVPR_2026_paper.html) |

正文补充（未入表）：Latent Consistency Models（[R] [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)）、Consistency Models Made Easy/ECT（[R] [arXiv:2406.14548](https://arxiv.org/abs/2406.14548)）、Diff-Instruct（NeurIPS 2023, [arXiv:2305.18455](https://arxiv.org/abs/2305.18455)）、f-distill（[R] [arXiv:2502.15681](https://arxiv.org/abs/2502.15681)）、SiDA（[R] [arXiv:2410.14919](https://arxiv.org/abs/2410.14919)）、ADM/DMDX（[R] [arXiv:2507.18569](https://arxiv.org/abs/2507.18569)）、Inductive Moment Matching/MMSD（[P] [PMLR v267](https://proceedings.mlr.press/v267/zhou25c.html)）、How to Build a Consistency Model（[A] [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/119201)）、Understanding/Accelerating/Improving MeanFlow Training（[P] [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Understanding_Accelerating_and_Improving_MeanFlow_Training_CVPR_2026_paper.pdf)）。

## 3. 方法演进脉络

**一致性线（轨迹压缩）**：Progressive Distillation（Salimans & Ho, ICLR 2022）逐次减半步数但需多阶段训练。CM（ICML 2023）改为直接学习 PF-ODE 的自一致映射，一步生成成为可能；iCT（ICLR 2024）去掉 EMA teacher、换 Pseudo-Huber 损失，使免蒸馏训练（CT）反超蒸馏（CD）；CTM（ICLR 2024）把"任意点→端点"推广为"任意时刻→任意时刻"的两时间映射并结合 GAN，一步 FID 进入 1.7-1.9 区间，其 γ-sampling 首次把质量-步数权衡变成连续旋钮；LCM 把 CD 搬进 Stable Diffusion 潜空间成为工程标配。之后的重点是稳定化与规模化：ECT 证明 CM 可从预训练扩散模型热启动微调；sCM（ICLR 2025 oral）以 TrigFlow 统一参数化，把连续时间 CM 训到 1.5B 参数、两步逼近最优扩散模型（差距 <10%）。与之平行，shortcut models（ICLR 2025）把步长作为条件输入、仅需 log2(T) 次 bootstrap；MeanFlow（NeurIPS 2025 oral）用平均速度恒等式从零训练一步模型（3.43@ImageNet-256，CVPR 2026 后续改进到 2.87）；IMM/MMSD（ICML 2025）用矩匹配（MMD，IPM 家族、W1 的近亲）获得分布级收敛保证。FMM（TMLR 2025）在理论上统一了这一整条线：CM/CTM/PD 都是两时间 flow map 的不同学习方案，其 Lagrangian/Eulerian 损失均控制教师-学生 W2；NeurIPS 2025 的后续把所有方案组织为 Eulerian/Lagrangian/Progressive 三族自蒸馏。

**分布匹配线（score distillation 系）**：SDS（DreamFusion）→ Diff-Instruct（积分 KL）→ DMD（CVPR 2024）：反向 KL 梯度等于真/假 score 之差，另用教师 ODE 的噪声-图像对做回归锚定——这本质上是一个"确定性耦合"的逐点传输约束；DMD2（NeurIPS 2024 oral）用 TTUR+GAN 去掉昂贵的耦合数据集并超越教师；SiD（ICML 2024）以 score 恒等式实现 data-free 蒸馏；f-distill 把这一族统一到一般 f-散度并指出 reverse-KL 的 mode-seeking 缺陷；**VDOT（CVPR 2026）走出散度家族，首次把熵正则 OT 距离引入 DMD**：OT plan 显式给出"每个假样本应被运往哪个真样本"，为优化方向加上几何护栏，缓解 few-step 场景 KL 的 zero-forcing 与梯度崩塌。

**对抗线（IPM/W1 对偶）**：ADD（ECCV 2024 oral）用 hinge-GAN + score distillation 造就 SDXL-Turbo；ASD（CVPR 2024）从 WGAN 完整推导出 SDS/VSD——二者分别对应固定次优判别器与不完整判别器优化，补全 W1 对偶的判别器更新后统一并改进了它们；SiDA、ADM/DMDX 继续把对抗项与 score 蒸馏混合（ADM 还把 DMD2 的 ODE 对 MSE 换成分布级损失）。这条线的"判别器"在数学上就是 IPM（含 W1）的对偶测试函数，是 OT 对偶结构在蒸馏中的隐式体现。

## 4. 与博客两个方向的关联

- **方向一（无须重训的轨迹对齐）**: 本子课题整体属于"重训/蒸馏"阵营，是方向一天然的成本上界与质量参照系（对齐 vs 蒸馏的 FID-NFE 对比基线）。更深的联系有三：(i) FMM 说明轨迹压缩的数学对象是两时间 flow map——免重训对齐可视为对该映射的 zero-shot 近似，其 W2 误差界可直接套用；(ii) CTM 的 anytime-to-anytime 跳跃与 γ-sampling 为对齐后的轨迹提供任意步数预算的解码接口；(iii) Issenhuth 证明"不动教师、只改噪声-数据耦合"即可重塑训练目标并降低传输成本——同样的耦合替换思想可以在免重训场景中离线重新配对 (noise, data) 实现轨迹对齐。
- **方向二（OT 引导跨域生成）**: 关联直接。VDOT 的熵 OT-DMD 本质是在两个生成分布之间做 OT 引导的分布对齐，把"教师分布→学生分布"替换为"源域→目标域"即得跨域蒸馏配方；ASD 的 WGAN 视角说明 score distillation 与 W1 对偶引导共享同一骨架（用 LoRA/textual-inversion 实现的可优化判别器即对偶势函数），可直接复用于 OT 引导的域间映射；DMD 的 ODE 对回归等价于"预计算的确定性耦合"，与跨域 OT 配对的构造同构。

## 5. 开放问题与可发论文的切入点

1. **OT-DMD 的图像/3D 版与理论**：把 VDOT 的熵 OT 项移植到图像蒸馏（SDXL/EDM2）与 text-to-3D score distillation；系统消融 Sinkhorn ε、minibatch 大小（minibatch-OT 偏差）、OT/KL/GAN 混合权重；理论上证明 few-step regime 下 OT 梯度非退化（对照 reverse-KL 的 zero-forcing），目标是一步 FID 与 mode coverage（recall/precision）同时改善。
2. **OT 耦合式一致性训练**：在 CM/shortcut/MeanFlow 的 bootstrap 目标中用 minibatch Sinkhorn 耦合替换独立耦合（把 OT-CFM 的思想搬进一致性训练，注意与 T09 的 rectified flow 重训路线区分），并与 Issenhuth 的 generator-induced coupling 正面对比；理论问题：更低曲率的耦合能否把 Li 等人的 O(d^{5/2}/ε) 离散化步数上界中的维度依赖降下来？
3. **W2 正则蒸馏 / 传输成本约束**：在 CD/CTM/DMD 损失上加 λ·E‖f_θ(x_t,t)−x_t‖² 型传输成本惩罚（或对输出-噪声对做 W2 约束），研究蒸馏得到的一步映射与 Monge 最优映射的偏差——PF-ODE 映射一般并非 OT 映射（高斯情形除外，见 Khrulkov et al., [arXiv:2202.07477](https://arxiv.org/abs/2202.07477)；一般反例由 Lavenant & Santambrogio 给出，理论细节归口理论子课题），但蒸馏的自由度允许"在同分布约束下选更直的映射"；预期副产物：更平滑的 latent 插值与更稳的少步外推。
4. **teacher-anchored Wasserstein 评测协议**：DMD2/SiDA/ADD 借 GAN+真数据"超越教师"，导致 FID 无法区分"忠实蒸馏"与"分布漂移"。建立以教师样本云为参照的 sliced-W2/W2 指标与 FID-NFE 帕累托基准（锚点：CD 3.55→CTM 1.73→sCM 两步 1.88@512→DMD2 一步 1.28@64→MeanFlow 从零 3.43@256），检验 W2(teacher, student) 随 NFE 是否呈幂律，为"多少步才够"给出预算公式。
5. **蒸馏动力学的 Wasserstein 梯度流解释**：把 DMD/Diff-Instruct 的生成器更新形式化为分布空间中某泛函的 Wasserstein 梯度流（JKO 格式），把 GAN 项解释为 W1 对偶修正，给出"蒸馏损失何时是 W2 测地凸"的充分条件，从而得到 few-step 蒸馏的收敛率定理——目前该族方法完全没有收敛性刻画。

## 6. 代码与资源

- CM 官方: https://github.com/openai/consistency_models ；CTM 官方: https://github.com/sony/ctm
- LCM: https://github.com/luosiallen/latent-consistency-model ；DMD2: https://github.com/tianweiy/DMD2
- SiD: https://github.com/mingyuanzhou/SiD ；shortcut models: https://github.com/kvfrans/shortcut-models
- MeanFlow: https://github.com/gsunshine/meanflow ；FMM/flow-maps（含三族自蒸馏算法）: https://github.com/nmboffi/flow-maps
- Generator-augmented flows: https://github.com/thibautissenhuth/consistency_GC ；ASD: https://github.com/2y7c3/ASD
- VDOT（含 14B 模型权重）: https://github.com/hhhh1138/VDOT ；UVCBench 统一视频创作基准: https://huggingface.co/datasets/yutongwang1012/UVCBench
- 常用基准：CIFAR-10 / ImageNet-64/256/512 的 FID-NFE 曲线；zero-shot COCO FID（文生图蒸馏）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2023_Song_Consistency_Models.pdf | Consistency Models | 成功 |
| 2024_Kim_Consistency_Trajectory_Models.pdf | Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion | 成功 |
| 2024_Yin_Distribution_Matching_Distillation.pdf | One-step Diffusion with Distribution Matching Distillation | 成功 |
| 2024_Dou_Theory_Consistency_Diffusion_Models.pdf | Theory of Consistency Diffusion Models: Distribution Estimation Meets Fast Sampling | 成功 |
| 2024_Boffi_Flow_Map_Matching.pdf | Flow Map Matching with Stochastic Interpolants: A Mathematical Framework for Consistency Models | 成功 |
| 2025_Issenhuth_Generator_Augmented_Flows.pdf | Improving Consistency Models with Generator-Augmented Flows | 成功 |
| 2024_Wei_Adversarial_Score_Distillation.pdf | Adversarial Score Distillation: When Score Distillation Meets GAN | 成功 |
| 2026_Wang_VDOT_Optimal_Transport_Distillation.pdf | VDOT: Efficient Unified Video Creation via Optimal Transport Distillation | 成功 |
