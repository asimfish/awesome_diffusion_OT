# T26 Gromov-Wasserstein 与跨空间生成对齐

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: GW 把 OT 从「同一空间内搬运质量」推广到「跨空间保结构对齐」，是扩散×OT 全景中处理不同维度、不同模态、无共同坐标系的生成对齐问题的核心工具。对内衔接熵正则/低秩计算（T04），对外支撑跨模态生成翻译（博客方向二）与图/分子等结构化数据生成；黎曼流形上的 OT 归 T28。

## 1. 核心问题与背景

GW 距离在 metric measure space 之间比较「内部距离结构」：耦合 π 最小化 pairwise 畸变 Σ|d_X(x,x')−d_Y(y,y')|²dπdπ，因此无需两空间共享坐标系即可对齐——这正是跨模态/跨维度生成对齐（图↔点云、基因表达↔染色质、语言↔视觉↔动作）的天然数学语言。三大瓶颈长期制约其落地：(i) 非凸二次指派问题（QAP），NP-hard 且局部极小编码了匹配对称性；(ii) 朴素求解 O(n³)~O(n⁴)、内存 O(n²)，难以到达生成模型所需规模；(iii) 对偶理论与统计样本复杂度直到 2024 年才补齐。2024–2026 年三条线并进：计算上出现低秩、CNT lifted、SDP 认证、神经连续求解器；理论上 Annals of Statistics 2024 给出对偶与尖锐收敛率；应用上 GENOT 把 entropic (F)GW 耦合装进流匹配骨架，moscot 把低秩 FGW 推到 170 万细胞 atlas，CVPR/ICML 一线把 GW 结构对齐引入视觉-语言盲匹配与 VLA 动作空间设计，图生成侧开始把 FGW 当作训练目标与理论保证工具。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ GENOT: Entropic (Gromov) Wasserstein Flow Matching with Applications to Single-Cell Genomics (Klein, Uscidda, Theis, Cuturi) | 2024·NeurIPS | [P] | 用条件流匹配参数化任意代价的 entropic (F)GW（含 unbalanced）耦合的条件分布，得到可采样的随机跨空间映射，实现单细胞跨模态翻译——「GW 流匹配」范式的确立之作 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc46e29f91e676747c584ca181cb0ea1-Abstract-Conference.html) |
| ⭐ Gromov-Wasserstein at Scale, Beyond Squared Norms (Houry, Feydy, Vialard) | 2026·ICML（种子库 [A]） | [A] | 识别出条件负定型（CNT）畸变代价大类，使 GW 化为 lifted 特征空间线性对齐 + 标准平方欧氏 OT：线性内存、二次（而非三次）时间、可微、可探索能量景观的 EGW solver，数十万点分钟级 | [arXiv](https://arxiv.org/abs/2602.06658) |
| ⭐ Gromov-Wasserstein Distances: Entropic Regularization, Duality and Sample Complexity (Zhang, Goldfeld, Mroueh, Sriperumbudur) | 2024·Annals of Statistics 52(4) | [P] | 通过辅助矩阵变量把二次 GW 线性化为 OT/EOT 族的下确界，建立首个对偶理论与尖锐经验收敛率：GW 为 n^{−2/max{min(dx,dy),4}}，EGW 达参数率 n^{−1/2} | [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-52/issue-4/GromovWasserstein-distances-Entropic-regularization-duality-and-sample-complexity/10.1214/24-AOS2406.full) |
| ⭐ It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data (Schnaus et al.) | 2025·CVPR | [P] | 把「无任何平行数据的视觉-语言匹配」形式化为 GW 型 QAP，改进 Hahn-Grant 对偶求解器，实证 platonic representation hypothesis 下基础模型嵌入可被无监督结构对齐 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Schnaus_Its_a_Blind_Match_Towards_Vision-Language_Correspondence_without_Parallel_Data_CVPR_2025_paper.pdf) |
| ⭐ LAST: Bridging Vision-Language and Action Manifolds via Gromov-Wasserstein Alignment (Lyu et al.) | 2026·ICML | [A] | 把 VLA 学习表述为 GW 对齐问题：Lie 代数 tokenizer 全局线性化动作流形 + 白化局部度量离散化，使动作空间的关系几何与 VL 语义嵌入统计兼容 | [ICML 官方页](https://icml.cc/virtual/2026/poster/62473) |
| Semidefinite Relaxations of the Gromov-Wasserstein Distance (Chen, Nguyen, Koh, Soh) | 2024·NeurIPS | [P] | GW 的 SDP 松弛给出可认证的全局下界与最优性证书，是非凸 GW「可认证计算」路线的代表 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8189d86a5d8dea0694d43bb90e01c14d-Abstract-Conference.html) |
| Linear-Time Gromov Wasserstein Distances using Low Rank Couplings and Costs (Scetbon, Peyré, Cuturi) | 2022·ICML | [P] | 低秩耦合 + 低秩代价分解把 GW 降至线性时间，是 moscot/OTT-JAX 规模化 GW 的算法基石 | [PMLR](https://proceedings.mlr.press/v162/scetbon22b.html) |
| Gromov-Wasserstein Averaging of Kernel and Distance Matrices (Peyré, Cuturi, Solomon) | 2016·ICML | [P] | entropic GW 的投影镜像下降求解器与 GW barycenter，现代计算 GW 的起点 | [PMLR](https://proceedings.mlr.press/v48/peyre16.html) |
| Optimal Transport for Structured Data with Application on Graphs（Fused GW）(Vayer/Titouan et al.) | 2019·ICML | [P] | FGW：特征项（Wasserstein）与结构项（GW）凸组合的联合传输，图比较、barycenter 与结构化生成的标准工具 | [PMLR](https://proceedings.mlr.press/v97/titouan19a.html) |
| Gromov-Wasserstein Autoencoders (Nakagawa, Togo, Ogawa, Haseyama) | 2023·ICLR | [P] | 抛弃似然目标，直接用 GW 度量匹配（不同维度的）latent 与 data 分布，把 meta-prior 表征学习变成跨空间结构匹配 | [OpenReview](https://openreview.net/forum?id=sbS10BCtc7) |
| Gromov-Wasserstein Alignment of Word Embedding Spaces (Alvarez-Melis, Jaakkola) | 2018·EMNLP | [P] | 无平行语料的跨语言词嵌入 GW 对齐，「嵌入空间结构对齐」整条线的奠基 | [ACL Anthology](https://aclanthology.org/D18-1214/) |
| Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild (Im et al.) | 2026·CVPR | [P] | 用 3D 结构先验 + anchor 线性化缓解 FGW 计算成本，做野外语义对应；展示 FGW 在视觉对应任务的工程化路径 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Im_Shape-of-You_Fused_Gromov-Wasserstein_Optimal_Transport_for_Semantic_Correspondence_in-the-Wild_CVPR_2026_paper.html) |
| Mapping Cells Through Time and Space with moscot (Klein et al.) | 2025·Nature | [P] | 基于 OTT-JAX 低秩 (F)GW 的统一框架，把 GW 型对齐推到 170 万细胞、20 个时间点的 atlas 规模（时空/跨模态映射），大规模 GW 落地的旗舰应用 | [Nature](https://www.nature.com/articles/s41586-024-08453-2) |
| Private Synthetic Graph Generation and Fused Gromov-Wasserstein Distance (Wirth, Aminian, Reinert) | 2026·AISTATS | [A] | 顶点级 ε-DP 属性图生成器，并用 FGW 距离给出生成分布与真实分布的精度理论保证——FGW 作为图生成「度量+证明工具」 | [OpenReview](https://openreview.net/forum?id=g5QpPIwSst) |
| MIRROR: Aligning Semantic Relations from Language to Image via Gromov-Wasserstein (Wang, Wang, Ding) | 2026·arXiv（自称 ECCV 2026 接收，待论文集核验） | [R] | 用 GW 型正则强制「概念间关系结构」在语言→视觉投影中保持，修复 MLLM 的关系推理盲区 | [arXiv](https://arxiv.org/abs/2606.29462) |

## 3. 方法演进脉络

**计算线：从 entropic 到「可扩展 + 可认证」两翼。** Peyré–Cuturi–Solomon (ICML 2016) 的 entropic GW 镜像下降确立了基本范式，Xu et al. (2019) 的递归划分把 GW 用于大图匹配；Scetbon et al. (ICML 2022) 用低秩耦合/代价实现线性时间，被 OTT-JAX 与 moscot (Nature 2025) 吸收成为 atlas 规模应用的引擎。2024 年后分出两翼：一翼追求**认证性**——NeurIPS 2024 的 SDP 松弛给出全局下界与最优性证书；另一翼追求**规模与可微性**——ICML 2026 的 GW-at-Scale 证明 CNT 代价类下 GW 分解为「lifted 特征线性映射 + 平方欧氏 OT」，得到线性内存、可微、能刻画能量景观对称性的 solver。切片方向持续供给廉价近似（sliced inner-product GW，[arXiv 2605.08546](https://arxiv.org/abs/2605.08546) [R]；min generalized sliced GW，[arXiv 2605.13753](https://arxiv.org/abs/2605.13753) [R]）。神经连续求解器仍不成熟：Korotin 组的 benchmark（NeuralGW，[arXiv 2303.05978](https://arxiv.org/abs/2303.05978) [R]）实测表明现有连续 GWOT 求解器严重依赖离散技巧、可靠性欠缺。

**理论线：** Zhang–Goldfeld 等 (AoS 2024) 的对偶化（GW = OT 族下确界）与样本复杂度补上了统计地基；2026 年延伸到偶数阶 GW 泛函的经验收敛（[arXiv 2605.11108](https://arxiv.org/abs/2605.11108) [R]）与 GW 量化/聚类的率（[arXiv 2608.11016](https://arxiv.org/abs/2608.11016) [R]）。unbalanced/partial 松弛（Séjourné et al. 2021 conic UGW，[arXiv 2009.04266](https://arxiv.org/abs/2009.04266)；Chapel et al. 2020 partial GW，[arXiv 2002.08276](https://arxiv.org/abs/2002.08276)）在 2025 年被整合进 fused 场景（FPGW，[arXiv 2502.09934](https://arxiv.org/abs/2502.09934) [R]）。

**生成对齐线（本课题主轴）：** Alvarez-Melis–Jaakkola (EMNLP 2018) 证明嵌入空间可被 GW 无监督对齐 → GWAE (ICLR 2023) 把 GW 变成生成模型的训练目标（latent↔data 跨维匹配）→ GENOT (NeurIPS 2024) 完成关键一跃：不再离散求解后做 barycentric 投影，而是用流匹配直接参数化 entropic (F)GW 耦合的条件分布，得到可外推、可采样的跨空间随机映射；Wasserstein Flow Matching（ICML 2025 [P]，[PMLR](https://proceedings.mlr.press/v267/haviv25a.html)）进一步把 FM 抬到分布族空间，与 GENOT 互补。对齐应用端，Blind Match (CVPR 2025) 与 GNN 的 GW 迁移对齐（NeurIPS 2025 [P]，[proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/30dee0ddc6c0b4907a9d587d3ce87979-Abstract-Conference.html)）说明预训练表征「结构上可对齐」；LAST (ICML 2026) 与 MIRROR ([R]) 把 GW 视角前置到模型设计——不再事后求耦合，而是重构动作/视觉空间使其与语义空间度量兼容；JK-EGW（[arXiv 2608.04234](https://arxiv.org/abs/2608.04234) [R]）用联合核把多模态一次性对齐进共享空间。

**图生成线：** FGW (ICML 2019) 提供度量后，FGW barycenter 混合进入表征与增广（AAAI 2025 OT Latent Mixer [P]，[AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/33849)）；AISTATS 2026 用 FGW 为差分隐私图生成器提供精度保证；SGW-GAN（[arXiv 2601.13417](https://arxiv.org/abs/2601.13417) [R]) 把 sliced GW 作为 GAN 的几何保持引导；相邻地，Bures-Wasserstein 流匹配图生成（[arXiv 2506.14020](https://arxiv.org/abs/2506.14020) [R]）说明「OT 位移插值 + 图」已成活跃方向，但 GW 版的图扩散/流匹配训练目标仍是空位。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **中度相关（工具层）**。GW 是「两个已训练模型的表示/轨迹能否不重训就对齐」的判定与求解工具：Blind Match 表明成熟基础模型的嵌入仅凭内部距离结构即可无监督匹配（platonic 假说），GW-at-Scale 的可微、线性内存 solver 使这种对齐能在推理期在线完成。可设想用 EGW 耦合把两个扩散模型在各自潜空间的采样轨迹逐时刻结构对齐（跨维 stitching），完全无须重训——但目前尚无论文直接做「GW × 扩散轨迹对齐」，属空白。
- 方向二（OT 引导跨域生成）: **直接核心**。GENOT 就是该方向在「跨空间」情形的范式：entropic (F)GW 耦合作为教师信号引导流匹配生成器，实现模态 A→模态 B 的翻译；LAST/MIRROR 展示 GW 结构兼容性如何注入 VLA 策略与 MLLM 生成；SGW-GAN 是 GW 判别式引导的直接实例。当前所有工作都是「先求 GW 耦合、再蒸馏进生成器」的一阶流方案，GW 引导的扩散/SB 版本尚缺。

## 5. 开放问题与可发论文的切入点

1. **跨维 Gromov-Schrödinger bridge**：GENOT 是确定性一阶 FM 骨架 + 静态 EGW 耦合。做法：把 entropic (F)GW plan 作为 SB 的端点耦合先验，在两个不同维度空间之间各自跑受控扩散、以 GW 畸变项替代路径空间 KL 中的代价，证明解的存在性/Γ-收敛到静态 GW，再在单细胞多组学（CITE-seq→RNA）与图↔点云翻译上对比 GENOT 的样本质量与轨迹稳定性。
2. **训练无关的潜空间缝合（GW stitching of diffusion latents）**：用 GW-at-Scale 的可微 CNT solver 在两个冻结生成模型（如不同分辨率/模态的扩散模型）潜空间间学一个轻量线性/正交映射，推理期直接把模型 A 的中间潜变量搬进模型 B 继续去噪。实验：跨模型图像编辑一致性、FID 与对齐畸变的 trade-off 曲线；理论：CNT 分解下映射的 Lipschitz 稳定性界。
3. **可认证且可扩展的对齐质量证书**：SDP 松弛 (NeurIPS 2024) 给出全局下界但难以过万点；结合低秩耦合约束或 CNT lifted 结构做「结构化 SDP」，输出跨模态检索/对齐 benchmark 上的可认证 GW gap——直接回答「Blind Match 找到的解离全局最优多远」。
4. **FGW 作为图扩散/流匹配的训练目标**：现状是 FGW 只用于评估、增广（mixup/barycenter）或 DP 保证（AISTATS 2026）。做法：用低秩 FGW 的可微近似作为 discrete flow matching 的终端损失（替代逐边交叉熵），在 QM9/MOSES 与 SBM 图上测结构统计的保真度；难点与卖点是 FGW 梯度的偏差—方差控制（可借 GW-at-Scale 的去偏结论）。
5. **GW plan 的统计稳定性 → 生成误差传播**：AoS 2024 只给出 GW **代价**的收敛率；生成对齐真正用的是**耦合本身**。证明 entropic GW plan 对样本扰动的稳定性（W₂ 意义），并推导「plan 估计误差 → GENOT 类生成分布误差」的 Lipschitz 界，给跨模态翻译提供第一个端到端统计保证。

## 6. 代码与资源

- [POT — Python Optimal Transport](https://pythonot.github.io/)：entropic/低秩 GW、FGW、partial GW、(F)GW barycenter 最全参考实现
- [OTT-JAX](https://ott-jax.readthedocs.io/)：低秩 GW/FGW、可微 Sinkhorn，moscot 与 GENOT 的底座
- [egw-solvers](https://github.com/guillaumeHoury/egw-solvers)：ICML 2026 GW-at-Scale 官方 PyTorch/KeOps 实现（CNT、多尺度、GW 梯度流/重心）
- [moscot](https://moscot.readthedocs.io/)：Nature 2025 单细胞时空/跨模态 (F)GW 框架（scverse 生态）
- [itsamatch](https://github.com/dominik-schnaus/itsamatch)：CVPR 2025 视觉-语言盲匹配（改进 Hahn-Grant QAP 求解器）
- [gwae](https://github.com/ganmodokix/gwae)：ICLR 2023 GW 自编码器官方实现
- [GW-Solvers](https://github.com/Ark-130994/GW-Solvers)：连续 GWOT benchmark 与 NeuralGW 实现
- 数据/基准：单细胞跨模态（CITE-seq、multiome，见 GENOT/moscot 论文）、图分类/匹配标准集（MUTAG/IMDB 等，FGW 系）、CINIC-10/CIFAR + DINOv2/SBERT 嵌入（Blind Match 盲对齐协议）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2024_klein_genot_gw_flow_matching.pdf | GENOT: Entropic (Gromov) Wasserstein Flow Matching with Applications to Single-Cell Genomics | 成功（NeurIPS 官方，11.9MB） |
| 2026_houry_gw_at_scale_cnt.pdf | Gromov-Wasserstein at Scale, Beyond Squared Norms | 成功（arXiv，11.0MB） |
| 2024_zhang_gw_duality_sample_complexity.pdf | Gromov-Wasserstein Distances: Entropic Regularization, Duality and Sample Complexity | 成功（arXiv，0.98MB） |
| 2025_schnaus_blind_match_vision_language.pdf | It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data | 成功（CVF 官方，0.52MB） |
| 2026_lyu_last_vla_gw_alignment.pdf | LAST: Bridging Vision-Language and Action Manifolds via Gromov-Wasserstein Alignment | 成功（arXiv，5.1MB） |
| 2026_wang_mirror_language_image_gw.pdf | MIRROR: Aligning Semantic Relations from Language to Image via Gromov-Wasserstein | 成功（arXiv，3.4MB） |
| 2024_chen_sdp_relaxations_gw.pdf | Semidefinite Relaxations of the Gromov-Wasserstein Distance | 成功（arXiv，0.89MB） |
