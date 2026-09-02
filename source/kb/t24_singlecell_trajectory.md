# T24 单细胞与生物轨迹推断中的 OT×流

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」最成熟的科学落地出口——单细胞测序是破坏性的，同一细胞无法被观测两次，实验只能给出无配对的时间快照，OT/Schrödinger bridge/flow matching 因此成为把快照缝合成连续动力学的天然语言。该领域也持续向方法核心回馈新问题设置（unbalanced 质量增减、多边缘、分支、细胞交互、条件化扰动），是 T03（SB 理论）、T07/T08（FM 与耦合设计）成果的最大真实试验场。SB 理论本身归 T03，分子构象生成归 T21，本笔记只覆盖细胞/组织尺度的应用与由应用驱动的方法创新。

## 1. 核心问题与背景

时序 scRNA-seq 给出发育或扰动过程在若干时间点的独立细胞群体快照 \(\{\hat\rho_{t_i}\}\)，每个快照是高维基因表达空间中的经验测度，且细胞间无对应关系。核心任务有四类：(i) 轨迹推断/快照插值——恢复连接各边缘的耦合或连续动力学 \(v_t, g_t\)（速度场+增殖/凋亡导致的质量变化）；(ii) 扰动响应预测——学习 control→perturbed 的传输映射并外推到未见药物/剂量/病人；(iii) 跨模态对齐——RNA/ATAC/蛋白等不同测量空间之间的细胞翻译（Gromov-Wasserstein 型问题）；(iv) 空间转录组配准——跨切片/跨时间点对齐带坐标的表达测度。四类问题共享同一数学骨架：在测度空间上解（熵正则、unbalanced、多边缘、fused-GW 变体的）OT 或 SB，而 2023 年后 flow matching 的 simulation-free 训练把这套骨架从「离散求解器」升级为「可外推的神经动力学模型」，形成本子课题 2024–2026 的爆发。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming (Waddington-OT, Schiebinger et al.) | 2019·Cell | [P] | 奠基：把发育建模为测度演化，相邻时间点间解带增殖率的熵正则 unbalanced OT，31.5 万细胞重编程谱系与祖先/命运分析 | [Cell](https://doi.org/10.1016/j.cell.2019.01.006) |
| TrajectoryNet (Tong et al.) | 2020·ICML | [P] | 首个连续化：CNF + 动态 OT 能量惩罚做快照间连续插值，可加密度/velocity 正则 | [PMLR](https://proceedings.mlr.press/v119/tong20a.html) |
| MIOFlow (Huguet et al.) | 2022·NeurIPS | [P] | 测地自编码器潜空间中训练 neural ODE、以流形 ground distance 的 OT 罚项插值快照，处理分叉/汇合 | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bfc03f077688d8885c0a9389d77616d0-Abstract-Conference.html) |
| ⭐ Learning Single-Cell Perturbation Responses using Neural Optimal Transport (CellOT, Bunne et al.) | 2023·Nature Methods | [P] | 扰动线奠基：ICNN 对偶势学 control→perturbed 的 Monge map，预测未见病人药物响应（4i+scRNA） | [Nat Methods](https://doi.org/10.1038/s41592-023-01969-x) |
| ⭐ Simulation-Free Schrödinger Bridges via Score and Flow Matching ([SF]²M, Tong et al.) | 2024·AISTATS | [P] | 静态熵正则/minibatch Sinkhorn 耦合 + score+flow 双回归免仿真解 SB，千维基因空间轨迹推断 SOTA | [PMLR](https://proceedings.mlr.press/v238/tong24a.html) |
| ⭐ GENOT: Entropic (Gromov) Wasserstein Flow Matching with Applications to Single-Cell Genomics (Klein et al.) | 2024·NeurIPS | [P] | 范式切换：用条件 FM 直接建模熵 OT 耦合的条件分布 π_ε(y\|x)，原生支持任意成本、unbalanced 与 (Fused) GW——发育/药物响应/ATAC→RNA 跨模态三合一 | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc46e29f91e676747c584ca181cb0ea1-Abstract.html) |
| Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation (Eyring, Klein, Uscidda et al.) | 2024·ICLR | [P] | 理论化的重标定方案把 unbalanced 塞进任意 Monge 估计器与 OT-FM（UOT-FM）：胰腺发育轨迹与扰动预测显著受益 | [OpenReview](https://openreview.net/forum?id=2UnCj3jeao) |
| TIGON (Sha, Qiu, Zhou & Nie) | 2024·Nat. Mach. Intell. | [P] | Wasserstein–Fisher–Rao 动态 unbalanced OT 的 neural ODE 求解：同时重建轨迹、增殖率与时序基因调控网络 | [NMI](https://www.nature.com/articles/s42256-023-00763-w) |
| ⭐ Mapping Cells Through Time and Space with moscot (Klein, Palla, Lange et al.) | 2025·Nature | [P] | 工程集大成：低秩熵 OT/GW/FGW 统一时序、空间、时空、谱系全部单细胞 OT 应用，多模态、170 万细胞×20 时间点 atlas 规模，实验验证 NEUROD2 | [Nature](https://www.nature.com/articles/s41586-024-08453-2) |
| DeepRUOT: Learning Stochastic Dynamics from Snapshots through Regularized Unbalanced Optimal Transport (Zhang, Li & Zhou) | 2025·ICLR (Oral) | [P] | 正则化 unbalanced OT（≈unbalanced SB）的深度求解器：Fisher 正则把 SDE 问题化成 ODE 约束，无先验地同时学增殖与转移、重建 Waddington 景观 | [OpenReview](https://openreview.net/forum?id=gQlxd3Mtru) |
| Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold (Atanackovic et al.) | 2025·ICLR | [P] | 把「初始群体」用 GNN 嵌入后 amortize 速度场——Wasserstein 流形上的向量场积分，泛化到未见病人的治疗响应 | [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/ebdb990471f653dffb425eff03c7c980-Paper-Conference.pdf) |
| Modeling Complex System Dynamics with Flow Matching Across Time and Conditions (MMFM, Rohbeck et al.) | 2025·ICLR (Spotlight) | [P] | 多边缘 FM：样条插值构造跨时间点平滑条件路径 + classifier-free guidance 跨条件共享动力学，补全缺失(时间点×扰动)组合 | [OpenReview](https://openreview.net/forum?id=hwnObmOTrV) |
| Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge (CytoBridge, Zhang et al.) | 2025·NeurIPS | [P] | UMFSB：unbalanced SB 加 mean-field 交互项，四网络（速度/增长/对数密度/交互势）从快照学细胞间相互作用 | [OpenReview](https://openreview.net/forum?id=Z6DJJIN8IJ) |
| DeST-OT: Alignment of Spatiotemporal Transcriptomics Data (Halmos et al.) | 2025·Cell Systems | [P] | 空间时序配准：semi-relaxed FGW 建模发育组织切片间的生长/凋亡/分化，提出 growth-distortion 与 migration 度量 | [Cell Systems](https://doi.org/10.1016/j.cels.2024.12.001) |
| Branched Schrödinger Bridge Matching (BranchSBM, Tang et al.) | 2026·ICLR | [A] | 分支 SB：把广义 SB 分解为多条带权 unbalanced 随机最优控制分支（每支独立速度+增长网络），建模命运分叉与扰动分歧 | [OpenReview](https://openreview.net/forum?id=ctq8BfUXWz) |

表外相邻工作（正文引用）：LineageOT（谱系条码信息进耦合的先驱，[Nat Comms 2021](https://doi.org/10.1038/s41467-021-25133-1)，[P]）、moslin（FGW 用谱系树距离对齐时间点，[Genome Biology 2024](https://doi.org/10.1186/s13059-024-03422-4)，[P]）、CondOT（条件 Monge map 学扰动连续参数化，NeurIPS 2022，[arXiv](https://arxiv.org/abs/2206.14262)，[P]）、Monge Gap（无 ICNN 约束的 Monge 估计器，[ICML 2023 PMLR](https://proceedings.mlr.press/v202/uscidda23a.html)，[P]）、Metric Flow Matching（数据流形黎曼度量下的测地插值，单细胞轨迹 SOTA，NeurIPS 2024，[P]，详见 T08）、Curly Flow Matching（非零参考漂移 SB 学非梯度/周期动力学——细胞周期需要 RNA velocity 做参考过程，[NeurIPS 2025](https://openreview.net/forum?id=7cqKVDgFZQ)，[P]）、Wasserstein Flow Matching（FM 提升到分布的分布：Bures-Wasserstein/点云空间上生成空间转录组 niche，[ICML 2025 PMLR](https://proceedings.mlr.press/v267/haviv25a.html)，[P]）、CellStream（自编码器与 unbalanced 动态 OT 联合学习动力学感知嵌入，[AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37041)，[P]）、3MSBM（动量多边缘 SB matching，[NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7c3875b86bd2b0639ab1e858c678af40-Abstract-Conference.html)，[P]）、Multi-Marginal Temporal Schrödinger Bridge Matching（因子化拟合多时间快照，[arXiv 2510.01894](https://arxiv.org/abs/2510.01894)，[R]）、CellFlow（Theis 组工业级条件 FM 扰动表型引擎：细胞因子/药物/敲除/类器官协议全覆盖，[bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1)，[R]）、轨迹推断统计理论（min-entropy 估计一致性，Lavenant–Zhang–Kirkpatrick–Schiebinger，Ann. Appl. Probab. 2024，[arXiv](https://arxiv.org/abs/2102.09204)，[P]）；空间侧前史：PASTE（FGW 切片对齐，[Nat Methods 2022](https://doi.org/10.1038/s41592-022-01459-6)，[P]）、PASTE2（部分对齐，[Genome Research 2023](https://doi.org/10.1101/gr.277670.123)，[P]）、SCOT（GW 跨组学对齐，[J Comput Biol 2022](https://doi.org/10.1089/cmb.2021.0446)，[P]）、NovoSpaRc（GW 空间重构，[Nat Protocols 2021](https://doi.org/10.1038/s41596-021-00573-7)，[P]）；综述：Optimal Transport for Single-Cell and Spatial Omics（Bunne, Schiebinger, Krause, Regev & Cuturi，[Nat Rev Methods Primers 2024](https://doi.org/10.1038/s43586-024-00334-2)，[B]）。

## 3. 方法演进脉络

**静态耦合时代（2019–2021）**：Waddington-OT 确立范式——细胞群体=概率测度、发育=测度演化，相邻快照间解带增殖修正的熵正则 unbalanced OT，直接读出祖先-后代耦合。LineageOT 把 CRISPR 谱系条码作为额外信息改造成本，SCOT/NovoSpaRc 用 GW 做跨组学与空间重构。这一代输出的是离散耦合矩阵，无法外推到新细胞或未观测时间。

**连续动力学化（2020–2024）**：TrajectoryNet 首先用 CNF+动态 OT 罚项把插值连续化；MIOFlow 把流约束到测地自编码器定义的数据流形上；TIGON 用 Wasserstein–Fisher–Rao 距离的 neural ODE 同时学速度与增殖并反推基因调控网络。痛点是这些方法都要仿真 ODE/PDE 训练，维度与规模受限。

**Simulation-free 革命（2023–2025）**：[SF]²M 把 minibatch Sinkhorn 耦合与 score+flow 双回归结合，免仿真近似 SB，千维基因空间直接可跑；Metric FM 把插值路径弯到数据流形测地线。GENOT 做出关键范式切换：不再学「动力学」而是把熵 OT 耦合本身当作条件生成模型（条件 FM 采样 π_ε(y|x)），一举原生支持任意成本、unbalanced 与 (F)GW 跨空间翻译。DeepRUOT（ICLR'25 oral）回到动力学但把 unbalanced 做成第一等公民：RUOT 框架用 Fisher 正则消掉 SDE 仿真、无先验学增殖；其谱系随后爆发——CytoBridge 加 mean-field 细胞交互（NeurIPS'25）、BranchSBM 做多目标分支（ICLR'26）、Curly-FM 用 RNA velocity 当非零参考漂移学细胞周期等非梯度动力学（NeurIPS'25）、CellStream 把嵌入与动态 OT 联合训练（AAAI'26）、3MSBM/MMTSBM 攻多时间边缘。

**条件化与规模化（2024–2026）**：扰动线从 CellOT 的 ICNN Monge map，经 Monge Gap、UOT-Monge（Eyring）的 unbalanced 修正，走向 FM 化与元学习：MMFM 用样条多边缘路径+classifier-free guidance 跨约百种化学扰动共享动力学；MFM 用 GNN 群体嵌入在 Wasserstein 流形上 amortize 速度场、泛化到未见病人；CellFlow 把这一切工程化成覆盖细胞因子/敲除/类器官协议的通用表型引擎。工程侧 moscot（Nature'25）用低秩 Sinkhorn+GPU 把全部 OT 应用推到 10⁶–10⁷ 细胞 atlas 规模并统一 API；空间侧 PASTE→DeST-OT 用 semi-relaxed FGW 处理发育切片的生长与迁移，WFM 则把 FM 提升到「分布的分布」直接生成组织 niche。整体趋势：离散耦合 → 连续流 → 免仿真 SB → unbalanced/分支/交互/条件全要素动力学 + atlas 规模工程。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: 双向强关联。其一，本领域的「对齐」大量是免训练的：WOT/moscot/DeST-OT 的耦合由 Sinkhorn 直接解出，无任何网络训练——分布漂移（新批次/新病人）时低秩重解一次即可，这是「对齐不必重训」的最有力生物版论据，其低秩熵 OT 工程栈（ott-jax）可直接搬到生成模型推理期对齐。其二，反向空白明显：细胞动力学流模型（[SF]²M/DeepRUOT 等）训练后若来了新时间点快照，目前只能重训；把 T12 式推理期 OT 矫正用于已训练细胞流模型（测试时用新快照做 transport-based steering）几乎无人做。
- 方向二（OT 引导跨域生成）: 本子课题是该方向最丰富的真实任务库，且 OT 在此不是加分项而是任务定义本身：control→perturbed 翻译（CellOT/UOT-Monge/CellFlow）、跨模态 ATAC↔RNA 生成（GENOT 用 FGW 耦合引导跨不相交空间的条件生成，是「OT 引导跨域生成」的教科书实现）、跨时间点生成（SB/FM 全家族）。GENOT 的「熵耦合条件分布 + 条件 FM」与 Eyring 的「unbalanced 重标定」是可直接迁移到图像/多模态跨域生成的两个模板。

## 5. 开放问题与可发论文的切入点

1. **免仿真 unbalanced SB 的统一收敛理论**：[SF]²M 的静态熵耦合路线与 DeepRUOT 的 Fisher 正则路线尚未统一。可证目标：「minibatch unbalanced Sinkhorn 耦合 + score/flow 回归」的组合估计器收敛到 RUOT/unbalanced SB 解及其速率（batch 大小 n、熵 ε、增长惩罚 τ 三参数联合），再把 CytoBridge 的交互项纳入得到带 mean-field 的推广；用 DeepRUOT 的合成基因调控网络做验证曲线。
2. **生物先验进入耦合的系统消融**：Curly-FM 只把 RNA velocity 放进参考漂移。系统研究「velocity/ATAC 可及性/谱系条码」三类先验分别放进 (a) SB 参考过程、(b) OT 成本矩阵（moslin 式）、(c) unbalanced 松弛权重的效果矩阵；C. elegans 与 zebrafish 有全谱系 ground truth，可做首个先验-注入位置的 benchmark 论文。
3. **扰动外推的分布回归对决与泛化界**：MFM（GNN 群体嵌入）、MMFM（classifier-free guidance）、CellFlow（条件编码器）三种条件化机制没有 head-to-head 比较。建立统一基准（scPerturb + Tahoe-100M 级数据），并从 Wasserstein 流形上的 meta-learning 视角给出「未见条件泛化误差 ≤ f(条件嵌入容量, 群体间 W₂ 半径)」的界——方法+理论各半的一篇。
4. **空间时序联合动力学（spatial SB）**：DeST-OT/moscot.spatiotemporal 仍是静态耦合，WFM 能生成 niche 但无时间维。做 (expression, location) 联合空间上的 FGW 型 bridge matching：证明 FGW geodesic 上条件路径的良定性，用 Stereo-seq MOSTA 小鼠器官发生数据验证，能同时输出迁移场与增殖场——空位明确且数据公开。
5. **推理期重校准（免重训）**：训练好的细胞流模型面对新快照/新批次时，用一次离散 OT 耦合构造 posterior steering（类比 T12 的推理期对齐），无需梯度更新地把流「拉回」新边缘；与 LoRA 式微调对比校准误差-成本帕累托。工程量小、与博客方向一直接呼应。

## 6. 代码与资源

- [moscot](https://github.com/theislab/moscot)（[docs](https://moscot.readthedocs.io)）— 时序/空间/时空/谱系全问题类，ott-jax 后端，atlas 规模
- [OTT-JAX](https://github.com/ott-jax/ott) — 低秩 Sinkhorn/GW，GENOT 官方实现所在库
- [TorchCFM](https://github.com/atong01/conditional-flow-matching) — OT-CFM/[SF]²M 官方库（含单细胞示例）
- [DeepRUOT](https://github.com/zhenyiizhang/DeepRUOT) / [CytoBridge](https://github.com/zhenyiizhang/CytoBridge-NeurIPS) — RUOT/UMFSB 官方实现
- [Curly-FM](https://github.com/kpetrovicc/curly-flow-matching) — 非梯度动力学（细胞周期）
- [CellFlow](https://github.com/theislab/CellFlow)（[docs](https://cellflow.readthedocs.io)）— 扰动表型 FM 引擎
- [CellOT](https://github.com/bunnech/cellot)、[WOT](https://github.com/broadinstitute/wot)、[TrajectoryNet](https://github.com/KrishnaswamyLab/TrajectoryNet)、[MIOFlow](https://github.com/KrishnaswamyLab/MIOFlow)、[TIGON](https://github.com/yutongo/TIGON)、[CellStream](https://github.com/PQ-Zhang/CellStream)、[DeST-OT](https://github.com/raphael-group/DeST_OT)、[moslin](https://github.com/theislab/moslin)
- 常用数据/benchmark：EB 胚状体分化（MIOFlow/[SF]²M）、CITE-seq & Multiome（NeurIPS 2021-22 挑战赛，[SF]²M/Metric FM 标准基准）、小鼠胰腺内分泌发育（moscot/Eyring）、C. elegans 全谱系（moslin）、血液发育（DeepRUOT/CytoBridge）、sciPlex 化学扰动（GENOT/MMFM）、scPerturb 扰动汇编（Nat Methods 2024）、Stereo-seq MOSTA 时空图谱（CellStream/spatial 方向）、axolotl 脑再生（DeST-OT）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2024_Tong_SF2M_simulation_free_schrodinger_bridge.pdf | Simulation-Free Schrödinger Bridges via Score and Flow Matching | 成功 |
| 2024_Klein_GENOT_entropic_GW_flow_matching.pdf | GENOT: Entropic (Gromov) Wasserstein Flow Matching with Applications to Single-Cell Genomics | 成功 |
| 2025_Zhang_DeepRUOT_regularized_unbalanced_OT.pdf | Learning Stochastic Dynamics from Snapshots through Regularized Unbalanced Optimal Transport | 成功 |
| 2025_Atanackovic_meta_flow_matching.pdf | Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold | 成功 |
| 2025_Haviv_wasserstein_flow_matching.pdf | Wasserstein Flow Matching: Generative Modeling Over Families of Distributions | 成功 |
| 2025_Petrovic_curly_flow_matching.pdf | Curly Flow Matching for Learning Non-gradient Field Dynamics | 成功 |
| 2025_Zhang_CytoBridge_unbalanced_meanfield_SB.pdf | Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge | 成功 |
| 2026_Tang_BranchSBM_branched_schrodinger_bridge.pdf | Branched Schrödinger Bridge Matching | 成功 |
