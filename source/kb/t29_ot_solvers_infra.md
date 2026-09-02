# T29 高性能 OT 求解器与训练基础设施

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」全景的**系统底座**：无论是训练期 batch 内 OT 配对、推理期 OT 引导，还是数据集级跨域耦合，最终都要落到"多大规模、多少毫秒、多少显存"上。它回答的问题是：现有 GPU/近似求解器把 OT 的可行规模推到了哪里，以及在扩散训练管线里引入 OT 的真实工程代价是多少。Sinkhorn 收敛理论归 T04，minibatch 耦合的统计偏差归 T08，本篇只管算法工程与系统。

## 1. 核心问题与背景

离散 OT 的朴素代价是 O(n³ log n)（精确 LP/匈牙利）或 O(n²) 时间与内存（Sinkhorn 每迭代一次矩阵-向量积），这决定了 OT 在深度学习里长期只能以"小 batch 配对"或"低维近似"的形态出现。2024–2026 年该瓶颈被从四个正交方向同时突破：(i) **kernel 级 IO 工程**——把 Sinkhorn 更新重写为 FlashAttention 同构的 online-LogSumExp 流式归约，显存从 O(n²) 降到 O(nd)（FlashSinkhorn、FastSinkhorn）；(ii) **更强的迭代算法**——二阶/拟牛顿（Sparse Newton、cuRegOT）、ε 退火分治（ProgOT）与一阶 LP 求解器复兴（PDOT/cuPDLP 系）；(iii) **结构近似**——低秩耦合（Scetbon 系→FRLC）从"有损近似"进化为层次化精确求解的构件（HiRef、HALO），把全秩 Monge 映射推到百万点；(iv) **sliced/投影系**——用 QMC、控制变量、树切片压方差降常数。与此同时，扩散/流匹配训练管线里"batch 内 OT 配对"的实际开销首次被系统量化（Immiscible Diffusion、Haxholli、Boïté），结论是中小 batch 下配对近乎免费、超大 batch 下成为真实瓶颈——这是所有"OT 改进扩散训练"工作的成本前提。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU (Ye et al.) | 2026·ICML Oral | [A] | 把 log-domain Sinkhorn 更新重写为 attention 同构的 online-LSE，Triton 融合 kernel 流式过 SRAM，O(nd) 显存 + 解析梯度/HVP/半对偶 c-transform；A100 上比 KeOps 前向快 9–32×、端到端最高 161× | [arXiv](https://arxiv.org/abs/2602.03067) |
| ⭐ Progressive Entropic Optimal Transport Solvers（ProgOT, Kassraie et al.） | 2024·NeurIPS | [P] | 沿时间轴分治 + ε 渐进退火调度，兼顾速度、鲁棒与统计意义，可输出 Monge map 估计；是"固定 ε Sinkhorn"之外的算法层标杆 | [NeurIPS 官方页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/22b6bc18be9c2bfaa48adc1122f0a971-Abstract-Conference.html) |
| Accelerating Sinkhorn Algorithm with Sparse Newton Iterations (Tang et al.) | 2024·ICLR | [P] | Sinkhorn 一阶缩放后接 Hessian 稀疏化的 Newton 迭代，超线性收敛；与硬件加速正交、可叠加 | [ICLR 官方页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3a819a53408e20b75d1954bf617ccc0a-Abstract-Conference.html) |
| cuRegOT: A GPU-Accelerated Solver for Entropic-Regularized OT (Qiu) | 2026·arXiv | [R] | 把 sparse-plus-low-rank 拟牛顿法 GPU 化：摊销稀疏符号分析、CPU/GPU 异步流水、融合梯度 kernel；在难例（小 η=0.001）上显著快于 POT/OTT-JAX/AccSinkhorn | [arXiv](https://arxiv.org/abs/2605.08793) |
| FastSinkhorn: Fast Log-Domain Sinkhorn with Warp-Level GPU Reductions (Xiao) | 2026·arXiv | [R] | 原生 CUDA warp-level shuffle 归约 + shared-memory tiling 的 log-domain Sinkhorn；ε 低至 1e-4 仍稳定，n=8192 时比 POT 快 12×、仅 256MB 显存 | [arXiv](https://arxiv.org/abs/2605.00837) |
| PDOT: a Practical Primal-Dual Algorithm and a GPU-Based Solver for OT (Lu & Yang) | 2024·arXiv | [R] | 用 restarted PDHG（cuPDLP 血统）做 matrix-free 高精度 OT：数据无关 O(1/ε) 复杂度，GPU 上高精度区间胜过 Sinkhorn 与商用 LP | [arXiv](https://arxiv.org/abs/2407.19689) |
| ⭐ A Memory-Efficient Hierarchical Algorithm for Large-scale OT（HALO, Xia et al.） | 2026·ICLR Poster | [A] | 层次多尺度 warm-start + active support 剪枝 + 因子化-free 一阶 LP 求解器（默认 cuPDLPx），O(n) 内存；1024² 像素图像 8.9× 加速、省 70.5% 显存 | [OpenReview](https://openreview.net/forum?id=CkOBcyntGd) |
| ⭐ Hierarchical Refinement: Optimal Transport to Infinity and Beyond（HiRef, Halmos et al.） | 2025·ICML Oral | [P] | 证明低秩耦合因子与 Monge map 共聚类，用低秩 OT 递归构造多尺度划分、log-linear 时间/线性空间恢复**全秩双射**，百万点规模超出 Sinkhorn 可及范围 | [OpenReview](https://openreview.net/forum?id=EBNgREMoVD)·[arXiv](https://arxiv.org/abs/2503.03025) |
| Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling（FRLC, Halmos et al.） | 2024·NeurIPS | [P] | latent coupling 因子化把低秩 OT 解耦成三个子 OT 问题，坐标镜像下降求解；统一 W/GW/FGW × 均衡/非均衡/半松弛，线性空间 | [NeurIPS 官方页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cfc1924c62e72e2cb0e0feeecb963241-Abstract.html) |
| Massively Scalable Sinkhorn Distances via the Nyström Method (Altschuler et al.)（奠基） | 2019·NeurIPS | [P] | Nyström 低秩核近似 + Sinkhorn 缩放的稳定性分析，近线性时间/内存；"核近似"路线的源头 | [arXiv](https://arxiv.org/abs/1812.05189) |
| Low-Rank Sinkhorn Factorization (Scetbon, Cuturi & Peyré)（奠基） | 2021·ICML | [P] | 不近似核而直接约束耦合的非负秩，任意 cost 通用；"低秩耦合"路线的源头，OTT-JAX 内置实现 | [arXiv](https://arxiv.org/abs/2103.04737)·[ICML 页](https://icml.cc/virtual/2021/poster/8545) |
| Quasi-Monte Carlo for 3D Sliced Wasserstein（QSW, Nguyen et al.） | 2024·ICLR | [P] | 用球面低差异点集替代 MC 投影方向，系统评测多种 QMC 构造；RQSW 随机化后保无偏可做 SGD——sliced 系的数值工程标杆 | [ICLR 官方页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4b6898c70d5b328deaf2216aefd8f77a-Abstract-Conference.html) |
| ⭐ Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment (Li et al.) | 2024·NeurIPS | [P] | batch 内图像-噪声线性指派（一行代码）+ 量化指派把开销压到 22.8ms@batch1024/A6000（KNN 变体 0.7ms），扩散训练提速最高 3×；"OT 配对进大规模训练管线"的成本样板 | [NeurIPS 官方页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a422a2f016c14406a01ddba731c0969a-Abstract-Conference.html) |
| Expected Batch Optimal Transport Plans and Consequences for Flow Matching (Boïté, Delon & Nadjahi) | 2026·arXiv | [R] | 训练-推理开销权衡的首个定量刻画：exact batch OT 配对 k≤1024 仅 +3% 训练时间、k=8192 达 +55%，大 OT batch 换低 NFE 采样质量（理论部分归 T08，此处取其成本结论） | [arXiv](https://arxiv.org/abs/2605.12174) |
| Minibatch Optimal Transport and Perplexity in Discrete Flow Matching (Haxholli et al.) | 2026·ICML | [A] | 离散 FM 中系统测量 OT 配对开销：约 3.4%（batch 128 级）至 ~12%（batch 2048–4096 需 GPU Sinkhorn）；开销与词表无关、随序列长度占比反而下降 | [OpenReview](https://openreview.net/forum?id=A8rmJlSET9)·[arXiv](https://arxiv.org/abs/2411.00759) |

表外支撑文献（正文引用）：Sliced Wasserstein Estimation with Control Variates（[ICLR 2024 [P]](https://openreview.net/forum?id=StYc4hQAEi)）、Tree-Sliced Wasserstein: Nonlinear Slicing（[ICML 2025 [P]](https://proceedings.mlr.press/v267/tran25c.html)）、Multisample Flow Matching（[ICML 2023 [P]](https://proceedings.mlr.press/v202/pooladian23a.html)）、OT-CFM（[TMLR 2024 [P]](https://arxiv.org/abs/2302.00482)）、GeomLoss（[AISTATS 2019 [P]](https://proceedings.mlr.press/v89/feydy19a.html)）、OTT-JAX（[arXiv 2022 [R]](https://arxiv.org/abs/2201.12324)）、cuPDLPx（[arXiv 2025 [R]](https://arxiv.org/abs/2507.14051)）、BCDNS 块坐标网络单纯形（[arXiv 2025 [R]](https://arxiv.org/abs/2506.21231)）。

## 3. 方法演进脉络

**第一代（2013–2019）：让 OT 能上 GPU。** Cuturi 2013 的熵正则把 OT 变成矩阵缩放，天然并行但 O(n²) 内存。2019 年出现两条绕开稠密核矩阵的路线：Altschuler 等用 **Nyström 低秩核近似**（近线性复杂度、有精度保证）；Feydy 的 GeomLoss 借 **KeOps 符号矩阵**在不物化 cost 矩阵的前提下做 tiled map-reduce——后者成为此后五年 PyTorch 生态的事实标准。

**第二代（2021–2024）：算法层面提速。** Scetbon–Cuturi–Peyré 改为**直接约束耦合的非负秩**（不再近似核，任意 cost 可用），开出"低秩耦合"分支；ICLR 2024 Sparse Newton 在一阶缩放后引入稀疏二阶迭代拿到超线性收敛；NeurIPS 2024 ProgOT 用 ε 退火 + 时间分治调和"快"与"准"。同期 LP 求解器社区的 PDLP/cuPDLP 革命外溢到 OT：PDOT（2024）证明 restarted PDHG 做 matrix-free 精确 OT 在 GPU 上可行，高精度区间反超 Sinkhorn。

**第三代（2024–2026）：低秩从近似变构件 + kernel 级 IO 工程。** FRLC（NeurIPS 2024）用 latent coupling 把低秩 OT 泛化到 GW/非均衡；HiRef（ICML 2025 Oral）发现低秩因子与 Monge map 的共聚类不变量，据此递归细化出**全秩双射**，log-linear 时间扫过百万点；HALO（ICLR 2026）把层次 warm-start、active 剪枝与 cuPDLPx 组合成 O(n) 内存的精确求解管线——低秩/层次与一阶 LP 两条线在此汇流。kernel 侧则复刻了 attention 的进化史：FastSinkhorn（2026 预印本）用 warp-level 归约做原生 CUDA log-domain Sinkhorn；FlashSinkhorn（ICML 2026 Oral）看破"稳定化 Sinkhorn 更新 = biased dot-product 的行 LSE = attention 归一化"，直接搬 FlashAttention 的 tiling/online-softmax，一次 pass 完成对偶更新，并配套流式梯度/HVP/c-transform，前向 9–32×、端到端最高 161×。cuRegOT 则把二阶方法也 GPU 化，补齐"难例小 ε"象限。sliced 系并行推进：QMC 点集与控制变量（ICLR 2024）压 MC 方差，树切片（ICML 2025）引入非线性投影结构。

**管线侧（2023–2026）：从"能用"到"算得清账"。** Multisample FM（ICML 2023，+4% 训练时间）与 OT-CFM（TMLR 2024，<1% 开销）确立 batch 内配对范式；Immiscible Diffusion（NeurIPS 2024）用量化指派把配对压到毫秒级并在 Stable Diffusion 级训练中验证；Haxholli（ICML 2026）与 Boïté（2026 预印本）分别给出离散 FM 与大 OT batch 情形的完整开销曲线——至此"OT 配对在扩散训练里花多少钱"第一次有了可引用的数字。

## 4. 与博客两个方向的关联

- **方向一（无须重训的轨迹对齐）**: 直接相关。推理期/后处理式的轨迹对齐往往要在采样循环里反复求两组样本（或两条轨迹边缘）间的耦合——FlashSinkhorn 的流式 c-transform/半对偶 kernel 与解析梯度使"每步在线 OT"从 O(n²) 显存降到 O(nd)，这是把对齐做进采样器而不拖垮吞吐的前提；Immiscible 的 22.8ms 量化指派给出了"轻量配对可以塞进任何循环"的经验上界。若对齐只需分布级信号，QSW/控制变量的低方差 sliced 估计器是更便宜的替代。
- **方向二（OT 引导跨域生成）**: 直接相关。跨域引导需要数据集级（而非 batch 级）的耦合或 transport 势：HiRef/HALO 把百万点全秩耦合变为可行，FRLC/低秩系提供可解释的粗粒度耦合；GeomLoss/OTT-JAX 的可微 Sinkhorn 势可直接作为 guidance 的梯度源。开销数据（Boïté 的 +55%@k=8192）提示：引导中不应每步重解 OT，而应预计算 plan/对偶势再 amortize 到采样过程。

## 5. 开放问题与可发论文的切入点

1. **把 IO-aware Sinkhorn 推广到平方欧氏之外**。FlashSinkhorn 依赖 cost 的 dot-product 分解，一般 cost（cosine、Mahalanobis、图测地近似）与 unbalanced/partial 的 KL-scaled 更新尚无 online-LSE 形式。具体做法：推导 generalized Sinkhorn（Chizat 系）更新的 shifted-potential 流式版本，写 Triton kernel，在 UOT/partial benchmark 上与 GeomLoss 同精度对比——种子库路线 A 明确点名此切口（非欧 cost、UOT kernel、超小 ε）。
2. **多 GPU/分布式 IO-aware Sinkhorn 完全空白**。FlashSinkhorn/FastSinkhorn/cuRegOT/HALO 全是单卡。可做 ring-attention 式的分片 online-LSE 归约（对偶势 all-reduce，cost tile 本地生成零通信），给出通信量下界证明，在 n≈10⁷–10⁸ 点云上验证弱扩展性——直接服务数据集级跨域耦合（方向二）。
3. **扩散训练中 OT 配对开销的统一 benchmark 缺失**。Immiscible（22.8ms）、Haxholli（3.4–12%）、Boïté（3–55%）的设定互不可比（solver、batch、维度、硬件、误差定义全不同）。建立统一 harness（固定 dtype/tolerance/计时边界），扫 batch×dim×solver（POT-EMD/Sinkhorn/FlashSinkhorn/量化指派/sliced）网格，输出"配对质量–墙钟时间–FID"三维 Pareto，并给出实践者查表——低垂果实，工作量可控且引用面大。
4. **OT 配对与训练流水线的重叠调度**。配对可在 CPU dataloader worker 或独立 GPU stream 中前瞻一个 step 异步执行，与 backward 重叠后有效开销理论上趋近 0，但无人实现与量化。具体：在 ImageNet 级 FM 训练中实现异步配对服务，测 k=8192（Boïté 显示对低 NFE 有益）时 overlap 前后的端到端吞吐差，若成立则"大 batch OT 配对免费"将改变 T08 里的 batch size 权衡结论。
5. **低秩/层次求解器的可微性**。HiRef/HALO/FRLC 当前是"求解即终点"，其输出 plan 对输入点云/cost 的导数（隐函数定理 vs 展开反传）没有公开实现，导致它们进不了需要梯度的 guidance/训练场景。给出 latent-coupling 因子化的隐式微分公式并入 OTT-JAX，可一举打通方向二的"大规模 + 可微"需求。

## 6. 代码与资源

| 资源 | 说明 | 链接 |
|---|---|---|
| FlashSinkhorn | Triton 流式 EOT，GeomLoss 兼容 `SamplesLoss` API，MIT；kernel 仓库 ot_triton | [repo](https://github.com/ot-triton-lab/flash-sinkhorn)·[kernels](https://github.com/ot-triton-lab/ot_triton) |
| POT | 最全 OT 基线库（exact EMD/Sinkhorn/UOT/partial/GW/barycenter）；batch 配对最常用的 `ot.emd` 在此 | [官网](https://pythonot.github.io/) |
| OTT-JAX | JAX 可微 OT：Sinkhorn、低秩（Scetbon 系内置）、几何对象；JIT 首编译需单独计时 | [文档](https://ott-jax.readthedocs.io/) |
| GeomLoss | PyTorch 大规模可微 Sinkhorn loss（tensorized/online/multiscale 三后端）；点云场景标杆 | [官网](https://www.kernel-operations.io/geomloss/) |
| KeOps | 符号 LazyTensor 引擎，免物化大核矩阵；GeomLoss online 后端的底层，非完整 OT solver | [官网](https://www.kernel-operations.io/keops/) |
| regot-cuda | cuRegOT 官方实现（SPLR 拟牛顿 GPU 版） | [repo](https://github.com/yixuan/regot-cuda) |
| cuPDLPx | GPU 一阶 LP 求解器（restarted Halpern PDHG），HALO 的默认底层 | [repo](https://github.com/MIT-Lu-Lab/cuPDLPx) |
| TorchCFM | OT-CFM/Multisample FM 参考实现，含 batch OT 配对代码路径 | [repo](https://github.com/atong01/conditional-flow-matching) |
| Immiscible Diffusion | 一行代码噪声指派 + 量化指派/KNN 实现 | [repo](https://github.com/yhli123/immiscible-diffusion) |
| Benchmark 数据 | DOTmark（图像 OT 标准集，HALO 用）、ModelNet10 点云（HALO/点云 OT 用）；统一计时字段建议沿用种子库 §6.1 模板 | 见各论文附录 |

选型速查：batch≤1k 精确配对→POT `ot.emd`（CPU 足够，Immiscible 证明毫秒级）；单卡大规模熵正则→FlashSinkhorn（平方欧氏）或 GeomLoss online（一般 cost）；需要可微/JAX/低秩→OTT-JAX；高精度/小 ε 难例→cuRegOT 或 PDOT；百万点全秩→HiRef/HALO；只要分布级 loss→sliced（QSW/控制变量）。

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2026_Ye_FlashSinkhorn_IO_Aware_EOT.pdf | FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU | 成功 |
| 2024_Kassraie_Progressive_Entropic_OT_Solvers.pdf | Progressive Entropic Optimal Transport Solvers | 成功 |
| 2025_Halmos_Hierarchical_Refinement_OT.pdf | Hierarchical Refinement: Optimal Transport to Infinity and Beyond | 成功 |
| 2024_Li_Immiscible_Diffusion_Noise_Assignment.pdf | Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment | 成功 |
| 2026_Boite_Expected_Batch_OT_Plans_FM.pdf | Expected Batch Optimal Transport Plans and Consequences for Flow Matching | 成功 |
| 2024_Lu_PDOT_GPU_Primal_Dual_OT.pdf | PDOT: a Practical Primal-Dual Algorithm and a GPU-Based Solver for Optimal Transport | 成功 |
| 2026_Qiu_cuRegOT_GPU_Entropic_OT.pdf | cuRegOT: A GPU-Accelerated Solver for Entropic-Regularized Optimal Transport | 成功 |
| 2026_Xiao_FastSinkhorn_Warp_Level_GPU.pdf | Fast Log-Domain Sinkhorn Optimal Transport with Warp-Level GPU Reductions | 成功 |
| —（未保存） | A Memory-Efficient Hierarchical Algorithm for Large-scale OT (HALO) | 失败（OpenReview PDF 两次尝试均返回 HTML 反爬页，按纪律删除；正文以 forum 页为准） |
