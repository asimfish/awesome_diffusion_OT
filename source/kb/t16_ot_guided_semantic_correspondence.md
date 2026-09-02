# T16 OT 代价先验引导的跨域语义对应

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是博客「方向二（OT 引导跨域生成）」的核心机制层——用最优传输代价/耦合作为先验，让扩散生成在两个分布之间挖掘最小成本映射，纠正跨域语义错乱。上承 OT 语义对应的判别式传统（SCOT 谱系），下接扩散 attention 与采样期 guidance；与 T13/T14（完整 I2I 框架）、T17（风格迁移）、T18（guidance 一般理论）互补，本笔记只收"OT 作为对应/对齐先验"的机制性工作。

## 1. 核心问题与背景

跨域生成与编辑中的语义错乱（属性泄漏 attribute leakage、物体错位 mislocated objects、many-to-one 错配）有一个共同根源：逐点最近邻检索和 softmax attention 都是**局部贪心匹配**——每个 query 独立选择最相似的 key，没有全局约束，导致多个源位置挤到同一目标语义上。OT 把匹配升格为**带边际约束的全局最小代价耦合**：质量守恒强制"语义预算"分配，熵正则 Sinkhorn 给出可微的软对应。这一视角带来两条落地路径：(a) **判别式对应**——在（扩散/DINO）特征空间上直接求 transport plan 作为语义对应（SCOT→GWOT-SC→Shape-of-You），代价设计从纯外观演化到外观+几何结构（GW/FGW）+外部先验（keypoint、3D）；(b) **生成式引导**——把 OT 耦合/代价作为先验挂进扩散的训练目标（OTCS）或采样过程（STORM、ASAG、OTComp），training-free 地纠正 attention 的传输结构。由于 softmax attention 本身可视为熵正则 OT 的单边归一化（Sinkformer 形式化了这一点），attention 矩阵成为注入 OT 先验最自然的接口。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| SCOT: Semantic Correspondence as an Optimal Transport Problem | 2020·CVPR | [P] | 奠基：首次把语义对应表述为 OT 问题，用显著性做边际、Sinkhorn 求全局 plan，抑制最近邻的 many-to-one 错配 | [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Semantic_Correspondence_as_an_Optimal_Transport_Problem_CVPR_2020_paper.html) |
| UNITE: Unbalanced Feature Transport for Exemplar-based Image Translation | 2021·CVPR | [P] | 用不平衡 OT + 自适应质量学习对齐条件输入与 exemplar 特征，解决跨域分布偏差下的稠密对应（完整 I2I 框架部分归 T13/T14） | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhan_Unbalanced_Feature_Transport_for_Exemplar-Based_Image_Translation_CVPR_2021_paper.html) |
| Sinkformers: Transformers with Doubly Stochastic Attention | 2022·AISTATS | [P] | 理论接口：softmax attention → Sinkhorn 双随机化，形式化"attention ≈ 熵正则 OT"，是后续所有 attention-OT 工作的依据 | [PMLR](https://proceedings.mlr.press/v151/sander22a.html) |
| Keypoint-Guided Optimal Transport (KPG-RL) | 2022·NeurIPS | [P] | 用 mask 约束 plan + 关系保持把少量标注 keypoint 语义先验注入 OT，支持异构空间与 partial 设定 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6091c5644d73637e3cccdcab52a7031f-Abstract-Conference.html) |
| Simultaneous Multiple-Prompt Guided Generation Using Differentiable OT | 2022·ICCC | [P] | 早期先驱：图像 patch ↔ 多 prompt 的可微 OT 距离直接作为生成引导损失（VQGAN-CLIP 时代） | [ICCC PDF](https://computationalcreativity.net/iccc22/wp-content/uploads/2022/06/ICCC-2022_22L_Tian-et-al..pdf) |
| PLOT: Prompt Learning with Optimal Transport for Vision-Language Models | 2023·ICLR (top-25%) | [P] | prompt/token 级 OT 对齐范式：多 prompt 与局部视觉特征集合做 Sinkhorn 内层对齐、外层学 prompt，防止 prompt 坍缩 | [OpenReview](https://openreview.net/forum?id=zqwryBoXYnh) |
| ⭐ OTCS: Optimal Transport-Guided Conditional Score-Based Diffusion Model | 2023·NeurIPS | [P] | 本方向奠基：L2 正则 OT 在非配对/半配对数据上建耦合，"按相容性重采样"引导条件分数模型训练，理论证明其实现 OT 数据传输 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/72c12e48c6135762f56bf188cd2479d2-Abstract-Conference.html) |
| ⭐ OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation | 2024·ECCV | [P] | MPSA 用 Sinkhorn 替换 Transformer 解码器 cross-attention 归一化，多文本 prompt 选择性对齐像素语义，ZS3 三个基准 SOTA | [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09900.pdf) |
| ⭐ STORM: Spatial Transport Optimization by Repositioning Attention Map | 2025·CVPR | [P] | training-free：定制空间传输代价的 OT 在早期去噪阶段重定位物体 cross-attention map，同时缓解物体错位/缺失/属性错配 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Han_Spatial_Transport_Optimization_by_Repositioning_Attention_Map_for_Training-Free_Text-to-Image_CVPR_2025_paper.html) |
| GWOT-SC: Gromov Wasserstein Optimal Transport for Semantic Correspondences | 2025·BMVC | [P] | 用 GW 空间平滑先验的 OT 匹配替代最近邻，DINOv2 单模型即可竞争 SD 特征 ensemble，效率高 5–10 倍 | [BMVC](https://bmvc2025.bmva.org/proceedings/721/) / [arXiv](https://arxiv.org/abs/2602.03105) |
| ⭐ ASAG: Adversarial Sinkhorn Attention Guidance | 2026·AAAI | [P] | 把 self-attention 重释为 OT，用 Sinkhorn 注入对抗代价构造"劣化分支"做 guidance，即插即用提升 T2I/IP-Adapter/ControlNet 保真度 | [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/37488) / [arXiv](https://arxiv.org/abs/2511.07499) |
| ⭐ Shape-of-You: Fused Gromov-Wasserstein OT for Semantic Correspondence in-the-Wild | 2026·CVPR | [P] | FGW 融合外观代价（W）与结构代价（GW），用 3D 结构先验 + anchor 线性化压低 FGW 计算成本，做 in-the-wild 语义对应 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Im_Shape-of-You_Fused_Gromov-Wasserstein_Optimal_Transport_for_Semantic_Correspondence_in-the-Wild_CVPR_2026_paper.html) |
| OTComp: Dual Optimal Transport for Multi-Concept Composition | 2026·ICML | [A] | 双 OT training-free 引导：质量守恒 OT 做结构草图对齐 + 几何引导 OT 做高频纹理残差传输，多概念组合无属性串扰 | [ICML](https://icml.cc/virtual/2026/poster/63327) / [代码](https://github.com/fuhao7i/OTComp) |
| TP-Blend: Textual-Prompt Attention Pairing for Object-Style Blending | 2026·arXiv | [R] | CAOF 用熵正则 OT 在 cross-attention 中按完整多头维度重分配特征向量，实现对象与风格双 prompt 的精确融合（风格部分归 T17） | [arXiv](https://arxiv.org/abs/2601.08011) |
| Optimal Transport for Rectified Flow Image Editing | 2025·arXiv | [R] | 用传输论轨迹校正统一 inversion-based 与 direct 编辑，training-free 大幅提升 FLUX 等模型编辑的重建/一致性 | [arXiv](https://arxiv.org/abs/2508.02363) |

统计：[P] 12 篇 · [A] 1 篇 · [R] 2 篇，共 15 篇；⭐ 必读 5 篇。

## 3. 方法演进脉络

本方向沿三条线演化，2024 后在扩散 attention 上合流。

**线 A：判别式语义对应的代价设计。** SCOT（CVPR 2020）首先把语义对应写成 OT 问题，用显著性调制边际，Sinkhorn 求全局 plan，解决最近邻的 many-to-one；UNITE（CVPR 2021）把它推进到跨域生成场景——条件输入与 exemplar 的分布天然偏差，于是引入不平衡 OT 与自适应质量学习；KPG-RL（NeurIPS 2022）示范了"先验注入"的另一形态：少量 keypoint 通过 mask 约束 plan 可行域并以关系保持传播引导。基础模型时代，GWOT-SC（BMVC 2025）发现 SD 特征在 ensemble 中的作用其实是提供空间平滑性，于是干脆把平滑性写进匹配算法——GW 结构项 + DINOv2 特征即可媲美 ensemble 且快 5–10 倍；Shape-of-You（CVPR 2026）进一步用 FGW 显式融合外观与结构两种代价，并以 3D 先验和 anchor 线性化控制计算量。代价设计的演化主线：纯外观 → 不平衡质量 → 关键点/掩码先验 → 空间/几何结构（GW/FGW）→ 外部 3D 先验。

**线 B：attention 即 OT 的接口化。** Sinkformer（AISTATS 2022）证明 softmax attention 是熵正则 OT 的单边化，换成 Sinkhorn 即得双随机 attention；PLOT（ICLR 2023）把 OT 对齐搬到 prompt-token 集合层面；OTSeg（ECCV 2024）第一次在多模态解码器里用 MPSA 整体替换 cross-attention 归一化，让多个文本 prompt 各自"运输"到不同像素语义。2025 起接口从"替换归一化"升级为"采样期操控传输结构"：STORM（CVPR 2025）用带空间代价的 OT 把物体 attention map 重定位到目标区域（发现早期去噪步最有效）；ASAG（AAAI 2026）反向操作——用对抗代价的 Sinkhorn 构造"劣化 attention"分支，替代 PAG/SEG 的启发式扰动，给 guidance 提供了 OT 语义下的原理性负样本；TP-Blend（2026, [R]）则用熵正则 OT 在 cross-attention 里做全头维特征重分配。

**线 C：采样/训练级 OT 耦合先验。** ICCC 2022 的多 prompt OT 引导是精神先驱；OTCS（NeurIPS 2023）是本子课题的枢纽工作——用 L2 正则（半）监督 OT 在非配对数据上估计耦合 π̂，再按相容性重采样伪配对来训练条件分数模型，并证明生成过程实现了 OT 意义下的数据传输（带界）；OT-RF-Edit（2025, [R]）把传输校正用于 rectified flow 编辑轨迹；OTComp（ICML 2026）给出 training-free 的双 OT 分解：先用质量守恒 OT 做结构对齐，再用几何引导 OT 传输高频纹理，明确把"结构/纹理"两类语义错乱分开治理。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **间接相关**。STORM/ASAG/OTComp 均为 training-free 推理期干预，与方向一共享"不动权重、只改采样"的哲学，但作用对象是 attention/特征传输结构而非扩散轨迹本身；OT-RF-Edit 的轨迹校正在形式上最接近方向一（对 rectified flow 轨迹做传输论修正），可作为两方向的连接点。
- 方向二（OT 引导跨域生成）: **正中核心**。OTCS 把"最小成本耦合 = 跨域配对先验"做成可训练机制，是"用 OT 代价先验辅助扩散挖掘最小成本映射"的直接实现；线 A（GWOT-SC、Shape-of-You）提供跨域语义对应的代价设计模板（外观 W + 结构 GW + 外部先验）；线 B（OTSeg、STORM、TP-Blend）证明 OT 先验能在 attention 层面直接纠正语义错乱（错位、泄漏、串扰）。三条线合起来就是方向二的机制工具箱。

## 5. 开放问题与可发论文的切入点

1. **FGW 对应闭环进扩散采样**：线 A 的判别式对应与线 C 的生成引导目前割裂。具体做法：在 SDEdit/FlowEdit 类编辑管线中，每 k 步用"源图 ↔ 当前去噪 latent"的扩散特征求 FGW plan，构造 correspondence 一致性损失做梯度引导；在 SPair-71k 关键点保持 + 编辑一致性（LPIPS/CLIP-dir）上与 attention 注入基线（P2P、MasaCtrl）对比。
2. **不平衡/partial OT 的 guidance 化**：跨域语义不守恒（物体增删、部件缺失）时 balanced OT 会强制错配。把 UNITE 的自适应质量学习搬进 ASAG 式采样期 attention 干预，做 mass-aware Sinkhorn attention guidance；在多物体编辑场景（BindEdit 型 benchmark）上测属性泄漏率与编辑成功率。
3. **"attention plan 偏离最优耦合 ⇒ 语义错乱"的定量理论**：ASAG 只有启发式动机。可证命题：在线性化 DiT 假设下，熵正则 OT plan 与 softmax attention 的偏差范数给出 attribute mis-binding 概率的上界；实验用可控合成 prompt 集验证界的紧致性。这也为 T18 的 guidance 理论提供一个具体案例。
4. **代价函数学习（inverse OT）**：现有代价全部手工（cosine、空间距离、3D 先验）。用 inverse OT 从少量人工标注对应反推 cost functional（接 ICLR 2024 Neural OT with General Cost Functionals 一线），验证学到的代价在未见类别上的对应泛化，目标是把 Shape-of-You 的 3D 先验替换为数据驱动先验。
5. **token 级 OT 一致性 metric**：把编辑前后 cross-attention plan 的（Gromov-）Wasserstein 距离做成编辑一致性/语义保持度量，与 CLIP-score、LPIPS、人工评分做相关性分析；低成本可发 benchmark/analysis 论文，并反哺切入点 1 的损失设计。

## 6. 代码与资源

- OTCS 官方代码: https://github.com/XJTU-XGU/OTCS （含非配对 SR、半配对 I2I 训练脚本）
- KPG-RL 官方代码: https://github.com/XJTU-XGU/KPG-RL
- OTSeg 官方代码: https://github.com/cubeyoung/OTSeg
- GWOT-SC 官方代码: https://github.com/fsnelgar/semantic_matching_gwot
- OTComp 官方代码: https://github.com/fuhao7i/OTComp
- PLOT 官方代码: https://github.com/CHENGY12/PLOT
- STORM 项目页: https://micv-yonsei.github.io/storm2025/
- OT 求解库: POT (https://pythonot.github.io/) 、OTT-JAX (https://ott-jax.readthedocs.io/)
- 语义对应 benchmark: SPair-71k、PF-PASCAL/PF-WILLOW、TSS；组合生成对齐 benchmark: T2I-CompBench、HRS-Bench

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2023_Gu_otcs_ot_guided_conditional_diffusion.pdf | Optimal Transport-Guided Conditional Score-Based Diffusion Model | 成功 |
| 2024_Kim_otseg_multiprompt_sinkhorn_attention.pdf | OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation | 成功 |
| 2025_Han_storm_spatial_transport_attention.pdf | Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis | 成功 |
| 2025_Snelgar_gwot_semantic_correspondences.pdf | Gromov Wasserstein Optimal Transport for Semantic Correspondences | 成功 |
| 2026_Kim_asag_adversarial_sinkhorn_attention_guidance.pdf | Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance | 成功 |
| 2026_Im_shape_of_you_fgw_semantic_correspondence.pdf | Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild | 成功 |
| 2021_Zhan_unite_unbalanced_feature_transport.pdf | Unbalanced Feature Transport for Exemplar-based Image Translation | 成功 |
