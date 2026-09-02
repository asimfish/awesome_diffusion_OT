# T17 风格迁移与域自适应中的 OT×扩散

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」在应用层的交汇点：OT 既作为颜色/纹理分布对齐的损失与感知度量（sliced Wasserstein 系），又作为扩散跨域生成的机制修正（DDIB latent 对齐、采样期引导），还作为域自适应/模型适配中特征分布对齐的正则。与 T16（跨域语义对应）互补：本课题关注**分布级**对齐，不关注点级/像素级语义对应。

## 1. 核心问题与背景

风格迁移的本质是**分布对齐**：把内容图的颜色/纹理特征分布推向风格参考的分布，同时保持内容结构。OT 恰好提供了带几何结构的分布距离与传输映射，因此从早期的 relaxed EMD 风格损失、sliced Wasserstein（SW）纹理损失，一路演化到扩散时代的采样期 SW 引导与 DDIB latent OT 对齐。域自适应（DA）/域泛化（DG）与之同构——源域与目标域特征分布的对齐长期以 OT 为主力工具；扩散模型的加入带来了新问题（预训练大模型如何低成本适配新域、如何用生成增广弥补目标域数据稀缺）与新机会（轨迹级分布分析、无须重训的引导式适配）。本子课题覆盖三层：损失/度量层（SW 风格损失、MS-SWD 色差）、扩散机制层（SW-Guidance、OT-ALD）、模型迁移层（WAT 微调、semi-dual OT 逐步域迁移、source-free DA）。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Multiscale Sliced Wasserstein Distances as Perceptual Color Difference Measures (MS-SWD) | 2024·ECCV | [P] | 多尺度 SW 距离做 training-free 感知色差度量，对非对齐图像对稳健，实证满足度量公理，可直接当图像/视频颜色迁移损失 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7025_ECCV_2024_paper.php) |
| ⭐ Color Conditional Generation with Sliced Wasserstein Guidance (SW-Guidance) | 2025·NeurIPS (spotlight) | [P] | 把可微 Sliced-1-Wasserstein 色彩距离塞进扩散采样循环做 training-free 颜色条件生成，胜过「先生成再色彩迁移」流水线 | [OpenReview](https://openreview.net/forum?id=r1Bx58M6It) |
| ⭐ Stochastic Interpolants for Revealing Stylistic Flows across the History of Art (Art-FM) | 2025·ICCV | [P] | 把艺术风格的历史演化建模为风格空间中的 OT 分布匹配，用 stochastic interpolants+DDIB 无配对对齐跨世纪艺术分布，并发布 65 万艺术品数据集 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Stochastic_Interpolants_for_Revealing_Stylistic_Flows_across_the_History_of_ICCV_2025_paper.pdf) |
| ⭐ OT-ALD: Aligning Latent Distributions with Optimal Transport for Accelerated Image-to-Image Translation | 2026·AAAI | [P] | 证明 DDIB 在有限 T 下两域 latent 分布错配必然导致翻译轨迹偏差，用显式 OT map 对齐 latent 后再反向去噪，平均提速 20.3%、FID 降 2.6 | [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/39886) |
| ⭐ Wasserstein-Aware Transfer: Class-Level Alignment for Robust Diffusion Model Adaptation (WAT) | 2026·AAAI | [P] | 分析扩散轨迹间 W 距离随 t 递减的规律，据此做源↔目标类级 OT 匹配指导扩散模型微调，并线性组合预训练/微调条件分支保知识 | [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/39365) |
| Color Transfer with Modulated Flows (ModFlows) | 2025·AAAI | [P] | 基于 rectified flow 的可逆 RGB 颜色迁移：在 OT plan 数据集上训练流+编码器预测流权重，新图像对零微调泛化，可处理 4K | [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/32470) |
| WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians | 2024·ECCV | [P] | 用熵正则 W2/EMD 直接匹配风格与内容场景的 3D 高斯分布，training-free 场景级 3DGS 风格迁移，把风格化从生成问题改写为显式分布匹配 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3174_ECCV_2024_paper.php) |
| GIST: Towards Photorealistic Style Transfer via Multiscale Geometric Representations | 2024·arXiv | [R] | 在小波/Contourlet 子带上用高斯假设下的闭式 W2 匹配做 training-free 照片级风格迁移，替代神经自编码框架 | [arXiv](https://arxiv.org/abs/2412.02214) |
| Scalable Motion Style Transfer with Constrained Diffusion Generation (KMCGs) | 2024·AAAI | [P] | 各风格域独立训练扩散模型，借 DDIB（熵正则 OT/SB 解释）桥接+关键帧流形约束梯度，可扩展到十种舞蹈动作风格 | [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/28889) |
| A Sliced Wasserstein Loss for Neural Texture Synthesis | 2021·CVPR（奠基） | [P] | 用 SWD（1D 投影排序闭式解）替代 Gram 矩阵作纹理损失，捕获完整特征分布而非二阶统计量 | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Heitz_A_Sliced_Wasserstein_Loss_for_Neural_Texture_Synthesis_CVPR_2021_paper.html) |
| Style Transfer by Relaxed Optimal Transport and Self-Similarity (STROTSS) | 2019·CVPR（奠基） | [P] | 用 relaxed EMD 定义风格损失+自相似保内容，OT 风格迁移的开山之作 | [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Kolkin_Style_Transfer_by_Relaxed_Optimal_Transport_and_Self-Similarity_CVPR_2019_paper.html) |
| Rethinking Flow-based Gradual Domain Adaptation via Semi-dual Optimal Transport | 2026·ICML | [A] | 用 semi-dual OT 重构 flow-based 逐步域自适应的中间域生成路径 | [OpenReview](https://openreview.net/forum?id=iqXzDUd36x) |
| Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing | 2025·CVPR | [P] | source-free 约束下用 OT 引导原型/特征传输，客户端自适应活体检测 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Optimal_Transport-Guided_Source-Free_Adaptation_for_Face_Anti-Spoofing_CVPR_2025_paper.html) |
| Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport | 2026·CVPR | [P] | 用 VLM 语义先验引导源原型与目标特征的 OT 对齐，source-free DA 新范式 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Han_Vision-Language_Model_Guided_Source-Free_Domain_Adaptation_via_Optimal_Transport_CVPR_2026_paper.html) |
| Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model (A2A-FM) | 2025·arXiv | [R] | 设计一个成本函数同时学所有条件分布对之间的 pairwise OT，支持连续条件的 all-to-all 风格/属性迁移，有无限样本极限收敛保证 | [arXiv](https://arxiv.org/abs/2504.03188) |

## 3. 方法演进脉络

**损失层（2019–2021，奠基）**：STROTSS（CVPR 2019）首次把风格损失从 Gram 矩阵换成 relaxed EMD，宣告「风格=特征分布、迁移=最优传输」的观点；Heitz 等（CVPR 2021）用 sliced Wasserstein 替代 Gram——1D 投影+排序即得闭式可微解，捕获完整分布且成本低。其更早的根源是 Pitié 等的迭代 1D 分布匹配颜色迁移算法，本身就是 SWD 的雏形。

**度量与新表示层（2024）**：分布匹配思想向两个方向外溢。其一是感知度量：MS-SWD（ECCV 2024）把 SW 升级为 CIELAB 空间的多尺度色差度量，training-free、对非对齐图像对稳健，反过来又能当颜色迁移的损失。其二是新表示：WaSt-3D（ECCV 2024）把熵正则 W2 从 VGG 特征空间搬到 3DGS 显式几何表示，风格化=两组 3D 高斯的分布匹配；GIST（arXiv 2024）在小波/Contourlet 子带用高斯闭式 W2 做照片级风格迁移。扩散侧此时以 DDIB（熵正则 OT/Schrödinger bridge 解释）为桥，KMCGs（AAAI 2024）借其做域独立训练的动作风格迁移。

**扩散机制层（2025）**：OT 与扩散采样正式融合。SW-Guidance（NeurIPS 2025 spotlight）把可微 SW-1 色彩距离的梯度直接注入去噪循环，training-free 实现颜色条件生成，解决「先生成后调色」语义失配问题；ModFlows（AAAI 2025，同组工作）用 rectified flow 学 OT plan 并以编码器泛化到新图像对；Art-FM（ICCV 2025）把风格演化整体建模为风格空间的 OT，用 stochastic interpolants 跨 500 年艺术分布对齐。

**模型迁移层（2026）**：走向理论化与适配。OT-ALD（AAAI 2026）给出 DDIB 有限 T latent 错配的误差定理，用显式 OT map 修正起点分布，兼得提速与降 FID；WAT（AAAI 2026）从扩散轨迹 W 距离随时间递减的观察出发，用类级 OT 匹配指导微调；DA 侧则有 semi-dual OT 逐步域迁移（ICML 2026）与 SFDA×OT（CVPR 2025/2026）。值得注意的空白：扩散增广式 DG/DA（如 FDS, WACV 2025；NOCDDA, IJCAI 2025）目前基本**不用 OT** 控制伪域分布，几何控制缺位。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **强关联**。SW-Guidance 是采样期 training-free 引导的标准实例——不动权重、只用 SW 梯度改写去噪轨迹；OT-ALD 在两条扩散轨迹的衔接点（latent 端点）做一次 OT 映射即消除轨迹偏差，两个预训练模型均无须重训；MS-SWD 提供了现成的可微 training-free 距离，可直接充当此类引导的势函数。
- 方向二（OT 引导跨域生成）: **强关联**。DDIB 系（KMCGs→OT-ALD）正是「跨域生成=两段扩散+OT 衔接」的机制化体现；Art-FM 把跨时代风格生成显式表述为分布间 OT；DA 侧的 semi-dual OT 逐步域迁移与 SFDA×OT 把 OT 对齐作为跨域适配的正则/引导，属于该方向在判别任务上的镜像。

## 5. 开放问题与可发论文的切入点

1. **把 MS-SWD 升级为扩散采样引导**：SW-Guidance 只用 RGB 像素分布上的 SW-1；换成 MS-SWD（CIELAB+多尺度金字塔）做 guidance 势函数，理论上更贴合人眼色差。实验：在 ContraStyles prompts + Unsplash 参考集上对比色彩保真（MS-SWD 指标）与语义一致（CLIP-T/CLIP-IQA），并测多尺度是否缓解 SW-1 引导在低分辨率 latent 上的噪声。
2. **DDIB latent 对齐的更优传输**：OT-ALD 用离散 OT map 修正起点；可改为（a）熵正则 map / Schrödinger bridge 并推导有限 T 下更紧的 W2 误差界，或（b）用 flow matching 学两 latent 分布间的连续 OT，摊销掉每批次求解。验证点：能否在更少去噪步数下保持 FID。
3. **给扩散增广式域泛化补上 OT 几何控制**：FDS/NOCDDA 等生成伪域时无分布距离约束。做法：在 latent 空间以 (sliced) W 距离约束伪域到源域的「距离带」（min-max：最大化多样性同时限制 transport 半径），或用 W-barycenter 插值构造中间域链（衔接 gradual DA 的 semi-dual OT 理论）。基准：PACS / DomainNet / OfficeHome。
4. **类级 OT 微调推广到不平衡/开集**：WAT 假设源类↔目标类近似双射；用 unbalanced OT / partial OT 处理类不平衡与开集（目标域新类无源对应），并把匹配从类质心推广到类条件分布间的 Bures-Wasserstein 距离。可直接在 WAT 的七个基准上做对照。
5. **SW 风格损失的理论刻画**：MS-SWD 只实证检验了度量公理；可证明多尺度 SW 在何种特征分布假设下与人眼色差单调一致（或给出反例），并研究其作为 style distillation 目标时对扩散蒸馏（一步生成器）风格保真的影响。

## 6. 代码与资源

- MS-SWD 官方实现: https://github.com/real-hjq/MS-SWD （另有 IQA-PyTorch 内置 `msswd` 学习版）
- SW-Guidance 官方实现: https://github.com/alobashev/sw-guidance （SD1.5/SDXL + DDIM，兼容 ControlNet）
- ModFlows 官方实现: https://github.com/maria-larchenko/modflows （HuggingFace 提供 checkpoints）
- Art-FM 代码+65 万艺术品数据集: https://github.com/CompVis/Art-fm
- WaSt-3D 项目页: https://compvis.github.io/wast3d/
- GIST 官方实现: https://github.com/renanrojasg/gist
- KMCGs 动作风格迁移: https://github.com/YIN95/ddst_motion （数据：100STYLE、AIST++）
- OT-RF（rectified flow 编辑的 OT 引导，边界参考）: https://github.com/marianlupascu/OT-RF
- 常用评测/数据：ContraStyles + Unsplash Lite（色彩条件生成）、PACS/DomainNet/OfficeHome（DA/DG）、POT 与 ott-jax（Sinkhorn/OT 求解器）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2024_He_msswd_color_difference.pdf | Multiscale Sliced Wasserstein Distances as Perceptual Color Difference Measures | 成功（ECVA 官方） |
| 2025_Lobashev_sw_guidance.pdf | Color Conditional Generation with Sliced Wasserstein Guidance | 成功（arXiv） |
| 2025_Ma_stylistic_flows_art.pdf | Stochastic Interpolants for Revealing Stylistic Flows across the History of Art | 成功（CVF 官方） |
| 2025_Wang_ot_ald_i2i.pdf | OT-ALD: Aligning Latent Distributions with Optimal Transport for Accelerated Image-to-Image Translation | 成功（arXiv） |
| 2026_Huang_wasserstein_aware_transfer.pdf | Wasserstein-Aware Transfer: Class-Level Alignment for Robust Diffusion Model Adaptation | 成功（AAAI OJS 官方） |
| 2025_Larchenko_modflows_color_transfer.pdf | Color Transfer with Modulated Flows | 成功（arXiv） |
| 2024_Kotovenko_wast3d_stylization.pdf | WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians | 成功（ECVA 官方） |
| 2021_Heitz_sliced_wasserstein_texture.pdf | A Sliced Wasserstein Loss for Neural Texture Synthesis | 成功（arXiv） |
