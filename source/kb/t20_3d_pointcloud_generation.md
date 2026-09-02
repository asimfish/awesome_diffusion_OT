# T20 3D/点云/几何生成中的 OT 与流

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题把「扩散×OT」从图像域延伸到几何域。点云/网格/3D Gaussian 本身就是离散测度，OT 在这里同时扮演三重角色：训练耦合（flow matching 的 noise-data 配对）、监督损失（EMD/Chamfer 及其修正）、表示结构化工具（把无序图元整理成规则网格）。这是全景中「数据即测度」最字面成立的方向。
> 边界: 分子构象归 T21；视频归 T19。equivariant OT flow（Klein 等、EquiFM）只作脉络交代，不在本表收录。

## 1. 核心问题与背景

点云是从曲面采样的无序点集，其置换不变性意味着生成模型实际工作在经验测度空间上：均匀离散测度间的 EMD 就是 W1/W2 距离，Chamfer Distance (CD) 是它的贪心松弛。由此产生三个层面的核心问题。其一，损失层面：CD 便宜但对密度失衡与离群点不敏感，EMD 忠实但 O(n³) 且要求等点数，如何构造介于两者之间、可微且可扩展的度量（DCD/HyperCD/InfoCD 谱系）。其二，耦合层面：flow matching/rectified flow 成为 2024-2026 年 3D 生成主干后，噪声-数据如何配对直接决定轨迹交叉程度与采样步数；点云的特殊性在于「形状内点级 OT」（置换求解）与「批级 minibatch OT」两个粒度并存，且 exact assignment 在 n~10³-10⁴ 时尚可精确解，这与图像域截然不同。其三，表示层面：3DGS/隐式场是非结构化参数集，OT 可把它们规整成扩散骨干可消化的网格（GaussianCube），或在 Gaussian 混合几何（Bures-Wasserstein/MW2）上直接比较与压缩资产。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Not-So-Optimal Transport Flows for 3D Point Cloud Generation (Hui et al.) | 2025·ICLR | [P] | 证明 equivariant/在线 OT 耦合在大点云上失效且完全拉直反而让 t≈0 处的场更难学，提出离线 superset OT 预计算 + 与独立耦合混合的 hybrid coupling，ShapeNet 无条件生成与补全双 SOTA | [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/f4dcb743e41af10d860562367a564bcd-Paper-Conference.pdf) / [arXiv](https://arxiv.org/abs/2502.12456) |
| ⭐ Wasserstein Flow Matching: Generative Modeling over Families of Distributions (Haviv et al.) | 2025·ICML | [P] | 把 FM 提升到测度空间：每个样本本身是一个分布（点云/Gaussian），沿 Wasserstein 测地线定义条件流，Gaussian 族用闭式 Bures-W 路径、点云用 entropic OT 估计；首个高维「分布的分布」生成器 | [PMLR](https://proceedings.mlr.press/v267/haviv25a.html) |
| ⭐ Unpaired Point Cloud Completion via Unbalanced Optimal Transport (UOT-UPC, Lee et al.) | 2025·ICML | [P] | 把无配对补全形式化为 UOT map 学习，marginal 松弛天然吸收类别不平衡；系统分析 cost 选择并论证 InfoCD 最适配 | [PMLR](https://proceedings.mlr.press/v267/lee25e.html) / [arXiv](https://arxiv.org/abs/2410.02671) |
| ⭐ GaussianCube: A Structured and Explicit Radiance Representation for 3D Generative Modeling (Zhang et al.) | 2024·NeurIPS | [P] | 定数化拟合 3DGS 后用 Jonker-Volgenant 线性指派（OT）把 Gaussians 摆进 N³ voxel 网格，使标准 3D U-Net 扩散直接可用；OT 在此充当「表示结构化」角色 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/b0b750c4189f19d0cd71375e9e17f83f-Paper-Conference.pdf) / [arXiv](https://arxiv.org/abs/2403.19655) |
| ⭐ SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis (Go et al.) | 2025·CVPR | [P] | 多视角 rectified flow 在 latent 空间联合生成图像/深度/相机位姿，经前馈 GSDecoder 输出 3DGS；training-free 反演/补绘统一生成与编辑 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Go_SplatFlow_Multi-View_Rectified_Flow_Model_for_3D_Gaussian_Splatting_Synthesis_CVPR_2025_paper.html) |
| Gaussian Herding across Pens (GHAP): An Optimal Transport Perspective on Global Gaussian Reduction for 3DGS | 2025·NeurIPS | [P] | 把 3DGS 压缩视为全局 Gaussian 混合约简，最小化 composite transport divergence，10% 图元几乎无损渲染 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/e79574cc3355e831cc276c845605ed72-Paper-Conference.pdf) |
| WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians (Kotovenko et al.) | 2024·ECCV | [P] | 直接用 entropic W2/Sinkhorn 在 Gaussian 参数分布间做显式匹配实现场景级风格迁移，零训练、纯优化 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03174.pdf) |
| Fast Point Cloud Generation with Straight Flows (PSF, Wu et al.) | 2023·CVPR | [P] | 把 rectified flow 的 reflow+蒸馏引入点云扩散，实现一步生成；3D 域「轨迹拉直」的开山之作 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Fast_Point_Cloud_Generation_With_Straight_Flows_CVPR_2023_paper.html) |
| Density-aware Chamfer Distance (DCD, Wu et al.) | 2021·NeurIPS | [P] | 指出 CD 密度盲区与 EMD 全局主导的双重缺陷，提出有界、密度敏感的折中度量并可作训练损失 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/file/f3bd5ad57c8389a8a1a541a76be463bf-Paper.pdf) |
| InfoCD: A Contrastive Chamfer Distance Loss for Point Cloud Completion (Lin et al.) | 2023·NeurIPS | [P] | 对比学习正则化 CD，等价于最大化底层曲面互信息下界；后被 UOT-UPC 选为最优 cost | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2ea1943896474b7cd9796b93e526f6f-Abstract.html) |
| TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models (Li et al.) | 2025·arXiv | [R] | 1.5B rectified flow transformer + SDF-VAE 的图生 3D 基础模型，2M 高质量样本；RF 线性轨迹成为 3D 资产工业界主干的代表 | [arXiv](https://arxiv.org/abs/2502.06608) |
| Neural Geometry Image-Based Representations with Optimal Transport | 2025·arXiv | [R] | Ricci flow 共形参数化后用 OT 校正面积畸变，得到保面积 geometry image（UV 域均匀采样），支持单趟重建与连续 LoD | [arXiv](https://arxiv.org/abs/2511.18679) |
| Texture Mapping via Optimal Mass Transport (Dominitz & Tannenbaum) | 2010·IEEE TVCG | [P] | 纹理/UV×OT 的奠基工作：共形初始化后经 OT 梯度流得到保面积贴图，最小化质量意义下的角度畸变 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2886313/) |
| Integrating Efficient Optimal Transport and Functional Maps for Unsupervised Shape Correspondence Learning (Le et al.) | 2024·CVPR | [P] | sliced Wasserstein 与 functional map 结合做无监督形状对应，为几何生成提供跨形状 OT 对齐 anchor | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Le_Integrating_Efficient_Optimal_Transport_and_Functional_Maps_For_Unsupervised_Shape_CVPR_2024_paper.html) |
| Improving Dynamic NeRFs with Optimal Transport | 2024·ICLR | [P] | 用 OT 约束时变隐式场（dynamic NeRF）的形变一致性，是 W 距离进入隐式表示优化的代表 | [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/568b6cc71889ea0b2aa74152ef9c28db-Abstract-Conference.html) |

正文另提及（不计入上表）：PointFlow（ICCV 2019，CNF+EMD/CD 评测奠基）、Achlioptas et al.（ICML 2018，EMD/CD 生成损失起点）、LION（NeurIPS 2022，latent 点扩散，[arXiv](https://arxiv.org/abs/2210.06978)）、Nguyen et al. 点集 sliced Wasserstein 距离（ICCV 2021）、HyperCD（双曲 CD）、RegGS（[arXiv 2507.08136](https://arxiv.org/abs/2507.08136)，MW2+Sinkhorn 做 3DGS 配准，[R]）、MMGS（[arXiv 2605.19304](https://arxiv.org/abs/2605.19304)，OT 聚合压缩 3DGS，[R]）。

## 3. 方法演进脉络

**度量线（2017-2023）**：Fan et al. 与 Achlioptas et al.（ICML 2018）确立 EMD/CD 作为点云生成的训练损失与评测（MMD/COV/1-NNA 的 CD/EMD 版本沿用至今）。EMD 即离散 W 距离但 O(n³)，于是出现一条「修 CD 使其逼近 OT 性质」的谱系：DCD（NeurIPS 2021）加密度感知与有界性，HyperCD 移到双曲空间，InfoCD（NeurIPS 2023）用对比正则把匹配点「摊开」以对齐分布。这条线在 2025 年被 UOT-UPC 反向消费：既然补全的 cost 决定 UOT map 质量，InfoCD 反而成为 neural OT 的最佳 cost——度量研究与传输研究在此合流。

**生成主干线（2019-2025）**：PointFlow（ICCV 2019）用 CNF 给出似然式点云生成；扩散接管后（Luo & Hu 2021、PVD、LION NeurIPS 2022），采样成本问题浮现。PSF（CVPR 2023）把 rectified flow 的 reflow+蒸馏搬进点云实现一步生成——「重训练式拉直」。2023-2024 年分子侧发展出 equivariant OT flow（Klein et al.；EquiFM，归 T21），通过置换求解使耦合对称化。NSOT（ICLR 2025）系统检验这条路在大点云（2048+ 点）上的可扩展性：在线 Hungarian 求解太贵、近似 OT 质量太差，且完全 OT 耦合把复杂度前移到 t≈0 的向量场；解法是离线在稠密 superset 上预计算一次 OT、训练时子采样继承配对，并与独立耦合混合。这一「not-so-optimal 更好学」的发现与图像域 minibatch OT-CFM（T08）的经验形成有趣对照。同期 WFM（ICML 2025）换了问题本身：不再把点云摊平成 R^{3n} 向量，而把每朵点云当作 Wasserstein 空间中的一个点，沿 W2 测地线做 FM，Gaussian 子流形用闭式 Bures-W 路径——「在测度空间上生成测度」。

**资产/表示线（2024-2026）**：3DGS 兴起后 OT 出现三种用法：GaussianCube（NeurIPS 2024）用线性指派把变长无序 Gaussians 规整进 voxel 网格喂给 3D U-Net 扩散；WaSt-3D（ECCV 2024）在推断期直接对两组 Gaussian 参数做 Sinkhorn 匹配实现零训练风格迁移；GHAP（NeurIPS 2025）与 MMGS 把压缩/聚合形式化为 Gaussian 混合的传输散度最小化，RegGS 用 MW2 做配准。大模型侧 TripoSG、SplatFlow（均 2025）确立 rectified flow 为 3D 资产生成的默认主干，但其「传输性」仅体现在直线概率路径，尚未利用几何域特有的 exact OT。UV/纹理线则从 Dominitz & Tannenbaum（2010）的保面积 OT 贴图，演进到 2025 年 OT-geometry-image 表示，把「参数化畸变校正」重新包装为神经压缩与重建问题。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: 强相关。NSOT 的核心证据——完全 OT 耦合虽拉直轨迹却让初始时刻向量场更难学——直接约束了「对齐应做到什么程度」；其离线 superset OT + 混合系数的做法本质是可调的轨迹对齐旋钮，且无需在线求解。PSF 代表「重训练式」拉直（reflow+蒸馏），恰是博客方向想绕开的成本基线。WaSt-3D 展示了几何域完全 training-free 的 OT 显式匹配范式（推断期 Sinkhorn 即所得），可视为轨迹对齐思想在参数空间的极端版本。
- 方向二（OT 引导跨域生成）: 强相关。UOT-UPC 是几何域最干净的实例：跨域（残缺→完整）映射直接学成 UOT map，marginal 松弛处理两域质量不守恒，cost 函数选择（InfoCD）决定语义对齐质量——这套「cost 设计 + unbalanced 松弛」框架可整体迁移到博客设想的跨域生成。WaSt-3D 的 scene-to-scene 传输和 Le et al. 的 OT 形状对应提供跨域 anchor 的构造方法；GaussianCube 说明 OT 还能充当跨模态生成管线中的「表示桥」。SplatFlow/TripoSG 的图像→3D 属条件生成，与 OT 引导仅间接相关。

## 5. 开放问题与可发论文的切入点

1. **混合耦合的定量理论**：NSOT 只给了 hybrid coupling 的经验配方。可证的目标：对 n 点经验测度，刻画耦合「OT 程度」（混合系数 λ）与 (a) 轨迹交叉率/曲率、(b) t≈0 处目标向量场 Lipschitz 常数之间的 trade-off，给出最优 λ 随 n 与形状复杂度的 scaling。实验：ShapeNet 上 sweep λ，测 Jacobian 范数谱与 NFE-质量 Pareto 前沿。
2. **去偏 Wasserstein 测地线的 FM**：WFM 用 entropic OT 近似测地线，存在 ε-偏置。改造：用 Sinkhorn divergence 的去偏 barycentric 映射构造条件路径，证明 ε→0 时学到的速度场一致收敛；实验对比固定 ε、退火 ε 与去偏版在点云 1-NNA 上的差异。
3. **补全中不平衡度 τ 的自适应估计**：UOT-UPC 的 mass 松弛参数目前全局固定。做一个由残缺率/遮挡几何预测 per-instance τ 的轻量模块（可用可见性先验监督），在类别失衡与变遮挡率基准上验证；理论上给出 τ 与缺失质量比的显式对应。
4. **Bures-Wasserstein 流匹配直接生成 3DGS 资产**：现状割裂——GaussianCube 只对位置做 assignment 后回到欧氏扩散，WFM 的 BW-FM 只处理单个 Gaussian 族，GHAP/RegGS 只做压缩/配准。空白点：把一个 3DGS 资产视为 Gaussian 混合测度，在 MW2/混合 Wasserstein 几何上直接做 flow matching（协方差沿 BW 测地线、分量间用 entropic 耦合），有望免除 voxel 化并天然支持变分量数。这是本子课题最有潜力的切入点。
5. **OT geometry image 上的潜空间扩散**：arXiv 2511.18679 的保面积 OT 参数化目前只用于压缩/LoD。把它作为规范 UV 域，在 geometry image 上跑 latent diffusion/FM（均匀采样消除极区过采样），并用 OT 对齐多视角纹理烘焙的接缝，可同时打通「网格生成」与「纹理 OT 对齐」两条线。

## 6. 代码与资源

- **NSOT**: 项目页 https://research.nvidia.com/labs/genair/not-so-ot-flow/
- **WFM**: 代码见 PMLR 论文页所附 WassersteinFlowMatching 仓库（JAX）
- **UOT-UPC**: https://github.com/LEETK99/UOT-UPC
- **PSF**: https://github.com/klightz/PSF
- **GaussianCube**: https://gaussiancube.github.io/
- **SplatFlow**: https://github.com/gohyojun15/SplatFlow
- **WaSt-3D**: https://compvis.github.io/wast3d/
- **GHAP**: https://github.com/DrunkenPoet/GHAP
- **DCD / InfoCD**: https://github.com/wutong16/Density_aware_Chamfer_Distance / https://github.com/Zhang-VISLab/NeurIPS2023-InfoCD
- **TripoSG**: https://github.com/VAST-AI-Research/TripoSG （权重在 HuggingFace VAST-AI/TripoSG）
- **库**: GeomLoss（大规模可微 Sinkhorn 点云损失，https://www.kernel-operations.io/geomloss/）、POT（https://pythonot.github.io/）、PyTorch3D（CD/EMD 实现）、TorchCFM（minibatch OT 耦合参考实现，https://github.com/atong01/conditional-flow-matching）
- **数据集/基准**: ShapeNet（生成：1-NNA/MMD/COV 的 CD 与 EMD 版）、PCN 与 MVP（补全）、OmniObject3D、Objaverse（资产级）、MVImgNet 与 DL3DV-7K（场景级 3DGS）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2025_Hui_not_so_optimal_transport_flows.pdf | Not-So-Optimal Transport Flows for 3D Point Cloud Generation | 成功（36.7MB，arXiv） |
| 2025_Haviv_wasserstein_flow_matching.pdf | Wasserstein Flow Matching: Generative Modeling over Families of Distributions | 成功（6.5MB，arXiv） |
| 2025_Lee_uot_upc_point_cloud_completion.pdf | Unpaired Point Cloud Completion via Unbalanced Optimal Transport | 成功（6.2MB，arXiv） |
| 2024_Zhang_gaussiancube_ot_structuring.pdf | GaussianCube: A Structured and Explicit Radiance Representation for 3D Generative Modeling | 成功（34.7MB，arXiv） |
| 2025_Go_splatflow_rectified_flow_3dgs.pdf | SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis | 成功（32.4MB，arXiv） |
| 2024_Kotovenko_wast3d_wasserstein_3d_gaussians.pdf | WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians | 成功（8.5MB，ECVA 官方 OA） |
| 2023_Wu_point_straight_flow.pdf | Fast Point Cloud Generation with Straight Flows | 成功（8.8MB，CVF 官方 OA） |

（7/7 全部通过校验：文件头 `%PDF` 且远大于 50KB）
