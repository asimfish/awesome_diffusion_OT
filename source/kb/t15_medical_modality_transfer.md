# T15 医学影像模态转换与 OT/SB/扩散

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「OT 引导跨域生成」（博客方向二）在医学影像的垂直落地：MRI↔CT、多序列 MRI 合成、低剂量 CT/PET 去噪、跨设备协调、病理染色转换。医学场景把 OT/SB 的理论卖点（不从高斯噪声出发、直连两个分布、成本函数可注入先验）变成了硬性临床需求（解剖一致、少数据可训、剂量学可验证），是检验「OT 先验是否真有用」的最严苛试验场。通用 I2I 方法（I2SB/UNSB/DDBM 本体）归 T13/T14，本笔记只收医学落地。

## 1. 核心问题与背景

医学影像模态转换要在「生成质量」之外满足三个医学特有约束：(i) **解剖一致性**——合成 CT 的骨骼边界错 1mm 就可能导致质子束射程计算错误，病理染色转换弄错细胞核形态会直接误导诊断；(ii) **少数据**——配对扫描（同一病人 MR+CT 严格配准）昂贵稀缺，多数场景只有未配对或弱配对（相邻切片）数据；(iii) **可信度**——临床采纳需要剂量学/下游任务级验证与不确定性量化，而非 FID。传统方案两极：CycleGAN 系（循环一致性约束弱、结构畸变）与条件扩散系（从高斯噪声出发、路径与任务无关、过平滑且慢）。OT/Schrödinger bridge/扩散桥提供第三条路：把转换建模为源模态分布到目标模态分布的（熵正则）最优传输，起点即源图像本身——传输成本天然惩罚不必要的改动，随机桥保留后验多样性，成本函数成为注入解剖/病理/物理先验的接口。2024–2026 年该思路已在放疗 sCT、PET/CT 去噪、虚拟染色、跨设备协调四条线上全面开花。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Harmonizing Optical Coherence Tomography Across Devices with Latent-Metric Schrödinger Bridges (LMSB, Wei et al., JHU) | 2025·NeurIPS | [P] | 指出 SB 的欧氏传输成本是医学解剖漂移根源，用可逆网络学 pullback 潜空间度量再训 SB，跨设备 OCT 协调保解剖 SOTA | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/08b60b4af0b8163b18553b15f5ce25d2-Abstract-Conference.html) / [OpenReview](https://openreview.net/forum?id=QU1SArYwKB) |
| ⭐ Anatomy-Conserving Unpaired CBCT-to-CT Translation via Schrödinger Bridge (ACSB, Shi et al.) | 2025·MICCAI | [P] | 熵正则 OT 解耦「模态伪影 vs 解剖」，AC-ViT 多尺度解剖先验 + 频率感知优化，无配对 CBCT→CT 跨部位泛化 | [MICCAI OA](https://papers.miccai.org/miccai-2025/paper/5303_paper.pdf) / [DOI](https://doi.org/10.1007/978-3-032-04965-0_5) |
| ⭐ Self-Consistent Recursive Diffusion Bridge for Medical Image Translation (SelfRDB, Arslan et al., Çukur 组) | 2025·Medical Image Analysis | [P] | 医学定制扩散桥：端点方差单调递增的噪声调度（软先验、抗测量噪声）+ 自洽递归采样，多对比 MRI 与 MRI↔CT SOTA | [DOI](https://doi.org/10.1016/j.media.2025.103747) / [arXiv](https://arxiv.org/abs/2405.06789) |
| ⭐ Diffusion Schrödinger Bridge Models for High-Quality MR-to-CT Synthesis for Proton Treatment Planning (DSBM, Li et al., PSI) | 2025·Medical Physics（arXiv 2024） | [P] | 首个用 DSBM 做质子放疗 sCT 并做剂量学级验证：46/77 对小数据训练，MAE 与骨 Dice 全面优于条件扩散，1%/1mm gamma 95.9–97.9%，NFE 大幅减少 | [DOI](https://doi.org/10.1002/mp.17898) / [arXiv](https://arxiv.org/abs/2404.11741) |
| ⭐ OT-StainNet: Optimal Transport Driven Semantic Matching for Weakly Paired H&E-to-IHC Stain Transfer (Guan et al.) | 2025·AAAI | [P] | 用 OT 在特征空间为弱配对（相邻切片错位）的 H&E–IHC 建立语义对应，把 OT 匹配变成监督信号驱动预训练扩散 LoRA 微调 | [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/32329) |
| Optimal Transport Driven CycleGAN for Unsupervised Learning in Inverse Problems (Sim, Oh, Kim, Jung & Ye) | 2020·SIAM J. Imaging Sciences | [P] | 奠基旧文：从 Kantorovich 对偶 + PLS 传输成本严格推导 cycleGAN 家族，前向算子知识可化简架构；统一无监督加速 MRI、低剂量 CT、显微镜超分 | [DOI](https://doi.org/10.1137/20M1317992) / [arXiv](https://arxiv.org/abs/1909.12116) |
| Implicit Image-to-Image Schrödinger Bridge for Image Restoration (I³SB, Wang et al., MGH) | 2024–2025·arXiv（v3 标注刊于 Pattern Recognition，DOI 未核验） | [R] | 把 I2SB 采样改成非马尔可夫（每步注入初始退化图），免重训复用预训练 I2SB，1/4 剂量腹部 CT 去噪与 4× 胸部 CT 超分少步数纹理更优 | [arXiv](https://arxiv.org/abs/2403.06069) |
| PET Image Denoising based on Diffusion Schrödinger Bridge Model | 2024·IEEE NSS/MIC/RTSD | [P] | 低剂量 PET 去噪：从低剂量 PET（而非高斯）起步的 DSBM + 解剖 MR 先验，避免条件扩散的过度增强 SNR，对未见数据稳健 | [DOI](https://doi.org/10.1109/NSS/MIC/RTSD57108.2024.10657633) |
| Fully Guided Neural Schrödinger Bridge for Brain MR Image Synthesis (FGSB, Yang et al., 汉阳大) | 2025·arXiv | [R] | 极少配对数据（2 个受试者）下的多序列 MRI 合成：两阶段生成-训练迭代 + 互信息一致性；可注入病灶 mask 先验保病灶 | [arXiv](https://arxiv.org/abs/2501.14171) |
| Heterogeneity-Adaptive Diffusion Schrödinger Bridge for PET-Guided Whole-Body MRI Translation (HA-DSB) | 2026·arXiv | [R] | 全身 MR 序列转换：VLM 区域上下文嵌入应对全身异质性，PET 代谢先验双阶段（前向噪声调制 + 反向注意力放大）保病灶保真 | [arXiv](https://arxiv.org/abs/2607.07401) |
| Weakly Supervised Virtual Immunohistochemistry Staining via Schrödinger Bridge (StainSB, Qiu et al.) | 2024·IEEE BIBM | [P] | 首批 SB 虚拟染色：区域颜色状态损失把病理相似性注入 H&E→IHC 生成，聚合策略平衡质量与病理一致性 | [DOI](https://doi.org/10.1109/BIBM62325.2024.10822509) |
| Topology-aware Diffusion Schrödinger Bridge for Unpaired H&E-to-IHC Stain Translation (TDSB) | 2026·IEEE JBHI | [P] | 把 UNSB 引入组织病理并修其二病：拓扑引导模块保腺体/细胞拓扑，双域自适应 patch-NCE 学 IHC 染色表征；7 个转换任务 SOTA + 病理医生评估 | [DOI](https://doi.org/10.1109/JBHI.2026.3668658) |
| PASB: Pathology-aware Schrödinger Bridge for Virtual Immunohistochemical Staining (Qiu et al.) | 2026·Medical Image Analysis | [P] | StainSB 升级：约束驱动对齐学习（高层病理语义监督）+ 相似度动态路径修正，下游诊断任务上接近真实 IHC | [DOI](https://doi.org/10.1016/j.media.2025.103869) |
| Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality (MOTFM, Yazdani et al.) | 2025·MICCAI | [P] | OT flow matching 进医学：直线路径少步采样，2D 超声/3D MRI、类别/掩码条件通吃，10 步优于 50 步 DDPM | [MICCAI OA](https://papers.miccai.org/miccai-2025/0343-Paper1056.html) / [arXiv](https://arxiv.org/abs/2503.00266) |
| Path and Bone-Contour Regularized Unpaired MRI-to-CT Translation (PaBoT, Sun et al.) | 2025·Computerized Medical Imaging and Graphics | [P] | 无配对 MRI→CT：潜空间 neural-ODE 流 + 最短传输路径正则（OT 味的路径长度最小化）+ 骨轮廓引导，下游骨分割保真最好 | [DOI](https://doi.org/10.1016/j.compmedimag.2025.102656) |

表外相邻工作（正文引用）：**Diffusion Bridge Models for 3D Medical Image Translation**（T1w↔DTI-FA 3D 桥，自述 EMBC 2025 未在 IEEE Xplore 核验，[arXiv 2504.15267](https://arxiv.org/abs/2504.15267)，[R]）；**USIGAN**（不平衡 OT 一致性挖掘用于弱配对虚拟染色，[arXiv 2507.05843](https://arxiv.org/abs/2507.05843)，[R]）；**PESB**（投影嵌入 SB 做稀疏视角 CT 重建，SPIE Medical Imaging 2025，[DOI](https://doi.org/10.1117/12.3048484)，[P]）与其扩展 **PEDB**（[arXiv 2510.22605](https://arxiv.org/abs/2510.22605)，[R]）；**DoseBridge**（DDBM 做 CT→质子剂量预测，[arXiv 2608.10173](https://arxiv.org/abs/2608.10173)，[R]）；**Fourier Diffusion GSR-PET**（按 MTF/NPS 物理谱构造桥端点做 PET 超分，[arXiv 2502.15055](https://arxiv.org/abs/2502.15055)，[R]）；**SynthRAD2025 挑战赛 FM 方案**（3D FM 做 MRI/CBCT→sCT，[arXiv 2510.04823](https://arxiv.org/abs/2510.04823)，[R]）。非 OT 背景基线：**SynDiff**（对抗扩散无监督医学翻译，IEEE TMI 2023，[DOI](https://doi.org/10.1109/TMI.2023.3290149)，[P]）；通用方法本体 **I2SB**（ICML 2023）与 **UNSB**（ICLR 2024）见 T13/T14。

## 3. 方法演进脉络

**前史（2018–2020，OT 作为无监督医学逆问题的理论骨架）**：Ye 组的 OT-CycleGAN 从 Kantorovich 对偶出发，证明「PLS 数据保真 + 深度逆路径惩罚」作为传输成本时 cycleGAN 架构可被严格推导，且前向算子（欠采样 MRI、CT 投影）已知时可砍掉一半生成器——OT 在此是理论解释器与架构化简器，应用即加速 MRI/低剂量 CT。这条「把物理前向模型编进传输成本」的线至今仍在 PESB/PEDB 的投影嵌入桥中延续。

**转折（2023–2024，从条件扩散到桥）**：SynDiff（TMI 2023）代表的对抗/条件扩散虽然质量高，但从高斯噪声出发：源图像只是旁路条件，路径与任务无关，采样慢且易过平滑。I2SB（ICML 2023，见 T13）给出可扩展的图像级 SB 后，医学界迅速跟进「起点=退化/源图像」范式：I³SB 把采样非马尔可夫化实现免重训少步 CT 修复；PET-DSBM 从低剂量 PET 起步并挂 MR 解剖先验；PSI 的 DSBM 用 46 对头颈 MR-CT 训练即超越条件扩散，并首次把评测拉到质子剂量学层面（gamma 通过率、剂量指数），确立「桥 = 少数据 + 少步数 + 剂量学可用」的临床叙事。SelfRDB 则针对医学测量噪声重设计桥本身：端点方差不归零（软先验）+ 递归自洽采样。

**分化（2025–2026，医学约束显式进入传输问题）**：三条支线并行。(a) **成本/度量设计**：LMSB 直击要害——SB 保解剖失败源于欧氏传输成本不度量解剖距离，学一个可逆网络把图像拉到「同解剖距离小」的潜度量空间再训 SB，是「成本函数即医学先验」的最干净证据；ACSB 用熵正则 OT 加 AC-ViT/频率约束在无配对 CBCT→CT 上落地同一哲学。(b) **OT 作弱监督匹配器**：病理染色线从 StainSB（区域颜色损失）到 OT-StainNet（OT 语义匹配解决相邻切片错位，把弱配对变成可用监督）、USIGAN（不平衡 OT 处理一对多与不完全匹配）、TDSB/PASB（拓扑/病理语义护栏），OT 的角色从生成骨架扩展到「对应关系发现」。(c) **流匹配轻量化**：MOTFM/PaBoT/SynthRAD 挑战方案把 OT-FM 的直线路径先验带进医学合成与翻译，换取 5–10 步推理；HA-DSB 则代表桥与多模态先验（VLM 区域嵌入、PET 代谢图）的融合方向。整体趋势：从「借用通用桥」走向「按医学约束重设计传输成本、端点分布与监督信号」，评测从像素指标走向剂量学与下游诊断任务。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: 间接但有清晰接口。I³SB 是医学界最接近的实例——不动预训练 I2SB 权重，只改采样过程（非马尔可夫注入源图像）就换来少步数高保真，说明「推理期轨迹干预」在医学桥上可行且需求强烈（临床要快）。FGSB 的生成期病灶引导、HA-DSB 的反向过程 PET 注意力放大，本质都是推理期把先验塞进已定桥动力学。可迁移问题：给任意已训练的医学条件扩散模型做推理期 OT 重排/引导，能否免重训消除解剖漂移？
- 方向二（OT 引导跨域生成）: 本课题就是方向二的落地样本库，且给出三个通用启示。(1) **成本函数是先验注入口**：LMSB 证明换度量（潜空间 pullback）直接决定解剖保真，比在网络里加模块更本质；(2) **OT 匹配可当监督**：OT-StainNet/USIGAN 用（不平衡）OT 从弱配对数据里挖对应关系，把「无监督」问题降级为「弱监督」；(3) **医学评测协议**（剂量学 gamma、下游分割/诊断一致性）为「OT 引导是否保语义」提供了比 FID 硬得多的可量化判据，博客方向二的实验设计可直接借用。

## 5. 开放问题与可发论文的切入点

1. **解剖等价类上的传输成本理论**：LMSB 的潜度量靠数据驱动学习，没有解剖不变性保证。可做：用配准形变场/分割一致性显式定义解剖等价类，在商空间上定义传输成本并证明「成本在等价类内为零 ⇒ SB 最优解只移动外观分量」；实验在 OCT 跨设备 + 多对比 MRI 上对照 LMSB，验证解剖漂移下界。
2. **3D 体积一致的医学桥**：现有 SB/桥方法几乎全是 2D 切片级（DSBM 用 2.5D 缓解），层间闪烁直接伤害剂量计算。做 slice-耦合 SB（相邻切片共享桥随机性 + 层间 OT 正则）或 latent 3D 桥，在 SynthRAD2025 上用 mDice/HD95 + 光子/质子 gamma 全指标验证——该 benchmark 目前无任何 SB 方法参赛，空位明显。
3. **任务感知传输成本（剂量学 OT）**：DSBM 只在事后做剂量学评测。把 HU→阻止本领的物理映射误差、或 OAR 区域加权直接写进传输成本/桥的漂移项，端到端优化 gamma 通过率而非 MAE；可证明加权成本下桥的存在唯一性，实验对照像素成本版本的剂量误差分布。
4. **桥后验的不确定性量化与临床可信度**：SB/桥天然支持多次采样出后验样本（I³SB、Fourier-PET 已用），但没人校准「voxel 方差 ↔ 真实误差」。做 conformal 校准的置信图，并验证其能否筛出解剖幻觉区域（如假骨、假病灶）——直接回应「可信度」这一医学落地最大阻力，方法量小、临床价值高。
5. **统一 med-bridge benchmark**：各工作数据划分互不相同（SelfRDB 用 IXI/BraTS/pelvis，ACSB 用私有 CBCT，染色各用 BCI/MIST），SB vs FM vs 条件扩散从未同场对照。搭一个覆盖 SynthRAD2025 + Mayo LDCT + BCI 的开源评测（像素/几何/剂量学/下游任务四层指标），本身可发 benchmark 论文并为后续方法文章供弹药。

## 6. 代码与资源

- [LMSB](https://github.com/Shuwen-Wei/lmsb) — NeurIPS 2025 OCT 协调官方实现
- [ACSB](https://github.com/Lalala-iks/ACSB) — MICCAI 2025 无配对 CBCT→CT
- [SelfRDB](https://github.com/icon-lab/SelfRDB) — Çukur 组医学扩散桥（多对比 MRI/MRI-CT）
- [MOTFM](https://github.com/milad1378yz/MOTFM) — 医学 OT flow matching，含 checkpoint 与合成数据集
- [HA-DSB](https://github.com/xyw-medical-research/HADSB) — PET 引导全身 MRI 转换
- [PaBoT](https://github.com/kennysyp/PaBoT) — 骨轮廓正则无配对 MRI→CT
- 通用底座：[I2SB](https://github.com/NVlabs/I2SB)、[UNSB](https://github.com/cyclomon/UNSB)（详见 T13/T14）
- 数据集/benchmark：[SynthRAD2025](https://synthrad2025.grand-challenge.org/)（890 MRI-CT + 1472 CBCT-CT，头颈/胸/腹，5 中心；11 指标含光子/质子剂量学，[数据 Zenodo](https://zenodo.org/records/14918089)、[指标代码](https://github.com/SynthRAD2025/metrics)、[数据集论文 Medical Physics](https://doi.org/10.1002/mp.17981)、[挑战报告 arXiv](https://arxiv.org/abs/2605.13555)）；SynthRAD2023（脑+盆腔，同系列首届）；[BCI](https://github.com/bupt-ai-cz/BCI)（H&E→HER2 IHC 配对 benchmark）与 MIST（弱配对 H&E-IHC 四染色）；BraTS / IXI（多对比 MRI 合成标准库）；Gold Atlas（盆腔 MR-CT）；Mayo/AAPM Low Dose CT Grand Challenge（1/4 剂量 CT）；ADNI（T1w↔DTI 桥实验用）
- 评测协议速查：图像层 MAE(HU)/PSNR/MS-SSIM → 几何层 mDice/HD95（骨与器官轮廓）→ 剂量学层 光子/质子 dose MAE、DVH、gamma（1%/1mm、2%/2mm、3%/3mm）→ 下游层 分割/分类/病理医生打分（TDSB、PASB 的临床评估范式）

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2025_Wei_LMSB_OCT_harmonization.pdf | Harmonizing OCT Across Devices with Latent-Metric Schrödinger Bridges (NeurIPS 2025) | 成功 |
| 2025_Shi_ACSB_CBCT_to_CT.pdf | Anatomy-Conserving Unpaired CBCT-to-CT Translation via Schrödinger Bridge (MICCAI 2025) | 成功 |
| 2025_Arslan_SelfRDB_medical_translation.pdf | Self-Consistent Recursive Diffusion Bridge for Medical Image Translation | 成功 |
| 2024_Li_DSBM_MR_to_CT_proton.pdf | Diffusion Schrödinger Bridge Models for High-Quality MR-to-CT Synthesis (proton planning) | 成功 |
| 2025_Wang_I3SB_image_restoration.pdf | Implicit Image-to-Image Schrödinger Bridge for Image Restoration | 成功 |
| 2025_Yang_FGSB_brain_MR_synthesis.pdf | Fully Guided Neural Schrödinger Bridge for Brain MR Image Synthesis | 成功 |
| 2025_Yazdani_MOTFM_medical_flow_matching.pdf | Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality (MICCAI 2025) | 成功 |
| 2020_Sim_OT_cycleGAN_inverse_problems.pdf | Optimal Transport Driven CycleGAN for Unsupervised Learning in Inverse Problems | 成功 |
