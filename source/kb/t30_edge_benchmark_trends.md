# T30 端侧部署、benchmark 与顶会趋势（博客落地场景：端侧图像生成）

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」全景的**落地与验收面**：少步生成（T09-T12 的算法产物）最终要在手机 NPU/笔记本 GPU 上跑起来，要用可信的评测协议证明质量没掉，还要用顶会趋势判断该往哪里投。它不覆盖具体加速算法（归 T09-T12），只回答"部署栈长什么样、怎么量、风往哪吹"。

## 1. 核心问题与背景

端侧图像生成受三重约束：**计算**（迭代采样×大 UNet/DiT，手机 SoC 上单步都昂贵）、**内存**（SD1.5 FP16 UNet 1.72GB、FLUX 12B 超 20GB，远超手机 DRAM 预算）、**算子/精度支持**（NPU 偏好静态图与 INT8/INT16，注意力与动态 shift 算子支持差）。工程解法是三条正交轴的组合：少步数（蒸馏/直线化，算法归 T09-T12）×模型压缩（量化/剪枝/小型化架构）×编译适配（kernel fusion、model-level tiling）。与 OT 的接口在于：少步化的可行性取决于轨迹几何（直线度/一致性），而新一代可部署模型（SD3、FLUX、SANA）的训练公式已经换成 flow matching/rectified flow——OT 耦合与轨迹拉直技术的"部署价值"因此系统性上升。评测是本课题第二主题：FID 本质是 Inception 特征上两个高斯拟合之间的 Wasserstein-2 距离（一个 OT 概念），但其正态假设与 Inception 偏差对扩散模型和少步模型系统性不公平，NFE-质量权衡曲线又缺乏统一协议——"怎么量"本身就是可发论文的开放问题。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds (Li et al.) | 2023 · NeurIPS | [P] | 端侧 T2I 开山：高效 UNet（冗余块识别+数据蒸馏压 VAE decoder）+ CFG 正则化步蒸馏，iPhone 14 Pro 上 8 步 <2s，FID/CLIP 超 SD1.5-50 步 | [NeurIPS](https://papers.neurips.cc/paper_files/paper/2023/hash/41bcc9d3bddd9c90e1f44b29e26d97ff-Abstract-Conference.html) |
| ⭐ MobileDiffusion: Instant Text-to-Image Generation on Mobile Devices (Zhao et al.) | 2024 · ECCV | [P] | 首个系统性的移动端扩散**架构设计空间研究**（<400M UNet）+ diffusion-GAN 一步采样兼容下游任务，iPhone 15 Pro 0.2s/512² | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07923.pdf) |
| EdgeFusion: On-Device Text-to-Image Generation (Castells et al.) | 2024 · arXiv（CVPR-W 系） | [R] | NPU 全栈工程样本：BK-SDM-Tiny + 改进 LCM 蒸馏 + 合成数据，W8/A16(UNet INT8 权重+INT16 激活)混合精度 + model-level tiling + kernel fusion，三星 Exynos 2400 NPU 2 步 <1s | [arXiv](https://arxiv.org/abs/2404.11925) |
| SDXS: Real-Time One-Step Latent Diffusion Models with Image Conditions (Song et al.) | 2024 · arXiv | [R] | UNet+VAE 双小型化 + 特征匹配/分数蒸馏的一步训练（显式做轨迹拉直 straightening），512² 达 100 FPS；ControlNet 蒸馏支持实时 image-to-image | [arXiv](https://arxiv.org/abs/2403.16627) |
| MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models... (Zhao et al.) | 2024 · ECCV | [P] | **少步模型专用 PTQ**：发现 1 步 SDXL-Turbo 量化瓶颈在文本嵌入 BOS 离群值；BOS-aware 量化 + 指标解耦敏感度分析 + 整数规划配比特，W4A8 仅 +0.5 FID（基线全崩） | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02212.pdf) |
| BitsFusion: 1.99 bits Weight Quantization of Diffusion Model (Sui et al.) | 2024 · NeurIPS | [P] | QAT 极限压缩：逐层最优比特分配 + 量化初始化 + 两阶段蒸馏训练，SD1.5 UNet 1.72GB→219MB（1.99 bit）且 TIFA/GenEval/人评反超全精度 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c64bc3f7796d31caa7c3e6b969bf7da-Abstract-Conference.html) |
| ⭐ SVDQuant: Absorbing Outliers by Low-Rank Component for 4-Bit Diffusion Models (Li et al.) | 2025 · ICLR | [P] | W4A4 量化范式：smoothing 把激活离群值移到权重，SVD 低秩分支吸收权重离群值；与 Nunchaku 引擎 kernel 融合 co-design，12B FLUX 跑进 16GB 笔记本 4090（3×提速）且免重量化支持 LoRA | [ICLR proc.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html) |
| ⭐ SnapGen: Taming High-Resolution Text-to-Image Models for Mobile Devices... (Chen et al.) | 2025 · CVPR Highlight | [P] | 端侧模型**从头训练**新范式：379M UNet 宏/微架构搜索 + 跨架构多级蒸馏（教师 SD3.5-Large）+ 对抗步蒸馏，iPhone 16 Pro-Max 1024² ~1.4s，GenEval 0.66 超 SDXL（7×大） | [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_SnapGen_Taming_High-Resolution_Text-to-Image_Models_for_Mobile_Devices_with_Efficient_CVPR_2025_paper.pdf) |
| SANA: Efficient High-Resolution Text-to-Image Synthesis with Linear Diffusion Transformers (Xie et al.) | 2025 · ICLR Oral | [A] | 可部署 DiT 路线：32× 深压缩 AE + linear attention DiT + 小 LLM 文本编码器 + FM 训练/Flow-DPM-Solver，0.6B 模型笔记本 GPU <1s/1024²，配 SVDQuant 4-bit 跑进 8GB | [OpenReview](https://openreview.net/forum?id=N8Oj1XhtYZ) |
| ⭐ Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models (Stein et al.) | 2023 · NeurIPS | [P] | 最大规模人评心理物理实验：**没有任何现有指标与人评强相关**；Inception-V3 特征系统性压低扩散模型排名；建议全面换用 DINOv2-ViT-L/14 特征算 FD | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0bc795afae289ed465a65a3b4b1f4eb7-Abstract-Conference.html) |
| Rethinking FID: Towards a Better Evaluation Metric for Image Generation (Jayasumana et al.) | 2024 · CVPR Highlight | [P] | 指出 FID 三宗罪：正态假设不成立、样本复杂度差、与人评矛盾（无法反映 T2I 迭代改进）；提出 CMMD = CLIP 嵌入 + Gaussian RBF 核 MMD（无分布假设、无偏、样本高效） | [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.pdf) |
| GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment (Ghosh et al.) | 2023 · NeurIPS D&B | [P] | 用目标检测器做组合能力评测（共现/计数/颜色/位置），实例级可解释、与人评强一致；已成端侧论文标配指标（SnapGen/SANA 均报告） | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a3bf71c7c63f0c3bcb7ff67c67b1e7b1-Abstract-Datasets_and_Benchmarks.html) |
| Human Preference Score v2 (HPS v2; Wu et al.) | 2023 · arXiv | [R] | 79.8 万人类偏好对训练的偏好模型 + HPD v2 基准，作为可扩展的"人评代理"；MobileDiffusion 等端侧工作用其验证主观质量 | [arXiv](https://arxiv.org/abs/2306.09341) |

⭐=必读 5 篇。表外提及：SANA-Sprint（ICCV 2025 Highlight，SANA 的连续时间一致性蒸馏一步版，算法归 T10）、SANA 1.5（ICML 2025）、SD-Turbo/UFOGen/LCM（一步/少步算法，归 T10-T12）、Q-Diffusion/PTQ4DM（多步模型早期 PTQ，被 MixDQ 对比）、DC-AE（ICLR 2025，SANA 的 32× AE 组件）。

## 3. 方法演进脉络

**部署线（2023→2026）**：SnapFusion（NeurIPS'23）确立"架构压缩+步蒸馏"双轴范式，把 SD1.5 压进 iPhone 2 秒。2024 年三条支线分化：MobileDiffusion 做系统性架构设计空间研究（UNet 每个组件的 FLOPs/质量权衡），把一步采样（diffusion-GAN）带到 0.2s；EdgeFusion 展示 **NPU 特有工程栈**（混合精度 PTQ、model-level tiling 省 DRAM、kernel fusion，Exynos 17K MAC NPU）；SDXS 用特征匹配+分数蒸馏做一步小模型并显式引入**轨迹拉直**。量化支线并行推进：MixDQ 首次指出**少步模型量化质的不同**——误差无多步迭代可摊销，瓶颈集中在文本嵌入 BOS 离群值；BitsFusion 用 QAT 探到 1.99 bit 极限；SVDQuant（ICLR'25）把 W4A4 做成范式（低秩分支吸收离群值+推理引擎 co-design），使 12B FLUX 上笔记本。2025 转折点是**从"压缩现有模型"到"为端侧从头设计"**：SnapGen 从头训 379M 模型配跨架构蒸馏拿下手机 1024²；SANA 换 DiT 底座（linear attention + 32× AE + FM 公式），0.6B 匹敌 FLUX-12B。趋势判断：FM/RF 公式已是新端侧模型默认训练目标（SD3→SANA→FLUX 生态），"轨迹几何→少步可行性→端侧延迟"的因果链使 OT/直线化研究直接服务部署。

**评测线**：FID（本质是 Inception 特征上高斯拟合的 W2 距离，即 Bures-Wasserstein——评测本身就是 OT 的应用）长期主导。2023 年 Stein et al. 用最大规模人评实验证明**没有指标与人评强相关**且 Inception 对扩散模型不公；2024 年 CMMD 进一步攻击正态假设与样本复杂度，提出 CLIP+MMD 替代；社区共识转向组合协议：分布级（FD-DINOv2/CMMD）+ 组合级（GenEval、DPG-Bench）+ 偏好级（HPSv2/ImageReward/人评）+ 效率维（NFE、参数量、端上延迟/内存）。少步评测的**协议缺陷**依旧：NFE-质量曲线各家采样器/CFG/seed 不统一，1-4 步区间 FID 对蒸馏模型的"过平滑"不敏感；轨迹直线度 S(Z)（T09）与曲率则从训练正则演化为**部署可行性的几何指标**，但尚未进入标准评测协议。

**顶会趋势粗统计**（检索日 2026-08-14；两套口径，均只计已接收论文）：

口径 A——OpenReview 全文提及级（term 匹配 title/abstract/keywords 等全部字段；accepted 判定=venue 字段含 poster/spotlight/oral/regular/Accept；API limit=1000 未触顶）：

| 关键词（提及级） | ICLR'24 | ICLR'25 | ICLR'26 | NeurIPS'23 | NeurIPS'24 | NeurIPS'25 | ICML'24 | ICML'25 | ICML'26 |
|---|---|---|---|---|---|---|---|---|---|
| "flow matching" | 7 | 46 | 144 | 6 | 32 | 88 | 13 | 56 | 167 |
| "optimal transport" | 33 | 47 | 55 | 57 | 49 | 73 | 33 | 43 | 114 |
| "rectified flow" | 2 | 8 | 7 | 0 | 8 | 20 | 1 | 8 | 12 |
| "Schrödinger bridge"（仅测 ICLR） | 5 | 6 | 7 | — | — | — | — | — | — |

口径 B——dblp 标题级（标题词前缀匹配 + venue/year facet；NeurIPS 2026 未举办、CVPR 2026 dblp 尚未收录、ICML/ICLR 部分年份因限流未采全，以口径 A 为主）：

| 关键词（标题级） | NeurIPS'23 | NeurIPS'24 | NeurIPS'25 | CVPR'23 | CVPR'24 | CVPR'25 |
|---|---|---|---|---|---|---|
| flow matching | 6 | 19 | 36 | 2 | 0 | 5 |
| optimal transport | 23 | 19 | 20 | 8 | 4 | 8 |
| rectified flow | 0 | 5 | 12 | — | — | — |
| diffusion（参照系） | — | — | — | 99 | 321 | 343 |

（另采得 ICML'24 标题级 flow matching=11。）

读数：(1) **FM 提及量逐年约 3 倍**（ICLR 7→46→144；ICML 13→56→167），已从方法词变成基础设施词；(2) OT 提及平稳上行、ICML'26 翻倍（43→114），主要作为 FM 文献的理论词汇渗透（minibatch-OT 耦合、W2 界）；(3) RF 术语在 NeurIPS'25 达峰（20）后于 ICLR'26 走平——被 FM 吸收（SD3 效应）；(4) CVPR 标题级 FM/OT 极少（≤8/年）而 diffusion 标题 300+/年：CV 应用层大量用 FM 但不入题名，**OT×扩散的理论主战场在 ML 三会，CVPR 是应用出口**；(5) 标题级≈提及级的 1/2-1/3（NeurIPS'25：36 vs 88），可分别当"主题级"下界与上界。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **落地验收面，直接相关**。端侧部署给"无须重训"提供了最硬的商业理由：SnapGen 级从头训练要大集群，reflow 类重训（InstaFlow 199 A100 天）端侧玩家养不起，而 scheduler/耦合层面的对齐可以叠在已量化模型上——SVDQuant 明确支持"LoRA 免重量化插拔"，意味着 LoRA 形态的轻量对齐模块可以直接部署进 W4A4 栈。评测侧，本课题给方向一提供验收协议：NFE-质量 Pareto 曲线 + 直线度 S(Z) + CMMD/GenEval 组合，替代单点 FID。注意 MixDQ 的教训：少步模型的薄弱环节可能在意想不到处（文本嵌入 BOS），对齐后模型需重做敏感度分析。
- 方向二（OT 引导跨域生成）: **间接相关，两个接口**。其一，端侧实时 image-to-image 是跨域生成的部署形态（SDXS 的 ControlNet 蒸馏做到 30-100 FPS 实时 i2i），OT 引导的跨域映射若要上端侧，必须走同样的"少步+量化"栈；其二，评测即 OT——FID 是高斯近似 W2，CMMD 之争本质是"分布距离该用什么 ground metric/统计量"，为 OT 背景的研究者提供了评测方法学的切入口（如用 Sinkhorn divergence 在 DINOv2 特征上替代 FD）。

## 5. 开放问题与可发论文的切入点

1. **少步×低比特的复合误差理论与实证**：量化噪声在 few-step ODE 离散化下如何累积？假设"轨迹越直，对权重/激活量化噪声越鲁棒"——在 SDXL-Turbo/FLUX-schnell/SANA-Sprint 上系统测 {W8A8,W4A8,W4A4}×NFE∈{1,2,4,8} 质量矩阵，并给 Euler 离散下量化扰动的局部误差界（可衔接 T09 的 W2 收敛界，把直线度参数引入误差常数）。目前 MixDQ/SVDQuant 只给单点结果，无交互项分析。
2. **建"端侧少步生成"标准 benchmark**：现状是各论文自报 NFE-质量曲线，采样器/CFG/seed/分辨率不统一，FID 又对少步模型有偏（Stein）。做一个固定 prompt 集（GenEval+DPG-Bench 子集）× 多指标（CMMD、FD-DINOv2、GenEval、HPSv2）× 实测设备延迟/内存/能耗（iPhone NPU、Exynos NPU、笔记本 4090）的公开 leaderboard，用 Pareto 前沿替代单点排名；顺带回答"FID/CMMD 在 NFE≤4 区间与人评的相关性哪个更高"。
3. **直线度作为可部署性的训练期代理指标**：对同一底模施加不同强度的轨迹对齐（reflow 轮数/耦合正则强度），测直线度 S(Z) 与"量化+少步后质量退化量"的相关性。若相关性成立，直线度可在训练期预测部署损耗，选型不必每次真机跑——直接支撑博客方向一的价值主张。
4. **Sinkhorn/OT 系评测指标对少步模型的公平性**：FID 的正态假设在少步模型输出分布（更集中、模式收缩）上错得更离谱。用 entropic OT（Sinkhorn divergence，OTT-JAX/FlashSinkhorn 可规模化到 50k 样本）在 DINOv2/CLIP 特征上做分布距离，与 CMMD/FD 系统对照人评相关性——评测方法学论文，工作量中等，NeurIPS D&B 或 CVPR 均可投。
5. **FM 原生模型的 NPU 适配空白**：现有 NPU 工程（EdgeFusion）都基于 DDPM 系 UNet；SD3/SANA 类 FM-DiT 的 NPU 移植没有公开研究——linear attention 的整型化、FlowMatch 调度器的 resolution-dependent shift 在静态图上的实现、Gemma 文本编码器端侧化都是空白。做一个"SANA-0.6B 上手机 NPU"的全栈工程论文（对标 EdgeFusion），即是第一篇 FM 模型 NPU 部署工作。

## 6. 代码与资源

**FM/RF 开源栈**
- HuggingFace diffusers：`FlowMatchEulerDiscreteScheduler`/`FlowMatchHeunDiscreteScheduler`（SD3/FLUX/SANA 默认，支持 resolution-dependent shift、dynamic shifting），及 LCM/TCD 等少步调度器 — https://huggingface.co/docs/diffusers/en/api/schedulers/flow_match_euler_discrete
- TorchCFM（OT-CFM/SB-CFM 参考实现，minibatch-OT 耦合训练入门首选）— https://github.com/atong01/conditional-flow-matching
- RectifiedFlow / InstaFlow 官方（reflow 系，详见 T09）— https://github.com/gnobitab/RectifiedFlow

**部署/压缩栈**
- SVDQuant 量化库 deepcompressor + 推理引擎 nunchaku（W4A4，支持 FLUX/SANA/LoRA）— https://github.com/mit-han-lab/deepcompressor · https://github.com/mit-han-lab/nunchaku
- MixDQ（少步模型混合精度 PTQ + HF pipeline + INT8 CUDA kernel）— https://github.com/thu-nics/MixDQ
- SANA 官方（训练+推理+ComfyUI/SGLang 集成，8GB 部署指南；含 SANA-Sprint 一步版）— https://github.com/NVlabs/Sana
- Apple ml-stable-diffusion（Core ML/ANE 移植参考，MobileDiffusion 用其做 iPhone 基准）— https://github.com/apple/ml-stable-diffusion
- Qualcomm AI Hub（Snapdragon NPU 上的量化 SD 系模型库，工程资源）— https://aihub.qualcomm.com/
- 项目页：SnapFusion https://snap-research.github.io/SnapFusion/ · BitsFusion https://snap-research.github.io/BitsFusion/ · SnapGen https://snap-research.github.io/snapgen/ · MobileDiffusion（Google 博客）https://research.google/blog/mobilediffusion-rapid-text-to-image-generation-on-device/

**评测栈**
- dgm-eval（Stein et al. 官方：17 指标 × 9 特征提取器，含 FD-DINOv2 leaderboard）— https://github.com/layer6ai-labs/dgm-eval
- CMMD 参考实现 — https://github.com/google-research/google-research/tree/master/cmmd
- GenEval — https://github.com/djghosh13/geneval ；HPSv2 — https://github.com/tgxs002/HPSv2
- OT 求解器（做 OT 系指标用）：POT https://pythonot.github.io/ · OTT-JAX https://ott-jax.readthedocs.io/

**趋势统计复现口径**：OpenReview API v2 `/notes/search?term="<关键词>"&content=all&group=<venue>/Conference&source=forum`，accepted=venue 字段含 poster/spotlight/oral/regular；dblp `https://dblp.uni-trier.de/search/publ/api?q=<词>+venue:<会>:+year:<年>:`（标题级，注意限流）。

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2023_Li_SnapFusion_t2i_mobile_two_seconds.pdf | SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds | 成功（24.9MB） |
| 2024_Zhao_MobileDiffusion_instant_t2i_mobile.pdf | MobileDiffusion: Instant Text-to-Image Generation on Mobile Devices | 成功（9.6MB） |
| 2024_Castells_EdgeFusion_ondevice_t2i.pdf | EdgeFusion: On-Device Text-to-Image Generation | 成功（5.1MB） |
| 2024_Zhao_MixDQ_fewstep_mixed_precision_quant.pdf | MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization | 成功（16.9MB） |
| 2024_Sui_BitsFusion_199bits_weight_quant.pdf | BitsFusion: 1.99 bits Weight Quantization of Diffusion Model | 成功（14.4MB） |
| 2025_Li_SVDQuant_4bit_diffusion.pdf | SVDQuant: Absorbing Outliers by Low-Rank Component for 4-Bit Diffusion Models | 成功（38.3MB） |
| 2023_Stein_exposing_flaws_gen_eval_metrics.pdf | Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models | 成功（42.7MB） |
| 2024_Jayasumana_Rethinking_FID_CMMD.pdf | Rethinking FID: Towards a Better Evaluation Metric for Image Generation | 成功（3.2MB） |
