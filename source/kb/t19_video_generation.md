# T19 视频生成与时序一致性中的 OT/流

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」全景的高算力应用末端：把 T07–T12 的流匹配/轨迹拉直/蒸馏工具链搬到时空维度最高的视频生成上（博客落地场景之一：高频/实时视频渲染），并回答"时序一致性能否用耦合/传输语言刻画"。调研发现视频侧 OT 有两个真实落点：(a) 少步蒸馏的目标函数（VDOT 的 OT-DMD）；(b) 帧间噪声/先验的分布保持传输（∫-noise 谱系）。通用图像加速见 T09–T12，3D 见 T20。

## 1. 核心问题与背景

视频扩散的推理成本 ≈ 图像成本 × 帧数：14B 级模型以 50 步生成 5 秒 720p 视频需数分钟，而直播、游戏、世界模型等交互场景要求 <40ms/帧。加速沿两条正交轴展开——把**步数**压到 1–4 步（蒸馏/一致性/对抗后训练），把**每步成本**压低（高压缩 VAE、缓存、稀疏注意力、流式缓冲）。视频特有的硬约束是**时序一致性**：少步化不能引入闪烁、漂移与"运动塌缩"（蒸馏后运动幅度显著下降）。OT/流在其中扮演三个角色：(1) **生成公式**——Sora 后的开源模型（HunyuanVideo、Wan、LTX-Video、Pyramid-Flow）已把 flow matching / rectified flow 作为训练标准，视频蒸馏的教师因此天然是流模型；(2) **蒸馏目标**——DMD 的反向 KL 在少步、高维视频分布上易 zero-forcing / 梯度塌缩，VDOT 用熵正则 OT（Sinkhorn）距离补充几何约束；(3) **帧间耦合**——把相邻帧的噪声场视为需"分布保持地传输"的对象，∫-noise 噪声传输方程给出严格解法，成为时序一致性的免训练工具。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ VDOT: Efficient Unified Video Creation via Optimal Transport Distillation (Wang et al.) | 2026 · CVPR (pp. 9273-9283) | [P] | 视频×OT 蒸馏的标志作：在 DMD 框架中用熵正则 OT（Sinkhorn，包络定理给梯度 ∂W/∂D=T*）约束真/假 score 分布对齐，缓解 KL 的 zero-forcing/梯度塌缩；把 VACE-Wan2.1-14B 统一创作模型蒸到 4 步，匹配 50–100 步基线，并发布 UVCBench | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_VDOT_Efficient_Unified_Video_Creation_via_Optimal_Transport_Distillation_CVPR_2026_paper.html) |
| ⭐ From Slow Bidirectional to Fast Autoregressive Video Diffusion Models (CausVid; Yin et al.) | 2025 · CVPR (pp. 22963-22974) | [P] | 把 DMD 扩到视频：双向教师**非对称蒸馏**因果自回归学生 + 教师 ODE 轨迹初始化，4 步、KV cache 流式 9.4 FPS，VBench-Long 84.27，零样本流式 V2V/I2V | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html) |
| ⭐ Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion (Huang et al.) | 2025 · NeurIPS Spotlight | [P] | 训练时自回归 self-rollout（KV 缓存）+ 视频级整体分布匹配损失，消除曝光偏差；随机梯度截断与滚动 KV cache，单 H100 17 FPS 亚秒延迟实时流式生成 | [OpenReview](https://openreview.net/forum?id=mSiN7i0BYH) |
| ⭐ Pyramidal Flow Matching for Efficient Video Generative Modeling (Jin et al.) | 2025 · ICLR | [P] | 视频原生流匹配设计：把去噪轨迹重写为空间金字塔分段流（仅末段全分辨率）+ 时间金字塔压缩历史，单一 DiT 端到端；20.7k A100 时训出 768p·24fps·10s | [OpenReview](https://openreview.net/forum?id=66NzcRQuOq) |
| Flowception: Temporally Expansive Flow Matching for Video Generation (Ifriqi, ..., R.T.Q. Chen) | 2026 · CVPR (pp. 16185-16195) | [P] | 概率路径中交错"离散帧插入 + 连续帧去噪"：非自回归、变长视频生成，训练 FLOPs 降 3 倍，缓解 AR 误差累积，统一 I2V 与插帧 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Ifriqi_Flowception_Temporally_Expansive_Flow_Matching_for_Video_Generation_CVPR_2026_paper.html) |
| FrameBridge: Improving Image-to-Video Generation with Bridge Models (Wang et al.) | 2025 · ICML, PMLR 267 | [P] | 把 I2V 从 noise-to-data 改写为 data-to-data **桥过程**（图像为先验），提出 SNR 对齐微调（扩散→桥模型迁移）与 neural prior；MSR-VTT 零样本 FVD 95 vs 扩散基线 192 | [PMLR](https://proceedings.mlr.press/v267/wang25q.html) |
| ⭐ How I Warped Your Noise: A Temporally-Correlated Noise Prior for Diffusion Models (Chang et al.) | 2024 · ICLR | [P] | 帧间噪声耦合奠基作：把离散噪声重释为连续积分噪声场（∫-noise），推导**噪声传输方程**做分布保持的跨帧噪声平流，免训练消除闪烁/纹理粘连 | [arXiv](https://arxiv.org/abs/2504.03072) |
| Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise (Burgert et al.) | 2025 · CVPR | [P] | 把 ∫-noise 提速为实时逐帧算法（前向/后向光流密度追踪，保 Gaussian 性），用 warped noise 微调 CogVideoX 等，统一实现物体运动/相机/运动迁移控制 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Burgert_Go-with-the-Flow_Motion-Controllable_Video_Diffusion_Models_Using_Real-Time_Warped_Noise_CVPR_2025_paper.html) |
| Diffusion Adversarial Post-Training for One-Step Video Generation (Seaweed-APT; Lin et al.) | 2025 · ICML, PMLR 267 | [P] | 扩散预训练后对**真实数据**对抗后训练（近似 R1 正则稳定训练），单次前向实时生成 2s·1280×720·24fps 视频——工业级一步视频的首个公开配方 | [arXiv](https://arxiv.org/abs/2501.08316) |
| Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation (AAPT; Lin et al.) | 2025 · NeurIPS | [P] | 把预训练视频扩散改造成 1NFE/帧的自回归实时交互生成器（单 H100 24fps 736×416、8×H100 720p），流式接收用户控制、可至分钟级 | [arXiv](https://arxiv.org/abs/2506.09350) |
| StreamDiT: Real-Time Streaming Text-to-Video Generation (Kodaira et al.) | 2026 · CVPR | [P] | 基于流匹配的**移动缓冲**训练（缓冲内帧带时变噪声水平），混合分区方案+少步蒸馏，4B 模型单 GPU 16 FPS 流式 512p 生成 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Kodaira_StreamDiT_Real-Time_Streaming_Text-to-Video_Generation_CVPR_2026_paper.html) |
| Learning Few-Step Diffusion Models by Trajectory Distribution Matching (TDM; Luo et al.) | 2025 · ICCV | [P] | 统一分布匹配与轨迹匹配的**数据自由**少步蒸馏（沿教师 ODE 轨迹逐段分布对齐），图像与视频通吃：CogVideoX-2B 蒸到 4 步且 VBench 超教师 | [arXiv](https://arxiv.org/abs/2503.06674) |
| T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback (Li et al.) | 2024 · NeurIPS | [P] | 视频一致性蒸馏 + 图文/视频文本奖励模型混合反馈，突破 VCM 质量瓶颈，4–8 步 VBench 超当时闭源模型（Gen-2/Pika） | [arXiv](https://arxiv.org/abs/2405.18750) |
| Taming Rectified Flow for Inversion and Editing (RF-Solver; Wang et al.) | 2025 · ICML, PMLR 267 | [P] | 免训练高阶求解器降低 rectified flow ODE 反演误差（精确解+泰勒展开），在 FLUX 与 **OpenSora 视频**上改进反演重建与编辑——视频流模型编辑的求解器基础 | [arXiv](https://arxiv.org/abs/2411.04746) |
| LTX-Video: Realtime Video Latent Diffusion (HaCohen et al.) | 2024–25 · arXiv | [R] | 实时渲染标杆：1:192 高压缩 Video-VAE（32×32×8/token）与去噪 DiT 整体协同设计 + rectified flow，H100 上 2 秒生成 5 秒 768×512·24fps 视频（快于播放速度） | [arXiv](https://arxiv.org/abs/2501.00103) |

表外近作（补充脉络）：**基础模型** Wan（开源 FM 视频套件 1.3B/14B，arXiv 2503.20314, [R]）、HunyuanVideo（13B FM，arXiv 2412.03603, [R]）；**蒸馏旁支** SF-V（对抗微调 SVD 单步，~23× 加速，NeurIPS 2024, [P]）、OSV（两阶段 GAN+一致性，1 步 FVD 171 超 AnimateLCM 8 步，CVPR 2025, [P]）、Motion Consistency Model（运动-外观解耦蒸馏，NeurIPS 2024, [P]）、AccVideo（教师合成轨迹数据集 SynVid 蒸 HunyuanVideo/Wan 至 5 步、8.5×，arXiv 2503.19462, [R]）、VideoLCM（首个视频 LCM，arXiv 2312.09109, [R]）、AnimateDiff-Lightning（跨模型渐进对抗蒸馏运动模块，arXiv 2403.12706, [R]）；**时间结构** Diffusion Forcing（逐 token 独立噪声水平，AR×扩散混合的概念源头，NeurIPS 2024, [P]，arXiv 2407.01392）、Pusa（帧级向量化时间步微调 Wan，~$500 解锁零样本 I2V/插帧/扩展，arXiv 2507.16116, [R]）；**编辑** FlowDirector（免反演：直接在数据空间沿流 ODE 演化做文本驱动视频编辑，arXiv 2506.05046, [R]）。

## 3. 方法演进脉络

**公式原生化（2024–2026）**：Sora 之后开源阵营全面转向流公式——HunyuanVideo、Wan、LTX-Video 均以 FM/RF 训练，使"视频教师即流模型"成为蒸馏的默认前提。视频原生的流设计随之出现：Pyramid-Flow 利用 FM 可在任意两分布间插值的灵活性，把去噪轨迹拆成分辨率金字塔分段流并用时间金字塔压缩自回归历史；Flowception（FM 原作者 R.T.Q. Chen 参与）更进一步，把"离散帧插入"编进概率路径本身，让视频长度与内容联合生成，兼得非自回归的稳定与 AR 的可变长。FrameBridge 则从先验侧改造：I2V 的起点不该是白噪声而是给定图像，data-to-data 桥过程（OT/SB 家族的近亲）显著提升外观一致性。

**蒸馏主线：从 KL 到 OT（2024.12–2026）**：图像 DMD 移植到视频始于 CausVid——双向教师非对称蒸馏因果学生 + ODE 初始化，4 步流式 9.4 FPS；Self-Forcing 指出其残留的曝光偏差，用训练时 self-rollout 让整段生成序列接受视频级分布匹配监督，达成单卡实时（17 FPS）；VDOT 补上目标函数一环：少步场景反向 KL 有 zero-forcing/梯度塌缩，用 Sinkhorn 求解的熵正则 OT 距离对真/假 score 样本做几何对齐（梯度即传输计划 T*），配判别器把统一创作模型（VACE）蒸到 4 步——这是 OT 作为一等公民进入视频蒸馏损失的首个 [P] 级工作。旁支路线并行：一致性系（VideoLCM→T2V-Turbo 加奖励→MCM 解耦运动/外观→OSV 两阶段），对抗系（SF-V→Seaweed-APT 一步 720p 实时→AAPT 自回归交互式），轨迹系（AccVideo 合成轨迹数据集；TDM 统一轨迹×分布匹配且数据自由）。值得注意：图像上成熟的 reflow 路线（T09）在视频上**几乎缺席**，公开工作均绕开整轨迹仿真而走 DMD/GAN/一致性。

**帧间耦合线（2024–2025）**：HIWYN 把噪声重释为连续积分场并推导噪声传输方程，实现严格分布保持的跨帧噪声平流——这本质是"约束边缘为 N(0,I) 的传输问题"的工程解；Go-with-the-Flow 将其提速为实时逐帧算法并规模化用于微调，把时序一致性与运动控制统一为"噪声耦合的结构设计"。RF-Solver/FlowDirector 则在编辑侧提供流 ODE 的精确反演与免反演演化。**实时工程线**：LTX-Video（高压缩 VAE + RF）、StreamDiT（移动缓冲 FM + 时变噪声）与蒸馏线汇流，把视频生成从分钟级推进到秒级乃至交互实时。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: **强关联，且是最佳落地场景**。(a) ∫-noise/Go-with-the-Flow 正是"无须重训的耦合对齐"：不动模型，只把各帧采样轨迹的**起点**（噪声）沿运动做分布保持传输，即显著改善时序一致性——这可直接解读为"帧间轨迹对齐"的 OT 实例，且给出了必须满足的边缘约束（保 Gaussian）作为理论抓手。(b) RF-Solver/FlowDirector 的免训练反演/流引导为"对齐已训好的视频流模型轨迹"提供求解器工具。(c) 重训式蒸馏（CausVid/VDOT，数千 H100/H200 GPU 时级）恰是"无须重训"方案要对标的成本靶子；高频渲染场景（本子课题的博客场景）对轻量对齐方案的需求最迫切。
- 方向二（OT 引导跨域生成）: **中等关联**。视频-视频翻译是跨域生成的时序版本：CausVid 支持零样本流式 V2V，VDOT 统一 V2V/MV2V 编辑并证明 OT 距离作为跨分布对齐损失在视频域可行且更稳；FrameBridge 的 data-to-data 桥表明"从结构化先验出发的传输过程"优于从噪声出发。但**显式 OT 约束的视频跨域翻译**（如带传输代价正则的 video style transfer）在 2024–2026 顶会几乎空白（检索仅见视频理解侧的时序 OT，如动作分割/因果 OT 域适应），这本身是方向二可拓展的处女地。

## 5. 开放问题与可发论文的切入点

1. **帧间噪声耦合的 OT 理论化**：HIWYN/GwtF 的 noise warping 是"沿光流的传输"但无最优性主张。可把帧间噪声耦合形式化为边缘约束 N(0,I) 的（熵正则/动态）OT 问题，证明 ∫-noise 传输是否为某代价下的 Monge 映射或给出反例；实验上用 Sinkhorn 在 latent 块上求帧间耦合替代光流 warp，量化"时序一致性（warp error/FVD）vs 运动多样性"的帕累托前沿。直接服务博客方向一。
2. **OT-DMD 的系统消融与缩放规律**：VDOT 只在 VACE-Wan-14B 一个设置上验证。固定教师（Wan2.1-1.3B 可低成本复现），对照 KL-DMD / OT-DMD / 对抗 三种目标，扫 Sinkhorn ε 与样本批大小，回答"OT 几何约束的增益如何随分辨率、时长、步数缩放"；用频谱与运动统计（光流幅度分布的 W2）诊断少步蒸馏的"运动塌缩"是否被 OT 项缓解。
3. **视频 reflow 的空白**：图像 reflow 已成熟（T09）而视频公开工作缺席——因为整轨迹仿真成本高且合成耦合会破坏时间结构。切入点：时空分解 reflow——空间维用 minibatch-OT 配对、时间维保持因果/光流耦合，只对短窗口做分段 reflow（PeRFlow 思路的时序版）；理论上分析视频数据流形上"直耦合"与时序一致性是否冲突。
4. **少步视频模型的 OT 一致性度量**：VBench 时序指标（subject/background consistency）过于粗糙，难以检测蒸馏引起的运动退化。提出基于帧间特征 Wasserstein 距离轨迹的"运动 OT 谱"，对比 4 步学生与 50 步教师，建立蒸馏质量的可复现 benchmark（可挂在 UVCBench/VBench 生态上）。
5. **流式外推的漂移作为累积传输误差**：Self-Forcing 消除了训练内曝光偏差但超出训练时长仍退化。把滚动 KV 外推的分布漂移建模为逐段传输误差的望远镜求和，用在线 OT 距离监测并做锚定分布再投影（anchor re-projection）正则，给出漂移的 TV/W2 上界——与 T05（Wasserstein 梯度流）的工具可对接。

## 6. 代码与资源

- VDOT（训练/推理/UVCBench）: https://github.com/hhhh1138/VDOT ；模型 https://huggingface.co/yutongwang1012/VDOT ；基准 https://huggingface.co/datasets/yutongwang1012/UVCBench
- CausVid: https://github.com/tianweiy/CausVid ；Self-Forcing: https://github.com/guandeh17/Self-Forcing
- Pyramid-Flow: https://github.com/jy0205/Pyramid-Flow ；Flowception 项目页见 CVF 论文（Meta，arXiv 附录含实现细节）
- LTX-Video（含蒸馏版权重）: https://github.com/Lightricks/LTX-Video ；Wan2.1: https://github.com/Wan-Video/Wan2.1 ；HunyuanVideo: https://github.com/Tencent/HunyuanVideo
- 噪声传输：∫-noise 项目页 https://warpyournoise.github.io/ ；Go-with-the-Flow: https://eyeline-research.github.io/Go-with-the-Flow/
- 编辑：RF-Solver-Edit: https://github.com/wangjiangshan0725/RF-Solver-Edit ；FlowDirector: https://flowdirector-edit.github.io/ ；Pusa: https://github.com/Yaofang-Liu/Pusa-VidGen
- 蒸馏工具链：TDM: https://tdm-t2x.github.io/ ；AccVideo: https://github.com/aejion/AccVideo ；FastVideo（DMD+稀疏注意力蒸 Wan/Hunyuan 的开源框架）: https://github.com/hao-ai-lab/FastVideo
- 评测：VBench / VBench-Long: https://github.com/Vchitect/VBench ；UVCBench（统一创作 18 任务）；MovieGenBench（CausVid 用）
- 实用提示：Wan/LTX/Hunyuan 均已入 diffusers；少步视频权重普遍存在"运动幅度下降"现象（APT/AAPT 论文均有讨论），对比评测时应报告运动统计而非仅帧质量

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2025_Wang_VDOT_OT_Distillation_Video.pdf | VDOT: Efficient Unified Video Creation via Optimal Transport Distillation | 成功 |
| 2024_Yin_CausVid_Fast_Autoregressive_Video.pdf | From Slow Bidirectional to Fast Autoregressive Video Diffusion Models | 成功 |
| 2025_Huang_Self_Forcing_Autoregressive_Video.pdf | Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion | 成功 |
| 2024_Jin_Pyramidal_Flow_Matching.pdf | Pyramidal Flow Matching for Efficient Video Generative Modeling | 成功 |
| 2024_Chang_How_I_Warped_Your_Noise.pdf | How I Warped Your Noise: A Temporally-Correlated Noise Prior for Diffusion Models | 成功 |
| 2026_Ifriqi_Flowception_Temporal_Flow_Matching.pdf | Flowception: Temporally Expansive Flow Matching for Video Generation | 成功 |
| 2025_Kodaira_StreamDiT_Realtime_Streaming_T2V.pdf | StreamDiT: Real-Time Streaming Text-to-Video Generation | 成功 |
| 2025_Burgert_Go_with_the_Flow_Warped_Noise.pdf | Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise | 成功 |
