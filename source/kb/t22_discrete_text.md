# T22 离散数据与文本中的扩散/流与最优传输

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题覆盖「扩散×OT」全景中的离散一侧：当数据空间没有连续几何（词表、序列、图符号）时，扩散/流模型如何定义，OT 又以何种形式注入——耦合选择、传输成本（Hamming/编辑距离/语义嵌入）、离散 Wasserstein 度量、以及蒸馏加速中的分布对齐。它是把连续 OT-FM/SB 主线（T07/T08/T03）推广到语言域的关键接口，也是扩散 LLM 采样加速的理论支点。
> 边界: 连续流匹配谱系归 T07/T08（本文仅交叉引用）；分子/蛋白/DNA 序列应用归 T21（本文只收其方法论骨架）。

## 1. 核心问题与背景

连续扩散/流匹配的成功依赖 \(\mathbb{R}^d\) 的几何：score 有定义、ODE 轨迹可拉直、OT 耦合可降低传输成本。而语言等离散数据生活在有限状态空间 \(\{1,\dots,S\}^d\) 上，没有梯度、没有确定性轨迹，前向过程只能是连续时间马尔可夫链（CTMC）的跳跃过程。这带来三个核心问题：(i) 如何定义离散空间上的「score」与生成过程（D3PM→SEDD→masked diffusion 的主线）；(ii) 连续 FM 里「rectification / OT 耦合拉直路径」的思想在离散空间失效——路径本质随机、无瞬时变量替换公式——那么「减少传输成本/转移次数」应如何重新形式化（离散动态 OT、Kantorovich 耦合、编辑距离成本）；(iii) 扩散 LM 的采样需要成百上千次迭代，few-step 加速依赖对多步转移核的分布蒸馏，其损失函数（KL、一致性、Sinkhorn/Wasserstein）本质是序列分布对齐问题。2024-2026 年该方向爆发：SEDD 拿下 ICML 2024 最佳论文，LLaDA/Dream 把扩散 LM 推到 8B 规模，而 OT 视角（Fisher-Flow 的黎曼 OT、minibatch-OT 离散流、DDOT 位置耦合、跨 tokenizer OT 蒸馏）正在成为提升训练效率与采样速度的统一语言。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD) | 2024·ICML (Best Paper) | [P] | 提出 score entropy 把 score matching 推广到离散空间（学概率比值），扩散 LM 首次在困惑度上压过 GPT-2，并可 32× 减少 NFE | [PMLR](https://proceedings.mlr.press/v235/lou24a.html) |
| ⭐ Discrete Flow Matching | 2024·NeurIPS | [P] | 通用离散概率路径族 + 后验参数化的生成速度公式 + corrector 采样，1.7B 模型显著缩小与 AR 的代码/文本生成差距 | [arXiv](https://arxiv.org/abs/2407.15595) |
| ⭐ Fisher Flow Matching for Generative Modeling over Discrete Data (Fisher-Flow) | 2024·NeurIPS | [P] | 把类别分布放到 Fisher-Rao 统计流形（球面正象限）上做连续 FM，用黎曼 OT 重耦合改善训练动力学，并证其梯度流最优降低前向 KL | [arXiv](https://arxiv.org/abs/2405.14664) |
| ⭐ Minibatch Optimal Transport and Perplexity Bound Estimation in Discrete Flow Matching | 2026·ICML | [A] | 首个离散流的动态 OT 式目标及其 Kantorovich 形式（成本=状态间相异度），minibatch-OT 耦合把达到同等生成困惑度的转移次数降至 1/32；另给出离散流困惑度上界 | [icml.cc](https://icml.cc/virtual/2026/poster/65787) · [arXiv](https://arxiv.org/abs/2411.00759) |
| ⭐ Flexible-length Text Infilling for Discrete Diffusion Models (DDOT) | 2025·EMNLP main | [P] | 首个灵活长度/位置文本填充的离散扩散：联合去噪 token 值与位置，用 sample-level OT 耦合保持相对语序、动态调整填充段位置与长度 | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1597/) |
| Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM) | 2021·NeurIPS | [P] | 离散扩散奠基：结构化转移矩阵（uniform/absorbing/离散化高斯）统一离散前向过程设计空间 | [arXiv](https://arxiv.org/abs/2107.03006) |
| Simple and Effective Masked Diffusion Language Models (MDLM) | 2024·NeurIPS | [P] | Rao-Blackwell 化的连续时间 ELBO 证明 masked diffusion 目标 = 加权 MLM 交叉熵混合，给 BERT 式编码器赋予有原则的生成能力 | [arXiv](https://arxiv.org/abs/2406.07524) |
| Simplified and Generalized Masked Diffusion for Discrete Data (MD4) | 2024·NeurIPS | [P] | 统一简化 masked diffusion 框架，支持状态依赖的 masking schedule；像素级离散建模超过同规模 AR | [arXiv](https://arxiv.org/abs/2406.04329) |
| Generative Flows on Discrete State-Spaces (Multiflow) | 2024·ICML | [P] | CTMC 离散流匹配框架（DFM 的直接前身），解耦训练与采样过程；蛋白质共设计应用详见 T21 | [arXiv](https://arxiv.org/abs/2402.04997) |
| Large Language Diffusion Models (LLaDA) | 2025·NeurIPS (Oral) | [P] | 8B 从零预训练+SFT 的 masked 扩散 LM，多基准比肩 LLaMA3-8B，证明扩散范式可规模化并破解 reversal curse | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/48b383b24230e0e6e649d9c98dae4d8c-Abstract-Conference.html) |
| Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (BD3-LM) | 2025·ICLR (Oral) | [P] | 块间自回归+块内扩散的插值族，支持 KV cache、任意长度生成与并行采样，扩散 LM 似然新 SOTA | [arXiv](https://arxiv.org/abs/2503.09573) |
| Edit Flows: Variable Length Discrete Flow Matching with Sequence-Level Edit Operations | 2025·NeurIPS | [P] | 在整条序列空间上定义 CTMC，转移=插入/删除/替换编辑操作（编辑距离几何），经辅助对齐过程+Bregman 散度损失可训练，原生支持变长生成 | [arXiv](https://arxiv.org/abs/2506.09018) |
| The Diffusion Duality (Duo) | 2025·ICML | [P] | 证明 uniform-state 离散扩散 = 底层高斯扩散经 argmax 投影而来；借对偶把课程学习与一致性蒸馏搬到离散域（DCD），采样加速两个数量级 | [arXiv](https://arxiv.org/abs/2506.10892) |
| Distillation of Discrete Diffusion through Dimensional Correlations (Di4C) | 2025·ICML | [P] | 用混合模型显式学习维度间相关性，把多步独立分解的教师蒸馏成 few-step 学生，并给出学生-教师分布距离上界 | [OpenReview](https://openreview.net/forum?id=jCEl0aJpF6) |
| Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation (MultiLevelOT) | 2025·AAAI (Oral) | [P] | token 级+序列级双层 OT（Sinkhorn）对齐不同词表的 logit 分布，实现任意教师→学生的跨 tokenizer LLM 蒸馏 | [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/34543) |

### 补充条目（次级/理论/预印本）

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| Beyond Autoregression: Fast LLMs via Self-Distillation Through Time (SDTT) | 2025·ICLR | [A] | 跨时间自蒸馏 masked 扩散 LM，32-64 token/步并行解码仍优于 GPT-2 级 AR（多篇 proceedings 交叉引用确认 ICLR 2025） | [arXiv](https://arxiv.org/abs/2410.21035) |
| Jump Your Steps | 2025·ICLR | [A] | 无额外计算下优化离散扩散采样时间表，最小化复合解码误差（Di4C 官方引用确认 ICLR 2025） | [arXiv](https://arxiv.org/abs/2410.07761) |
| Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data (RADD) | 2025·ICLR | [A] | 证明 absorbing 扩散的 concrete score 可分解为条件分布×时间标量，与任意序 AR 等价，支持缓存加速 | [arXiv](https://arxiv.org/abs/2406.03736) |
| Sinkhorn Distance Minimization for Knowledge Distillation (SinKD) | 2024·LREC-COLING; 2025·IEEE TNNLS | [P] | 用 Sinkhorn 距离替代 KL/RKL/JS 做 LLM logit 蒸馏，批级重构感知分布几何，规避 mode-averaging/collapsing | [arXiv](https://arxiv.org/abs/2402.17110) · [DOI](https://doi.org/10.1109/TNNLS.2024.3501335) |
| Towards Cross-Tokenizer Distillation: Universal Logit Distillation (ULD) | 2024·arXiv | [R] | 最早用 Wasserstein 距离做跨词表 logit 蒸馏的损失（MultiLevelOT 的直接前驱） | [arXiv](https://arxiv.org/abs/2402.12030) |
| Optimal Transport-Based Token Weighting for Preference Optimization (OTPO) | 2025·ACL main | [P] | 用 unbalanced OT 在 chosen/rejected 回复间算语义对齐权重，重加权 DPO 的 token 级损失，统一 SimPO/SamPO 等为特例 | [ACL Anthology](https://aclanthology.org/2025.acl-long.1035/) |
| Convergence Analysis of Discrete Diffusion Model: Exact Implementation through Uniformization | 2025·J. Mach. Learn. 4(2) | [P] | 用 CTMC uniformization 精确模拟反向链，给出超立方体上 TV/KL 收敛保证，对齐连续扩散最优结果 | [DOI](https://doi.org/10.4208/jml.240812) |
| Efficient Sampling with Discrete Diffusion Models: Sharp and Adaptive Guarantees | 2026·COLT (PMLR 336) | [P] | τ-leaping 达 \(\tilde O(d/\varepsilon)\) KL 复杂度（消去词表 S 依赖）+匹配下界；masked 情形由「有效总相关」自适应控制，可对结构化数据亚线性 | [PMLR](https://proceedings.mlr.press/v336/dmitriev26a.html) |
| Dimension-free Convergence of Discrete Diffusion Models | 2026·arXiv | [R] | 伴随方程框架给出任意 IPM 下完全不依赖词表大小 S 的收敛界，首次同时覆盖 masked（奇异先验）与 uniform | [arXiv](https://arxiv.org/abs/2605.17232) |
| Consistent Diffusion Language Models (CDLM / MPDC) | 2026·arXiv | [R] | 离散域没有 PF-ODE，提出以精确后验桥为「轨迹」的多路径离散一致性训练（teacher-free），统一 masked diffusion/连续一致性/渐进蒸馏为解析极限 | [arXiv](https://arxiv.org/abs/2605.00161) |
| An Optimal Transport View of Activation Steering in Masked Diffusion Models | 2026·ICLR Workshop (TTU) | [A] | 把 dLLM 激活转向统一为仿射 OT map（矩匹配估计），推理时零开销提升 LLaDA/Dream 指令遵循 +6.5~11.9 分 | [OpenReview](https://openreview.net/forum?id=3JM0DTKxgE) |
| Dream 7B: Diffusion Large Language Models | 2025·arXiv | [R] | AR 初始化+上下文自适应 token 级噪声重调度训练的 7B 扩散 LM，规划类任务显著优于同规模 AR | [arXiv](https://arxiv.org/abs/2508.15487) |
| d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation | 2026·arXiv（repo 自称 ICML 2026，未经官方页核验） | [R] | 伪轨迹蒸馏使 LLaDA/Dream 级 dLLM 接近每步多 token 的极限吞吐 | [arXiv](https://arxiv.org/abs/2601.07568) |

## 3. 方法演进脉络

**第一阶段（2021-2023）：把扩散搬进离散空间。** D3PM（NeurIPS 2021）用结构化转移矩阵定义离散前向链；Campbell et al.（NeurIPS 2022）给出连续时间 CTMC 框架与 τ-leaping 采样。此阶段模型的似然与生成质量均落后 AR。

**第二阶段（2024）：目标函数的正确打开方式。** SEDD 提出 score entropy，把「学 score」替换为「学概率比值」，第一次让扩散 LM 在困惑度上超过 GPT-2；随后 MDLM、MD4、RADD 三条并行工作把 masked（absorbing）扩散的 ELBO 化简为加权交叉熵，揭示其与任意序自回归的等价性——离散扩散从「异域移植」变成「良定义的似然模型」。平行地，流匹配谱系出现：Multiflow（ICML 2024）建立 CTMC 离散流，DFM（NeurIPS 2024）给出通用路径族+corrector 采样并扩到 1.7B 代码模型。

**第三阶段（2024-2025）：几何与 OT 的注入。** 连续 FM 的 rectification/OT 耦合无法直接照搬——离散路径本质随机。三条绕行路线出现：(a) **连续化**：Fisher-Flow 把类别分布嵌入 Fisher-Rao 球面，在流形上做 FM 并用黎曼 OT 重耦合（此思路上承 T07 的几何 FM）；Duo 则证明 uniform 离散扩散本就是高斯扩散的 argmax 投影，把连续域的课程学习与一致性蒸馏「无损搬运」过来。(b) **重新定义离散动态 OT**：Haxholli et al.（ICML 2026）提出以状态间相异度为成本的 Kantorovich 目标，minibatch-OT 耦合源-目标分布，等效于离散版「拉直」——同等困惑度下转移数降到 1/32。(c) **改状态空间几何**：Edit Flows 把 CTMC 定义在整条序列空间上，转移即插入/删除/替换，隐式采用编辑距离几何；DDOT 在 token 位置上做 sample-level OT 耦合，解决灵活长度填充。

**第四阶段（2025-2026）：规模化与 few-step 加速。** LLaDA 8B（NeurIPS 2025 Oral）、Dream 7B 证明扩散 LM 可与 LLaMA3 级 AR 竞争；BD3-LM 用块结构调和 AR 与扩散。加速侧形成「蒸馏即分布对齐」的谱系：SDTT（跨时间自蒸馏）→ Di4C（维度相关性混合模型蒸馏）→ Duo 的 DCD（对偶导出的离散一致性）→ CDLM（后验桥多路径一致性，teacher-free）→ d3LLM（伪轨迹蒸馏）。理论侧从 uniformization 精确采样（Chen & Ying）推进到 COLT 2026 的 sharp τ-leaping 界与 2026 年的维度无关 IPM 界，为「离散扩散的传输复杂度」提供了度量基础。

## 4. 与博客两个方向的关联

- **方向一（无须重训的轨迹对齐）**：中等强度关联，但「对齐对象」需换概念——离散空间无 ODE 轨迹可拉直，对应物是 CTMC 的跳跃次数/转移成本。Minibatch-OT DFM 正是离散版轨迹对齐：不改架构、只改训练耦合即可 32× 降低 NFE；Duo/DCD 与 CDLM 表明「一致性蒸馏」这一连续域轨迹自洽工具可经对偶或后验桥迁移，属后训练（非重训）加速。推理时干预的直接例子是 OT activation steering（ICLR 2026 WS）：仿射 OT map 转向 dLLM 激活，零开销、无须重训。DDOT 的 OT 位置耦合同样发生在采样结构层面。
- **方向二（OT 引导跨域生成）**：间接相关但接口清晰。跨 tokenizer 蒸馏（ULD→SinKD→MultiLevelOT）本质是「跨词表域」的分布桥接——成本矩阵定义在 logit/嵌入空间，Sinkhorn 求耦合，这正是离散域间 OT 引导的知识迁移；OTPO 把 chosen/rejected 回复间的 unbalanced OT 耦合用于偏好对齐，属「序列域间语义传输」。若把博客的跨域生成理解为图像风格/域迁移，则文本对应物（风格迁移、改写）尚缺一个显式 OT-guided 离散扩散工作，这本身是空白（见下）。

## 5. 开放问题与可发论文的切入点

1. **离散 kinetic-optimal 路径定理**：连续 FM 有 kinetic energy 最优路径理论，离散侧只有 Haxholli et al. 的经验性 Kantorovich 目标。可证：给定 Hamming/图距离成本，masked 与 uniform 前向过程的 rate matrix 何时是离散 Benamou-Brenier（Maas 2011 有限马尔可夫链 Wasserstein 度量）意义下的测地线；推导最优 scheduler 的闭式条件。实验：text8/OWT 上对比理论最优 schedule 与 loglinear/cosine 的 NFE-困惑度前沿。
2. **语义成本的 minibatch-OT 离散耦合**：Haxholli 的成本只用状态相异度（≈Hamming）。改用冻结嵌入空间的语义距离或编辑距离作为 ground cost 做源-目标 minibatch 耦合，验证 few-step 生成质量（gen-PPL、MAUVE、熵）增益；理论上结合 T08 的 minibatch OT 偏差结果，分析离散情形耦合偏差随 batch size 的衰减率。
3. **Edit Flows × unbalanced OT**：插入/删除天然对应质量创生/湮灭，即 unbalanced OT。把 Edit Flows 的辅助对齐采样替换为 Levenshtein 成本下的最优（部分）耦合，可证训练方差降低、期望编辑次数逼近编辑距离下界；这会把「编辑距离成本的离散 OT」从隐式变成显式可微模块。
4. **任意教师→扩散学生的 OT 蒸馏统一框架**：现有 dLLM 加速蒸馏（SDTT/Di4C/DCD）全是同模型自蒸馏，而跨 tokenizer OT 蒸馏（MultiLevelOT）只服务 AR→AR。二者拼接即新题：用双层 OT 损失把强 AR 教师（不同词表）蒸馏进 masked 扩散学生，同时继承 few-step 采样——一个实验闭环即可验证（AR 7B → dLLM 1B，比较 OT 损失 vs 逐 token KL 的下游/速度前沿）。
5. **W1 而非 KL/TV 的离散扩散收敛理论**：2026 年的收敛工作（COLT 2026、dimension-free IPM）已指出 KL 在 masked 奇异先验下发散、TV 带 S 依赖。用 Hamming ground metric 的 W1（或一般 IPM 中挑 Lipschitz 类）建立 τ-leaping/uniformization 的收敛界，并论证 W1 界更能解释大词表下经验成功；顺带可给出「OT 意义下的最优早停时间」。

## 6. 代码与资源

- SEDD 官方: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- MDLM 官方（含博客/教程视频）: https://github.com/kuleshov-group/mdlm · https://s-sahoo.com/mdlm
- MD4 (DeepMind): https://github.com/google-deepmind/md4
- Fisher-Flow: https://github.com/olsdavis/fisher-flow
- BD3-LM: https://github.com/kuleshov-group/bd3lms
- Duo（含 DCD 蒸馏与 checkpoints）: https://s-sahoo.com/duo
- Di4C (Sony): https://github.com/sony/di4c
- LLaDA: https://ml-gsai.github.io/LLaDA-demo/ ；Dream 7B: https://github.com/DreamLM/Dream
- d3LLM（dLLM 加速全家桶，含 Fast-dLLM/dParallel 复现）: https://github.com/hao-ai-lab/d3LLM
- 文本 OT 损失: MultiLevelOT https://github.com/2018cx/Multi-Level-OT · SinKD https://github.com/2018cx/SinKD · OTPO https://github.com/Mimasss2/OTPO
- 常用基准: text8、OpenWebText、LM1B、One-Billion-Word/Yelp（填充）、HumanEval/MBPP（代码）；评价用 generative perplexity（GPT-2/Llama 评估器）、MAUVE、序列熵
- 综述 [B]: Optimal and Diffusion Transports in Machine Learning (2025-12, https://arxiv.org/abs/2512.06797)——含离散/动态 OT 与生成模型统一视角，适合作为 T22 与 T01/T05 的桥梁读物

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2024_Lou_SEDD_score_entropy.pdf | Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution | 成功（1.0MB，%PDF 校验通过） |
| 2024_Gat_discrete_flow_matching.pdf | Discrete Flow Matching | 成功（2.7MB，%PDF 校验通过） |
| 2024_Davis_fisher_flow.pdf | Fisher Flow Matching for Generative Modeling over Discrete Data | 成功（3.8MB，%PDF 校验通过） |
| 2026_Haxholli_minibatch_ot_dfm.pdf | Minibatch Optimal Transport and Perplexity Bound Estimation in Discrete Flow Matching | 成功（0.7MB，%PDF 校验通过） |
| 2025_Zhang_ddot_ot_infilling.pdf | Flexible-length Text Infilling for Discrete Diffusion Models (DDOT) | 成功（2.1MB，ACL 官方 OA，%PDF 校验通过） |
| 2024_Sahoo_mdlm_masked_diffusion.pdf | Simple and Effective Masked Diffusion Language Models | 成功（1.0MB，%PDF 校验通过） |
| 2025_Sahoo_diffusion_duality.pdf | The Diffusion Duality | 成功（1.3MB，%PDF 校验通过） |
| 2025_Havasi_edit_flows.pdf | Edit Flows: Variable Length Discrete Flow Matching with Sequence-Level Edit Operations | 成功（1.3MB，%PDF 校验通过） |
