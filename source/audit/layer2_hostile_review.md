# Layer-2 敌意审稿报告：造假取证与自洽性审查

> 审计日期：2026-08-14 ｜ 审计者：ARIS Layer-2 hostile reviewer（全新上下文，与被审产出零利益关联）
> 裁定原则：**no-new-blocker ≠ acquittal**。本报告只出具 flag，不负责修复。

## 1. 审计范围与方法

- 对象：`REPORT_DIFFUSION_OT_20260814.md`（主报告）、`kb/t01–t30`（30 份笔记）、`refs/MASTER_BIBLIOGRAPHY.md`（477 条聚合引用库）。
- 方法：15 条指定靶标 + 9 条自选抽查 + 4 条报告级荣誉加扫，共 **28 项独立网络核验**（arXiv API 元数据、icml.cc/iclr.cc/neurips.cc 官方 virtual 页、CVF/ECVA/AAAI OJS/PMLR/OpenReview/Nature/SIAM DOI、dblp、Semantic Scholar API 交叉），每项核验【存在性/会议归属/荣誉/内容相符】四维；另对主报告按 hack-pattern 家族（范围膨胀/幻影结果/自评美化/荣誉通胀/趋势口径）逐节扫描，并用脚本实测 PDF 计数与引用库去重。
- 反爬处理：IEEE DOI 直连 406、OpenReview 一次人机验证页，均按预案改用 Semantic Scholar/dblp/搜索引擎替代源双证，无条目因反爬弃核。

## 2. 靶标核验表（指定 15 条，全部完成）

| # | 论断 | 存在性 | 归属 | 荣誉 | 内容相符 | 证据链接 | 裁定 |
|---|---|---|---|---|---|---|---|
| 1 | FlashSinkhorn：ICML 2026 Oral、IO-aware Sinkhorn、最高 161×（t29） | ✓ arXiv 2602.03067（Ye et al.，作者一致） | ✓ ICML 2026 | ✓ Oral（官方 virtual 页 + 官方 repo 注 top 0.7%） | ✓ online-LSE/O(nd)/前向最高 32×、端到端最高 161× 与摘要逐字吻合；笔记"9–32×"下限为论文内细节，无矛盾 | [icml.cc/virtual/2026/oral/71180](https://icml.cc/virtual/2026/oral/71180) · [arXiv](https://arxiv.org/abs/2602.03067) | **PASS** |
| 2 | VDOT：CVPR 2026、熵正则 OT 引入 DMD 视频蒸馏（t10/t19） | ✓ arXiv 2512.06802（Wang et al.） | ✓ CVPR 2026 正式论文集，页码 9273–9283 与 t19 一致 | —（无荣誉声明） | ✓ 熵 OT 替代/增强 KL-DMD、4 步匹敌 50–100 步、UVCBench、VACE-Wan2.1-14B 均与 CVF 摘要一致 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_VDOT_Efficient_Unified_Video_Creation_via_Optimal_Transport_Distillation_CVPR_2026_paper.html) | **PASS** |
| 3 | Hertrich–Chambolle–Delon《On the Relation between Rectified Flows and OT》：NeurIPS 2025、reflow 不动点非最优反例（t09） | ✓ arXiv 2505.19712 | ✓ arXiv 官方注释 "Accepted for NeurIPS 2025" | —（无荣誉声明） | ✓ 全文核验：明确构造"Non-optimal Fixed Point of R_p"（Fig.1b/c、Prop.19）、证"零损失≠最优"（§4.2）、推翻 L2022 Thm 5.6 的等价性——笔记与报告头号论断的表述准确 | [arXiv](https://arxiv.org/abs/2505.19712) | **PASS** |
| 4 | 《Wasserstein Proximal Operators Describe SGM...》：SIMODS 2026（t05） | ✓ arXiv 2402.06162（Zhang/Liu/Li/Katsoulakis/Osher，作者一致） | ✓ DOI 10.1137/24M1644584 解析为 SIAM J. Math. Data Sci.，出版 2026-07-15 | — | ✓ WPO=SGM、MFG（FP+HJB）、核模型解记忆化，与笔记一致 | [doi.org/10.1137/24m1644584](https://doi.org/10.1137/24m1644584) | **PASS** |
| 5 | OTP-FM《Multimarginal FM with OT Potentials》：ICML 2026（t27） | ✓ arXiv 2606.05327（Kansal et al.，与本地 PDF 文件名一致） | ✓ arXiv 自注 "Accepted to the Forty-Third ICML"（=2026）+ 笔记附 icml.cc poster 页；[A] 标注恰当 | — | ✓ 中间边缘化为动态 OT 作用量势项、simulation-free、单细胞/海洋/气象数据集，与摘要一致 | [arXiv](https://arxiv.org/abs/2606.05327) | **PASS** |
| 6 | UniDB++：TPAMI 2026（t14） | ✓ arXiv 2505.21528（Pan/Zhu 等 7 人，作者一致） | ✓ Semantic Scholar API：venue=IEEE TPAMI，DOI=10.1109/TPAMI.2026.3710696（与笔记逐字一致）+ PubMed 42412664 双证；dblp 期刊版未收录（滞后）；IEEE 直连 406 反爬 | — | ✓ 闭式逆向 SDE + data-prediction + SDE-Corrector、5–20× 免训练加速、DBIM 为特例，与笔记一致；UniDB 会议版 ICML 2025 亦经 dblp 证实 | [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2505.21528) · [arXiv](https://arxiv.org/abs/2505.21528) | **PASS** |
| 7 | SEDD：ICML 2024 最佳论文（t22） | ✓ | ✓ ICML 2024（PMLR v235） | ✓ 官方 awards 页列 Best Paper（Lou/Meng/Ermon） | ✓ score entropy、压过 GPT-2、32× NFE 削减均与官方摘要一致 | [icml.cc/virtual/2024/awards_detail](https://icml.cc/virtual/2024/awards_detail) | **PASS** |
| 8 | moscot：Nature 2025、170 万细胞（t24） | ✓ | ✓ Nature（s41586-024-08453-2，2025 年刊出；笔记"2025·Nature"准确） | — | ✓ 摘要原文 "1.7 million cells from mouse embryos across 20 time points"；NEUROD2 实验验证亦见于摘要 | [Nature](https://www.nature.com/articles/s41586-024-08453-2) | **PASS** |
| 9 | SW-Guidance：NeurIPS 2025 spotlight（t17） | ✓ arXiv 2503.19034（Lobashev/Larchenko/Guskov） | ✓ OpenReview venue 字段 "NeurIPS 2025 spotlight" | ✓ spotlight | ✓ training-free 可微 SW-1 色彩引导进采样循环、胜"先生成后调色"，一致 | [OpenReview r1Bx58M6It](https://openreview.net/forum?id=r1Bx58M6It) | **PASS** |
| 10 | Golden Noise for Diffusion Models：ICCV 2025（t12） | ✓ arXiv 2411.09502（Zhou et al.） | ✓ ICCV 2025 正式论文集 pp. 17688–17697 | — | ✓ 10 万对噪声数据集（NPD）、SVD 结构先验小网络、即插即用跨模型泛化，与 CVF 摘要一致 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Golden_Noise_for_Diffusion_Models_A_Learning_Framework_ICCV_2025_paper.html) | **PASS** |
| 11 | DeepRUOT：ICLR 2025 Oral（t24） | ✓ arXiv 2410.00844（Zhang/Li/Zhou） | ✓ ICLR 2025 | ✓ 官方 virtual Oral 页 | ✓ Fisher 正则 RUOT、无先验学增殖、Waddington 景观，一致 | [iclr.cc/virtual/2025/oral/31800](https://iclr.cc/virtual/2025/oral/31800) | **PASS** |
| 12 | Immiscible Diffusion：NeurIPS 2024、训练加速 3×（t12） | ✓ arXiv 2406.12303（Li et al.） | ✓ NeurIPS 2024 proceedings | — | ✓ 论文原文 "**up to** 3x"（CIFAR 一致性模型；CelebA 1.3×）+ 22.8ms@1024/A6000——**笔记"最高 3×"准确**；主报告 §0 丢限定词见第 4 节 F3 | [nips.cc poster](https://nips.cc/virtual/2024/poster/93906) | **PASS**（笔记层） |
| 13 | 《On Fitting Flow Models with Large Sinkhorn Couplings》存在性与作者归属（t08） | ✓ arXiv 2506.05526 | ✓ [R] 预印本标注恰当（arXiv 无接收记录，检索亦未见） | — | ✓ 作者 Stephen Zhang / Alireza Mousavi-Hosseini / Michal Klein / Marco Cuturi 与笔记逐字一致；"n 提升 3–4 个数量级（自 256）+ 低 ε 才见收益"与摘要一致（≈10⁵–10⁶） | [arXiv](https://arxiv.org/abs/2506.05526) | **PASS** |
| 14 | Generator Matching：ICLR 2025 Oral（t07） | ✓ arXiv 2410.20587（Holderrieth et al.） | ✓ ICLR 2025 | ✓ OpenReview venue "ICLR 2025 Oral" + 官方 virtual Oral 页 | ✓ 任意 Markov 过程统一 FM/扩散/离散扩散/jump、支持叠加与多模态，一致 | [iclr.cc/virtual/2025/oral/31846](https://iclr.cc/virtual/2025/oral/31846) | **PASS** |
| 15 | 《Gromov-Wasserstein at Scale, Beyond Squared Norms》：ICML 2026（t26） | ✓ arXiv 2602.06658（Houry/Feydy/Vialard，作者一致） | ✓ Feydy 本人主页刊 "ICML 2026"；[A] 标注恰当 | — | ✓ CNT 畸变代价类→lifted 线性对齐、线性内存/二次时间、可微、数十万点分钟级，一致 | [jeanfeydy.com/research](https://www.jeanfeydy.com/research.html) · [arXiv](https://arxiv.org/abs/2602.06658) | **PASS** |

## 3. 自选抽查表（9 条：优先 2026 条目、荣誉标注、漂亮数字）

| # | 论断 | 存在性 | 归属 | 荣誉 | 内容相符 | 证据链接 | 裁定 |
|---|---|---|---|---|---|---|---|
| B1 | HiRef：ICML 2025 **Oral**、百万点全秩 Monge（t29） | ✓ arXiv 2503.03025 | ✓ PMLR v267:21629 | ✓ OpenReview "ICML 2025 oral" + 作者主页 | ✓ 低秩因子共聚类不变量→递归全秩双射、log-linear 时间，一致 | [OpenReview EBNgREMoVD](https://openreview.net/forum?id=EBNgREMoVD) | **PASS** |
| B2 | LLaDA：NeurIPS 2025 **Oral**、8B 比肩 LLaMA3-8B（t22） | ✓ arXiv 2502.09992 | ✓ NeurIPS 2025 proceedings（hash 与笔记链接一致） | ✓ 官方 virtual Oral 页（Dec 5 场次） | ✓ 8B 从零、SFT、reversal curse，一致 | [neurips.cc/virtual/2025/oral/118609](https://neurips.cc/virtual/2025/oral/118609) | **PASS** |
| B3 | MeanFlow：NeurIPS 2025 **Oral**、1-NFE FID 3.43（t07/t10） | ✓ arXiv 2505.13447（Geng et al.） | ✓ NeurIPS 2025 proceedings | ✓ 官方 virtual Oral 页 | ✓ 平均速度恒等式、从零训练、ImageNet-256 FID 3.43，一致 | [neurips.cc oral/130275](https://neurips.cc/virtual/2025/loc/mexico-city/oral/130275) | **PASS** |
| B4 | W-Flow：一步 ImageNet-256 **FID 1.29**、约 100× 加速、Ermon/Candès 署名（t05，[R]） | ✓ arXiv 2605.11755 | ✓ [R] 标注恰当（无接收记录） | — | ✓ 作者 Han/Li/Guo/Xu/Ermon/Candès 与笔记逐字一致；Sinkhorn 散度 WGF→一步蒸馏、FID 1.29、~100×，与摘要一致 | [arXiv](https://arxiv.org/abs/2605.11755) | **PASS** |
| B5 | OT-ALD：AAAI **2026**、提速 20.3%、FID 降 2.6（t17） | ✓ arXiv 2511.11162 | ✓ AAAI-26 官方 OJS，40(31):26760–26768，DOI 10.1609/aaai.v40i31.39886 | — | ✓ 原文 "20.29% / 2.6 on average"（笔记 20.3% 为合理舍入）；DDIB latent 错配定理+OT map 修正，一致 | [AAAI OJS 39886](https://ojs.aaai.org/index.php/AAAI/article/view/39886) | **PASS** |
| B6 | Haxholli et al.：ICML **2026**、转移次数降至 1/32（t22/t08/t29） | ✓ arXiv 2411.00759 v5 | ✓ icml.cc 官方 poster 页（2026-07-06 场次）；[A] 恰当 | —（Poster，笔记未虚标荣誉） | ✓ v5/ICML 版摘要 "up to 32 times (1024 to 32)"；注意旧版（ICLR 2026 投稿）仅claim 8×，笔记取的是录用版数字，正确 | [icml.cc/virtual/2026/poster/65787](https://icml.cc/virtual/2026/poster/65787) | **PASS** |
| B7 | HALO：ICLR **2026** Poster、1024² 图像 8.9× 提速、省 70.5% 显存（t29） | ✓ | ✓ OpenReview venue "ICLR 2026 Poster"（Xia/Zhu/Liang/Zhang） | ✓ Poster（未虚标） | ✓ 8.9×/70.5%@n=1024²、active 剪枝、默认 cuPDLPx，与 OpenReview 摘要逐字一致 | [OpenReview CkOBcyntGd](https://openreview.net/forum?id=CkOBcyntGd) | **PASS** |
| B8 | sCM：ICLR 2025 **Oral**、1.5B、两步 FID 1.88@512（t10） | ✓ arXiv 2410.11081（Lu & Song） | ✓ ICLR 2025 proceedings | ✓ 官方 virtual Oral 页（笔记链接同页）+ 作者主页 "Oral [Top 1.8%]" | ✓ TrigFlow、2.06/1.48/1.88、差距<10%，一致 | [iclr.cc/virtual/2025/oral/31868](https://iclr.cc/virtual/2025/oral/31868) | **PASS** |
| B9 | Self Forcing：NeurIPS 2025 **Spotlight**、单 GPU 17 FPS 实时（t19） | ✓ arXiv 2506.08009（Huang et al.） | ✓ NeurIPS 2025 proceedings | ✓ 官方页 "2025 Spotlight Poster" + 官方 repo | ✓ self-rollout+KV cache、视频级分布匹配、亚秒延迟单 H100 17FPS，一致 | [nips.cc poster/116208](https://nips.cc/virtual/2025/loc/san-diego/poster/116208) | **PASS** |

**荣誉加扫**（主报告直接引用的其余荣誉，全部证实）：RFM《Flow Matching on General Geometries》ICLR 2024 Oral（官方页并显示 Outstanding-Paper **Honorable Mention**——报告只称 Oral，未夸大反而低报）；HIWYN《How I Warped Your Noise》ICLR 2024 Oral（t19 所附 arXiv 2504.03072 与作者主页"2025-04 上传 arXiv"吻合，非异常）；DMD2 NeurIPS 2024 Oral；Go-with-the-Flow CVPR 2025 Oral（官方 virtual Oral 页）。**荣誉通胀家族：0 起。**

## 4. 报告级信号发现

### F1｜证据计数三方不自洽 + "已标注"不实 —— **HIGH**
- **位置**：主报告头部第 4 行、§2、§11；`refs/MASTER_BIBLIOGRAPHY.md` 头部。
- **引文**：报告头部"**451 篇独立文献（[P] 372 正式论文集 / [A] 26 官方接收 / [R] 64 预印本 / [B] 15 教材综述）**"；引用库头部"共 477 条（跨课题重复保留）…去重后**约 446 篇**独立文献"；报告 §11"跨课题重复（**26 条，已在主引用库保留并标注**）""451 篇独立文献中 [P]+[A] 占 88%"。
- **取证**：① 372+26+64+15=**477**，是含重复的引用行口径，被当作"451 篇独立文献"的分解展示（477≠451）；② 本审计对 477 数据行做归一化标题去重，得**约 445 个唯一条目、约 32 条重复行**——与引用库"约 446"吻合，与报告"451 篇/重复 26 条"不符（报告多报约 5–6 篇）；③ 逐行核查重复条目（如 MMFM 同文出现于 T24 与 T27 两节）**无任何逐行重复标注**，仅引用库头部一句总述——"已在主引用库保留并标注"与实物不符；④ "88%"=398/451，分子取 477 行口径、分母取去重口径，混用口径（同口径为 398/477≈83%）。
- **理由**：旗舰证据声明（报告第一屏）在报告与引用库两份文档间互相矛盾，且含一句与实物不符的核验性陈述。**不涉及文献造假**（本审计 28 项外部核验全过、477 行分级计数在引用库层面经复核准确、234 篇 PDF/2.8GB 实测吻合），定性为自评美化+数字自洽性失败。

### F2｜头号论断之二建立在 [R] 预印本上，未按自订纪律加限定词 —— **HIGH**
- **位置**：主报告 §0 第 2 点；§4.2、§9 Top10 #5 沿用。
- **引文**："'OT 耦合增益微弱'的旧共识正在翻盘。2025 年**证明**这是 batch 太小（n≈256）的伪象，n≈10⁶ 级 Sinkhorn 耦合 + 低熵正则才见真收益（T08）"。
- **取证**：该论断唯一来源是 arXiv 2506.05526（[R]，无接收记录；t08 笔记正确标注 [R]）。报告 §2 承诺"**预印本结论一律带 [R] 限定词表述**"，但此处（一页速览三大论断之一、Top10 #5 的支柱）无任何限定词；且"证明"把实验性消融（原文 "We show that…in synthetic and image generation tasks"）说成了定论级结果。
- **理由**：违反报告自我声明的证据纪律，发生在最高杠杆位置。内容与预印本本身相符（非造假），故不升 BLOCKER。

### F3｜"一行代码提速 3×"丢失"最高"限定 —— **ADVISORY**
- **位置**：主报告 §0 第 3 点（"Immiscible Diffusion，一行代码提速 3×"）。
- **取证**：论文原文 "**up to** 3x faster training"，且 3× 仅在 CIFAR 一致性模型上取得（CelebA 1.3×、tiny-ImageNet 1.2×，见 NeurIPS 正式版结论节）；t12/t29 笔记均正确写"最高 3×/加速至 3×"，报告转述时丢限定词。

### F4｜§1 把 OT 并入"两年 3×"口径，与 §8 自述矛盾 —— **ADVISORY**
- **位置**：§1 核验表首行"FM/**OT** 理论词汇渗透率两年 3×"。
- **取证**：t30 口径 A 数据中 OT 提及为 ICLR 33→47→55（1.4×/1.2×）、ICML 33→43→114，报告 §8 自己的判读是"OT 提及**平稳上行**、ICML'26 翻倍"——3× 只适用于 FM。另注：FM"逐年约 3 倍"对第一年（实为 4.3–6.6×）是低估，方向保守、原始数字在 t30 全量披露，不构成夸大。

### F5｜经验现象升格为"理论合法性" —— **ADVISORY**
- **位置**：§5"'耦合是数据内在属性、可离线预计算跨模型复用'（ICML 2024 可复现性现象）给了它**理论合法性**"。
- **取证**：被引工作（Zhang et al., ICML 2024）为经验观察，t12 自己称之为"经验补钉"；"理论合法性"措辞越过证据等级。

### F6｜SynthRAD2025 "无 SB 方法参赛" —— **UNVERIFIABLE**（记录，不判 PASS）
- **位置**：§1、§6、§9 #7（该空位主张直接支撑 Top10 第 3/7 号选题）。
- **取证**：负向存在性断言，本审计无法穷尽核验；kb 内部自洽（t15 同时收录 SynthRAD2025 的 **FM** 参赛方案 [R]（arXiv 2510.04823）并引用挑战报告 arXiv 2605.13555，说明检索确实覆盖了参赛方案且区分了 FM/SB），报告 §11 亦将"SynthRAD2025 结果公布"列为更新触发点。按协议标 UNVERIFIABLE。

### 正面记录（非发现，供后续层参考）
- t27 自曝派单描述口误（NeurIPS 2025→实为 UAI 2026）并与报告 §2 质量事件披露一致；d3LLM（"repo 自称 ICML 2026 未核验"）、MIRROR（"自称 ECCV 2026 待核验"）、EMBC 桥、IJCAI-AOT 等均如实降级 [R] 并注明未核验——证据分级纪律在笔记层执行良好。
- "234 篇本地 PDF（2.8GB）"实测：`papers/` 恰 234 个 PDF、2.8G，吻合。
- 趋势统计（§8/t30）：双口径（OpenReview 提及级/dblp 标题级）、三处未发布会议的剔除、API 复现命令均在 t30 如实交代，报告 §8 数字与 t30 表格逐项一致；除 F4 外口径自洽。
- HIWYN 的 arXiv 2504.03072（2025 年 4 月才上传的 2024 年 ICLR 论文）经作者主页证实，非伪造 ID。

## 5. 总裁定

| 等级 | 数量 | 明细 |
|---|---|---|
| **BLOCKER** | **0** | — |
| **HIGH** | **2** | F1 证据计数三方不自洽+"已标注"不实；F2 头号论断建立在 [R] 上且违反自订限定词纪律 |
| **ADVISORY** | **4** | F3 丢"最高"限定；F4 OT 并入 3× 口径；F5 "理论合法性"措辞；F6 计为 UNVERIFIABLE 记录（不判 PASS 亦不计入 ADVISORY 之外） |

核验总量：**28 项外部核验（15 指定 + 9 自选 + 4 荣誉加扫）＝ 28 PASS / 0 FLAG / 0 BLOCKER**。指定靶标中最可疑的画像（2026 年会议、Oral/最佳论文荣誉、161×/1.29 FID/170 万细胞等漂亮数字、Ermon–Candès 级署名）逐一经官方源证实为真；[P]/[A]/[R] 分级在笔记层与实况一致，未发现任何一例"预印本冒充接收"或荣誉虚标。

**声明：no-new-blocker ≠ acquittal。**本轮未发现新的 BLOCKER，仅意味着在本轮抽样深度（28/451+ 条文献、四维核验、报告级五族扫描）下未检出造假证据，不构成对知识库全部 477 条引用行、全部数字与全部"空位/空白"主张（尤其负向存在性断言）的无罪认定。两条 HIGH 发现（证据计数不自洽、[R] 纪律在头号论断处失守）应在下一修订轮修复后复核；未抽样的约 420 条引用行仍处于未审计状态。
