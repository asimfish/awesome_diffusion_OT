# 扩散模型 × 最优传输：深度调研报告

> 日期：2026-08-14 ｜ 方法：30 个并行文献调研 agent + 主控聚合审计（流水线参考 PaperOrchestra，证据纪律参考 PaperSpine）
> 证据底座：**477 条引用行（[P] 372 正式论文集 / [A] 26 官方接收 / [R] 64 预印本 / [B] 15 教材综述；跨课题重复约 30 条，按标题归一化去重约 445–455 篇）+ 234 篇本地 PDF（2.8GB）**
> 修订记录：2026-08-14 按 ARIS Layer-2 敌意审稿修正数字口径与 [R] 限定表述（审计全文见 `_audit/`）；2026-08-25 §11 触发点增量复审——SynthRAD2025 空位经官方报告证实、FlashSinkhorn 无新 release、ECCV/NeurIPS 未到期（见 `_audit/INCREMENTAL_REVIEW_20260825.md`）
> 知识库：`~/Desktop/research/diffusion_ot_survey/`（本报告为总纲，30 份子课题笔记在 `kb/`，逐条出处见 `refs/MASTER_BIBLIOGRAPHY.md`）

---

## 0. 一页速览

**博客的大方向判断成立，且比作者说的更有弹药**：扩散×OT 在 2024→2026 处于爆发期（flow matching 顶会提及量连续两年约 3 倍增长），理论、算法、落地三层都有明确空位。但有三处需要修正后再讲故事，否则会被审稿人反杀：

1. **"直线轨迹 ≈ 最优传输"是陷阱**。NeurIPS 2025 已给出正式反例：rectified flow 的不动点可以不是 OT 映射，"损失趋零"也不等于最优（T09）。同期 Khrulkov 猜想（DDPM encoder ≈ Monge map）也早被 Lavenant–Santambrogio 反例证伪（T02）。**正确讲法**：OT 是设计语言和正则化手段，不是扩散模型自动实现的性质——"量化次优度"本身就是公开的可发论文问题。
2. **"OT 耦合增益微弱"的旧共识正在被挑战**。2025 年的预印本证据（Zhang/Klein/Cuturi，arXiv 2506.05526，**[R] 级、待正式接收**）表明这是 batch 太小（n≈256）的伪象，n≈10⁶ 级 Sinkhorn 耦合 + 低熵正则才见真收益（T08）；但 3D 点云上又有"完全 OT 反而更难学"的反证（T20）。**"OT 该用多少"是精确可研究的量，不是站队问题。**
3. **方向一的正确定位是"推理/数据管线的耦合工程"**。真正免重训的插入点有七个环节（T12），从数据管线噪声指派（Immiscible Diffusion，一行代码、**最高** 3× 提速——数据集相关，CelebA 约 1.3×）到黄金噪声传输映射（ICCV 2025），这条线已有正式论文集级先例且空位明确。

**最值得投入的三件事**（完整 Top 10 见第 9 节）：
- **免训练 batch 级保边缘噪声重排**：Hungarian/Sinkhorn 一次性指派替代 top-1 检索，理论（保边缘+方差降低）与实验都是空位，算力需求小（T12）；
- **保 OT 耦合的单步桥蒸馏 + 耦合漂移度量**：蒸馏后终端耦合是否漂移目前零保证，医学模态转换直接受益（T14/T15；SynthRAD2025 官方挑战报告 arXiv 2605.13555 证实 25 队零 SB 参赛，2026-08-25 复审核验）；
- **OT-aware 采样调度理论**：把 Align-Your-Steps 的 KL 泛函换成 Benamou–Brenier 动能泛函，证"等旋转误差 = W₂ 最优调度"（T11），纯理论+免训练实验，正中"数学严谨性"的审稿口味。

---

## 1. 动机核验：博客论断 vs 证据

博客核心论点：感知分数内卷终结 → 审稿人转向数学严谨性 → 扩散×OT 是未来 1–2 年高校实验室的高性价比赛道，两个具体方向是"无须重训的轨迹对齐"与"OT 引导跨域生成"。逐条对照本库 477 条引用核验：

| 博客论断 | 核验结论 | 关键证据（出处笔记） |
|---|---|---|
| 纯改结构刷分难生存，社区转向采样加速的数学严谨性 | **方向一致**。FM 提及量两年约 3×、OT 提及平稳上行（ICML 2026 翻倍）；ICML 2024 SEDD、ICLR 2024 Oral RFM 等最佳论文级荣誉全部给了"公式化"工作 | T30 趋势统计、T07、T22 |
| 扩散本质是高维布朗运动，少步数会崩 | **基本成立但要说准**：前向是 OU 过程；少步崩溃的根因是 PF-ODE 轨迹弯曲+离散化误差，曲率可测可优化 | T11（GITS 轨迹几何）、T09 |
| 引入 OT = 给生成轨迹加最短路径约束 | **要加限定**：reflow 拉直≠收敛到 OT（反例已发表）；OT 是可选的耦合/正则设计语言 | T09、T02 |
| 流匹配+几何测度理论的故事受欢迎 | **成立**：FM 已是 SD3/FLUX 等大模型默认公式；ICLR/ICML/NeurIPS 是主战场，CVPR 是应用出口 | T30、T07 |
| 不需要卷从头预训练，设计映射机制换算力 | **成立**：方向一全线免重训；方向二可用预训练模型做桥（DDIB 式串联） | T12、T14 |
| 端侧图像生成、高频视频渲染是好落地 | **成立且有数据**：SnapGen 379M 手机 1.4s 出 1024²；视频侧 VDOT（CVPR 2026）已把熵 OT 写进蒸馏目标；视频 reflow 仍是空白 | T30、T19 |
| 医学模态转换是方向二的价值垂直 | **成立且有具体空位**：SB 系已入 Medical Physics 剂量学验证；SynthRAD2025 官方报告证实零 SB 参赛（8/25 已核验） | T15 |

**修正后的调研动机（本报告采用）**：不做"OT 万能"叙事；把 OT 当作耦合选择、调度设计、蒸馏正则、跨域先验四种可插拔机制，每处都以"可证明的界 + 免重训或轻训练的实验"为纲。

---

## 2. 调研方法与证据口径

**流水线**（PaperOrchestra 式七阶段的调研版）：任务拆解（6 板块 30 子课题，边界互斥）→ 30 个 agent 并行检索（种子库 `ot_variants_survey` + WebSearch + arXiv API）→ 会议归属官方核验（PMLR / OpenReview venue 字段 / CVF / ECVA / AAAI OJS / NeurIPS proceedings；"submitted to"一律不算）→ 按七节模板写结构化笔记 → 下载核心 PDF 并校验（`%PDF` 头 + >50KB）→ 主控机械聚合（477 条引用行）→ 本报告综合。

**证据分级**（PaperSpine 式句级支撑纪律）：[P] 正式论文集 / [A] 官方已接收 / [R] 预印本 / [B] 教材综述。本报告正文所有强论断都能回溯到 `kb/` 笔记中带链接的表行；预印本结论一律带 [R] 限定词表述。

**质量事件记录**：T22 首次运行被上游安全策略误拦截（零产出，重试成功）；T01 有 1 篇 PDF 因 arXiv 限流放弃（按纪律记失败不硬重试）；T27 核验出派单描述中一处会议归属口误（实为 UAI 2026），种子库本身无误；T23 发现两篇撤稿预印本并如实降级为 [R]。

**已知盲区**：ECCV 2026 论文集未发布（官方窗口 2026-08-13 起）；NeurIPS 2026 尚未放榜；CVPR 2026 dblp 未完全收录。这三处在第 8 节趋势判读中已标注口径。

---

## 3. 领域全景图

```
                    ┌─ A 理论基础 ────────────────────────────────┐
                    │ T01 数学装备  T02 扩散≟OT  T03 SB 理论      │
                    │ T04 熵正则   T05 WGF/JKO  T06 收敛统计      │
                    └──────────────┬──────────────────────────────┘
                                   │ 提供：耦合语言·动态形式·误差界
        ┌──────────────────────────┼───────────────────────────────┐
        ▼                          ▼                               ▼
┌─ B 轨迹拉直(方向一) ─┐  ┌─ C 跨域生成(方向二) ─┐  ┌─ E OT 变体前沿 ────┐
│ T07 FM 谱系          │  │ T13 神经 OT map      │  │ T25 UOT/partial    │
│ T08 OT-CFM 耦合      │  │ T14 扩散桥 I2I       │  │ T26 GW 跨空间      │
│ T09 Rectified Flow   │  │ T15 医学模态转换 ★   │  │ T27 MMOT/重心      │
│ T10 蒸馏的 OT 视角   │  │ T16 OT 语义对应      │  │ T28 黎曼流形 FM    │
│ T11 免训练求解器     │  │ T17 风格/域适应      │  └────────────────────┘
│ T12 推理期 OT 对齐 ★ │  │ T18 条件 OT/guidance │
└──────────┬───────────┘  └──────────┬───────────┘
           │      落地模态（D 板块）  │
           ▼                          ▼
┌─ D: T19 视频 · T20 3D · T21 分子 · T22 离散文本 · T23 语音 · T24 单细胞 ─┐
└─ F 系统底座: T29 FlashSinkhorn/求解器栈 · T30 端侧部署/评测/趋势 ────────┘
```

**演进大事记**（详细谱系见各笔记第 3 节）：
- 2021–2023 奠基：DSB（SB 进生成）、Flow Matching、Rectified Flow、NOT/I2SB/DDIB（翻译三范式）、Khrulkov 猜想与反例；
- 2024 展开：OT-CFM（TMLR）、DDBM 统一桥设计、Immiscible Diffusion（数据管线耦合）、SD3 采用 RF、EDM/AYS/GITS 把采样调度原理化、FM 进语音/分子/流形；
- 2025 收口与翻盘：stochastic interpolants 统一理论（JMLR）、IMF 指数收敛率、"reflow≠OT"反例、大规模 Sinkhorn 耦合挑战"OT 无用论"（[R] 预印本）、SVDQuant/SnapGen 端侧转折、moscot 推到 170 万细胞；
- 2026 制高点：FlashSinkhorn（ICML Oral，Sinkhorn=attention 同构）、OTP-FM（多边际势能松弛）、VDOT（熵 OT 进视频蒸馏，CVPR）、UniDB++（TPAMI）、SGM=WPO 理论桥（SIMODS）。

---

## 4. 理论主线：审稿人在乎的"数学严谨性"从哪来

### 4.1 辩论线一："扩散≟OT"——从猜想、反例到量化次优度

Khrulkov et al.（ICLR 2023）证明高斯情形 DDPM encoder 恰为 Monge map 并猜想一般成立；Lavenant & Santambrogio 用 3 页反例（Hessian 非交换障碍）证伪；2026 年 Dumont et al. 延续这条线。**当前状态**：PF-ODE 映射"接近但不等于"OT，而"接近多少"（次优度的量化上界与地形图）是原作者明示的 open problem，且高斯精确 W₂ 解（ICML 2025）+ tensor-train FP 求解器等工具已齐备（T02）。

### 4.2 辩论线二："拉直≠OT"——rectified flow 的理论边界

Liu et al. 的 reflow 以 O(1/K) 速率拉直轨迹，一轮 reflow 实践上就足够直（NeurIPS 2024）；但 NeurIPS 2025 反例证明 reflow 不动点可以非最优、训练损失趋零≠OT。**含义**：拉直的价值在于少步采样（NFE-质量），不在"最优性"；补"何种数据条件下 reflow 不动点唯一且=W₂-OT"的正定理是清晰的理论空位（T09）。3D 侧 NSOT（ICLR 2025）进一步显示完全 OT 耦合会让 t≈0 的速度场更难学——**耦合"OT 程度"与可学性存在 trade-off**（T20），与 T08 的大 batch 证据（[R] 预印本）合起来构成"OT 用量"这个新研究变量。

### 4.3 收口线：SGM = Wasserstein proximal、SB 五代成熟

SIMODS 2026 证明 score-based 生成模型隐式实现 Wasserstein proximal 算子并以此解释/缓解记忆化（T05）；SB 侧经 IPF→IMF/matching→轻量化（LightSB-M 闭式势）→在线/离散/多边缘→NeurIPS 2025 IMF 指数收敛率的五代演进，2026 年已有专著收口（T03）。stochastic interpolants（JMLR 2025）把 flows/diffusions/SB 统一进一个框架（T07）。**含义**：底层理论骨架已稳，新论文的增量应放在"学习误差下的收敛""耦合的统计性质""推理期机制"这些骨架之上的空位，而不是再造框架。

### 4.4 工具箱线：收敛与统计理论

扩散收敛界已从多项式推进到 O(d/T)（TV，最弱假设）与内在维数自适应 O(k/T)；FM 有近 minimax 保证；OT map 估计率（plug-in / entropic / smooth）体系完整（T06）。**关键空白**：现有 FM 统计理论只覆盖独立耦合——**OT-CFM 的端到端 W₂ 收敛率**（以熵 map 估计率为中间量）是 diffusion×OT 的天然交叉定理；"加速是否吃掉统计精度"可用三项误差分解（map 估计+score+离散化）+ 迭代×样本联合下界正面回答（T06×T08）。

**写论文的三条红线**（从 30 份笔记的失败案例与反例总结）：① 不说"我们的方法实现了最优传输"，说"以 OT 为设计目标并度量偏差"；② 少步质量比较必须固定 NFE 口径并报告多样性指标（FID 对少步模型系统性失真，见 T30）；③ 会议归属与"已接收"表述必须官方可核验（本知识库 [P]/[A]/[R] 全部带链接）。

---

## 5. 方向一深评：无须重训的轨迹对齐

**成立性**：证据充分。这条线已有 NeurIPS 2024（Immiscible Diffusion）、ICCV 2025（Golden Noise）等正式论文集先例，且"耦合是数据内在属性、可离线预计算跨模型复用"（ICML 2024 可复现性现象）给了它理论依据（T12）。

**七个可插入测度耦合的环节**（按管线顺序，来自 T12，逐环节代表工作见笔记）：
1. 数据管线噪声指派（训练前，Immiscible/量化指派 22.8ms@1024）；
2. 初值检索/搜索（NoiseQuery、top-1-of-k 选择）；
3. 初值连续变换（噪声优化、Golden Noise 传输映射）；
4. OT 桥接 prior（一步半离散 OT + 短程扩散，Monge–Ampère 理论化）；
5. 轨迹中段对齐（采样中途投影/引导）；
6. 跨帧噪声传输（视频 ∫-noise，ICLR 2024）；
7. inversion 侧耦合（编辑一致性）。

**明确空位**（可直接立项）：
- **batch 级保边缘重排**：环节 2 与 5 之间的空白——batch 内 (条件, 噪声) 一次性 Hungarian/Sinkhorn 指派，每个噪声恰用一次、边缘严格保持；理论卖点是保边缘性+方差降低，实验用 T2I-CompBench/GenEval（T12）；
- **实例级选择的分布漂移理论**：top-1-of-k 是 order-statistics 耦合，其 W₂/KL 漂移上界无人刻画（T12）；
- **OT-aware 调度**：AYS 的 KLUB 换成 Benamou–Brenier 动能泛函；样本级"距 OT 偏差"（弧长/端点距离比）驱动自适应 NFE 分配（T11）；
- **视频 reflow**：图像 reflow 成熟、视频公开工作缺席；时空分解 reflow + 直耦合与时序一致性的冲突分析（T19）。

**风险与审稿攻击点**：改耦合 vs 保边缘的张力（任何实例级挑选都使有效初始分布偏离标准高斯）——必须报告多样性/漂移指标；与"更大 teacher 蒸馏"基线的对比要公平（固定算力口径）；"免训练"卖点要求方法对采样器/模型即插即用（跨 SD3/FLUX 验证）。

---

## 6. 方向二深评：OT 引导跨域生成

**三条技术路线**（互补而非竞争，接口处是空位）：
1. **神经 OT map 直译**（T13）：NOT 系 saddle-point 求解器 + UOTM 非平衡分支；结构保持强、纹理弱；评测基建老化（2021 W2 benchmark 已测不动 2024-2026 方法）；
2. **扩散桥/SB 翻译**（T14）：I2SB/DDIB/DDBM→UniDB++（TPAMI 2026）；质量高但采样贵，蒸馏后"终端耦合是否漂移"零保证；
3. **OT 代价先验做 guidance/语义对应**（T16/T17/T18）：OTCS（NeurIPS 2023）开创"OT 耦合=跨域配对先验"；FGW 外观+结构代价（CVPR 2026）、attention 重释为 OT 的采样引导（AAAI 2026）、sliced Wasserstein 色彩引导（NeurIPS 2025 spotlight）；条件几何的正确度量已由 JMLR 2025 给出（joint W₂ 不控制 posterior W₂）。

**医学垂直**（博客点名的价值场景，T15）：SB 系已过剂量学验证关（DSBM MR→CT，Medical Physics 2025）、无配对 CBCT→CT（MICCAI 2025）、跨设备 OCT 协调（NeurIPS 2025）。**具体空位**：SynthRAD2025 官方挑战报告（arXiv 2605.13555，2026-08-25 核验）证实 25 队提交零 SB/OT 方法——解剖商空间传输成本 + 3D 耦合 SB 是一条能直接刷公开榜的路线；桥后验 conformal 校准筛"解剖幻觉"是安全性卖点。

**跨路线空位**：一步 OT map（快、结构强）+ 少步扩散 refinement（纹理强）的混合管线（T13×T14 接口）；非配对桥在可算 ground-truth EOT 基准上的"耦合误差 vs FID"检验——"更接近最优传输是否等于更好翻译"至今没被正面回答（T14）。

**风险**：医学赛道数据合规与评测协议成本高；语义对应线要防"OT 只是 attention 重命名"的质疑（需消融证明耦合结构本身带来增益）；guidance 强度与分布偏移的理论（CFG 的 W₂ 刻画，T18）尚缺，先证界再上大实验更稳。

---

## 7. 高价值周边方向速览

| 模态/变体 | 成熟度 | 一句话状态 | 空位举例（出处） |
|---|---|---|---|
| 语音（T23） | ★★★★ 最成熟落地 | TTS 收敛到"FM+infilling"极简范式，SB 统治增强/修复 | Sway Sampling 理论化；首个非配对 SB voice conversion |
| 分子/科学（T21） | ★★★★ 全 [P] 证据 | SE(3) 流形 OT 流范式确立（FoldFlow/FlowMM） | 可扩展群-quotient 等变 OT 耦合；条件黎曼 OT 统一跨任务框架 |
| 单细胞（T24） | ★★★★ | moscot 推到 170 万细胞规模，unbalanced 动态线活跃 | 免仿真 unbalanced SB 收敛理论；推理期 transport steering |
| 视频（T19） | ★★★ 快速上升 | VDOT 把熵 OT 写进视频蒸馏（CVPR 2026）；∫-noise 奠基帧间传输 | 视频 reflow 完全空白；帧间噪声耦合的 OT 理论化 |
| 3D/点云（T20） | ★★★ | Wasserstein FM 范式跃迁（把每朵点云当 W 空间的点） | Bures-Wasserstein 几何上直接生成 3DGS 资产 |
| 离散/文本（T22） | ★★★ | SEDD（ICML 2024 最佳论文）后 masked diffusion 起飞 | Edit Flows×unbalanced OT；离散 kinetic-optimal 路径定理 |
| UOT/partial（T25） | ★★★ | "非平衡 Monge=重缩放平衡映射"使 UOT 即插即用 | WFR 生灭率作推理期模式再平衡；污染率自适应 τ |
| GW（T26） | ★★★ | CNT 代价类让 GW 线性内存进工作流（ICML 2026） | 跨维 Gromov-Schrödinger 桥；plan 稳定性→生成误差界 |
| MMOT/重心（T27） | ★★★ | OTP-FM 统一多边际 FM（ICML 2026） | OT 势即插即用引导；扩散模型 barycentric merging |
| 黎曼流形（T28） | ★★★ | RFM（ICLR 2024 Oral）奠基，WFM 提升到分布族空间 | 黎曼 rectified flow 完全空白；BW 空间脑影像增广 |
| 系统（T29） | ★★★ | FlashSinkhorn＝attention 同构（ICML 2026 Oral），161× | 分布式 IO-aware Sinkhorn 空白；OT 配对开销统一 benchmark |
| 端侧/评测（T30） | ★★★ | 范式转向"为端侧从头设计 FM 模型"（SnapGen/SANA） | 直线度作可部署性代理指标；DINOv2 特征 Sinkhorn divergence 替代 FID |

---

## 8. 顶会趋势与竞争格局

**数据**（T30，口径 A=OpenReview/proceedings 提及级，口径 B=dblp 标题级作下界；ECCV 2026/NeurIPS 2026 未发布、CVPR 2026 dblp 未全收录已剔除）：

- Flow matching 提及级接收数**逐年约 3 倍**：ICLR 2024→2026：7→46→144；ICML：13→56→167；NeurIPS 2023→2025：6→32→88。FM 已从"方法词"变成"基础设施词"。
- OT 提及平稳上行，ICML 2026 翻倍（43→114），主要作为 FM 文献的理论词汇渗透（minibatch 耦合、W₂ 界）。
- "Rectified flow"术语在 NeurIPS 2025 达峰后走平——被 FM 吸收（SD3 效应）。
- CVPR 标题级 FM/OT 极少（≤8/年）而 diffusion 标题 300+/年：**理论主战场在 ML 三会，CVPR/MICCAI 是应用出口**——与博客"发 ICLR 或 CVPR"的投递建议一致，但分工不同：理论证明投 ML 三会，垂直落地（医学/视频/端侧）投 CV 会。

**竞争判读**：耦合工程（方向一）与桥式翻译（方向二）都已有大组进场（Meta/NVIDIA/Google 在 FM 基础设施，Vector/Mila 在 SB/CFM，Skoltech 系在神经 OT），但本报告第 5/6 节列出的空位多为**理论补全型**和**接口缝合型**——恰是高校实验室相对大厂的比较优势（不需要预训练算力，需要证明与精巧实验）。窗口期判断：以 FM 渗透速度和 2026 年已出现的收口性专著看，**理论空位的窗口约 12–18 个月**，落地垂直（医学/端侧）窗口更长。

---

## 9. 可发论文切入点 Top 10

从 30 份笔记约 60 个候选中按「新颖性 × 可行性（无需大规模预训练）× 与博客两方向的契合度 × 审稿风险」综合排序：

| # | 切入点 | 类型 | 方向 | 算力 | 出处 |
|---|---|---|---|---|---|
| 1 | 免训练 batch 级保边缘噪声重排（Sinkhorn 一次性指派 + 保边缘/方差理论） | 方法+理论 | 一 | 低 | T12 |
| 2 | OT-aware 采样调度：Benamou–Brenier 动能泛函替代 KLUB，证 W₂ 最优调度定理 | 理论+免训练 | 一 | 低 | T11 |
| 3 | 保 OT 耦合的单步桥蒸馏 + 耦合漂移度量（医学模态转换直接受益） | 方法 | 二 | 中 | T14/T15 |
| 4 | reflow 正定理：何种数据条件下不动点唯一且=W₂-OT（先高斯混合族完全刻画） | 理论 | 一 | 低 | T09 |
| 5 | OT-CFM 端到端统计收敛率（熵 map 估计率作中间量），回答"加速 vs 统计精度" | 理论 | 一 | 低 | T06/T08 |
| 6 | PF-ODE 次优度量化（原作者 open problem，工具已齐：高斯精确解+FP 求解器） | 理论 | 通用 | 低 | T02 |
| 7 | 解剖商空间成本 + 3D 耦合 SB 刷 SynthRAD2025（官方挑战报告证实 25 队零 SB/OT 参赛，post-challenge 榜开放至 2030-03；赛中 FM/扩散系剂量指标显著逊于 CNN/GAN，入场须以剂量学为设计目标）+ conformal 幻觉筛查 | 应用 | 二 | 中 | T15 |
| 8 | FGW 语义对应闭环进扩散采样：编辑一致性引导（判别线与生成线的缝合） | 方法 | 二 | 中 | T16 |
| 9 | 视频侧：时空分解 reflow（空白）或帧间噪声 Sinkhorn 耦合替代光流 warp | 方法 | 一 | 中高 | T19 |
| 10 | OT 系评测指标：DINOv2 特征上 Sinkhorn divergence 替代 FID + 人评相关性（D&B） | 评测 | 通用 | 低 | T30 |

**第二梯队**（各笔记第 5 节共 ~50 条，择要）：CFG 分布偏移的 W₂ 刻画与最优 w 调度（T18）；WFR 生灭率作推理期模式再平衡（T25）；OTP-FM 势转 guidance 的误差传播界（T27）；黎曼 rectified flow（T28）；跨维 Gromov-Schrödinger 桥（T26）；分布式 IO-aware Sinkhorn（T29）；MeanFlow/flow map 的 W₂ 误差理论（T07）；OT 耦合式一致性训练冲击维度上界（T10）；Sway Sampling 理论化与非配对 SB 声转换（T23）；直线度×量化复合误差界（T30）。

**选题组合建议**：#1+#2 构成"推理期耦合工程"连击（共享代码基建）；#3+#7 构成"医学桥"主线（方法+落地各一篇）；#4/#5/#6 是纯理论线，适合数学背景强的学生单兵作战。

---

## 10. 12 周行动建议

- **W1–2 装备**：Peyré《OT for Machine Learners》(2025) 通读 → Benamou–Brenier 动态形式精读；用高斯族 Bures–Wasserstein 闭式解搭"随学随验"沙盒（T01 路线）。同步跑通 torchcfm + POT/OTT-JAX。
- **W3–4 复现三件套**：Immiscible Diffusion（一行代码级）、OT-CFM（torchcfm 官方）、DPM-Solver++/AYS 调度对比；在 CIFAR/小 T2I 上建立 NFE-质量-多样性三轴评测脚手架（直接服务 #1/#2）。
- **W5–8 立项冲刺**：主攻 #1（batch 保边缘重排）：先跑 top-1 检索 vs Sinkhorn 指派的对照，同步写保边缘引理；#2 作为并行理论线推导动能泛函调度。产出 workshop 级初稿。
- **W9–12 扩展与投递决策**：若 #1 实验强 → 冲 ICLR 2027 主会；若理论线（#2/#4/#5）先成熟 → 补 GMM 闭式验证投 ICML；医学线（#3/#7）按 MICCAI 2027 时间表布局数据合规。
- **持续动作**：ECCV 2026 论文集（8 月下旬起）与 NeurIPS 2026 放榜（9 月底）后按第 11 节触发点增量更新知识库。

---

## 11. 质量审计与局限（PaperSpine 式收尾）

**覆盖度自检**：30/30 子课题笔记七节齐全；477 条引用中 [P]+[A] 共 398 条、占 83%（按去重约 450 篇计约 88%），杜绝了"预印本冒充接收"；博客点名的四个关键词（FlashSinkhorn、黎曼流形、Barycenter、流匹配）分别由 T29/T28/T27/T07-T12 深度覆盖；两个方向各有专属课题群与空位清单。

**证据分布的诚实说明**：[R] 64 篇集中在 2025H2–2026 前沿（如视频蒸馏、GW 桥、黎曼 NOT），结论以"待正式接收"限定；T23 两篇撤稿预印本已降级标注；趋势统计为提及级口径，作"主题热度上界"解读，标题级口径作下界。

**方法局限**：① 30 agent 并行检索存在少量跨课题重复（约 30 条，26–32 之间随标题归一化规则浮动；在主引用库中按课题分组保留、未逐行标注）；② 检索以英文顶会为主，期刊（TPAMI/TMLR/JMLR/SIMODS）覆盖到但非穷尽，中文社区实践经验未纳入；③ 顶会趋势数字受 OpenReview 反爬与 dblp 收录延迟影响，已双口径对冲；④ "算力需求"评级基于论文报告的规模推断，未实测。

**下次更新触发点**（2026-08-25 增量复审，详见 `_audit/INCREMENTAL_REVIEW_20260825.md`）：ECCV 2026 论文集上线——8/25 查 ECVA 尚未上线，9 月会议期复查，届时补 T14/T16/T17/T19/T20 的 [P] 条目；NeurIPS 2026 放榜（9 月底）；FlashSinkhorn——8/25 已核：自 v0.3.3（2026-04）后无新 release，仍限平方欧氏 cost 与单卡，T29 空位 #1/#2 未被填，ICML 2026 卷未上 PMLR 故维持 [A]；SynthRAD2025——8/25 已核：挑战报告 [arXiv 2605.13555](https://arxiv.org/abs/2605.13555) 证实 25 队零 SB/OT 参赛，T15 空位成立，post-challenge 榜开放至 2030-03（已同步 §9 #7 与幻灯片）。

---

## 附录：30 子课题一览

| 板块 | 课题 | 一句话亮点 |
|---|---|---|
| A | T01 OT 数学基础 | Peyré 2025 双新讲义 + BW 闭式沙盒学习法 |
| A | T02 扩散≟OT | 猜想-反例辩论线完整谱系，次优度量化是 open problem |
| A | T03 Schrödinger 桥 | 五代演进收口，IMF 指数收敛率 + 2026 专著 |
| A | T04 熵正则/Sinkhorn | ε 从超参变调度对象（PROGOT），QOT 稀疏耦合无人用于生成 |
| A | T05 WGF/JKO | SGM=WPO 理论桥（SIMODS 2026），S-JKO 达扩散级 FID |
| A | T06 收敛统计 | O(d/T)+内在维数自适应；OT-CFM 统计率空白 |
| B | T07 FM 谱系 | SI 统一框架（JMLR 2025）；噪声调度理论-实践鸿沟 |
| B | T08 OT-CFM | 大规模 Sinkhorn 挑战"OT 无用论"（[R]）；期望 batch plan 理论 |
| B | T09 Rectified Flow | "直≠OT"反例定论；一轮 reflow 即够直 |
| B | T10 蒸馏 OT 视角 | VDOT 熵 OT-DMD（CVPR 2026）；W₂ 上界理论地基 |
| B | T11 免训练求解器 | GITS 轨迹几何 + AYS 调度原理化；OT-aware 调度空白 |
| B | T12 推理期对齐 | 七环节清单；保边缘重排是最干净的空位 |
| C | T13 神经 OT 翻译 | NOT 范式 + UOTM 分支；评测基建老化待更新 |
| C | T14 扩散桥 I2I | I2SB/DDIB/DDBM→UniDB++；保耦合蒸馏零保证 |
| C | T15 医学模态 | 剂量学验证已过关；SynthRAD2025 官方证实零 SB 参赛（8/25 核验） |
| C | T16 语义对应 | OTCS 奠基；FGW 代价设计模板（CVPR 2026） |
| C | T17 风格/域适应 | SW-Guidance 免训练色彩引导（NeurIPS 2025 spotlight） |
| C | T18 条件 OT/guidance | 条件 W 距离的正确几何（JMLR 2025）；CFG 的 W₂ 刻画空白 |
| D | T19 视频 | VDOT + ∫-noise；视频 reflow 完全空白 |
| D | T20 3D/点云 | "完全 OT 反而难学"反证；W 空间 FM 范式跃迁 |
| D | T21 分子/科学 | 全 [P] 证据；SE(3) OT 流范式（FoldFlow/FlowMM） |
| D | T22 离散/文本 | SEDD 最佳论文后起飞；唯一显式离散动态 OT（ICML 2026） |
| D | T23 语音 | 落地最成熟；Sway Sampling 理论化是现成抓手 |
| D | T24 单细胞 | moscot 170 万细胞；unbalanced 动态线（DeepRUOT Oral） |
| E | T25 UOT/partial | UOT 即插即用定理；WFR 第二控制量思路 |
| E | T26 GW | CNT 代价类线性内存（ICML 2026）；跨维桥空白 |
| E | T27 MMOT/重心 | OTP-FM 制高点；势转 guidance 空白 |
| E | T28 黎曼 FM | RFM Oral 奠基；黎曼 reflow 完全空白 |
| F | T29 求解器 | FlashSinkhorn=attention 同构 161×；分布式空白 |
| F | T30 端侧/趋势 | FM 提及 3×/年；"从头设计 FM 端侧模型"范式转折 |

*报告完。逐条出处：`refs/MASTER_BIBLIOGRAPHY.md`（477 行带链接）；作业规范与复现方法：`_brief/AGENT_BRIEF.md`。*
