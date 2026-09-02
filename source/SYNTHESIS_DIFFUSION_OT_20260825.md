# 扩散模型 × 最优传输：深度综合报告

> 版本：2026-08-25 ｜ 性质：在 30 课题知识库（477 条引用、234 篇 PDF、ARIS 审计通过）之上的**综合与洞察层**，按「问题是什么 → 别人的理论 → 经典工作 → 最新工作 → 我们能做什么」重组织
> 方法：PaperOrchestra 式管线（大纲 → 文献综合 → 分节写作 → 内容精炼 → 图示），证据纪律沿用 PaperSpine（[P] 正式论文集 / [A] 官方接收 / [R] 预印本 / [B] 教材综述，逐条可溯源至 `kb/` 与 `refs/MASTER_BIBLIOGRAPHY.md`）
> 与主报告的分工：`REPORT_DIFFUSION_OT_20260814.md` 是调研收口报告（动机核验/趋势/Top-10）；本文档是**深度综合**——形式化问题、梳理定理骨架、给出经典与前沿的完整地图、沉淀跨课题洞察。8/25 新增：SynthRAD2025 官方核验、FlashSinkhorn 复查、以及 8 月新论文（c-RF 理论、Beckmann 传输模型等）
> HTML 版：`SYNTHESIS_DIFFUSION_OT_20260825.html`（同内容，带全景图与交互导航）

---

## 0. 导读：五个问题的一句话答案

| 问题 | 一句话答案 |
|---|---|
| **问题是什么** | 生成建模本质是把噪声/源分布**搬运**成数据分布；搬运的路径几何（曲率→NFE）、耦合选择（谁配谁）、最优性（是不是 OT）、跨域语义（保什么改什么）、计算与统计代价，构成五个可形式化的核心问题 P1–P5 |
| **别人的理论** | 骨架已收口：扩散 = KL 的 Wasserstein 梯度流（JKO/WPO），SB = 熵正则 OT 的动态形式（IMF 指数收敛），FM 统计上与扩散等价（almost minimax）；但「扩散 encoder ≈ OT」被反例证伪、「拉直 = OT」被反例证伪——OT 是设计语言，不是自动性质 |
| **经典工作** | 数学经典五件套（Brenier 1991 / McCann 1997 / JKO 1998 / Benamou–Brenier 2000 / Otto 2001）+ 计算三件套（Cuturi 2013 Sinkhorn / Score-SDE 2021 / DSB 2021）+ 范式三线（FM / RF / SI，2022–2023 同期奠基） |
| **最新工作** | 2025–2026 三个制高点：理论收口（IMF 指数率、FM minimax、O(d/T)、**8 月新出的 c-RF 收敛定理**）、一步生成新范式（MeanFlow / W-Flow / **Beckmann 自治流**）、基础设施拐点（FlashSinkhorn＝attention 同构，ICML 2026 Oral） |
| **我们可以做什么** | 高校比较优势在**理论补全型**与**接口缝合型**空位：免训练 batch 级保边缘重排（#1）、OT-aware 采样调度（#2）、保耦合蒸馏（#3）、医学 SB 刷榜（#7，8/25 官方证实零 SB 参赛且榜单开放至 2030）——完整机会地图见 §6 |

**先读什么**：赶时间读 §5（十条洞察）+ §6.1（Top-10 更新版）；要理论装备读 §2；要文献地图读 §3–§4。

---

## 1. 问题是什么：五个可形式化的核心问题

### 1.0 元问题

生成建模 = 测度传输：构造映射/过程把易采样的 \(\mu_0\)（高斯噪声或源域数据）变成 \(\mu_1\)（数据分布）。扩散模型用一条 SDE/ODE 实现这次搬运，最优传输问一个规范性问题——**这次搬运是不是（应不应该是、能不能是）代价最小的？** 两个领域在 Benamou–Brenier 动态形式处共享同一套语法：

\[ W_2^2(\mu_0,\mu_1) = \min_{(\rho_t, v_t)} \int_0^1 \mathbb{E}_{\rho_t}\|v_t\|^2 \, dt \quad \text{s.t.} \quad \partial_t \rho_t + \nabla\cdot(\rho_t v_t) = 0 \]

任何生成流的动能 ≥ \(W_2^2\)，缺口就是「弯曲程度」。由此分解出五个具体问题。

### P1 路径几何问题（为什么慢）

**表述**：PF-ODE 采样轨迹 \(x_t\) 的曲率决定离散化误差，进而决定最小可行 NFE。求：轨迹曲率的解析刻画、最优时间离散、以及把轨迹「弄直」的代价。
**已知**：轨迹近似躺在 2D 子空间、呈与内容无关的「回旋镖」形（AMED CVPR 2024 / GITS ICML 2024，[P]）；高斯数据下逆向 SDE/PF-ODE 有解析解与精确 W₂ 误差分解，Heun 格式最优（Pierret–Galerne ICML 2025，[P]）。
**未解**：曲率—最优调度的第一性定理（现有 AYS/GITS 是优化启发式）；免训练求解器的信息论下界。

### P2 耦合选择问题（谁配谁）

**表述**：训练/推理默认独立耦合 \(q(x_0)q(x_1)\)，回归目标互相冲突导致速度场弯曲交叉。求：在训练（minibatch OT 重配对）与推理（噪声指派/检索/优化）两侧选择耦合 \(\pi(x_0,x_1)\) 的原则、偏差与收益。
**已知**：batch 内 OT 重配对使路径直线化、梯度方差下降（Multisample FM ICML 2023，OT-CFM TMLR 2024，[P]）；但期望 batch 耦合 \(\pi_k\) 受维度诅咒支配、不随训练消失（Fatras AISTATS 2020 → Boïté et al. 2026 系统理论，[R]）；n≈10⁶ 的大 Sinkhorn + 低熵正则才见真收益（Zhang/Klein/Cuturi，arXiv 2506.05526，[R]）；条件生成中无条件 OT 耦合反而有害（C²OT ICCV 2025，[P]）。
**核心张力**：改耦合 vs 保边缘——任何实例级挑选都使有效初始分布偏离 \(\mathcal N(0,I)\)。

### P3 跨域传输问题（保什么、改什么）

**表述**：无配对翻译 X→Y 中「该保留什么」没有先验定义；OT 用传输代价给出规范答案。求：map（一步）、bridge（多步随机）、guidance（引导）三条路线的原则性框架与互补边界。
**已知**：SB = 熵正则 OT 的动态形式，是 unpaired translation 最正统框架（DSB 2021 → DSBM 2023 → UniDB++ TPAMI 2026，[P]）；神经 OT map 直译快而结构强但纹理弱（NOT ICLR 2023 Spotlight 系，[P]）；「翻译 = 两段 EOT 串联」有显式证明（DDIB ICLR 2023，[P]）。
**未解**：「更接近最优传输是否等于更好翻译」从未被正面回答；蒸馏后终端耦合是否漂移零保证。

### P4 最优性问题（扩散 ≟ OT）

**表述**：PF-ODE 定义的确定性 encoder map 是不是二次代价的 Brenier 最优映射？若不是，差多少？
**已知**：高斯情形严格成立（Khrulkov ICLR 2023，[P]）；一般情形**不成立**——流映射逐时刻是梯度场（无穷小最优）但复合后一般不是凸函数梯度，障碍是 Hessian 非交换项（Lavenant–Santambrogio 2022，[P]；前驱 Tanana 2021，[P]）；数值上「几乎最优」。分布层面 score matching 损失上界控制 W₂（Kwon et al. NeurIPS 2022，[P]）。
**未解（且是原作者明示的 open problem）**：次优度的定量上界。**8/25 新进展**：c-RF 理论（arXiv 2608.02487，[R]）在拉直侧给出高斯情形充要条件（协方差可交换）——见 §4.1。

### P5 计算与统计问题（多贵、多准）

**表述**：(a) OT 求解器的规模化（Sinkhorn O(n²) 内存/IO 瓶颈）；(b) 从 n 样本估计 OT map 的统计率；(c) 扩散/FM 的采样收敛与端到端统计保证。
**已知**：FlashSinkhorn 把 Sinkhorn 更新重写为 attention 同构的 online-LSE，O(nd) 显存、A100 上端到端最高 161×（ICML 2026 Oral，[A]）；OT map 估计 minimax 率 \(n^{-2\alpha/(2\alpha-2+d)}\)（Hütter–Rigollet AoS 2021，[P]）在高维像素空间必然失效 → 必须显式引入低维/结构假设；扩散迭代复杂度 O(d/T)（TV）且自适应内在维数 O(k/T)（Li–Yan JMLR 2025，[P]）；FM almost minimax，最优方差调度 σ_t≍√t（Fukumizu ICLR 2025，[P]）。
**未解**：OT-耦合训练（非独立耦合）的统计率完全空白；「n 样本 × T 步」联合下界（加速是否吃掉统计精度）无人证。

---

## 2. 别人的理论：定理骨架与四大张力

### 2.1 数学装备层（拿来即用的定理）

| 定理 | 内容 | 在扩散×OT 中的角色 |
|---|---|---|
| Kantorovich 对偶（1942） | OT 的线性规划对偶，位势函数 \(f,g\) | 对偶势梯度 = 天然 guidance 场 |
| Brenier 定理（1991，CPAM，[P]） | 二次代价最优映射存在唯一且 = 凸势梯度 \(T=\nabla\varphi\) | 一切「encoder ≈ OT map」讨论的根基；联结 Monge–Ampère |
| McCann 位移插值（1997） | \(W_2\) 测地线 = \(((1-t)\mathrm{Id}+tT)_\#\mu\) | 「理想直线轨迹」的规范定义 |
| JKO 格式（1998，SIAM J. Math. Anal.，[P]） | Fokker–Planck = KL 在 \(W_2\) 度量下的梯度流 | 扩散的「耗散」变分结构 |
| Benamou–Brenier（2000，Numer. Math.，[P]） | \(W_2^2\) = 连续性方程约束下最小动能 | 扩散的「测地」变分结构；轨迹直度的规范账本 |
| Otto calculus（2001） | \(\mathcal P_2\) 作为形式黎曼流形 | 测度空间优化的微分几何词汇 |
| 半离散 OT（KMT 2019，JEMS，[P]） | 连续源→离散目标，阻尼牛顿全局线性收敛 | 精确匹配「先验 → 有限数据集」的工程现实 |
| 熵正则/Sinkhorn（Cuturi 2013，[P]） | ε-正则化 OT，GPU 可并行 | 一切 minibatch 耦合与 SB 的计算底座 |

学习路线（T01）：Peyré《OT for Machine Learners》2025 讲义（[B]，arXiv 2505.06589）通读 → Santambrogio 2015 补严格证明 → 高斯族 Bures–Wasserstein 闭式解搭「随学随验」沙盒。

### 2.2 辩论线一：「扩散 ≟ OT」——猜想、反例、量化（T02）

- **正方**：Khrulkov et al.（ICLR 2023，[P]）证明多元高斯情形 DDPM encoder 恰为 Monge map，猜想一般成立（tensor-train Fokker–Planck 数值支持）。
- **反方定论**：Lavenant & Santambrogio（Appl. Math. Lett. 2022，[P]）三页反例证伪：PF-ODE 速度场**逐时刻**是梯度场（无穷小意义「局部 OT」），但复合后的流映射一般不再是凸函数梯度——障碍恰是 \(D^2 u\) 与 \(D^2(\log\det D^2 u - \frac12\|\nabla u\|^2)\) 的非交换性。
- **精确图景**：逐时刻最优 ≠ 全局最优；经验上近乎最优；**缺口未被量化（原作者明示 open problem，工具已齐：高斯精确解 ICML 2025 + FP 求解器）**。
- **正面补丁**：特定条件下有限时间区间上 PF 确为 Monge map（P. Zhang NeurIPS 2023，[P]）；分布层 \(W_2 \le \sqrt{\text{score matching loss}}\times C\)（Kwon NeurIPS 2022，[P]）；Föllmer/扩散型映射虽非最优却有 OT 映射尚无法证明的 Lipschitz 收缩性（Mikulincer–Shenfeld PTRF 2024，[P]）——「扩散传输」开始有独立于 OT 的正则性理论。
- **建设性新方向（2025–2026）**：把 drift 动力学显式约束回 OT——约束漂移模型收敛到 Monge map（Dumont et al. 2603.25182，[R]）、Monge–Ampère 流（Deb–Liang 2504.09279，[R]）。

### 2.3 辩论线二：「拉直 ≠ OT」——rectified flow 的理论边界（T09）

- RF 奠基（Liu et al. ICLR 2023，[P]）证明三件事：rectification 保边缘且**单调不增一切凸传输代价**；直线度 \(\min_{k\le K}S(Z^k)=O(1/K)\)；直耦合 ⇔ 插值路径不相交，是 c-最优的**必要非充分**条件（仅 1D 重合）。
- 反例定论（Hertrich–Chambolle–Delon NeurIPS 2025，[P]）：迭代 rectification 存在**非最优不动点**、损失趋零不蕴含最优——「reflow ≈ OT 求解器」叙事正式终结。
- 实证修正（rfpp NeurIPS 2024，[P]）：现实设置下**一轮 reflow 即近乎直**；Rectified Diffusion（ICLR 2025，[P]）更进一步：本质是「预训练模型配对 + 重训」而非直线度本身。
- **8 月新进展（gap check 增补）**：《Computational and Statistical Guarantees of the c-Rectified Flow》（arXiv 2608.02487，[R]）给出：高斯情形普通 reflow 收敛到 OT 耦合 **当且仅当源/目标协方差矩阵可交换**；c-rectified flow（速度场投影到梯度类）在紧性+一致可积假设下**恒收敛**到 OT 耦合，且有一步收缩与指数收敛率；配合新的 minimax score 估计率得到 d≥3 的率最优 OT 估计器。——这直接部分回答了 T09 的公开问题（见 §6.1 对切入点 #4 的更新）。

### 2.4 Schrödinger 桥理论：五代求解器与收口（T03）

SB 问题：路径测度上 \(\min_{P} KL(P\|Q)\) s.t. 两端边缘约束；静态投影 = 熵正则 OT；\(\varepsilon\to 0\) 收敛到确定性 OT。五代演进：

1. **深度 IPF**（DSB NeurIPS 2021 Spotlight；SB-FBSDE ICLR 2022，[P]）：SGM 恰为第一次 IPF 迭代；痛点是交替训练、误差累积「遗忘」。
2. **IMF / bridge matching**（IDBM JMLR 2023；DSBM NeurIPS 2023，[P]）：关键洞见——SB 是唯一既 Markov 又属参考桥 reciprocal 类的过程；交替两投影每步保两端边缘。
3. **轻量化/广义化**（LightSB ICLR 2024；LightSB-M ICML 2024——任意耦合单次 matching 即证恢复 SB；GSBM ICLR 2024 纳入任务 state cost，[P]）。
4. **在线/离散/少步/半监督**（α-DSBM NeurIPS 2024 单网络在线；ASBM NeurIPS 2024 与 CSBM ICML 2025 离散 D-IMF；FSBM ICLR 2025 Oral <8% 配对样本作 feedback，[P]）。
5. **理论收口（2025–2026）**：IMF 首个非渐近**指数收敛率**（NeurIPS 2025，[P]）；「Sinkhorn bridge」统计分析统一 matching 系估计量（2510.22560，[R]）；IPMF 证明双向交替启发式收敛（ICLR 2026，[A]）；Tang 220 页专著（2603.18992，[B]）标志教材化。

### 2.5 梯度流与 Wasserstein proximal：扩散的第二套构造原理（T05）

- JKO 神经化：ICNN 凸势（Mokrov NeurIPS 2021，[P]）→ 逐块 CNF（JKO-iFlow NeurIPS 2023 Spotlight，[P]）→ **S-JKO**（ICML 2024，[P]）借 JKO↔UOT 等价把复杂度 O(K²)→O(K)，CIFAR-10 FID 2.62，WGF 生成模型首次逼近 SOTA。
- **SGM = Wasserstein proximal 算子**（SIMODS 2026，[P]）：score-based 模型隐式实现交叉熵的（正则化）WPO，MFG 最优性条件 = 前向受控 FP + 后向 HJB；核公式直接解释并缓解记忆化。
- 反问题线：JKOnet（AISTATS 2022）→ JKOnet*（NeurIPS 2024 Oral，[P]）从快照学驱动能量，一阶最优性条件把双层优化降为单层二次损失。
- 前沿：测度空间二阶理论（PWGF 逃鞍点，NeurIPS 2025，[P]）；W-Flow 把 Sinkhorn-WGF 整条演化蒸馏成一步生成器，ImageNet-256 **1-NFE FID 1.29**（2605.11755，[R]）。

### 2.6 收敛与统计理论：审稿人的度量衡（T06）

| 问题 | 当前最好结果 | 出处 |
|---|---|---|
| 采样迭代复杂度（给定 score） | TV：O(d/T)，仅需一阶矩 + L² score；KL：Õ(d/ε)（复合分析，[R]） | Li–Yan ICLR/JMLR 2025 [P]；Jain–Zhang 2508.16306 [R] |
| 内在维数自适应 | 流形数据 KL 步数对 k 线性且 sharp；DDPM 无需知道 k 自动达近 k-线性 | Potaptchik COLT 2025 [P]；Huang–Wei–Chen MOR 2026 [P] |
| 端到端统计（score 从样本学） | 扩散是 Besov 类近 minimax 分布估计器；score 估计率 Θ̃(n^{-2/(d+4)})（维数灾难坐实） | Oko ICML 2023 [P]；Wibisono COLT 2024 [P] |
| FM 理论 | almost minimax（1≤p≤2 Wasserstein），σ_t≍√t 是最优调度；W₂ 误差界 | Fukumizu ICLR 2025 [P]；Benton TMLR 2024 [P] |
| PF-ODE（确定性）理论 | TV 率 O(k/T) 自适应内在维数；端到端近 minimax（需同时控 Jacobian 误差） | Tang–Yan Info. Inference 2025 [P]；Cai–Li 2503.09583 [R] |
| OT map 估计 | minimax 率 n^{-2α/(2α-2+d)}；可计算 plug-in 同最优 + CLT；一般函数空间（Poincaré+度量熵）覆盖神经 map | Hütter–Rigollet AoS 2021；Manole AoS 2024；Divol AoS 2025（均 [P]） |
| 熵正则侧 | entropic map 估计器兼顾率与可扩展性；lower complexity adaptation：率只取决于「简单」一方 | Pooladian–Niles-Weed [R]；Groppe–Hundrieser JMLR 2024 [P] |

**关键空白（= 我们的机会）**：现有 FM 统计理论只覆盖**独立耦合**——OT-CFM 的端到端收敛率是 diffusion×OT 的天然交叉定理（§6.1 #5）。

### 2.7 随机最优控制统一视角（T02）

Hopf–Cole/HJB 把扩散训练目标（ELBO）解释为 verification theorem（Berner TMLR 2024，[P]）；SOC 求解化为回归（SOCM NeurIPS 2024，[P]）；reward 微调 = memoryless SOC，必须用 σ(t)=√(2η_t) 消初值偏差（Adjoint Matching ICLR 2025 Spotlight，[P]）；路径积分/WKB 是同一结构的物理表述（ICML 2024，[P]）。**对我们的意义**：把终端 reward 换成传输代价泛函，即得「把预训练扩散控制到目标耦合」的原理性机制——方向二的理论工具箱。

### 2.8 四大理论张力（本报告的分析主轴）

1. **耗散 vs 测地**：同一 \(\mathcal P_2\) 流形上，扩散走 KL 梯度流（JKO），OT 走测地线（BB）——扩散×OT 的一切纠葛源于这两种变分结构的不重合。
2. **逐时刻最优 vs 全局最优**：PF-ODE 每一瞬都是梯度场，复合起来却不是——「局部 OT ≠ 全局 OT」（Lavenant 反例的本质）。
3. **直 vs 最优**：直线化降代价但不动点未必最优（Hertrich 反例）；c-RF 投影修复最优性（2608.02487）——「直线度」与「最优性」是两个自由度。
4. **加速 vs 统计精度**：迭代复杂度与统计率各自 sharp，联合前沿无人刻画——「快了会不会不准」还没有定理级答案。

---

## 3. 经典工作：奠基年表（The Canon）

### 3.1 数学经典（1781–2013）

| 年份 | 工作 | 为什么是经典 |
|---|---|---|
| 1781 | Monge《论土方的搬运》 | 提出原始映射形式，非凸且可能无解 |
| 1942 | Kantorovich 松弛 | 耦合上的 LP + 对偶，OT 成为可分析对象 |
| 1991 | **Brenier 极分解定理**（CPAM，[P]） | 二次代价最优映射 = 凸势梯度；与 Monge–Ampère、凸分析焊接 |
| 1997 | McCann 位移插值 | 分布形变的标准语言（\(W_2\) 测地线） |
| 1998 | **JKO 格式**（SIAM JMA，[P]） | Fokker–Planck = KL 的 Wasserstein 梯度流 |
| 2000 | **Benamou–Brenier 动态形式**（Numer. Math.，[P]） | \(W_2^2\) = 最小动能；今天 PF-ODE/FM 分析的通用语法 |
| 2001 | Otto calculus | \(\mathcal P_2\) 的黎曼几何化 |
| 2013 | **Cuturi Sinkhorn**（NeurIPS，[P]） | 熵正则把 OT 带进 GPU 时代 |
| 2014 | Léonard SB 综述（DCDS，[B]） | Schrödinger 1932 问题的现代梳理：路径熵最小化 ⇔ EOT |

### 3.2 生成建模接口经典（2021–2023）

| 年份·会议 | 工作 | 奠基点 |
|---|---|---|
| 2021·ICLR Oral | Score-SDE（Song et al.，[P]） | VP/VE-SDE + PF-ODE 统一框架，定义了「encoder map」这个讨论对象 |
| 2021·ICLR | DDIM（[P]） | 采样确定化 = PF-ODE 一阶指数离散，10–50 步实用化 |
| 2021·NeurIPS Spotlight | **DSB**（De Bortoli et al.，[P]） | SB 进生成建模：SGM = 第一次 IPF 迭代 |
| 2021·NeurIPS | W2 benchmark（Korotin et al.，[P]） | 神经 OT 求解器的第一个 ground-truth 评测：「下游好 ≠ map 准」 |
| 2022·NeurIPS | EDM（Karras et al.，[P]） | 统一训练/采样设计空间（ρ=7 调度、Heun、churn），few-NFE 公共基准 |
| 2022·NeurIPS | DPM-Solver（[P]） | 半线性结构定制指数积分器，NFE 数百 → 10–20 |
| 2023·ICLR | **Flow Matching**（Lipman et al.，[P]） | conditional path + CFM 目标：simulation-free 训练 CNF |
| 2023·ICLR | **Rectified Flow**（Liu et al.，[P]） | 线性插值 + reflow 迭代拉直；「直线换算力」纲领 |
| 2023·ICLR（JMLR 2025 完整版） | Stochastic Interpolants（Albergo & Vanden-Eijnden，[P]） | 任意两分布插值统一 flows/diffusions/SB |
| 2023·ICLR | **Khrulkov 猜想**（[P]）＋ Lavenant–Santambrogio 反例（2022，[P]） | 「扩散≟OT」辩论定型：高斯成立、一般证伪、量化开放 |
| 2023·ICLR | **DDIB** / NOT（[P]） | 翻译 = 两段 EOT 串联；weak OT 统一 saddle-point 求解器 |
| 2023·ICML | **I2SB**（[P]） | 边界对给定时 SB tractable 化，桥模型工业可用性首证 |
| 2023·ICML | Multisample FM（[P]）＋ OT-CFM（TMLR 2024，[P]） | batch 级 OT 耦合进入 FM 训练：直线化 + 方差下降 |
| 2023·NeurIPS | **DSBM/IMF**（[P]） | Markov×reciprocal 双投影：SB 求解不再累积误差 |
| 2023·NeurIPS | UOTM（[P]） | UOT 半对偶生成模型：outlier 稳健 + 训练稳定 |

### 3.3 规模化与工业化（2024）

SD3 大规模消融确立 RF 公式 + logit-normal 时间采样为工业标准（ICML 2024 Oral，[P]）；SiT 在 DiT 骨干上完成 interpolant 四轴消融（ECCV 2024，[P]）；DDBM 统一桥设计空间（ICLR 2024，[P]）；Immiscible Diffusion 证明数据管线级噪声指派一行代码加速训练最高 3×（NeurIPS 2024，[P]）；AYS/GITS 把采样调度原理化（ICML 2024，[P]）；LightSB-M/α-DSBM/ASBM 把 SB 轻量化在线化（ICML/NeurIPS 2024，[P]）；UOT-FM 证明 unbalanced map = 重缩放边际的 balanced map、即插即用（ICLR 2024，[P]）；moscot 把 OT 推到 170 万细胞（T24）；FM 进语音/分子/流形（T21/T23/T28）。

---

## 4. 最新工作：2025–2026 前沿地图

### 4.1 理论收口线

| 工作 | 出处·分级 | 一句话 |
|---|---|---|
| IMF 指数收敛率 | NeurIPS 2025 [P] | SB 求解迭代的首个非渐近 KL 指数率（覆盖强/弱对数凹） |
| 「reflow≠OT」反例 | Hertrich et al. NeurIPS 2025 [P] | 非最优不动点存在、损失趋零≠最优 |
| **c-RF 计算统计保证** | arXiv 2608.02487 [R]（8 月新） | 高斯情形普通 reflow → OT **iff 协方差可交换**；c-RF 恒收敛 + 指数率 + d≥3 minimax 最优 OT 估计 |
| FM almost minimax | Fukumizu et al. ICLR 2025 [P] | 统计上 FM 与扩散等价；σ_t≍√t 最优 |
| O(d/T) / O(k/T) | Li–Yan JMLR 2025 [P]；MOR 2026 [P] | 最弱假设下线性维数依赖，自适应内在维数且最优 |
| SGM = Wasserstein proximal | SIMODS 2026 [P] | MFG（FP+HJB）刻画 score 模型；解释/缓解记忆化 |
| Sinkhorn bridge 统计 | 2510.22560 [R] | [SF]²M/DSBM/BM²/LightSB(-M) 估计量一致，泛化分析全家族生效 |
| Entropy-Controlled FM | 2602.22265 [R] | 熵率预算约束的 FM 变分原理 = 带显式熵乘子的 SB；Γ-收敛到 OT；模式覆盖证书 |
| DFM 维数改进 KL 界 | 2606.16610 [R] | Brownian DFM 离散化误差的 SOTA 维数依赖 |
| 条件 W 距离几何 | JMLR 2025 [P]（T18） | joint W₂ 不控制 posterior W₂——条件生成度量的正确性基础 |

### 4.2 一步生成新范式线

| 工作 | 出处·分级 | 一句话 |
|---|---|---|
| MeanFlow | NeurIPS 2025 Oral [P] | 平均速度场 + MeanFlow identity 从头训练一步生成，ImageNet-256 1-NFE FID 3.43 |
| W-Flow | 2605.11755 [R] | Sinkhorn-WGF 整条演化蒸馏进一步生成器，1-NFE FID 1.29 |
| **Beckmann Transport Models** | 2608.01692 [R]（8 月新，Albergo/Vanden-Eijnden 组） | **自治（时间无关）速度场**精确映射两分布（目标低维支撑时）；一步映射满足守恒方程可直接学；给 Beckmann 流量约束以动力学意义，统一回收 Poisson Flow 与 Equilibrium Matching |
| PMOT | 2608.05666 [R]（8 月新） | 标量势参数化广义 Benamou–Brenier，零损失解恢复一般 p-代价 OT 映射与动力学 |
| Flow Map Matching / Transition Matching | TMLR 2025 [P] / NeurIPS 2025 [P] | 学习对象从瞬时速度改为两时间流映射/离散转移核 |
| LBM | ICCV 2025 Highlight [P] | latent Brownian bridge matching 蒸馏到 1 NFE，覆盖重光照/去物体等产品任务 |
| CAF / HRF | NeurIPS 2024 / ICLR 2025 [P] | 常加速度/层级 ODE：放弃常速假设，允许路径相交 |

### 4.3 耦合工程线（方向一的训练侧）

| 工作 | 出处·分级 | 一句话 |
|---|---|---|
| 大规模 Sinkhorn 耦合 | 2506.05526 [R]（Apple） | n≈10⁶ 分片 Sinkhorn：「OT 耦合无用」是小 batch 伪象，大 n + 低 ε 显著收益 |
| Semidiscrete couplings | 2509.25519 [R] | 绕开 minibatch：全数据集预计算半离散对偶势，训练时查表配对 |
| LOOM-CFM | ICLR 2025 [P] | 跨 batch 存储/交换局部最优配对，近零开销逼近全局 plan |
| C²OT | ICCV 2025 [P] | 条件生成中无条件 OT 耦合有害（条件偏斜先验）；成本矩阵加条件加权修复 |
| 期望 batch plan 理论 | Boïté et al. 2605.12174 [R] | π_k 大 batch 一致性、半离散收敛速率、FM 流良定性的首个系统理论 |
| Designing OT Flows | 2606.04092 [R] | 换视角：设计低频投影先验使**恒等耦合本身最优** |
| 离散 FM × minibatch OT | ICML 2026 [A] | 耦合选择进入离散流匹配（perplexity 视角） |

### 4.4 推理期对齐线（方向一的免训练侧，T12 七环节）

可插入测度耦合的七个环节：① 数据管线噪声指派（Immiscible 系，[P]）→ ② 初值检索/搜索（NoiseQuery ICCV 2025 Highlight 0.2ms 查库；verifier×搜索设计空间 CVPR 2025，[P]）→ ③ 初值连续变换（Golden Noise ICCV 2025 [P]；NoiseRefine 把 CFG 折叠进初值 ICLR 2026 [A]）→ ④ OT 桥接 prior（半离散 OT 一步桥 + 短程扩散，2410.13431 [R]）→ ⑤ 轨迹中段对齐（search-over-paths；batch 中段重排**几乎空白**）→ ⑥ 跨帧噪声传输（∫-noise ICLR 2024 Oral；Go-with-the-Flow CVPR 2025 Oral，[P]）→ ⑦ inversion 侧耦合。
**支撑事实**：噪声↔样本耦合是数据内在属性、跨模型可复用（可复现性现象，ICML 2024，[P]）——离线构造（噪声库、golden noise 数据集）因此有普适价值。
**几何修正前沿**：欧氏梯度噪声优化会推离高斯 typical set（Oracle Noise 2604.23540，[R]）。

### 4.5 桥与翻译线（方向二）

| 工作 | 出处·分级 | 一句话 |
|---|---|---|
| UniDB++ | TPAMI 2026 [P] | SOC 统一桥框架的闭式逆向解 + SDE-Corrector，免训练加速 5–20×，DBIM 为特例 |
| DBIM / CDBM | ICLR 2025 / NeurIPS 2024 [P] | 桥的 DDIM 对应物（25×）/ 桥的一致性蒸馏（4–50×） |
| FSBM | ICLR 2025 Oral [P] | <8% 配对样本作 state feedback 的半监督 SB |
| CSBM / 3MSBM / Reflected SBM | ICML 2025 / NeurIPS 2025 [P] / 2607.03626 [R] | SB 推广到离散空间/多边缘动量/反射（有界域）动力学 |
| DIOTM / OTP / ENOT | ICLR 2025 / ICML 2025 / NeurIPS 2024 [P] | 静态 map 线的稳定化：位移插值正则、spurious 解充分条件、expectile 近似 c-transform |
| UNOT | ICML 2025 [P] | FNO 摊销预测熵正则位势，Sinkhorn 初始化 7.4× |
| LSB | 2411.14863 [R] | 预训练 SD 三预测子线性组合免训练近似 SB |
| Bridge vs FM 统一比较 | 2509.24531 [R]（ICLR 2026 在审） | SOC 视角证明 FM 是 DB 的退化特例（无 drift） |
| 医学垂直（T15） | Medical Physics 2025 / MICCAI 2025 [P] | DSBM MR→CT 过剂量学验证关；LMSB「成本函数即医学先验」 |

### 4.6 系统与评测线

- **FlashSinkhorn**（ICML 2026 Oral，[A]）：稳定化 log-domain Sinkhorn 更新 = biased dot-product 的行 LSE = **attention 归一化**，直接搬 FlashAttention tiling，前向 9–32×、端到端最高 161×，O(nd) 显存。8/25 复查：v0.3.3（2026-04）后无新 release，仍限平方欧氏 + 单卡——**分布式空位仍开放**。
- 低秩/层次线：FRLC（NeurIPS 2024）→ HiRef（ICML 2025 Oral，log-linear 全秩双射到百万点）→ HALO（ICLR 2026，O(n) 内存精确管线）（均 [P/A]）。
- 端侧：SnapGen 379M 参数手机 1.4s 出 1024²；SVDQuant 4-bit；范式转向「为端侧从头设计 FM 模型」（T30）。
- 评测：FID 对少步模型系统性失真（T30）；**SynthRAD2025 官方教训（8/25 核验）：图像质量不是剂量准确性的充分替代**（MS-SSIM×Dice ρ=0.78–0.79，与剂量指标仅中等相关）。

### 4.7 顶会趋势（数据见主报告 §8）

FM 提及级接收量连续两年约 3×（ICLR 7→46→144；ICML 13→56→167；NeurIPS 6→32→88）；OT 平稳上行（ICML 2026 翻倍 43→114）；「rectified flow」术语被 FM 吸收走平；理论主战场在 ML 三会，CVPR/MICCAI 是应用出口。窗口期判断：理论空位约 12–18 个月，应用垂直更长。

---

## 5. 十条核心洞察（Insights）

**I1 · OT 是设计语言，不是扩散模型的自动性质。** 两条辩论线（§2.2/§2.3）都以反例收尾：encoder map 不是 Brenier map，reflow 不动点不是 OT。正确姿势是把 OT 当**耦合选择、调度设计、蒸馏正则、跨域先验**四种可插拔机制，每处都度量「距 OT 的偏差」而非宣称「实现了 OT」。这直接决定论文的表述红线（§6.4）。

**I2 · 「OT 用量」是一个精确可研究的变量，不是站队问题。** 三组证据构成张力三角：大 batch Sinkhorn（n≈10⁶ + 低 ε）才见真收益（[R] 2506.05526）；3D 点云上完全 OT 耦合反而让 t≈0 速度场更难学（NSOT ICLR 2025，[P]）；条件生成中无条件 OT 耦合系统性有害（C²OT ICCV 2025，[P]）。⇒ 「耦合的 OT 程度 × 数据几何 × 条件结构 → 可学性」的定量刻画是全新研究轴。

**I3 · 免重训的价值集中在两个自由度：耦合与调度。** 推理管线有七个可插耦合的环节（§4.4），调度侧五条原理化路线（AYS/GITS/LD3/OSS/DM-NonUniform）已取代手工调度。二者的交点——**batch 级保边缘重排**（环节②与⑤之间）与 **OT-aware 调度理论**（动能泛函替代 KLUB）——是当前最干净的两个空位（Top-10 #1/#2）。

**I4 · 直线化与最优性已正式解耦，而 8 月的 c-RF 理论把它们重新接上。** 「直 ⇒ 少步」是数值分析事实，「直 ⇒ 最优」是伪命题（Hertrich 反例）；rfpp 证明一轮 reflow 就够直、Rectified Diffusion 证明本质是配对改进。c-RF（2608.02487，[R]）给出修复路径：把速度场投影到梯度类则恒收敛到 OT——「拉直」与「最优」的关系从叙事之争变成投影算子的选择问题。

**I5 · 理论骨架已收口，论文级空位在「接缝处」。** SB 有指数收敛率、FM 有 minimax、扩散有 O(d/T)——但这些定理相互不搭界：OT-耦合下的 FM 统计率（独立耦合假设失效）、学习误差下的 IMF 收敛、迭代×样本联合下界、跨域引导的端到端 oracle 不等式，全部空白。**新框架论文的时代结束了，接缝定理的时代开始了。**

**I6 · 蒸馏的「保耦合」是零保证地带。** 桥/SB 蒸馏（CDBM/IBCD/LBM）追求少步，但蒸馏后终端耦合是否漂移无任何定理；医学翻译恰恰最在乎这个（解剖结构漂移 = 临床事故）。「保 OT 耦合的蒸馏 + 耦合漂移度量」同时是理论空位和医学刚需（Top-10 #3）。

**I7 · 基础设施出现拐点，且空位已经 8/25 官方核验。** FlashSinkhorn 看破「Sinkhorn = attention」后，OT 求解器进入 FlashAttention 式工程时代；但非欧 cost 的 online-LSE 化与**多 GPU 分布式**（ring-attention 式分片归约）仍完全空白（v0.3.3 后无 release）。谁先做出「分布式 IO-aware Sinkhorn」，谁就拥有数据集级跨域耦合的入场券。

**I8 · 评测存在系统性错位，OT 自己可以当解药。** FID 对少步模型失真（T30）；SynthRAD2025 官方报告证实图像相似度与剂量准确性只有中等相关——「像素好看 ≠ 下游正确」在医学里是定量事实。机会：DINOv2 特征上的 Sinkhorn divergence 替代 FID（Top-10 #10）+ 任务侧「下游指标优先」的评测设计。

**I9 · 竞争格局对高校有利的区域是明确的。** 大组（Meta/NVIDIA/Google/Apple）占据 FM 基础设施与大规模耦合工程；Vector/Mila 占 SB/CFM 谱系；Skoltech 系占神经 OT。但 Top-10 里的理论补全型（#2/#4/#5/#6）与接口缝合型（#1/#8/#10）都不需要预训练算力——需要的是证明与精巧实验，恰是高校比较优势。时间窗：理论约 12–18 个月。

**I10 · 医学垂直的入场条件已官方验证，且门槛写得很清楚。** SynthRAD2025 挑战报告（arXiv 2605.13555，8/25 核验）：25 队零 SB/OT 方法；FM/扩散系每任务仅 3 队且剂量指标显著逊于 CNN/GAN（质子 γ 是最大短板：顶四队 86.4–88.6% vs 光子 99%+）；post-challenge 榜开放至 2030-03。⇒ 空位真实存在，但**入场必须以剂量学为设计目标**（解剖商空间成本 + 3D 耦合 + conformal 幻觉筛查），纯像素指标的扩散方法已被证明不够。

---

## 6. 我们可以做什么：机会地图与行动方案

### 6.1 Top-10 切入点（2026-08-25 更新版）

| # | 切入点 | 类型 | 算力 | 8/25 状态更新 |
|---|---|---|---|---|
| 1 | 免训练 batch 级保边缘噪声重排：batch 内 (条件,噪声) 一次性 Hungarian/Sinkhorn 指派，每噪声恰用一次、边缘严格保持；理论卖点 = 保边缘性 + 方差降低；评测 T2I-CompBench/GenEval | 方法+理论 | 低 | 空位仍在（T12 环节②/⑤ 之间） |
| 2 | OT-aware 采样调度：把 AYS 的 KLUB 换成 Benamou–Brenier 动能泛函，证「等旋转误差 = W₂ 最优调度」定理；TORS 的恒定总旋转经验律是待证靶子 | 理论+免训练 | 低 | 空位仍在 |
| 3 | 保 OT 耦合的单步桥蒸馏 + 耦合漂移度量：证明 consistency bridge 蒸馏保终端耦合的条件；Sinkhorn 散度约束的「保耦合蒸馏」损失 | 方法 | 中 | 空位仍在；医学直接受益 |
| 4 | reflow 正定理 | 理论 | 低 | **⚠ 被 2608.02487 部分解决**：高斯情形充要条件（协方差可交换）+ c-RF 恒收敛已被证。**转向建议**：(a) 高斯混合/流形数据的非渐近刻画（该文只做高斯）；(b) c-RF 投影的实用化（大规模近似投影算子）；(c) 把其 minimax 率接到 SD3/FLUX 级实验诊断 |
| 5 | OT-CFM 端到端统计收敛率：熵 map 估计率作中间量，回答「加速 vs 统计精度」 | 理论 | 低 | 空位仍在（c-RF 文只覆盖 rectification 路线，minibatch-OT 耦合训练的率仍空白） |
| 6 | PF-ODE 次优度量化：Lavenant 明示 open problem；工具已齐（高斯精确解 + tensor-train FP） | 理论 | 低 | 空位仍在 |
| 7 | 解剖商空间成本 + 3D 耦合 SB 刷 SynthRAD2025 + conformal 幻觉筛查 | 应用 | 中 | **✅ 8/25 官方证实**：25 队零 SB/OT；榜开放至 2030-03；**须以剂量学为设计目标**（质子 γ 短板） |
| 8 | FGW 语义对应闭环进扩散采样：编辑一致性引导（判别与生成的缝合） | 方法 | 中 | 空位仍在（T16） |
| 9 | 视频侧：时空分解 reflow（公开工作缺席）或帧间噪声 Sinkhorn 耦合替代光流 warp | 方法 | 中高 | 空位仍在（T19） |
| 10 | OT 系评测指标：DINOv2 特征 Sinkhorn divergence 替代 FID + 人评相关性（D&B 论文） | 评测 | 低 | 空位仍在；SynthRAD 教训强化其动机 |

**组合建议**（不变）：#1+#2 构成「推理期耦合工程」连击（共享代码基建）；#3+#7 构成「医学桥」主线（方法+落地）；#4（转向后）/#5/#6 是纯理论线。

### 6.2 全景空位地图（按类型，30 课题 §5 精选）

**理论补全型**（低算力、高时窗压力）：
- 量化 PF-ODE 次优度地形图：∫|S∞−∇u*|²dμ₀ ≤ C·(非交换项范数) 型上界；近高斯 log-concave 数据下随 Fisher 信息差收敛的猜想（T02）
- 流形数据下 encoder map 定性：法向坍缩 ⇒ 切向趋于流形间 OT map（先「线性子空间+高斯噪声」精确模型，T02）
- 学习误差下的 IMF/α-IMF 收敛：把 Sinkhorn-bridge 统计（2510.22560）与 IMF 收缩估计拼接（T03）
- 偏差→曲率→NFE 的端到端传导界：「速度场曲率 ≤ f(batch n, ε, d)」+ 大规模验证曲线（T08）
- 实例级噪声选择（top-1-of-k）的 order-statistics 耦合漂移上界（T12）
- 免训练 vs solver 蒸馏的信息论下界：给定 NFE 预算的最小 W₂/FID 差距（T11）
- reward 微调 = 熵正则 OT：Adjoint Matching 终端 reward 换传输代价泛函，σ→0 极限恢复 tilted Monge map（T02）
- 平均速度/flow map 的误差传播界（MeanFlow 系尚无 Benton/Fukumizu 式保证，T07）
- CFG 分布偏移的 W₂ 刻画与最优 guidance 强度调度（T18）
- weak/UOT 半对偶的可计算 duality-gap 误差证书（T13）

**方法/接口缝合型**（中算力）：
- 一步 OT map 粗对齐 + 冻结扩散 2–4 步细化的混合管线（T13×T14 接口，量化「传输成本-保真-FID」三方权衡）
- 半离散 JKO/UOT 势作免训练 guidance：目标域小样本在线解轻量半对偶，势梯度注入 PF-ODE（T05）
- Monge–Ampère 桥规模化：熵正则半离散对偶/ICNN 摊销，推到 SDXL/FLUX latent（T12）
- LOOM 与 semidiscrete 之间的在线全局耦合（cluster 级 anchor OT，内存 O(K)，T08）
- 端点可解码性 × Sinkhorn barycentric 投影解码（TJS 接口，T11×T12）
- 语义成本系统消融：CLIP/DINO/GW 结构成本对直线度/FID/条件一致性的影响 + 「保边缘+最小条件 skew」充要条件（T08）
- 离散/多模态数据的耦合成本设计（编辑距离/GW/token 级部分 OT，几乎空白，T08/T22）
- 免 reflow 的单阶段拉直正则：Burgers 残差作 LoRA 微调正则（T09）
- 多域翻译的可学习 barycenter 中继（DDIB 的高斯中继未必最优，T14×T27）

**应用垂直型**（中-中高算力，窗口更长）：
- 医学桥主线：3D 体积一致 SB（slice-耦合/latent 3D 桥）刷 SynthRAD2025 全指标（T15，8/25 已核验入场条件）
- 统一 med-bridge benchmark：SynthRAD + Mayo LDCT + BCI，四层指标（像素/几何/剂量学/下游），本身可发 D&B（T15）
- 视频时空分解 reflow / 帧间 Sinkhorn 耦合（T19）
- 黎曼 rectified flow（完全空白）与 BW 空间脑影像增广（T28）
- WFR 生灭率作推理期模式再平衡（T25）；OT 势即插即用引导（OTP-FM 势转 guidance 的误差传播界，T27）
- 跨维 Gromov-Schrödinger 桥（T26）；Sway Sampling 理论化 + 非配对 SB 声转换（T23）

**系统/评测型**：
- 分布式 IO-aware Sinkhorn：ring-attention 式分片 online-LSE 归约 + 通信量下界证明，n≈10⁷–10⁸ 点云弱扩展性（T29，8/25 核验空位仍在）
- 非欧 cost / UOT 更新的 online-LSE 流式化（FlashSinkhorn 只覆盖平方欧氏，T29）
- OT 配对开销统一 benchmark：固定 dtype/tolerance/计时边界，扫 batch×dim×solver 网格出三维 Pareto 查表（T29，低垂果实）
- 新一代 neural OT map 精度 benchmark（2021 W2 benchmark 已测不动新方法，T13）
- 直线度作端侧可部署性代理指标；直线度×量化复合误差界（T30）

### 6.3 12 周行动计划（承接主报告 §10，起点 2026-08-25）

- **W1–2 装备**：Peyré《OT for Machine Learners》通读 + Benamou–Brenier 精读；Bures–Wasserstein 闭式沙盒；跑通 torchcfm + POT/OTT-JAX。**新增**：精读 c-RF（2608.02487）与 BTM（2608.01692）——它们直接改变 #4 的定位与一步生成的理论叙事。
- **W3–4 复现三件套**：Immiscible（一行代码）、OT-CFM（torchcfm）、DPM-Solver++/AYS 对比；建 NFE-质量-多样性三轴评测脚手架。
- **W5–8 立项冲刺**：主攻 #1（batch 保边缘重排：top-1 检索 vs Sinkhorn 指派对照 + 保边缘引理）；并行 #2 理论线（动能泛函调度）。
- **W9–12 扩展与投递**：#1 实验强 → ICLR 2027 主会；理论线先熟 → 补 GMM 闭式验证投 ICML；医学线按 MICCAI 2027 布局数据合规。
- **持续触发点**：ECCV 2026 论文集（9 月会议期）、NeurIPS 2026 放榜（9 月底）、PMLR ICML 2026 卷（FlashSinkhorn [A]→[P]）、FlashSinkhorn ≥v0.4 或多 GPU 支持。

### 6.4 写作红线（从 30 份笔记的反例与失败案例总结）

1. 不说「我们的方法实现了最优传输」——说「以 OT 为设计目标并度量偏差」（反例线 I1）。
2. 少步比较必须固定 NFE 口径并报告多样性指标（FID 对少步失真）。
3. 会议归属与「已接收」表述必须官方可核验；预印本结论一律 [R] 限定。
4. 任何实例级耦合修改必须报告边缘漂移/多样性（改耦合 vs 保边缘张力）。
5. 条件生成中的 OT 声明必须过 C²OT 检查（条件偏斜先验）；语义对应线须消融证明「耦合结构本身带来增益」而非 attention 重命名。

---

## 7. 附录

### 7.1 证据底座

- 30 份子课题笔记（`kb/t01–t30`，七节模板，491KB）；477 条引用（[P]372/[A]26/[R]64/[B]15，去重约 445–455 篇）；234 篇本地 PDF（2.8GB，`papers/`）；主引用库 `refs/MASTER_BIBLIOGRAPHY.md`。
- 审计链：`_audit/ARIS_AUDIT_20260814.md`（三层审计，no-new-blocker）→ `_audit/INCREMENTAL_REVIEW_20260825.md`（触发点复审：SynthRAD 官方证实、FlashSinkhorn 空位仍在、ECCV/NeurIPS 未到期）。
- 本文档新增引用（8/14 检索截止后，均 [R] 预印本）：c-RF 保证（arXiv 2608.02487）、Beckmann Transport Models（arXiv 2608.01692）、PMOT（arXiv 2608.05666）、Entropy-Controlled FM（arXiv 2602.22265）、DFM KL 界（arXiv 2606.16610）——以上未入主引用库统计口径，待下轮知识库增量更新时合并。

### 7.2 文档谱系

`INDEX.md`（导航）→ `REPORT_DIFFUSION_OT_20260814.md`（调研收口报告）→ **本文档**（深度综合层）→ `slides/DIFFUSION_OT_SLIDES_20260814.html`（20 页汇报 PPT）→ `SYNTHESIS_DIFFUSION_OT_20260825.html`（本文档的 HTML 版）。

*报告完。2026-08-25。*
