## 3. 别人的理论：定理骨架与四大张力

### 3.1 数学装备层（拿来即用的定理）

| 定理 | 内容 | 在扩散×OT 中的角色 |
|---|---|---|
| Kantorovich 对偶（1942） | OT 的线性规划对偶，位势 $f,g$ | 对偶势梯度 = 天然 guidance 场 |
| Brenier 定理（1991，[P]） | 二次代价最优映射存在唯一且 $T=\nabla\varphi$，$\varphi$ 凸 | 一切「encoder ≈ OT map」讨论的根基 |
| McCann 位移插值（1997） | $W_2$ 测地线 $=((1-t)\mathrm{Id}+tT)_\#\mu$ | 「理想直线轨迹」的规范定义 |
| JKO 格式（1998，[P]） | Fokker–Planck = KL 的 $W_2$ 梯度流 | 扩散的「耗散」变分结构 |
| Benamou–Brenier（2000，[P]） | $W_2^2$ = 连续性方程约束下最小动能 | 扩散的「测地」变分结构；轨迹直度的账本 |
| 半离散 OT（KMT 2019，[P]） | 连续源 → 离散目标，Laguerre 胞腔，阻尼牛顿全局线性收敛 | 匹配「先验 → 有限数据集」的工程现实；MPNA 的人口极限（§8.2） |
| 熵正则 / Sinkhorn（2013，[P]） | ε-正则化 OT，GPU 可并行 | minibatch 耦合与 SB 的计算底座 |

学习路线（T01 深读的结论）：Peyré《OT for Machine Learners》（`reports/2505.06589.md`）通读 → Santambrogio 2015 补严格证明 → Mérigot–Thibert 半离散讲义（`reports/2003.00855.md`）→ 高斯族 Bures–Wasserstein 闭式解搭沙盒。

### 3.2 辩论线一：「扩散 ≟ OT」（T02）

正方：Khrulkov 等证明多元高斯情形 DDPM encoder 恰为 Monge map 并猜想一般成立（`reports/2202.07477.md`）。反方定论：Lavenant–Santambrogio 三页反例——PF-ODE 速度场逐时刻是梯度场，复合后的流映射一般不是凸函数梯度。正面补丁：特定条件下有限时间区间上 PF 确为 Monge map（`reports/2311.03886.md`）；分布层 $W_2\le C\sqrt{\text{score matching loss}}$（`reports/2212.06359.md`）；Föllmer/扩散型映射有 OT 映射尚无法证明的 Lipschitz 收缩性。建设性方向：把 drift 显式约束回 OT（约束漂移模型、Monge–Ampère 流）。**精确图景**：逐时刻最优 ≠ 全局最优；经验上近乎最优；缺口未被量化。

### 3.3 辩论线二：「拉直 ≠ OT」及其八月修复（T09）

RF 奠基（`reports/2209.03003.md`）证明三件事：rectification 保边缘且单调不增一切凸传输代价（Thm 3.3/3.5）；直线度 $\min_{k\le K}S(Z^k)=O(1/K)$（Thm 3.7）；直耦合 ⇔ 插值路径不相交，是 c-最优的必要非充分条件，仅 1D 重合（Thm 3.8–3.10）。深读还确认：$O(1/K)$ 是对 $\min_k$ 的界且假设每步精确求解，网络近似误差不在定理里；一步 FID 4.85 的「SOTA」比较范围是 U-Net 一步模型（Table 1(b) 自列 StyleGAN-XL 1.85）。

反例定论：Hertrich–Chambolle–Delon（NeurIPS 2025）证明迭代 rectification 存在非最优不动点、损失趋零不蕴含最优。实证修正：rfpp 一轮 reflow 即近乎直；Rectified Diffusion 指出本质是「预训练配对 + 重训」。

**八月的两个修复定理**：(i) c-RF（2608.02487）：高斯情形普通 reflow 收敛到 OT 耦合当且仅当源/目标协方差可交换；速度场投影到梯度类后恒收敛到 OT 耦合，有一步收缩、指数率与 d≥3 的 minimax 最优 OT 估计。(ii) reflow×minibatch-OT 极限点（2608.07042）：与固定 batch 的 minibatch OT 交替迭代的极限是 N-循环单调耦合，在梯度场条件下收敛到 OT 映射。两者一致指向：**「直线度」与「最优性」是两个自由度，加一个投影/单调性约束就能把它们重新接上**。

### 3.4 Schrödinger 桥：五代求解器与 Q3 的精细化（T03）

SB：路径测度上 $\min_P KL(P\|Q)$ s.t. 两端边缘约束；静态投影 = 熵正则 OT；$\varepsilon\to0$ 收敛到确定性 OT。五代演进：深度 IPF（DSB，SGM 恰为第一次 IPF 迭代）→ IMF/bridge matching（DSBM，`reports/2303.16852.md`：SB 是唯一既 Markov 又属参考桥 reciprocal 类的过程）→ 轻量化（LightSB-M 任意耦合单次 matching 即证恢复 SB）→ 在线/离散/少步/半监督（α-DSBM、CSBM、FSBM）→ 理论收口（IMF 首个非渐近指数率；Sinkhorn bridge 统计；IPMF；Tang 220 页专著）。Q3 新阶段（§5）：参考过程该怎么选（PRISM）、端点约束怎么松（SDDBMs）、混合分布桥的连续性界（2608.13383）、去噪扩散 = 高温 SB（2608.25094）。

### 3.5 梯度流与 Wasserstein proximal（T05）

JKO 神经化：ICNN 凸势 → 逐块 CNF → S-JKO 借 JKO↔UOT 等价把复杂度 O(K²)→O(K)。SGM = Wasserstein proximal 算子（`reports/2503.01998.md` 一线）：score 模型隐式实现交叉熵的正则化 WPO，MFG 最优性条件 = 前向受控 FP + 后向 HJB，核公式解释记忆化。反问题线 JKOnet*；W-Flow 把 Sinkhorn-WGF 整条演化蒸馏成一步生成器（1-NFE FID 1.29，[R]）。Q3：WGF 成为微调工具——奖励引导一步模型（2608.29647）、CVaR 极端事件（2608.11544）。

### 3.6 收敛与统计理论：审稿人的度量衡（T06）

| 问题 | 当前最好结果 | 出处 |
|---|---|---|
| 采样迭代复杂度（给定 score） | TV：O(d/T)，仅需一阶矩 + L² score；KL：Õ(d/ε) | Li–Yan JMLR 2025 [P]；2508.16306 [R] |
| 内在维数自适应 | 流形数据 KL 步数对 k 线性且 sharp；DDPM 自动近 k-线性 | Potaptchik COLT 2025；Huang–Wei–Chen MOR 2026 |
| 端到端统计 | 扩散是 Besov 类近 minimax 分布估计器；score 估计率 $\tilde\Theta(n^{-2/(d+4)})$ | Oko ICML 2023；Wibisono COLT 2024 |
| FM 理论 | almost minimax（1≤p≤2）；$\sigma_t\asymp\sqrt t$ 最优；$W_2$ 误差界 | Fukumizu ICLR 2025；Benton TMLR 2024 |
| PF-ODE（确定性） | TV 率 O(k/T) 自适应内在维数；端到端近 minimax 需同时控 Jacobian 误差 | Tang–Yan 2025；2503.09583 [R] |
| OT map 估计 | minimax 率 $n^{-2\alpha/(2\alpha-2+d)}$；plug-in 同最优 + CLT；一般函数空间覆盖神经 map | Hütter–Rigollet；Manole；Divol（均 AoS）|
| 熵正则侧 | entropic map 估计器兼顾率与可扩展性；率只取决于「简单」一方 | Pooladian–Niles-Weed；Groppe–Hundrieser JMLR 2024 |
| 函数空间 FM（Q3 新） | 有限系数/点值离散化下速度目标强 L² 收敛 + 端到端 Wasserstein 界 | 2608.04531 [R] |

关键空白（= 我们的机会）：现有 FM 统计理论只覆盖独立耦合——OT-CFM 的端到端收敛率是 diffusion×OT 的天然交叉定理（§8.1 #5）。

### 3.7 随机最优控制统一视角（T02）

Hopf–Cole/HJB 把 ELBO 解释为 verification theorem（`reports/2211.01364.md`）；SOC 求解化为回归（SOCM）；reward 微调 = memoryless SOC，须用 $\sigma(t)=\sqrt{2\eta_t}$ 消初值偏差（Adjoint Matching）。对我们的意义：把终端 reward 换成传输代价泛函，即得「把预训练扩散控制到目标耦合」的原理性机制。

### 3.8 四大理论张力（本报告的分析主轴）

1. **耗散 vs 测地**：同一 $\mathcal P_2$ 流形上，扩散走 KL 梯度流（JKO），OT 走测地线（BB）——一切纠葛源于两种变分结构不重合。
2. **逐时刻最优 vs 全局最优**：PF-ODE 每一瞬是梯度场，复合起来不是（Lavenant 反例的本质）。
3. **直 vs 最优**：直线化降代价但不动点未必最优（Hertrich）；c-RF 投影与 N-循环单调条件修复最优性（八月）。
4. **加速 vs 统计精度**：迭代复杂度与统计率各自 sharp，联合前沿无人刻画。

深读新增的第五条工程张力（来自 T14/T12）：**改耦合 vs 保边缘**——实例级噪声/端点选择提高兼容性但改变有效初始分布；文献几乎不报告边缘漂移。这是 §8.2 的立项动机。
