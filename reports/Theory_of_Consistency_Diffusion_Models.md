# Theory of Consistency Diffusion Models: Distribution Estimation Meets Fast Sampling

> Zehao Dou, Minshuo Chen, Mengdi Wang, Zhuoran Yang · ICML (PMLR v235) 2024 · [PMLR](https://proceedings.mlr.press/v235/dou24a.html) · 证据级 [P] · 课题 T10 一致性模型与少步蒸馏的 OT 视角
> **一句话**：把一致性模型训练形式化为 Wasserstein 距离最小化，给出蒸馏与免蒸馏两种训练的 W1 估计率，蒸馏情形与原扩散模型同阶。

## 1. 问题

扩散模型采样需要 500 到 1000 步（Sec. 1），因为 score 网络规模大（作者举例最小的 stable diffusion 模型超过 890M 参数，Sec. 1）。一致性模型（consistency models, CM）通过训练一个网络直接学习概率流 ODE 的自一致映射，把多步采样压缩到极少步甚至一步，但此前缺乏统计理论。作者提出的开放问题是：一致性模型估计数据分布的统计误差率是多少？与 vanilla 扩散模型相比如何？（Sec. 1）

此前 Song et al. (2023) 只给出了迭代训练算法，训练目标不清晰（Sec. 3）。Lyu et al. (2023) 的采样理论假设 score 函数和多步 backward 采样器已被准确估计，而本文不要求这些假设，直接给出样本复杂度界（Sec. 1.1）。本文的定位是「首个一致性模型统计理论」（Abstract）。

## 2. 方法

核心思想：把一致性模型的训练形式化为 Wasserstein 距离最小化问题。一致性网络 $f_\theta(x,t)$ 满足边界条件 $f_\theta(x,\varepsilon)=x$，对 $t\in(\varepsilon,T]$ 由自由形式网络 $F_\theta(x,t)$ 给出（Eq. 6）。理想情况下，$F_\theta$ 把同一 ODE 轨迹上的任意点映射到同一端点 $x_\varepsilon$（Sec. 3）。

训练目标基于时间不变性：对离散时间点 $\varepsilon=t_0<t_1<\cdots<t_N=T$，取子集 $\{\tau_k\}_{k\in[N']}$（$\tau_k=t_{kM}$，$N=N'M$），要求

$$f_\theta(\cdot,\tau_k)_\sharp X_{\tau_k} \overset{\text{law}}{=} f_\theta(\cdot,\tau_{k-1})_\sharp X_{\tau_{k-1}} \overset{\text{law}}{=} X_\varepsilon \quad \forall k\in[N'].$$

对应的总体损失为

$$\sum_{k=1}^{N'} W_1\left(f_\theta(\cdot,\tau_k)_\sharp X_{\tau_k},\; f_\theta(\cdot,\tau_{k-1})_\sharp X_{\tau_{k-1}}\right), \tag{Eq. 7}$$

其中 $X_t=\text{Law}(x_t)=m(t)p_{\text{data}}\star\mathcal{N}(0,\sigma(t)^2I)$（Eq. 4 下的 VP-SDE 边际）。实际用经验分布 $\widehat p_{\text{data}}=\frac{1}{n}\sum_j\delta_{x_j}$ 替换 $p_{\text{data}}$，得到经验损失（Eq. 8）。

两种训练方式（Sec. 3）：

- **蒸馏（distillation, CD）**：用预训练 score 估计器 $s_\phi(x,t)$ 沿概率流 ODE 做 $M$ 步离散更新，从 $x_{\tau_k}\sim X_{\tau_k}$ 得到 $\widehat x_{\tau_{k-1}}^{\phi,M}=G^{(M)}(x_{\tau_k},\tau_k;\phi)$。单步更新为 $\widehat x_{t_{k-1}}^\phi = x_{t_k}-\Delta t\cdot\Phi(x_{t_k},t_k;\phi)$，VP 框架下 $\Phi(x_{t_k},t_k;\phi)=-\frac{\beta(t_k)}{2}x_{t_k}-\frac{\beta(t_k)}{2}s_\phi(x_{t_k},t_k)$（Eq. 9–10）。训练目标：
$$L_N^{\text{CD}}(\theta;\phi)=\sum_{k=1}^{N'} W_1\left(f_\theta(\cdot,\tau_k)_\sharp X_{\tau_k},\; f_\theta(\cdot,\tau_{k-1})_\sharp \widehat X_{\tau_{k-1}}^{\phi,M}\right), \tag{Eq. 11}$$
其中 $\widehat X_{\tau_{k-1}}^{\phi,M}=G^{(M)}(\cdot,\tau_k;\phi)_\sharp X_{\tau_k}$。优化在 $\text{Lip}(R)$ 函数类上做（Eq. 12）。

- **免蒸馏（isolation, CT）**：不用预训练 score 模型，改用 Tweedie 公式的经验版本。Lemma 3.1 证明经验近似 $-\mathbb{E}_{x_0\sim\widehat p_{\text{data}}}[\frac{x_t-m(t)x_0}{\sigma(t)^2}\mid x_t]$ 恰好等于经验分布 $\widehat X_t=m(t)\widehat p_{\text{data}}\star\mathcal{N}(0,\sigma(t)^2I)$ 的 score $\nabla\log\widehat p_t(x_t)$，无需额外训练。训练目标：
$$L_N^{\text{CT}}(\theta)=\sum_{k=1}^{N'} W_1\left(f_\theta(\cdot,\tau_k)_\sharp X_{\tau_k},\; f_\theta(\cdot,\tau_{k-1})_\sharp X_{\tau_{k-1}}\right), \tag{Eq. 15}$$
优化同样在 $\text{Lip}(R)$ 上做（Eq. 16）。

与 Song et al. (2023) 原始训练的区别：本文取 $\lambda(\cdot)\equiv 1$、$\theta^-=\theta$、允许多步 ODE solver、用 $W_1$ 替代逐点度量 $d(\cdot,\cdot)$（Sec. 3 末）。作者指出 $W_1$ 比逐点 $l_2$ 弱，小 $l_2$ 蕴含小 $W_1$，因此本文分析在更弱条件下成立并覆盖 $l_2$ 情形。

## 3. 理论结果

**假设**（Sec. 4）：

- Assumption 4.1（Gaussian tail）：$p_{\text{data}}$ 二次连续可微且有 Gaussian tail，即存在 $\alpha_1,\alpha_2>0$ 使 $P_{X\sim p_{\text{data}}}[\|X\|_2\ge R_0]\le P_{Z\sim\mathcal{N}(0,I)}[\|Z\|_2\ge\frac{R_0-\alpha_1}{\alpha_2}]$ 对所有 $R_0>\alpha_1$ 成立；这直接给出有限二阶矩 $M_2^2=\mathbb{E}_{X\sim p_{\text{data}}}\|X\|_2^2<\infty$。
- Assumption 4.2（Lipschitz score）：$\nabla\log p_t(\cdot)$ 对任意 $t\in[0,T]$ 是 $L$-Lipschitz。
- Assumption 4.3（Lipschitz 基线）：基线一致性模型 $f_{\theta^*}(\cdot,t)$ 对任意 $t\in[\varepsilon,T]$ 是 $R$-Lipschitz。作者用 Caffarelli (1992) 的变换正则性论证该假设自然（Remark 4.1），并强调不要求访问 $\theta^*$（Remark 4.2）。
- Assumption 4.4（有界系数）：$\underline\beta\le\beta(t)\le\overline\beta<\frac{1}{d\log n+d^2\log(d/\varepsilon)}$ 对所有 $t\in[\varepsilon,T]$。
- Assumption 4.5（有界支撑，仅 isolation 用）：$P_{X\sim p_{\text{data}}}[\|X\|_2\le R_0]=1$。作者说明这是为了给经验 score $\nabla\log\widehat p_t(\cdot)$ 提供 Lipschitz 连续性以替代 Assumption 4.2；Remark 4.6 指出可放松到 sub-Gaussian tail。

**Theorem 4.1（蒸馏）**：在 Assumptions 4.1–4.4 下，存在 score 估计器 $s_\phi(\cdot,t)$ 使从 Eq. 12 得到的 $\widehat f_\theta$ 满足

$$\mathbb{E}\left[W_1\left(\widehat f_\theta(\cdot,T)_\sharp\mathcal{N}(0,I),\; p_{\text{data}}\right)\right] \lesssim \sqrt{d}R\exp(-\underline\beta T/2) + \frac{R\overline\beta dLT}{\sqrt{M}} + 6RN'n^{-1/d} + R\overline\beta\sqrt{d}\,\varepsilon_{\text{score}}\cdot\sqrt{\frac{TN'}{\varepsilon}} + \sqrt{d\overline\beta\varepsilon},$$

其中 $R$ 是优化问题的 Lipschitz 约束，期望对数据集取，$\varepsilon_{\text{score}}=O(n^{-1/(d+5)})$ 是 score 估计误差。五个误差项依次为：前向过程收敛误差、ODE 离散化误差、经验-总体集中差距、score 估计误差、早停误差（Sec. 4 解释）。

**Remark 4.3**：取 $\underline\beta,\overline\beta\asymp\frac{1}{d\log n}$，$T=(\log n)^3$，$M=d^2n^{1/(d+5)}$，$N'=\log n$，$\varepsilon=\sqrt{TN'}n^{-1/(d+5)}=\log^2 n\cdot n^{-1/(d+5)}$，得到

$$\mathbb{E}\left[W_1\left(\widehat f_\theta(\cdot,T)_\sharp\mathcal{N}(0,I),\; p_{\text{data}}\right)\right] \lesssim \sqrt{\log n}\cdot n^{-\frac{1}{2(d+5)}},$$

即 $\widetilde O(n^{-1/(2(d+5))})$，与 Chen et al. (2023a) 的 vanilla 扩散模型分布估计率一致。作者在 Remark 4.4 承认该率对维度 $d$ 有指数依赖，但指出这是无进一步假设下的最优；低维结构（Chen et al., 2023a）或参数化设定（Yuan et al., 2024）可改善。

**Theorem 4.2（免蒸馏）**：在 Assumptions 4.3–4.5 下，从 Eq. 16 得到的 $\widehat f_\theta$ 满足

$$\mathbb{E}\left[W_1\left(\widehat f_\theta(\cdot,T)_\sharp\mathcal{N}(0,I),\; p_{\text{data}}\right)\right] \lesssim \sqrt{d}R\exp(-\underline\beta T/2) + Rn^{-1/d} + \frac{d\overline\beta R_0^2T}{\underline\beta^2\varepsilon^2\sqrt{M}} + \sqrt{d\overline\beta\varepsilon}.$$

**Remark 4.5**：取 $\underline\beta,\overline\beta\asymp\frac{1}{d\log n}$，$\varepsilon=n^{-2/d}$，$T=d(\log n)^3$，$M=d^2(\log n)^8 n^{10/d}$，得到 $\mathbb{E}[W_1(\widehat f_\theta(\cdot,T)_\sharp\mathcal{N}(0,I),p_{\text{data}})]\lesssim n^{-1/d}$，即 $\widetilde O(n^{-1/d})$。作者注明该率与蒸馏情形不可直接比较，因为训练过程不同（Remark 4.5）。

**证明结构**（Sec. 5）：构造基线 DDPM solver $f_{\theta^*}$（$N$ 层 ResNet 结构，注入预训练 score 估计器），利用最优性不等式 $L_N^{\text{CD}}(\widehat\theta;\phi)\le L_N^{\text{CD}}(\theta^*;\phi)$（Eq. 18）把性能差距分解为四项 $I_1+I_2+I_3+I_4$（Lemma 5.2）。$I_1,I_2$ 是 ODE 离散化误差，$I_3,I_4$ 是经验-总体 $W_1$ 集中差距。Lemma 5.3 给出 $I_1+I_2\lesssim R\overline\beta dL\cdot\frac{T}{\sqrt{M}}+R\overline\beta\sqrt{d}n^{-1/(d+5)}\cdot\sqrt{\frac{TN'}{\varepsilon}}$；Lemma 5.4 给出 $\mathbb{E}[I_3+I_4]\le 6RN'n^{-1/d}$。Lemma 5.5 给出 $W_1(X_T,\mathcal{N}(0,I))\lesssim\sqrt{d}\exp(-\underline\beta T/2)$；Lemma 5.6 给出 DDPM solver 估计误差 $W_1(f_{\theta^*}(\cdot,T)_\sharp X_T,p_{\text{data}})\lesssim\overline\beta Ld\sqrt{T}\Delta t+\overline\beta\sqrt{\frac{dT}{\varepsilon}}n^{-1/(d+5)}+\sqrt{d\overline\beta\varepsilon}$。Lemma 5.1 引用 Chen et al. (2023a) 的 score 网络逼近结果，给出 mean integrated squared error $\widetilde O(\frac{1}{\varepsilon}n^{-2/(d+5)})$。

## 4. 实验与数字

本文是纯理论论文，**无实验**。全文没有数据集、基线或数值实验表格。所有数字均来自定理与假设中的理论量（如 $n^{-1/(2(d+5))}$、$n^{-1/d}$、$890M$ 参数的 stable diffusion 模型规模举例、$500$–$1000$ 步采样步数）。

## 5. 在 OT×扩散地图中的位置

本文是 T10（一致性模型与少步蒸馏的 OT 视角）的**理论奠基工作**，对应课题背景中「理论层——蒸馏/一致性损失如何控制学生与教师（或数据）分布之间的 Wasserstein 距离」这一条线。具体关系：

- **继承**：直接建立在 Song et al. (2023) 的 CM 训练算法之上，把其逐点 $l_2$ 损失推广为 $W_1$ 分布差异最小化；统计率结果与 Chen et al. (2023a) 的 vanilla 扩散模型估计率对齐（Remark 4.3 明确引用）。采样理论方面与 Lyu et al. (2023) 互补——后者假设 score 与采样器已准确，本文给出达到该准确度的样本复杂度。
- **与 FMM（TMLR 2025）的关系**：FMM 在理论上统一了 CM/CTM/PD 等两时间 flow map 学习方案，证明 Lagrangian/Eulerian 蒸馏损失控制教师-学生 $W_2$。本文是更早的、针对 CM 单一方法的 $W_1$ 统计估计率结果，两者共同构成「一致性训练 = 分布差异最小化」的理论证据链。本文的 $W_1$ 目标与 FMM 的 $W_2$ 上界在度量选择上不同，但方向一致。
- **与 Li 等人的离散化步数下界**：课题背景提到 Li 给出 $O(d^{5/2}/\varepsilon)$ 离散化步数下界；本文 Theorem 4.1 的离散化误差项 $\frac{R\overline\beta dLT}{\sqrt{M}}$ 是上界方向，$M$ 的选取（Remark 4.3 中 $M=d^2n^{1/(d+5)}$）体现维度依赖，但未讨论下界。
- **在推理管线中的位置**：本文处理的是「训练目标 → 分布估计误差」这一环节，不涉及采样调度优化或耦合选取。它把 CM 的训练从算法层面提升到统计估计层面，为后续「多少步才够」「蒸馏是否保真」等问题提供了误差分解框架（前向收敛、离散化、集中、score 估计、早停五项）。

## 6. 局限与批评

**作者承认的**：

1. 蒸馏情形的估计率 $\widetilde O(n^{-1/(2(d+5))})$ 对维度 $d$ 有指数依赖，作者明确说这是无进一步假设下的最优，但实际数据有低维结构时该率过悲观（Remark 4.4）。
2. Isolation 情形需要 Assumption 4.5（有界支撑），比蒸馏情形的 Gaussian tail 强得多；作者解释这是为了经验 score 的 Lipschitz 连续性，并说可放松到 sub-Gaussian（Remark 4.6），但未在定理中给出放松后的完整证明。
3. 本文用 $W_1$ 替代 Song et al. (2023) 的逐点 $l_2$ 损失，作者承认 $W_1$ 是更弱的度量（Sec. 3 末），因此理论覆盖的是比实际训练目标更宽松的设定。

**读出来的**：

1. 所有定理都假设优化问题（Eq. 12、Eq. 16）达到**全局最优** $\widehat\theta$，且函数类 $\text{Lip}(R)$ 的 $R$ 出现在最终误差界中。实际 CM 训练用深度网络 + EMA + 随机优化，既不保证全局最优，也不保证严格 $R$-Lipschitz；定理没有刻画优化误差与 $R$ 的估计误差。
2. Assumption 4.4 要求 $\overline\beta<\frac{1}{d\log n+d^2\log(d/\varepsilon)}$，即噪声调度上界随 $n$ 和 $d$ 缩小。这与实际 VP-SDE 中 $\beta(t)$ 通常取线性函数（Sec. 2 提到「usually chosen as a linear function」）的设定不一致——实际 $\beta(t)$ 不随样本量 $n$ 调整。定理的调参（Remark 4.3、4.5）是理论存在性结果，不是可操作的训练配置。
3. 蒸馏情形的 score 估计误差项 $R\overline\beta\sqrt{d}\,\varepsilon_{\text{score}}\sqrt{\frac{TN'}{\varepsilon}}$ 依赖 $\varepsilon_{\text{score}}=O(n^{-1/(d+5)})$，但 Lemma 5.1 的 score 逼近结果是引用 Chen et al. (2023a) 的 ReLU 网络类，本文没有给出该网络类与 $s_\phi$ 的具体对应关系，也没有讨论 score 网络的训练误差（只假设存在这样的估计器）。

## 7. 对我们的启发

1. **误差分解框架可直接用于 OT-aware 采样调度（接切入点 #2）**：Theorem 4.1 把 $W_1$ 误差拆成五项，其中离散化误差 $\frac{R\overline\beta dLT}{\sqrt{M}}$ 与 score 估计误差 $R\overline\beta\sqrt{d}\,\varepsilon_{\text{score}}\sqrt{\frac{TN'}{\varepsilon}}$ 都显式依赖时间网格 $\{\tau_k\}$ 与子步数 $M$。如果我们在 CM 训练中引入 OT 耦合（如 minibatch Sinkhorn 替换独立耦合），可以沿用这个分解，检查耦合选择是否改变集中差距项 $6RN'n^{-1/d}$ 或 score 误差项的系数。本文的 $W_1$ 目标本身就是一个可操作的分布级训练损失，OT 耦合的引入相当于在 Eq. 11/15 的 pushforward 分布之间加一个传输计划约束，理论上可尝试把 $W_1$ 换成 entropic OT 并重跑 Lemma 5.2–5.4 的分解。
2. **Isolation 情形的 $n^{-1/d}$ 率提示免训练 CM 的维度瓶颈**：Theorem 4.2 的 $\widetilde O(n^{-1/d})$ 比蒸馏情形的 $\widetilde O(n^{-1/(2(d+5))})$ 慢得多，且需要 $M=d^2(\log n)^8 n^{10/d}$ 的巨额子步数（Remark 4.5）。这从统计上解释了为什么免蒸馏 CT 在低维或低内在维数据上可行、在高维图像上需要蒸馏或结构假设。对我们做「免训练 batch 级保边缘噪声指派 MPNA」（切入点 #1）的直接含义：如果 MPNA 能降低经验 score 的方差或改善其 Lipschitz 常数，就可能把 isolation 的 $n^{-1/d}$ 率向蒸馏情形的率靠拢，值得在定理层面检验 Assumption 4.5 的 Lipschitz 条件是否因耦合而放松。
3. **「蒸馏保真」的度量选择**：本文用 $W_1$ 而非 $l_2$ 作为训练目标，并证明 $W_1$ 更弱。这为课题背景中「teacher-anchored Wasserstein 评测协议」（切入点 #4）提供了理论支撑：如果实际 CM 用 $l_2$ 训练，那么 $W_1$ 上界是更保守的保真度量；反过来，如果我们想度量蒸馏学生与教师分布的 $W_2$ 随 NFE 的幂律，本文的误差分解（特别是前向收敛项 $\sqrt{d}R\exp(-\underline\beta T/2)$ 与早停项 $\sqrt{d\overline\beta\varepsilon}$）给出了 NFE 无关的固有误差下限，可作为幂律拟合的截距项。

## 8. 资源

代码：未公开（纯理论论文，无代码）。

相关论文 arXiv id 互链：

- Song et al. (2023) Consistency Models：arXiv:2303.01469
- Lyu et al. (2023)（CM 采样理论）：原文参考文献未给 arXiv id，仅列作者与年份
- Chen et al. (2023a)（低维数据扩散模型 score 逼近与分布恢复）：arXiv:2302.07194
- Chen et al. (2022)（minimal data assumptions 的扩散采样理论）：arXiv:2209.11215
- Benton et al. (2023)（stochastic localization 线性收敛）：arXiv:2308.03686
- Block et al. (2020)（denoising auto-encoder + Langevin）：arXiv:2002.00107
- De Bortoli et al. (2021)（diffusion Schrödinger bridge）：NeurIPS 34
- Caffarelli (1992)（凸势映射正则性）：J. Amer. Math. Soc. 5(1):99–104
- Yuan et al. (2024)（参数化 score 估计率）：原文参考文献未给 arXiv id
