# Diffusion Models are Minimax Optimal Distribution Estimators

> Kazusato Oko, Shunta Akiyama, Taiji Suzuki · ICML 2023（PMLR 202；文本为 arXiv 2303.01861v1） · [arXiv](https://arxiv.org/abs/2303.01861) · 证据级 [P] · 课题 T06 扩散/流生成模型的收敛性与统计理论
> **一句话**：真密度在 Besov 类 $B^s_{p,q}([-1,1]^d)$ 且经验 score matching 被正确最小化时，扩散模型的估计率为 TV $\tilde O(n^{-s/(2s+d)})$（minimax）、$W_1$ $\tilde O(n^{-(s+1-\delta)/(2s+d)})$（几乎 minimax）；低维子空间数据率只依赖 $d'$。

## 1. 问题
此前的扩散收敛理论都假设 score 已给定（$L^2$ 或逐点误差小），从未回答「用 $n$ 个样本学 score 后，生成分布离真分布多远」这一分布估计问题（Sec. 1）。唯一例外 De Bortoli 2022 得到 $W_1$ 的 $n^{-1/d}$，但那是经验测度收敛率（Dudley 下界），不能随密度光滑度改进。本文首次把「score 网络逼近—经验风险泛化—误差传导到分布」三段打通，问：扩散模型是否是 minimax 最优的分布估计器？

## 2. 方法
**设定**（Sec. 2）。前向 OU $dX_t=-\beta_tX_tdt+\sqrt{2\beta_t}dB_t$，$X_t|X_0\sim N(m_tX_0,\sigma_t^2)$，$1-m_t\simeq t\wedge1$、$\sigma_t\simeq\sqrt t\wedge1$；反向从 $N(0,I_d)$ 起、$T=\tilde O(1)$、在 $\underline T>0$ 早停。训练目标为区间 $[\underline T,T]$ 上的去噪 score matching Eq.(1)/(2)，假设类为稀疏 ReLU 网络 $\Phi(L,W,S,B)$（Def. 2.1）。

**Step 1 逼近（Sec. 3）**。核心构造是**diffused B-spline 基**：Besov 函数可由 $N$ 个张量积 B-spline $M^d_{k,j}$ 以 $L^2$ 误差 $N^{-s/d}$ 逼近（Lemma 3.2），而 $p_t=\int p_0(y)K_t(x|y)dy$ 对 $p_0$ 线性，于是 $p_t\approx\sum\alpha_{k,j}E_{k,j}(x,t)$，$E_{k,j}=\prod_iD_{k,j}(x_i,t)$ 是 B-spline 与高斯核的一维卷积；$D_{k,j}$、$m_t$、$\sigma_t$、有理函数 $\nabla p_t/p_t$ 各由 polylog 大小的网络实现（Lemma 3.3/3.4）。**Theorem 3.1**：存在 $\varphi_{\rm score}\in\Phi$，对所有 $t\in[\underline T,T]$，
$$\int p_t\|\varphi_{\rm score}(x,t)-\nabla\log p_t(x)\|^2dx\lesssim\frac{N^{-2s/d}\log N}{\sigma_t^2},$$
$L=O(\log^4N)$、$\|W\|_\infty=O(N\log^6N)$、$S=O(N\log^8N)$、$B=e^{O(\log^4N)}$，且 $\|\varphi_{\rm score}(\cdot,t)\|_\infty=O(\sigma_t^{-1}\log^{1/2}N)$。大 $t$ 时噪声带来的光滑性（Lemma 3.5：$|\partial^kp_t|\le C_a/\sigma_t^k$）给出更紧的 **Lemma 3.6**：$t\ge2t_*$、$t_*\ge N^{-(2-\delta)/d}$ 时误差 $\lesssim N^{-2(s+1)/d}/\sigma_t^2$，网络稀疏度只需 $S=O(t_*^{-d/2}N^{\delta/2})$。

**Step 2 泛化（Sec. 4）**。Lemma 4.1（Vincent 2011）：DSM 损失 = $L^2(p_t)$ score 误差 + 常数。限制 $\|\varphi(\cdot,t)\|_\infty\lesssim\log^{1/2}n/\sigma_t$ 后 $\sup\|\ell_s\|_\infty\lesssim\log^2n$，覆盖数 $\log\mathcal N\lesssim SL\log(\cdot)$（Lemma 4.2）。**Theorem 4.3**：经验最小化子满足 $\mathbb E\int_{\underline T}^T\!\!\int\|\hat s-\nabla\log p_t\|^2p_t\lesssim\inf_{s\in\Phi'}(\text{逼近})+\frac{\sup\|\ell\|_\infty\log\mathcal N}{n}+\delta$，取 $N=n^{d/(2s+d)}$ 得 Eq.(4)：$\lesssim n^{-2s/(d+2s)}\log^{18}n$。Sec. 4.1 讨论对 $t,x_t$ 也抽样时：均匀抽 $t$ 需 $M\gtrsim n\underline T^{-1}$ 个样本；改为 $\mu(t)\propto1/t$ 抽样并配权 $\lambda(t)\propto t$ 则 $M=n$ 足够。

**Step 3 传导到分布（Sec. 5）**。TV：$\mathbb E\,\mathrm{TV}(X_0,\hat Y_{T-\underline T})\lesssim\mathrm{TV}(X_0,X_{\underline T})+\mathrm{TV}(X_T,N(0,I))+\mathrm{TV}(\bar Y,Y)$，末项用 Girsanov（Prop. D.1）$\lesssim\sqrt{\int\mathbb E\|\hat s-\nabla\log p_t\|^2dt}$（Eq.(6)）。$W_1$：**切换网络**——按 $t_1=2n^{-(2-\delta)/(2s+d)}<\dots<t_{K^*}$（$K^*=O(\log n)$）分段，每段用 Lemma 3.6 的小网络单独训练；Lemma 5.5 给 $W_1(\bar Y^{(i-1)},\bar Y^{(i)})\le\tilde O(1)\sqrt{t_{i-1}\,\mathbb E\int_{t_{i-1}}^{t_i}\|\hat s-\nabla\log p_t\|^2}$：越靠近 $t=0$ 的 score 越难学，但其误差对 $W_1$ 的贡献被路径位移 $\sqrt{t_i}$ 压缩——「Wasserstein 是运输距离」是拿到额外 $n^{-1/(2s+d)}$ 的关键。

## 3. 理论结果
- **假设**：2.4 $p_0$ 支撑于 $[-1,1]^d$，$C_f^{-1}\le p_0\le C_f$，$p_0\in U(B^s_{p,q})$，$s>d(1/p-1/2)_+$（允许不连续密度，对比 Lipschitz 假设的 Lee/Chen 系列）；2.5 $\beta_t\in[\underline\beta,\bar\beta]$ 且 $C^\infty$；2.6 边界带 $[-1,1]^d\setminus[-1+a_0,1-a_0]^d$（$a_0\approx n^{-(1-\delta)/(d+2s)}$）上 $p_0\in C^\infty$——为补 $p_t$ 在支撑边缘无下界时 $\nabla p_t/p_t$ 的误差放大；作者称可换成 LSI 式慢衰减且不伤 minimax 率。
- **Theorem 5.1（TV）**：$\underline T=n^{-O(1)}$、$T=\frac{s\log n}{\beta(d+2s)}$ 时 $\mathbb E[\mathrm{TV}(X_0,\hat Y_{T-\underline T})]\lesssim n^{-s/(2s+d)}\log^{\frac{5d+8s}{2d}}n$。**Proposition 5.2**（下界）：$\inf_{\hat\mu}\sup_{p\in B^s_{p,q}}\mathbb E\,\mathrm{TV}\gtrsim n^{-s/(2s+d)}$ ⇒ 对数因子内 minimax。
- **Theorem 5.4（$W_1$）**：任意固定 $\delta>0$，$\mathbb E[W_1(X_0,\hat Y_{T-\underline T})]\lesssim n^{-(s+1-\delta)/(d+2s)}$，取 $\underline T=n^{-2(s+1)/(d+2s)}$、$T=\frac{(s+1)\log n}{\beta(d+2s)}$。**Proposition 5.3**（Niles-Weed–Berthet 2022）：$W_1$ 下界 $n^{-(s+1)/(2s+d)}$（$d\ge2$）⇒ 差 $n^{\delta/(2s+d)}$，「几乎」minimax。
- **Theorem 5.7（离散化）**：步长 $\eta$ 的指数积分器格式 $\mathbb E[\mathrm{TV}(X_0,Y^d)]\lesssim\tilde O(\eta^2\underline T^{-3}+n^{-s/(d+2s)})$，$\eta=\underline T^{1.5}n^{-s/(2s+d)}=\mathrm{poly}(n^{-1})$ 即可忽略——步数是 $n$ 的多项式，不是本文优化对象。
- **Theorem 6.4（内在维数）**：$p_0$ 支撑在 $d'$ 维线性子空间 $V=\{Ax\}$、其坐标密度 $q\in U(B^s_{p,q}([-1,1]^{d'}))$ 且有上下界（Assumption 6.1–6.3）时 $\mathbb E[W_1]\lesssim n^{-(s+1-\delta)/(d'+2s)}$：指数只含 $d'$。
- Remark 5.6：切换网络可能在实践中由隐式正则化 + 随 $t$ 增大的权重 $\lambda(t)$ 隐式实现——这是推测，非定理。

## 4. 实验与数字
无实验；全文理论（ICML 版正文 + 附录 A–F 证明）。

## 5. 在 OT×扩散地图中的位置
T06 统计线的开山之作，其「diffused B-spline 逼近 + 覆盖数泛化 + 时间分段切换网络」三件套被 2405.20879（Fukumizu 等）整体搬到 flow matching、用 Alekseev–Gröbner 替代 Girsanov 后得到 $W_r$（$r\le2$）几乎 minimax；2402.15602（Zhang 等）用核估计器替换网络并去掉密度下界假设 2.4/2.6；2503.09583（Cai–Li）在 PF-ODE 上做同类端到端。它依赖 2209.11215 式的 Girsanov TV 传导，但把「score 给定」换成「score 学出」。Theorem 6.4 是流形/低维统计率线（后续 Azangulov 等、2410.09046）的第一个 quantitative 结果，只覆盖线性子空间。对 T04 的 OT map 统计（1905.05828、2107.12364）：两边同用 Niles-Weed–Berthet 的 Wasserstein minimax 下界作标尺，可直接比较「学分布」与「学 map」的样本复杂度。对综合报告的「加速是否吃掉统计精度」张力：Theorem 5.7 说离散化步数多项式即可，不给加速空间。

## 6. 局限与批评
作者承认的：(1) 边界高光滑假设 2.6 是技术性的（Sec. 2.1）；(2) $W_1$ 最优率依赖显式切换多网络（Remark 5.6 只给间接证据）；(3) 每个 $x_{0,i}$ 需 poly 个 $(t,x_t)$ 样本或改 $t$ 的抽样分布（Sec. 4.1）。

我读出来的：(1) 立方体支撑 + 密度上下界排除了流形、多峰零密度区与无界支撑——和 Sec. 6 的「线性子空间」一样，是对图像数据的强理想化；(2) 结论针对**经验风险的精确最小化子**，优化是否能到达该解不在范围内（作者也明说）；(3) $\log^{18}n$、$\log^{(5d+8s)/(2d)}n$ 这类对数幂在有限 $n$ 下不可忽略；(4) 网络规范 $B=e^{O(\log^4N)}$ 极大，和实际训练网络的参数范围不对应；(5) 「minimax」的 max 取在 Besov 球上，任何有额外结构的数据都可能有更快的率——本文的最优性是最坏情形意义的。

## 7. 对我们的启发
1. Lemma 5.5 的「$\sqrt{t_i}\times$ 该段 score 误差」权重是设计 OT-aware 训练权重（切入点 #2）的定理依据：在 Wasserstein 目标下应把学习预算向大 $t$ 倾斜，而非均匀或向小 $t$ 倾斜；可直接用它论证 $\lambda(t)$ 随 $t$ 增大的调度。
2. Sec. 6 的子空间结果 + 2405.20879 的 FM 版本，组合起来给潜空间 OT 引导（#7 医学 SB）一个可引用的论断：在 $d'$ 维潜空间做扩散/流，率的指数只含 $d'$——这是「先降维再做 OT 引导」的统计学理由。
3. 做 $n$-scaling 验证实验时，应按 Theorem 5.1/5.4 的处方设 $\underline T=n^{-O(1)}$、$T\asymp\log n$；若不早停，误差会被 $t\to0$ 处 score 爆炸主导而看不到 minimax 率。

## 8. 资源
代码：未公开。
互链：2405.20879（FM 版本）、2402.15602（去密度下界）、2503.09583（PF-ODE 版本）、2209.11215（Girsanov 传导）、2402.07747（score 估计本身的 minimax 率）、2410.09046（流形数据）、1905.05828 / 2107.12364（同标尺的 OT map 统计率）。
