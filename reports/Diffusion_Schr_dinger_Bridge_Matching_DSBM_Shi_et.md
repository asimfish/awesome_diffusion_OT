# Diffusion Schrödinger Bridge Matching

> Yuyang Shi, Valentin De Bortoli, Andrew Campbell et al. · NeurIPS 2023 · [arXiv:2303.16852](https://arxiv.org/abs/2303.16852) · 证据级 [P] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：提出 IMF 交替 Markov/reciprocal 投影，用 bridge-matching 回归实现 DSBM，解决 DSB 误差累积与遗忘问题。

## 1. 问题

本文解决 Schrödinger Bridge（SB）问题的数值求解。SB 是动态熵正则最优传输（EOT）：给定参考扩散 $Q$，在所有满足两端边缘约束 $P_0=\pi_0, P_T=\pi_T$ 的路径测度中，找与 $Q$ 的 KL 散度最小者（Eq. 6）。SB 恢复 EOT 的动态版本，而 Denoising Diffusion Models（DDMs）和 Flow Matching Models（FMMs）不保证逼近 OT 映射（Sec. 1）。

此前方法有两类不足。经典数值方法（Bernton et al. 2019; Chen et al. 2016; Finlay et al. 2020; Caluya and Halder 2021; Pavon et al. 2021）局限于低维（Sec. 1）。基于 Iterative Proportional Fitting（IPF）的深度方法 DSB（De Bortoli et al. 2021）和同类算法（Vargas et al. 2021; Chen et al. 2022）虽然经验上可扩展，但数值误差随迭代累积（Fernandes et al. 2021），且 IPF 迭代不保持两端边缘（Table 1）。此外，DSB 需要完整轨迹缓存，训练中还会出现对参考桥的「遗忘」（Sec. 4）。

## 2. 方法

核心思想是 **Iterative Markovian Fitting（IMF）**：交替做 Markov 投影和 reciprocal 投影。Markov 投影 $M^\star = \mathrm{proj}_{\mathcal{M}}(\Pi)$ 把混合桥测度 $\Pi = \Pi_{0,T} Q^{|0,T}$ 投影到 Markov 路径测度空间 $\mathcal{M}$，对应 SDE 漂移为

$$v^\star_t(x_t) = \sigma_t^2 \mathbb{E}_{\Pi_{T|t}}[\nabla \log Q_{T|t}(X_T|X_t) \mid X_t = x_t] \tag{Definition 1}$$

reciprocal 投影 $\Pi^\star = \mathrm{proj}_{\mathcal{R}(Q)}(P) = P_{0,T} Q^{|0,T}$ 把任意路径测度投影到参考桥的 reciprocal 类 $\mathcal{R}(Q)$（Definition 3）。IMF 序列定义为

$$P_{2n+1} = \mathrm{proj}_{\mathcal{M}}(P_{2n}), \quad P_{2n+2} = \mathrm{proj}_{\mathcal{R}(Q)}(P_{2n+1}) \tag{Eq. 8}$$

关键性质：IMF 每一步都保持 $P^n_0 = \pi_0$ 和 $P^n_T = \pi_T$，与 IPF 形成对比（Table 1）。SB 是唯一同时满足 Markov、属于 $\mathcal{R}(Q)$、且两端边缘正确的路径测度（Proposition 5）。

**DSBM 算法**（Algorithm 1）用 bridge-matching 回归实现 IMF。Markov 投影通过最小化

$$\theta^\star = \arg\min_\theta \int_0^T \mathbb{E}_{\Pi_{t,T}}\big[\|\sigma_t^2 \nabla \log Q_{T|t}(X_T|X_t) - v_\theta(t, X_t)\|^2\big]/\sigma_t^2 \, dt \tag{Eq. 10}$$

学习前向漂移；同时利用 Markov 投影的时间对称性（Proposition 9），学习一个等价的后向表示（Eq. 13–14）。实际中交替前向/后向投影，以消除单方向投影在 $\pi_T$ 上累积的偏差（Sec. 4）。reciprocal 投影只需从 $M_{0,T}$ 采样端点对，再采样参考桥 $Q^{|0,T}$，无需完整轨迹缓存。

两种初始化：**DSBM-IPF** 用 $\Pi^0_{0,T} = Q_{0,T}$（等价于 IPF 迭代，Proposition 10）；**DSBM-IMF** 用独立耦合 $\Pi^0_{0,T} = \pi_0 \otimes \pi_T$；**DSBM-IMF+** 用 mini-batch EOT 求解器得到的近似 SB 耦合初始化（Sec. 6）。平衡态下可构造概率流 ODE：$dZ^\star_t = \{f_t(Z^\star_t) + \tfrac{1}{2}[v_{\theta^\star}(t, Z^\star_t) - v_{\phi^\star}(t, Z^\star_t)]\}dt$（Sec. 4）。

## 3. 理论结果

- **Proposition 2**：Markov 投影是 reverse KL 下的投影，即 $M^\star = \arg\min_{M \in \mathcal{M}} \mathrm{KL}(\Pi|M)$，且保持边缘 $M^\star_t = \Pi_t$ 对所有 $t \in [0,T]$。
- **Proposition 4**：reciprocal 投影是 forward KL 下的投影，$\Pi^\star = \arg\min_{\Pi \in \mathcal{R}(Q)} \mathrm{KL}(P|\Pi)$。
- **Proposition 5**：满足 $P_0 = \pi_0, P_T = \pi_T$、Markov 且属于 $\mathcal{R}(Q)$ 的路径测度唯一，等于 $P^{\mathrm{SB}}$。
- **Lemma 6**：Markov 投影和 reciprocal 投影各满足一个 Pythagorean 定理。
- **Proposition 7**：$\mathrm{KL}(P^{n+1}|P^{\mathrm{SB}}) \le \mathrm{KL}(P^n|P^{\mathrm{SB}}) < \infty$，且 $\lim_{n \to +\infty} \mathrm{KL}(P^n|P^{n+1}) = 0$。
- **Theorem 8**：IMF 序列有唯一不动点 $P^\star = P^{\mathrm{SB}}$，且 $\lim_{n \to +\infty} \mathrm{KL}(P^n|P^\star) = 0$。作者注明该收敛结果首次出现在同期工作 Peluchetti (2023, Theorem 2)，本文给出更简单的证明（Sec. 3.2）。
- **Proposition 10**：当函数族足够丰富时，以 $\Pi^0_{0,T} = Q_{0,T}$ 初始化的最优 DSBM 序列与 DSB 的 IPF 迭代序列一致，即 $M^n = \tilde{P}^n$ 对 $n \ge 1$。

所有理论结果均在「mild assumptions」下陈述，原文未给出精确的假设条件清单。

## 4. 实验与数字

**2D 实验**（Table 2）：在 moons、scurve、8gaussians、moons-8gaussians 四个数据集上，以 2-Wasserstein 距离（Euler 20 步）和路径能量为指标。DSBM-IPF 的 2-Wasserstein 为 0.140/0.140/0.315/0.812（±1 SD，5 seeds）；DSBM-IMF 为 0.144/0.145/0.338/0.838；DSBM-IMF+ 为 0.123/0.130/0.276/0.802。对比：DSB 为 0.190/0.272/0.411/0.987；OT-CFM 为 0.111/0.102/0.253/0.716（最佳）；RF 为 0.129/0.126/0.267/1.522。路径能量上 DSBM-IPF 为 1.598/2.110/14.91/42.16，低于 DSB（原文 Table 2 中 DSB 路径能量未报告，标为「-」）和 SB-CFM（1.649/2.144/15.08/45.69）。作者结论：不用 OT 求解器时 DSBM 优于 FM 和 CFM；DSBM 在所有数据集上优于 DSB；DSBM-IMF+ 在 SB 方法中采样误差最低，且路径能量低于 SB-CFM 全部数据集（Sec. 6）。

**高维高斯实验**（$d=50$，Figure 3, Table 3）：DSBM 的方差和协方差估计随迭代保持准确，而 DSB、IMF-b、RF 的方差估计随迭代变差（Figure 3）。Table 3 报告 21 个均匀时间点的平均 $\mathrm{KL}(P_t|P^{\mathrm{SB}}_t) \times 10^{-3}$：$d=5$ 时 DSB 3.26±1.60，SB-CFM 1.45±0.73，DSBM-IPF 1.23±0.23，DSBM-IMF 1.34±0.51；$d=20$ 时 DSB 13.0±3.49，SB-CFM 12.3±1.47，DSBM-IPF 4.42±0.76，DSBM-IMF 5.05±0.95；$d=50$ 时 DSB 32.8±1.28，SB-CFM 49.4±3.91，DSBM-IPF 8.75±0.87，DSBM-IMF 9.76±1.67。作者结论：$d=5$ 时与 SB-CFM 相近，高维时 DSBM 明显更准确（Sec. 6）。

**MNIST/EMNIST 迁移**（Figure 4, 5）：DSBM 视觉质量高于 DSB 和 RF，且训练中不出现质量退化；OT-CFM 在高维下样本质量变差；DSBM 运行时间比 DSB 约少 30%（Sec. 6）。

**CelebA 64×64 迁移**（Figure 6, 7, 8）：$\sigma^2 \in \{0.01, 0.1, 1, 10\}$ 的消融显示，FID 随 $\sigma$ 增大先改善后变差，LPIPS 对齐度随 $\sigma$ 增大单调变差（Figure 7）。同一 $\sigma=1$ 下，128×128 的对齐度优于 64×64，与噪声调度应随分辨率缩放的观点一致（Sec. 6）。

**AFHQ 512×512 迁移**（Figure 9）：cat↔wild 双向迁移生成逼真且与输入相似的样本（Sec. 6）。

**非配对流体降尺度**（Figure 10, 11）：在 Bischoff and Deck (2023) 的 64×64→512×512 数据集上，DSBM-IPF 和 DSBM-IMF 在所有频率类别上的 $\ell_2$ 距离均低于 Diffusion-fb 和 Random 基线（Figure 11）。

## 5. 在 OT×扩散地图中的位置

本文处于「第二代：IMF / bridge matching（2023）」的核心位置，与 Peluchetti (2023, IDBM) 独立提出同一 IMF 框架（作者在 Sec. 5 明确注明）。它继承 DSB（De Bortoli et al. 2021）的 SB 生成建模叙事，但用 reciprocal 投影替代 IPF 的时间反演迭代，切断了误差累积链条。它同时吸收 Bridge Matching（Peluchetti 2021; Liu et al. 2022b）的 Markov 投影回归工具，把 SB 求解降为每步一次 bridge-matching 回归。

在理论张力上，本文对应「边缘迭代（IPF/Sinkhorn）vs 过程结构迭代（Markov/reciprocal）」的对偶路线。Table 1 明确将 IMF 定位为 IPF 的对偶：IPF 保持 Markov 性和 $\mathcal{R}(Q)$ 但交替破坏两端边缘，IMF 保持两端边缘但交替破坏 Markov 性和 reciprocal 性。Proposition 10 建立了 DSBM-IPF 与 DSB 的等价性，说明 IMF 框架是 IPF 的推广而非完全替代。

被后续工作取代/发展的方向：α-DSBM（NeurIPS 2024）将 IMF 连续化为 SB Flow 实现在线单网络更新；ASBM（NeurIPS 2024）和 CSBM（ICML 2025）建立离散时间/离散空间的 D-IMF 理论；NeurIPS 2025 给出 IMF 的非渐近指数收敛率。本文的 DSBM-IMF 与 Rectified Flow（Liu et al. 2023b）的关系是 $\sigma \to 0$ 的确定性极限（Sec. 5），这为后续「SB 与 flow matching 谱线」的讨论提供了锚点。

## 6. 局限与批评

作者承认的局限（Sec. 7）：
1. DSBM 对一般传输问题最有效，在 CIFAR-10 生成建模上相比 Bridge/Flow Matching 只有微小改进（见 Appendix I.6）。
2. 虽然比 DSB 高效，但缓存步骤仍需从学习到的过程采样。
3. $\sigma$ 较小时 EOT 问题数值求解更困难。

读出来的局限：
1. 所有理论结果依赖「mild assumptions」但未精确列出，且 Theorem 8 的收敛是 KL 意义下的，不给出收敛速率；实际算法中 Markov 投影由有限样本回归近似，Proposition 7 的单调性在近似投影下不严格成立——作者在 Sec. 4 中承认「approximate minimization (10) may not admit $M^{n+1}_T = \pi_T$ exactly」，这正是需要交替前向/后向投影的原因。
2. DSBM-IMF+ 依赖 mini-batch EOT 求解器（Fatras et al. 2021; Flamary et al. 2021）提供初始化耦合，这引入了额外的计算依赖，且原文未报告该初始化本身的成本。
3. 实验中的 2D 结果（Table 2）显示 DSBM 在 moons-8gaussians 上 2-Wasserstein 为 0.812–0.838，远差于 OT-CFM 的 0.716，说明在低维、OT 求解器可用时 DSBM 并非最优选择；作者自己也承认 OT-CFM 在低维下表现最好（Sec. 6）。

## 7. 对我们的启发

1. **保边缘的交替投影可作为免训练 batch 级噪声指派的替代视角**：IMF 的核心卖点是每步保持两端边缘。若我们的 MPNA（免训练 batch 级保边缘噪声指派）需要在不训练网络的情况下构造近似 SB 耦合，可借鉴 reciprocal 投影的「端点重采样 + 参考桥插值」结构：用 mini-batch Sinkhorn 得到端点耦合后，直接采样 Brownian/OU 桥，无需学习漂移。这对应 DSBM-IMF+ 的初始化思路，但可去掉后续 Markov 投影迭代。
2. **概率流 ODE 提供 OT-aware 采样调度的现成公式**：平衡态下 $dZ^\star_t = \{f_t + \tfrac{1}{2}[v_{\theta^\star} - v_{\phi^\star}]\}dt$ 保持 SB 边缘。若我们要做 OT-aware 采样调度，可直接用 DSBM 学到的前向/后向漂移差构造 ODE，避免重新训练；注意原文明确警告该 ODE 的路径测度不是 $P^\star$，端点耦合不是 EOT 计划（Sec. 4），因此用于「保耦合蒸馏」时需额外惩罚耦合偏移。
3. **医学 SB 刷 SynthRAD 的可行性信号**：本文在非配对流体降尺度（64×64→512×512）上展示了 DSBM 的实用性，且仅需「slightly noising the low resolution input」（Sec. 6）。这提示 DSBM 可作为 SynthRAD 类医学图像超分辨/跨模态任务的基线：非配对设置、高维、需要保持输入一致性，与本文 AFHQ 和流体实验的设置高度吻合。但需注意作者承认 $\sigma$ 小时 EOT 数值困难，医学图像通常需要低噪声，需在 $\sigma$ 调度上做消融。

## 8. 资源

代码：https://github.com/yuyang-shi/dsbm-pytorch （原文 Sec. 1 脚注 2 给出）。

相关论文互链：
- De Bortoli et al. 2021, Diffusion Schrödinger Bridge with applications to score-based generative modeling（DSB，NeurIPS 2021，arXiv:2103.01360）
- Peluchetti 2023, Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling（IDBM，JMLR 2023，arXiv:2304.00917，同期独立工作）
- Liu et al. 2023b, Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow（arXiv:2209.03003，DSBM-IMF 的 $\sigma \to 0$ 极限）
- Tong et al. 2023, Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport（OT-CFM/SB-CFM，arXiv:2302.00482）
- Chen et al. 2022, Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory（SB-FBSDE，ICLR 2022，arXiv:2110.11291）
- Bunne et al. 2023, The Schrödinger Bridge between Gaussian Measures has a Closed Form（AISTATS 2023，arXiv:2302.05777，高斯闭式解用于 $d=50$ 实验）
