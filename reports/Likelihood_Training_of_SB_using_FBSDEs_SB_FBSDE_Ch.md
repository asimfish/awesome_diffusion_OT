# Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory

> Tianrong Chen, Guan-Horng Liu, Evangelos A. Theodorou · ICLR 2022 · [arXiv](https://arxiv.org/abs/2110.11291) · 证据级 [P] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：用 FBSDE 把 SB 的 PDE 最优性条件改写成可训练的精确对数似然，SGM 的 ELBO 是其 $Z_t\equiv0$ 特例。

## 1. 问题
SGM 的前向扩散必须是线性或退化漂移才能有解析条件 score，且要跑到足够长时间才近似高斯先验，导致采样慢、噪声调度 $g(t)$ 同时影响先验逼近与损失权重 $\lambda(t)$（Sec. 1, 2.1）。SB 允许在有限时间内、用可学习的非线性前向过程连接任意两分布，但 2021 年之前的 SB 生成模型要么做多段拼接（Wang et al. 2021），要么退回 IPF 的半桥交替优化（DSB、Vargas et al. 2021），都不是「对数似然训练」，与现代生成模型训练脱节；SB 的最优性由两条耦合 PDE（Eq.(6)）刻画，而 SGM 的似然目标是沿 SDE 的积分（Eq.(3)），两个数学对象看起来毫无关系。本文要回答：SB 的最优性条件能否被写成一个对数似然目标，并且严格包含 SGM 的目标？

## 2. 方法
起点是随机最优控制视角：SGM 与 SB 的前/后向 SDE 都属于控制仿射 SDE $dX_t=A\,dt+Bu\,dt+C\,dW_t$（Eq.(8)），区别只是把 score 还是 $\nabla\log\hat\Psi$ 当作控制。工具是非线性 Feynman–Kac（Lemma 2）：一类抛物 PDE 的解可沿前向 SDE 路径由一对前向-后向 SDE 表示。

**Theorem 3（SB 的 FBSDE 表示）**：SB 最优性 PDE（Eq.(6)，$\Psi,\hat\Psi$ 满足 $\Psi\hat\Psi|_0=p_{data}$、$\Psi\hat\Psi|_T=p_{prior}$）等价于耦合 SDE 组

$$dX_t=(f+gZ_t)dt+g\,dW_t,\quad dY_t=\tfrac12\|Z_t\|^2dt+Z_t^{\top}dW_t,\quad d\hat Y_t=\big(\tfrac12\|\hat Z_t\|^2+\nabla\!\cdot(g\hat Z_t-f)+\hat Z_t^{\top}Z_t\big)dt+\hat Z_t^{\top}dW_t\quad\text{(Eq.(13))}$$

其中 $Y_t=\log\Psi$、$Z_t=g\nabla\log\Psi$、$\hat Y_t=\log\hat\Psi$、$\hat Z_t=g\nabla\log\hat\Psi$（Eq.(14)），且 $Y_t+\hat Y_t=\log p^{SB}_t(X_t)$。$Z,\hat Z$ 就是 SB 的前向 / 后向策略。

**Theorem 4（SB 的对数似然）**：沿前向 SDE Eq.(13a) 从 $x_0$ 出发，

$$\log p^{SB}_0(x_0)=\mathbb E[\log p_T(X_T)]-\int_0^T\mathbb E\Big[\tfrac12\|Z_t\|^2+\tfrac12\|\hat Z_t\|^2+\nabla\!\cdot(g\hat Z_t-f)+\hat Z_t^{\top}Z_t\Big]dt\quad\text{(Eq.(16))}$$

用网络 $Z(\cdot;\theta),\hat Z(\cdot;\varphi)$ 替换后得下界 $\mathcal L_{SB}$。令 $(Z_t,\hat Z_t)=(0,g s_t)$ 即回到 SGM 的 Eq.(3)——这只在 $p^{(1)}_T=p_{prior}$ 时最优；否则前向策略 $Z_t$ 把过程「拉回」先验、后向策略补偿。Corollary 5 给 SB 概率流 ODE $dX_t=[f+gZ-\tfrac12g(Z+\hat Z)]dt$（Eq.(17)），用于算 NLL。

**训练（Algorithm 1–3）**：目标是散度型（Hutchinson 估计 $\nabla\cdot\hat Z$），不是 DSB / Vargas 的均值匹配回归。低维用联合训练（Alg. 2，保留整条轨迹计算图）；图像用交替训练（Alg. 3）：缓存前向轨迹训 $\hat Z$（Eq.(18)），再利用 SB 对称性（Theorem 11）缓存后向轨迹训 $Z$（Eq.(19)），每 ~1500 步刷新缓存，结构上对应 IPF。**采样（Algorithm 4）**：predictor 用 $\hat Z$ 做 Euler–Maruyama，corrector 用 $\nabla\log p^{SB}_t=(Z_t+\hat Z_t)/g$ 做 Langevin 校正（Eq.(20)(21)，SNR $r=0.05$，Eq.(59)）。骨干为 VE-SDE（$f=0$），$T=1$，CIFAR10 200 步、其他 100 步。

## 3. 理论结果
- **Lemma 2**（非线性 Feynman–Kac，引自 Exarchos & Theodorou 2018）：在 $f,G,h,\phi$ 连续、$f,G$ 对 $x$ 一致 Lipschitz、$h$ 对 $z$ 二次增长条件下，FBSDE Eq.(9) 与 PDE Eq.(10) 的解沿前向路径几乎必然重合。
- **Theorem 3**：假设 $\Psi,\hat\Psi\in C^{1,2}$，SB 最优性 PDE Eq.(6) 有 FBSDE 表示 Eq.(13)–(14)。
- **Theorem 4**：SB 模型在 $x_0$ 处的精确对数似然为 Eq.(15)=Eq.(16)。Eq.(15) 含不可算的 $\nabla\log p^{SB}_t$，Eq.(16) 消去它，只剩策略与散度项。
- **Corollary 5**：SB 的概率流 ODE，Remark 10 指出用 flow-based 方法算其似然恰好回到 Eq.(16)。
- **Theorem 11**（Appendix B）：对称地给出从 $x_T$ 出发的 $\mathcal L_{SB}(x_T)$，支撑交替训练。
- Appendix C 指出 DSB 的均值匹配回归目标（Eq.(55)）可视为 Eq.(15) 中 $\|Z_t+\hat Z_t-g\nabla\log p^{SB}_t\|^2$ 项的离散近似，且 DSB 用的经典 SB 模型 $\sqrt{2\gamma}dW_t$ 只有离散化后步长单调递增才能对上 SGM 的 $g(t)dW_t$。
- 无收敛性、无近似误差界；所有结果是恒等式层面的。

## 4. 实验与数字
数据集：GMM / checkerboard 玩具集、MNIST（padding 到 32×32）、CelebA（resize 到 32×32）、CIFAR10。网络：Toy FC-ResNet 0.76M×2；MNIST reduced UNet 1.95M×2；CelebA UNet 39.63M×2；CIFAR10 $Z$ 用 UNet 39.63M、$\hat Z$ 用 NCSN++ 62.69M（Table 4）。超参：lr 2e-4（CIFAR10 1e-5），时间步 100（CIFAR10 200），batch 400/200/200/64，$p_{prior}$ 方差 1/1/900/2500（Table 3），EMA 0.99。

| 设置 | 指标 | SB-FBSDE | 对照 | 来源 |
|---|---|---|---|---|
| CIFAR10 | NLL ↓ (bits/dim) | 2.96 | SDE deep sub-VP 2.99；ScoreFlow 2.74；VDM 2.49；LSGM 3.43 | Table 1 |
| CIFAR10 | FID ↓（50k 样本 vs 训练集） | 3.01 | DOT 15.78；Multi-stage SB 12.32；DGflow 9.63；SDE sub-VP 2.92；ScoreFlow 5.7；VDM 4.00；LSGM 2.10 | Table 1 |
| CIFAR10 | 同上（正文数字） | NLL 2.98、FID 3.18 | — | Sec. 4 正文（与 Table 1 不一致，见 §6） |
| MNIST 前向过程 | $\mathrm{KL}(p_T\|p_{prior})$ 随训练阶段 | 线性与退化漂移下均随 stage 下降，远低于 SGM 固定值 | SGM 值不变 | Figure 5 |
| CelebA / CIFAR10 | FID vs 有无 Langevin corrector | 加 corrector 在所有训练阶段一致降低 FID | — | Figure 6 |
| CIFAR10 效率 | 训练时间 / 采样时间 / 显存 | 训练 +6.8%、采样 −80%、显存 2–2.5× | 相对 SGM | Sec. 3.3 |
| MNIST / CelebA | 定量 | 作者未报告（理由：预处理不可比） | — | Sec. 4 |

## 5. 在 OT×扩散地图中的位置
第一代「深度 IPF 与 likelihood」的第二根支柱（另一根是 DSB）。它与 DSB 的分工：DSB 把 IPF 半步实现为均值匹配回归，SB-FBSDE 把 SB 最优性实现为散度型似然目标，并首次在连续时间证明「SGM 训练目标 = SB 目标的退化特例」，使 SB 可以借用 Langevin corrector 等 SGM 技巧。它是 T03 中 GSBM（Liu et al. 2024，同组，把 FBSDE/随机控制推广到带 state cost 的广义 SB）、DMSB / 3MSBM（Chen、Theodoropoulos，相空间多边缘）以及 Reflected SB（Deng et al. 2024，`2607.03626` 批评其需要高阶导数与全路径采样）的方法学源头。被第二代 IMF/matching 系在效率上取代：LightSB-M 的 Sec. 4.1 把它归为「迭代反转 Markov 过程、会发散或遗忘先验」的 IPF 系。对应综合报告张力：「学习前向过程 vs 固定噪声调度」——Figure 5 是「可学习前向扩散比手工 $g(t)$ 更接近先验」的最早直接证据。

## 6. 局限与批评
作者承认（Sec. 3.3）：散度计算与维护两套网络使显存增至 SGM 的 2–2.5 倍；交替训练需要缓存轨迹。

我读出来的：(1) CIFAR10 数字自相矛盾——Table 1 写 NLL 2.96 / FID 3.01，正文写 2.98 / 3.18，且 FID 是相对训练集算的，与多数文献相对测试集的做法不同，横向比较要打折；(2) 「outperforms prior OT baselines by a large margin」成立，但相对 SGM 只是「comparable」（FID 3.01 vs sub-VP 2.92、LSGM 2.10）；(3) 交替训练（Alg. 3）本质仍是 IPF，只是目标换成散度型，DSBM 后来指出的误差累积与遗忘问题同样适用，论文没有讨论；(4) 图像实验依赖 Langevin corrector（Figure 6），而 corrector 用到的 $\nabla\log p^{SB}_t=(Z+\hat Z)/g$ 只在两策略都收敛到最优时成立，训练早期用它做校正缺乏理由；(5) MNIST / CelebA 没有任何定量结果。

## 7. 对我们的启发
1. **#2 OT-aware 调度**：Theorem 4 把「前向调度选得不好」量化成前向策略 $Z_t$ 的非零能量 $\tfrac12\|Z_t\|^2$；可用一个轻量 $Z$ 网络的收敛能量作为「当前噪声调度离 SB 最优有多远」的诊断量，不必重训生成器。
2. **保耦合蒸馏 / 似然评估**：Corollary 5 的概率流 ODE 给 SB 模型精确 NLL，是评估「蒸馏后是否仍是同一个 SB」的可计算指标之一（与 DSBM 的概率流 ODE 一致）。
3. **#7 医学 SB**：Eq.(16) 是散度型目标，不依赖参考桥的闭式（不像 Brownian bridge matching），适合非线性、物理先验驱动的参考过程（如器官形变模型），代价是 Hutchinson 估计与双网络。

## 8. 资源
- 代码：https://github.com/ghliu/SB-FBSDE
- 相关报告：`Diffusion_Schr_dinger_Bridge_DSB_De_Bortoli_et_al`（同期竞争）、`Generalized_SB_Matching_GSBM_Liu_et_al`、`Deep_Momentum_Multi_Marginal_SB_DMSB_Chen_et_al`、`Momentum_Multi_Marginal_SB_Matching_3MSBM_Theodoro`（同组后续）、`Diffusion_Schr_dinger_Bridge_Matching_DSBM_Shi_et`（取代者）、`2607.03626`（对 FBSDE 路线的批评）、`2005.10963`（随机控制视角来源）
