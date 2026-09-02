# Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling (DSB)

> Valentin De Bortoli, James Thornton, Jeremy Heng et al. · NeurIPS 2021 (Spotlight) · [proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html) · 证据级 [P] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：用 score 网络实现 IPF 的每个半步（学时间反转），首次把 SB 求解扩展到图像；SGM 是其第一次迭代。
> ⚠ 未读全文，依据摘要（清单一句话贡献、KB 笔记，以及 `Diffusion_Schr_dinger_Bridge_Matching_DSBM_Shi_et`、`Likelihood_Training_of_SB_using_FBSDEs_SB_FBSDE_Ch`、`2409.09376` 对本文的转述与基线数字）。

## 1. 问题
SGM 依赖前向扩散在无限时间趋于高斯先验，采样步数多；SB 能在有限时间内连接任意两分布，但 2021 年前的 SB 数值方法限于低维（DSBM Sec. 1 的综述）。本文要用深度网络在高维上求 SB。

## 2. 方法
据 DSBM Sec. 2.2 转述：IPF 序列 $\tilde P^{2n+1}=\arg\min\{\mathrm{KL}(\tilde P|\tilde P^{2n}):\tilde P_T=\pi_T\}$、$\tilde P^{2n+2}=\arg\min\{\mathrm{KL}(\tilde P|\tilde P^{2n+1}):\tilde P_0=\pi_0\}$，初始 $\tilde P^0=Q$；每个迭代是上一迭代的时间反转（以另一端边缘初始化），DSB 用网络逐轮学习该时间反转，DDM 恰为第一次迭代。据 SB-FBSDE Appendix C 转述：训练目标是均值匹配回归 Eq.(55)——$B_{k+1}\leftarrow\arg\min\mathbb E\|B_{k+1}(X_{k+1})-(X_{k+1}+F_k(X_k)-F_k(X_{k+1}))\|^2$，SDE 模型为经典 $dX_t=f\,dt+\sqrt{2\gamma}dW_t$，Euler–Maruyama 离散步长 $\gamma_k$。需要缓存整条轨迹（DSBM Sec. 4 脚注 5 与 Sec. 4 末）。

## 3. 理论结果
据 `2409.09376` Sec. 3.1 转述：De Bortoli et al. 2021 的 Propositions 4、5 与 Section 3.5 建立了 DIPF 迭代的收敛性质；DSBM Proposition 10 证明以 $Q_{0,T}$ 初始化的 DSBM 与 DSB 的 IPF 迭代逐轮相同。具体定理陈述未读全文，未见。

## 4. 实验与数字
本文自身数字未见。作为基线出现的数字：DSBM Table 2（2D，$W_2$）：DSB moons 0.190±0.049、scurve 0.272±0.065、8gaussians 0.411±0.084、moons-8gaussians 0.987±0.324（均劣于 DSBM-IPF 0.140/0.140/0.315/0.812）；DSBM Table 3（高斯 d=5/20/50，KL×10⁻³）：DSB 3.26/13.0/32.8 vs DSBM-IPF 1.23/4.42/8.75；DSBM Figure 3：DSB 的 SB 协方差估计随迭代恶化；DSBM 称其运行时间比 DSB 省约 30%。`2409.09376` Table 1（EOT benchmark，$\mathrm{KL}(S\|P)$，$\varepsilon=0.1$，d=2/16/64/128）：DIPF 0.59/2.39/7.93/34.77 vs I-BM 0.03/0.20/1.24/5.70。

## 5. 在 OT×扩散地图中的位置
T03 第一代的起点，确立「SB = 有限时间扩散生成 + 连续状态 Sinkhorn」叙事；与 SB-FBSDE（散度型似然目标）并列为深度 IPF 的两种实现。被第二代 IMF/DSBM 取代：后者指出 DSB 的三个缺陷——误差逐轮累积（Fernandes 2021）、需整条轨迹缓存、实践中「遗忘」参考桥 $Q_{|0,T}$（DSBM Sec. 4）。SB-FBSDE Appendix C 另指出其 $\sqrt{2\gamma}$ 常扩散模型只有离散化后步长单调递增才对应 SGM 的 $g(t)$。

## 6. 局限与批评
后续论文指出（非作者自述，作者自述未见）：(1) IPF 每半步只保一端边缘（DSBM Table 1）；(2) 时间反转需整条轨迹，内存与计算重；(3) 近似误差逐轮累积，高维 d=50 高斯上协方差随迭代漂移（DSBM Figure 3）；(4) 「遗忘」先验（Vargas & Nüsken 2023，LightSB-M Sec. 4.1 转引）。

## 7. 对我们的启发
1. DSB 的「SGM = 第一次 IPF 迭代」视角是解释「为什么预训练扩散模型只是 SB 的一个粗糙起点」的标准说法，可用于综合报告。
2. 其误差累积案例（DSBM Figure 3）是评估任何迭代式对齐/蒸馏方案时应复现的诊断实验。

## 8. 资源
- 代码：未见（未读全文）
- 相关报告：`Diffusion_Schr_dinger_Bridge_Matching_DSBM_Shi_et`（取代者）、`Likelihood_Training_of_SB_using_FBSDEs_SB_FBSDE_Ch`（同期竞争）、`2409.09376`（DIPF 基线）、`2005.10963`（IPF/Fortet 理论来源）
