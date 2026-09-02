# Light and Optimal Schrödinger Bridge Matching

> Nikita Gushchin, Sergei Kholkin, Evgeny Burnaev et al. · ICML 2024 (PMLR v235) · [PMLR](https://proceedings.mlr.press/v235/gushchin24a.html) · 证据级 [P] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：证明对任意耦合做一次「最优投影」即得 SB；高斯混合势参数化使其在 4 核 CPU 上分钟级求解。

## 1. 问题
Bridge matching 系 SB 求解器的瓶颈是「先要有 EOT 耦合」。两条现有路线各有硬伤（Sec. 1, Sec. 2.3）：(a) OT-CFM / [SF]²M 用 minibatch 离散 OT 近似 EOT 耦合，经验耦合相对真耦合有偏，偏差直接进入 SB 解；(b) DSBM 用 IMF 交替 Markov 投影与 reciprocal 投影，理论上从任意耦合收敛到 SB，但每次 Markov 投影都由神经网络回归实现，误差逐轮累积，导致 $t=1$ 边缘偏离 $p_1$。IPF 系（DSB、SB-FBSDE 等）同样是迭代反转 Markov 过程，会发散或「忘掉」Wiener 先验（Sec. 4.1 引 Vargas & Nüsken 2023）。

本文的目标：找一种投影，使**任意**输入耦合 $\pi\in\Pi(p_0,p_1)$ 经**一次** matching 回归就落在 SB 上，既不依赖 minibatch 近似也不迭代。

## 2. 方法
核心思想：把投影目标集从「Markov 过程」换成「SB 集合」。定义 $\mathcal S$ 为所有 SB 的集合（对某对边缘 $p_0^S,p_1^S$ 是 $\mathrm{KL}(\cdot\|W^\epsilon)$ 极小者，Eq.(12)），对 reciprocal 过程 $T_\pi=\int W^\epsilon_{|x_0,x_1}d\pi$ 定义最优投影（optimal projection, OP）

$$\mathrm{proj}_{\mathcal S}(T_\pi)=\arg\min_{S\in\mathcal S}\mathrm{KL}(T_\pi\|S)\quad\text{(Eq.(13))}$$

Theorem 3.1 说这一步投影直接等于 $p_0,p_1$ 之间的 SB $T^*$，与 $\pi$ 无关。Theorem 3.2 给可计算目标：对由 adjusted Schrödinger 势 $v$ 决定的 SB $S_v$（起点 $p_0$），

$$\mathrm{KL}(T_\pi\|S_v)=C(\pi)+\frac{1}{2\epsilon}\int_0^1\!\!\int\Big\|g_v(x_t,t)-\frac{x_1-x_t}{1-t}\Big\|^2dp_{T_\pi}(x_t,x_1)dt\quad\text{(Eq.(15))}$$

形式上就是 DSBM 的 bridge-matching 回归损失 Eq.(10)，区别只在回归函数 $g_v$ 被限制为「由势 $v$ 通过 Eq.(4) 生成的 SB 漂移」，而不是自由神经网络。

参数化沿用 LightSB：$v_\theta(x)=\sum_{k=1}^K\alpha_k\mathcal N(x|\mu_k,\epsilon\Sigma_k)$（Eq.(7)），漂移 $g_\theta$ 有闭式 Eq.(16)。Algorithm 1（LightSB-M）：采 $(x_0,x_1)\sim\pi$（独立耦合、minibatch OT 均可）、$t\sim U[0,1]$、$x_t\sim\mathcal N(tx_1+(1-t)x_0,\epsilon t(1-t)I)$，最小化 $\|g_\theta(x_t,t)-(x_1-x_t)/(1-t)\|^2$，SGD 更新 $\theta$。推断有两种：Euler–Maruyama 解学到的 SDE，或直接从闭式条件耦合 $\pi_\theta(x_1|x_0)$（Eq.(6)）采 $x_1$、再用 Brownian bridge 自相似性补轨迹，完全不解 SDE（Sec. 3.2）。

Theorem 3.3：OP 目标 Eq.(15) 等于 LightSB/EgNOT 的 KL 目标 $\mathcal L_0(v)$（Eq.(5)）加常数——matching 目标与 EBM 目标在此统一，LightSB-M 因而继承 LightSB 的逼近与泛化理论。Appendix C 另给出用神经网络参数化势 $\varphi$ 的 HardSB-M 变体（MC 估计漂移），只作方向性展示。

## 3. 理论结果
- **Theorem 3.1（reciprocal 过程的 OP）**：假设 Wiener 先验 $dW^\epsilon_t=\sqrt\epsilon dW_t$，$p_0,p_1\in\mathcal P(\mathbb R^D)$ 绝对连续且方差、微分熵有限，$\pi\in\Pi(p_0,p_1)$ 任意。结论：$\arg\min_{S\in\mathcal S}\mathrm{KL}(T_\pi\|S)=T^*$（Eq.(14)）。含义：解不依赖输入耦合。
- **Theorem 3.2（OP 的可计算目标）**：对 $S_v\in\mathcal S(p_0)$，Eq.(15) 成立，$C(\pi)$ 与 $v$ 无关。
- **Theorem 3.3（与 EgNOT/LightSB 目标等价）**：Eq.(15) 左端 $=\tilde C(\pi)+\mathcal L_0(v)$。
- 证明在 Appendix A（本文未逐行核对）。理论只覆盖精确优化 $\mathcal S$ 的情形；有限 $K$ 的高斯混合只是 $\mathcal S$ 的子集，论文未给有限 $K$ 下的误差界。

## 4. 实验与数字
数据集：EOT/SB mixtures benchmark（Gushchin et al. 2023b；$D\in\{2,16,64,128\}$，$\epsilon\in\{0.1,1,10\}$，有真 EOT 耦合）；MSCI 单细胞（Kaggle "Open Problems – Multimodal Single-Cell Integration"，PCA 到 50/100/1000 维，day2→4 评 day3、day3→7 评 day4）；FFHQ 1024×1024 在 ALAE 512 维 latent 上做 unpaired 翻译。基线：LightSB、DSBM、SF2M-Sink（DSBM 用 10 轮 IMF × 10000 步、MLP+位置编码，Appendix B.5）。

| 设置 | 指标 | LightSB-M (ID) | LightSB-M (MB) | DSBM | SF2M-Sink | LightSB | 来源 |
|---|---|---|---|---|---|---|---|
| benchmark $\epsilon=0.1$, D=2/16/64/128 | cBW²₂-UVP ↓ (%) | 0.04/0.18/0.77/1.66 | 0.02/0.1/0.56/1.32 | 5.2/16.8/37.3/35 | 0.54/3.7/9.5/10.9 | 0.03/0.08/0.28/0.60 | Table 1 |
| benchmark $\epsilon=1$ | 同上 | 0.09/0.18/0.47/1.2 | 0.09/0.18/0.46/1.2 | 0.3/1.1/9.7/31 | 0.2/1.1/9/23 | 0.05/0.09/0.24/0.62 | Table 1 |
| benchmark $\epsilon=10$ | 同上 | 0.12/0.19/0.36/0.71 | 0.13/0.18/0.36/0.71 | 3.7/105/3557/15000 | 0.31/4.9/319/819 | 0.07/0.11/0.21/0.37 | Table 1 |
| benchmark（GT 耦合作输入） | 同上，$\epsilon=0.1$ | 0.02/0.1/0.49/1.16 | — | — | — | — | Table 1 |
| MSCI, DIM=50/100/1000 | energy distance ↓ | 2.347±0.11 / 2.174±0.08 / 1.35±0.05 | 2.33±0.09 / 2.172±0.08 / 1.33±0.05 | 2.46±0.1 / 2.35±0.1 / 1.36±0.04 | 2.66±0.18 / 2.52±0.17 / 1.38±0.05 | 2.31±0.27 / 2.16±0.26 / 1.27±0.19 | Table 2 |
| MSCI 训练时间 | 50/100/1000 维 | 58 s / 60 s / 147 s（4 CPU 核） | 80/80/176 s（4 CPU 核） | 6.6/6.6/8.9 min（V100） | 8.4/8.4/13.8 min（V100） | 65/66/146 s（4 CPU 核） | Table 2 |
| benchmark 动态 KL | $\mathrm{KL}(T^*\|S)$ / $\mathrm{KL}(S\|T^*)$ | 0.0093 / 0.0099 | — | 0.2950 / 0.39 | 0.6422 / 1.0765 | — | Table 5 |
| FFHQ man→woman（ALAE latent, $\epsilon=0.1$） | FID ↓ | 0.852 | 0.859 | 0.859 | 0.8613 | — | Table 8 |

采样效率（benchmark $\epsilon=0.1$, D=16，Table 6/7）：直接从 $\pi_\theta(x_1|x_0)$ 采样 0.00058 s、cBW²₂-UVP 0.09；Euler–Maruyama 需约 500 步才到 0.09，1000 步耗时 12.61 s，10 步只有 1.53。作者的两点观察（Sec. 5.2）：ID / MB / GT 三种输入耦合得到几乎相同的解，印证 Theorem 3.1；DSBM 增加 IMF 轮数并未改善质量，且 DSBM 与 SF2M-Sink 在 $\epsilon=10$ 时都因需学高幅值 SDE 而失效。

## 5. 在 OT×扩散地图中的位置
处于 SB 五代史的第三代「轻量化」：它把 LightSB（ICLR 2024，KL 目标 + 高斯混合势）重新解释成 bridge matching，并用 Theorem 3.3 把 EBM 路线（EgNOT）与 matching 路线（DSBM/[SF]²M）接到同一目标上。与 DSBM 的关系是**竞争 + 理论对照**：DSBM 交替投影到「Markov 集」和「reciprocal 集」，本文一步投影到二者的交集 $\mathcal S$。与 [SF]²M 的关系是竞争：[SF]²M 的正确性依赖输入耦合是 EOT 耦合，本文对任何耦合都成立。后续被 arXiv 2510.22560 纳入「Sinkhorn bridge」统计分析家族（LightSB(-M) 的估计量与 [SF]²M/DSBM-IMF/BM² 在特定设定下一致）；同组的 ASBM、CSBM 走另一条 D-IMF 路线。对应综合报告的张力「一次投影 vs 迭代投影」「参数化表达力 vs 免仿真」；在推理管线里占「从耦合恢复动态过程」这一环。

## 6. 局限与批评
作者承认：(1) 高斯混合参数化对大规模生成任务可能不够，图像实验只能在 ALAE latent 上做（Sec. 6, 5.4）；(2) 只处理 Wiener 先验，其他先验需变量替换（Sec. 6）；(3) 混合参数化与 benchmark 构造原理同源，可能带来 inductive bias（Sec. 5.2）。

我读出来的：(1) Table 1 中 LightSB 在几乎所有格子都优于 LightSB-M（如 $\epsilon=0.1$, D=128：0.60 vs 1.66），而 Theorem 3.3 说两者目标等价——LightSB-M 的实际增益是「统一叙事 + 可接任意耦合」，不是精度；(2) 图像域 FID 四种方法都在 0.852–0.8613，差异在噪声内，不能据此说图像翻译上有优势；(3) 「一步投影、与 $\pi$ 无关」只在精确优化 $\mathcal S$ 时成立，有限 $K$ 混合下的投影集是 $\mathcal S$ 的真子集，论文未给近似误差界；(4) DSBM 基线只用 10 轮 IMF 与 MLP，$\epsilon=10$ 时的 15000% 更像调参失败而非方法极限。

## 7. 对我们的启发
1. **#1 保边缘噪声指派 MPNA**：LightSB-M 在 latent 空间几分钟就能给出闭式 EOT 条件耦合 $\pi_\theta(x_1|x_0)$，可作为「参考耦合 oracle」量化任一噪声指派与 EOT 耦合的偏离（用 Table 5 那套动态 KL 度量）。
2. **#3 保耦合蒸馏**：Theorem 3.1 表明任意耦合投影到 $\mathcal S$ 都回到同一 SB，可用作蒸馏后少步模型的 latent 空间「耦合修正」后处理，不需重训 teacher。
3. **KB 开放问题 2（Light-3MSB）**：作者把「更一般先验的 light 求解器」列为未来方向，把 Eq.(16) 的闭式推广到 OU / 相空间参考过程，是本课题可直接接手的空白。

## 8. 资源
- 代码：https://github.com/SKholkin/LightSB-Matching（PyTorch，每个实验一份 notebook）
- 相关报告：`Light_Schr_dinger_Bridge_LightSB_Korotin_et_al`（前身）、`Diffusion_Schr_dinger_Bridge_Matching_DSBM_Shi_et`（竞争）、`Simulation_free_Score_Flow_Matching_SF_M_Tong_et_a`（竞争）、`2510.22560`（统计分析涵盖本方法）、`2409.09376`（另一非迭代路线）、`Adversarial_SB_Matching_ASBM_Gushchin_et_al`、`Categorical_SB_Matching_CSBM_Ksenofontov_Korotin`（同组后续）
