# I2SB: Image-to-Image Schrödinger Bridge

> Guan-Horng Liu et al. · ICML 2023 · [arXiv](https://arxiv.org/abs/2302.05872) · 证据级 [P] · 课题 T14 扩散桥 / Schrödinger 桥的图像到图像翻译
> **一句话**：把 SB 边界一侧设为 Dirac delta，得到边缘可解析的 tractable SB，simulation-free 训练，ImageNet-256 修复任务超越条件扩散、媲美已知退化算子的逆方法。

## 1. 问题

图像修复（inpainting、超分、去模糊、JPEG 修复）是病态逆问题，现代数据驱动方法通常用条件生成来学「给定退化分布采样干净分布」。此前的扩散/分数生成模型（SGM）做修复时，生成过程一律从高斯白噪声出发（Sec. 1, p.2），噪声几乎不含干净数据分布的结构信息；而退化图像本身比随机噪声结构信息丰富得多，从噪声出发是否适合 I2I 翻译并不清楚。

Schrödinger Bridge（SB）理论上把生成过程推广为两个任意分布之间的熵正则最优传输，可以起点设在退化图像而非高斯。但此前的 SB 数值求解框架（如迭代投影法，Chen et al. 2021b）与 SGM 训练框架独立发展，在高维下计算不利：256×256 分辨率上 SB 比 SGM 慢 6×、内存多 3×（Figure 2, p.2），且存在离散化误差、高方差甚至发散问题（Sec. 1, p.2）。因此问题是：能否构造一个 SB 子类，既保留「从退化图像出发」的结构优势，又能复用 SGM 的 scalable 训练框架。

## 2. 方法

核心思想：把 SB 的一侧边界设为 Dirac delta 分布 $p_A(\cdot)=\delta_a(\cdot)$，即每个干净图像 $a$ 作为一个锚点。此时 SB 的耦合约束被打破，反向漂移 $\nabla\log\hat\Psi$ 变成另一个线性 SDE 的 score function，可以用标准 denoising score matching 训练（Corollary 3.2, p.4）。作者称这类为 tractable SB，它把 SGM 的终端分布从高斯推广到任意 $p_B(\cdot|X_0)$，同时保持 SGM 的计算框架（Figure 3, p.2）。

关键公式：

**（1）tractable SB 的边界条件**（Corollary 3.2, Eq.(10), p.4）：
$$\hat\Psi(\cdot,0)=\delta_a(\cdot),\qquad \Psi(\cdot,1)=\frac{p_B}{\hat\Psi(\cdot,1)}$$
当 $p_A$ 是 Dirac delta 时，$\hat\Psi(x,0)$ 不再依赖 $\Psi$，耦合被解除。

**（2）给定边界对的解析后验**（Proposition 3.3, Eq.(11), p.5）：
$$q(X_t|X_0,X_1)=\mathcal{N}(X_t;\mu_t(X_0,X_1),\Sigma_t),\quad \mu_t=\frac{\bar\sigma_t^2}{\bar\sigma_t^2+\sigma_t^2}X_0+\frac{\sigma_t^2}{\bar\sigma_t^2+\sigma_t^2}X_1,\quad \Sigma_t=\frac{\sigma_t^2\bar\sigma_t^2}{\bar\sigma_t^2+\sigma_t^2}\cdot I$$
其中 $\sigma_t^2:=\int_0^t\beta_\tau d\tau$，$\bar\sigma_t^2:=\int_t^1\beta_\tau d\tau$。训练时直接从该高斯采样 $X_t$，无需解非线性扩散；生成时从 $X_1\sim p_B$ 出发跑 DDPM，只要预测的 $X_0^\epsilon$ 接近真实 $X_0$，就诱导相同的边际密度。

**（3）训练目标**（Eq.(12), p.5）：
$$\|\epsilon(X_t,t;\theta)-\frac{X_t-X_0}{\sigma_t}\|$$
设 $f:=0$，网络参数化与 SGM 相同（U-Net，ADM 初始化），无需条件模块。

训练流程（Algorithm 1, p.5）：采样 $t\sim U([0,1])$、$X_0\sim p_A$、$X_1\sim p_B(\cdot|X_0)$，按 Eq.(11) 采样 $X_t$，对 Eq.(12) 做梯度下降。生成流程（Algorithm 2, p.5）：从 $X_N\sim p_B$ 出发，用网络预测 $X_0^\epsilon$，按 DDPM 递归后验采样 $X_{n-1}\sim p(X_{n-1}|X_0^\epsilon,X_n)$。默认 1000 步、quadratic discretization（Sec. 5.1, p.6），噪声调度为两端收缩的对称 $\beta_t$（Figure 6, p.6）。

## 3. 理论结果

**Theorem 3.1**（p.4）：当 Schrödinger 系统 Eq.(6) 成立时，$\nabla\log\hat\Psi(X_t,t)$ 和 $\nabla\log\Psi(X_t,t)$ 分别是两个线性 SDE（Eq.(9a)、(9b)）的 score function。即 SB 的非线性漂移被吸收进线性 SDE 的初始条件 $\hat\Psi(\cdot,0)$ 中。

**Corollary 3.2**（p.4）：当 $p_A=\delta_a$ 时，$\hat\Psi(\cdot,0)=\delta_a$，$\Psi(\cdot,1)=p_B/\hat\Psi(\cdot,1)$，解除了 $\Psi$ 对 $\hat\Psi(x,0)$ 的依赖。作者指出 Dirac delta 假设也隐含出现在 SGM 的 denoising objective Eq.(3) 中——先对每个数据点 $a$ 算 $\nabla\log p(X_t,t|X_0=a)$，再对 $X_0\sim p_A$ 平均。当 $p_B=\hat\Psi(\cdot,1)\approx\mathcal{N}(0,I)$ 时，前向漂移消失（$\Psi(\cdot,t)=1$），框架退化为 SGM（p.4）。

**Proposition 3.3**（p.5）：$f:=0$ 时，SB 给定边界对的后验有解析高斯形式 Eq.(11)，且该后验等于 DDPM 递归后验采样 $q(X_n|X_0,X_N)=\int\prod_{k=n}^{N-1}p(X_k|X_0,X_{k+1})dX_{k+1}$ 的边际。

**Proposition 3.4**（p.5）：当 $\beta_t\to 0$，SDE 退化为 ODE $dX_t=v_t(X_t|X_0)dt$，$v_t(X_t|X_0)=\frac{\beta_t}{\sigma_t^2}(X_t-X_0)$，其解是 Eq.(11) 的后验均值。该 OT-ODE 不是 SGM 文献中的 probability flow ODE，只在随机性消失时模拟 OT plan。

**Corollary 3.5**（p.5）：当 $\beta_t:=\beta$ 足够小且恒定，$v_t=\frac{X_t-X_0}{t}$，$\mu_t=(1-t)X_0+tX_1$，恢复 OT displacement（McCann, 1997）。

## 4. 实验与数字

数据集：ImageNet 256×256；超分报告全验证集，其余任务报告 10k 验证子集（Sec. 5.1, p.6）。指标：FID 和预训练 ResNet50 的 Classifier Accuracy（CA）。基线：CSGM 类（Palette、ADM）、DIM 类（DDRM、DDNM、ΠGDM，需已知退化算子）、SB 类（CDSB）。

**Table 2（p.7）4× 超分**：

| Filter | Method | FID↓ | CA↑ |
|---|---|---|---|
| Pool | DDRM | 14.8 | 64.6 |
| Pool | DDNM | 9.9 | 67.1 |
| Pool | ΠGDM | 3.8 | 72.3 |
| Pool | ADM | 3.1 | 73.4 |
| Pool | CDSB | 13.0 | 61.3 |
| Pool | **I2SB** | **2.7** | 71.0 |
| Bicubic | DDRM | 21.3 | 63.2 |
| Bicubic | DDNM | 13.6 | 65.5 |
| Bicubic | ΠGDM | 3.6 | 72.1 |
| Bicubic | ADM | 14.8 | 66.7 |
| Bicubic | CDSB | 13.6 | 61.0 |
| Bicubic | **I2SB** | **2.8** | 70.7 |

**Table 3（p.7）JPEG 修复**：

| QF | Method | FID-10k↓ | CA↑ |
|---|---|---|---|
| 5 | DDRM | 28.2 | 53.9 |
| 5 | ΠGDM | 8.6 | 64.1 |
| 5 | Palette | 8.3 | 64.2 |
| 5 | CDSB | 38.7 | 45.7 |
| 5 | **I2SB** | **4.6** | **67.9** |
| 10 | DDRM | 16.7 | 64.7 |
| 10 | ΠGDM | 6.0 | 71.0 |
| 10 | Palette | 5.4 | 70.7 |
| 10 | CDSB | 18.6 | 60.0 |
| 10 | **I2SB** | **3.6** | **72.1** |

**Table 4（p.7）Inpainting**：

| Mask | Method | FID-10k↓ | CA↑ |
|---|---|---|---|
| Center 128×128 | DDRM | 24.4 | 62.1 |
| Center 128×128 | ΠGDM | 7.3 | 72.6 |
| Center 128×128 | DDNM | 15.1 | 55.9 |
| Center 128×128 | Palette | 6.1 | 63.0 |
| Center 128×128 | CDSB | 50.5 | 49.6 |
| Center 128×128 | **I2SB** | **4.9** | 66.1 |
| Freeform 10%-20% | DDRM | 9.7 | 67.6 |
| Freeform 10%-20% | DDNM | 3.2 | 73.6 |
| Freeform 10%-20% | Palette | 4.0 | 73.7 |
| Freeform 10%-20% | CDSB | 8.5 | 71.2 |
| Freeform 10%-20% | **I2SB** | **2.9** | **74.9** |
| Freeform 20%-30% | DDRM | 8.6 | 71.9 |
| Freeform 20%-30% | ΠGDM | 5.3 | 75.3 |
| Freeform 20%-30% | DDNM | 4.2 | 70.8 |
| Freeform 20%-30% | Palette | 4.1 | 71.8 |
| Freeform 20%-30% | CDSB | 16.5 | 64.5 |
| Freeform 20%-30% | **I2SB** | **3.2** | 73.4 |

**Table 5（p.7）去模糊**：

| Kernel | Method | FID-10k↓ | CA↑ |
|---|---|---|---|
| Uniform | DDRM | 9.9 | 68.0 |
| Uniform | DDNM | 3.0 | 75.5 |
| Uniform | Palette | 4.1 | 74.0 |
| Uniform | CDSB | 15.5 | 65.1 |
| Uniform | **I2SB** | 3.9 | 73.7 |
| Gaussian | DDRM | 6.1 | 72.5 |
| Gaussian | DDNM | 2.9 | 75.6 |
| Gaussian | Palette | 3.1 | 75.4 |
| Gaussian | CDSB | 7.7 | 71.1 |
| Gaussian | **I2SB** | 3.0 | 75.0 |

关键结论（Sec. 5.2, p.7）：I2SB 在 9 个任务中的 6 个上超越标准 CSGM（Palette、ADM）；在 9 个任务中的 7 个上取得 SOTA FID；在 JPEG 修复（两个 QF）和 inpainting（Freeform 10-20%）上刷新 CA 记录；在所有任务上大幅超越 CDSB。I2SB 不需要知道退化算子，却匹配了 DIM 的性能。

**采样效率**（Figure 8, p.8）：在 inpainting（Freeform 20-30%）上，I2SB 只需 2~10 NFE 即可达到 Palette 至少 100 NFE 才能达到的相近最佳性能。Figure 9（p.8）显示 I2SB 在 NFE=2 时就能为 inpainting 补出语义结构，Palette 在小 NFE 下产生噪声补绘或对比度偏移。

**OT-ODE 消融**（Table 6, p.8）：用后验均值替代随机采样（即 OT-ODE）时，JPEG 修复 QF=5 的 FID 差 +5.3、CA 差 -4.7；QF=10 的 FID 差 +4.2、CA 差 -3.8；去模糊 Uniform 的 FID 差 -0.3、CA 差 +6.0；Gaussian 的 FID 差 -0.6、CA 差 +4.1。即 OT-ODE 偏向确定性映射可行的任务（去模糊），对不确定性大的任务（JPEG）不利。

**通用 I2I 翻译**（Table 7, p.9）：edges→shoes 上 Pix2pix FID 73.9，I2SB NFE=1 为 73.9、NFE=5 为 54.2、NFE=1000 为 37.8；day→night 上 Pix2pix 196.4，I2SB NFE=1 为 196.3、NFE=5 为 185.8、NFE=1000 为 153.6。

**与 inpainting GAN 对比**（Table 8, p.9）：Freeform 10%-20% 上 DeepFillv2 FID 6.7、HiFill 7.5、I2SB(NFE=1) 4.1、Palette(NFE=1) 9.6；Freeform 20%-30% 上 DeepFillv2 9.4、HiFill 12.4、I2SB(NFE=1) 6.7、Palette(NFE=1) 19.8。推理时间（V100 16G）：DeepFillv2 0.01 s/image、HiFill 0.03、I2SB 0.14、Palette 0.14。

## 5. 在 OT×扩散地图中的位置

I2SB 是「扩散桥 × 图像翻译」方向中「配对桥」路线的奠基工作之一，与 BBDM（latent Brownian bridge）同期把 I2I 从条件生成范式推向桥范式。它抓住的关键理论张力是：SB 的耦合约束（Eq.(6b)）导致数值求解与 SGM 框架脱节，而 Dirac delta 边界假设把一侧耦合解除，使 SB 退化为「每个锚点的条件 score matching」——这正是后来 DDBM 等统一框架中「给定边界对时桥可 simulation-free 训练」这一设计空间的早期实例。

在推理管线环节上，I2SB 对应「配对翻译的桥构造与训练」：它不学耦合（配对数据已给定耦合），只学给定耦合下的桥。这与 DDIB 的「两段 SGM latent 拼接」形成对照——DDIB 免重训但走高斯中继，I2SB 重训但起点是信息丰富的退化图。I2SB 的 Proposition 3.4（OT-ODE）把桥的随机性消去后连接到 OT displacement，为后来「桥 vs 流」的统一讨论（如 UniDB 的 SOC 视角）提供了一个早期注脚。

被取代关系：I2SB 的 Dirac delta 假设在 DDBM/UniDB 等框架中被更一般的 bridge matching / SOC 形式收编，但其「配对修复 simulation-free 训练」的工程配方（解析后验 + DDPM 采样 + 对称噪声调度）被后续工作直接继承。

## 6. 局限与批评

作者承认的局限（Sec. 5.3, p.9）：tractability 要求训练时已知配对数据（clean-degraded pairs），这限制了 I2SB 用于 unpaired 翻译（如 CycleGAN、DDIB 场景）。作者称配对数据在修复任务中「通常几乎零成本可得」，但这一说法只适用于合成退化，真实退化（如 RealSR）配对数据昂贵。

读出来的局限：

1. **Dirac delta 边界的泛化依赖神经网络外推**。作者在 p.4 承认 $\delta_a$ 的奇异性「可能阻碍对训练样本之外的泛化」，但仅以「神经网络强泛化能力」带过，没有给出分布外退化（如训练时未见过的模糊核）的实验证据。这是把理论困难转嫁给网络容量，而非解决。

2. **Table 2 中 Pool 超分和 Table 5 去模糊上 I2SB 的 CA 低于 ADM/Palette/DIM**（Pool: I2SB 71.0 vs ADM 73.4；Uniform 去模糊: I2SB 73.7 vs DDNM 75.5）。作者用「FID 更低」来主张优势，但 FID 与 CA 的 trade-off 未被解释——低 FID 可能来自模式坍缩或过度平滑，CA 下降提示类别语义保持变差。原文未报告 LPIPS、PSNR 等保真度指标，无法判断「FID 更低」是否以牺牲与输入的忠实度为代价。

3. **与 DIM 的对比不对称**。DIM 需要已知退化算子，I2SB 不需要，这是 I2SB 的优势；但 DIM 是「免训练/轻训练」地利用预训练扩散先验，I2SB 需要为每个任务从头训练（虽然用 ADM 初始化）。原文没有报告训练成本对比，只说 SB 此前比 SGM 慢 6×、内存 3×（Figure 2），I2SB 自身的训练开销未量化。

## 7. 对我们的启发

1. **可接切入点 #3（端点奇异与噪声调度）**：I2SB 的对称 $\beta_t$ 调度（Figure 6）是直接沿用 De Bortoli et al. 2021 和 Chen et al. 2021a 的建议，没有针对 Dirac delta 端点的奇异性做设计。Proposition 3.3 的解析后验给出了 $\mu_t$ 和 $\Sigma_t$ 的闭式，可以直接分析 $t\to 0$ 时 $\Sigma_t\to 0$ 的速率与 score 目标方差的关系，设计端点自适应的 $\beta_t$ 调度，检验能否缓解 DDBM 类模型在 $t\to T$ 端点的欠拟合（课题背景提到的 2026 预印本问题）。

2. **可接切入点 #2（保 OT 结构的一步蒸馏）**：Table 8 显示 I2SB 在 NFE=1 时已经超过 GAN 基线（Freeform 10-20%: FID 4.1 vs DeepFillv2 6.7），但 NFE=1 的 I2SB 与 NFE=1000 的 I2SB 之间仍有差距（Table 7: edges→shoes 73.9 vs 37.8）。这说明「一步蒸馏保桥结构」有明确收益空间。I2SB 的 Proposition 3.4 提供了 OT-ODE 的确定性极限，可以作为蒸馏目标或正则项，检验蒸馏后终端耦合是否保持。

3. **可接切入点 #7（医学 SB）**：I2SB 的配对训练假设在医学图像翻译中天然成立（如 CBCT→MDCT 有配准对），且其「从退化图出发、少 NFE 可用」的特性对临床推理时间敏感场景有直接价值。可以按 I2SB 的配方在 SynthRAD 类数据上复现，重点验证低 NFE 下的结构保真度（需补 LPIPS/SSIM，弥补原文只报 FID/CA 的不足）。

## 8. 资源

代码：https://i2sb.github.io/ （项目页与代码，原文 Sec. 1 给出）。

相关论文 arXiv id 互链：
- DDIB: arXiv:2111.05458（Su et al. 2022，原文引用）
- DSBM（Diffusion Schrödinger Bridge Matching）: arXiv:2203.16852（De Bortoli et al. 2021 的后续，课题背景提及）
- DDBM: arXiv:2303.05852（课题背景提及）
- CDSB: Shi et al. 2022（原文 Table 2-5 基线，arXiv id 原文未给全，未见）
- Palette: arXiv:2111.05826（Saharia et al. 2022，原文引用）
- DDRM: arXiv:2201.11793（Kawar et al. 2022a，原文引用）
- DDNM: arXiv:2204.04327（Wang et al. 2022b，原文引用，原文参考文献未列 arXiv id，按已知信息补）
- ΠGDM: arXiv:2205.11487（Song et al. 2022，原文引用，原文参考文献未列 arXiv id，按已知信息补）
