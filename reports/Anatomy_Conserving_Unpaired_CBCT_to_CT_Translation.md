# Anatomy-Conserving Unpaired CBCT-to-CT Translation via Schrödinger Bridge

> Ke Shi, Song Ouyang, Gang Liu et al.（武汉大学 / 华中科大协和医院） · MICCAI 2025 · [MICCAI OA](https://papers.miccai.org/miccai-2025/paper/5303_paper.pdf) · [DOI](https://doi.org/10.1007/978-3-032-04965-0_5) · 证据级 [P] · 课题 T15 医学影像模态转换与 OT/SB/扩散
> **一句话**：把 UNSB 搬到无配对 CBCT→CT，换成 AC-ViT 生成器加焦点频率损失；H&N/胸部同区与跨区评测 PSNR/MAE 均最优，但只有像素指标。

## 1. 问题
CBCT 是图像引导放疗（IGRT）的日常扫描模态，但散射伪影与噪声让它不能直接用于剂量计算；sCT 生成的目标是把 CBCT 的可及性与 CT 的剂量学精度结合（Sec. 1）。作者对已有三类方法的判断：监督 U-Net 系需要严格配对且配准误差不可避免；CycleGAN 系的循环一致性约束弱、有结构畸变风险；条件扩散模型在反向去噪中把剂量计算所依赖的高频细节抹平（Abstract, Sec. 1）。作者认为共同的缺陷是「只做全局强度匹配、不保护对放疗敏感的解剖结构」。

论文要做的是：在无配对设定下，用 Schrödinger bridge（SB）直接连接 $\pi_{CBCT}$ 与 $\pi_{CT}$，并通过生成器架构与频率约束保住解剖细节，同时在解剖区域不同（头颈 H&N、胸部）之间泛化。

## 2. 方法
ACSB 的骨架是 UNSB（Kim et al. 2023，原文 [16]）：熵正则 OT 目标 + 马尔可夫链插值采样 + 对抗/patchNCE 正则。新增部分是 AC-ViT 生成器、焦点频率损失与跨区评测协议。

- **SB 目标（Eq.(1)）**：$\min_Q \mathbb E_Q\big[\int_0^1 \tfrac12\|u(t,x_t)\|^2dt\big]+\lambda D_{KL}(Q\|W_\tau)$，s.t. $x_0\sim\pi_{CBCT},x_1\sim\pi_{CT}$，其中控制 $u$ 由 AC-ViT 生成器 $G_\theta$ 参数化，$W_\tau$ 为方差 $\tau$ 的 Wiener 测度。
- **插值机制 IPM（Eq.(2)–(3)）**：给定生成器预测 $x_1(x_{t_k})=G_\theta(x_{t_k},t_k)$，下一状态 $x_{t_{k+1}}\sim\mathcal N\big(\alpha_{k+1}x_1(x_{t_k})+(1-\alpha_{k+1})x_{t_k},\ \sigma^2_{k+1}I\big)$，$\alpha_{k+1}=\frac{t_{k+1}-t_k}{1-t_k}$，$\sigma^2_{k+1}=\tau\alpha_{k+1}(1-\alpha_{k+1})$。训练（随机时间步）与生成（多步迭代）共用同一条链；作者把 $\alpha$ 解释为「保留 $x_{t_k}$ 解剖 vs 引入目标域外观」的权衡系数。
- **损失（Eq.(6)–(10)）**：$\mathcal L_{OT}=\mathbb E\|x_{t_i}-x_1(x_{t_i})\|^2$（传输成本，Eq.(6)）；patch 判别器对抗损失 $\mathcal L_{adv}$（Eq.(7)）；输入 CBCT $x_0$ 与合成 CT 之间的 patchNCE（Eq.(8)，特征取自 AC-ViT 多尺度层）；焦点频率损失 $\mathcal L_{freq}=\|\mathcal F(x_1(x_{t_k}))-\mathcal F(x_1)\|_1+\big\|\|\mathcal F(x_1(x_{t_k}))\|_1-\|\mathcal F(x_1)\|_1\big\|_2$（Eq.(9)），只在以真 CT 作输入的恒等映射分支上施加。总目标 $\mathcal L_{ACSB}=\mathcal L_{adv}+\lambda_{OT}\mathcal L_{OT}+\lambda_{NCE}\mathcal L_{patchNCE}+\lambda_{freq}\mathcal L_{freq}$（Eq.(10)），权重取值原文未给。
- **AC-ViT 生成器（Sec. 2.3, Fig. 1b）**：两层下采样卷积 → 基于 FFT 的谱感知位置嵌入（替代标准位置嵌入，用于抑制网格伪影）→ 时间步 $t$ 与噪声水平 $\sigma$ 的通道仿射条件编码 → 12 个 Transformer 编码块，每块为 LN→MHSA→残差（Eq.(4)）再 LN→LFFN→残差（Eq.(5)），LFFN 把 token 序列还原为 2D 图（Seq2Img），过两个 1×1 卷积与一个深度 3×3 卷积后再展平 → 对称轻量解码器出 CT。
- **训练/推理流程（Sec. 2.4）**：随机取时间步 $t_i$，从 $x_0\sim\pi_{CBCT}$ 跑 $i$ 次 IPM 得 $x_{t_i}$，生成器给出 $x_1(x_{t_i})$，与独立采样的真 CT $x_1$ 一起算上述损失，生成器与 patch 判别器联合训练。推理从 $x_{t_0}=x_0$ 出发迭代 $N$ 步（$N$ 取值原文未给）。

## 3. 理论结果
无理论结果。SB 目标与 IPM 链直接引用 UNSB，论文未给任何新命题。

## 4. 实验与数字
**数据（Sec. 3.1）**：某三级转诊中心的两个专家整理数据集，H&N 与胸部各一套，原始为严格配准的 CBCT–CT 3D 配对扫描，沿 Z 轴切成 2D 切片；训练 2,232 对、测试 480 对（两区合计还是各自，原文未说明）。训练时把配对随机打乱、随机裁 256×256，即「用配对数据模拟无配对训练」。Adam，lr 2e-4，前 100 epoch 固定后 100 epoch 线性衰减，batch 4，RTX 3090。
**基线（Sec. 3.2）**：Pix2Pix、CycleGAN、CUT（CNN 系）、SynDiff、UNSB（扩散/桥系）、ResViT（Transformer 系），均用公开实现重训。
**指标**：全部为像素相似度——SSIM↑、PSNR↑、MAE↓、RMSE↓。MAE/RMSE 的单位原文未标注（数值量级 1.6–21，显然不是 HU）。**没有任何剂量学、HU 误差、分割/Dice 或下游任务指标**；解剖保真只靠 Fig. 2/3 的箭头标注定性说明。

| 设置 | 方法 | SSIM↑ | PSNR↑ | MAE↓ | RMSE↓ | 来源 |
|---|---|---|---|---|---|---|
| 同区 H&N→H&N | ACSB | 0.962 | 33.791 | 1.655 | 5.569 | Table 1 |
|  | ResViT | **0.963** | 33.483 | 1.753 | 5.716 | Table 1 |
|  | UNSB | 0.957 | 33.350 | 1.885 | 5.878 | Table 1 |
|  | CycleGAN / CUT | 0.949 / 0.951 | 33.067 / 32.997 | 1.962 / 1.978 | 6.228 / 6.051 | Table 1 |
|  | SynDiff | 0.917 | 29.154 | 3.994 | 9.568 | Table 1 |
| 同区 Chest→Chest | ACSB | 0.933 | 31.457 | 2.018 | 7.021 | Table 1 |
|  | UNSB | 0.932 | 31.187 | 2.080 | 7.286 | Table 1 |
|  | CUT | 0.931 | 31.381 | 2.054 | 7.089 | Table 1 |
|  | ResViT | 0.928 | 31.128 | 2.041 | 7.207 | Table 1 |
| 跨区 H&N→Chest（H&N 训练直接测胸部） | ACSB | 0.916 | 29.437 | 3.203 | 9.121 | Table 2 |
|  | ResViT | 0.901 | 27.963 | 3.503 | 10.430 | Table 2 |
|  | UNSB | 0.897 | 26.369 | 5.184 | 13.080 | Table 2 |
|  | SynDiff | 0.854 | 22.42 | 9.528 | 20.284 | Table 2 |
| 跨区 Chest→H&N | ACSB | 0.949 | 29.640 | 2.980 | 8.835 | Table 2 |
|  | UNSB | 0.947 | 29.133 | 3.030 | 9.386 | Table 2 |
|  | CUT | 0.943 | 29.190 | 3.473 | 9.415 | Table 2 |
|  | SynDiff | 0.814 | 17.848 | 20.921 | 33.318 | Table 2 |
| 消融（H&N 同区） | 4 次下采样，无 $\mathcal L_{freq}$ | 0.909 | 29.051 | 2.862 | 9.242 | Table 3 |
|  | 2 次下采样，无 $\mathcal L_{freq}$ | 0.961 | 33.689 | 1.717 | 5.592 | Table 3 |
|  | 2 次下采样 + $\mathcal L_{freq}$（完整） | 0.962 | 33.791 | 1.655 | 5.569 | Table 3 |

读数：(i) 同区设定 ACSB 相对最强基线的优势很小（H&N MAE 1.655 vs ResViT 1.753；胸部 PSNR 31.457 vs CUT 31.381），H&N 的 SSIM 还输给 ResViT。(ii) 优势主要在跨区：H&N→Chest 上 UNSB 的 MAE 从同区 2.080 涨到 5.184，ACSB 只从 2.018 涨到 3.203；SynDiff 两个跨区方向都崩掉（MAE 9.528 / 20.921）。(iii) 消融里生成器保留分辨率（下采样 4→2 次）带来 MAE 2.862→1.717 的大头，$\mathcal L_{freq}$ 只再降到 1.655。(iv) 作者在 Fig. 3 指出基线的两种失败模式——「解剖层面结构解体」与「幻觉出不存在的组织界面」（Sec. 3.2），但没有对幻觉给任何度量。所有表格无标准差、无显著性检验。

## 5. 在 OT×扩散地图中的位置
- **继承**：UNSB（T14 `UNSB_Unpaired_Neural_Schr_dinger_Bridge`）的直接医学落地——Eq.(1)–(3) 与 SB/adv/patchNCE 三损失都是 UNSB 的；ACSB 的增量是生成器（AC-ViT）、$\mathcal L_{freq}$ 与跨区协议。它与 T15 里的 PaBoT（`2505.03114`，无配对 MRI→CT）、LMSB（`Harmonizing_Optical_Coherence_Tomography_Across_De`）同属「无配对桥 + 解剖保真」支线，但三者的保解剖手段不同：ACSB 靠架构与频率损失，PaBoT 靠骨轮廓正则，LMSB 靠改传输成本的度量。
- **竞争**：SynDiff（对抗扩散无监督医学翻译）与 ResViT（同为 Çukur 组），二者在本文全部设置里被压过；有配对数据的 DSBM（`2404.11741`）与 SelfRDB（`2405.06789`）则是另一条线，不直接可比。
- **张力**：本文是「像素指标 vs 任务指标」张力的反面样本——题目写 anatomy-conserving，评测只有 SSIM/PSNR/MAE/RMSE；SynthRAD2025 已表明像素指标与剂量学只有中等相关，所以本文关于「保解剖」的结论目前只有定性证据。
- **推理管线位置**：无配对源→目标的 SB 翻译环节（起点=源图像、$N$ 步随机插值），无推理期先验注入。

## 6. 局限与批评
作者承认：论文没有 limitation 段落，未列出任何局限。

我读出来的：
1. **「无配对」是模拟出来的**：数据本身是严格配准的配对切片，训练时打乱；这比真实无配对场景（不同患者、不同解剖分布）容易，而且既然有配对，完全可以报告 HU-MAE、骨 Dice 甚至剂量学指标，论文却没有。
2. **保解剖无度量**：题目与贡献 (1)(2) 都以「解剖保真」「解耦伪影与解剖」为核心，但没有一个几何或下游指标（轮廓 Dice/HD95、分割一致性、幻觉率），Fig. 2/3 的箭头是唯一证据；「解耦」也没有任何解耦度量。
3. **同区增益在噪声范围内**：与 ResViT/CUT/UNSB 的差距在 0.001 SSIM、0.1–0.3 dB 量级，无标准差与显著性检验；「pioneer adaptation of SB to cross-modal medical translation」的表述也过强——2024 年的 DSBM MR→CT（`2404.11741`）与 PET-DSBM 都在前。
4. **SB 只剩名义**：Eq.(6) 的 $\mathcal L_{OT}$ 只有传输成本项，UNSB 目标里的熵项在训练损失中没有出现；$\tau$、$N$、$\lambda$ 取值全部缺失，可复现性依赖代码。
5. 单中心、2D 切片级、MAE 单位不明，跨区实验只做两个部位。

## 7. 对我们的启发
1. **#7 医学 SB · 跨区协议当作解剖分布偏移的压力测试**：「H&N 训练→胸部直接测」是零成本的 anatomical shift 基准，本文显示对抗扩散（SynDiff）在此崩溃而 SB 系退化平缓（Table 2）；我们做 med-bridge benchmark 时应把它列为独立的鲁棒性维度，并补上骨/器官 Dice 使其成为真正的解剖漂移度量。
2. **架构瓶颈先于损失设计**：Table 3 中保留分辨率（少下采样一次）带来的 MAE 改善约是 $\mathcal L_{freq}$ 的 7 倍。给医学桥设计「保解剖」模块之前，先确认生成器没有分辨率瓶颈。
3. **把 IPM 的 $\alpha_k$ 看成解剖/外观权衡的旋钮**：作者的解读（Sec. 2.2）暗示可对 $\alpha_k$ 做空间自适应（如骨区小步、软组织大步），这与 Top-10 #2 OT-aware 调度的「按局部结构调步长」思路一致，可在同一代码基上快速验证。

## 8. 资源
- 代码：https://github.com/Lalala-iks/ACSB
- 相关报告：`UNSB_Unpaired_Neural_Schr_dinger_Bridge`（方法骨架）、`2404.11741`（配对 DSBM sCT，含剂量学评测）、`2505.03114`（PaBoT，无配对 MRI→CT + 骨轮廓）、`Harmonizing_Optical_Coherence_Tomography_Across_De`（LMSB，以度量改造治解剖漂移）、`2405.06789`（SelfRDB，配对医学桥；同组 SynDiff/ResViT 为本文基线）
