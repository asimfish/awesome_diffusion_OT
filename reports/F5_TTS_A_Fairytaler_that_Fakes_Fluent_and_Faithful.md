# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

> Chen et al. · ACL 2025 · [ACL Anthology](https://aclanthology.org/2025.acl-long.313/) · 证据级 [P] · 课题 T23 语音与音频中的流匹配与 Schrödinger 桥
> **一句话**：F5-TTS 用 ConvNeXt 精炼文本表示解决 E2 TTS 的对齐鲁棒性问题，并提出免重训的推理期 Sway Sampling，RTF 0.15。

## 1. 问题

F5-TTS 解决的是非自回归（NAR）TTS 中「文本-语音对齐」与「推理效率」两个问题。自回归（AR）TTS 模型通过逐 token 预测隐式建模时长，但存在推理延迟和 exposure bias（Sec.1）；NAR 模型受益于并行推理，但文本与语音之间的对齐建模变得关键且困难。此前 NAR 路线中，NaturalSpeech 3 和 Voicebox 使用 frame-wise 音素对齐，Matcha-TTS 使用 monotonic alignment search 加音素级时长模型；E2 TTS 则完全去掉音素和时长预测器，直接把字符用 filler token 填充到 mel 长度后输入。作者报告 E2 TTS 存在收敛慢和鲁棒性低的问题：在 Mandarin 小模型实验中，E2 TTS 在 800K updates 时 WER 为 9.63，且始终有 7% 测试样本失败（WER>50%），无法通过 re-ranking 解决（Sec.5.1）。作者将原因归结为 E2 TTS 把 padded 字符序列与语音序列直接拼接，导致语义特征与声学特征深度纠缠，且有效信息长度差距大（Sec.3.2）。

## 2. 方法

核心设计分两部分：ConvNeXt 文本精炼和推理期 Sway Sampling。

**ConvNeXt 文本精炼**：字符序列 $z$（含 filler token 填充到 mel 长度）先经过 ConvNeXt V2 blocks，再与 noisy speech 和 masked speech 在特征维拼接，进入 DiT 骨干（adaLN-zero）。作者称这给了文本输入「individual modeling space」，让文本在 in-context learning 前先自我准备，缓解有效信息长度不匹配的问题（Sec.3.2）。模型总参数量 335.8M（DiT 22 层 + ConvNeXt V2 4 层，Sec.4 Training）。

**Sway Sampling**：训练时 flow step 仍用均匀采样 $t \sim U[0,1]$，推理时对均匀采样 $u$ 施加单调变换：

$$f_{\text{sway}}(u; s) = u + s \cdot \left(\cos\left(\frac{\pi}{2}u\right) - 1 + u\right) \tag{7}$$

其中 $s \in [-1, \frac{2}{\pi-2}]$。$s<0$ 时采样密度向 $t\to 0$（早期）倾斜，$s=0$ 退化为均匀采样。作者的理由是 CFM 模型在早期（$t\to 0$）从纯噪声勾勒语音轮廓，文本-语音对齐由最初几步决定，因此给 ODE solver 更多早期步的信息可提升对齐精度（Sec.3.2）。该方法不改变训练，可直接应用于任何 CFM 模型。

**训练与推理流程**：训练任务是 text-guided speech infilling——给定 masked speech $(1-m)\odot x_1$ 和完整文本 $z$，预测被 mask 的 mel 段 $m\odot x_1$。CFM 损失为 OT 形式：

$$L_{\text{CFM}}(\theta) = \mathbb{E}_{t,q(x_1),p(x_0)} \left\| v_t((1-t)x_0 + tx_1) - (x_1 - x_0) \right\|^2 \tag{3}$$

推理时以参考音频 mel $x_{\text{ref}}$ 和拼接文本 $z_{\text{ref}\cdot\text{gen}}$ 为条件，从采样噪声 $x_0$ 出发用 Euler ODE solver 积分到 $x_1$，丢弃参考音频部分后经 Vocos 声码器转波形（Sec.3.1, Sec.4）。时长估计直接用生成文本与参考文本的字符数比例，无独立时长模型（Sec.3.1）。

## 3. 理论结果

无理论结果。Sway Sampling 的系数 $s$ 范围与单调性在 Sec.3.2 给出，但未提供关于最优 $s$ 或离散化误差的理论分析。

## 4. 实验与数字

**训练数据**：Emilia 多语数据集，过滤后约 95K 小时英中数据；小模型消融用 WenetSpeech4TTS Premium 子集（945 小时 Mandarin）。Base 模型训练 1.2M updates，batch size 307,200 audio frames（0.91 小时），8 张 NVIDIA A100 80G，超过一周（Sec.4）。

**测试集**：LibriSpeech-PC test-clean（作者自建 1127 样本子集并发布）、Seed-TTS test-en（1088 样本）、Seed-TTS test-zh（2020 样本）。

**主结果**（Table 1, LibriSpeech-PC test-clean）：

| 模型 | #Param. | #Data | WER(%)↓ | SIM-o↑ | RTF↓ |
|---|---|---|---|---|---|
| Ground Truth (1127 samples) | - | - | 2.23 | 0.69 | - |
| Vocoder Resynthesized | - | - | 2.32 | 0.66 | - |
| CosyVoice | ~300M | 170K Multi. | 3.59 | 0.66 | 0.92 |
| FireRedTTS | ~580M | 248K Multi. | 2.69 | 0.47 | 0.84 |
| E2 TTS (32 NFE) | 333M | 100K Multi. | 2.95 | 0.69 | 0.68 |
| F5-TTS (16 NFE) | 336M | 100K Multi. | 2.53 | 0.66 | **0.15** |
| F5-TTS (32 NFE) | 336M | 100K Multi. | **2.42** | 0.66 | 0.31 |

RTF 在 NVIDIA RTX 3090 上以 10s 语音推理时间计算（Table 1 注）。F5-TTS 16 NFE 的 RTF 0.15 是表中最低；WER 2.42（32 NFE）优于表中所有开源基线（CosyVoice 3.59、FireRedTTS 2.69、E2 TTS 2.95），但 SIM-o 0.66 低于 E2 TTS 的 0.69。

**Seed-TTS 测试集**（Table 2）：F5-TTS 32 NFE 在 test-en 上 WER 1.83、SIM-o 0.67、CMOS 0.31、SMOS 3.89；test-zh 上 WER 1.56、SIM-o 0.76、CMOS 0.21、SMOS 3.83。test-en 上 WER 1.83 优于 CosyVoice（3.39）、FireRedTTS（3.82）、E2 TTS（2.19），但差于 Seed-TTSDiT（1.733*，基线论文报告值）；test-zh 上 WER 1.56 优于 CosyVoice（3.10）、E2 TTS（1.97），差于 FireRedTTS（1.51）和 Seed-TTSDiT（1.178*）。作者注明 Seed-TTS 训练数据和模型规模大数个量级（Sec.5）。

**架构消融**（Sec.5.1, Fig.2, 155M 小模型，945h Mandarin，800K updates，Seed-TTS test-zh）：F5-TTS（32 NFE w/o SS）WER 4.17、SIM 0.54；E2 TTS WER 9.63、SIM 0.53。纯 adaLN DiT（F5-TTS−Conv2Text）无法学会对齐；MMDiT 学得快但崩溃快，产生严重重复话语。F5-TTS+Conv2Audio 比 F5-TTS WER 增加 1.61、SIM 增加 0.01；F5-TTS+LongSkip 和 E2 TTS+Conv2Text 均显著退化。

**Sway Sampling 消融**（Sec.5.2, Fig.3）：更负的 $s$ 值进一步提升性能。默认推理配置为 CFG strength 2、Sway Sampling 系数 $-1$（Sec.5）。「leak and override」实验：将推理初始噪声替换为泄漏了参考音频信息的输入 $(1-t')x_0 + t'x'_{\text{ref}}$（$t'=0.1$），使用 Sway Sampling 时模型能覆盖泄漏内容并跟随文本提示生成，不用则失败，输出被泄漏信息主导（Sec.5.2）。

## 5. 在 OT×扩散地图中的位置

F5-TTS 处于「FM-OT 进入语音」主线的极简化分支。它直接继承 E2 TTS 的「字符 + filler token 填充、无音素/时长模型」范式，而 E2 TTS 又基于 Voicebox 的 text-guided infilling + CFM 框架。F5-TTS 的贡献是把 E2 TTS 的 flat U-Net Transformer 换成 DiT + ConvNeXt 文本精炼，解决对齐鲁棒性；其 OT-CFM 损失（Eq.3）与 Lipman et al. 2022 的 OT 路径一致，训练时 $(x_0, x_1)$ 独立采样，未使用 batch 内 OT 耦合——这对应课题背景中「minibatch-OT 耦合在语音上从未被系统消融」的空白点。

Sway Sampling 是「免重训改轨迹」思想在语音推理端的落地：它不改变训练分布，只重分配推理时 ODE solver 的 flow step 密度，与 T07 的 kinetic optimal path 和课题背景中「OT-aware 采样调度」切入点直接相关。作者在 Sec.5.2 明确表示未来将把它与 training-time noise schedulers 和蒸馏技术结合。与 Bridge-TTS 的 informative prior 路线不同，F5-TTS 仍从高斯噪声出发（$x_0 \sim N(0,I)$），未采用数据端先验。

## 6. 局限与批评

**作者承认的**（Sec.6 Limitations）：
1. mel 频谱序列长度仍远长于文本模态，需要更高效且通用的连续表示来进一步提升效率。
2. 缺乏对副语言细节（如情感）的细粒度控制。

**读出来的**：
1. Sway Sampling 的系数 $s$ 是启发式单参数，默认取 $-1$（Sec.5），无理论依据或按输入自适应的机制；Fig.3 只展示小模型上的 $s$ 扫描，未给出 base 模型上 $s$ 的敏感性分析（Appendix B.3 有更多消融但正文未展开）。
2. 时长估计用字符数比例（Sec.3.1），在跨语言或 code-switching 场景下可能失准；作者未报告该简单策略在 Seed-TTS test-zh 上的时长误差影响。
3. 与 Seed-TTSDiT 的对比不公平：后者训练数据「several million hours」、模型规模大数个量级（Sec.5），F5-TTS 的「state-of-the-art」结论需限定在可比规模内。

## 7. 对我们的启发

1. **Sway Sampling 的理论化与自适应版**（对应切入点 #2）：F5-TTS 的 $s$ 是全局固定启发式。可结合 T07 的 kinetic optimal path 或轨迹曲率估计，推导「给定预训练 FM 速度场，最小化离散化误差的最优时间步分布」，并做按输入文本/音频难度自适应的 $s$ 预测器。改动仅在推理端，可在 F5-TTS 开源栈上直接验证 4-16 NFE 区间的 WER/SIM 变化。
2. **minibatch-OT 耦合消融**（对应切入点 #4）：F5-TTS 训练时 $(x_0, x_1)$ 独立采样（Eq.3），未使用 batch 内 OT 耦合。可在 F5-TTS 或 Matcha 规模复现，比较独立采样 vs. batch 内 OT 耦合在少步推理区间的质量增益；语音 mel 的强结构（谐波、共振峰）可能使耦合收益显著，负结果也有信息量。
3. **Sway Sampling 的跨模型移植验证**：作者声称该方法可无缝应用于任何 CFM 模型（Sec.3.2），但正文只在 F5-TTS 上验证。可在 Matcha-TTS、Voicebox 复现或 CosyVoice 的 FM 解码器上做移植实验，量化 $s$ 的最优值与模型架构/训练数据的关系——这是低成本、可直接产出的实验。

## 8. 资源

- 代码与 checkpoint：https://SWivid.github.io/F5-TTS/ （作者声明已开源，Sec.6）
- 自建 LibriSpeech-PC 4-to-10s 子集（1127 样本）已发布（Sec.4）
- 相关论文：E2 TTS (arXiv:2406.18009)、Voicebox (NeurIPS 36)、Matcha-TTS (ICASSP 2024)、DiTTo-TTS (arXiv:2406.11427)、Seed-TTS (arXiv:2406.02430)、Emilia (arXiv:2407.05361)、WenetSpeech4TTS (arXiv:2406.05763)
