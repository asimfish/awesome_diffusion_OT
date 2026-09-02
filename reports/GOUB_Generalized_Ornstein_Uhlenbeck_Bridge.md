# GOUB: Generalized Ornstein-Uhlenbeck Bridge

> 作者：课题清单未提供 · ICML 2024 · [PMLR](https://proceedings.mlr.press/v235/yue24d.html) · 证据级 [P] · 课题 T14 扩散桥 / Schrödinger 桥的图像到图像翻译
> **一句话**：对广义 OU 过程施加 Doob h-transform 消掉稳态方差，得到点对点修复桥并统一多种桥为特例；另给 Mean-ODE 变体。

⚠ 未读全文，依据摘要。本条目摘要为空，以下仅依据课题清单/课题笔记的一句话贡献；§3/§4 不含任何数字。

## 1. 问题

均值回复型 SDE（课题笔记提到的 IR-SDE 一类）把退化图作为 OU 过程的均值，但终点仍带稳态方差，不是精确到达退化图的点对点映射（依据课题清单"消掉稳态方差"的表述推断）。GOUB 要在 OU 型参考过程上得到端点精确钉住的桥。

## 2. 方法

课题清单转述：对广义 Ornstein–Uhlenbeck（GOU）过程施加 Doob h-transform，使终端方差为零、过程精确到达给定终点，从而实现修复任务中"退化图→清晰图"的点对点映射；证明多种已有桥是其特例；另提出只沿均值走的 Mean-ODE 变体。具体 SDE 系数、训练目标（课题笔记称与最大似然/分数匹配均可适配）：原文未读，未见。

## 3. 理论结果

原文未读，未见。课题清单称"统一多种桥为特例"，定理内容未见。

## 4. 实验与数字

- 任务：图像修复、去雨、超分（课题清单）；课题笔记的常用基准列表提到 Rain100H 去雨与 DIV2K 超分。
- 结论措辞（课题清单）：修复/去雨/超分 SOTA。数字：原文未读，未见。

## 5. 在 OT×扩散地图中的位置

GOUB 与 DDBM（2309.16948）是同一 Doob h-transform 机制的两种参考过程选择（OU 型 vs VE/VP 型）；UniDB（2505.21528）随后把两者都收为 SOC 终端惩罚 $\gamma\to\infty$ 的特例，并以 GOU 过程为其主要实例（UniDB 摘要中"现有用 h-transform 的桥"）。它偏向低层修复任务，与 I2SB 竞争同一赛道但用 OU 均值项显式利用退化图。Mean-ODE 变体对应"随机桥 vs 确定性均值路径"的张力，与 I2SB 的 OT-ODE 消融（Table 6）是同一问题。

## 6. 局限与批评

作者承认的：原文未读，未见。

我读出来的（基于方法类型）：
- h-transform 精确钉住终点，按 UniDB 的论证这对应 SOC 中忽略控制代价的极限，可能导致细节过平滑；GOUB 自身是否观察到该现象需读原文。
- 修复类基准以 PSNR/SSIM/LPIPS 为主，与 T14 翻译类基准（FID）口径不同，跨论文比较需注意。

## 7. 对我们的启发

1. **#7 医学 SB**：OU 均值项天然适合"退化图=均值先验"的医学去噪/超分设定，GOUB 是比 I2SB 更贴近该结构的起点。
2. **#2 OT-aware 调度**：Mean-ODE vs 随机 GOUB 的对比可补一条"确定性 vs 随机"的证据链，与 I2SB Table 6、LBM 消融并列。
3. 读全文后核对其"特例统一"与 DDBM/UniDB 的统一声明是否一致，避免三篇互相声称包含对方。

## 8. 资源

- 代码：https://github.com/Hammour-steak/GOUB（据课题笔记）
- 互链：2309.16948（DDBM，同机制不同参考过程）、2505.21528（UniDB，将其收为特例）、UniDB_UniDB（UniDB++）、I2SB_Image_to_Image_Schr_dinger_Bridge（修复赛道竞争者）
