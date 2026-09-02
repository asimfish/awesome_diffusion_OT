# Feedback Schrödinger Bridge Matching (FSBM)

> Theodoropoulos et al. · ICLR 2025 (Oral) · [OpenReview](https://openreview.net/forum?id=k3tbMMW8rH) · 证据级 [P] · 课题 T03 Schrödinger Bridge 与扩散生成
> **一句话**：半监督 SB——用少于 8% 的预配对样本作为状态反馈嵌入广义 EOT，作者报告训练加速且泛化更好。
> ⚠ 未读全文，依据摘要（清单一句话贡献与 KB 笔记；无 abstract 文本）。

## 1. 问题
标准 SB / bridge matching 假设两域完全非配对；实际任务常有少量配对样本（如少数已知对应的细胞或图像对），纯非配对方法浪费了这一监督，全配对方法（如 I²SB 类 bridge matching）又不能用大量非配对数据。本文要把少量配对点作为「反馈」注入 SB。

## 2. 方法
据清单一句话贡献：把配对样本作为 state feedback 项写进广义 EOT / 广义 SB 的代价（KB 归入 GSBM 那条「任务 state cost」路线），再用动态 matching 求解；配对比例 <8%。具体的反馈项形式与算法未读全文，未见。

## 3. 理论结果
未读全文，未见。

## 4. 实验与数字
未读全文，无数字；「显著加速训练并提升泛化」为清单转述，未核对幅度。

## 5. 在 OT×扩散地图中的位置
T03 第四代「半监督」分支，方法学上属 GSBM（`Generalized_SB_Matching_GSBM_Liu_et_al`，同一课题组的广义 SB / 条件随机控制路线）的延伸，与 3MSBM 同组。KB 把它标为方向二「关键点引导的跨域生成」的直接理论模板：少量跨域配对点作为 feedback 引导整体 transport。

## 6. 局限与批评
未读全文，作者自述未见。可推断：(1) 反馈项破坏 Wiener 参考下 EOT 的标准结构，LightSB 类闭式与 `2510.22560` 的统计理论可能不再适用；(2) 配对样本的选取偏差会直接进入耦合。

## 7. 对我们的启发
1. **#1 保边缘噪声指派 MPNA / 方向二**：若医学配准或跨模态任务有少量 landmark 对应，FSBM 提供把它们纳入 SB 目标的模板。
2. 需要全文核对「<8% 配对」的实验设定与基线（是否与 GSBM、DSBM 在同一基准比较）。

## 8. 资源
- 代码：未见
- 相关报告：`Generalized_SB_Matching_GSBM_Liu_et_al`（方法学前身）、`Momentum_Multi_Marginal_SB_Matching_3MSBM_Theodoro`（同组）、`Diffusion_Schr_dinger_Bridge_Matching_DSBM_Shi_et`
