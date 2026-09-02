# 扩散模型 × 最优传输 知识库总索引

> 构建: 2026-08-14 凌晨 | 方法: 30 个并行文献调研 agent（每 agent 一个子课题）+ 主控聚合
> 证据分级: [P] 正式论文集 / [A] 官方已接收 / [R] 预印本 / [B] 教材综述（口径与 `../ot_variants_survey` 一致）

## 目录结构

```
diffusion_ot_survey/
├── INDEX.md                          # 本文件
├── REPORT_DIFFUSION_OT_20260814.md   # 调研收口报告（动机核验/趋势/Top-10）
├── SYNTHESIS_DIFFUSION_OT_20260825.md    # 深度综合报告（问题/理论/经典/前沿/机会+洞察）★
├── SYNTHESIS_DIFFUSION_OT_20260825.html  # 同上 HTML 版（全景图+管线图+张力图，浏览器打开）
├── _brief/AGENT_BRIEF.md             # 30 个 agent 的统一作业规范
├── kb/t01–t30_*.md                   # 30 份子课题结构化笔记（七节模板）
├── papers/t01–t30/                   # 各课题核心论文 PDF
├── refs/MASTER_BIBLIOGRAPHY.md       # 477 条聚合引用（机械生成）
├── slides/DIFFUSION_OT_SLIDES_20260814.html  # 20 页学术汇报 PPT（单文件）
└── _audit/                           # ARIS 三层审计（20260814）+ 触发点增量复审（20260825）
```

衍生项目（代码仓放 `~/Code/` 避开 iCloud 驱逐）：

| 项目 | 位置 | 对应 | 状态（2026-09-01） |
|---|---|---|---|
| MPNA 免训练 batch 级保边缘噪声指派 | `~/Code/mpna/`（`PROPOSAL.md` + `sandbox/`） | 综合报告 §6.1 Top-10 #1 · 路线图 W5–W8 | 立项书 v0.1 + 沙盒完成（Lemma 1 / Prop. 2 数值精确复现；Hungarian B=128 以 1 次评分/输出达 top-1-of-25 等效增益且零漂移）；下一步 B1 pilot（SDXL-Turbo + GenEval） |

## 总量: 30 份笔记 | 477 条引用（去重约 445–455 篇: [P]372/[A]26/[R]64/[B]15）| 234 篇 PDF（2.8GB）

### A. 理论基础

| 课题 | 笔记 | 论文表 | PDF |
|---|---|---|---|
| T01 OT 数学基础（面向生成模型研究者的最小必要集） | [t01_ot_foundations.md](kb/t01_ot_foundations.md) | 15 篇（B11 P4） | 7 篇 |
| T02 扩散模型与 OT 的理论联系 | [t02_diffusion_ot_theory.md](kb/t02_diffusion_ot_theory.md) | 15 篇（P13 R2） | 8 篇 |
| T03 Schrödinger Bridge 与扩散生成 | [t03_schrodinger_bridge.md](kb/t03_schrodinger_bridge.md) | 25 篇（A1 B3 P16 R5） | 8 篇 |
| T04 熵正则 OT 与 Sinkhorn 在生成建模中的角色 | [t04_entropic_sinkhorn_gen.md](kb/t04_entropic_sinkhorn_gen.md) | 15 篇（P13 R2） | 8 篇 |
| T05 Wasserstein 梯度流与 JKO 格式生成模型 | [t05_wasserstein_gradient_flow.md](kb/t05_wasserstein_gradient_flow.md) | 15 篇（P14 R1） | 8 篇 |
| T06 扩散/流生成模型的收敛性与统计理论 | [t06_convergence_statistics.md](kb/t06_convergence_statistics.md) | 15 篇（P13 R2） | 8 篇 |

### B. 流匹配与轨迹拉直（博客方向一）

| 课题 | 笔记 | 论文表 | PDF |
|---|---|---|---|
| T07 Flow Matching 基础谱系 | [t07_flow_matching_foundations.md](kb/t07_flow_matching_foundations.md) | 15 篇（B1 P14） | 7 篇 |
| T08 OT-CFM 与 minibatch OT 耦合 | [t08_ot_cfm_minibatch.md](kb/t08_ot_cfm_minibatch.md) | 15 篇（A1 P11 R3） | 7 篇 |
| T09 Rectified Flow 与轨迹拉直 | [t09_rectified_flow.md](kb/t09_rectified_flow.md) | 15 篇（P12 R3） | 8 篇 |
| T10 一致性模型与少步蒸馏的 OT 视角 | [t10_consistency_distillation_ot.md](kb/t10_consistency_distillation_ot.md) | 16 篇（P16） | 8 篇 |
| T11 免训练采样器与 ODE 求解器 | [t11_fast_ode_solvers.md](kb/t11_fast_ode_solvers.md) | 22 篇（P16 R6） | 8 篇 |
| T12 推理阶段的 OT 对齐与噪声-样本耦合 | [t12_inference_time_ot_alignment.md](kb/t12_inference_time_ot_alignment.md) | 15 篇（A1 P11 R3） | 8 篇 |

### C. 跨域生成与翻译（博客方向二）

| 课题 | 笔记 | 论文表 | PDF |
|---|---|---|---|
| T13 神经 OT 映射与无配对图像翻译 | [t13_neural_ot_translation.md](kb/t13_neural_ot_translation.md) | 15 篇（P14 R1） | 8 篇 |
| T14 扩散桥 / Schrödinger 桥的图像到图像翻译 | [t14_bridge_i2i.md](kb/t14_bridge_i2i.md) | 15 篇（P14 R1） | 8 篇 |
| T15 医学影像模态转换与 OT/SB/扩散 | [t15_medical_modality_transfer.md](kb/t15_medical_modality_transfer.md) | 15 篇（P12 R3） | 8 篇 |
| T16 OT 代价先验引导的跨域语义对应 | [t16_ot_guided_semantic_correspondence.md](kb/t16_ot_guided_semantic_correspondence.md) | 15 篇（A1 P12 R2） | 7 篇 |
| T17 风格迁移与域自适应中的 OT×扩散 | [t17_style_domain_adaptation.md](kb/t17_style_domain_adaptation.md) | 15 篇（A1 P12 R2） | 8 篇 |
| T18 条件生成与 guidance 的 OT 形式化 | [t18_conditional_ot_guidance.md](kb/t18_conditional_ot_guidance.md) | 15 篇（A2 P11 R2） | 8 篇 |

### D. 模态扩展

| 课题 | 笔记 | 论文表 | PDF |
|---|---|---|---|
| T19 视频生成与时序一致性中的 OT/流 | [t19_video_generation.md](kb/t19_video_generation.md) | 15 篇（P14 R1） | 8 篇 |
| T20 3D/点云/几何生成中的 OT 与流 | [t20_3d_pointcloud_generation.md](kb/t20_3d_pointcloud_generation.md) | 15 篇（P13 R2） | 7 篇 |
| T21 分子与科学计算中的 OT 流生成 | [t21_molecules_science.md](kb/t21_molecules_science.md) | 13 篇（P13） | 8 篇 |
| T22 离散数据与文本中的扩散/流与最优传输 | [t22_discrete_text.md](kb/t22_discrete_text.md) | 28 篇（A5 P18 R5） | 8 篇 |
| T23 语音与音频中的流匹配与 Schrödinger 桥 | [t23_speech_audio.md](kb/t23_speech_audio.md) | 15 篇（P11 R4） | 8 篇 |
| T24 单细胞与生物轨迹推断中的 OT×流 | [t24_singlecell_trajectory.md](kb/t24_singlecell_trajectory.md) | 15 篇（A1 P14） | 8 篇 |

### E. OT 变体前沿

| 课题 | 笔记 | 论文表 | PDF |
|---|---|---|---|
| T25 非平衡/部分 OT 在生成建模中的应用 | [t25_unbalanced_partial_ot_gen.md](kb/t25_unbalanced_partial_ot_gen.md) | 15 篇（A1 P12 R2） | 8 篇 |
| T26 Gromov-Wasserstein 与跨空间生成对齐 | [t26_gromov_wasserstein_gen.md](kb/t26_gromov_wasserstein_gen.md) | 15 篇（A3 P11 R1） | 7 篇 |
| T27 多边际 OT 与 Wasserstein 重心的生成应用 | [t27_multimarginal_barycenter_gen.md](kb/t27_multimarginal_barycenter_gen.md) | 15 篇（A4 P9 R2） | 8 篇 |
| T28 黎曼流形上的流匹配与 OT | [t28_riemannian_manifold_fm.md](kb/t28_riemannian_manifold_fm.md) | 15 篇（A1 P12 R2） | 8 篇 |

### F. 系统与生态

| 课题 | 笔记 | 论文表 | PDF |
|---|---|---|---|
| T29 高性能 OT 求解器与训练基础设施 | [t29_ot_solvers_infra.md](kb/t29_ot_solvers_infra.md) | 15 篇（A3 P8 R4） | 8 篇 |
| T30 端侧部署、benchmark 与顶会趋势（博客落地场景：端侧图像生成） | [t30_edge_benchmark_trends.md](kb/t30_edge_benchmark_trends.md) | 13 篇（A1 P9 R3） | 8 篇 |

## 使用建议

1. 要「问题/理论/经典/前沿/我们能做什么」的深度综合与洞察，读 `SYNTHESIS_DIFFUSION_OT_20260825.md`（或 HTML 版）；要调研收口结论读 `REPORT_DIFFUSION_OT_20260814.md`；再按板块下钻对应 `kb/` 笔记。
2. 每份笔记第 5 节是"可发论文的切入点"，报告第 9 节有全局排序后的精选。
3. 找某篇论文：先查 `refs/MASTER_BIBLIOGRAPHY.md`（按课题分组，含链接与证据级），PDF 在对应 `papers/tNN/`。
4. OT 本身的系统学习路线见姊妹知识库 `../ot_variants_survey/`（教材精读指南 + OT 全景目录）。
5. 要动手做 #1（保边缘噪声指派），读 `~/Code/mpna/PROPOSAL.md`（问题→定理→方法→沙盒证据→claim-driven 实验计划），沙盒一条命令复现。
6. 报告可信度与时效见 `_audit/`：ARIS 三层审计（20260814，裁决 no-new-blocker）与触发点增量复审（20260825，SynthRAD2025 空位经官方证实、FlashSinkhorn 空位仍在）。
