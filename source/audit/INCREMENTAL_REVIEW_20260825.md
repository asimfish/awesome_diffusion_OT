# 增量复审 2026-08-25 — 主报告 §11 更新触发点核查

> 依据：`REPORT_DIFFUSION_OT_20260814.md` §11「下次更新触发点」与 `ARIS_AUDIT_20260814.md` Layer-3 裁决（"SynthRAD2025 参赛清单、ECCV/NeurIPS 2026 官方结果发布后应触发增量复审"）。
> 执行：2026-08-25 凌晨 | 方法：官方来源在线核验（ecva.net / proceedings.mlr.press / GitHub / PyPI / arXiv / grand-challenge.org），全部证据附指针。

## 一、触发点逐项裁决

| # | 触发点 | 状态 | 核验证据 | 后续动作 |
|---|---|---|---|---|
| 1 | ECCV 2026 论文集上线 | **未触发** | 2026-08-25 抓取 ecva.net/papers.php（3.6MB），年份串统计最高仅 eccv_2024，无 2026 条目 | 9 月会议期复查；届时补 T14/T16/T17/T19/T20 的 [P] 升级 |
| 2 | NeurIPS 2026 放榜 | **未到期** | 预期 9 月底 | 10 月初复查 |
| 3 | FlashSinkhorn 开源 release | **已核·空位未被填** | GitHub `ot-triton-lab/flash-sinkhorn`：最新 release 仍为 v0.3.3（2026-04-06，PyPI v0.3.3.post1 同日）；仍限平方欧氏 cost、单卡（unbalanced 仅经 `reach` 参数）；ICML 2026 Oral 已宣讲（2026-07-09），但 PMLR 尚无 ICML 2026 卷（现存最新 ICML 卷为 v267 = ICML 2025） | T29 空位 #1（非欧 cost 流式化）与 #2（多 GPU 分布式）均仍开放；FlashSinkhorn 维持 [A] 分级，PMLR 上卷后升 [P] |
| 4 | SynthRAD2025 结果公布 | **已触发·已消化** | 挑战报告 [arXiv 2605.13555](https://arxiv.org/abs/2605.13555)（Rogowski et al., Medical Image Analysis 投稿版）：Task 1（MRI→CT）12 队 / Task 2（CBCT→CT）13 队有效提交；报告全文对 "Schrödinger / bridge / Wasserstein / Sinkhorn / optimal transport" 检索均为 0 命中 | T15 空位由"据检索未见（待核验）"升级为"官方报告证实"；已同步主报告 §9 #7、§11 与幻灯片 5 处 |

## 二、SynthRAD2025 详细发现（更新切入点 #7 的证据基础）

1. **空位证实**：25 份有效提交中 0 个 Schrödinger 桥、0 个 OT 耦合方法。扩散/FM 系每任务仅 3 队且排名靠后——Task 1 中 imi-graz（全 3D flow matching，第 12/12 名）、SEU & Rennes（Swin-VNet DDPM，第 11 名）、Faking it（VS-DDPM，第 7 名）。
2. **新增限定（已写入 #7）**：FM/扩散组在光子 γ 通过率上与 CNN/GAN 组差异显著（Task 1 三个解剖区均显著；Mann–Whitney，α=0.01）；挑战报告明确"图像质量不是剂量准确性的充分替代"（MS-SSIM 与 Dice ρ=0.78–0.79，但与剂量指标仅中等相关）。⇒ #7 的入场设计必须以剂量学指标为目标——质子 γ 是最大短板（Task 2 顶四队仅 86.4–88.6%，而其光子 γ 达 99%+）。
3. **可行性增强**：post-challenge 排行榜持续开放提交至 **2030-03-01**（grand-challenge.org），SB 方法可随时提交获官方口径评测，降低 #7 的评测成本与"自报数字"审稿风险。
4. **笔记溯源**：T15 笔记（8/14 版）第 6 节已收录该挑战报告 arXiv 链接与 FM 参赛方案（arXiv 2510.04823），其第 5 节空位表述在当时口径下准确；本次复审不改 kb 笔记，仅升级主报告与幻灯片口径。

## 三、本次修改清单（全部留痕）

- `REPORT_DIFFUSION_OT_20260814.md`：头部修订记录追加 1 条；§0 速览、§1 动机核验表、§6 方向二、§9 #7、附录 C 行共 5 处 SynthRAD 口径由"待核验"升级为"官方证实"（#7 另加剂量学限定）；§11 触发点全量更新状态。
- `slides/DIFFUSION_OT_SLIDES_20260814.html`：5 处近等长替换（第 5 页"最值得投入的三件事"、方向二空位页 callout 与触发点行、Top-10 表 #7 行、第 19 页审计触发点行）。
- `INDEX.md`：目录树补 `slides/` 与 `_audit/` 两行；使用建议追加第 5 条（审计与复审指引）。
- 新增本文件。
- **未改动**：30 份 kb 笔记、`refs/MASTER_BIBLIOGRAPHY.md`、`ARIS_AUDIT_20260814.md`（历史文档冻结）。

## 四、下次复审排程

- **9 月中旬**（ECCV 2026 会议期 9/8–13 后）：查 ECVA 论文集，触发 [P] 升级批处理。
- **10 月初**：NeurIPS 2026 放榜复查；顺带复查 PMLR ICML 2026 卷（FlashSinkhorn [A]→[P]）。
- **随时**：FlashSinkhorn 新 release ≥ v0.4 或宣布多 GPU 支持时，重估 T29 空位 #2 与报告 §9 第二梯队"分布式 IO-aware Sinkhorn"条目。
