# Mapping Cells Through Time and Space with moscot

> Klein, Palla, Lange et al. · Nature 2025 · [Nature](https://www.nature.com/articles/s41586-08453-2) · 证据级 [P] · 课题 T24 单细胞与生物轨迹推断中的 OT×流
> **一句话**：moscot 用低秩熵 OT/GW/FGW 统一四类单细胞 OT 应用，扩到 170 万细胞×20 时间点 atlas 规模，实验验证 NEUROD2。

⚠ 未读全文，依据摘要

## 1. 问题

时序 scRNA-seq 给出若干时间点的独立细胞群体快照，每个快照是高维基因表达空间中的经验测度，细胞间无对应关系。核心任务有四类：轨迹推断/快照插值、扰动响应预测、跨模态对齐、空间转录组配准。此前方法分散：Waddington-OT 只做相邻快照间离散耦合，LineageOT 依赖谱系条码，SCOT/NovoSpaRc 用 GW 做跨组学与空间重构，但输出都是离散耦合矩阵，无法外推到新细胞或未观测时间；TrajectoryNet、MIOFlow、TIGON 等连续动力学方法要仿真 ODE/PDE 训练，维度与规模受限。工程侧缺少一个统一 API 把全部 OT 应用推到 atlas 规模。

## 2. 方法

moscot 的核心思想是用低秩熵正则 OT 及其变体（GW、FGW）统一四类单细胞 OT 应用，配合 GPU 加速把规模推到 10⁶–10⁷ 细胞。摘要未给出具体公式编号与算法步骤；从课题背景可知其技术路线为：低秩 Sinkhorn 求解熵 OT 耦合，GW/FGW 处理跨空间翻译与空间配准，统一 API 覆盖时序、空间、时空、谱系四类应用。原文截断，未见具体公式与训练/采样流程。

## 3. 理论结果

摘要未报告定理、引理或理论保证。原文截断，未见。

## 4. 实验与数字

摘要未给出具体实验数字。课题背景提供：170 万细胞 × 20 时间点 atlas 规模（来源：调研 agent 一句话贡献，非摘要原文）；实验验证 NEUROD2（来源：调研 agent 一句话贡献，非摘要原文）。摘要本身未含数据集、基线或数值表。原文截断，未见。

## 5. 在 OT×扩散地图中的位置

moscot 位于「静态耦合时代」的工程规模化终点：它把 Waddington-OT 的离散耦合范式、GW 跨组学翻译、FGW 空间配准统一进一个低秩 Sinkhorn + GPU 框架，推到 atlas 规模。与 [SF]²M、GENOT、DeepRUOT 等 simulation-free 连续动力学路线是互补关系：moscot 输出离散耦合矩阵，不学可外推的神经动力学；GENOT 把熵 OT 耦合本身当作条件生成模型，是 moscot 静态耦合的自然后续。在推理管线中，moscot 对应「离散求解器」环节，而非「可外推的神经动力学模型」环节。

## 6. 局限与批评

作者承认的：摘要未报告。读出来的：moscot 输出离散耦合矩阵，无法外推到新细胞或未观测时间（课题背景明确）；摘要未含理论保证；规模数字与 NEUROD2 验证来自调研 agent 转述，非摘要原文，需读全文核实。

## 7. 对我们的启发

1. moscot 的低秩 Sinkhorn + GPU 工程路线可直接作为 #1（免训练 batch 级保边缘噪声指派 MPNA）的底层求解器：用 moscot 的离散耦合做 posterior steering 的初始化，无需梯度更新。
2. moscot 的统一 API 设计可作为 #2（OT-aware 采样调度）的工程参照：把时序、空间、跨模态四类问题放进同一接口，便于后续接 flow matching 外推模块。
3. moscot 的 atlas 规模（170 万细胞）提示 #7（医学 SB）的数据规模上限：在 SynthRAD 等医学数据上做 SB 时，可先借 moscot 做粗耦合，再训练连续流。

## 8. 资源

代码链接：未公开（摘要未提供；Nature 页面可能含代码，原文未读，未见）。相关论文 arXiv id 互链：Waddington-OT、LineageOT、SCOT、NovoSpaRc、TrajectoryNet、MIOFlow、TIGON、[SF]²M、GENOT、DeepRUOT、CytoBridge、BranchSBM、Curly-FM、CellStream、3MSBM/MMTSBM、CellOT、MMFM、MFM、CellFlow、PASTE、DeST-OT、WFM（均来自课题背景，具体 arXiv id 原文未读，未见）。
