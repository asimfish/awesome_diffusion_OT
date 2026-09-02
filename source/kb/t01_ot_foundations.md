# T01 OT 数学基础（面向生成模型研究者的最小必要集）

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是全项目的"数学地基"，覆盖 Monge/Kantorovich、对偶、Brenier、Benamou–Brenier 动态形式与半离散 OT 五块核心装备，并给出"从哪学最快"的路线。熵正则/Sinkhorn 细节归 T04，Wasserstein 梯度流归 T05，统计估计率归 T06；本笔记只在边界处给指针。

## 1. 核心问题与背景

本子课题回答"做扩散×OT 需要哪些数学装备、从哪学最快"。OT 研究以最小代价把一个概率分布搬运成另一个：Monge 形式求映射，Kantorovich 松弛为耦合上的线性规划并给出对偶理论；由此诱导的 Wasserstein 距离让"分布的空间"具备度量与测地结构。对扩散/流匹配研究者，这套语言不可绕过，因为生成模型本质上就是"把噪声分布运到数据分布"：Brenier 定理刻画最优确定性映射（凸势梯度，联结 Monge–Ampère 与凸分析）；Benamou–Brenier 把 \(W_2\) 写成动能最小化的动态流（与 probability-flow ODE / flow matching 同一语法）；半离散 OT 精确对应"连续先验 → 有限样本数据集"的真实设定。掌握这五块，即拥有阅读扩散×OT 文献的最小充分装备。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Peyré & Cuturi, *Computational Optimal Transport* | 2019·Foundations and Trends in ML | [B] | 计算 OT 标准教材：离散 OT、对偶、动态形式、barycenter 全覆盖，"从零到能跑代码"的主线读物 | [arXiv](https://arxiv.org/abs/1803.00567) · [官方站+PDF](https://optimaltransport.github.io/) |
| ⭐ Peyré, *Optimal Transport for Machine Learners* | 2025·arXiv 课程讲义 | [B] | 面向 ML 的现代精简版：Monge/Kantorovich、Brenier、对偶、动态形式、Bures 度量、梯度流，并直连 GAN/扩散/transformer，配可运行 notebook | [arXiv](https://arxiv.org/abs/2505.06589) · [网站](https://www.gpeyre.com/ot4ml/) |
| ⭐ Santambrogio, *Optimal Transport for Applied Mathematicians* | 2015·Birkhäuser | [B] | 应用数学侧标准参考：Kantorovich 对偶、Brenier、Benamou–Brenier、W 空间几何的严格但可读证明 | [作者稿 PDF](https://math.univ-lyon1.fr/~santambrogio/OTAM-cvgmt.pdf) |
| Villani, *Topics in Optimal Transportation*（2003, AMS GSM 58）/ *Optimal Transport: Old and New* | 2009·Springer Grundlehren 338 | [B] | 理论百科全书：正则性、几何（Ricci 曲率）方向的终极参考；不建议作为第一本，查证明时用 | [Springer](https://link.springer.com/book/10.1007/978-3-540-71050-9) · [作者免费预印](https://www.ceremade.dauphine.fr/~mischler/articles/VBook-O&N.pdf) |
| Figalli & Glaudo, *An Invitation to Optimal Transport, Wasserstein Distances, and Gradient Flows* (2nd ed) | 2023·EMS Textbooks | [B] | 146 页最短严格入门：对偶、Brenier、W 距离、JKO/Otto 微积分，含带解答习题，一学期课体量 | [EMS](https://ems.press/books/etb/258) · [同作者免费 ETH 讲义](https://people.math.ethz.ch/~afigalli/lecture-notes-pdf/An-introduction-to-optimal-transport-and-Wasserstein-gradient-flows.pdf) |
| Ambrosio, Brué & Semola, *Lectures on Optimal Transport* (2nd ed) | 2024·Springer UNITEXT 169 | [B] | SNS 二十年课程沉淀；给出两种自包含的 Kantorovich 对偶证明，通向几何/泛函不等式与 PDE | [Springer](https://link.springer.com/book/10.1007/978-3-031-76834-7) |
| Mérigot & Thibert, *Optimal Transport: Discretization and Algorithms* | 2021·Handbook of Numerical Analysis 22 | [B] | 半离散 OT 最佳系统讲义：Laguerre cell、damped Newton、离散化误差分析 | [arXiv](https://arxiv.org/abs/2003.00855) |
| Chewi, Niles-Weed & Rigollet, *Statistical Optimal Transport* | 2024·arXiv 讲义（Saint-Flour） | [B] | 统计侧系统讲义（经验测度收敛、估计率）；T01 只需其数学预备章，深入归 T06 | [arXiv](https://arxiv.org/abs/2407.18163) |
| ⭐ Brenier, "Polar factorization and monotone rearrangement of vector-valued functions" | 1991·Comm. Pure Appl. Math. 44(4) | [P] | 二次代价下最优映射存在唯一且 = 凸势梯度；一切"扩散潜码 ≈ OT 映射"讨论的理论根基 | [DOI](https://doi.org/10.1002/cpa.3160440402) |
| ⭐ Benamou & Brenier, "A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem" | 2000·Numerische Mathematik 84 | [P] | \(W_2^2\) = 连续性方程约束下的最小动能；扩散/流匹配轨迹分析的通用语言与数值入口 | [DOI](https://doi.org/10.1007/s002110050002) |
| Kitagawa, Mérigot & Thibert, "Convergence of a Newton algorithm for semi-discrete optimal transport" | 2019·J. Eur. Math. Soc. 21(9) | [P] | 半离散 OT 阻尼牛顿法的全局线性收敛，半离散数值求解的理论支柱 | [DOI](https://doi.org/10.4171/jems/889) · [arXiv](https://arxiv.org/abs/1603.05579) |
| "A Combinatorial Algorithm for Semi-Discrete Optimal Transport" | 2024·NeurIPS | [P] | 与主流 smooth dual/Newton 不同的组合算法路径，半离散 OT 的最新进展 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2d950a2cfd8a75124c178a89545b97fd-Abstract-Conference.html) |
| Peyré, *Optimal and Diffusion Transports in Machine Learning* | 2025·arXiv 综述 | [B] | 直接把扩散与 OT 两条主线串成统一框架（Eulerian/Lagrangian、BB、梯度流、transformer token 流）；本项目的桥梁综述 | [arXiv](https://arxiv.org/abs/2512.06797) |
| Montesuma, Mboula & Souloumiac, "Recent Advances in Optimal Transport for Machine Learning" | 2025·IEEE TPAMI 47(2) | [B] | 2012–2023 OT×ML 全景应用综述（生成、迁移、RL 与计算 OT 扩展）；查应用先查它 | [DOI](https://doi.org/10.1109/tpami.2024.3489030) · [arXiv](https://arxiv.org/abs/2306.16156) |
| Pereira & Amini, "A Survey on Optimal Transport for Machine Learning: Theory and Applications" | 2025·IEEE Access 13 | [B] | 入门友好的 2025 应用综述，含历史脉络与对偶/熵正则数学预备 | [DOI](https://doi.org/10.1109/ACCESS.2025.3539926) · [arXiv](https://arxiv.org/abs/2106.01963) |

## 3. 方法演进脉络

理论主线：Monge（1781）提出"搬土"原始形式——直接求映射 \(T\)，非凸且可能无解；Kantorovich（1942）松弛为耦合 \(\pi\) 上的线性规划，给出对偶理论与存在性，OT 从此成为可分析的对象。1987–1991 年 Brenier 的极分解定理是第二次革命：二次代价下最优映射唯一且为凸势梯度 \(T=\nabla\varphi\)，把 OT 与 Monge–Ampère 方程、凸分析焊接在一起；Gangbo–McCann（1996）借 twist 条件推广到一般代价。McCann（1997）的位移插值把 \(W_2\) 测地线写成 \(((1-t)\mathrm{Id}+tT)_\#\mu\)，"分布形变"有了标准语言。Benamou–Brenier（2000）给出动态形式：\(W_2^2\) 等于连续性方程约束下的最小动能，第一次把静态耦合问题改写为时间上的流，这正是今天 probability-flow ODE、flow matching 分析所用的语法（Otto 2001 进一步把 \(W_2\) 空间视为无穷维黎曼流形，归 T05）。计算主线：半离散 OT 从计算几何的 power/Laguerre diagram 出发（Aurenhammer 等 1998），经 Mérigot（2011）多尺度方法，到 Kitagawa–Mérigot–Thibert（2019）证明阻尼牛顿全局线性收敛、Lévy 的 3D 实现，成为"连续源→离散目标"设定的成熟工具；熵正则/Sinkhorn（Cuturi 2013，归 T04）则把 OT 带进 GPU 时代。教材演进清晰反映重心迁移：Villani（2003/2009）奠定理论百科 → Santambrogio（2015）面向应用数学 → Peyré–Cuturi（2019）面向计算与数据科学 → Figalli–Glaudo（2021/2023）、Ambrosio–Brué–Semola（2024）课程化精炼 → Chewi–Niles-Weed–Rigollet（2024）统计化 → Peyré（2025 两部）直接面向生成模型研究者：学习材料已从"分析定理"转向"ML 最小数学装备 + 可运行代码"。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: 该方向的全部理论词汇都来自本子课题。Benamou–Brenier 动能给出"轨迹直度"的规范度量（任何 advection 场的动能 ≥ \(W_2^2\)，gap 即弯曲程度），可作为无须重训的对齐质量评估量；McCann 位移插值给出"对齐后的理想轨迹"（直线插值即 \(W_2\) 测地线）；Brenier 定理指明最直耦合的形态（凸势梯度），高斯情形有 Bures–Wasserstein 闭式解，可为对齐算法提供精确 sanity check。
- 方向二（OT 引导跨域生成）: Kantorovich 对偶势是天然的 guidance 能量（其梯度即搬运方向）；Brenier map 是跨域翻译的原则性目标；半离散 OT 恰好匹配"连续源分布 → 离散目标样本集"的工程现实——Laguerre cell 把源空间按目标样本划分为引力盆，对偶权重可用 damped Newton 稳定求解，为 data-anchored guidance 提供可计算结构；\(W_2\) 与对偶 gap 亦可作跨域生成的评价度量。

## 5. 开放问题与可发论文的切入点

1. 半离散对偶势做 training-free guidance：在目标域样本上用 damped Newton（KMT 2019）解半离散 OT，把对偶势梯度作为额外 drift 注入预训练扩散采样器；可证目标为经验测度时该 drift 逼近分段 Brenier map；实验在 2D toy + 小规模图像跨域迁移上对比 classifier guidance 的 FID 与 transport cost。
2. BB 动能作为轨迹直度的统一账本：证明（或证伪）rectified-flow/ReFlow 类迭代使 Benamou–Brenier 动能单调不增、极限达到 \(W_2^2\) 下界的充分条件；把 KE\((v)-W_2^2\) gap 做成可复现的"对齐质量"基准指标。
3. 扩散潜码映射何时是 Brenier map：在高斯情形（已知成立）之外，给出一般数据分布下 PF-ODE time-map 偏离凸势梯度的可计算判据（如 cyclical monotonicity 违反率），用 Gaussian mixture 闭式解构造正例/反例，划定"DDPM encoder ≈ OT map"命题的适用边界。
4. 一般代价与 twist 条件在感知潜空间的对应：在 VAE/感知度量诱导的非二次代价下检验 Gangbo–McCann 结构能否给出更语义化的跨域映射，并刻画 twist 失效时 guidance 场的多值性与失稳模式。
5. Bures–Wasserstein 沙盒基准：构建 Monge map、BB 测地线、扩散边缘全部闭式可算的高斯族测试台，系统性证伪各类"轨迹对齐 ≈ OT"宣称，开源为社区 benchmark（资源型论文切入）。

## 6. 代码与资源

- 库：[POT (Python Optimal Transport)](https://pythonot.github.io/)（LP/半离散/barycenter 全家桶）；[OTT-JAX](https://ott-jax.readthedocs.io/)（GPU/自动微分）；[GeomLoss](https://www.kernel-operations.io/geomloss/)（大规模几何损失）；[Geogram](https://github.com/BrunoLevy/geogram)（Lévy，含 3D 半离散 OT）；[pysdot](https://github.com/sd-ot/pysdot)（Mérigot–Thibert 讲义配套的半离散 OT）
- 课程与教程：[OT4ML 网站](https://www.gpeyre.com/ot4ml/)（slides+notebook，与 2025 讲义同步）；[Computational OT for ML（2025 课程）](https://mathurinm.github.io/otml/)；[Göttingen Computational OT](https://ot.cs.uni-goettingen.de/teaching_cot.html)；[OT for Unsupervised Learning tutorial](https://optimaltransporttutorial.github.io/)
- 免费全文：[Peyré–Cuturi 官方 PDF](https://optimaltransport.github.io/pdf/ComputationalOT.pdf)；[Santambrogio 作者稿](https://math.univ-lyon1.fr/~santambrogio/OTAM-cvgmt.pdf)；[Figalli ETH 讲义](https://people.math.ethz.ch/~afigalli/lecture-notes-pdf/An-introduction-to-optimal-transport-and-Wasserstein-gradient-flows.pdf)；[Villani 预印全文](https://www.ceremade.dauphine.fr/~mischler/articles/VBook-O&N.pdf)
- 交互与视频：[OT4ML interactive book](https://www.gpeyre.com/ot4ml/myst/_build/html/index.html)；[Transport: An Interactive Introduction](https://pablowilliams.github.io/Transport/)；[Chizat OT primer 视频](https://www.broadinstitute.org/talks/primer-tutorial-optimal-transport)；[Cuturi mini-tutorial](https://www.youtube.com/watch?v=W29-YKYQLBY)
- 推荐学习路线（最快路径）：① Peyré *OT for Machine Learners*（2025）通读 + notebook（1–2 周，建立全景）→ ② Santambrogio 第 1、5、6 章补严格证明（对偶/Brenier/BB）→ ③ Mérigot–Thibert 半离散章 + POT/pysdot 上手 → ④ Peyré *Optimal and Diffusion Transports*（2025）衔接扩散视角；需要统计保证时跳 T06（Chewi et al.），需要 Sinkhorn 跳 T04。

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2019_Peyre_Computational_Optimal_Transport.pdf | Computational Optimal Transport (FnT ML) | 成功 (42.4MB) |
| 2025_Peyre_OT_for_Machine_Learners.pdf | Optimal Transport for Machine Learners (arXiv 2505.06589) | 成功 (27.8MB) |
| 2025_Peyre_Optimal_and_Diffusion_Transports.pdf | Optimal and Diffusion Transports in Machine Learning (arXiv 2512.06797) | 成功 (0.5MB) |
| 2015_Santambrogio_OT_for_Applied_Mathematicians.pdf | Optimal Transport for Applied Mathematicians（作者稿） | 成功 (7.9MB) |
| 2021_Figalli_Intro_OT_and_Wasserstein_Gradient_Flows.pdf | An Introduction to Optimal Transport and Wasserstein Gradient Flows（ETH 讲义） | 成功 (0.4MB) |
| 2021_Merigot_Thibert_OT_Discretization_and_Algorithms.pdf | Optimal Transport: Discretization and Algorithms (arXiv 2003.00855) | 成功 (0.8MB) |
| 2009_Villani_OT_Old_and_New_preprint.pdf | Optimal Transport: Old and New（2006 Saint-Flour 终稿，作者免费分发版） | 成功 (11.0MB) |
| （未保存） | Recent Advances in Optimal Transport for Machine Learning (arXiv 2306.16156) | 失败（arXiv 限流，按纪律停止重试；HAL 无全文，线上可读） |
