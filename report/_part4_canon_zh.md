## 4. 经典工作：奠基年表（The Canon）

### 4.1 数学经典（1781–2014）

| 年份 | 工作 | 为什么是经典 |
|---|---|---|
| 1781 | Monge《论土方的搬运》 | 原始映射形式，非凸且可能无解 |
| 1942 | Kantorovich 松弛 | 耦合上的 LP + 对偶，OT 成为可分析对象 |
| 1991 | **Brenier 极分解定理**（CPAM，[P]） | 二次代价最优映射 = 凸势梯度；与 Monge–Ampère、凸分析焊接 |
| 1997 | McCann 位移插值 | $W_2$ 测地线：分布形变的标准语言 |
| 1998 | **JKO 格式**（SIAM JMA，[P]） | Fokker–Planck = KL 的 Wasserstein 梯度流 |
| 2000 | **Benamou–Brenier**（Numer. Math.，[P]） | $W_2^2$ = 最小动能；PF-ODE/FM 分析的通用语法 |
| 2001 | Otto calculus | $\mathcal P_2$ 的黎曼几何化 |
| 2013 | **Cuturi Sinkhorn**（NeurIPS，[P]，`reports/1306.0895.md`） | 熵正则把 OT 带进 GPU 时代 |
| 2014 | Léonard SB 综述（[B]） | Schrödinger 1932 问题的现代梳理：路径熵最小化 ⇔ EOT |

### 4.2 生成建模接口经典（2021–2023）

| 年份·会议 | 工作 | 奠基点 | 深读 |
|---|---|---|---|
| 2021·ICLR Oral | Score-SDE | VP/VE-SDE + PF-ODE 统一框架，定义了「encoder map」 | `reports/2011.13456.md` |
| 2021·ICLR | DDIM | 采样确定化 = PF-ODE 一阶指数离散 | `reports/2010.02502.md` |
| 2021·NeurIPS Spotlight | **DSB** | SB 进生成建模：SGM = 第一次 IPF 迭代 | `reports/2106.01357.md` |
| 2021·NeurIPS | W2 benchmark | 神经 OT 求解器首个 ground-truth 评测：「下游好 ≠ map 准」 | `reports/2106.01954.md` |
| 2022·NeurIPS | EDM | 统一训练/采样设计空间，few-NFE 公共基准 | `reports/2206.00364.md` |
| 2022·NeurIPS | DPM-Solver | 半线性结构定制指数积分器，NFE 数百 → 10–20 | `reports/2206.00927.md` |
| 2023·ICLR | **Flow Matching** | conditional path + CFM 目标：simulation-free 训练 CNF | `reports/2210.02747.md` |
| 2023·ICLR | **Rectified Flow** | 线性插值 + reflow：「直线换算力」纲领 | `reports/2209.03003.md` |
| 2023·ICLR / JMLR 2025 | Stochastic Interpolants | 任意两分布插值统一 flows/diffusions/SB | `reports/2303.08797.md` |
| 2023·ICLR | **Khrulkov 猜想** + Lavenant 反例 | 「扩散≟OT」定型：高斯成立、一般证伪、量化开放 | `reports/2202.07477.md` |
| 2023·ICLR | **DDIB** / NOT | 翻译 = 两段 EOT 串联；weak OT 统一 saddle-point 求解器 | `reports/2203.08382.md` |
| 2023·ICML | **I2SB** | 边界对给定时 SB tractable 化，桥模型工业可用性首证 | `reports/2302.05872.md` |
| 2023·ICML / TMLR 2024 | Multisample FM + OT-CFM | batch 级 OT 耦合进入 FM 训练：直线化 + 方差下降 | `reports/2302.00482.md` |
| 2023·NeurIPS | **DSBM/IMF** | Markov × reciprocal 双投影：SB 求解不再累积误差 | `reports/2303.16852.md` |
| 2023·NeurIPS | UOTM | UOT 半对偶生成模型：outlier 稳健 | `reports/2305.14777.md` |

### 4.3 规模化与工业化（2024）

SD3 大规模消融确立 RF 公式 + logit-normal 时间采样为工业标准（ICML 2024 Oral）；SiT 在 DiT 骨干上完成 interpolant 四轴消融（ECCV 2024）；DDBM 统一桥设计空间（`reports/2309.16948.md`：Theorem 2 只保证边际，见 §2 P3）；Immiscible Diffusion 证明数据管线级噪声指派一行代码加速训练最高 3×（`reports/2406.12303.md`）；AYS/GITS 把采样调度原理化；LightSB-M/α-DSBM/ASBM 把 SB 轻量化在线化；UOT-FM 证明 unbalanced map = 重缩放边际的 balanced map；moscot 把 OT 推到 170 万细胞（T24）。
