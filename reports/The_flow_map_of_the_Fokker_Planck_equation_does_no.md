# The flow map of the Fokker–Planck equation does not provide optimal transport

> Hugo Lavenant, Filippo Santambrogio · Appl. Math. Lett. 133:108225, 2022（arXiv 无独立编号，作者版见 [cvgmt](https://cvgmt.sns.it/paper/5469/)）· 证据级 [P] · 课题 T02 扩散模型与 OT 的理论联系
> **一句话**：严格证明 Fokker–Planck 流映射 $S_\infty$ 一般不是到标准高斯的 OT 映射，障碍是 $D^2u$ 与 $D^2(\log\det D^2u-\frac12|\nabla u|^2)$ 的非交换性；同时留下"量化次优度"的开放问题。

## 1. 问题

Khrulkov & Oseledets（arXiv 2202.07477，即 `2202.07477`）猜想：沿 Fokker–Planck 方程 $\partial_t\mu-\nabla\cdot(x\mu)-\Delta\mu=0$（Eq.(1)）的 Wasserstein 速度 $v(t,x)=-x-\nabla\log\mu_t(x)$（Eq.(3)）积分 ODE $\partial_tS=v(t,S)$，$S_0=\mathrm{Id}$（Eq.(4)），所得极限映射 $S_\infty$ 是 $\mu_0\to\gamma=\mathcal N(0,I)$ 的最优传输映射（**Conjecture 1**）。Kim & Milman（Math. Ann. 2012）在研究反向热流构造收缩映射时已认为这应当为假，但没给证明；Tanana（2021）对 drift $-Ax$、高斯初值给出了反例，但其论证依赖 $\mu$ 与 $\gamma$ 协方差的非交换性，当 $\gamma$ 是标准高斯（DDPM 的情形）时失效，只留下一个无误差估计的数值暗示（"Comparison with a previous counterexample"）。

本文要做的：对 $\gamma$ 为各向同性标准高斯、$\mu_0$ 光滑快衰减的情形，给出严格反例，并且不依赖高斯显式公式。

## 2. 方法

**预备**：$S_{t\#}\mu_0=\mu_t$；$|v(t,x)|\le e^{-t}\|v(0,\cdot)\|_{L^\infty}$（Eq.(6)，来自势 $|x|^2/2$ 的一致凸性 / Bakry–Émery / JKO 估计），故所有轨迹有极限 $S_\infty$，且 $S_{\infty\#}\mu_0=\gamma$。判据是 Brenier 定理（Theorem 4）：光滑传输映射 $T$ 最优 $\iff T=\nabla u$，$u$ 凸 $\iff DT$ 处处对称半正定；此时 $u$ 满足 Monge–Ampère 方程 $\det D^2u=\nu_1/\nu_2(\nabla u)$（Eq.(7)）。以及 Prop. 5：光滑微分同胚 $T$ 是 $\nu_1\to\nu_2$ 的 OT 映射 $\iff T^{-1}$ 是 $\nu_2\to\nu_1$ 的 OT 映射。

**为什么直接微分行不通**（Sec."Notations"后段）：Conjecture 1 没有断言 $S_t$ 在 $\mu_0\to\mu_t$ 之间最优。若 $DS_t$ 对任意 $t$ 对称，由行列式条件与连续性可推出正定从而最优（高斯情形正是如此），但 Kim–Milman 已证一般 $DS_t$ 非对称——然而这不排除 $DS_\infty$ 对称，所以问题比看起来难。

**换视角——用半群性与逆映射导出必要条件**：假设 Conjecture 1 对整条曲线 $(\mu_t)_{t\ge0}$ 都成立。则 $S_\infty\circ S_t^{-1}$ 是 $\mu_t\to\gamma$ 的 OT 映射，其逆 $S_t\circ S_\infty^{-1}$ 是 $\gamma\to\mu_t$ 的 OT 映射（Prop. 5）。记 $T=S_\infty^{-1}$，则 $D(S_t\circ T)=DS_t(T)\,DT$ 对所有 $t$ 对称（Eq.(8)）。对 $x$ 微分 Eq.(4) 得 $\partial_tDS_t=-DS_t-D^2\log\mu_t(S_t)DS_t$，在 $t=0$ 处对 Eq.(8) 求时间导数：$[-\mathrm{Id}-D^2\log\mu_0(T(x))]DT(x)$ 对称。由 $DT$ 对称推出 $D^2\log\mu_0(T)$ 与 $DT$ 处处可交换，右复合 $T^{-1}$ 后即
$$D^2\log\mu_0(x)\ \text{与}\ DS_\infty(x)\ \text{处处可交换}.$$
写 $S_\infty=\nabla u$，对 Monge–Ampère 取对数 $\log\det D^2u=\log\mu_0+\frac12|\nabla u|^2+\text{const}$，得必要条件
$$\forall x,\quad D^2\Big(\log\det D^2u(x)-\tfrac12|\nabla u(x)|^2\Big)\ \text{与}\ D^2u(x)\ \text{可交换}\quad\text{(Eq.(9))}.$$
左边含 $u$ 的四阶导数，右边只有二阶——一般不该成立。

**构造反例**：只要找到凸 $u$ 使 Eq.(9) 在某点失效（Eq.(10)），令 $\mu_0=(\nabla u)^{-1}_\#\gamma$，则 $\nabla u$ 是 $\mu_0\to\gamma$ 的 OT 映射而 Eq.(9) 不成立，于是 Conjecture 1 对 $(\mu_t)_{t\ge0}$ 中至少一个不成立。取 $u=|x|^2/2+\varepsilon\varphi$，$\varphi$ 光滑紧支，$\varepsilon$ 小则 $u$ 凸、$\nabla u$ 是紧集外为恒等的 $C^\infty$ 微分同胚。线性化：$D^2u=\mathrm{Id}+\varepsilon D^2\varphi$，$D^2(\log\det D^2u-\frac12|\nabla u|^2)=-\mathrm{Id}+\varepsilon\{D^2\Delta\varphi-D^2[x\cdot\nabla\varphi]\}+o(\varepsilon)$。在 $x=0$ 处 $D^2[x\cdot\nabla\varphi](0)=2D^2\varphi(0)$ 自动可交换，只需 $D^2\Delta\varphi(0)$ 与 $D^2\varphi(0)$ 不交换。取 $\mathbb R^2$ 上 $\varphi(x_1,x_2)=x_1x_2+x_1^4$：$D^2\varphi=\begin{pmatrix}12x_1^2&1\\1&0\end{pmatrix}$，$D^2\Delta\varphi=\begin{pmatrix}24&0\\0&0\end{pmatrix}$，在 $(0,0)$ 处不交换；再乘一个在原点邻域恒为 1 的光滑截断函数得到紧支。

## 3. 理论结果

- **Proposition 2（主结果）**：在 $\mathbb R^2$ 上存在光滑紧支 $\varphi$，使 $u=|x|^2/2+\varepsilon\varphi$（$\varepsilon$ 小）凸、$\nabla u$ 为 $C^\infty$ 微分同胚，且对 $\mu_0=(\nabla u)^{-1}_\#\gamma$ 生成的曲线 $(\mu_t)_{t\ge0}$，Conjecture 1 对某个 $\mu_t$（$t\ge0$）不成立。作者明确指出：不能精确地对 $\mu_0$ 本身证伪，只能对曲线上某个 $\mu_t$；但所有 $\mu_t$ 光滑、log-concave（Kim–Milman Theorem 1.2）、指数衰减，因此反例落在"很好的"分布类中。
- **必要条件 Eq.(9)** 及其三种自动成立的情形：$u$ 为凸二次函数（高斯，与 Khrulkov Theorem 3.1、Kim–Milman 一致）；$\mu_0$ 径向对称；$u$ 可分 $u=\sum_iu_i(x_i)$。
- **速度衰减估计 Eq.(6)**：$|v(t,x)|\le e^{-t}\|v(0,\cdot)\|_\infty$，保证 $S_\infty$ 存在。
- 无定量结论。

## 4. 实验与数字

无数值实验。文中唯一"数字"是反例矩阵：$D^2\varphi(0)=\begin{pmatrix}0&1\\1&0\end{pmatrix}$，$D^2\Delta\varphi(0)=\begin{pmatrix}24&0\\0&0\end{pmatrix}$。作者在结语中承认 Khrulkov 等人的数值证据表明 $S_\infty$ "几乎最优"，并把量化最优性缺陷列为未来研究方向（Concluding remarks）。

## 5. 在 OT×扩散地图中的位置

- **辩论线的裁决**（KB §3）：本文终结了"扩散 encoder 精确等于 Monge 映射"的猜想，把结论精确化为"逐时刻是梯度场（无穷小 OT），复合后一般不是凸函数梯度"。KB §5 第 1 点的开放问题——上界 $\int|S_\infty-\nabla u^*|^2d\mu_0\le C\cdot$(非交换项范数)——直接源于本文结语。
- **与前驱 / 后继**：Tanana（`1709.06464`）是高斯 + 非标准 $\gamma$ 的反例；本文更强（标准 $\gamma$）且不依赖显式公式。Khrulkov（`2202.07477`）对本文的复述"只在一点不同"不准确（见该报告 §6）。Pierret–Galerne（`2405.14250`）Prop. 3 在高斯情形确认"有限区间 PF-ODE = OT 映射"，正是本文列出的第一类例外。P. Zhang et al.（`2311.03886`）声称在"特定条件"下有限区间上 PF 是 Monge 映射，读那篇时要对照本文的 $DS_t$ 非对称论断检查条件。
- **建设性方向**：Dumont–Lacombe–Vialard（`2603.25182`）在传输映射空间内把漂移动力学约束到 OT 映射凸集，是对本文"复合后离开凸梯度类"这一障碍的直接回应。
- 对 KB §4 方向一 / 二的含义：纯重参数化不可能把 PF-ODE 轨迹变成精确 OT 测地线；跨域直接复合两个 encoder 没有最优性保证。

## 6. 局限与批评

作者承认：
1. 反例只能证明曲线 $(\mu_t)$ 上某个时刻的 $\mu_t$ 违反猜想，不能指定是 $\mu_0$（Prop. 2 后的说明）。
2. 没有量化 $S_\infty$ 与 OT 映射的差距；数值上二者几乎一致（Concluding remarks）。

我读出的：
3. 反例是 $O(\varepsilon)$ 扰动的高斯，差距本身也是 $O(\varepsilon)$——它证明"不精确相等"，但没有排除"差距在所有实际分布上都可忽略"的可能；这正是 Khrulkov 一方仍可坚持"近乎 OT"的空间。
4. 论证依赖"对整条曲线都成立"这一加强假设（通过半群性）；从逻辑上，猜想对单个 $\mu_0$ 成立而对其后继 $\mu_t$ 不成立并未被排除，因此严格说本文证伪的是"对所有光滑快衰减 $\mu_0$ 成立"的全称命题。
5. 全文只处理 $\mathbb R^2$ 与标准 OU 漂移；对 VP-SDE 一般 $\beta(t)$ 调度的推广是时间重参数化，不影响结论，但文中未提。

## 7. 对我们的启发

1. **KB §5 第 1 点的直接执行方案**：把 Eq.(9) 的交换子 $[D^2(\log\det D^2u-\frac12|\nabla u|^2),\,D^2u]$ 当作"次优度密度"，猜想 $\int|S_\infty-\nabla u^*|^2d\mu_0\lesssim\varepsilon^2\|[\cdot,\cdot]\|^2$；先在本文的 $u=|x|^2/2+\varepsilon\varphi$ 族上用摄动展开算出二阶项，再用 Khrulkov 的 TT 求解器或 Tanana 的 $DS_\infty$ 非对称检验数值验证。
2. **切入点 #1（MPNA）**：本文告诉我们预训练 encoder 的雅可比非对称部分就是"非 OT 分量"；保边缘噪声指派可以把 $\mathrm{skew}(DS_\infty)$ 的范数作为正则项或诊断量。
3. **切入点 #3（PF-ODE 的 Lipschitz / 收缩理论）**：本文用到的 Kim–Milman 结果（$\mu_t$ 保持 log-concave）与 Eq.(6) 的衰减估计是研究确定性 encoder 收缩性的起点。

## 8. 资源

- 代码：无（纯数学短文）。
- 相关报告：`2202.07477`（被证伪的猜想）、`1709.06464`（高斯反例前驱）、`2405.14250`（高斯闭式解）、`2311.03886`（有限区间正面结论）、`2603.25182`（约束到 OT 映射集的漂移模型）、`2111.11521`（Föllmer / Brownian 传输映射的收缩理论）。
