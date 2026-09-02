# T14 扩散桥 / Schrödinger 桥的图像到图像翻译

> 检索日期: 2026-08-14 | 状态: 完成
> 定位: 本子课题是「扩散×OT」交叉中应用最成熟的分支：把图像到图像翻译（I2I）直接建模为两个数据分布之间的扩散桥 / Schrödinger 桥（SB，熵正则 OT 的动力学形式），覆盖配对（修复、超分）与非配对（域翻译）两条赛道。SB 基础理论与 IMF/bridge matching 求解器归 T03；纯 neural OT map（OTM/NOT 系）归 T13；医学影像垂直应用（CT/MRI/PET 桥）归 T15，本笔记仅在边界处提及。

## 1. 核心问题与背景

I2I 翻译的本质是把源域分布搬运到目标域分布。标准扩散模型只能"从高斯噪声出发"，做 I2I 时必须依赖条件网络或推理期引导，既浪费了源图像携带的结构信息（退化图像本身就是重建的强先验），又缺乏传输层面的理论解释。扩散桥与 SB 把 I2I 直接写成两个任意分布之间的随机过程：SB 对应熵正则最优传输，一般扩散桥则用 Doob h-transform / Brownian bridge 把过程钉在给定端点上。好处有三：(i) 起点是信息丰富的源图而非纯噪声，所需采样步数与累积误差更小；(ii) 配对（修复/超分/翻译）与非配对（域迁移）能在同一数学框架内处理，前者学"给定耦合下的桥"，后者要同时学"耦合+桥"；(iii) 传输代价与耦合的 OT 语义为"翻译应保留什么、改变什么"提供了原则性答案。2023 年起该方向爆发，已从单点方法演化出统一框架（DDBM/UniDB）、加速采样（DBIM/CDBM）与工业级单步应用（LBM）。

## 2. 关键论文表

| 论文 | 年份·会议/来源 | 证据 | 一句话贡献 | 链接 |
|---|---|---|---|---|
| ⭐ Dual Diffusion Implicit Bridges (DDIB) | 2023·ICLR | [P] | 两个独立预训练扩散的 PF-ODE latent 拼接实现零配对/免联合训练翻译，理论上等价于"源→latent→目标"两段 Schrödinger 桥（熵正则 OT）串联，精确循环一致 | https://openreview.net/forum?id=5HLoTvVGDe |
| ⭐ I2SB: Image-to-Image Schrödinger Bridge | 2023·ICML | [P] | 边界对给定时 SB 退化为边缘解析可算的 tractable 类，simulation-free 训练；ImageNet-256 修复/超分/去模糊/JPEG 修复超越条件扩散，媲美已知退化算子的逆问题法 | https://proceedings.mlr.press/v202/liu23ai.html |
| BBDM: Brownian Bridge Diffusion Models | 2023·CVPR | [P] | 首个把 I2I 建模为（VQGAN latent 上）Brownian bridge 双向扩散过程而非条件生成的工作 | https://openaccess.thecvf.com/content/CVPR2023/html/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.html |
| Diffusion Schrödinger Bridge Matching (DSBM) | 2023·NeurIPS | [P] | IMF + bridge matching 的通用 SB 求解器，为翻译类 bridge 方法提供算法底座（理论细节见 T03，此处作谱系锚点） | https://proceedings.neurips.cc/paper_files/paper/2023/hash/c428adf74782c2092d254329b6b02482-Abstract-Conference.html |
| ⭐ Denoising Diffusion Bridge Models (DDBM) | 2024·ICLR | [P] | 一般化 bridge score matching 统一设计空间（VE/VP 桥），退化情形回收标准扩散与 OT-Flow-Matching；配对翻译（edges2handbags、DIODE）显著超基线 | https://openreview.net/forum?id=FKksTayvGo |
| ⭐ UNSB: Unpaired Neural Schrödinger Bridge | 2024·ICLR | [P] | 利用 SB 自相似性将其分解为对抗学习序列（时间条件判别器+正则化），首次在高分辨率非配对 I2I（horse2zebra 等）上成功 | https://proceedings.iclr.cc/paper_files/paper/2024/file/5491280797f3192b895bce84eb83df8d-Paper-Conference.pdf |
| GOUB: Generalized Ornstein-Uhlenbeck Bridge | 2024·ICML | [P] | 对广义 OU 过程施加 Doob h-transform 消掉稳态方差，实现点对点修复映射并统一多种桥为特例；修复/去雨/超分 SOTA，另给 Mean-ODE 变体 | https://proceedings.mlr.press/v235/yue24d.html |
| Stochastic Interpolants with Data-Dependent Couplings | 2024·ICML (Spotlight) | [P] | 在 stochastic interpolant 框架内形式化"数据依赖耦合"，把退化图像作为条件基分布，超分/修复上验证依赖耦合的收益 | https://proceedings.mlr.press/v235/albergo24a.html |
| ASBM: Adversarial Schrödinger Bridge Matching | 2024·NeurIPS | [P] | 离散时间 IMF（D-IMF）只学少数转移概率，用 DD-GAN 实现，几步推理达到连续 IMF 百步的非配对翻译质量（CelebA 128） | https://openreview.net/forum?id=L3Knnigicu |
| CDBM: Consistency Diffusion Bridge Models | 2024·NeurIPS | [P] | 学习 bridge PF-ODE 的一致性函数，提出 consistency bridge distillation/training 两种范式，采样加速 4-50×，两步生成可用 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/29d4e09f060a95118762296d240b5e63-Abstract.html |
| ⭐ DBIM: Diffusion Bridge Implicit Models | 2025·ICLR | [P] | 把 DDBM 推广为非马尔可夫桥（DDIM 的 bridge 对应物），诱导新 ODE 与高阶求解器，免训练加速 25×；booting noise 保翻译多样性与语义插值 | https://openreview.net/forum?id=eghAocvqBk |
| UniDB: Unified Diffusion Bridge via SOC | 2025·ICML | [P] | 用随机最优控制统一扩散桥：Doob h-transform 是终端惩罚 γ→∞ 的特例，可调 γ 改善细节保真；统一 DDBM/GOUB 等 | https://proceedings.mlr.press/v267/zhu25o.html |
| ⭐ LBM: Latent Bridge Matching | 2025·ICCV (Highlight) | [P] | VAE latent 上的 Brownian bridge matching + 蒸馏，单步（1 NFE）完成重光照/去物体/深度法线估计/阴影生成；消融显示随机桥优于其零噪声极限（流匹配） | https://openaccess.thecvf.com/content/ICCV2025/html/Chadebec_LBM_Latent_Bridge_Matching_for_Fast_Image-to-Image_Translation_ICCV_2025_paper.html |
| UniDB++（UniDB 期刊版）| 2026·IEEE TPAMI | [P] | 推导 UniDB 逆向 SDE 精确闭式解 + data-prediction 参数化 + SDE-Corrector，免训练加速 5-20×，低步数（5-10）保感知质量，DBIM 为其特例 | https://arxiv.org/abs/2505.21528 (DOI: 10.1109/TPAMI.2026.3710696) |
| Latent Schrödinger Bridge (LSB) | 2024·arXiv | [R] | 把 SB PF-ODE 速度场分解为源/目标/噪声三个预测子的线性组合，用预训练 Stable Diffusion + prompt 优化免训练近似，低 NFE 非配对翻译胜过 SDEdit/DDIB | https://arxiv.org/abs/2411.14863 |

2025-2026 前沿预印本雷达（均 [R]，未见正式接收记录）：

- IBCD: Single-Step Bidirectional Unpaired Translation via Implicit Bridge Consistency Distillation — 隐式桥一致性蒸馏 + 分布匹配，单步双向非配对翻译，无对抗损失（OpenReview 显示 ICLR 2026 在审）。https://arxiv.org/abs/2503.15056
- Diffusion Bridge or Flow Matching? A Unifying Framework and Comparative Analysis — SOC 视角首次统一比较桥与流匹配的建模假设与代价函数。https://arxiv.org/abs/2509.24531
- LADB: Latent Aligned Diffusion Bridges — 半监督（少配对+多非配对）域翻译。https://arxiv.org/abs/2509.08628
- RDBM: Residual Diffusion Bridge Model — 残差视角统一桥式修复，避免全局注噪伤害未退化区域。https://arxiv.org/abs/2510.23116
- UDBM: Uncertainty-Aware Diffusion Bridge — 松弛终端约束 + 像素级不确定性引导的 all-in-one 修复。https://arxiv.org/abs/2601.21592
- Bridging Day and Night — 非配对昼夜翻译中的目标类幻觉检测与抑制，关注下游语义忠实而非 FID。https://arxiv.org/abs/2602.15383
- DBMSolver — 利用桥 SDE/ODE 半线性结构的指数积分器免训练采样器，NFE 降 5×。https://arxiv.org/abs/2605.05889

## 3. 方法演进脉络

2022-2023（奠基）：DDIB 最早把两个独立训练的扩散模型经 PF-ODE latent 拼接成翻译管线，并证明其等价于"源→高斯、高斯→目标"两段 Schrödinger 桥的串联，给出熵正则 OT 解释与（离散化误差内的）精确循环一致性——这是"免重训 + OT 语义"路线的起点。BBDM 第一次把 I2I 写成 latent Brownian bridge 的双向扩散，摆脱条件生成范式。I2SB 抓住"边界对给定时 SB 属于 tractable 类、边缘可解析"这一关键性质，把配对修复做成 simulation-free 的大规模训练，在 ImageNet-256 上首次展示桥模型的工业可用性；平行线还有 mean-reverting SDE（IR-SDE，ICML 2023，https://proceedings.mlr.press/v202/luo23b.html），是后来 GOUB 的前身。

2023-2024（框架化）：DSBM 以 IMF+bridge matching 给 SB 提供不再累积误差的通用求解器（归 T03）；DDBM 把 bridge score matching 推成统一设计空间，让配对翻译从"设计一个桥"转向"在框架内选设计点"；GOUB 用 Doob h-transform 统一 OU 型桥并证明多种桥是其特例；stochastic interpolants 一支则从耦合角度切入（data-dependent couplings）。非配对一侧，UNSB 利用 SB 自相似性把问题化为对抗学习序列，首次做通高分辨率非配对翻译；ASBM 把 IMF 离散时间化并用 DD-GAN 实现几步翻译。

2024-2026（加速、统一与落地）：加速线复刻"扩散的老配方"——CDBM 做桥的一致性蒸馏（4-50×），DBIM 构造非马尔可夫桥得到 DDIM 对应物与高阶求解器（25×），DBMSolver 用指数积分器进一步免训练提效；统一线上 UniDB 用随机最优控制收编 Doob-h 类桥（h-transform 为 γ→∞ 特例），其 TPAMI 2026 版（UniDB++）给出闭式逆向解，"Diffusion Bridge or Flow Matching?"在 SOC 视角下统一比较桥与流两条路线；落地线上 LBM 把 Brownian bridge matching 搬进 VAE latent 并蒸馏到单步，覆盖重光照/去物体/深度估计等真实产品任务，LSB 则完全免训练地用预训练 Stable Diffusion 逼近 SB ODE。非配对赛道 2026 年的关注点开始从 FID 转向语义忠实度（昼夜翻译的目标类幻觉抑制）。医学垂直（I3SB、CBCT→MDCT、PET 引导 MRI 翻译等）见 T15。

## 4. 与博客两个方向的关联

- 方向一（无须重训的轨迹对齐）: 强关联。DDIB 就是零重训翻译的原型——两个预训练扩散的 PF-ODE 轨迹在高斯 latent 处对齐拼接；LSB 更进一步，把 SB 速度场分解为预训练 Stable Diffusion 的三个预测子线性组合，完全免训练地"组装"出 SB 轨迹；DBIM / DBMSolver / UniDB++ 则是推理期对桥轨迹重参数化的免训练采样器。三者共同说明：预训练模型 + 轨迹层面的桥接/对齐即可实现跨域传输，可直接作为方向一的证据链与方法库。
- 方向二（OT 引导跨域生成）: 本子课题即该方向的主战场。SB 是熵正则 OT 的动力学形式；DDIB 显式证明"翻译=两段 EOT 串联"；UNSB/ASBM 在非配对设定下逼近 EOT 耦合（传输代价最小化 + KL 约束）；LBM 的消融显示带随机性的桥优于其零噪声极限（即 OT 直线插值的流匹配），提示"熵正则强度"是跨域生成质量的可调旋钮。

## 5. 开放问题与可发论文的切入点

1. 非配对桥的"最优性漂移"度量：UNSB/ASBM/LSB 学到的耦合与真 EOT 耦合的偏差缺乏系统量化。可在能算 ground-truth EOT 的合成基准（高斯混合、SB benchmark）上同时测 coupling 误差与 FID，并提出带传输代价正则的判别器/损失修正，检验"更接近 EOT 是否等于更好翻译"。
2. 保 OT 结构的一步蒸馏：CDBM/IBCD/LBM 蒸馏后是否仍保持桥的边际与终端耦合无理论保证。切入：证明 consistency bridge 蒸馏在何种条件下保持终端耦合不变；设计加 Sinkhorn 散度约束的"保耦合蒸馏"损失，在 edges2handbags/DIODE 上对比蒸馏前后耦合漂移。
3. 端点奇异与噪声调度：DDBM 类模型在 t→T 端点欠拟合（2026 预印本 arXiv:2605.28962 已指出）。切入：把端点行为与 OT map 正则性联系，基于 UniDB 的可调终端惩罚 γ 设计端点自适应 schedule，给出 FID–LPIPS 权衡曲线的经验定律。
4. 配对与非配对的中间地带：LADB 刚试水半监督桥。切入：少量配对样本校准非配对桥的耦合（anchored bridge matching），在真实退化超分（RealSR/DRealSR）与遥感翻译上验证"每增加 1% 配对数据换多少 FID/一致性"。
5. 多域翻译的中继分布选择：DDIB 式 latent 中继固定为高斯，N 域翻译时 O(N) 模型但中继未必传输最优。切入：把中继分布参数化为可学习 barycenter（与 T13 的 neural OT、T03 的多边缘 SB 衔接），证明多域翻译总代价上界并在 AFHQ 三域上验证。

## 6. 代码与资源

- I2SB 官方: https://github.com/NVlabs/I2SB（项目页 https://i2sb.github.io/）
- DDIB 官方: https://github.com/suxuann/ddib
- BBDM 官方: https://github.com/xuekt98/BBDM
- DDBM 官方: https://github.com/alexzhou907/DDBM
- UNSB 官方: https://github.com/cyclomon/UNSB
- DBIM/CDBM（清华 thu-ml 桥模型套件）: https://github.com/thu-ml/DiffusionBridge
- LBM 官方（含 HuggingFace demo）: https://github.com/gojasper/LBM
- UniDB: https://github.com/UniDB-SOC/UniDB ；UniDB++: https://github.com/2769433owo/UniDB-plusplus
- GOUB: https://github.com/Hammour-steak/GOUB
- ASBM: https://github.com/Daniil-Selikhanovych/ASBM
- Stochastic interpolants couplings: https://github.com/interpolants/couplings
- IBCD 项目页: https://hyn2028.github.io/project_page/IBCD/index.html
- 常用基准: edges2handbags(64²)、DIODE-outdoor(256²)、ImageNet-256 修复套件（I2SB 定义）、AFHQ cat↔dog↔wild、horse2zebra、CelebA male↔female(128²)、DIV2K 超分、Rain100H 去雨（GOUB）。

## 7. 本地 PDF 清单

| 文件名 | 论文标题 | 下载状态 |
|---|---|---|
| 2023_Liu_i2sb_image_to_image_schrodinger_bridge.pdf | I2SB: Image-to-Image Schrödinger Bridge | 成功 |
| 2023_Su_ddib_dual_diffusion_implicit_bridges.pdf | Dual Diffusion Implicit Bridges for Image-to-Image Translation | 成功 |
| 2023_Li_bbdm_brownian_bridge_diffusion.pdf | BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models | 成功 |
| 2024_Zhou_ddbm_denoising_diffusion_bridge_models.pdf | Denoising Diffusion Bridge Models | 成功 |
| 2024_Kim_unsb_unpaired_neural_schrodinger_bridge.pdf | Unpaired Image-to-Image Translation via Neural Schrödinger Bridge | 成功 |
| 2025_Zheng_dbim_diffusion_bridge_implicit_models.pdf | Diffusion Bridge Implicit Models | 成功 |
| 2025_Chadebec_lbm_latent_bridge_matching.pdf | LBM: Latent Bridge Matching for Fast Image-to-Image Translation | 成功 |
| 2026_Zhu_unidb_plusplus_soc_fast_sampling.pdf | A Unified and Fast-Sampling Diffusion Bridge Framework via Stochastic Optimal Control (UniDB++, TPAMI 2026) | 成功 |
