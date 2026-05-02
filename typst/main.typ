#import "@preview/blind-cvpr:0.7.0":cvpr2025,  conf-name,  conf-year,  eg,  etal,  indent
#import "/logo.typ":LaTeX,    TeX

#let affls = (
  one:(institution:"Ocean University of China",    location:"Qingdao China"),   
  airi:("AIRI",    "Qingdao",    "China"),   
  skoltech:(
    department:"CS",   
    institution:"Ocean University of China",   
    location:"Qingdao",   
    country:"China"),   
)

#let authors = (
  (name:"陈中浩",    affl:("one",    ),    email:"czh2886@stu.ouc.edu.cn"),   
)

// ===== 中文支持（必须在 #show:cvpr2025.with(...) 之前注册）=====
// 拉丁字符走 Times New Roman（保持 CVPR 风格）; CJK 字符的字体由 typst 按字重自动选:
//   - regular  → SimSun（宋体,   正文）
//   - bold     → SimSun 无 bold 变体,   typst 跳过,   落 SimHei（黑体）→ 真实加粗
// 这样标题/strong 中的中文（如"摘要""1. 引言"）能像英文标题一样真正加粗. 
#set text(font:(
  (name:"Times New Roman",    covers:"latin-in-cjk"),   
  "SimSun",               // 正文 CJK:regular
  "SimHei",               // 标题/strong 的 CJK:自带 bold
  "Microsoft YaHei",      // 兜底（同时含 regular 和 bold）
))

// 行距调整必须注入到 abstract / body / appendix 各自内容的开头
// 因为 cvpr2022 模板内部硬编码了 set par(leading:0.532em),   外层 set 会被它覆盖

// 外层 show regex:**只设 lang/region,   不强制 font**. 
// 原因:此前在 show regex 里强制 font=(SimSun,   ...),   这是字符级局部 set,   
// 会覆盖任何 show heading/strong 内的 font 设置,   导致标题永远回到 SimSun. 
// 字体选择改由外层 set text 的 font 列表 + heading/strong 内局部 set 控制. 
#show regex("[\p{Han}\u3000-\u303f\uff00-\uffef]"):set text(
  lang:"zh",   
  region:"cn",   
)

// 标题/strong 中 CJK 字体选择:
// - SimHei 优先:单字重黑体（元数据≈regular）,   字形比 SimSun 粗但比 YaHei Bold 轻,   
//   更接近中文期刊标题习惯; weight=bold 上下文下 typst 用 closest-match 仍选 SimHei. 
// - Microsoft YaHei 兵底:万一 SimHei 加载失败,   走 YaHei（Regular 或 Bold variant 由 weight 决定）. 
#let cjk-bold-fallback = (
  (name:"Times New Roman",    covers:"latin-in-cjk"),   
  "SimHei",   
  "Microsoft YaHei",   
)

// 摘要标题:直接用 text() 显式构造,   把 font/weight/size 写死. 
// 原因:abstract 是 cvpr2022 在内部 `*Abstract*` 生成的 strong[Abstract],   
// 外层 show strong/heading 的注册时机比 cvpr2022 内部 set 要早,   无法捕获这个 strong. 
// 直接用 text() 绕开所有 show/set 链,   text 元素的 explicit font/weight 字段胜过任何外层 set. 
#show "Abstract":text(
  font:cjk-bold-fallback,   
  weight:"bold",   
  size:12pt,   
)[摘要]

#show:cvpr2025.with(
  title:text(size:22pt,    weight:"bold",    font:cjk-bold-fallback)[前馈三维重建的频域不匹配分析与修正],   
  authors:(authors,    affls),   
  keywords:(),   
  abstract:[
    
    #set par(leading:0.80em)
    前馈式单图三维重建模型（如 LRM）实现了亚秒级推理,  但其渲染质量始终不及逐场景优化方法. 探究这一性能瓶颈面临一个基础障碍:Triplane 表征存在大量渲染等价的冗余解,  导致常规的残差计算极易被冗余参数间的无意义偏移所主导,  从而掩盖了真实的映射误差. 为此,  我们引入 Pred-Init 策略,  以编码器的预测输出作为优化的初始值,  有效排除了冗余解的干扰,  确立了可靠的误差基准. 
    基于该基准,  我们揭示了编码器与解码器之间存在严重的频域不匹配现象:编码器产生的残差能量有近半数集中在 Triplane 的高频段,  但受限于自身的低通特性,  解码器对这部分高频误差几乎不响应; 相反,  仅占总误差能量 15%–17% 的低频偏差,  在解码阶段被放大了 30 至 50 倍,  最终主导了 53%–77% 的终端渲染退化. 
    针对这一明确机制,  我们提出了 LoRA-FreqLoss 联合修正方案. 该方案在架构层面利用低秩自适应限制网络拟合高频噪声的参数自由度,  在损失计算层面利用频率加权将优化梯度引导至解码器敏感的低频段. 该方法极其轻量,  仅需 147K 可训练参数（占全模型 0.09%）,  即可在保留验证集上使 LPIPS 显著下降 11.4%、PSNR 提升 1.9 dB,  且未引入任何样本退化. 代码与实验结果已公开:#link("https://github.com/Pocon041/OpenLRM"). 
  ],   
  bibliography:bibliography("main.bib"),   
  appendix:include "appendix.typ",   
  accepted:true,   
  id:none,   
)

// 在 body 入口注入行距覆盖（位于 cvpr2022 函数内部 set par 之后,   因此能赢）
#set par(leading:0.75em)

// body 入口再注册 show heading / show strong. 
// 注册时机:在 cvpr2022 函数内的 `show heading.where(level:1):h1` 之后,   
// 按 typst "最近优先"原则,   这里的 show 会在 cvpr2022 的 h1 转换**之前**应用到 heading 元素,   
// 从而让 set text(font:...) 被后续 h1 内部的 set text(weight) 继承 → YaHei Bold. 
#show heading:it => {
  set text(font:cjk-bold-fallback)
  it
}
#show strong:it => {
  set text(font:cjk-bold-fallback)
  it
}

= 引言 <sec:intro>

将单图三维重建从耗时的逐场景优化推进为前馈式推理是当前生成式视觉的核心议程. 以 LRM~@hong2024lrm 为代表的大尺度重建模型借助 Transformer 在 Objaverse~@deitke2023objaverse 等多视图数据上端到端学习从二维图像到三维隐式表示的映射,   将原本需要数千次迭代的 NeRF~@mildenhall2020nerf 或 3DGS~@kerbl20233dgs 场景拟合压缩至单次前向. 围绕这一范式涌现了 OpenLRM~@he2023openlrm、Instant3D~@li2024instant3d、DMV3D~@xu2024dmv3d、TripoSR~@tochilkin2024triposr 等开源与变体方案,   分别在编码器骨干（DINOv2~@oquab2024dinov2）、底层表征（Triplane~@chan2022eg3d、3D Gaussians、显式网格）与外部级联（多视图扩散）层面持续扩展. 然而,   无论数据规模如何扩大,   前馈模型在生成高频几何与精细纹理时仍存在显著的感知质量缺陷. 现有改进多通过引入外部 2D 扩散先验（如 DreamFusion~@poole2023dreamfusion）、级联多视图重投影或更换底层表征来绕开编码器-解码器内部的特征传播缺陷. 本文从该缺陷本身出发. 

诊断的前提是能建立一个可靠的潜空间残差度量基准. 理论上,  给定三维物体的多视角监督,  可通过逐场景优化获取理想 Triplane ,  并以其与编码器预测值的差值来定位映射误差.  但 Triplane~@chan2022eg3d 表征不唯一,   即从不同初始化优化得到的两个 理想 Triplane 在数值上几乎正交（逐通道相关性接近零、L1 距离约 2.0）,   渲染出来却几乎一样（§3.2）. 这一实证结果表明,  Triplane 的解空间存在高度冗余,  包含大量数值差异大但渲染等价的隐式解. 如果直接以随机初始化的参考 Triplane 作为基准进行残差分析,  测量结果将被等价解间巨大的无效偏移所主导,  从而完全掩盖了编码器的真实映射偏差. 对此我们采用一个简单的修复:把参考 Triplane 优化的初值设为编码器预测 $T_"pred"$（Pred-Init）,   将参考强制锚定到与编码器预测同一个等价类中,   使残差真正反映可被微调修正的编码器映射偏差. 这一锚定是后续所有频域分析能成立的前提. 

在 Pred-Init 基准上,  我们对编码器预测残差进行了频谱分析,  发现了一种频域不匹配现象. 其中编码器残差 47--52% 集中在 Triplane 的高频段,   但高频截断实验证明,   完全消除这部分高频残差对渲染质量的改善不足 7%; 与之对比,   仅占总残差 15--17% 的低频残差却主导了 53--77% 的渲染退化（§3.2）. 这一现象的原因是解码器具有的低通的网络特性. 以 OpenLRM 采用的 OSGDecoder（仅含两层 128 维隐层）为例,  受限于 Rahaman 等~@rahaman2019spectralbias 揭示的频谱偏置,  以及 Tancik 等~@tancik2020fourier 在坐标网络中观察到的频率响应不对称性,  该解码器在频域上表现出强烈的低通特性:其对低频扰动的敏感度达到了高频的 2.4 倍. 


已有缓解频谱偏置的工作（如 FreeNeRF~@yang2023freenerf 的频率正则化,  或 Cai 等~@cai2024batchnorm 基于 NTK 谱视角的分析）均聚焦于单一坐标网络内部的收敛过程. 本文关注的则是本文则揭示了前馈架构中误差分布与解码器响应的频域不一致现象:编码器误差最集中的频段,  恰好是解码器最不敏感的盲区. 这种不匹配导致标准 MSE 损失在 Triplane 空间中极力优化的残差大部分与最终的感知质量无关,  模型的梯度容量被无效地消耗于解码器本就会自动抑制的高频噪声上. 


既然主导渲染退化的有效误差集中于平滑的低频结构,  那么真正需要修正的映射残差具有极低的维度. 基于这一推断,  我们提出了与解码器响应特性严格对齐的微调方案 LoRA-FreqLoss. 在架构设计上,  我们将低秩自适应（LoRA~@hu2022lora,   限制秩 $r <= 4$）不仅视为一种高效微调的工具,  更将其作为一种结构性正则化:极低的秩约束天然排斥对复杂高频噪声的拟合,  从而在参数更新空间切断了高频过拟合的通道. 在优化目标上,  我们引入频率加权损失,  将梯度显式重定向至解码器高度敏感的低频段,  确保训练轨迹与渲染的感知收益严格对齐. 该方法极其轻量,  仅需全模型 0.09% 的可训练参数. 


综上所述,  本文的核心贡献可归纳为以下三点:

- 确立隐空间误差量化新基准:揭示了 Triplane 隐式表征的参数冗余如何导致测量歧义,  并提出 Pred-Init 策略,  有效排除等价解间的无效偏移. 

- 揭示跨模块频域不匹配机制:基于上述无偏基准,  量化了编码器误差分布与解码器低通响应之间的严重频域脱节. 证明了标准 MSE 训练会将大量梯度容量徒劳地消耗在感知无关的高频段,  为前馈模型的质量瓶颈提供了底层解释. 

- 提出参数高效的频域对齐修正方案:基于上述结论设计了 LoRA-FreqLoss. 通过低秩架构约束避免高频过拟合,  结合频率加权损失重定向低频梯度,  在仅 0.09% 的参数预算下实现 LPIPS 显著降低 11.4%、PSNR 提升 1.9 dB,  且无任何样本退化. 
= 方法 <sec:method>

== 参考 Triplane 与解码器频率响应 <sec:method-triplane>

我们以 OpenLRM-Mix-Base-1.1（DINOv2-Base 编码器、12 层 Transformer 主干、OSGDecoder 渲染器）为研究对象. 给定一个三维物体的 $V = 32$ 个多视角监督对,   我们定义两类 Triplane 用于残差分析:编码器从单视角图像前向得到的预测 Triplane $T_"pred"$,   以及通过逐场景优化获得的参考 Triplane $T_"ref"$:

$ T_"ref" = arg min_T sum_(v=1)^V norm(cal(R)(T,    c_v) - I_v)_2^2 $ <eq:triplane-opt>

其中 $cal(R)(T,    c)$ 是用 Triplane $T$ 在相机 $c$ 下的渲染输出,   ${(I_v,    c_v)}_(v=1)^V$ 是 GT 视图与相机参数对,   解码器与体渲染器全程冻结. 优化器用 Adam（学习率 0.01,   余弦衰减）,   共 2000 次迭代. 在解码器冻结的前提下,   $T_"ref"$ 给出 32 视角监督下的解码器输入侧渲染质量上界. 

§1 已说明,   直接用 $T_"ref"$ 作残差基准会受 Triplane 表征非唯一性的干扰:从随机噪声初始化优化得到的 $T_"ref"^"rand"$ 与从 $T_"pred"$ 初始化优化得到的 $T_"ref"^"init-pred"$ 之间逐通道相关性仅 0.12,   L1 距离约 2.05,   但二者渲染质量几乎相同（详见 §3.2）.  我们把 $T_"ref"$ 优化的初值设为 $T_"pred"$（即 Pred-Init）,   强制把参考锚定到与编码器预测同一个等价类中,   使残差 $Delta T = T_"pred" - T_"ref"^"init-pred"$ 真正反映编码器的可修正误差. 本文后续所有分析均基于该锚定基准; 除非另作说明,   下文 $T_"ref"$ 一律指 $T_"ref"^"init-pred"$. 

为量化解码器对 Triplane 各频段扰动的响应强度,  我们将 Triplane 在每个二维平面上进行 2D 离散傅里叶变换,  按归一化径向频率 $f \in [0,   1]$ 划分频段. 对无偏基准 $T_"ref"$ 注入幅度受控的带通扰动 $Delta_"f"$,  并定义频段敏感度:

$ H(f) = norm(cal(D)(T_"ref" + delta_f) - cal(D)(T_"ref"))_1 / norm(delta_f)_1 $ <eq:transfer>

其中 $cal(D)$ 为解码器与体渲染器,   $delta_f$ 满足 $norm(delta_f)_1 = epsilon$. $H(f)$ 的含义是单位 Triplane 频率扰动在渲染输出上引发的变化幅度. 具体我们将 $[0,    1]$ 均匀划分为 10 段,   每段注入 $epsilon = 0.5$ 的带通噪声并从 8 个均匀视角渲染计算 L1 变化以得到 $H(f)$ 的数值估计. 结合编码器残差的功率谱密度 $|Delta T(f)|^2$,   频率 $f$ 处的有效渲染误差可定义为:

$ E_"eff" (f) = |Delta T(f)|^2 dot.c H(f) $ <eq:eeff>

当 $|Delta T(f)|^2$ 与 $H(f)$ 频率轴上反相关就标志着频域不匹配:编码器误差能量集中的频段恰好是解码器抑制的频段,   标准 MSE 损失对应的梯度信号被解码器压制.  

为了进一步分析这种频率不匹配现象,  我们需要引入频域分离误差分析. 具体而言,  我们以归一化截止频率 $f_c = 0.3$ 对残差 $Delta T$ 进行高低通分离,  分别构造高通和低通的理想 Triplane:$T_"pred" - Delta T_"low"$ 与 $T_"pred" - Delta T_"high"$,  并评估两者在 8 个验证视角下的 L1 误差. 若人为消除某频段的误差后,  终端渲染质量实现了显著跃升,  则可从机制上确证该频段误差在总体 $E_"eff"(f)$ 中占据了绝对主导地位. 

同时,  我们对低频残差 $Delta T_"low"$ 进行了正交剥离,  将其拆分为 DC 分量（即每通道的空间均值,  仅占 $48 times 3 = 144$ 个标量自由度）与去除 DC 后的结构化低频分量. 拆分的目的是为了验证:低频误差对渲染质量的破坏,  究竟是仅仅来源于全局颜色与亮度的简单偏移,  还是源于更实质性的空间结构错位. 

== LoRA-FreqLoss:与解码器频率响应对齐的低秩频域微调 <sec:method-lora>


基于上述误差剥离分析,  我们得出三个相互印证的诊断结论:（i）有效渲染误差高度集中于低频段; （ii）该低频误差不可由简单的 DC 校准抵消,  呈现空间相干的结构性偏移; （iii）主导渲染质量的有效误差空间维度,  远低于 Triplane 的名义维度（$3 times 48 times 64 times 64$）. 这三个结论指明了我们在微调的时候,   参数空间应严格限制在主导渲染质量的低频低维子空间内,  同时优化方向应显式地向解码器敏感的频段引导.  我们据此提出 LoRA-FreqLoss——一个由架构层约束与优化层引导相耦合的联合微调方案. 


架构层借助 LoRA~@hu2022lora 将编码器参数更新约束在秩 $r$ 的子空间内:

$ bold(y)' = bold(y) + bold(B) bold(A) bold(y) dot.c alpha / r $ <eq:lora>

其中 $bold(A) in RR^(r times d)$、$bold(B) in RR^(d times r)$,   $bold(B)$ 以零矩阵初始化以保证训练起点的零扰动; $alpha$ 为缩放因子. 我们通过 forward hook 将适配器注入 Transformer 主干的 24 个注意力层输出（12 层 $times$ {交叉注意力,    自注意力}）. 与 LoRA 在 LLM 与扩散模型中作为显存优化手段的标准用途不同,   本文的低秩约束直接源于误差分析结论:高拟合复杂的高频噪声通常高度依赖高秩的参数空间,  而这部分高频特征在解码阶段又会被低通特性所抹除. 因此,  将更新矩阵的秩严格限制在极低水平,  本质上是从网络架构层面剥夺了模型拟合冗余高频噪声的参数自由度,  使其自然退化为一种强有力的结构正则化器. 

对于优化层,  我们对像素空间残差的 2D FFT 幅度施加频段加权后再做 L2:

$ cal(L)_"freq" = EE[ abs(cal(F)^(-1) [ w(f) dot.c cal(F)(I_"pred" - I_"GT") ])^2 ] $ <eq:freqloss>

其中 $cal(F)$ 为 2D FFT,   $w(f) = w_"low"$ 当 $|f| < f_"cutoff"$（本文取 $w_"low" = 5.0$、$f_"cutoff" = 0.3$）、其余频段为 1. 该操作直接放大了低频残差在反向传播时产生的惩罚信号. 由于前文测得解码器对低频扰动最为敏感,  微调时的梯度更新必须强制向低频区域倾斜. 尽管公式形式与 Mip-NeRF~@barron2021mipnerf 的抗锯齿监督或 FreeNeRF~@yang2023freenerf 的频率正则化相似,  但本文的权重设定并非基于平滑先验的设定,  而是直接用于补偿解码器固有的低通滤波偏置. 

在此方案中,  LoRA 从架构层面限制了可更新的参数子空间,  FreqLoss 从目标层面引导了梯度的优化方向. 单独使用 LoRA,  有限的低秩参数容量仍会被错误消耗在解码器必然抑制的高频细节上（详见 §3.3 的 Std-MSE 对照）; 单独使用 FreqLoss,  则无法从网络结构上剥夺拟合复杂高频噪声的参数自由度（详见附录 E 的全参数微调退化案例）. 两者的正交结合,  确保了微调过程在参数架构与损失目标两个维度上,  均与解码器的固有频率响应达成严格对齐. 

= 实验 <sec:exp>

== 实验设置 <sec:setup>

- *数据集*:来自 Objaverse~@deitke2023objaverse 的 21 个三维物体,   覆盖几何复杂度从凸面光滑（石头）到凹面密集（办公室）的不同分布. 

- *渲染器*:Blender 4.2 Cycles GPU 渲染 $512 times 512$ RGBA 图像（每物体 32 视角）,   按 16:5 划分训练集与验证集. 模型.  预训练 OpenLRM-Mix-Base-1.1（DINOv2-Base 冻结,   12 层 Transformer,   约 170M 参数）. 

- *模型*:采用预训练的 OpenLRM-Mix-Base-1.1（DINOv2-Base 编码器冻结,  12 层 Transformer 主干,  约 170M 参数）.  

- *评估指标*:L1、PSNR、SSIM、LPIPS（AlexNet）,   均在保留验证集上评估. 微调设置.  总计 500 步,   AdamW（"wd"=0.01）,   "lr"=$5 times 10^(-4)$,   每步随机采样一个样本与一个视角. 

- *训练平台*:单张5090显卡（32G）,  CUDA  12.8,   Python  3.12(ubuntu22.04)

== 频域不匹配的实证诊断 <sec:diag>
我们首先证明 Triplane 等价类的存在并验证 Pred-Init 策略的必要性; 随后测量编码器残差谱与解码器敏感度谱以揭示频率响应的负相关; 接着通过频段截断实验将相关性观测上升为机制结论; 最后通过分解关键低频误差排除朴素的全局亮度偏移假设,  并辅以可见性消融排除信息缺失的替代假设. 

为研究 Triplane 解空间的结构,   我们对每个样本分别以随机噪声和 $T_"pred"$ 为初值优化参考 Triplane,   并比较所得 $T_"ref"^"rand"$、$T_"ref"^"init-pred"$ 与 $T_"pred"$ 之间的关系（@tab:init-strategy,   石头样本 fed1c）. 

#figure(
  caption:[初始化策略对参考 Triplane 残差度量的影响. ],   
  placement:top,   
  table(
    columns:4,   
    align:(left,    center,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 3 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 3 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([比较对象],    [逐通道相关性],    [L1 距离],    [最优缩放 $alpha$]),   
    table.hline(stroke:0.4pt),   
    [$T_"pred"$ vs $T_"ref"^"rand"$],    [$-0.005$],    [1.98],    [0.07],   
    [$T_"pred"$ vs $T_"ref"^"init-pred"$],    [0.840],    [0.78],    [1.016],   
    [$T_"ref"^"rand"$ vs $T_"ref"^"init-pred"$],    [0.12],    [2.05],    [---],   
    table.hline(stroke:0.9pt),   
  )
) <tab:init-strategy>

随机初始化与 Pred-Init 各自收敛到的两个参考 Triplane 在通道维度上几乎正交（相关性仅 0.12）,   二者间 L1 距离与 $T_"pred"$ 到 $T_"ref"^"rand"$ 的距离量级相同,   但二者均能良好渲染目标物体; 与之对比,   $T_"pred"$ 与 $T_"ref"^"init-pred"$ 的相关性高达 0.84、最优缩放 $alpha approx 1$,   表明两者的残差具有线性关系,  能够直接代表真实的预测误差. 这证明了 Triplane 表征空间中存在多个数值上完全不同但渲染等价的解. 我们进一步在每个解周围注入 100 个随机单位扰动（$epsilon = 0.1$,   详见附录 C）,   观察到所有方向引发的渲染变化标准差仅占均值的 0.4%--0.7%. 这一低方差特性证明,  这些等价解并非偶然的特殊情况:在每个解的附近,  无论怎么微调特征数值,  都不会改变渲染结果; 但不同的等价解之间,  数值差异却极大.  这确立了 §2.1 中 Pred-Init 的必要性:所有后续分析必须使用 $T_"ref"^"init-pred"$ 作为基准,   否则残差度量将被等价解之间的冗余偏移所主导而失去意义. 


#figure(
  caption:[
   呈现 $Delta T = T_"pred" - T_"ref"^"init-pred"$ 在三个二维平面上的能量分布. 高能量区域集中在物体轮廓与表面材质边界连续延伸,  且跨平面分布具有明显对应性. 
  ],   
  placement:top,   
  kind:image,   
  image("exps/residual_vis/503d24924ffa431690fa75fe37a60f17_triplane_residual_heatmap.png",    width:100%),   
) <fig:residual-heatmap>
\


#figure(
  caption:[
   把残差转换到 3D 表面点上的逐点 RGB 误差. 直方图呈双峰:低误差峰主要由可见区域贡献,   高误差峰主要由遮挡区域贡献. 两峰间隔仅 1.16$times$（详细数据见后文 @tab:visibility）,   远小于"遮挡导致信息论极限"假设所预期的差距,   初步暗示遮挡并非主要瓶颈——这一观察将在后文可见性消融中得到严格验证. 
  ],   
  placement:top,   
  kind:image,   
  image("exps/residual_vis/8476c4170df24cf5bbe6967222d1a42d_error_histogram.png",    width:90%),   
) <fig:error-hist>
\


为揭示编码器残差与解码器敏感度在频率轴上的关系,   我们在 Pred-Init 基准上对 $Delta T = T_"pred" - T_"ref"^"init-pred"$ 做 2D FFT 并按频段累计能量（@tab:residual-energy）,   同时通过 §2.1 的带通扰动注入实验测量解码器敏感度（@tab:decoder-sens）. 

#figure(
  caption:[Triplane 残差能量在三个频段的分布（Pred-Init 基准,   跨样本一致）. ],   
  placement:top,   
  table(
    columns:4,   
    align:(left,    center,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 3 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 3 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([频段],    [石头],    [办公室],    [楼梯]),   
    table.hline(stroke:0.4pt),   
    [低频 (0--15%)],    [17.3%],    [15.3%],    [16.1%],   
    [中频 (15--40%)],    [25.9%],    [23.8%],    [24.5%],   
    [高频 (40--100%)],    [47.4%],    [52.1%],    [49.8%],   
    table.hline(stroke:0.9pt),   
  )
) <tab:residual-energy>

#figure(
  caption:[解码器对各频段扰动的敏感度（$Delta_"render"$ 占比）. ],   
  placement:top,   
  table(
    columns:3,   
    align:(left,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 3 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 2 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([频段],    [石头],    [办公室]),   
    table.hline(stroke:0.4pt),   
    [低频 (0--0.3)],    [40.6%],    [41.7%],   
    [中频 (0.3--0.6)],    [32.2%],    [29.6%],   
    [高频 (0.6--1.0)],    [27.3%],    [28.7%],   
    table.hline(stroke:0.9pt),   
  )
) <tab:decoder-sens>

两表共同构成频域不匹配的核心证据. 编码器残差能量跨样本一致地集中于高频段（47--52%）,   跨石头、办公室、楼梯三类几何复杂度差异极大的物体差异 $<= 5$ 个百分点,   证明它是编码器的系统性行为而非物体特定属性——编码器对 Triplane 执行了隐式的低通滤波. 与此同时解码器对低频扰动的响应是高频的 2.4 倍,   OSGDecoder 仅含 2 层 128 维隐层,   在频域上等价于一个低通放大器,   与 Rahaman 等~@rahaman2019spectralbias 报告的 MLP 频谱偏置完全吻合. 两条频谱在频率轴上呈现明确的反相关:编码器最 _能产生_ 误差能量的频段恰好是解码器最 _不响应_ 的频段. 

这种负相关源于编码器与解码器之间的架构不对称. 编码器（DINOv2 + Transformer）在 2D 图像 patch 上运作,   全局注意力机制天然能捕获多尺度特征,   理论上有足够容量产生高频 Triplane 特征,   但其输出必须经一个仅含 2 层 128 维隐层的浅层 MLP 解码. 后者受固有频谱偏置的限制~@rahaman2019spectralbias,   对低频函数的拟合能力远强于高频. 这一不对称形导致了训练偏差:即便编码器具备生成高频细节的能力,  但在基于渲染的 MSE 监督下,  高频特征接收到的有效回传梯度极其微弱. 因此编码器在训练中将容量优先分配给低频特征精度,   留下高频特征欠定义. 我们观察到的" Triplane 高频残差大但渲染无关"现象是正是这种架构级频率脱节导致的结果

#figure(
  caption:[三个样本下的 Triplane 残差功率谱分布（Pred-Init 基准）,   纵轴为各频段的能量占比. ],   
  placement:top,   
  kind:image,   
  image("exps/frequency_analysis_predinit/cross_sample_frequency.png",    width:95%),   
) <fig:cross-sample-freq>
\

@fig:cross-sample-freq 直直观展示了编码器残差谱的右偏分布特征. 无论样本复杂度如何变化,  高频段的能量占比始终接近半数,  而低频段稳定维持在 15%–17%. 三条高度重合的柱状图为前述系统性偏差提供了直观的视觉支撑. 

#figure(
  caption:[解码器频谱传递函数 $H(f)$ 的测量. 橙色曲线为单位 Triplane 扰动引发的渲染响应幅度,   蓝色柱为各频段的有效误差贡献 $E_"eff" (f)$. ],   
  placement:top,   
  kind:image,   
  image("exps/decoder_analysis/fed1c0493e364d70a54d97cb81b7bc9b_decoder_sensitivity.png",    width:95%),   
) <fig:decoder-sens>
\

@fig:decoder-sens 进一步将解码器侧的两个关键变量叠加展示. 橙色 $H(f)$ 曲线随频率升高而下降,   体现解码器对高频的不敏感; 蓝色柱给出有效误差贡献 $E_"eff" (f) = |Delta T(f)|^2 dot.c H(f)$ 则呈现出明显的左偏. 这意味着,  尽管编码器自身的误差主要在高频,  但在叠加了解码器的衰减作用后,  真正影响最终渲染结果的误差集中到了低频段. 

为把上述反相关上升为结论,   我们执行了针对单一频段的截断实验. @tab:causal 报告将 §2.1 的频段选择性修正应用于 $T_"pred"$ 的结果. 

#figure(
  caption:[不同频段的渲染改善幅度. ],   
  placement:top,   
  table(
    columns:4,   
    align:(left,    center,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 2 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 3 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([修正频段],    [占总残差能量],    [石头改善],    [办公室改善]),   
    table.hline(stroke:0.4pt),   
    [仅修正低频],    [15--17%],    [$+76.9%$],    [$+53.1%$],   
    [仅修正高频],    [47--52%],    [$+1.3%$],    [$+7.2%$],   
    table.hline(stroke:0.9pt),   
  )
) <tab:causal>

占比仅 15%–17% 的低频残差,  在修正后贡献了 53%–77% 的渲染改善; 而占据近半数能量（47%–52%）的高频残差,  彻底消除后仅带来 1%–7% 的微弱提升. 低频与高频误差的有效放大比率达到了 30 至 50 倍. 这一定量结论彻底证实了频域不匹配现象:标准 MSE 训练在 Triplane 空间中所极力最小化的误差能量,  绝大多数与终端感知质量毫无关联,  模型的梯度容量被大规模浪费在了被解码器抑制的高频段上. 

#figure(
  caption:[不同频段的渲染对比. 左:原始 $T_"pred"$ 渲染; 中:仅修正低频后的渲染; 右:GT. ],   
  placement:top,   
  kind:image,   
  image("exps/spatial_freq/503d24924ffa431690fa75fe37a60f17_correction_effect.png",    width:100%),   
) <fig:correction-effect>
\

@fig:correction-effect 提供了上述结论的可视化结果. 中间列的低频修正结果与右侧 Ground Truth 几乎不可区分,  表明仅修复极小比例的低频残差即可挽回绝大部分感知退化; 反之,  对高频段执行同等修复（定量数据见 @tab:causal）在视觉上几乎不产生任何可察觉的变化. 

为了探究这部分主导退化的低频误差是否仅为颜色偏移,  我们对 $Delta T_"low"$ 进行了正交分解（解耦为 144 个标量的空间均值 DC 分量,  以及去除 DC 后的结构化低频分量）,  并分别进行消融测试（@tab:lowfreq-decomp）. 

#figure(
  caption:[低频残差的内部分解. ],   
  placement:top,   
  table(
    columns:3,   
    align:(left,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 3 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 2 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([修正类型],    [石头改善],    [办公室改善]),   
    table.hline(stroke:0.4pt),   
    [仅 DC（每通道均值偏移,   144 个标量）],    [$+7.7%$],    [$+2.4%$],   
    [结构化低频（去除 DC 后）],    [$+54.9%$],    [$+26.6%$],   
    [DC + 结构化低频（完整低频）],    [$+76.9%$],    [$+53.1%$],   
    table.hline(stroke:0.9pt),   
  )
) <tab:lowfreq-decomp>

结果显示,  仅修正 DC 分量对渲染的改善极其有限（个位数百分比）,  这排除了编码器产生全局亮度或颜色漂移这类低级错误的假设. 真正主导感知退化的,  是去除均值后的结构化低频偏差. 

#figure(
  caption:[低频内部结构分解. 从左至右:原始残差、DC 分量、去除 DC 后的结构化低频残差. ],   
  placement:top,   
  kind:image,   
  image("exps/lowfreq_decomp/fed1c0493e364d70a54d97cb81b7bc9b_lowfreq_decomp.png",    width:100%),   
) <fig:lowfreq-decomp>
\

@fig:lowfreq-decomp 的可视化进一步支持了这一定量结论:DC 分量表现为纯色块且不包含任何空间几何信息; 而剥离 DC 后的结构化低频残差仍清晰保留了物体轮廓与材质过渡区域的形状畸变. 

最后,  我们通过消融测试排除了另一项合理的替代假设:即认为编码器的误差瓶颈源自源视角的遮挡导致的几何信息缺失. 我们根据三维表面点的可见性对其进行分组,  并独立计算局部 RGB 误差（@tab:visibility）. 

#figure(
  caption:[按可见性区域的 Triplane 误差对比. ],   
  placement:top,   
  table(
    columns:4,   
    align:(left,    center,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 2 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 3 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([样本],    [可见区域 RGB 误差],    [遮挡区域 RGB 误差],    [比值]),   
    table.hline(stroke:0.4pt),   
    [办公室],    [0.346],    [0.401],    [$1.16 times$],   
    [石头],    [0.160],    [0.185],    [$1.16 times$],   
    table.hline(stroke:0.9pt),   
  )
) <tab:visibility>

统计表明,  遮挡区域的渲染误差仅比可见区域高出约 16%,  这一极微小的差异推翻了遮挡是导致性能下降核心瓶颈的假设. 这证明了前馈模型的感知缺陷本质上来源于特征传播网络（编码器到解码器）本身的频率响应错位,  而非单一视角输入带来的信息论极限. 这也从侧面印证了本文不引入额外多视图先验、转而专注于修正网络架构频率特征的合理性. 

#figure(
  caption:[可见性分组的误差比较. 可见与遮挡区域的误差直方图几乎重叠. ],   
  placement:top,   
  kind:image,   
  image("exps/visibility_analysis/fed1c0493e364d70a54d97cb81b7bc9b_visibility_error.png",    width:95%),   
) <fig:visibility>
\

@fig:visibility 的分布图同样显示,  可见组与遮挡组的误差直方图几乎完全重合. 综合前述 @fig:error-hist 中的双峰观察,  这一结果表明网络重建失败主要表现为全局结构层面的频段偏差,  而不是局部遮挡区域的彻底崩溃. 

== LoRA-FreqLoss 主实验 <sec:main-exp>

我们在保留验证集上系统评估了 §2.2 提出的 LoRA-FreqLoss 联合修正方案. @tab:main 详细对比了基线模型（无微调）、LoRA 结合标准 MSE 损失、以及 LoRA 结合 FreqLoss（秩均设为 $r = 8$）的指标差异. 附录 D 和 E 分别补充了测试时优化（TTO）变体的失效分析及全参数微调导致样本退化的局限性验证.  

#figure(
  caption:[微调策略在验证集上的表现. ],   
  placement:top,   
  table(
    columns:6,   
    align:(left,    center,    center,    center,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 3 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 5 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([方法],    [可训练参数],    [L1 ],    [PSNR],    [SSIM ],    [LPIPS ]),   
    table.hline(stroke:0.4pt),   
    [baseline],    [---],    [0.0320],    [19.9],    [0.889],    [0.160],   
    [LoRA + Std MSE ($r=8$)],    [295K],    [0.0278],    [21.5],    [0.897],    [0.154],   
    [LoRA + FreqLoss ($r=8$)],    [295K],    [0.0268],    [22.0],    [0.900],    [0.146],   
    table.hline(stroke:0.9pt),   
  )
) <tab:main>

LoRA-FreqLoss 仅用全模型 0.17% 的可训练参数（295K）,   在所有四项指标上同时取得最优; 相对基线 L1 降低 16.2%、PSNR 提升 2.1 dB、LPIPS 降低 9.3%. 在相同参数预算下,   引入频率加权损失带来 LPIPS 多降 5.1 个百分点的额外提升. 这恰好对应 §2.2 的设计预测:频率加权显著放大对感知质量影响最大的低频梯度. 低秩约束 + 频率加权的组合还系统优于参数量 170 倍的全量微调（详见附录 E）,   说明在频率加权损失的引导下,   低秩架构约束从物理层面切断了高频拟合通道,  使得宝贵且有限的优化容量被集中分配到了关键结构误差的修正上. 

#figure(
  caption:[LoRA-FreqLoss 训练过程的 4 项指标收敛曲线（橙:FreqLoss; 蓝:Std MSE）. ],   
  placement:top,   
  kind:image,   
  image("exps/lora_freq/convergence_4metrics.png",    width:100%),   
) <fig:convergence>
\

@fig:convergence 显示两种损失函数在训练过程的指标演化. 在 L1 与 PSNR 这种纯像素级指标上,  FreqLoss 相对 Std MSE 保持着稳定但幅值有限的优势; 但在 LPIPS 这一衡量感知结构一致性的指标上,  两者的性能差距随训练步数急剧扩大,  并在收敛时拉开显著身位. 这直接证实了在显式重定向低频梯度后,  网络的更新权重被优先分配给了感知最为敏感的结构属性上. 全部实验在 500 步内收敛完毕且全程未出现过拟合震荡. 



为验证有效误差子空间确实是低维的,   我们系统消融 LoRA Rank $r in {2,    4,    8,    16,    32}$,   并平行报告 FreqLoss 与 Std MSE 两条对照线的 LPIPS（@tab:rank-ablation）. 

#figure(
  caption:[LoRA Rank 消融实验. ],   
  placement:top,   
  table(
    columns:7,   
    align:(left,    center,    center,    center,    center,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 6 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 6 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([Rank],    [参数量],    [L1  (Freq)],    [PSNR  (Freq)],    [SSIM  (Freq)],    [LPIPS (Freq)],    [LPIPS (Std)]),   
    table.hline(stroke:0.4pt),   
    [---],    [---],    [0.0320],    [19.9],    [0.889],    [0.161],    [0.161],   
    [2],    [74K],    [0.0274],    [21.6],    [0.897],    [0.154],    [0.154],   
    [4],    [147K],    [0.0267],    [21.8],    [0.899],    [0.143],    [0.155],   
    [8],    [295K],    [0.0260],    [21.8],    [0.899],    [0.143],    [0.152],   
    [16],    [590K],    [0.0266],    [21.8],    [0.897],    [0.149],    [0.156],   
    [32],    [1.18M],    [0.0263],    [21.8],    [0.898],    [0.154],    [0.163],   
    table.hline(stroke:0.9pt),   
  )
) <tab:rank-ablation>

该消融实验以独立的数据链路为“频域不匹配”现象提供了内禀维度的验证. 测试显示,  无论 Rank 取值如何,  FreqLoss 分支始终优于 Std MSE 分支,  这排除了频率对齐的收益依赖于某个特定 Rank 的质疑. 其中,  配置为 Rank-4 的极小模型（参数量 147K,  占比 0.09%）成功达到 LPIPS 的全局最优拐点（0.143）,  而低至 Rank-2 的变体（74K）也已大幅抛离无微调基线. 这证实了与渲染质量深度绑定的有效偏差确实位于极低维的子空间内. 

当 $r >= 16$ 时,  网络在 L1 误差未恶化的前提下,  LPIPS 出现明确的性能劣化（数值从 0.143 到 0.149 到 0.154 递增）. 这表明,  过剩的参数容量反而诱使网络引入了引发视觉伪影的高频模式以强行压低整体均方误差. 这正是频率脱节机制在优化过程中的直观表现:标准损失在过度冗余的参数空间内无法区分高频噪声与低频结构,  最终导致 PSNR 指标表面维稳,  但代表视觉感知质量的 LPIPS 单调恶化. 数据趋势表明,  在 $r >= 4$ 的参数区间内,  纯像素指标（PSNR/SSIM）已完全饱和失灵,  唯有感知指标（LPIPS）能够真实反映模型重建质量. 

#figure(
  caption:[LoRA Rank 消融的 4 项指标曲线. FreqLoss 始终优于 Std MSE,   Rank-4 处达到 LPIPS 最优拐点. ],   
  placement:top,   
  kind:image,   
  image("exps/lora_ablation/rank_ablation.png",    width:100%),   
) <fig:rank-ablation>
\

@fig:rank-ablation 将表格数据进行了可视化呈现. 图中反映出两个核心机制:其一,  FreqLoss 具备跨 Rank 级别的鲁棒优越性（橙线全程位于蓝线下方）; 其二,  LPIPS 曲线在 Rank-4 处表现为标准的抛物线拐点（先降后升）. 这一拐点形态直接确认了参数自由度过载后极易被冗余高频噪声污染的现象. 


= 结论 <sec:conclusion>

本文系统揭示了前馈式单图三维重建中的频域不匹配现象:编码器产生的残差能量高度集中于 Triplane 的高频段,  但受限于固有低通特性,  解码器对这些高频误差几乎不响应; 相反,  解码器将仅占总能量 15%–17% 的低频结构性偏差放大了 30 至 50 倍,  并由此主导了绝大部分的渲染退化. 这种不对称性导致标准 MSE 微调会将大量的梯度容量徒劳地浪费在被解码器抑制的高频段上. 为了准确观测这一现象,  本文首先解决了一个潜空间测量问题,  即 Triplane 解空间存在大量数值完全不同但渲染等价的解. 如果直接以随机初始化的参考目标进行对比,  计算出的残差将主要反映等价解之间的冗余参数偏移,  而非编码器真实的映射偏差. 基于上述机制结论,  我们提出的 LoRA-FreqLoss 方案仅需动用全模型 0.09% 的参数（$r = 4$,  147K）,  通过在架构上将更新限制在极低秩的子空间、在损失函数上将梯度强制重定向至解码器敏感的低频段,  即可实现 LPIPS 降低 11.4%、PSNR 提升 1.9 dB 且无任何样本退化,  在感知质量上全面超越了参数量大 170 倍的全量微调. 

说点实话,  毕竟是个作业. 首先此实验确实说明了LRM架构下的这种Encoder-Decoder不对称的问题. 由于算力和资金的限制,  我没有改Decoder等等需要重新从头训练的东西,  而是用了16个模型使用Lora微调然后5个验证,  结果上来说,  在生成模型的物理拓扑结构上来说确实有明显改进. 但是代价是牺牲了生成模型的高频纹理细节. 我觉得对于2026年现在来说的话,  虽然大多研究都堆在3DGS或者Sparse Volume,  但是Triplane还是作为一个基础. 对于这种高频损失的问题的话,  我认为两阶段、或者双分支是很好的解决方法. 目前两阶段的方法也不少吧. 这里由于我个人的精力问题也就不再进一步探讨二阶段了. 

// 附录 A--E 已通过 cvpr2025.with(appendix:include "appendix.typ") 在参考文献之后自动插入
