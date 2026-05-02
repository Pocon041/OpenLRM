// 附录入口注入行距覆盖（同 main.typ 中 body 开头的 set par,   原因见那里的注释）
#set par(leading:0.75em)

// 附录入口再注册 show heading / show strong（同 main.typ body 入口的写法）. 
// 原因:appendix 是通过 include 作为 content 传给 cvpr2022 模板渲染的,   
// 这些 show 规则在 cvpr2022 内部的 show heading.where 之后注册 → 优先应用 → 让 CJK 字体走 YaHei Bold. 
#let cjk-bold-fallback = (
  (name:"Times New Roman",    covers:"latin-in-cjk"),   
  "SimHei",   
  "Microsoft YaHei",   
)
#show heading:it => {
  set text(font:cjk-bold-fallback)
  it
}
#show strong:it => {
  set text(font:cjk-bold-fallback)
  it
}

= 附录 <sec:appendix>

== A. 实验配置细节 <sec:app-a>

渲染.  Blender 4.2 Cycles GPU; $512 times 512$ 分辨率; 32 个均匀分布视角; 白色背景（RGBA 合成）; 自动相机距离. 

模型.  编码器:DINOv2-Base（ViT-B/14,   全程冻结）. Transformer:12 层,   768 维,   交叉注意力 + 自注意力. Triplane:$3 times 48 times 64 times 64$. 解码器:OSGDecoder（2 层 MLP,   128 维隐层）. 渲染器:ImportanceRenderer（粗 64 + 精 64 采样点）. 

LoRA 配置.  24 个适配器（12 层 $times$ 2 种注意力类型）,   通过 forward hook 注入. 初始化:$bold(A) tilde cal(N)(0,    0.01)$,   $bold(B) = bold(0)$,   $alpha = 2 r$. 

== B. 频率加权损失实现细节 <sec:app-b>

```
1. 计算残差:e = I_pred − I_GT
2. 2D FFT 逐通道:E = F[e]
3. 权重掩码:w(f) = 5.0 (|f| < 0.3) else 1.0
4. 逆 FFT:e_w = F⁻¹[w · E]
5. 损失:L = mean(e_w²)
```

== C. 等价类的局部各向同性验证 <sec:app-c>

对每个 Pred-Init 参考 Triplane $T_"ref"^"init-pred"$ 周围施加 100 个随机单位扰动（$epsilon = 0.1$）:

#figure(
  caption:[表 C1. 等价类局部各向同性测量. ],   
  placement:top,   
  table(
    columns:5,   
    align:(left,    center,    center,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 2 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 4 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([样本],    [维度],    [$Delta_"render"$ 均值],    [$Delta_"render"$ 标准差],    [标准差/均值]),   
    table.hline(stroke:0.4pt),   
    [石头],    [589,   824],    [$7.56 times 10^(-4)$],    [$5 times 10^(-6)$],    [0.7%],   
    [办公室],    [589,   824],    [$4.88 times 10^(-3)$],    [$1.9 times 10^(-5)$],    [0.4%],   
    table.hline(stroke:0.9pt),   
  )
) <tab:isotropy>

所有扰动方向引发的渲染变化标准差均小于均值的 1%,   证明等价类是全局拓扑特征（多个 L1 约 2.0 分隔的高维近各向同性区域）而非局部线性退化. 

== D. 测试时优化的失败:编码器作为特征分布约束 <sec:app-d>

§3.2 的诊断结论指向了一个自然的问题：既然主导渲染退化的误差集中在低频结构上, 能否在测试阶段直接通过渲染监督来修改 Triplane, 从而获得与微调等价的改善？为验证训练时修正方案的必要性, 我们系统评估了多种测试时优化（TTO）策略。

#figure(
  caption:[表 D1. 测试时优化策略的渲染改善与视觉质量. ],   
  placement:top,   
  table(
    columns:3,   
    align:(left,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 5 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 2 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([方法],    [L1 改善],    [视觉质量]),   
    table.hline(stroke:0.4pt),   
    [Triplane 优化（无约束）],    [$-36.4%$],    [严重伪影],   
    [Triplane + 邻近约束],    [$-5.3 tilde -7.3%$],    [自然],   
    [Token 空间优化],    [$-49 tilde -64%$],    [部分伪影],   
    [Token 空间 + LPIPS],    [$-42.5%$],    [伪影持续],   
    [仅低频优化],    [$-36.4%$],    [伪影],   
    table.hline(stroke:0.9pt),   
  )
) <tab:tto>

如表 D1 所示, 直接在测试时优化 Triplane 确实能大幅降低像素级的 L1 误差, 但渲染出的图像却出现了灾难性的视觉崩坏（如异常的金属高光和几何变形）。这完全印证了 §3.2 中关于等价解的结论：由于解码器存在多对一的映射关系, 在缺乏足够视角约束的情况下, 优化算法极易找到那些数值误差极低、但严重违背物理常理的错误特征解。即使在优化目标中引入感知损失（LPIPS）, 也无法消除这些伪影。这表明, 仅靠少量的训练视角渲染监督, 根本无法为解码器的反向求解提供足够的几何约束。

测试时优化的全面失败, 揭示了编码器在前馈架构中常被忽视的另一项核心功能：它不仅负责输出预测特征, 更重要的是, 它通过庞大的预训练数据, 为特征空间圈定了一个合理的分布范围。绕开编码器直接修改隐特征, 必然会导致数值脱离这一正常的分布轨道。因此, 要想安全有效地修正误差, 微调操作必须发生在编码器内部, 以确保修正后的特征仍然符合其原有的分布规律。这正是本文选择在训练阶段使用 LoRA 进行架构内干预的底层逻辑, 也彻底排除了事后直接修改特征的可行性。

#figure(
  caption:[测试时优化的失败示例. 从左至右:输入图像、TTO 后渲染（伪影明显）、原始 $T_"pred"$ 渲染、GT. ],   
  placement:top,   
  kind:image,   
  image("exps/latent_refine/503d24924ffa431690fa75fe37a60f17_render_comparison.png",    width:100%),   
) <fig:tto>
\

== E. 全量微调对比与样本退化分析 <sec:app-e>

我们以全量微调（$tilde$50M 参数全部可训练）作为对照组,   验证低秩约束的内禀正则化效应. @tab:fullft 是参数效率与过拟合风险的总览,   @tab:per-sample 给出 5 个验证样本上的逐样本 L1. 

#figure(
  caption:[表 E1. 全量微调与 LoRA-FreqLoss 的参数效率对比. ],   
  placement:top,   
  table(
    columns:5,   
    align:(left,    center,    center,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 4 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 4 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([方法],    [参数量],    [L1 增益],    [LPIPS 增益],    [是否过拟合]),   
    table.hline(stroke:0.4pt),   
    [全量 FT + Std MSE],    [$tilde$50M],    [$+16.5%$],    [---],    [是（楼梯样本退化 38%）],   
    [全量 FT + FreqLoss],    [$tilde$50M],    [$+19.6%$],    [---],    [中等],   
    [LoRA + Std MSE],    [295K],    [$+13.1%$],    [$+4.2%$],    [否],   
    [LoRA + FreqLoss],    [295K],    [$+16.2%$],    [$+9.3%$],    [否],   
    table.hline(stroke:0.9pt),   
  )
) <tab:fullft>

#figure(
  caption:[表 E2. 5 个验证样本上的全量微调 L1 表现. ],   
  placement:top,   
  table(
    columns:6,   
    align:(left,    center,    center,    center,    center,    center),   
    row-gutter:0pt,   
    stroke:none,   
    inset:(x,    y) => (
      top:if y == 0 or y == 1 { 5pt } else { 2.6pt },   
      bottom:if y == 0 or y == 5 { 5.4pt } else { 2.6pt },   
      left:if x == 0 { 0pt } else { 5pt },   
      right:if x == 5 { 5pt } else { 0pt },   
    ),   
    table.hline(stroke:0.9pt),   
    table.header([样本],    [基线 L1],    [Std-FT L1],    [Freq-FT L1],    [Std 变化],    [Freq 变化]),   
    table.hline(stroke:0.4pt),   
    [机械装置],    [0.0443],    [0.0439],    [0.0436],    [$+0.9%$],    [$+1.6%$],   
    [斧头],    [0.0140],    [0.0067],    [0.0069],    [$+52.3%$],    [$+50.7%$],   
    [楼梯建筑],    [0.0320],    [0.0441],    [0.0441],    [$-37.7%$],    [$-37.9%$],   
    [护目镜盒],    [0.0126],    [0.0092],    [0.0096],    [$+26.9%$],    [$+23.4%$],   
    [石头],    [0.0620],    [0.0463],    [0.0456],    [$+25.3%$],    [$+26.5%$],   
    table.hline(stroke:0.9pt),   
  )
) <tab:per-sample>

全量微调虽然在 L1 平均上略胜（$+19.6%$ vs LoRA + FreqLoss 的 $+16.2%$）,   但代价显著:楼梯建筑这一验证样本渲染质量退化 38%,   证明 50M 全量参数的更新中存在大量被解码器抑制但仍被 MSE 损失激励的高频伪梯度,   使训练分布外的样本被错误地破坏. LoRA-FreqLoss 用 1/170 的参数实现可比的 L1、更优的 LPIPS、且无任何样本退化——与 §3.3 的 Rank 消融结果（rank $>= 16$ 时过拟合显现）共同确认:低秩约束本身就是有效的正则化器. 

#figure(
  caption:[5 个验证样本上的渲染对比（基线 / Std-FT / Freq-FT / LoRA-Freq / GT）. LoRA-Freq 在所有样本上保持稳定改善而无退化,   全量微调在楼梯建筑样本上明显退化. ],   
  placement:top,   
  kind:image,   
  image("exps/finetune_freq_v2/render_comparison_labeled.png",    width:100%),   
) <fig:fullft-comparison>
\
