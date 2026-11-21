# 息肉分割新增全文阅读：问题、算子、协议与证据

日期：2026-11-21。本轮从出版方/会议官方地址获取并提取7篇全文，阅读方法、实现、结果和消融部分；录用以会议官方页面为依据。下面的数值均为作者报告，未在本项目复现。MICCAI和WACV是会议，不能称作SCI一区期刊。本报告补充此前已读的DSCNet、DEPR-Net等期刊文献。

## 1. MEGANet，WACV 2024

**问题。** 低对比边界与尺度变化；低层细节沿金字塔反复平滑会衰减。

**真实计算。** EGA输入是当前编码特征、上一级预测和图像高频图。上级预测生成反向注意力和Laplacian边界注意力；图像Laplacian提供第三条高频证据。分别调制编码特征，再拼接卷积、学习注意力及CBAM细化。一个重要细节：从浅层高频图直接缩小得到各尺度高频证据，避免每级重新高斯平滑持续损失边缘。

**训练。** 标准900+550训练池；352；batch16；200轮；SGD lr1e-3、momentum.9、wd1e-5、poly调度；水平/垂直翻转、旋转、.75/1/1.25多尺度；多层BCE+Dice监督；RTX3090。

**证据与局限。** 消融分别检查反向注意力、边界注意力、高频分支与CBAM，并比较高频图生成方式。作者也指出Kvasir背景复杂，高频图会带入噪声，去掉该分支的版本在该集反而最好。因此“高频越多越好”不成立。

**对我们。** 任何细节保留方案应同时检测黏膜假阳性；简单边缘图拼接已不是新贡献。

全文：https://openaccess.thecvf.com/content/WACV2024/papers/Bui_MEGANet_Multi-Scale_Edge-Guided_Attention_Network_for_Weak_Boundary_Polyp_Segmentation_WACV_2024_paper.pdf

## 2. PolyMamba，MICCAI 2025

**问题。** 固定扫描的空间感知偏置，以及计算中局部高频细节减弱。

**真实计算。** PVT特征作为空间先验，调制VSS/SS2D输出；随后1×1卷积、ReLU与残差处理。DFE对浅层特征FFT，以两个不同Gaussian频率滤波器的差取出特定频带，IFFT回空间后调制上采样的深层特征。区别于一般“加Mamba”，它给出了空间先验与频率补偿两项明确职责。

**训练。** 900+550；AdamW lr1e-4、wd1e-4；100轮、batch8、RTX3090；旋转、翻转、color jitter；weighted BCE+IoU。文中352用于示例/分析，完整输入与多尺度执行细节仍应在源码中核验，不凭示例补齐超参。

**证据。** Table2从无先验/无Gaussian的ClinicDB .916、ETIS .786，提高至完整模型 .940/.829，分别加入组件也有正收益。Table1五集Dice依次为Kvasir .919、ClinicDB .940、CVC300 .904、ColonDB .815、ETIS .829。

**局限。** 模块组合消融不等于独立证明“扫描偏置/频谱衰减”是全部误差来源。论文首页给出的Parker-rfu/PolyMamba仓库在本轮API请求返回404，不能称已获得实现。

全文：https://papers.miccai.org/miccai-2025/paper/3872_paper.pdf

## 3. SAM-MaGuP，MICCAI 2025

**问题。** SAM迁移时的弱边界、息肉与黏膜混淆。

**真实计算。** 3/5/7卷积构造多尺度表示，1D与2D Mamba建模通道/空间关系；边界蒸馏BDC在训练中用GT分离前景和背景特征，以前景为Q、背景为K/V做cross-attention，再将增强表示蒸馏回编码表示。BDC推理时移除；另有预测pseudo-mask作为提示。阶段一适配编码特征，阶段二联合适配器与解码器。

**训练。** 标准1450训练；352；.75/1/1.25；Adam lr1e-5、batch8、200轮、A100。需进一步确认两阶段各自轮数、实际SAM规格及输入适配方式；不能仅依据背景介绍中的ViT-H/16就锁定运行骨干。

**证据。** 完整模型报告Kvasir/ClinicDB/CVC300/ColonDB/ETIS Dice .947/.953/.927/.859/.854。Table3渐进添加MSD、1D/2D Mamba和BDC；最后BDC使ETIS .849→.854，不能把全部收益归给蒸馏。

**可复现性。** 查到官方仓库，但树c1a7507de8b6c00b583e24309f584fef8a9776f7仅README和架构图，无Python文件。作为强结果参照值得重视，尚不能据此安排“直接运行原版”。GT参与训练不等于测试泄露，关键是推理时不使用GT支路。

全文：https://papers.miccai.org/miccai-2025/paper/3165_paper.pdf

官方：https://papers.miccai.org/miccai-2025/0522-Paper3165.html

## 4. ESPNet，MICCAI 2025

**问题。** 新中心、扁平/小息肉、IBD炎症背景中的泛化。

**真实计算。** DeiT编码；相邻特征token通过注意力和GCN增强；基于FSPNet的feature-shrinkage decoder构造mask/edge双金字塔。边缘特征在多个尺度拼入mask特征，CBAM融合，尺度间残差保存信息；四尺度mask结构损失及edge BCE监督。

**数据。** 不是常见1450训练：804 Kvasir、1182 PolypGen C1–C5、2014 SUN-SEG，共4000张。SUN样本为每正例约20张非连续帧。测试Kvasir-Sessile、SUN-SEG-Easy Seen、ETIS、ColonDB、PolypGen-C6以及95张私有IBD图像。视频来源需要患者/序列划分审计，不能仅随机按帧拆分。

**训练。** 384，patch16，100轮、batch8；Adam lr1e-4、50轮降十倍；翻转/旋转；作者报告4张L40S48GB约9小时。这不是单张4090可直接复用的耗时承诺。

**证据。** ETIS baseline Dice .763，加Edge FSD .809，加CBAM .807，加残差完整 .827；表明额外注意力并非每步都涨。ColonDB .770→.783→.794→.802。

**对我们。** 是研究泛化协议与细节保存的好参照。不能把它的4000图成绩放入1450图表格，直接宣称结构更强或更弱。官方仓库有8个Python文件与训练入口，但未完成运行验证。

全文：https://papers.miccai.org/miccai-2025/paper/4428_paper.pdf

官方：https://papers.miccai.org/miccai-2025/0307-Paper4428.html

## 5. CD-PolypNet，MICCAI 2025

**问题。** SAM浅层特征适配不足，以及弱边界。

**真实计算。** SSFD利用受监督解码层特征指导早期编码层，并加入层内约束。EFB计算图像Canny边缘，编码特征经7×7 Gaussian(σ3)得到低频，作差为高频；高频与Canny图拼接，经1×1–ReLU–1×1–sigmoid产生门控；边界增强分量与3×3语义分量以iAFF融合。

**训练。** 1024 bicubic，[0,1]；Adam lr.01，batch4，30轮，每5轮lr乘.8；水平翻转和±15°旋转；RTX3090。划分段同时出现80/10/10与PraNet固定测试集描述，具体1450池如何进一步划分必须从数据清单查证，不能自行推定。

**需核验。** 消融表的部分Dice/IoU组合在文本提取中出现IoU高于Dice；在相同二值掩码、同样逐图聚合下应有Dice≥IoU，需回原表及代码确认，不能照抄后参与排名。层内归一化/排序公式的可实现性也值得核验，不能只接受“去相关”名称。

**源码。** 官方树99b5faddfcb813ca606a2f79206eed3709bf3f65含29个Python文件，存在SSFD和EFGConv实现。比空仓库更具复现条件，但尚未完成内容逐式/前后向核验。

全文：https://papers.miccai.org/miccai-2025/paper/2291_paper.pdf

官方：https://papers.miccai.org/miccai-2025/0139-Paper2291.html

## 6. Lite-π，MICCAI 2026

**问题。** 轻量分割网络缺乏跨域表征能力，多个基础模型直接部署成本高。

**真实计算。** 学生深层特征经1×1投影与卷积编码，三条小解码支路分别重建SAM、DINOv2、OneFormer特征；这些是逐图空间特征重建，不是统一任务词，也不是单纯文本余弦图。三个生成表示拼接、1×1、Transformer、自学生latent残差、Mamba再送原分割解码器。基础模型作为训练监督来源，推理走学生生成支路。

**训练。** 标准1450；352；多尺度.75/1/1.25，水平/垂直翻转；Adam lr1e-4，batch6，100轮，V10032GB；BCE+Dice及重建项，重建权重按30/60轮分为.1/.3/.5。

**证据与限制。** 多基础模型消融并不在每一列单调提升：PraNet的2-FM→3-FM，ColonDB 76.1→75.9、ETIS 73.7→72.9，尽管源域提高。论文将UFMI作为统一模块，未充分拆分重建、Transformer、Mamba各部分的独立贡献。不能概括为“三教师总是最好”。

**源码。** 官方树8f5fe697d040843f9bf37e2144ea70f85f0bb4f5共5文件、0个Python文件。当前不能直接运行原版。

全文：https://papers.miccai.org/miccai-2026/paper/5591_paper.pdf

官方：https://papers.miccai.org/miccai-2026/0495-Paper5591.html

## 7. Polyp-AFMamba，MICCAI 2026

**问题。** 空间扫描方向、频率偏置及不同区域/尺度所需的解码操作不同。

**真实计算。** 通道分四组进行不同起点的二维扫描；块DCT与样本相关top-k频率掩码；训练时5%掩码扰动。三通道辅助头预测前景/边界/背景，低频与深层特征经Mamba生成动态卷积核；三类区域分别调制特征，在dilation1/3/5/7下解码后融合。区域先验以GT派生监督训练，推理应该使用预测先验。

**证据。** Table3有三模块组合消融，完整ClinicDB .946、Kvasir .932、ColonDB .819、ETIS .825、EndoScene .917。不能据小目标可视化单独证明全面的小目标提升，仍需面积分桶。

**复现疑点。** 所读全文实验段未明确列出epoch/batch/lr，不能自行补齐。按重要性top-k定义的LF/HF是否严格对应物理低/高频，需要实现解释；top-k路径如何训练、两张扰动mask是否保持互补也需验证。不能照图写出相似代码后称精确复现。

**源码。** 官方摘要链接的仓库树faaeea7901de399237b6869c9ac50326a87ba0e1目前只有README，无Python。

全文：https://papers.miccai.org/miccai-2026/paper/4551_paper.pdf

官方：https://papers.miccai.org/miccai-2026/0032-Paper4551.html

## 汇总：哪些想法已被覆盖

| 宽泛设计 | 直接相关先例 | 不能单独据此宣称创新 |
|---|---|---|
| 浅层高频指导边界 | MEGANet、PolyMamba、CD-PolypNet | Laplacian/FFT/Gaussian拼接 |
| SAM加Mamba适配 | SAM-MaGuP | 替换一个注意力层或加SS2D |
| 前景/背景/边界分解 | Polyp-AFMamba及此前PraNet-V2的前背景建模 | 三分支命名与门控 |
| 边缘/区域双解码 | ESPNet | 只加边界监督和CBAM |
| 多基础模型迁移 | Lite-π | 把多个骨干输出拼接 |

## 对本项目的具体判断

1. 当前较强的论文结果已有ETIS .82–.85甚至更高报告。不能满足于在弱基线上增益.003；同时不能忽略数据协议、选模及源码空缺，直接将最高报告当成可信可复现最优。
2. 主基线尚不锁死：SAM2-UNet与Polyp-PVT是可落地起点；新增优先审计CD-PolypNet的真实算子/指标一致性，MEGANet作为边界对照。SAM-MaGuP是强参考但缺代码；ESPNet应作为多中心协议独立实验。
3. 不再提出“多尺度+注意力+频率+区域形状”的宽泛组合。待检验的问题是：粗尺度语义错误时，细尺度证据能否推翻错误更新，而不是被持续压低。必须先在强基线上做层间错误传播诊断，再论证某种更新规则的必要性。
4. 关键实验至少包括：同骨干原版、普通参数量匹配融合、仅小目标采样/缩放增强、拟议结构；报告逐图漏检、小目标Recall/Dice、边界误差、参数和时间；在同一源域验证规则下锁定配置。GT面积只能做训练监督/事后分桶，不能在测试推理中提供门控。
5. 目前还没有证据证明上述更新规则本身新颖或有效。后续应查对应方向的跨尺度更新、竞争融合、细到粗反馈文献，明确区别后再设计模块和绘图。

本轮完成文献阅读和公开仓库存在性检查，没有完成以上模型的GPU复现，也没有改动当前算法。原文提取与仓库树记录保存在工作区 `polyp_baseline_audit_20261121/deep_reading/`。
