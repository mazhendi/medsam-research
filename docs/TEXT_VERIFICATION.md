# 两阶段文本定位与方向性修复：实现与运行

状态：可运行的研究原型；不是Simple-ViLMedSAM的IPP/BID复现，不是已经验证的新颖方法或精度提升。现有基线、控制器及backbone-control命令保持独立。

## 实际结构

- 原始自动SAM2.1基线全部冻结，单次前向输出视觉预测和三尺度特征。
- 官方BiomedCLIP图像/文本编码器全部冻结。提取ViT的14×14上下文化patch tokens，通过其预训练图像投影和文本向量计算余弦相似度。它是空间语义线索，不是经过逐像素标注校准的概率，也不是IPP信息瓶颈归因图。
- 文本定位头：三尺度投影、patch响应与文本条件融合，逐级输出logits残差，加到冻结视觉基线。输出层零初始化，初始等于视觉基线。
- 修复头：融合多尺度图像特征、视觉/文本预测、差值和熵，预测恢复/抑制logit残差。文本先验背景区域执行恢复，前景区域执行抑制，初始严格保持文本定位预测。

文字只有类别词时也可产生逐图不同的patch响应，但定位头可能忽略文字或只记住任务。因此自动评估同时输出清空相似度和文本向量的text_null对照，以及预测变化量。text_null只隔离条件输入，不是与训练充分的纯视觉模型完全等价的对照。

输入SAM使用原项目的1024方形缩放；BiomedCLIP对同一个整图视野缩放到224并使用官方mean/std，不做中心裁剪以避免空间错位。这与官方分类预处理的中心裁剪可能不同，已经记录在provenance。patch readout是预训练投影在patch tokens上的应用，不意味着BiomedCLIP原本就有精确密集分割能力。

## 1. 安装文本可选依赖与下载权重

原项目的SAM2依赖、官方SAM2权重和你训练的原始best.pt继续保留。仓库根目录：

```bash
git pull
pip install -e ".[dev,text]"
medsam prepare-biomedclip --output checkpoints/biomedclip
```

下载官方模型约748 MiB，另外下载BERT配置/词表。固定版本：BiomedCLIP `9f341de24bfb00180f1b847274256e9b65a3a32e`，tokenizer `d673b8835373c6fa116d6d8006b33d48734e305d`。bundle.json记录来源和逐文件SHA256；加载时验证文件身份。不将这些权重提交Git。

下载目录必须不存在；如果完整目录已在本地，跳过下载命令。网络中断留下不完整目录时，先使用新的输出目录重试并在后续命令指定对应路径；程序不自动删除用户目录。下载需要Hugging Face网络访问，准备完整bundle后加载不需要联网。

固定open_clip_torch 2.32.0，transformers 4.45–4.49与timm 1.0系列。本模块不加载远程自定义代码；权重按weights_only模式加载。已有CUDA版torch满足依赖时应保留原环境，不为本模块改装CPU版torch。

## 2. 第一阶段：文本定位与互补性测量

先按[逐图描述说明](IMAGE_CAPTIONS.md)生成并检查prompts.json。建议先用`--epochs 1`在新输出目录做本机运行检查，然后正式运行：

```bash
medsam train-text-localizer --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --biomedclip-dir checkpoints/biomedclip --prompt "retinal blood vessels" --prompt-map outputs/chase_image_prompts/prompts.json --output outputs/chase_text_localizer --epochs 30
```

- 仅训练文本定位头；基线和BiomedCLIP不更新。
- 全部监督来自原训练掩码，BCE+Dice，AdamW lr=1e-3，weight_decay沿用配置；head宽度默认16，可用--width修改。
- 为避免冻结特征精度和分支差异造成难以定位的问题，当前训练使用FP32，不启用配置中的AMP。batch_size/worker/seed沿用原配置。4090实际显存峰值需要用户本机确认，不能从CPU检查推断。
- 每个epoch报告原图网格训练/验证指标，所有模式保持原阈值和边界容差。包含visual、text、text_null。
- `--prompt-map`传入逐图JSON `{ "Image_04L": "..." }`，必须覆盖训练和验证ID，缺失时报错；训练文本全部相同时拒绝逐图实验。统一任务词对照需明确指定`--task-only`。映射及来源记录嵌入checkpoint并用于第二阶段和预测；新图需要另传覆盖新ID的`--prompt-map`。不能使用测试掩码或测试错误图生成提示。

`complementarity.csv`按图记录：文本预测的FN/FP、视觉分支能正确覆盖/排除的比例，以及清空文本条件后的概率L1差和掩码差异。只有一个类别、全部文本相同时打乱文字不会改变实验，不是有效消融。

**检查点：如果最佳模型停在第0轮，或者text与text_null几乎相同，先不要把第二阶段成绩解释为文本方法的贡献。** 进一步检查表示能力、优化与语义响应质量。互补率只表示相对于现有标注的潜在可修复空间，不意味着真实临床错误或分支统计独立。

## 3. 第二阶段：冻结两路，训练方向性修复

确定第一阶段有可用信号后运行：

```bash
medsam train-text-repair --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --biomedclip-dir checkpoints/biomedclip --localizer outputs/chase_text_localizer/best_localizer.pt --output outputs/chase_text_repair --repair-mode full --epochs 30
```

第二阶段自动继承第一阶段的提示策略、宽度和通道数；冻结其定位头。只更新修复头，训练中不更新图像/文本编码器或视觉基线。

令文本logits为zT，H=stopgrad[zT>=logit(threshold)]，修复头输出dR,dS：

- 背景操作域：恢复logits zR=zT+dR。
- 前景操作域：抑制logits zS=-zT+dS。
- 最终logits：`where(H, zT-dS, zT+dR)`。

这等价于设计中的条件概率合成，并避免显式logit(sigmoid)的数值误差。零初始化严格保持原先验。定向监督仅在对应操作域上计算：恢复分支拟合Y，抑制分支拟合1-Y；空操作域贡献为0，FOV外不计入损失。

**数学边界：H固定时，这个定向损失与最终logits的分域归一化BCE代数等价，不能声称提供独立的新监督信息。** 因此增加balanced-residual控制：同样的分域损失、同样的定位干预，只去掉方向性参数路由。新颖性不能单靠这一损失或恢复/抑制命名。

训练还将与真值位置无关的随机矩形在先验logits上置为-6或+6，模拟局部遗漏/假激活。图像、目标文字和视觉特征不变，重新计算操作域及修复标签。原始/干预两视图的监督损失取平均，加上最终概率一致性；默认direction-weight=0.5、consistency-weight=0.1，属于待验证超参数。只改变定位先验，不把“另一个目标的文字”错误地监督为同一掩码。

## 4. 消融

第二阶段--repair-mode支持：

|mode|方向性监督|定位干预与一致性|输出合成|
|---|---|---|---|
|full|有|有|互斥操作域|
|no-intervention|有|无|互斥操作域|
|no-direction|无|有|互斥操作域|
|residual|无|有|普通有符号残差 zT+dR-dS|
|balanced-residual|等价的分域归一化BCE|有|普通有符号残差 zT+dR-dS|

五组使用相同两通道输出头和参数量。residual的两通道线性组合与条件合成仍有不同参数化/梯度，不能视为完全等优化难度；no-intervention也比成对训练少一次修复头前向。先验干预复用冻结特征，不重复编码图像。

可在固定第一阶段权重后逐组运行，输出目录分别命名。比较完整方法是否超过visual、text和普通residual，而不仅比较是否超过较弱的文本先验。当前未实现其他论文的复现，不能声称已与IPP/BID或RELiF公平比较。

## 5. 固定权重评估与无标注预测

先在验证集检查导出一致性；设置锁定后再将manifest换为独立测试集：

```bash
medsam evaluate-text-model --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --biomedclip-dir checkpoints/biomedclip --localizer outputs/chase_text_localizer/best_localizer.pt --repair outputs/chase_text_repair/best_repair.pt --manifest data/retina/CHASE_DB1/val.csv --output outputs/chase_text_eval
```

不传--repair则只评估第一阶段。加载核对SAM基线、BiomedCLIP及第一阶段文件哈希、阈值/掩码约定/边界容差，审计训练、验证与外部数据重叠。不执行测试时参数更新或先验干预。

无标注预测：

```bash
medsam predict-text-model --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --biomedclip-dir checkpoints/biomedclip --localizer outputs/chase_text_localizer/best_localizer.pt --repair outputs/chase_text_repair/best_repair.pt --manifest data/retina/CHASE_DB1/val.csv --output outputs/chase_text_predictions
```

predict仅读取id/image及原图，不读取manifest的mask/FOV列；即使它们指向不存在的文件也不需要。输出原始尺寸掩码及ID/提示映射。它没有验证集真值驱动的自动选区。

## 输出与解读

- best_localizer.pt / last_localizer.pt；best_repair.pt / last_repair.pt：轻量训练参数及身份绑定，仍需原SAM基线和两套官方权重。第0轮参与最佳选择。
- summary.json、history.jsonl、provenance.json、initial_validation.json：训练/验证结果、模型配置和身份、依赖版本、时间及GPU显存；selected_identity标记最佳是否为第0轮。
- diagnostics.json：FN/FP互补比例和清空文本后的变化汇总；比例为像素池化统计，不是受试者级置信结论。
- per_image.csv：visual/text/text_null/repaired逐图结果；原图Dice、IoU、边界F1、P/R、clDice、中心线宽度分组。
- complementarity.csv：两路错误互补及文本条件敏感性。
- repair_transitions.csv：相对文本先验修复了多少FN/FP，又制造了多少新FN/FP。
- *_errors.png：红色漏分、蓝色误分、绿色正确；*_probability.png为固定0–1映射到8-bit灰度，便于可视化，不能当无损概率文件。
- *_patch_cosine.png：原始patch余弦[-1,1]按固定线性范围映射显示，不是误差图或校准置信度。
- *_recovery_probability.png / *_suppression_probability.png：仅展示对应操作域内的条件错误概率；两个residual模式不输出这两张图以避免误解。
- images.csv：编号、样本ID和实际使用文本。

零文本条件仅用于诊断；性能不变不必然证明全部文本无用，但与概率变化量一起可帮助识别忽略条件的退化。图文条件路径若被忽略，或者修复只等同于增加视觉容量，则不支持方法主张。中心线指标仅主要用于血管，不应当作息肉和胸片的一般主指标。

## 当前验证范围

- 单元/流程测试使用显式标记的小型视觉网络和文本测试夹具，验证恒等初始化、操作方向、空域损失、梯度隔离、确定性干预、身份绑定与不读标注的预测；它们不验证BiomedCLIP的语义效果。
- 另用真实SAM2.1官方权重、官方BiomedCLIP和CHASE图像做CPU端到端训练/保存/重载/评估/预测检查。该检查基线是官方权重加随机任务token构造的API基线，不是用户已训练best.pt，因此其数值不能当真实实验结果。
- 未验证用户4090上的显存、速度和实际增益；未完成跨数据集、近邻方法复现或论文新颖性证明。

第一轮建议先报告第一阶段的summary/history/complementarity与代表性语义响应图，再决定第二阶段投入，避免再次在缺少有效先验信号时扩大消融。

## 高分辨率采样对照

新增冻结BiomedCLIP重叠裁块模式，保持SAM和定位头不变。参见[TILED_EVIDENCE.md](TILED_EVIDENCE.md)，先运行1轮，再做正确描述与统一任务词对照。
