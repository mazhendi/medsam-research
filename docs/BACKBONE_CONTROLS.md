# 先建立可学习的强对照：解码器继续训练与多尺度残差头

当前监督控制器实验没有有效收益证据。本阶段暂停控制器，只比较特征接入与常规训练。多尺度残差头是实验对照，不是已确认的新颖模块。

## 四组结构

|mode|训练什么|输入信息|
|---|---|---|
|decoder|原基线掩码解码器与任务token，编码器/提示编码器继续冻结|原SAM2多尺度解码路径，不增加头|
|deep|冻结整个基线，只训练原规格256→32→1的1×1残差头|最深层64×64特征|
|multiscale|冻结整个基线，训练逐级融合残差头|SAM2解码器已经投影的32×256×256、64×128×128、256×64×64特征|
|pooled|与multiscale完全相同的残差头|先将每层池化到64×64，再插值恢复原层尺寸，去掉新增头输入的细尺度变化|

特征尺寸对应1024输入的官方SAM2.1 Hiera-Small，运行时会记录实际形状和通道。所有模式都对整图运行原基线，不做patch裁剪、不使用控制器、无额外标注。原基线本身已经使用高分辨率特征，修改的是新增纠错路径能否访问这些特征，不能说此前SAM完全没看高分辨率信息。

multiscale头将三层特征各投影到16通道，从64→128→256逐级上采样，与对应层相加并经过3×3卷积/GELU，输出单通道残差，再插值到1024加在初始logits上。最终仍有256→1024插值，没有增加原图RGB浅层分支，不能保证恢复所有极细结构。

残差输出层零初始化，deep/pooled/multiscale的第0轮均与原基线一致。pooled与multiscale结构/参数量相同，能比仅比较deep更好地隔离细尺度信息作用；pooled操作本身也可能改变特征分布，所以它是诊断性消融，不是完美无干扰的因果证明。

## 训练设置与可比范围

所有模式均从同一原始best.pt独立开始，BCE+Dice、同一数据划分、阈值、种子、batch size与epoch数。在线提取特征，不做内存缓存，避免不同模式使用不同精度的缓存。沿用配置中的AMP训练、AdamW权重衰减和梯度裁剪1；训练/验证指标均使用FP32推理及原图标注网格。

默认学习率：decoder=1e-4，新增头=1e-3；可用--learning-rate显式覆盖。由于可训练参数、学习率和反向计算成本不同，这些是合理强基线，不是严格等参数/等计算比较。需要正式方法对比时应分别进行预先约定的验证集调参并报告成本。deep也需在新入口重跑，不能把旧Adam/无裁剪缓存实验当作完全同配置对照。

decoder是继续训练原任务token和解码器，和原始基线的可训练参数集合一致；这是从旧权重开始的fresh optimizer训练，不是恢复原优化器状态。如果原best.pt来自第50轮，再训练50轮意味着大约累计100轮数据暴露。新增头也站在已经训练好的基线上，不能与从官方权重训练50轮简单视为等预算。

## 建议先跑两组，各50轮

仓库根目录运行，传入原始基线best.pt，不是之前的correction/granularity权重：

```bash
git pull
pip install -e ".[dev]"
medsam train-backbone-control --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_control_decoder --mode decoder --epochs 50
medsam train-backbone-control --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_control_multiscale --mode multiscale --epochs 50
```

Windows PowerShell可以直接使用以上单行命令，输出目录不能已经存在。这次不运行监督控制器，不需要重新从官方权重训练原基线。

第10轮先查看原图训练Dice/Recall/clDice与损失是否变化。decoder表现好而multiscale无效，说明额外头未胜过普通继续微调；multiscale表现好，则补齐两个诊断对照：

```bash
medsam train-backbone-control --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_control_pooled --mode pooled --epochs 50
medsam train-backbone-control --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_control_deep --mode deep --epochs 50
```

如50轮末训练和验证均持续改善，可将下一组完整比较统一设置100轮；当前无resume入口，应使用新输出目录从同一基线重新开始。若训练不动，不应盲目增加轮数；若训练提升、验证不提升，则研究泛化而非声称梯度断开。

## 输出

- provenance.json：实际特征形状、可训练参数量、优化设置、原始基线哈希及数据审计。
- initial_validation.json：更新前原基线指标。
- history.jsonl：每轮训练损失、原图训练与验证Dice/Precision/Recall/clDice/细中粗中心线召回、裁剪前梯度范数。
- summary.json：原基线与最佳验证模型指标、最佳轮次、时间和显存。
- per_image.csv、paired_deltas.csv：逐图指标和变化；保留dataset/group以便受试者内聚合。
- *_errors.png、*_mask.png：原图尺寸结果与错误叠加图。
- best_control.pt、last_control.pt：只保存训练参数及实验身份，仍需原基线best.pt与官方SAM权重。

第0轮也参与最佳模型选择，防止强行报告比原基线更差的训练后模型。最佳选择当前依据验证Dice；必须同时报告Recall/clDice，不能用Dice选优后的微升掩盖结构退化。若需要改变选模标准，应提前约定并统一重跑，不能按测试表现更换。

结构专用中心线指标主要用于血管，不应当作息肉/胸片的通用主指标。CHASE只有两名验证受试者，本阶段只能验证流程与初步现象，不能支撑稳定泛化结论。

## 固定模型评估

下面先用验证集检查导出的模型；等结构和超参数锁定后，才在独立测试集评估：

```bash
medsam evaluate-backbone-control --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --control outputs/chase_control_multiscale/best_control.pt --manifest data/retina/CHASE_DB1/val.csv --output outputs/chase_control_multiscale_check
```

评估审计训练/验证/外部数据重叠、核对基线哈希和评估设置，不做测试时更新。
