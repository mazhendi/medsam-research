# 区域监督精度控制器：可检验的研究原型

本模块检验：少量标注下，按训练集内部跨受试者更新效果选择区域监督精度，是否优于固定精确监督及普通元重加权？这是待验证的研究假设，不是已确认创新或性能提升。它不是此前的 patch 纠错模型，没有 detail/context 裁剪。

## 改了什么

保留已训练 SAM 2.1 基线并完全冻结，整图提取最深层图像特征和初始 logits。新增同一规格的分割适配头（256→32→1 的 1×1 卷积，中间 GELU），输出上采样的 logits 残差。输出层零初始化，因此第0轮结果与基线一致。

训练专用控制器把特征、当前前景概率及熵池化到8×8区域，经过小型1×1卷积网络，输出各区域像素监督比例 alpha。控制器不读取真值。默认 alpha=0.1+0.9*sigmoid(z)，初始0.55；0.1的下限是防止完全取消像素BCE的保守设计值，不是理论最优值。

每个区域有两种 BCE：

1. pixel：逐像素 BCE，在有效FOV内平均。
2. occupancy：先分别计算区域内平均预测概率与真实前景比例，再计算两者的 BCE。允许不同像素布局对应相同占比，因此它不能单独保留血管结构或精确边界。

整体损失：`0.5 * 区域平均[alpha*pixel+(1-alpha)*occupancy] + 0.5*全图soft-Dice`。所有对照都保留相同的全图soft-Dice，所以本原型只改变 BCE 部分的精度，并不是完全撤去模糊区域的像素监督。FOV为空的区域不参与区域损失；网格必须整除模型输入尺寸，保证不重叠分区。

区域等权平均与原基线按像素平均并不在部分FOV时完全相同，因此完整方法应与本入口的pixel对照比较，不能把改变适配头/优化器的收益归给控制器。

## 元训练：真正可微的虚拟更新

从同一dataset抽取不同group的训练图A/B（group应填写受试者或视频序列）。在A上计算控制器加权损失，使用`create_graph=True`计算适配头梯度，构造一次虚拟SGD更新；用虚拟参数在B上计算固定BCE+Dice，反传更新控制器。随后使用更新后的控制器权重，真正更新适配头。

虚拟与实际适配更新都采用无动量、无权重衰减的SGD，默认学习率0.01；控制器采用Adam，默认0.001。特征及控制器输入概率停止梯度，但通过虚拟参数更新保留控制器的元梯度。冻结基线不会更新。

A/B都是训练样本，不使用独立验证集更新控制器。每个dataset至少需要两个不同group；不能把同一儿童双眼当作不同受试者。不同dataset之间不做元配对，避免不同任务标签直接混用。group缺失会报错，而不是擅自视每张图为独立患者。

重要限制：原始基线已经见过所有训练组，B只对当前虚拟更新是未参与样本，不是对基线完全未见的受试者。B的标签也不是已知无噪声真值，不能因此宣称辨识了真实标注错误或观察者歧义。该目标测量的是相对于现有训练标注的迁移效果。

## 五种模式

|mode|含义|
|---|---|
|pixel|alpha恒为1，适配头只用精确BCE+Dice|
|fixed|alpha恒为0.5，固定混合两类BCE|
|curriculum|alpha从接近下限逐轮增加到1，不能回退|
|meta|区域控制器通过跨受试者元目标学习混合比例，可升可降|
|reweight|相同控制器及元训练，但只调整像素BCE的区域权重（有效区域权重均值归一化），不混入occupancy|

所有模式使用相同适配头、随机种子、训练图顺序和真实适配更新次数。meta/reweight还有额外的虚拟更新与query计算，不能声称五组计算成本完全相同；summary记录训练+验证用时，应另补等时间/等计算对照。控制器可逆仅表示结构允许权重升降；是否真正学到有用的回退，应检查training_weights.csv，不能预先声称已经实现有意义的课程。

## 先运行 pixel 与 meta

在仓库根目录，保留原SAM2权重及原基线best.pt。不要传入best_correction.pt。

```bash
git pull
pip install -e ".[dev]"
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_pixel --mode pixel --epochs 30
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_meta --mode meta --epochs 30
```

Windows PowerShell可用以上单行命令。输出目录不能已存在。原YAML无须修改。每组运行先提取一次冻结全图特征缓存到CPU内存，后续训练不重复运行SAM编码器；因此推理图仍为SAM全图+轻量残差头，没有局部高分辨率信息。当前CPU缓存采用FP32，1024输入、256×64×64特征时，每张图的特征/logits/目标/FOV合计约16 MiB，1000张约16 GiB；大数据集需要足够内存，后续应改磁盘缓存或分批提取。缓存阶段与适配训练阶段分别计时；GPU峰值字段明确只覆盖缓存完成后的训练/验证，不代表提取SAM特征的峰值。

完成后补齐其余三组：

```bash
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_fixed --mode fixed --epochs 30
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_curriculum --mode curriculum --epochs 30
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_reweight --mode reweight --epochs 30
```

## 输出与判断

- summary.json：最佳验证轮次、global/adapted原图指标、计时与数据/权重身份。
- initial_validation.json：检查零初始化是否与基线一致。
- history.jsonl：损失、验证指标、控制器元梯度范数、平均像素权重。
- training_weights.csv：每图每区域逐epoch权重；只有meta/reweight代表学习控制，其他模式是固定/课程值。
- validation_weights.csv：最佳控制器在验证图上的权重，仅用于分析，不参与预测或更新。
- per_image.csv 与错误图：原图网格指标及漏分/误分。
- best_granularity.pt / last_granularity.pt：轻量分割头与训练控制器；最佳选择允许保留第0轮恒等适配头。

当前summary含Dice、IoU、边界F1、Precision/Recall及clDice；逐图CSV另含细/中/粗中心线召回。对息肉、胸片，中心线分组指标不具有同等解释，不应作为通用主指标。

至少需要看到meta胜过pixel、fixed、curriculum及reweight，并在不同受试者划分、随机种子、样本数下复现。还要判断收益是否只是正则化或训练成本增加造成。若alpha几乎恒定、只会单调升高，或者reweight同样有效，就没有证据支持“区域可逆精度控制”的贡献。

只有两名验证受试者的CHASE实验只是初步排错；少样本主张还需要用一致的低样本协议重新训练每个基线，不能把已用更多数据训练的SAM基线带入真正的few-shot设置。

## 固定适配头评估

下列例子仍用验证集检查导出的模型。等训练设计和设置锁定后，再使用独立test.csv，不用测试结果挑模式/epoch。

```bash
medsam evaluate-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --adapter outputs/chase_granularity_meta/best_granularity.pt --manifest data/retina/CHASE_DB1/val.csv --output outputs/chase_granularity_meta_check
```

评估只加载冻结基线与分割适配头，不调用控制器，不更新参数，检查适配头绑定的基线哈希和评估设置，并审计训练/验证/外部数据重叠。

本轮实现不意味着证明新颖性；还需与可学习课程、元重加权、多粒度/部分标签监督等方法作系统比较。
