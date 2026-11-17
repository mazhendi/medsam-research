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

新版默认虚拟与实际适配更新都采用Adam，学习率0.001，不使用权重衰减或梯度裁剪。虚拟步骤读取真实优化器已有一阶/二阶矩和步数，对当前一步保留可微计算，不修改真实优化器状态；历史状态停止梯度。控制器同样采用Adam，学习率0.001。可用`--optimizer sgd --learning-rate 0.01`复现旧版无动量SGD。特征及控制器输入概率停止梯度，但通过虚拟参数更新保留控制器的元梯度。冻结基线不会更新。

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
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_pixel --mode pixel --optimizer adam --learning-rate 0.001 --epochs 50
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_meta --mode meta --optimizer adam --learning-rate 0.001 --epochs 50
```

Windows PowerShell可用以上单行命令。输出目录不能已存在。原YAML无须修改。每组运行先提取一次冻结全图特征缓存到CPU内存，后续训练不重复运行SAM编码器；因此推理图仍为SAM全图+轻量残差头，没有局部高分辨率信息。当前CPU缓存采用FP32，1024输入、256×64×64特征时，每张图的特征/logits/目标/FOV合计约16 MiB，1000张约16 GiB；大数据集需要足够内存，后续应改磁盘缓存或分批提取。缓存阶段与适配训练阶段分别计时；GPU峰值字段明确只覆盖缓存完成后的训练/验证，不代表提取SAM特征的峰值。

完成后补齐其余三组：

```bash
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_fixed --mode fixed --optimizer adam --learning-rate 0.001 --epochs 50
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_curriculum --mode curriculum --optimizer adam --learning-rate 0.001 --epochs 50
medsam train-granularity --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_granularity_reweight --mode reweight --optimizer adam --learning-rate 0.001 --epochs 50
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

## 先决检查：pixel几乎不学习时，暂停控制器结论

如果pixel模式30轮的固定分割损失和Dice几乎不变，先运行以下**仅训练集的小样本拟合诊断**，不要继续用弱适配头判断meta是否有效：

```bash
medsam probe-adapter --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --output outputs/chase_adapter_probe --steps 100 --samples 2
```

固定两张训练图组成完整小批次；各组使用相同初始化和相同数据，分别检查SGD学习率0.01/0.1/1.0、Adam学习率0.001。输出summary.json、probe_history.csv、per_layer.csv，包括：

- 原始特征尺度、固定BCE+Dice损失、模型输入网格上的训练Dice；
- 每步参数梯度、真实更新量、相对初始化的漂移；
- 预测logits残差幅度，区分“参数变了但预测几乎没变”；
- 数值发散的显式错误记录，而非推荐使用发散学习率。

probe指标是选中训练小批次的拟合情况，Dice为合并像素计数的结果，不是原图验证集宏平均，不能与summary中的验证Dice直接比较。它不读取验证结果选最优项，也不会修改基线或正式训练默认优化器。

解释应基于结果：

1. SGD弱、Adam或更大SGD步长明显学动：优先排查优化尺度，下一版需同步修改真实/虚拟更新并重新做元梯度检查，不能只把实际SGD替换为Adam。
2. 几组都难以拟合：继续检查64×64深层特征和1×1残差头的表示能力，以及冻结特征是否包含所需信息；优化器扫描本身不能证明是哪一项限制。
3. 训练能拟合、验证不提高：再讨论泛化、过拟合与控制器，而不是把验证无提升等同梯度断开。

零初始化使第一个更新中前层卷积梯度为零是预期行为；后续是否获得非零梯度和有效更新才关键。已有单元测试验证了这一过程，但合成数据可学不意味着真实已收敛基线仍有可利用的表示空间。

更新后的train-granularity额外记录initial_training.json、逐epoch固定像素BCE+Dice训练目标、适配头梯度/真实更新量、预测残差、alpha的标准差和范围。原始train_loss随alpha改变了目标，不适合作为meta跨epoch学习进度的唯一依据。1e-8的元梯度非零不代表控制器学到有价值的选择，Adam也可能积累这种梯度而造成alpha漂移；平均alpha接近1可能只是在回到像素监督。


## 优化修正与轮数建议

旧版默认SGD 0.01在用户已训练基线上的残差更新几乎为零。因此新版先提供Adam作为独立优化对照，保持适配头结构和所有监督目标不变；这不是已经证实的完整修复。尚未取得用户真实权重上的拟合诊断，不能排除64×64特征的表示瓶颈。

- 先运行`probe-adapter --steps 200 --samples 2`。这是200次小批次参数更新，不是200个epoch；先确认训练损失、实际参数漂移和预测残差明显变化。
- 然后只运行pixel模式50轮。CHASE16张训练图对应800次真实适配更新。第10轮就检查`training_fixed_objective`、适配更新范数和logits残差；若仍几乎不变，不应靠加到200轮解决。
- pixel可学习后再跑meta50轮；同一批对照使用相同优化器、学习率、种子和轮数，且记录额外元训练开销。
- 若50轮末训练和验证仍在改善，可以预先把下一组完整对照预算定为100轮（1600次真实更新）；目前没有resume入口，需新输出目录从同一基线初始化，不可将前后预算混作公平比较。
- 若训练损失下降、验证早已平台或下降，则不延长。已有best checkpoint保留第0轮候选和整个预算内的最佳验证轮次，测试集保持锁定。

Adam并不保证控制器更有效：在梯度符号不变的首次更新中，自适应归一化可能减弱其对损失权重幅度的敏感性，因此即使分割头学动，元梯度仍可能很弱。代码没有人为放大元梯度掩盖问题；若出现该情况，应研究元目标与虚拟更新的可辨识性，而不是把alpha漂移当成功。新增测试覆盖零初始化有限梯度、已有动量状态下虚拟/真实Adam更新一致性，以及控制器超梯度有限差分校验。

probe的summary新增`supervision_gradient_contrast`，比较固定占比/像素监督（两者都保留Dice）的梯度范数、夹角及差值。此诊断只检查两个选项是否在初始头上可区分，不证明它们都能提高泛化。
