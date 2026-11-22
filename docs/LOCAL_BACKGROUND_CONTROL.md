# 局部正样本与难背景学习：训练对照

本功能用于检验背景判别是否是瓶颈，不是新增网络模块，不声称创新成立或保证提分。
现有SAM2-UNet、原生基线、SCDS/SBCA/RCSR、缩小增强配置均保留。

## 实际做什么

每张原训练图产生三个训练视图：

1. 完整图及原标签。
2. 包含全部GT包围区域、带随机上下文的局部图及对应标签；裁剪宽高为GT包围框的1.2–2.5倍，受原图边界限制。不是只保留目标、去掉全部上下文。
3. 纯背景局部图，全零标签。背景候选宽高分别为原图15%–40%，不得与GT及保护区相交；保护半径为短边的1%，至少2像素。候选全部来自训练图。

均缩放到352、按相同规则随机翻转、ImageNet归一化；mask沿用bicubic软目标。背景纯度按训练GT检查，不能保证标注之外不存在未标注病灶。

每张图最多尝试80个位置以获得4个背景候选。不足4个但至少找到1个时重复合法候选；一个都没有时，第三视图回退为**原完整图＋原标签**，绝不强行将目标标成背景。正样本resize后若前景核心消失，也回退整图。history记录背景有效数及回退次数。

## 三个独立配置

| 配置 | 第三个视图的选择 | 输出 |
|---|---|---|
| native_local_random.yaml | 同一随机候选池的首个候选 | runs/native_local_random |
| native_local_hard.yaml | 当前模型预测平均前景概率最高的候选 | runs/native_local_hard |
| native_local_matched_full.yaml | 三个梯度视图均使用同一完整图 | runs/native_local_matched_full |

hard组前2轮使用随机选择，之后开始hard选择。候选坐标由seed、epoch、源图index确定，不依赖模式或worker调度；相同源图/epoch的候选池在random/hard间一致。难度由各组当前模型在线计算，不使用外部测试结果或测试标签。

**计算预算明确对齐：** 所有组都对全部4个候选评分（eval、no_grad，每次物理batch16），再进行3次顺序的梯度前向/反向；三个损失各除3，最后只执行一次optimizer.step。三组每步均7次前向、3次反向，20轮均1820次optimizer更新。每步16张源图形成三视图，不等于48张独立随机图像。

因此这些实验明显比旧单整图训练更贵。不能把local组与旧基线的差距全部归因于难例挖掘：**核心比较是hard vs random**。matched_full仅作附加计算预算控制，三个完整图视图相同、没有新增独立信息，在确定性模型下平均梯度应接近一次整图训练。

## 损失与模型

三个模块全关，仍用Hiera-L＋Adapter＋原生Mask Decoder，不增加网络参数。
完整图/正局部图用weighted BCE＋weighted IoU。合法纯背景仅用BCE；回退整图使用原结构损失。三视图等权、保持完整图与正样本监督，避免只压低前景而伤害召回。没有额外对象存在性头、对比损失或GT推理提示。

固定1450源图、352、batch16、20轮、FP32、AdamW lr1e-4，cosine、seed42、固定末轮；从原官方sam2_hiera_large.pt初始化，不加载测试挑选权重。默认不叠加缩小增强。程序拒绝混用模块/缩小配置，以及用train-native-sam2误跑local配置。

## 运行

先拉取代码并做短探针（使用真实训练图与官方权重）：

```bash
git pull
pip install -e '.[dev]'
medsam probe-local-background --config configs/native_local_hard.yaml --steps 2 --batch-size 1
```

探针直接测试hard选择与三视图更新，报告loss、回退数、前后向次数、CUDA峰值；不代表完整训练效果，也不执行训练轮数的warmup。batch16的显存需要本机验证。

分别训练随机和难背景两组：

```bash
medsam train-local-background --config configs/native_local_random.yaml
medsam train-local-background --config configs/native_local_hard.yaml
```

可选附加控制：

```bash
medsam train-local-background --config configs/native_local_matched_full.yaml
```

数据仍读取data/polyp_full/train_all.csv，程序自动裁剪，不生成需要手工整理的新图片。输出目录必须不存在。checkpoint记录配置、来源manifest哈希、更新步数；未实现断点续训。日志中selected_background_score仅统计合法背景，不包含回退的完整图。

测试不裁图、不做候选选择，沿用原预测命令：

```bash
medsam predict-native-sam2 --checkpoint runs/native_local_hard/last.pt --manifest /path/to/ETIS.csv --output outputs/native_local_hard_etis --normalization raw
```

比较Dice、目标面积分桶Recall、FP/GT、完全无重叠率。Precision上涨但Recall明显下降不算成功。推理成本与原生基线相同；训练成本增加。

这些配置为预先固定的诊断对照，不自动搜索采样超参。开发时应在训练池内部另立开发协议；外部测试只能作为探索性观察，不能从ETIS/ColonDB抽难例加入训练或反复依其调整配置。

2026-11-22验证：全套111项测试通过，新文件ruff检查通过。覆盖保护区排除、全前景图回退、正样本完整性、候选确定性、同预算前后向、负样本BCE梯度；真实Hiera-T及原生SAM2解码器的352三视图前后向通过（随机初始化，仅用于集成测试）。尚未验证本机官方Hiera-L权重加载、batch16显存或完整训练精度。
