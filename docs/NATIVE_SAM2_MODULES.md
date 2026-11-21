# 原生SAM2解码器：SCDS / SBCA / RCSR实验实现

本分支是**待验证的研究原型**，不是此前Dice 0.8450的SAM2-UNet，也不宣称已经证明新颖性或精度提升。原SAM2-UNet及所有训练权重保持独立。

## 实际接入方式

```text
归一化图像 352×352
  → SAM2 Hiera-L，原权重冻结，各block前Adapter可训练
  → 四级特征：88/44/22/11
  → [SCDS]：F3/F4作结构参照，分别增强F1/F2，原通道数不变
  → 原生FPN neck：移除最粗11网格，保留88/44/22
  → 主图像嵌入22×22×256；原生conv_s0/s1投影高分辨率特征
  → 原生Two-Way Transformer
      [SBCA]：包裹第一层Token→Image注意力，仅修改单掩码Token的更新
  → 原生第一层转置卷积：22→44，256→64
      [RCSR]：区域Token＋44×44高分辨率特征指导重采样
  → 原生高分辨率相加、归一化、激活
  → 原生第二层转置卷积及88网格高分辨率融合
  → 原生mask-token MLP生成动态权重并投影掩码
  → logits上采样到352；推理再恢复原图
```

三个模块均有独立开关。没有RFB或U形解码器，没有边界Token，不使用测试GT、人工点框、IoU oracle或对象存在性门控。自动任务提示为4个可学习嵌入，直接作为稀疏prompt embedding传入解码器；原Prompt Encoder只提供no-mask embedding和位置编码，并冻结。

**与草图的具体化差别：**

- SCDS保留F1/F2的144/288原通道，内部宽度64；不是将二者直接拼成同尺寸图。
- SBCA背景参照是注意力模块内部的可学习查询，不额外插入原生self-attention的Token序列。其他质量/多掩码/任务Token仍走原生注意力。
- RCSR保留预训练转置卷积，在第一次上采样之后、high-res相加之前做采样细化。不能将它描述成已替换原生转置卷积。
- 原生neck与Mask Decoder可训练；Hiera原参数、Prompt Encoder冻结，Adapter可训练。

## 计算与初始化

### SCDS

浅层1×1投影→3×3/5×5深度卷积→逐位置两尺度softmax；F3/F4投影并各池化到最多8×8，加入二维坐标与层级编码；细节作Q，结构作K/V；由[D,C,|D−C|]预测门控；Gated FFN后投回原通道，以零初始化的可学习残差gain加回原特征。

### SBCA

复用被包裹注意力的Q/K/V/out预训练投影。对2/4/8像素窗口，先做窗口内softmax，再对窗口的**平均logit**做窗口间softmax；padding位置无概率质量。这里不是logsumexp汇聚，避免两级注意力代数退化成原全局softmax。多个尺度由查询条件的权重融合。

单掩码查询读取目标证据，内部背景查询读取背景证据；学习lambda后形成out_proj(z_target−lambda·z_background)。以零初始化gain在原生更新与新更新之间插值，其他Token更新保持原生值。

背景查询不能只靠命名获得功能：训练时对两路归一化空间证据的log比值加入BCE，目标为训练GT在22×22网格的面积占比，默认系数0.1。GT只进入loss，不进入forward。设置evidence_loss_weight=0可单独消融该监督。均匀注意力时证据概率约0.5，不是已经校准的前景概率；不能将其解释为可靠性保证。

### RCSR

区域Token经MLP，high-res特征经1×1投影，与待细化特征共同预测9点采样偏移和权重。偏移预测零初始化，偏移额外限制为每轴±2输出像素；固定3×3邻域加偏移，经grid_sample双线性读取，softmax加权。以零初始化gain将重建结果与原上采样结果相接。

因此初始RCSR整体为恒等映射；其内部均匀9点重建本身并非恒等映射。所有模块零gain时保持原生输出，但分支内部通常要在gain打开后才收到主损失梯度；探针至少执行2步。

## 配置与消融

| 配置 | 开关 | 输出目录 |
|---|---|---|
| configs/native_sam2_baseline.yaml | 全关 | runs/native_sam2_baseline |
| configs/native_sam2_scds.yaml | 仅SCDS | runs/native_sam2_scds |
| configs/native_sam2_sbca.yaml | 仅SBCA | runs/native_sam2_sbca |
| configs/native_sam2_rcsr.yaml | 仅RCSR | runs/native_sam2_rcsr |
| configs/native_sam2_modules.yaml | 三模块 | runs/native_sam2_modules |

共同设置1450张、固定352、batch16、20轮、同种子、FP32、AdamW lr1e-4/wd5e-4、cosine，翻转及bicubic软标签沿用新基线数据接口。原生解码器采用的lr不同于SAM2-UNet的1e-3；各原生分支间设置相同，不能称与SAM2-UNet完全相同超参。当前无早停、无验证选模、无断点续训。

环境沿用README中固定提交的SAM2；预训练文件仍是已下载的 `checkpoints/sam2_hiera_large.pt`，不是SAM2.1权重或UNet的last.pt。1450张清单沿用 `data/polyp_full/train_all.csv`。输出路径相对配置文件，已有目录不覆盖。

## 先探针，再训练

```bash
git pull
pip install -e '.[dev]'

medsam probe-native-sam2 --config configs/native_sam2_modules.yaml --steps 3 --batch-size 1
```

探针使用真实训练图和官方预训练权重，报告loss、模块梯度范数及CUDA峰值显存；这不是性能评估。零增益造成的第一步内部分支梯度为零属预期，2–3步后还没有梯度才需要追查。可再用 --batch-size 16测完整反向显存；云端CPU测试不能保证4090 batch16一定装得下。

先跑原生对照，再跑单模块及完整模型：

```bash
medsam train-native-sam2 --config configs/native_sam2_baseline.yaml
medsam train-native-sam2 --config configs/native_sam2_scds.yaml
medsam train-native-sam2 --config configs/native_sam2_sbca.yaml
medsam train-native-sam2 --config configs/native_sam2_rcsr.yaml
medsam train-native-sam2 --config configs/native_sam2_modules.yaml
```

全部从同一官方预训练文件初始化，不能加载已训练UNet模型冒充同基线消融。主损失为单输出weighted BCE+weighted IoU；仅SBCA开启时默认追加证据监督。baseline本身若明显不收敛，先排查原生路线，不把三模块结果直接与0.8450相减作为模块增益。

## 导出

```bash
medsam predict-native-sam2 --checkpoint runs/native_sam2_modules/last.pt --manifest /path/to/ETIS.csv --output outputs/native_modules_etis_raw --normalization raw
```

可另用独立目录导出 --normalization per-image。输出原图概率PNG、predictions.csv和provenance.json，兼容已有评估映射格式。预测无需GT或原始预训练文件；checkpoint包含完整模型及模块配置。逐图归一化、固定阈值、测试清单须在所有模型间保持一致。

## 验证边界

已做组件测试、部分窗口归一化、恒等初始化、Token选择、背景监督梯度、区域条件采样梯度以及原生352接口集成测试。真实权重文件、1450张数据和4090完整训练需在用户本机验证；不将随机初始化smoke test称为精度复现。

2026-11-21验证记录：全套97项测试通过，新文件ruff检查通过。CPU真实Hiera-L＋原生解码器＋三个模块，随机初始化352前后向通过；输出[1,1,352,352]，总参数219,176,328，可训练参数7,020,812，各残差gain梯度有限。预训练文件加载与完整GPU显存尚未在本环境验证。
