# Proposed SCSI–BCSC replacement decoder

状态：结构实现与可运行训练分支；尚无完整训练结果，不声称已经提升指标或确立新颖性。
保留PVTv2-B2编码器，CIM、CFM、原相似性聚合模块全部移除。原版配置/权重独立保留。

## 和设计图的对应

352输入得到F1/F2/F3/F4，通道64/128/320/512，空间88/44/22/11。
各自1×1投影为64通道，F4投影得到语义S4。

1. **SCSI**：前三层分别做Haar分解，得到LL/LH/HL/HH。
   每个位置四个子带token，注意力矩阵4×4。
   对齐后的S4经独立投影，以`1+tanh(context)`分别调制Q/K；V来自本层子带。
   注意力后两层通道MLP，再逆Haar返回原网格。没有丢弃高频，也不宣称学习后无损。
   子带注意力有可学习对角偏置，初始值2；它不是空间分割掩码。
2. **BCSC**：相邻三对同时读取各自原输入，不串行覆盖。
   每对有独立的“细读粗”和“粗读细”；Q来自接收层，K/V来自另一层。
   两层拼接预测18个偏移通道。规则3×3采样点加`tanh`限制的±2源像素偏移。
   grid_sample使用align_corners=False，像素中心坐标，border边界。
   16维瓶颈Q/K/V做局部交叉注意力，随后与接收特征拼接经卷积MLP。
   中间层接收两侧消息后拼接融合，端点各接收一条消息，输出P1…P4。
3. **区域—边界解码器**：两路从P4开始逐级上采样，与Pi拼接。
   每级先形成区域r与边界b，再同时由[r,b]和[b,r]更新两路。
   这里用拼接+逐点/深度卷积MLP，不使用粗掩码乘掉浅层特征。
   区域路径在P2对应44网格和P1对应88网格各输出一个头，边界头位于88网格。
   三个logit都上采样回输入尺寸。

实现明确使用GroupNorm(8 groups)+GELU。模块A的通道MLP为Linear–GELU–Linear；
BCSC/解码器的卷积MLP为1×1–GN–GELU–DW3×3–GELU–1×1。
图中的抽象MLP与解码器应以这里的展开和代码为准；最终论文图还需同步这些算子细节。

## 损失与推理

两个区域头各用与原版相同的structure loss，相加。
边界辅助项为0.2×structure loss，目标由当前尺度训练掩码二值化后3×3形态学梯度产生。
不使用外部边缘标签。该监督改变也是实验因素，提供no_edge_aux配置检验。
推理仅用`sigmoid(upsample(coarse_logit + refined_logit))`，不读取掩码，不使用边界GT。
原版两头44网格，新版44/88网格；需用相同解码器的control分离容量/分辨率收益与模块收益。

## 训练

先按docs/POLYP_PVT.md完成依赖安装、固定官方源码和ImageNet PVTv2-B2权重准备。
默认352、1450张、batch16、99轮、FP32、三尺度256/352/448，AdamW等设置继承原基线。
不加载已经训练的Polyp-PVT完整decoder checkpoint；同一ImageNet编码器初始化，从头训练新解码器。
所有编码器和新模块参数都训练。

```bash
medsam train-polyp-pvt --config configs/subband_pvt_full.yaml
```

新架构名`subband_pvt_b2`，与原`polyp_pvt_b2`检查点不通用。
完整模型保存至`runs/subband_pvt_352_full/last.pt`，不会覆盖原版。
不自动改变batch/精度；4090 batch16峰值未在此环境验证，若OOM先保留报错再制定对照。

## 预测

```bash
medsam predict-polyp-pvt --config configs/subband_pvt_full.yaml --checkpoint runs/subband_pvt_352_full/last.pt --manifest data/polyp_test/Kvasir_dedup.csv --output outputs/subband_pvt/Kvasir --inference-size 352 --normalization raw
```

五集一键导出（只有352，不扫测试尺寸）：

```bash
medsam sweep-polyp-pvt --config configs/subband_pvt_full.yaml --checkpoint runs/subband_pvt_352_full/last.pt --manifest-dir data/polyp_test --output outputs/subband_pvt_full_raw --sizes 352 --normalization raw
```

若原版采用per-image，必须两边都用per-image，并采用相同测试清单和指标定义。
记录每个实验的配置、初始化哈希、训练步数及原图概率图。

## 对照配置

| 配置 | SCSI | BCSC | 双路解码 | 边界辅助权重 |
|---|---|---|---|---|
| polyp_pvt_1450.yaml | 原版 | 原版 | 原版 | 无 |
| subband_pvt_control.yaml | 关 | 关 | 开 | 0.2 |
| subband_pvt_scsi_only.yaml | 开 | 关 | 开 | 0.2 |
| subband_pvt_bcsc_only.yaml | 关 | 开 | 开 | 0.2 |
| subband_pvt_full.yaml | 开 | 开 | 开 | 0.2 |
| subband_pvt_no_edge_aux.yaml | 开 | 开 | 开 | 0 |

control仍包含新的区域—边界解码器，并非参数量严格匹配的随机替代。
这些对照用于定位初步收益；尚不能独立证明“偏移采样”和“语义条件子带交互”的每个机制。
如主实验有稳定收益，再补固定采样、去语义条件、参数量匹配等内部对照。
不根据五测试集选最佳轮，也不保证所有模块组合都优于原版。

## 已完成的实现检查

- 项目全套146项测试通过，包括原版预测流程回归。
- Haar重构/能量/梯度，语义Q/K梯度，源像素采样坐标，双向交互均有检查。
- 真实PVTv2-B2结构（随机初始化）+默认64通道新解码器：352、batch2前向与反向通过，
  全部可训练参数有梯度且有限；256/448前向通过。
- 新模型25,355,519参数。上述CPU结构检查不是预训练权重复现实验，
  也不是4090 batch16显存验证，更不代表分割性能提升。
