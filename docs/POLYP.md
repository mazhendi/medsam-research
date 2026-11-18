# 息肉分割：自己的SAM 2.1基线

模型为冻结SAM 2.1 Hiera-Small图像编码器、可学习任务token和可训练mask decoder，BCE+Dice；没有文本分支、VLM、LoRA或额外纠错头。配置为`configs/polyp.yaml`，从官方Small权重初始化，不自动加载视网膜best.pt。

## 配对与划分

```bash
medsam manifest --images /path/to/Kvasir-SEG/images --masks /path/to/binary_masks --dataset Kvasir-SEG --output data/polyp/kvasir_all.csv
```

命令只配对，不自动划分。根据明确协议生成`data/polyp/train.csv`、`val.csv`和外部测试清单；填写已知患者、视频或序列group，同组不跨训练/验证。CSV中的图像路径相对其所在目录。标准五测试集协议与新建的分组划分不同，需记录具体样本名单，不能互称。CVC-300与EndoScene/CVC相关数据需检查来源重叠。

图像支持JPG/PNG等8-bit输入，掩码必须为单通道二值图。原始Kvasir掩码可能是RGB JPEG，不能直接当作二值PNG：需要明确记录灰度转换/阈值并导出二值掩码，再运行配对和审计；程序不会静默二值化多值或RGB标签。

可用Kvasir与ClinicDB构建训练/验证，并将ColonDB、ETIS、CVC-300保留为外部测试；这只是实验设计选择，不会自动替用户确定划分。外部测试不用于调阈值、选轮次或修改模型。

```bash
medsam audit data/polyp/train.csv data/polyp/val.csv
medsam train --config configs/polyp.yaml
medsam evaluate --config configs/polyp.yaml --checkpoint outputs/polyp_baseline_seed42/best.pt --manifest data/polyp/etis_test.csv --output outputs/polyp_etis_seed42
```

预测不需要目标标签：

```bash
medsam predict --config configs/polyp.yaml --checkpoint outputs/polyp_baseline_seed42/best.pt --manifest data/polyp/inference.csv --output outputs/polyp_predictions_seed42
```

输出目录已存在时拒绝覆盖。保留原视网膜配置和权重，息肉另建输出目录。旧基线检查点格式保持不变；加载还需原始官方Small权重。默认50轮是起步配置，不是已经证实的最优轮数。报告Dice、IoU、Boundary F1、前景比例、逐图结果及计算成本。息肉不应把血管宽度/中心线指标作为主指标。
