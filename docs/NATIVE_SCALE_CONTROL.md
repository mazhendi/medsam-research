# 尺度覆盖诊断对照

这是一组训练数据增强对照，不是创新模块，不承诺提高精度。原生基线、SAM2-UNet及三个模块的默认配置均不变。

优先用现有预测诊断错误：

```bash
python tools/diagnose_native_archive.py native_sam2_baseline_2026-11-21.zip
```

脚本直接读取原大ZIP，导出baseline_diagnosis.csv，无需上传大包、解压图片或重新训练。包括GT面积、固定0.5 Dice/Precision/Recall、FP/FN、空预测、前背景概率与逐图最佳阈值。最佳阈值使用测试GT，仅作校准/定位诊断，不是合法的部署成绩，也不能用它确定正式测试阈值。

若要验证尺度覆盖因素，运行独立配置：

```bash
medsam train-native-sam2 --config configs/native_sam2_shrink.yaml
```

与native_sam2_baseline.yaml相比，只增加训练时缩小增强并更换输出目录。仍为1450张、352、batch16、20轮、同种子、lr1e-4；三个模块全关。输出runs/native_sam2_shrink，固定末轮，无测试选模。

每张图以0.3概率同步缩小RGB与mask，线性缩放系数均匀取0.5–0.8，再随机放入352画布。RGB双线性、mask bicubic软标签沿用现有接口；mask外部填0，RGB填该图的平均颜色，避免反射填充复制出没有标注的息肉。缩小后若不存在>=128的mask像素，则放弃这次缩小，保留原样。填充仍可能引入人工接缝，这只是诊断对照，不是真实小病灶的完整模拟。

缩小操作使用独立的worker种子随机流，不额外消耗原翻转随机流；默认概率0时原路径和随机序列不变。探针与推理不应用该增强。日志provenance记录概率、尺度区间与填充方式。

预测命令沿用原生分支：

```bash
medsam predict-native-sam2 --checkpoint runs/native_sam2_shrink/last.pt --manifest /path/to/ETIS.csv --output outputs/native_shrink_etis_raw --normalization raw
```

使用与原基线完全相同的测试清单和评估处理。外部测试分析是探索性的，后续增强参数/架构选择应在训练池内部开发划分完成，不反复根据外部结果调概率与尺度。
