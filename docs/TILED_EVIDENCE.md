# 高分辨率局部BiomedCLIP响应对照

新增`--evidence-mode tiled`。SAM仍整图前向，冻结的BiomedCLIP改用重叠裁块；没有增加可训练参数，也没有改变文本定位头。默认仍为global，旧checkpoint按原来的整图方式加载。

## 采样方式

对于当前1024×1024模型输入，`--tile-grid 2`使用512×512窗口、256步长，共9个窗口。每窗独立缩至官方224输入，提取14×14 patch余弦响应，以正值Hann权重融合为28×28。每个窗口仍使用同一张图的完整描述，不生成裁块描述。不会把全图描述中的方向词转换成窗口局部坐标，也没有解决文本位置绑定问题。

这里保留的是相对整图224缩放更细的视觉信息；不是改变ViT的16像素patch，也不是恢复原始图中已经丢失的细节。当前数据加载仍先将原图缩放为SAM的1024输入。融合在余弦响应层进行，不重新训练BiomedCLIP，不混入全局响应。

`--tile-grid 4`会使用256窗口、128步长，49个窗口，输出56×56。先跑grid=2，不建议直接使用49窗口。`--tile-batch-size 1`控制同时编码的窗口数，较大值可能提速但增加显存；它不会改变窗口位置或响应含义。

## 先运行1轮

```bash
medsam train-text-localizer --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --biomedclip-dir checkpoints/biomedclip --prompt "retinal blood vessels" --prompt-map outputs/chase_image_prompts/prompts.json --evidence-mode tiled --tile-grid 2 --tile-batch-size 1 --output outputs/chase_tiled_smoke --epochs 1
```

先确认本机显存、速度和导出响应，再改为新输出目录、30轮。保持此前的描述JSON、数据划分、种子、width和优化器不变，避免同时改变多个因素。

正式比较三组：

1. `global`＋正确描述：现有实验。
2. `tiled`＋正确描述：上述命令，30轮。
3. `tiled`＋统一任务词：删除`--prompt-map ...`，替换为`--task-only`，使用新输出目录。

另可加`tiled`＋`--prompt-pairing shuffle`。当描述有重复时，ID置换无法保证文字都改变；必须检查provenance中各划分的`effective_text_change_fraction`，不能把不充分的文字变化解释为完全错配。text_null依然输出，但它是同一训练头的消融，不是独立训练的纯视觉对照。

判断：若tiled改善Recall/细血管召回，但正确描述不优于统一任务词或有效错配，证据支持高分辨率视觉采样，不支持文本贡献。不能仅凭Dice小幅上升宣称文本有效。

## 保存与成本

采样模式、grid、窗口批量和每图窗口数写入checkpoint的`evidence_sampling`，评估、预测和第二阶段自动继承；不能给旧global头直接换成tiled后当作同一个模型。训练输出已有wall_seconds和CUDA峰值显存，需一并比较。9个窗口意味着BiomedCLIP图像编码样本数变为9倍，不代表整套模型耗时正好9倍。

真实官方权重CPU测试覆盖28×28输出、有限值和文本向量保持一致；几何测试核对坐标、拼接边缘、样本隔离和窗口批量一致性。4090性能与分割精度尚未验证。本改动是采样诊断对照，不是已经证实的新方法。
