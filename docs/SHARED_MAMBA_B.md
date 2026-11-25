# 多尺度共享Mamba B：外层残差版

配置：`configs/polyp_pvt_slots_three_head_shared_b.yaml`。B₃/B₄均替换为本版本，原A₁/A₂、原C、深层Cat＋1×1融合和三头监督不变。它不是已失败的跨尺度注意力B1；该B1和原B均保留。

## 与确认图对应的结构

- X为32通道；1×1 Conv32→32得到P。
- P作为完整输入分别送入DWConv3×3、5×5、7×7，不分组、不压成每路8/16通道。每条路径均为32通道。
- 每路Flatten＋transpose得到B×HW×32，并经过独立LayerNorm(32)。
- 三路沿batch维度拼接为(3B)×HW×32，一次调用同一个Mamba-1。
- 沿batch拆回三路，恢复空间网格。Concat为96通道，1×1 Conv96→32。
- 最终Y=X+F(X)，配置明确`b_residual: true`。没有组间递进加法，也没有自定义联合状态核心、跨尺度Q/K/V或Channel Shuffle。

**共享参数，不共享状态。** 三路等价于同一个Mamba分别处理三条序列，各自从零状态开始；不能沿序列轴拼接，否则会产生跨尺度状态串接。B₃和B₄是两个独立模块，只在各自模块内部共享三种尺度的Mamba参数。

默认Mamba输入32通道、expand=2（内部64通道）、d_state=16、d_conv=4，保留单向光栅扫描及标准输出SiLU门控。恢复网格后没有额外的分支逐点卷积，与确认图一致；仅最终融合为96→32。

## 运行

```bash
git pull --ff-only
medsam probe-abc-pvt --config configs/polyp_pvt_slots_three_head_shared_b.yaml --batch-size 2 --image-size 352
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_shared_b.yaml
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_three_head_shared_b.yaml --checkpoint runs/polyp_pvt_slots_three_head_shared_b/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_three_head_shared_b_raw --sizes 352 --normalization raw
```

独立训练目录：`runs/polyp_pvt_slots_three_head_shared_b`。需要从同一PVTv2-B2预训练重新训练，不能加载原B或失败B1的完整权重；预测配置必须配套新检查点。已有b_variant和b_residual检查防止误用。原Full配置默认original且b_residual=false，不改变旧图或旧权重。

1450训练图、基础352、batch16、99轮、AdamW lr=1e-4/weight_decay=1e-4、梯度裁剪0.5、seed42。`augmentation: none`；仍保留256/352/448三尺度训练，辅助权重[0.5,0.5]，推理仅最终头。测试与原Full使用相同完整798张清单、相同归一化/阈值口径。

## 检查与限制

CPU检查包含：堆叠一次与独立调用三次一致、修改一个尺度不污染其他序列状态、两个深层位置22/11网格、全部参数梯度、最终投影置零时输出严格等于X、三头训练/主头推理、旧配置兼容性。

尚未完成GPU训练，先在4090运行探针确认开销。共享参数不代表三路只有一路计算成本：Mamba实际处理3B条序列，仍有三路激活与计算。这里是待验证结构，不能据此宣称稳定性/泛化提升或已验证原创性。
