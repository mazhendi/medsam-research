# 候选B：双轴双向Mamba，原A/C三头无增强

配置：`configs/polyp_pvt_slots_three_head_dual_axis_b.yaml`。同时替换B₃和B₄，保持原A₁/A₂、原C、深层Cat＋1×1融合、主预测头和两个辅助头。原B、Shared B、失败的跨尺度B1均保留为独立实验。

## 对应确认图的结构

1. 输入X为32×H×W。每个位置沿通道执行LayerNorm(32)，然后1×1投影32→32。
2. 水平路径：打包为(B·H)×W×32，每一行是独立序列。
3. 垂直路径：打包为(B·W)×H×32，每一列是独立序列。
4. 每条路径正反向共用同一个Mamba-1参数实例。正序和反序沿batch维度堆叠计算，状态独立；反向输出翻回原顺序后，与正向输出沿通道拼接64通道，通过逐点线性投影64→32。
5. 水平与垂直输出恢复为32×H×W，Concat64通道，1×1卷积64→32。
6. 与原始X相加：Y=X+F(X)，配置`b_residual: true`。

两个方向共享参数，不共享状态；两轴参数不共享；B₃/B₄也不共享参数。默认每个Mamba为输入32通道、expand2（内部64通道）、state16、因果深度Conv1D核4、SiLU及标准SSM输出门控。图中两个Linear32→64是一次Linear32→128后分割的两个参数块，代码继续使用既有SelectiveMamba1实现。

没有多尺度二维深度卷积，没有原B的分组递进相加，也不引入新的空间注意力或自定义状态更新。不是官方SS2D复现。虽然空间分支不使用二维卷积，Mamba内部仍保留因果一维卷积。

代码选项`b_variant: dual_axis`；不要同时设置历史`b_scan: axial`（该开关属于另一个旧实现）。旧配置缺省`b_variant: original`，不改变其计算图或检查点。推理已有b_variant/b_residual校验，必须使用训练配套配置。

## 本机运行

```bash
git pull --ff-only
medsam probe-abc-pvt --config configs/polyp_pvt_slots_three_head_dual_axis_b.yaml --batch-size 2 --image-size 352
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_dual_axis_b.yaml
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_three_head_dual_axis_b.yaml --checkpoint runs/polyp_pvt_slots_three_head_dual_axis_b/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_three_head_dual_axis_b_raw --sizes 352 --normalization raw
```

独立输出：`runs/polyp_pvt_slots_three_head_dual_axis_b`。从同一PVTv2-B2预训练初始化重新训练，不载入旧结构完整权重。1450训练图、基础352、batch16、99轮、seed42、AdamW lr/weight_decay均1e-4、梯度裁剪0.5，沿用256/352/448三尺度训练，`augmentation: none`，辅助权重[0.5,0.5]。推理仅主预测头。

评估与Full保持相同完整798张清单、归一化、阈值、检查点选择及HD95坐标口径。已有测试清单若采用独立目录/后缀，请配套使用同一目录，不要混入旧去重版本。

## 验证与研究边界

CPU测试核对正反向输出对齐、真实Mamba批次堆叠与独立调用等价、非方形网格逐行逐列参考、单行单列边界、22/11网格所有参数梯度、最终投影置零时严格恒等、三头训练/主头推理及旧路径兼容。

尚未在此环境完成GPU训练或速度/显存测量，先运行4090探针。该方案为待验证结构，不保证精度或效率提升；轴向和双向Mamba已有研究基础，不能只据此宣称强原创性。当前原Full继续作为已验证主模型，不覆盖历史结果。
