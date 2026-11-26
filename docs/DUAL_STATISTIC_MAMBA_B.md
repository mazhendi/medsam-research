# 双统计多尺度Mamba B候选

配置：`configs/polyp_pvt_slots_three_head_dual_stat_b.yaml`。B₃、B₄同时替换为独立的DualStatisticMambaB实例，原A₁/A₂、原C、深层Cat＋1×1融合、一个主头与两个辅助头均保持不变。原Full和所有历史变体保留。

## 对应确认图的实际结构

输入X(32×H×W) → 1×1投影32→32得到P，再分为两条结构对称、参数独立的支路：

- 平均统计支路：AvgPool3×3(stride1,padding1) → 并行DW3×3/DW5×5/DW7×7 → Concat96通道 → 1×1 Conv96→32 → Flatten＋LayerNorm32 → 独立Mamba-1 → 恢复网格。
- 极值响应支路：MaxPool3×3(stride1,padding1) → 同样的并行DW3/5/7、Concat96、1×1 Conv96→32、LN和独立Mamba-1 → 恢复网格。

两路各为32通道，Concat64 → 1×1 Conv64→32，与原始X相加。默认外层残差开启。没有组间递进相加、跨尺度Softmax、共享Mamba、跨层注意力或读写条件化SSM。两路中的卷积权重和Mamba参数均不共享；B₃与B₄之间也不共享。

所有池化/二维卷积保持网格。平均池化明确`count_include_pad=False`，边界均值只计有效邻居；最大池化采用PyTorch标准负无穷填充，避免负值特征被补零取代。深度卷积仍使用普通零填充。

352输入时，B₃输出32×22×22，B₄输出32×11×11。两路Mamba均采用单向光栅扫描，输入32通道、expand2（内部64通道）、d_state16、因果DWConv1D核4，SSM及SiLU门控沿用既有实现。每条序列独立初始化状态。Mamba内部结构不变。

## 训练与评估

```bash
git pull --ff-only
medsam probe-abc-pvt --config configs/polyp_pvt_slots_three_head_dual_stat_b.yaml --batch-size 2 --image-size 352
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_dual_stat_b.yaml
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_three_head_dual_stat_b.yaml --checkpoint runs/polyp_pvt_slots_three_head_dual_stat_b/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_three_head_dual_stat_b_raw --sizes 352 --normalization raw
```

独立训练目录`runs/polyp_pvt_slots_three_head_dual_stat_b`。从同一PVTv2-B2预训练重新训练，不加载原B或其他变体的完整权重。已有b_variant/b_residual检查要求预测配置与训练一致；旧配置默认original，保持兼容。

训练设置：1450图、batch16、99轮、seed42、AdamW lr与weight_decay均1e-4、梯度裁剪0.5、基础352，保留256/352/448三尺度更新。`augmentation: none`，无随机旋转/裁剪/翻转/颜色扰动。辅助权重[0.5,0.5]，推理只执行主头。与Full使用相同完整798张清单和评估口径，注意不要混入去重版清单或不同HD95坐标。

## 验证与限制

CPU测试覆盖池化边界及负值输入的邻域参考、两路参数独立、3/5/7卷积与96→32接口、22/11网格全参数梯度、零融合输出时的恒等残差、三头训练/主头推理、旧路径兼容。

当前尚未进行GPU精度训练，先在4090跑探针。双池化可能损失弱响应或保留背景强响应；7×7在11×11网格的边缘依赖填充。不能预先把平均支路解释为前景、最大支路解释为边界，也不能保证泛化收益或原创性。独立双Mamba会增加计算，效率需实测。后续以不池化、单池化及共享/独立Mamba等同条件对照定位作用。
