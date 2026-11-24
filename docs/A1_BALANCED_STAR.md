# A₁均衡双支路＋Star式交互（三头，无随机增强）

这是独立实验，只替换F₁上的A₁。原三头配置、原A₂、单向B、C、Cat＋1×1融合以及两个训练辅助头不变；不启用逐级融合或双向扫描。

## 实际结构

输入X先经1×1卷积得到P。局部支路分别执行DW3×3和DW5×5并取逐元素平均；上下文支路执行DW5×5和DW7×7并取平均。两路的5×5卷积独立，不共享权重。

每路聚合特征H经独立Star式交互：`PW_out(ReLU6(PW_a(H)) * PW_b(H))`，三个PW均为C→C。再经Sigmoid生成权重，与P逐元素相乘。两路结果Concat为2C通道，最终1×1卷积恢复C通道。无外层残差、无差分、无可变形采样。

A₁为64×88×88（352输入时）；A₂仍为原SpatialContextA，128通道。两个Star实例独立。两支路处理层级对称，但大核支路计算量更大。

Star式交互依据：[Rewrite the Stars, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Rewrite_the_Stars_CVPR_2024_paper.html)。这里是受其启发的门控组件，不是完整StarNet复现，不据此宣称原创性或ETIS性能提升。

## 配置与训练

新配置：`configs/polyp_pvt_slots_three_head_a1_star.yaml`；独立输出：`runs/polyp_pvt_slots_three_head_a1_star`。

- 1450训练图，基础尺寸352，batch 16，99轮，seed 42。
- AdamW，lr=1e-4，weight_decay=1e-4，梯度裁剪0.5；其余沿用原三头流程。
- `augmentation: none`：不做随机裁剪、旋转、翻转或颜色增强。
- 保留原多尺度训练：同批次依次256/352/448，各一次参数更新；不是三个模型。
- 主损失权重1，B₃/B₄辅助损失各0.5；推理仅主头。

从PVTv2-B2预训练权重重新训练，用原三头同预算结果作对照；不要把旧三头last.pt当作新结构权重加载。预测时必须使用新配置与对应检查点，配置不匹配会拒绝加载。旧配置和已有权重不覆盖。

```bash
git pull --ff-only
medsam probe-abc-pvt --config configs/polyp_pvt_slots_three_head_a1_star.yaml --batch-size 2 --image-size 352
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_a1_star.yaml
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_three_head_a1_star.yaml --checkpoint runs/polyp_pvt_slots_three_head_a1_star/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_three_head_a1_star_raw --sizes 352 --normalization raw
```

评估与原三头使用同一数据划分、概率图口径和固定检查点规则。实现检查不代表已完成GPU训练；待实测逐集Dice、HD95及小目标分桶，不能预先保证增益。
