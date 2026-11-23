# 固定352三头主模型：辅助监督对照

主模型回到 `polyp_pvt_slots_three_head.yaml`：A1/A2、原单向raster B3/B4、
三层对齐Cat＋1×1融合、原C、C后的最终预测头；B3/B4各接一个训练辅助头。
不启用axial或progressive。既有三头成绩为用户报告Dice宏平均0.8628，未在本机重跑。
合并axial+progressive配置保留为历史对照，不再作为主模型推荐。

## 仅改变辅助损失权重

| 配置 | B3辅助权重 | B4辅助权重 | 输出目录 |
|---|---:|---:|---|
| polyp_pvt_slots_three_head.yaml | 0.5 | 0.5 | runs/polyp_pvt_slots_three_head |
| polyp_pvt_slots_three_head_weak_aux.yaml | 0.2 | 0.1 | runs/polyp_pvt_slots_three_head_weak_aux |
| polyp_pvt_slots_three_head_no_aux.yaml | 0 | 0 | runs/polyp_pvt_slots_three_head_no_aux |

最终损失权重始终为1。no_aux保留相同网络、辅助头参数和初始化顺序，仅将两项
辅助损失系数设为0，两个辅助头不提供学习信号；仍计算辅助logit以保持同一训练前向接口。
推理在三组中都不执行辅助头，仅用final。已有slots_loss直接支持0权重，无需改模块。
三组初始化来源、种子、数据、优化器、轮数、训练尺寸策略与评估口径保持一致。
权重差异影响训练，不会使同一检查点在推理时变出另一组结果，需从相同预训练来源重新训练。

## 运行

```bash
git pull
medsam probe-abc-pvt --config configs/polyp_pvt_slots_three_head_weak_aux.yaml --batch-size 2 --image-size 352
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_weak_aux.yaml
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_no_aux.yaml
```

两组为独立实验，训练可按顺序执行。旧0.5/0.5结果已经完成，不需要重新训练旧目录。
保持1450张、352基准输入、batch16、99轮、AdamW1e-4、weight_decay1e-4、种子42、
原三倍率更新（256/352/448）；推理固定352。这不是更换为512训练。
各组独立输出，任何指定输出目录已存在时训练会拒绝运行，不覆盖旧权重。
使用最新版代码不要求回退Git；默认扫描/融合开关仍是raster/aligned。

评估弱辅助组示例：

```bash
medsam sweep-polyp-pvt --config configs/polyp_pvt_slots_three_head_weak_aux.yaml --checkpoint runs/polyp_pvt_slots_three_head_weak_aux/last.pt --manifest-dir data/polyp_test --output outputs/polyp_pvt_slots_three_head_weak_aux_raw --sizes 352 --normalization raw
```

no_aux组同步替换配置、checkpoint和output后缀。若此前基线使用per-image归一化，
统一使用`--normalization per-image`和独立目录，不能混用指标口径。
配置选择依靠训练来源内部验证，不按测试集挑选不同模型；统一报告五集与尺寸分桶。
这两组尚无训练结果，不预先认定弱监督更优。

检查记录：52项相关测试通过、1项可选集成测试跳过。新增检查确认两份配置
仅改变aux_weights和独立输出目录；0权重损失与final-only相等，两个辅助头梯度为0，
主预测及B4仍有梯度。未运行真实数据完整训练。
