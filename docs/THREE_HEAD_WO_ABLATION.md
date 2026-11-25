# 原三头模型的w/o删除消融

Full使用 `configs/polyp_pvt_slots_three_head.yaml`（原A₁/A₂），不是Polyp-PVT原架构。Polyp-PVT作为外部基线，另表报告。

| 配置后缀 | 删除内容 | 删除后连接 |
| --- | --- | --- |
| 原three_head | 无 | 原A₁/A₂、B₃/B₄、C |
| three_head_wo_a | A₁、A₂ | F₁/F₂直接接各自原通道投影，不恢复Polyp-PVT ca/sa |
| three_head_wo_b | B₃、B₄ | Identity；投影后特征同时送融合与原辅助头 |
| three_head_wo_c | C | 对齐E、U，Concat(E,U)，1×1卷积64→32，再接原最终头；不恢复原SAM |

w/o C的1×1卷积是为维持输出接口提供的基础融合对照，不是完全无参数旁路。三组均无原CFM，仍使用深层Cat＋1×1融合。B、C以外未删除的组件保持原版；不启用A★。

## 配置语义与兼容性

新增 `slots.ablation: wo_a / wo_b / wo_c`，默认`none`。三个w/o配置中a/a_f2/b/c仍为true，表示以完整模型为母体；构造模型时ablation仅删除指定组件。不要再把旧a/c开关改成false：旧开关用于恢复Polyp-PVT组件，语义不同。验证器要求w/o以完整三头、cfm=False配置为基础。

Full配置、默认计算图和权重键不变。每个w/o独立输出，不覆盖Full或历史实验。推理检查点校验加入ablation，必须使用对应训练配置。

## 训练

1450训练图，352基础尺寸，batch16，99轮，AdamW lr=1e-4、weight_decay=1e-4、梯度裁剪0.5、seed42。无随机增强，保留原256/352/448三尺度更新。均保留主头和B₃/B₄两个仅训练辅助头，辅助权重[0.5,0.5]。每组从同一PVTv2-B2预训练初始化，重新训练；不要从Full检查点删层直接评估。

以下三个训练命令依次执行（同一张4090不建议同时训练）。若已有Full结果，无需重跑Full。

```bash
git pull --ff-only
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_wo_a.yaml
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_wo_b.yaml
medsam train-polyp-pvt --config configs/polyp_pvt_slots_three_head_wo_c.yaml
```

对应输出为 `runs/polyp_pvt_slots_three_head_wo_a`、`_wo_b`、`_wo_c`。可先用probe-abc-pvt加相同config、`--batch-size 2 --image-size 352`做本机GPU探针。

## 评估

完整798张清单（Kvasir100、ClinicDB62、CVC300 60、ColonDB380、ETIS196），各组同一概率导出方式和同一阈值规则。按[新评估协议](EVAL_PROTOCOL_20261125.md)分列固定0.5与256阈值平均；不混入旧去重清单。具体论文是否可比需核对其归一化与指标实现，固定0.5不自动等于所有文献口径。

为各配置对应的last.pt重新推理，输出目录也独立。Full与w/o必须使用同一个固定末轮/预先约定选模规则。报告逐集指标、宏平均及参数/计算成本；关键组补相同的多个种子，不能凭单次小差异宣称有效。当前实现检查仅证明计算图和梯度连通，未在此环境完成GPU精度训练。
