# Region–shape 改进前的检索与设计依据

检索日期：2026-11-19。检索方向：多候选分割、空间专家路由、医学分割歧义、分歧/不确定区域细化、息肉不确定性。

**结论：多候选、空间权重、不确定性引导细化均有大量先例。下面的窗口竞争是待检验的工程研究变体，不宣称“首次”或已经具备 ICML 新颖性。**

## 证据范围

本次从会议官方公开 PDF 阅读了 Probabilistic U-Net、PointRend、Spatial MoE、Tyche、AmbiSSL、SPA，以及 CVPR 2026 的 Delving Aleatoric Uncertainty，重点看方法与实验设置。MoSE 的 OpenReview 正文请求返回 403，因此其记录仅依据可检索的录用 PDF 首页/摘要和作者公开说明；UM-Net 仅核验已发表书目信息与 PubMed 摘要。两者不冒充全文精读。

PDF 留在本地检索工作目录，不随项目上传；以下给出原始来源与方法定位，避免把第三方论文文本复制进仓库。检索不是系统综述，也不能证明不存在更接近的已有方法。

## 已发表/录用来源

| 工作及正式来源 | 状态核验 | 问题与设计 | 实验范围/与本项目的区别 |
|---|---|---|---|
| [A Probabilistic U-Net for Segmentation of Ambiguous Images](https://proceedings.neurips.cc/paper_files/paper/2018/hash/473447ac58e1cd7e96172575f48dca3b-Abstract.html) | NeurIPS 2018 正式论文集；全文 §2 | U-Net 与条件 VAE 结合，采样整体分割假设，学习多种合理解释及其频率 | 肺部病灶、合成歧义 Cityscapes；不是单一标签息肉 Dice 优化协议 |
| [PointRend: Image Segmentation as Rendering](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) | CVPR 2020 正式论文集；全文 §3.1–3.2 | 在选出的不确定位置结合细粒度特征和粗预测，以点头细化；训练同时保留随机覆盖 | COCO、Cityscapes 等实例/语义分割；说明“按不确定性取细节”不是新想法 |
| [Spatial Mixture-of-Experts](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4c5e2bcbf21bdf40d75fddad0bd43dc9-Abstract-Conference.html) | NeurIPS 2022 Main Conference Track；全文 §2 | 细粒度空间路由、稀疏专家执行、路由监督与专家误差抑制 | 主要为空间密集回归/天气等；其共享空间结构假设不能原样套到任意息肉位置 |
| [Modeling Multimodal Aleatoric Uncertainty in Segmentation with Mixture of Stochastic Experts](https://openreview.net/forum?id=KE_wJD2RK4)；[作者实现与说明](https://github.com/gaozhitong/MoSE-AUSeg) | ICLR 2023 录用 PDF 首页可检索；正文访问受限 | 随机专家和门控表示多模态分割，作者说明采用 Wasserstein-like 目标 | LIDC-IDRI、合成歧义 Cityscapes；本次仅摘要/作者说明，不据此复现损失细节 |
| [Tyche: Stochastic In-Context Learning for Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Rakic_Tyche_Stochastic_In-Context_Learning_for_Medical_Image_Segmentation_CVPR_2024_paper.html)；[作者项目](https://tyche.csail.mit.edu/) | CVPR 2024；全文 §3.1.1–3.1.3 | SetBlock 让候选与上下文集合交互，best-candidate loss 训练多样结果；另一变体用上下文增强产生随机性 | 跨医学任务上下文分割，含多观察者数据；和我们的单任务、单标签训练目标不同 |
| [Annotation Ambiguity Aware Semi-Supervised Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Kumari_Annotation_Ambiguity_Aware_Semi-Supervised_Medical_Image_Segmentation_CVPR_2025_paper.html) | CVPR 2025；全文 §3 | AmbiSSL：随机剪枝解码器产生多样伪标签，半监督潜分布学习，以及跨解码器监督 | LIDC-IDRI、ISIC 的论文协议；需要区分标注集与未标注集，不是普通全监督五息肉协议 |
| [SPA: Efficient User-Preference Alignment against Uncertainty in Medical Image Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_SPA_Efficient_User-Preference_Alignment_against_Uncertainty_in_Medical_Image_Segmentation_ICCV_2025_paper.html) | ICCV 2025；全文 §3.2–3.4 | 用偏好分布生成候选，选择少数代表候选供用户反馈，并更新偏好 | REFUGE2、LIDC-IDRI、QUBIQ；有人机交互，不能据此声称我们的自动分歧图等价于临床偏好分布 |
| [Delving Aleatoric Uncertainty in Medical Image Segmentation via Vision Foundation Models](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Delving_Aleatoric_Uncertainty_in_Medical_Image_Segmentation_via_Vision_Foundation_CVPR_2026_paper.html) | CVPR 2026 官方公开录用版 PDF；全文 §2–3 | 根据基础模型表示的特征多样性/奇异值能量构造难度量，再用于数据筛选、优化与标签去噪 | LiTS、TotalSegmentator、WORD、FeTA、KiTS23；难度评分并不等于我们候选的空间方差 |
| [UM-Net: Rethinking ICGNet for polyp segmentation with uncertainty modeling](https://pubmed.ncbi.nlm.nih.gov/39316997/) | Medical Image Analysis 99:103347，2025；2024 在线发表 | 摘要报告颜色迁移和不确定性建模/方差修正 | 五个息肉数据集；仅摘要级核验，也足以说明“息肉不确定性建模”已有直接先例 |

## 对原先三个建议的修正

1. **多支撑候选生成：先不实现。** 多头与多专家本身不能保证有用的多样性。现有 K=4 候选是否塌缩，应先用两两 IoU、概率差和候选选择参考分数诊断。单观察者标签也不能直接验证“所有候选都是临床合理解”。
2. **窗口竞争：实现为单变量实验。** 不另加三个专家/三个新头，不换解码器，复用当前兼容性 MLP；仅将原有候选的内外池化、一致性评分细化到窗口，再用整图评分锚定。这样比较更容易解释，但不是空间路由思想的新发明。
3. **分歧引导高分辨率细化：先不实现。** PointRend 等已有明确先例。先检查分歧与原始分辨率错误的 AUROC，尤其注意“所有候选一起错”的低分歧漏检。候选差异不应自动命名为校准不确定性。

## 本次实现与相关方法的边界

- 我们没有复制论文源码，不宣称复现以上方法。
- 新版本是密集软窗口聚合，不是 Spatial MoE 的稀疏专家执行；所有候选仍计算，不能声称减少编码或 FLOPs。
- 仍是原来的区域—形状交替更新，再通过原稠密/稀疏提示和 SAM2 解码器产生单个最终预测。稀疏提示继续用整图候选权重；窗口权重只用于空间均值、方差与后续区域更新。
- 当前可信的贡献表述只能是“提出并检验整图先验锚定的局部形状竞争，研究它在区域—形状协同推理中的作用”，不能写“首次局部专家路由”“解决医学歧义”或“保证泛化”。
- 窗口竞争是否值得保留，要由相同基线/参数/数据/预算/种子的 global 与 window 对照决定；查到先例不等于获得新颖性证明，模块实现完成不等于有效。
