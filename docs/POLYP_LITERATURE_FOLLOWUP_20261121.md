# 息肉分割文献续查：发表、协议、源码完整性

日期：2026-11-21。补充前一份 `POLYP_BASELINE_AUDIT_20261121.md`。本轮完成两篇开放全文精读、多个作者仓库与出版记录核对；付费墙文献只标注已读摘要，不声称读过完整实验。

## 一、FPN-in-FPN，MICCAI 2025

- 录用证据：MICCAI Society 官方录用论文页面，DOI 10.1007/978-3-032-05141-7_20。不是仅凭 arXiv 判断。
- 资料：已读官方开放全文，核对仓库 API 文件树。
- 问题：单向多尺度融合不能充分结合高级语义与低级细节。
- 机制：在编码器与解码器之间组织嵌套的双向金字塔，改变多尺度信息流；VAN-B0 编码，主设置堆叠两次融合。
- 数据：训练 Kvasir 900 + ClinicDB 550；五集测试 100/62/60/380/196。
- 训练：120 epochs、batch 16、AdamW，lr=1e-4、wd=1e-4；翻转、旋转和多尺度训练，A100。具体增强分布不能由这些概述推断。
- Table 1：按 Kvasir/ClinicDB/CVC-T/ColonDB/ETIS，Dice=.914/.921/.900/.786/.802；4.01M 参数，MACs=2.55G。
- 源码状态：所查树 SHA `032046696d8b3b6c3651ed6bb38d202fb4695137` 只有一个 README 文件。论文给出代码链接，但此时不能据此称为完整可运行实现。
- 对本项目的启发：轻骨干也能有较强外部表现；融合拓扑与分辨率保留值得研究。不能直接把现有双向融合改名作为创新，需与已有双向金字塔区分。

来源：https://papers.miccai.org/miccai-2025/0351-Paper4899.html ；https://github.com/Yejin0111/FPN-in-FPN

## 二、PraNet-V2，Computational Visual Media 2026

- 正式发表 DOI 10.26599/CVM.2025.9450510，12(2):493–500。已读开放全文，并检查 PyTorch 训练和加载脚本。
- 问题与模块：将背景作为独立受监督对象，而非只从前景置信度取反；DSRA 用独立前/背景头及标签空间融合做细化。
- 数据：全文 §3.1 写 Kvasir 按 80/10/10 划分，ClinicDB 62 测试其余训练；外部评估为 CVC-300 与 ETIS。表 1 没有 ColonDB。不能默认它就是标准 1450 训练、五集测试。
- 输入与优化：352，多尺度 .75/1/1.25，Adam lr=1e-4。公开 PyTorch 默认 batch=8、epoch=30，循环 `range(1,opt.epoch)` 实际 29 轮；每批三个尺度各更新一次。
- 结果：PVTv2-B2 版本表 1 Dice 为 Kvasir .9152、ClinicDB .9309、CVC-300 .8989、ETIS .7635；Res2Net 版本 ETIS .6405。新版本名称不等于所有数据集最优。
- 数据加载：当前 `PolypDataset` 路径为 Resize、ToTensor、ImageNet Normalize，未在此路径看到随机旋转/翻转；不能将多尺度更新自动解释成随机几何增强。
- 可运行性问题：所下载 `binary_seg/MyTrain_med.py` 在建模前调用 `sys.exit(0)`，直接执行会提前结束。
- 选模问题：后续代码设置评估集为 CVC-300 和 ClinicDB，并按两个返回 Dice 之和保存 best。若这两者作为正式测试集，我们不能采用该选模规则。此处是源码观察，不据此推断作者全部发表结果的生成过程。
- SHA：仓库树 `ca08522ae77cbda6c5056609e7951b502dbb08f2`；所下载训练文件 SHA256 `ff206f9beefef95c60a7a67804e8608e45a3534abee671efbf320d1e074cbc63`。分支内容会变化，复现前应固定提交并再次核验。

来源：https://www.sciopen.com/article/10.26599/CVM.2025.9450510 ；https://github.com/ai4colonoscopy/PraNet-V2

## 三、Pattern Recognition 候选

### MSBP-Net

- 正式记录：Pattern Recognition 170 (2026), 112101，DOI 10.1016/j.patcog.2025.112101。
- 已核对出版摘要与方法概述，完整实验受到付费墙限制。
- PVTv2 编码，修改感受野块 mRFB、边界预测 BPM、浅层过滤 SFM；多尺度深度可分离卷积、SE、反向注意力及自主提示是其机制组成。
- 尚未核实该文对应的官方可运行仓库；不能将检索到的同名病理分类 MSBP_Net 当作其源码。
- 检索页包含“citing articles”，其中 LISA-Net 和其他模型的链接不能归到 MSBP-Net 名下。未核实训练超参，不补写猜测值。

来源：https://doi.org/10.1016/j.patcog.2025.112101

### Fu-TransHNet

- 正式记录：Pattern Recognition 170 (2026), 112116，DOI 10.1016/j.patcog.2025.112116。
- 本轮核对出版记录、源码与数据准备，不声称已读付费全文。保留“源码版本是否完整对应期刊最终版”的待确认项。
- 作者所指仓库存在训练、推理和预处理文件。`process_train.py` 从已有 TrainDataset 再做 80/20 划分，未固定 Python random 种子；图像与 mask 分别 os.listdir 后 zip，未按 stem 验证配对，因此不能直接用于我们的数据。
- 预处理 RGB、352×352，mask 的 cv2.resize 没有显式指定最近邻。当前训练默认 epoch=200、batch=20、lr=7e-5；这些为代码默认值，不能自动等同期刊全部实验。
- 仓库树 SHA `87787076298b557e73f53ff5068cb8f855da3985`。

来源：https://doi.org/10.1016/j.patcog.2025.112116 ；https://github.com/ChenxiLuo-code1/Fu-TransHNet

## 四、KBS 补充：Trans-SAM

- 正式记录：KBS 310 (2025), 112909，DOI 10.1016/j.knosys.2024.112909。
- 已检查官方 README、训练脚本、数据加载。提出 IPF 图像感知适配器与 MDT 多尺度域适配器，目标是参数高效迁移；可自动分割。
- 当前源码按单个输入数据集内部 80/10/10 划分。不能将单数据集 Kvasir 设置直接与训练 Kvasir+ClinicDB、测试三个外部集的协议合并排名。
- 作为编码器适配参照有价值；尚无本项目标准五集统一实测，不应据此取代主基线。
- 仓库树 SHA `8fd53ea699035acafaebeb7f73e4027f66a51588`。

来源：https://doi.org/10.1016/j.knosys.2024.112909 ；https://github.com/wuyanlin-wyl/Trans-SAM

## 五、对主基线选择的更新

上一轮建议 SAM2-UNet 优先复现，是出于公开实现、基础模型路线与 4090 可行性，并未证明它是最强息肉模型。本轮仍不据论文排行榜强行确定主基线。

实际候选保持 SAM2-UNet 与 Polyp-PVT：前者评估基础模型迁移路线，后者评估任务专用 PVT 路线。PraNet-V2 可作背景建模参照；FPN-in-FPN 适合研究信息流，但当前缺实现。DSCNet、DEPR-Net 和 MSBP-Net 应继续补源码或全文证据，不能用其未验证复写版充当“原版”。

比较须区分两条轨道：

1. 作者协议复现：遵守对应论文数据、预处理和预算，明确论文与代码差异。
2. 本项目公平比较：固定训练/验证/测试清单及输出评估口径；训练配置通过源域验证确定，固定最终测试；同骨干的模块消融保持预算一致。

单个 epoch 并非统一预算单位。多尺度方法可能每批更新三次，必须额外记录 optimizer steps、样本曝光次数、输入尺度与耗时。不能直接用“都训 50 轮”声称预算相同。

下一项研究问题仍聚焦息肉：高分辨率的小目标证据，能否在跨尺度更新中免于被错误粗语义覆盖，同时抑制黏膜假阳性。它目前是待验证问题，不是已成立创新。先在强基线测量其是否存在，再决定模块。

本轮没有修改模型算法，也没有声称完成以上方法的训练复现。

## 六、继续寻找开放全文的补充

另已找到并下载 Fu-TransHNet 的 2023 作者预印本： https://arxiv.org/abs/2301.06892 ，题为 Cooperation Learning Enhanced Colonic Polyp Segmentation Based on Transformer-CNN Fusion。其作者列表、题名与 2026 PR 期刊版有变化，不能当成期刊最终版逐项等同的证据。

已核对该公开稿 §IV.A–B：900 Kvasir + 550 ClinicDB 训练，五集剩余共 798 张测试，352×352，RTX 3090，Adam lr=7e-5，200 epochs，lambda=1。该段未明确列出 batch size 和增强分布，不从当前代码反推为公开稿原实验设置。

公开稿的设计把 Transformer、CNN 和融合输出看作三个学习视角，通过 GLFF 同尺度融合、DFM 跨尺度密集融合及多视角自适应加权共同训练。方法启发可以先研究，期刊最终版实验仍保留待核实项。

MSBP-Net 已找到作者单位的正式成果介绍：https://zxxy.suse.edu.cn/2025/0711/c2707a197732/page.htm 。该页可支持发表与设计概述，不是论文全文，不能据其最高 Dice 或速度概述填充五集实验表。本次搜索仍未核实其开放完整 PDF。
