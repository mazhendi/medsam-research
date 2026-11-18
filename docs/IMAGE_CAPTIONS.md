# 逐图文本：生成、检查与对照

这次更新增加原图→本地视觉语言模型→逐图描述JSON→冻结BiomedCLIP文本编码器的流程。SAM、文本融合头和14×14 patch响应算法没有改变。逐图描述是否改善分割，需要实验验证；它不会自动修复此前细血管定位信号不足的问题。

## 1. 安装与生成

```bash
git pull
pip install -e ".[dev,text,captions]"
medsam generate-image-prompts --manifests data/retina/CHASE_DB1/train.csv data/retina/CHASE_DB1/val.csv --task "retinal blood vessels" --output outputs/chase_image_prompts
```

默认独立加载Qwen/Qwen2.5-VL-3B-Instruct，固定版本，CUDA FP16，贪心生成；首次需下载数GB权重。生成时不加载SAM。4090运行速度和显存尚未实测。清单路径以本地实际位置为准。

仅读取原图，不读取掩码、FOV、分割预测或错误图。四项内容为外观质量、目标可见性、图像相对区域差异和不确定性，不要求生成诊断、精确尺寸或不存在的细节。换息肉或X片时修改`--task`，具体目标必须明确。

输出：

- `prompts.json`：图像ID到完整文本的映射，可直接用于训练。
- `records.jsonl`：原始回复、解析后的四项描述、错误和质量警告。
- `provenance.json`、`prompts.metadata.json`：模型版本、生成指令、图像哈希及映射哈希。
- `summary.json`：完成数、不同文本数及指令照抄警告数。

先人工检查描述是否符合图像，尤其是位置、可见性及不确定性。模型描述不是专家标注，警告检测也不完整。相似图像可以得到相同描述；程序不添加ID或编造细节制造差异。全部训练描述相同时，逐图训练会报错。

失败不会用统一任务词补齐，也不会导出不完整映射。修复原因后用原命令追加`--resume`，保留成功项，只重试未完成项；改变模型、指令或原图需使用新目录。解析支持完整JSON或完整的四项有序标签，缺字段时报错。

描述超出BiomedCLIP的文本长度限制会报错，不静默截断。人工修订可直接编辑映射，保留元数据以继续核验原图；训练记录映射哈希及其是否经过修改。自行提供无元数据JSON也支持，但无法核验图像哈希。

## 2. 三组第一阶段实验

```bash
medsam train-text-localizer --config configs/retina_chase.yaml --checkpoint outputs/chase_observer1_seed42/best.pt --biomedclip-dir checkpoints/biomedclip --prompt "retinal blood vessels" --prompt-map outputs/chase_image_prompts/prompts.json --output outputs/chase_text_matched --epochs 30
```

先将epochs改为1做运行检查。正式对照使用同一配置、初始化、种子和轮数：

1. 正确配对：上述命令。
2. 错误配对：追加`--prompt-pairing shuffle`，改用新输出目录。训练和验证各自按数据集内部做无自身配对的置换，不跨划分；记录实际文本变化比例。每组至少两张图。
3. 统一任务词：删除`--prompt-map ...`，改用`--task-only`，使用新输出目录，复现原来的任务词条件。

每组仍输出visual/text/text_null。检查正确配对相对错配和统一词是否有稳定收益，同时比较Recall、clDice及细血管召回，不能只看Dice。先不进入第二阶段；30轮是与上次对应的诊断预算，不保证收敛。

## 3. 新图与测试

新图先独立生成描述，再给`evaluate-text-model`或`predict-text-model`追加`--prompt-map 新目录/prompts.json`。逐图checkpoint遇到未知ID不会退回统一文本。第二阶段继承第一阶段的文本策略和映射。

若预先生成测试图描述，必须固定生成模型、指令和检查规则，仅使用测试原图，不能依据测试标签或指标修改提示策略。描述生成不属于分割训练监督。

## 验证范围

本环境用真实SmolVLM-500M权重和一张CHASE原图完成CPU生成及导出，模型输出出现照抄指令，已标记；该小模型只用于接口检查，不作为医学描述质量证据。默认Qwen-3B及4090尚未实测。夹具测试覆盖训练文本接入、缺失拒绝、图像身份检查、失败恢复及错配对照，不代表描述有医学价值或方法已有创新。
