<div align="center">
<h1>
  星辰语义大模型-TeleChat3
</h1>
</div>

<p align="center">
   🦉 <a href="https://github.com/Tele-AI/TeleChat3" target="_blank">github</a> • 🤗 <a href="https://huggingface.co/Tele-AI" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/TeleAI" target="_blank">ModelScope</a> • 💬 <a href="https://github.com/Tele-AI/Telechat/blob/master/images/wechat.jpg" target="_blank">WeChat</a>
</p>

# 目录

- [模型介绍](#模型介绍)
- [能力评估](#能力评估)
- [推理](#gpu-推理)
  - [vLLM](#vllm)
  - [SGLang](#sglang)
- [微调](#gpu-微调)
  - [LLaMA-Factory](#llama-factory)
- [国产化适配](#国产化适配)
- [声明、协议、引用](#声明协议引用)

# 最新动态
- 2025.12.23 开源 **TeleChat3-105B-A4.7B-Thinking**、**TeleChat3-36B-Thinking**

# 模型介绍

### 星辰语义大模型-TeleChat3

- 星辰语义大模型**TeleChat3**是由中国电信人工智能研究院研发训练的大语言模型，该系列模型**完全基于国产算力**训练。


### 模型结构

**TeleChat3**的模型结构配置如下表所示：

|      | Layers | Hidden Size | FFN Intermediate | Attention |  Routed Experts | Experts per Token | Shared Experts |
|------|-----------|------|------|-----|-----|-----|---|
| 105B-A4.7B | 45    | 2560        | 7680  |  MLA  | 192 | 4 | 1 |
| 36B  | 64        | 6144        | 24576   | GQA | - | - | - |


本次发布版本和下载链接见下表：
|                               | modelscope | Huggingface |
|-------------------------------|------------|-------------|
| TeleChat3-105B-A4.7B-Thinking | -          | -           |
| TeleChat3-36B-Thinking        | -          | -           |

# 能力评估
为了全面体现模型效果，针对六个维度（知识、数学、创作、代码、Agent、指令）进行模型能力评测，所有模型均评测Thinking思考模式，具体评测效果如下：

| 评测集                        | 任务类型  | Qwen3-30B-A3B | Qwen3-30B-A3B-Thinking-2507 | Qwen3-32B | GPT-OSS-120B | TeleChat3-105B-A4.7B-Thinking | TeleChat3-36B-Thinking  |
|----------------------------|-------|---------------|--------------------|-----------|--------------|----------------------|----------------|
| MMLU-Pro                   | 知识    | 78.4          | 80.9               | 75.37     | 79.19        | 78.5                 | 80.89          |
| GPQA-Diamond               | 知识    | 65.8          | 67.68              | 68.4      | 80.1         | 66                   | 70.56          |
| Creative writing v3        | 创作    | 79.1          | 84.4               | 81        | 80.77        | 82.1                 | 84.33          |
| IFEval                     | 指令    | 86.5          | 88.9               | 90        | 82.4         | 83.7                 | 82.96          |
| Math-500                   | 数学    | 98            | 94.4               | 97.2      | 90           | 91                   | 95             |
| AIME2024                   | 数学    | 80.4          | 76.7               | 81.4      | 73.3         | 71.1                 | 73.3           |
| AIME2025                   | 数学    | 70.9          | 85                 | 72.9      | 83.3         | 69.7                 | 73.3           |
| Livecodebench(24.08-25.05) | 代码    | 63.11         | 66.89              | 69        | 74.01        | 66.5                 | 69             |
| IFEvalCode                 | 代码    | 20.95         | 20.45              | 28        | 25.73        | 23                   | 26             |
| HumanEval-X                | 代码    | 84.88         | 88.29              | 76.1      | 89.76        | 87.3                 | 92.67          |
| SWE Bench verify           | 代码    | 21            | 26                 | 28        | 44           | 42                   | 51             |
| BFCL-V3                    | Agent | 69.1          | 72.4               | 70.3      | 65.25        | 65.9                 | 68             |
| Tau2-Bench                 | Agent | 31.3          | 47.7               | 41.73     | -            | 58                   | 63.6           |


# 推理

### 本地推理

当前模型推理兼容了单卡和多卡推理，以及针对长文推理做了部分优化工作。

**模型推理方法示范**

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
tokenizer = AutoTokenizer.from_pretrained('./TeleChat3-36B-Thinking', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('./TeleChat3-36B-Thinking', trust_remote_code=True, device_map="auto",torch_dtype=torch.bfloat16)
prompt = "生抽与老抽的区别？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages,
    tokenize=False,
    add_generation_prompt=True
>>>	)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    top_p=0.95,
    temperature=0.6,
    repetition_penalty=1.05,
    max_new_tokens=2048
)
response = tokenizer.decode(generated_ids[0], skip_special_tokens=False,spaces_between_special_tokens=False)
answer = response.split("</think>")[-1].strip()

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
生抽和老抽是两种不同的酱油，它们在风味、色泽和用途上都有所区别。

1.颜色：生抽的颜色比较淡，而老抽的颜色较深。生抽的颜色呈红褐色或棕红色，而老抽的颜色则呈棕黑色。

2.味道：生抽具有鲜美的咸味和微甜的味浅，而老抽浓郁，颜色较深。根据个人口味和烹饪需求选择不同的酱油类型可以获得更好的口感和菜肴效果。
```

### 服务化推理

### vLLM
[vLLM](https://github.com/vllm-project/vllm) 是一个快速且易于使用的 LLM 推理和服务库。vLLM 最初由加州大学伯克利分校的天空计算实验室开发，现已发展成为一个由学术界和工业界共同贡献的社区驱动项目。

TeleChat3 已支持使用 vllm 进行推理，具体使用方式参考文档 [TeleChat3-vLLM 推理文档](./tutorial/vllm.md)。我们也在`eval/`中提供了部署与推理thinking模型的脚本示例。

### SGLang

[SGLang](https://github.com/sgl-project/sglang) 是一个用于大型语言模型和视觉语言模型的快速服务框架。通过共同设计后端运行时和前端语言，它使您与模型的交互更快、更可控。

TeleChat3 已支持使用 SGLang 进行推理，具体使用方式参考文档 [TeleChat3-SGLang 推理文档](./tutorial/sglang.md)。

### 推理注意事项

1. TeleChat3-36B-Thinking 系列模型在 chat template 中加入了一些适配复杂推理模型的特性：
    - TeleChat3-36B-Thinking 系列模型在 chat template 中加入了`<think>\n`符号以确保推理时能够生成 reason 过程。如果借助 `transformers` 库推理，并采用`apply_chat_template`方法，且 `add_generation_prompt` 设为`True`，则将会在推理时自动拼接`<think>\n`符号。
    - TeleChat3-36B-Thinking 系列模型在进行多轮推理时不应传入之前轮次回答中的`<think>..</think>`过程，在chat template 中已经实现了对多轮历史信息的自动处理。

2. TeleChat3-36B-Thinking 系列模型推理参数选择
    - 在推理数学、代码任务时，建议使用`repetition_penalty=1.0, top_p=0.95`、`temperature`设为`1.1-1.2`之间的值进行推理。
    - 在推理通用任务时，建议使用`repetition_penalty=1.05, temperature=0.6, top_p=0.95`的推理设置，可以有效减少重复生成现象。

# 微调

### LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是一个专注于大语言模型（LLM）开发和优化的开源平台，旨在简化模型训练和部署的过程。该平台提供了多种工具和框架，支持用户根据特定需求自定义和扩展语言模型。通过 LLaMA-Factory，研究人员和开发者可以更高效地探索和实现最新的自然语言处理技术，例如 LoRA、QLoRA、Pre-Training、Supervised Fine-Tuning、DPO Training等。

TeleChat3 已支持使用 LLaMA-Factory 进行微调、权重合并、推理、部署，具体使用方式参考文档 [TeleChat3-LLaMA-Factory微调文档](./tutorial/telechat_llama_factory.md)。


# 国产化适配

### 昇腾 Atlas 800T A2 训练服务器实现训练、推理适配

#### 核心组件：

- 昇思 MindSpore：该框架是华为开发的深度学习框架，旨在为AI应用提供高效、灵活的开发环境。它支持多种硬件平台，并具有自动微分、模型优化等功能，适合各种深度学习任务。

- MindSpore Transformers：该框架的目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件，提供业内主流的 Transformer 类预训练模型和 SOTA 下游任务应用，涵盖丰富的并行特性。期望帮助用户轻松的实现大模型训练和创新研发。

**当前星辰语义大模型 TeleChat3 支持昇腾 Atlas 800T A2 训练服务器，可基于昇思 MindSpore 框架以及 MindSpore Transformers 框架进行模型训练和评测，详情请看 [TeleChat3国产化](./tutorial/TeleChat3_国产化运行.md)。如果您对 MindFormers 相关特性有疑问，也可以查看 [MindFormers官方代码和文档](https://gitee.com/mindspore/mindformers/tree/dev/)。**

TeleChat3 系列模型性能方面，具体对比如下：

| Model Size    | Performance (samples/p/s) | NPUs | Epochs |
|:---------------| :-------------- |:------- | :----- |
| 105B-A4.7B  | 0.1002  | 4096  |     1      |
| 36B   | 0.0633   |   2048   |     1      |



# 致谢

在此，我们要向开源社区的伟大贡献致以最深切的谢意——正是站在这些巨人的肩膀上，我们才得以眺望更远的风景。

特别的向DeepSeek团队表达我们诚挚的感激。借鉴其模型架构的设计智慧，为我们模型的训练过程赋予了显著的稳定性与效率，使探索之路更为平稳而清晰。


# 声明、协议、引用

### 声明

我们在此声明，不要使用 TeleChat 模型及其衍生模型进行任何危害国家社会安全或违法的活动。同时，我们也要求使用者不要将 TeleChat 模型用于没有安全审查和备案的互联网服务。我们希望所有使用者遵守上述原则，确保科技发展在合法合规的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用 TeleChat 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。


### 引用

如需引用我们的工作，请使用如下 reference:

```
@misc{wang2025technicalreporttelechat2telechat25,
      title={Technical Report of TeleChat2, TeleChat2.5 and T1}, 
      author={Zihan Wang and Xinzhang Liu and Yitong Yao and Chao Wang and Yu Zhao and Zhihao Yang and Wenmin Deng and Kaipeng Jia and Jiaxin Peng and Yuyao Huang and Sishi Xiong and Zhuo Jiang and Kaidong Yu and Xiaohui Hu and Fubei Yao and Ruiyu Fang and Zhuoru Jiang and Ruiting Song and Qiyi Xie and Rui Xue and Xuewei He and Yanlei Xue and Zhu Yuan and Zhaoxi Zhang and Zilu Huang and Shiquan Wang and Xin Wang and Hanming Wu and Mingyuan Wang and Xufeng Zhan and Yuhan Sun and Zhaohu Xing and Yuhao Jiang and Bingkai Yang and Shuangyong Song and Yongxiang Li and Zhongjiang He and Xuelong Li},
      year={2025},
      eprint={2507.18013},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.18013}, 
}
```
