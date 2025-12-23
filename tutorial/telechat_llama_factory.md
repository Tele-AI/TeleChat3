# LLaMA-Factory
我们将介绍如何使用 LLaMA-Factory 微调 TeleChat2 模型。

* 支持单卡和多卡分布式训练
* 支持全参数微调、LoRA、Q-LoRA 和 DoRA 。

# 安装
开始之前，我们建议安装transformers==4.53.2,lamafactory0.9.3.请确保你已经安装了以下代码库：

1. 根据 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 官方指引构建好你的环境
2. 安装下列代码库（可选）：

    ```bash
    pip install deepspeed
    pip install flash-attn --no-build-isolation
    ```

3. 如你使用 [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)  ，请确保你的CUDA版本在11.6以上。


# 准备数据
LLaMA-Factory 在 data 文件夹中提供了多个训练数据集，您可以直接使用它们。如果您打算使用自定义数据集，请按照以下方式准备您的数据集。

1. 请将您的数据以 json 格式进行组织，并将数据放入 data 文件夹中。LLaMA-Factory 支持以 alpaca 或 sharegpt 格式的数据集。

* alpaca 格式的数据集应遵循以下格式：
    ```json
    [
        {
            "instruction": "user instruction (required)",
            "input": "user input (optional)",
            "output": "model response (required)",
            "system": "system prompt (optional)",
            "history": [
                ["user instruction in the first round (optional)", "model response in the first round (optional)"],
                ["user instruction in the second round (optional)", "model response in the second round (optional)"]
            ]
        }
    ]
    ```
* sharegpt 格式的数据集应遵循以下格式：
    ```json
    [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "user instruction"
                },
                {
                    "from": "gpt",
                    "value": "model response"
                }
            ],
            "system": "system prompt (optional)",
            "tools": "tool description (optional)"
        }
    ]
    ```
2. 在 data/dataset_info.json 文件中提供您的数据集定义，并采用以下格式：

* 对于 alpaca 格式的数据集，其 dataset_info.json 文件中的列应为：

    ```json
    "dataset_name": {
    "file_name": "dataset_name.json",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
        "history": "history"
        }
    }
    ```
* 对于 sharegpt 格式的数据集，dataset_info.json 文件中的列应该包括：

    ```json
    "dataset_name": {
        "file_name": "dataset_name.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
    ```

# 训练
下载模型，例如模型位置为："./telechat3_36B"

替换完毕后，使用如下telechat3_full_sft.yaml即可使用示例数据进行全参微调过程。

```Bash
### model
model_name_or_path: ./telechat3_36B
trust_remote_code: true


### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: ./llamafactory/examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: identity,alpaca_en_demo
# tokenized_path : /workspace/sft_minrecords/summary 如果需要高阶微调，需要用tokenized_path
template: telechat2
cutoff_len: 8192
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/telechat3-36b/full/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

如果需要使用Telecht3-32k进行训练，建议使用4台及以上的A800机器，使用如下命令进行训练：

```Bash
llamafactory-cli train telechat3_full_sft.yaml
```



