# SGLang
我们建议您在部署TeleChat时尝试使用[sglang](https://github.com/sgl-project/sglang)。

## 安装

默认情况下，你可以通过 pip 在新环境中安装sglang,需要安装transformers==4.53.2进行推理，对于 sglang 建议可以使用0.5.0rc2：
```
pip install sglang 
```

请留意预构建的sglang对torch和其CUDA版本有强依赖。


## OpenAI兼容的API服务
```
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path Telechat3-36B --host 0.0.0.0 --trust-remote-code"
)

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")


import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="Telechat3-36B",
    messages=[
        {"role": "user", "content": "生抽和老抽的区别"},
    ],
    temperature=0,
    max_tokens=8192,
)

print_highlight(f"Response: {response}")
```
