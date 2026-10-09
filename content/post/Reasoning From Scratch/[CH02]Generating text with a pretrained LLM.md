---
title: "[CH02]Generating text with a pretrained LLM"
description: "利用预训练LLM生成文本"
date: 2026-09-26T00:26:43+08:00
lastmod: 2026-09-26T00:26:43+08:00
draft: false

categories:
  - Reasoning From Scratch
tags:
  - LLM
  - RLHF

toc: true
math: true
mermaid: true
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1790353663827_image.png
---

<!--more-->





## 零、写在前面

这一章就是简单过一下模型部署流程。

---

## 一、部署预训练 LLM

### 1.1 环境搭建 

```bash
uv init --bare         
uv python pin 3.12
uv add reasoning-from-scratch
```

### 1.2 硬件需求

训练 LLM 非常昂贵。对头部 LLM 公司来说，在加入任何推理技术之前，单纯训练一个新的 base 模型，算力成本从低位的 100–1000 万美元到高位超过 5000 万美元都很常见。

例如，DeepSeek V3 模型（它是 DeepSeek R1 推理系统所基于的 base checkpoint）在 **2048 块 Nvidia H800 GPU** 上训练了约 **11 周**，估计成本为 **550 万美元**。（DeepSeek V3 是近期少数完全公开算力信息的模型之一。）

此外，根据技术报告，最终那次训练运行消耗了 **1480 万 GPU 小时**的算力；能耗约为 **620 MWh**，大致相当于一个普通美国家庭约 55 年的用电量。

所以，我们会使用一个**相对较小但能力不错**的预训练 LLM，并在它之上实现推理技术。

可以看一下电脑的算力配置：

```python
import torch

print(f"PyTorch version {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA/ROCm GPU: {torch.cuda.get_device_name(0)}")
elif torch.xpu.is_available():
    print(f"Intel GPU: {torch.xpu.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("Apple Silicon GPU")
else:
    print("Only CPU")
```

```text
PyTorch version 2.10.0+cu128
CUDA/ROCm GPU: NVIDIA GeForce RTX 4090
```

### 1.3 LLM input
![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1790411913324_image.png)

先下一个 tokenizer

```python
from reasoning_from_scratch.qwen3 import download_qwen3_small

download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")
```

该命令会下载 `tokenizer-base.json` 文件（约 6 MB），并保存在 `qwen3` 子目录下。

然后把 tokenizer 配置从文件加载进 `Qwen3Tokenizer`：

```python
from pathlib import Path
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer

tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1790412006383_image.png)

```python
prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
```

```python
text = tokenizer.decode(input_token_ids_list)
print(text)
```

```text
'Explain large language models.'
```

```python
for i in input_token_ids_list:
    print(f"{i} --> {tokenizer.decode([i])}")
```

输出如下：

```text
840 --> Ex
20772 --> plain
3460 --> large
4128 --> language
4119 --> models
13 --> .
```

### 1.4 加载预训练模型
![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1790412258616_image.png)

我们使用 **Qwen3 0.6B** 作为预训练 base 模型。现在加载它的预训练权重，

先写一个获取device的接口

```python
def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
                # torch 2.10: inductor 会去读旧的 allow_tf32，混用检测会抛错
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

```

然后下载 Qwen3 0.6B 的权重：

```python
download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
```

输出如下：

```text
qwen3-0.6B-base.pth: 100% (1433 MiB / 1433 MiB)
✓ qwen3/tokenizer-base.json already up-to-date
```

下载完模型权重后，现在可以把 `Qwen3Model` 类实例化，并通过 PyTorch 的 `load_state_dict` 方法把预训练权重加载进去：

```python
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B)          #A
model.load_state_dict(torch.load(model_path)) #B
model.to(device)                              #C
```

- `#A` 用随机权重作为占位，实例化一个 Qwen3 模型
- `#B` 把预训练权重加载进模型
- `#C` 把模型转移到指定设备（例如 `"cuda"`）

>   注意，如果 `device` 是 `"cpu"`，`model.to(device)` 这一步会被跳过，因为模型默认已经在 CPU 内存里了。
>

```text
Qwen3Model(
  (tok_emb): Embedding(151936, 1024)
  (trf_blocks): ModuleList(
    (0-27): 28 x TransformerBlock(
      (att): GroupedQueryAttention(
        (W_query): Linear(in_features=1024, out_features=2048, bias=False)
        (W_key): Linear(in_features=1024, out_features=1024, bias=False)
        (W_value): Linear(in_features=1024, out_features=1024, bias=False)
        (out_proj): Linear(in_features=2048, out_features=1024, bias=False)
        (q_norm): RMSNorm()
        (k_norm): RMSNorm()
      )
      (ff): FeedForward(
        (fc1): Linear(in_features=1024, out_features=3072, bias=False)
        (fc2): Linear(in_features=1024, out_features=3072, bias=False)
        (fc3): Linear(in_features=3072, out_features=1024, bias=False)
      )
      (norm1): RMSNorm()
      (norm2): RMSNorm()
    )
  )
  (final_norm): RMSNorm()
  (out_head): Linear(in_features=1024, out_features=151936, bias=False)
)
```

### 1.5 文本生成

```python
prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
print(f"Number of input tokens: {len(input_token_ids_list)}")

input_tensor = torch.tensor(input_token_ids_list)   #A
input_tensor_fmt = input_tensor.unsqueeze(0)        #B
input_tensor_fmt = input_tensor_fmt.to(device)

with torch.inference_mode():
    output_tensor = model(input_tensor_fmt)         #C

output_tensor_fmt = output_tensor.squeeze(0)        #D
print(f"Formatted Output tensor shape: {output_tensor_fmt.shape}")
```

- `#A` 把 Python 列表转换为 PyTorch 张量
- `#B` 增加一个额外的维度
- `#C` 生成输出
- `#D` 移除那个额外的维度

输出：

```text
Number of input tokens: 6
Formatted Output tensor shape: torch.Size([6, 151936])
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1790412648285_image.png)

```python
last_token = output_tensor_fmt[-1]
print(last_token)
```

>   在上一步中使用 `torch.inference_mode()`，可以避免存储生成阶段并不需要的梯度信息。这能减少内存占用，通常也能提升速度。
>

这会打印出与最后一个 token 对应的 151,936 个值：

```text
tensor([ 7.3750, 2.0312, 8.0000,  ..., -2.5469, -2.5469, -2.5469],
       dtype=torch.bfloat16)
```

然后，我们可以用 `argmax` 函数取得该张量中最大值所在的位置：

```python
print(torch.argmax(last_token, dim=-1, keepdim=True))
```

结果是：

```text
tensor([20286])
```

返回的这个整数是该向量中最大值的**位置**，它同时也就对应着生成 token（`last_token`）的 **token ID**。我们可以通过 tokenizer 把它翻译回文本：

```python
print(tokenizer.decode([20286]))
```

这会打印出生成的 token：

```text
Large
```

然后写一个基础的文本生成函数：

```python
@torch.inference_mode()                                   #A
def generate_text_basic_stream(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    model.eval()                                          #B
    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]                     #C
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break                                         #D
        yield next_token                                  #E
        token_ids = torch.cat([token_ids, next_token], dim=1)  #F
```

- `#A` 关闭梯度追踪，以获得速度和内存效率
- `#B` 把模型切换到评估模式，以获得确定性行为（最佳实践）
- `#C` 获取最后一个 token 的分值
- `#D` 如果批次中所有序列都生成了 EOS，则停止
- `#E` 每生成一个 token 就立刻 yield 出去
- `#F` 把新预测出的 token 追加到序列上

然后写一个简单的prompt：

```python
prompt = "Explain large language models in a single sentence."
input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device                          #A
).unsqueeze(0)

max_new_tokens = 100                       #B

for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
):
    token_id = token.squeeze(0).tolist()   #C
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True                         #D
    )
```

- `#A` 把输入 token ID 转移到模型所在的那个设备（CPU 或 GPU）上
- `#B` 让模型最多生成 100 个新 token
- `#C` 把输出的 token ID 从 PyTorch 张量转换为 Python 列表
- `#D` 关闭缓冲，使 token 能被实时打印出来

生成的输出文本如下：

```text
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.<|endoftext|>Human language
is a complex and dynamic system that has evolved over millions of
years to enable effective communication and social interaction. It is
composed of a vast array of symbols, including letters, numbers, and
words, which are used to convey meaning and express thoughts and
ideas. The evolution of language has
```

> **TIP**： 第一个输出词 `" Large"` 前面有空格是**预期行为**：很多 tokenizer 在词出现在前文之后时，会把前导空格一起编码进 token。清单 2.2 中 token 是逐个流式输出、紧跟在输入文本之后的，所以这个前导空格自然地出现在第一个吐出的 token 里。如果我们想要更干净的输出，可以对第一个 token 或最终拼接好的字符串调用 `.lstrip()`。

在推理（训练之后生成文本）时，我们通常希望模型一产生特殊 token `<|endoftext|>` 就立刻停止。这个 token 的 ID 是 **151643**，我们可以这样确认：

```python
print(tokenizer.encode("<|endoftext|>"))
```

这个 token ID 也保存在 `tokenizer.eos_token_id` 属性中。

```python
for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id    #A
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

- `#A` 传入 end-of-sequence（eos）token ID

输出如下：

```text
Large language models are artificial intelligence systems that can
understand, generate, and process human language, enabling them to
perform a wide range of tasks, from answering questions to writing
articles, and even creating creative content.
```

然后写一个小工具函数，用来测量文本生成过程的运行时间。

```python
import warnings

def generate_stats(output_token_ids, tokenizer, start_time, end_time):
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")
    for name, backend in (("CUDA", getattr(torch, "cuda", None)),
                          ("XPU", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():       #A
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )
            if hasattr(backend, "synchronize"):
                backend.synchronize()                            #B
            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()
```

- `#A` 检查我们是否真的在使用这个后端
- `#B` 如果支持就 synchronize（对异步后端很重要）

```python
import time

start_time = time.time()
generated_ids = []
for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
    next_token_id = token.squeeze(0)
    generated_ids.append(next_token_id)                     #A
end_time = time.time()

output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)
```

4090 跑的结果：

```text
 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.

Time: 1.56 sec
26 tokens/sec
Max CUDA memory allocated: 1.51 GB
```

### 1.6 KV Cache

因为这种自回归的输出，每次 next token 的预测都要跟前缀计算注意力，如果每次都对整个前缀算一遍 k v 的话就很慢，所以会有 KV Cache 这种东西，用于加速模型推理。

除了 KV Cache，后面还会用模型编译来加快推理。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1790413082920_image.png)

我们只需在之前的文本生成函数中启用 KV Cache 即可。

```python
from reasoning_from_scratch.qwen3 import KVCache

@torch.inference_mode()
def generate_text_basic_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])   #A
    model.reset_kv_cache()                            #A

    out = model(token_ids, cache=cache)[:, -1]        #B
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break
        yield next_token
        out = model(next_token, cache=cache)[:, -1]   #C
```

- `#A` 初始化 KV 缓存
- `#B` 第一轮，和之前一样把整个输入提供给模型
- `#C` 后续的迭代只把 `next_token` 作为输入喂给模型

我们来给这个函数计时，看看它是否真的带来性能收益：

```python
start_time = time.time()
generated_ids = []
for token in generate_text_basic_stream_cache(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
    next_token_id = token.squeeze(0)
    generated_ids.append(next_token_id)
end_time = time.time()

output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)
```

输出是：

```text
 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform tasks such as answering questions, writing text, and even creating content based on user input.

Time: 1.44 sec
27 tokens/sec
Max CUDA memory allocated: 1.46 GB
```

因为生成的文本太短了，在 4090 上这点运算不是瓶颈，所以效果不是很明显（

### 1.7 PyTorch 模型编译
`torch.compile` 会分析模型内部的**计算图**，尝试把成组的运算转换成更优化的 kernel，从而减少 Python 开销和其他执行低效之处。这能改善文本生成时的运行时性能，尤其是当我们**在循环中反复调用同一段模型代码**时。

这一点在后面跑更大规模评估和更复杂的推理工作流时会很重要，因为哪怕每步只有 modest 的提速，也能省下可观的总运行时间。

```python
major, minor = map(int, torch.__version__.split(".")[:2])
if (major, minor) >= (2, 8):
    # This avoids retriggering model recompilations
    # in PyTorch 2.8 and newer
    # if the model contains code like self.pos = self.pos + 1
    torch._dynamo.config.allow_unspec_int_on_nn_module = True

model_compiled = torch.compile(model)
```

值得注意的是，**第一次**使用编译后的模型执行时可能比平时更慢，因为要做初始的编译和优化。为了更好地衡量性能提升，我们会把文本生成过程重复跑多次。

首先，我们用**不带缓存**的那个生成函数来测试。

```python
for i in range(3):                                       #A
    start_time = time.time()
    generated_ids = []
    for token in generate_text_basic_stream(
        model=model_compiled,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id)
    end_time = time.time()

    if i == 0:                                           #B
        print("\n\nWarm-up run")                         #B
    else:
        print(f"\n\nTimed run {i}:")
    output_token_ids_tensor = torch.cat(generated_ids, dim=0)
    generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)
    print(f"\n{30*'-'}\n")
```

- `#A` 我们把 token 生成跑三遍
- `#B` 第一遍被标记为 "Warm-up run"（预热运行）

输出如下：

```text
 Large language models are
 artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.

Warm-up run


Time: 0.51 sec
80 tokens/sec
Max CUDA memory allocated: 1.48 GB

------------------------------

 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.

Timed run 1:


Time: 0.49 sec
83 tokens/sec
Max CUDA memory allocated: 1.48 GB

------------------------------

 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.

Timed run 2:
...
Max CUDA memory allocated: 1.48 GB

------------------------------

Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

效 果 拔 群

对比下 KV 缓存版本。

```python
for i in range(3):
    start_time = time.time()
    generated_ids = []
    for token in generate_text_basic_stream_cache(
        model=model_compiled,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id)
    end_time = time.time()
    if i == 0:
        print("\n\nWarm-up run")
    else:
        print(f"\n\nTimed run {i}:")
    output_token_ids_tensor = torch.cat(generated_ids, dim=0)
    generate_stats(
        output_token_ids_tensor, tokenizer, start_time, end_time
    )
    print(f"\n{30*'-'}\n")
```

输出如下：

```text
 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.

Warm-up run


Time: 153.92 sec
0 tokens/sec
Max CUDA memory allocated: 1.52 GB

------------------------------

 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.

Timed run 1:


Time: 0.53 sec
76 tokens/sec
Max CUDA memory allocated: 1.46 GB

------------------------------

 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.

Timed run 2:
...
Max CUDA memory allocated: 1.46 GB

------------------------------

Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

### 1.9 小结
使用 LLM 生成文本涉及多个关键步骤：

- **搭建编码环境**，以便运行 LLM 代码并安装必要的依赖。
- **加载一个预训练的 base LLM**（例如 Qwen3 0.6B），后续章节会为它扩展推理能力。
- **初始化并使用 tokenizer**，它把文本输入转换成 token ID，并把输出解码回人类可读的形式。
- LLM 的文本生成遵循**顺序式（自回归）**过程：模型一次生成一个 token，通过预测下一个最可能的 token 来实现。
- 文本生成的速度与效率可以通过以下方式提升：
  - **KV 缓存**：存储中间状态，避免在每一步重新计算此前已经处理过的输入 token。
  - **模型编译**：使用 `torch.compile` 优化运行时性能。

## 二、REFERENCE

[Build a Reasoning Model (From Scratch)，Sebastian Raschka](https://www.manning.com/books/build-a-reasoning-model-from-scratch?utm_source=raschka&utm_medium=affiliate&utm_campaign=book_raschka2&a_aid=raschka&a_bid=4c3c5398&chan=mm_github)

[[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)](https://github.com/rasbt/reasoning-from-scratch/tree/main)

