---
title: "[CH04]Improving reasoning with inference-time scaling"
description: "Let's think step by step"
date: 2026-10-10T20:55:09+08:00
lastmod: 2026-10-10T20:55:09+08:00
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

通过 prompt 激活 CoT 来增强 reasoning 能力是一种比较常见且容易的方式，计算量增加，但是提高了推理质量。

关于 CoT 之前我也有整理：[Paper-Reading](https://github.com/Equinox-2003/Paper-Reading)，见 CoT 部分





## 一、inference-time scaling

### 1.1 什么是 inference-time scaling

总的来说，提升推理能力主要有两条路线：

1.  增加**训练**算力，
2.  增加**推理**算力（也叫 inference-time scaling 或 test-time scaling）。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791637597817_image.png)

LLM 通常两者**兼用**：靠大量的训练时算力加上更多的推理时算力来共同提升推理能力。

**inference-time scaling（test-time compute scaling）**，指的是在模型**已经训练完成之后**，在生成答案的时候多花一些计算来提升回复质量。我们不改变模型权重，而是让同一个训练好的模型在每道题上做更多工作——例如生成更多 token、想得更久、采样多个答案，或者反复精炼自己的答案。

我们聚焦三种实用且基础的方法：

1.  **扩展思维链回复**，即提示模型解释自己的推理过程。这是最简单的技术，但能大幅提升准确率。
2.  **自洽性并行采样（self-consistency）**，让模型生成多份回复，选出出现次数最多的那个。
3.  **迭代式自我精炼（self-refinement）**，让模型跨多个步骤检查并改进自己的推理和答案。（下一篇blog再做）

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791637827959_image.png)

这些方法的 Scale 体现在，推理时生成更多 **token**。

### 1.2 加载预训练模型

```python
import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer
)

device = get_device()
device = torch.device("cpu") #A

model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False
)

#A 删掉这一行就可以在 GPU 上跑（如果你的机器支持）
```

拿上一章用过的 MATH-500 里的一道题来试试这个模型：

```python
from reasoning_from_scratch.ch03 import render_prompt

raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)
print(prompt)
```

格式化后的 prompt 如下：

```markdown
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}

Question:
Half the value of $3x-9$ is $x+37$. What is the value of $x$?

Answer:
```

为了方便比较不同策略，所以改一下文本生成函数：

```python
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache

def generate_text_stream_concat_flex(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
    generate_func=None, #A
    **generate_kwargs #A
):
    if generate_func is None: #B
        generate_func = generate_text_basic_stream_cache

    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)

    generated_ids = []
    for token in generate_func(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        **generate_kwargs, #C
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())

        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)

#A 新增参数：接受一个文本生成函数，以及额外的参数
#B 如果没传生成函数，就默认用 generate_text_basic_stream_cache，和第 3 章一样
#C 需要时可以把额外的参数传给这个生成函数
```

这样方便我们后面选择温度采样、截断之类的。

```python
response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_basic_stream_cache
)
```

```text
 \boxed{20}
```

注意这个答案是**错的**，正确答案是 **83**。

### 1.3 用思维链提示生成更好的回复

思维链提示是一种经典、简单又有效的技术：它修改输入 prompt，去鼓励 LLM 生成一段解释，也就是所谓的**思维链**（chain-of-thought，也叫推理链），如下图：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791638286770_image.png)

```python
prompt_cot = prompt + " \n\nExplain step by step."

response_cot = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=True,
)
```

```text
 To solve the problem, we need to find the value of \( x \) such that half the value of \( 3x - 9 \) is equal to \( x + 37 \).

### Step 1: Set up the equation
According to the problem, half the value of \( 3x - 9 \) is equal to \( x + 37 \). This can be written as:
\[
\frac{1}{2}(3x - 9) = x + 37
\]

### Step 2: Eliminate the fraction
To eliminate the fraction, multiply both sides of the equation by 2:
\[
2 \cdot \frac{1}{2}(3x - 9) = 2(x + 37)
\]
Simplifying both sides:
\[
3x - 9 = 2x + 74
\]

### Step 3: Solve for \( x \)
Subtract \( 2x \) from both sides to isolate \( x \):
\[
3x - 2x - 9 = 74
\]
Simplify:
\[
...

### Final Answer:
\[
\boxed{83}
\]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

trade-off：模型现在答对了，但它比之前**多花了大量 token**。

虽然模型在这个例子里给出了正确答案，但**并非所有问题都能从思维链提示中获益**。在一些很简单的题目上，它甚至可能让模型表现变差——因为模型可能会生成错误的解释，然后把自己带偏。这个现象也被叫做 **"过度思考"（overthinking）**。

也不是每个模型都能从"Explain step by step"（或类似指令）里获益。这里我们用的是一个简单 base 模型，它并不总会主动生成解释，所以思维链提示能带来明显帮助。而**训练过的推理模型**——比如上一章练习里用到的 Qwen3 `"reasoning"` 变体——本来就会在回复旁边写好解释，不需要、也不会从这种思维链提示里额外获益。

>   Q：**思维链为什么能提升准确率？**
>
>   A：
>
>   思维链提示要求模型把导向最终答案的**中间步骤**写出来。这从两个很实际的方面帮忙：
>
>   -   第一，逐步走一遍给了模型更多**自我纠正**的机会。
>   -   第二，分步推理**匹配了很多训练样本的写法**。比如大型数学和逻辑数据集里通常带着详细的解答过程，所以要求思维链相当于让模型对齐它已经学过的模式。
>
>   但与此同时，思维链**不是正确性的保证**。它照样可能产出错误的推理；对特别简单的问题，它甚至可能引入一些不必要的步骤，反而带来更多错误。换句话说，思维链能提升很多推理任务的准确率，但并非**普适**有益。
>
>   总的来说，思维链回答并没有给模型提供**新知识**，而是改变了模型**使用已有知识的方式**。这种转变在很多情况下能带来更可靠的答案——对数学、代码、逻辑问题以及其他多步问题尤其如此。

### 1.4 temperature scaling

**自洽性采样（self-consistency sampling）**，正式出处是 Google Research 的论文 *Self-Consistency Improves Chain-of-Thought Reasoning in Language Models*（https://arxiv.org/abs/2203.11171 ）。

>   之前我有对这个论文写过博客：[Self Consistency泛读](https://equinox.wiki/post/paper-reading/cot/self-consistency%E6%B3%9B%E8%AF%BB/)

步骤是比较简单的：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791638286770_image.png)

#### 1.4.1 next-token

```python
ex_prompt = "The capital of Germany is"

response = generate_text_stream_concat_flex(
    model, tokenizer, ex_prompt, device,
    max_new_tokens=1, verbose=True
)
```

```text
 Berlin
```

看起来很简单，但底层其实发生了好几个步骤：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791638954586_image.png)

生成文本时输入先被转成 token ID：

```python
input_token_ids = torch.tensor(
    tokenizer.encode(ex_prompt), device=device
).unsqueeze(0)
print(input_token_ids)
```

```text
tensor([[ 785, 6722,  315, 9856,  374]])
```

然后我们拿出model最后一层输出的logits：

```python
with torch.inference_mode():
    next_token_logits = model(input_token_ids)[:, -1]
print(next_token_logits.shape)
```

```text
torch.Size([1, 151936])
```

```python
max_token_id = torch.argmax(next_token_logits)
print(f"Token ID: {max_token_id}")
print(f"Decoded token: '{tokenizer.decode([max_token_id])}'")
```

```text
Token ID: 19846
Decoded token: ' Berlin'
```

```python
import matplotlib.pyplot as plt

def plot_scores_bar(
    next_token_logits, start=19_800, end=19_900,
    arrow=True, ylabel="Logit value"
):
    x = torch.arange(start, end) #A
    logits_section = next_token_logits[0, start:end].float().cpu() #B

    plt.bar(x, logits_section) #C
    plt.xlabel("Vocabulary index")
    plt.ylabel(ylabel)

    if arrow: #D
        max_idx = torch.argmax(logits_section)
        plt.annotate(
            "Berlin",
            xy=(x[max_idx], logits_section[max_idx]),
            xytext=(x[max_idx] - 25, logits_section[max_idx] - 2),
            arrowprops={
                "facecolor": "black", "arrowstyle": "->", "lw": 1.5
            },
            fontsize=10,
        )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_scores_bar(next_token_logits)

#A 选取词表的一个子区间
#B .cpu() 是 to(torch.device("cpu")) 的简写
#C 绘制所选区间内的 logits（分数）
#D 画一个箭头来标出最大的那个分数
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791638954586_image.png)

#### 1.4.2 *temperature* 

温度——更准确地说，是我们选定的 **温度参数**——改变 logits（token 分数）有多尖锐或多平缓，进而影响下一个 token 如何被选中。

```python
def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature
```

```python
def plot_logits_with_temperature(
    next_token_logits, start=19_800, end=19_900,
    temps=(0.5, 5.0),
):
    x = torch.arange(start, end)
    logits_orig = next_token_logits[0, start:end].float().cpu()

    logits_scaled = [ #A
        scale_logits_by_temperature(logits_orig, T) for T in temps #A
    ] #A

    plt.plot(x, logits_orig, label="Original logits", lw=2) #B
    plt.plot( #B
        x, logits_scaled[0], #B
        label=f"T={temps[0]} (sharper)", ls="--", lw=1 #B
    ) #B
    plt.plot( #B
        x, logits_scaled[1], #B
        label=f"T={temps[1]} (flatter)", ls=":", lw=3 #B
    ) #B

    # Highlight max logit
    max_idx = torch.argmax(logits_orig) #C
    plt.annotate( #C
        "Berlin",
        xy=(x[max_idx], logits_orig[max_idx]),
        xytext=(x[max_idx] - 25, logits_orig[max_idx] + 2),
        arrowprops={"facecolor": "black", "arrowstyle": "->", "lw": 1.5},
        fontsize=12,
    )

    plt.xlabel("Vocabulary index")
    plt.ylabel("Logit value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_logits_with_temperature(
    next_token_logits,
    temps=(0.5, 5.0) #D
)

#A 应用温度缩放
#B 绘制所选区间内的 logits（分数）
#C 画箭头标出最大的分数
#D 用温度 0.5 和 5 执行重缩放并绘图
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791639130711_image.png)

较大的温度（5.0）让分数分布**平坦得多**，而较小的温度（0.5）让它**尖锐得多**。

>   Q：
>
>   **为什么叫"温度"（WHY TEMPERATURE?）**
>
>   A：
>
>   "温度"这个词来自物理学：温度控制系统中的随机性或运动程度。在 LLM 里我们借用同一个想法，来控制模型选择下一个 token 时是更"自信"还是更"有创造性"——接下来两节会看到具体效果。

#### 1.4.3 从概率分布中采样下一个 token

```python
rescaled_logits = scale_logits_by_temperature(next_token_logits, 5.0) #A

next_token_probas = torch.softmax( #B
    rescaled_logits, dim=-1
)

#A 步骤 3.2：重缩放下一个 token 的分数
#B 步骤 3.3：把重缩放后的 logits 转成概率分数
```

```python
print("Probability sum:", torch.sum(next_token_probas))
```

```text
Probability sum: tensor(1., dtype=torch.bfloat16)
```

```python
plot_scores_bar(
    next_token_probas, arrow=False, ylabel="Probability value"
)
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791639285588_image.png)

```python
print("Token ID 19,846 probability:", next_token_probas[:, 19846])
```

```text
Token ID 19,846 probability: tensor([0.0003], dtype=torch.bfloat16)
```

```python
print("Highest probability:", max(next_token_probas.squeeze(0)))
```

```text
Highest probability: tensor(0.0003, dtype=torch.bfloat16)
```

采样过程可以这样实现：

```python
torch.manual_seed(123)
print(
    "Sampled token:",
    torch.multinomial(next_token_probas.cpu(), num_samples=1)
)
```

写一个函数来采样多个token：

```python
def count_samples(probas, num_samples=1000, threshold=1, tokenizer=None):
    samples = torch.multinomial( #A
        probas.cpu(), num_samples=num_samples, replacement=True
    )
    counts = torch.bincount(samples.squeeze(0), minlength=1) #B

    for i, c in enumerate(counts):
        if c > threshold: #C
            if tokenizer is None:
                print(f"Vocab index {i}: {c.item()}x")
            else:
                print(f"'{tokenizer.decode([i])}': {c.item()}x")

#A 按照概率抽取样本
#B 统计每个索引被选中的次数
#C 打印高频出现的词表索引（条目）
```

```python
torch.manual_seed(123)
count_samples(next_token_probas, tokenizer=tokenizer)
```

```text
'}': 2x
' your': 2x
' Oh': 2x
' dos': 2x
' grocery': 2x
' Going': 2x
' Troy': 2x
' overdose': 2x
'וא': 2x
' thêm': 2x
```

试一个更低的温度 0.35：

```python
torch.manual_seed(123)
probas_lowT = torch.softmax(
    scale_logits_by_temperature(next_token_logits, 0.35), dim=-1
)
```

```python
count_samples(probas_lowT, tokenizer=tokenizer)
```

```text
'
': 2x
' __': 130x
' Berlin': 453x
' ____': 173x
' ______': 213x
' Hamburg': 5x
' _____': 19x
```

#### 1.4.4 温度缩放加进文本生成函数

然后把文本生成函数改成带 temperature 的：

```python
from reasoning_from_scratch.qwen3 import KVCache

@torch.inference_mode()
def generate_text_temp_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    out = model(token_ids, cache=cache)[:, -1]  #A
    for _ in range(max_new_tokens):

        ########################################
        # NEW:
        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            logits = scale_logits_by_temperature(out, temperature)  #B
            probas = torch.softmax(logits, dim=-1)  #C
            next_token = torch.multinomial(probas.cpu(), num_samples=1)  #D
            next_token = next_token.to(orig_device)

        #########################################
        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]
#A 步骤 3.1：拿到 logits
#B 步骤 3.2：对 logits 施加温度缩放
#C 步骤 3.3：转成概率
#D 步骤 3.4：按概率采样 token
```

```python
torch.manual_seed(123)
response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_temp_stream_cache,  #A
    temperature=1.1
)
#A 使用新的、基于温度缩放的文本生成函数
```

```text
 We begin with the equation \(\frac{1}{2}(3x - 9) = x + 37\).
First, distribute the \(\frac{1}{2}\) on the left-side of the equation to get \(3x - 9 = 2(x + 37)\).
Then simplify the right side of the equation by using the distributive property and combining like terms: \(3x - 9 = 2x + 74\).
Next, subtract \(2x\) from both sides of the equation to get \(x - 9 = 2x + 74\).
Next, subtract \(x\) from both sides of the equation to reduce the number of \(x\) terms on the right side to exactly 1: \(-9 = x + 74\).
Finally, subtract 74 from both sides of the equation to get \(x = -83\).
The value of \(x\) is \(\boxed{-83}\).
```

### 1.5 top-p 采样

温度缩放加上 `torch.multinomial` 采样会**提高**回复的多样性——好坏参半。具体说，按上一节的做法，我们可能采到和用户查询毫无关系的"怪 token"。

我们通过增加一个 **top-p 滤波器**来改进采样过程，让那些置信度极低的 token 不再被误采。top-p 采样过程也叫 **nucleus sampling（核采样）**。

#### 1.5.1 选出一组 top-p token

```python
toy_logits = torch.tensor(  #A
    [-0.7, -3.0, 0.1, -1.2, 2.0, -1.0, -0.5, -2.0, 0.3, 1.5]
)

toy_logits_scaled = scale_logits_by_temperature(toy_logits, 1.0)  #B
toy_probas = torch.softmax(toy_logits_scaled, dim=-1)  #C

plt.bar(  #D
    torch.arange(len(toy_probas)), toy_probas,
    alpha=0.5
)

plt.ylim([0, 1])
plt.xlabel("Vocabulary index")
plt.ylabel("Probability")
plt.show()
#A 步骤 3.1：拿到 logits（这里：10 个 token 的玩具 logits）
#B 步骤 3.2：施加温度缩放（这里用 1.0 作为占位值）
#C 步骤 3.3：转成概率
#D 把概率画成柱状图
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791639869663_image.png)

然后计算累积概率和：

```python
sorted_probas, sorted_idx = torch.sort(toy_probas, descending=True)  #A
cumsum = torch.cumsum(sorted_probas, dim=-1)  #B

plt.bar(
    torch.arange(len(sorted_probas)), sorted_probas,
    alpha=0.5
)
plt.step(
    torch.arange(len(cumsum)), cumsum,
    where="mid", color="C1", label="Cumulative sum"
)

plt.ylim([0, 1])
plt.xlabel("Token rank (sorted by probability)")
plt.ylabel("Probability")
plt.show()
#A 步骤 4.1：按概率降序排序
#B 步骤 4.2：计算累积和
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791639909347_image.png)

然后写一个简单的 top-p 过滤规则：

```python
top_p = 0.8
keep_mask = cumsum <= top_p
n_kept = torch.sum(keep_mask).item()
print("Cumulative sum:", cumsum)
print("Tokens kept:", n_kept)
```

```text
Cumulative sum: tensor([0.4538, 0.7290, 0.8119, 0.8798, 0.9170, 0.9475, 0.9701, 0.9886, 0.9969,
        1.0000])
Tokens kept: 2
```

一个常用的变种：

```python
keep_mask = (cumsum - sorted_probas) < top_p
n_kept = keep_mask.sum().item()
print("Tokens kept:", n_kept)
```

它符合 top-p 过滤的那个定义：**保留最小的若干个 token，使累积概率质量至少达到 p**。

可视化：

```python
plt.bar(
    torch.arange(len(sorted_probas)), sorted_probas,
    alpha=0.5, label="Sorted probabilities"
)
plt.step(
    torch.arange(len(cumsum)), cumsum, where="mid",
    color="darkorange", label="Cumulative sum"
)

#A
plt.axhline(
    top_p, color="red", linestyle="--",
    label=f"top_p = {top_p}"
)
plt.axvline(
    n_kept - 0.5, color="gray", linestyle=":",
    label=f"Top-p cutoff at {n_kept} tokens"
)

plt.xlabel("Token rank (sorted by probability)")
plt.ylabel("Probability")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(0, 1.05)
plt.show()

#A 高亮截断位置
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791640045813_image.png)

#### 1.5.2 带 top-p 的文本生成

```python
def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas

    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)  #A
    cumprobas = torch.cumsum(sorted_probas, dim=1)  #B

    prefix = cumprobas - sorted_probas  #C
    keep = prefix < top_p               #C
    keep[:, 0] = True                   #D

    kept_sorted = torch.where(          #E
        keep, sorted_probas,
        torch.zeros_like(sorted_probas)
    )
    #F
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)

    #G
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom
#A 步骤 4.1：按概率降序排序
#B 步骤 4.2：累积和
#C 步骤 4.3.1：保留"前缀累积质量（每个 token 之前的质量）< top_p"的 token
#D 当 top_p <= 0 时，作为兜底只保证保留概率最高的那个 token
#E 步骤 4.3.2：把截断之外的置零
#F 步骤 4.3.3：映射回原始顺序
#G 步骤 4.4：重归一化，使总和为 1
```

```python
@torch.inference_mode()
def generate_text_top_p_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.,
    top_p=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    out = model(token_ids, cache=cache)[:, -1]  #A
    for _ in range(max_new_tokens):

        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            logits = scale_logits_by_temperature(out, temperature)  #B
            probas = torch.softmax(logits, dim=-1)  #C

            # (新增) 步骤 4：对概率应用 top-p 过滤器
            probas = top_p_filter(probas, top_p)  #D

            next_token = torch.multinomial(probas.cpu(), num_samples=1)  #E
            next_token = next_token.to(orig_device)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]
#A 步骤 3.1：拿到 logits
#B 步骤 3.2：对 logits 施加温度缩放
#C 步骤 3.3：转成概率
#D （新增）步骤 4：对概率应用 top-p 过滤器
#E 步骤 3.4：按概率采样 token
```

跑一下：

```python
torch.manual_seed(123)
response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.5,
    top_p=0.8,
)
```

```text
 \boxed{18}
```

还是错的。

### 1.6 self-consistency

**自洽性采样（self-consistency sampling）**，主要就是基于前面的代码搭出一条完整的**基于投票的推理时流水线**。

写一个自洽性投票：

```python
from reasoning_from_scratch.ch03 import extract_final_candidate
from collections import Counter

def self_consistency_vote(
    model, tokenizer, prompt, device,
    num_samples=10, temperature=0.8, top_p=0.9, max_new_tokens=2048,
    show_progress=True, show_long_answer=False, seed=None,
):
    full_answers, short_answers = [], []

    # 1) 采样多个答案
    for i in range(num_samples):  #A
        if seed is not None:
            torch.manual_seed(seed + i + 1)

        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )

        # 2) 从每个答案中抽取最终的（短）答案
        short = extract_final_candidate(  #B
            answer, fallback="number_then_full"  #B
        )
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")

    # 3) 选出出现次数最多的最终答案（自洽性投票）
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)

    mc = counts.most_common()  #C
    if not mc:
        majority_winners, final_answer = [], None
    else:
        top_freq = mc[0][1]
        majority_winners = [s for s, f in mc if f == top_freq]
        final_answer = mc[0][0] if len(majority_winners) == 1 else None

    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }
#A 1) 采样多个答案
#B 2) 从每个答案中抽取最终的（短）答案
#C 3) 选出出现次数最多的最终答案（自洽性投票）
```

```python
results = self_consistency_vote(
    model,
    tokenizer,
    prompt,
    device=device,
    num_samples=5,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    seed=123,
    show_progress=True,
)
```

```text
[Sample 1/5] → '83'
[Sample 2/5] → '22'
[Sample 3/5] → '54'
[Sample 4/5] → '83'
[Sample 5/5] → '66'
```

如果加上CoT：

```python
results = self_consistency_vote(
    model,
    tokenizer,
    prompt + "\n\nExplain step by step.",
    device=device,
    num_samples=5,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    seed=123,
    show_progress=True,
)
```

```text
[Sample 1/5] → '83'
[Sample 2/5] → '83'
[Sample 3/5] → '83'
[Sample 4/5] → '83'
[Sample 5/5] → '83'
```

>   注意到出现绝对众数的时候抽样就可以停止了，也就是说前面的代码还可以加一个小优化，这里就略了。
>





## 二、REFERENCE

[Build a Reasoning Model (From Scratch)，Sebastian Raschka](https://www.manning.com/books/build-a-reasoning-model-from-scratch?utm_source=raschka&utm_medium=affiliate&utm_campaign=book_raschka2&a_aid=raschka&a_bid=4c3c5398&chan=mm_github)

[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch/tree/main)

