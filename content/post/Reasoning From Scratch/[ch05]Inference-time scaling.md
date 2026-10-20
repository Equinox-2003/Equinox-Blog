---
title: "[Ch05]Inference-time scaling via self-refinement"
description: "一个自我改进（self-refinement）循环，让 LLM 迭代地改进自己的答案"
date: 2026-10-14T14:14:18+08:00
lastmod: 2026-10-14T14:14:18+08:00
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

另一种推理时做 scaling 的方法便是 self-refinement。其实仍然是 promp engineering。对于 llm 的输出，我们按照某种规则设置一个打分函数（也可以用 LLM as Judge）。

然后让模型对自己的输出进行 refine，如果得分有提升，我们就迭代下去。

不过这不一定必然使得模型输出得到改进，也可能强化错误输出。





## 一、Inference-Time Scaling via Self-Refinement

### 1.1 对 response 打分并迭代改进

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791959572130_image.png)

**CoT（chain-of-thought prompting）**通过修改 prompt，激发 model 更长输出，提高答案准确率。这对那些天生不给出"类推理"解释的 base 模型特别有用。已经训练成推理模型的模型通常**不会**从这种推理时缩放中获益，因为它们本来就会解释自己的答案。

**Self-consistency** 让模型并行产生若干答案，然后对这些候选做一次简单的多数投票来选出最终答案。

尽管 Self-consistency 非常简单，它却常常带来**大幅**的准确率提升，这也是它在"准确率优先于延迟"的 LLM 应用中成为常见选择的原因。比如  DeepSeekMath-V2 和 Google 的 Gemini 3 Deep Think 模式

自洽性的一个缺点是：**多数投票要求答案足够短、彼此可比较**。

所以一种更为通用的技术便是**自我改进**（self-refinement），让 LLM 学会迭代地改进自己的答案。

不过在实现自我改进之前，我们先实现**打分函数**（scoring functions），用来比较和排序不同的答案。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791960032794_image.png)

### 1.2 加载预训练 model

```python
import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer
)

device = get_device()

model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False
)
```

然后用温度缩放和 top-p 采样做一次简单生成：

```python
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache
)

raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)
prompt_cot = prompt + "\n\nExplain step by step."

torch.manual_seed(0)
response_1 = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9
)

```

```text
 ### Step 1: Understand the problem

The problem states that half the value of \(3x - 9\) is equal to \(x + 37\). We need to find the value of \(x\).

### Step 2: Translate the problem into an equation

Let's translate the given statement into an equation. Half of \(3x - 9\) is equal to \(x + 37\).

\[
\frac{1}{2}(3x - 9) = x + 37
\]

### Step 3: Solve the equation

We will solve the equation step by step to find the value of \(x\).

\[
\frac{1}{2}(3x - 9) = x + 37
\]

First, distribute the \(\frac{1}{2}\) on the left side:

\[
\frac{1}{2} \cdot 3x - \frac{1}{2} \cdot 9 = x + 37
\]
...
### Final Answer

\[
\boxed{83}
\]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

当然，换个种子输出就不同了：

```python
torch.manual_seed(3)
response_2 = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9,
)
```

```text
 We start with the equation:

\[
\frac{1}{2} \times (3x - 9) = x + 37
\]

Step 1: Multiply both sides of the equation by 2 to eliminate the fraction:

\[
2 \times \frac{1}{2} \times (3x - 9) = 2 \times (x + 37)
\]

Simplifying both sides:

\[
3x - 9 = 2x + 74
\]

Step 2: Subtract \(2x\) from both sides to get all the \(x\) terms on one side:

\[
3x - 2x - 9 = 74
\]

Simplifying:
...
Final Answer:

\[
\boxed{83}
\]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

虽然答案都对，但是第二个回复显然短得多：

```python
print("Response 1 characters:", len(response_1))
print("Response 1 tokens:", len(tokenizer.encode(response_1)))
print("\nResponse 2 characters:", len(response_2))
print("Response 2 tokens:", len(tokenizer.encode(response_2)))
```

```text
Response 1 characters: 1498
Response 1 tokens: 586

Response 2 characters: 533
Response 2 tokens: 231
```

更短的回复**不一定**更好。如果两个回复都得到了同一个正确答案，判断哪个更好并不直观，它往往取决于人类在**清晰度、有用性、以及中间步骤的部分正确性**上的偏好。

>   对回复的**中间步骤**打分仍然是一个活跃的研究方向。

如果两个回复的定性价值相当，那么有一点是确定的：**更短的回复更便宜**，因为它需要生成的 token 更少，因此更受欢迎。

### 1.3 启发式规则评分

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791960415327_image.png)

打分器可以是一个**启发式**（heuristic），也就是简单的规则式方法；也可以是**另一个 LLM** 来给答案评级，这通常叫 **LLM-as-a-judge**；还可以依赖**内部的概率分数或似然**。

>   LLM-as-a-judge 例：
>
>   ![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791960474675_image.png)

然后写一个简单的打分器：

```python
from reasoning_from_scratch.ch03 import extract_final_candidate
import math

def heuristic_score(
    answer,
    prompt=None,                                  #A
    brevity_bonus=500.0,
    boxed_bonus=2.0,
    extract_bonus=1.0,
    fulltext_bonus=0.0,
):
    score = 0.0

    #B 奖励那些有最终 boxed 值的答案
    cand = extract_final_candidate(answer, fallback="none")
    if cand:
        score += boxed_bonus

    #C 如果没有 boxed 值，就给弱一点的奖励
    else:
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus
        else:
            cand = extract_final_candidate(
                answer, fallback="number_then_full"
            )
            if cand:
                score += fulltext_bonus

    #D 加一个随文本长度衰减的简洁奖励
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score
#A 一个占位参数，本节里不使用它
#B 奖励那些有最终 boxed 值的答案
#C 如果没有 boxed 值，就给弱一点的奖励
#D 加一个随文本长度衰减的简洁奖励
```

这里就是根据答案抽取的干净程度以及输出的长度有多短来打分的。

然后绘制一个惩罚曲线：

```python
import matplotlib.pyplot as plt

def plot_brevity_curve(brevity_bonus, max_len=2048):
    lengths = torch.arange(1, max_len)
    scores = 1.5 * torch.exp(-lengths / brevity_bonus)

    plt.figure(figsize=(4, 3))
    plt.plot(lengths, scores)
    plt.xlabel("Text length (number of characters)")
    plt.ylabel("Score contribution")
    plt.tight_layout()
    #plt.savefig("brevity_curve.pdf")
    plt.show()

plot_brevity_curve(500)
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791960934585_image.png)

### 1.4 token logits

LLM 是基于 next-token 的概率预测，那么我们可以利用这个来做一套评分指标。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791961451611_image.png)

对于模型的一串输出，我们根据每个token的概率，计算该输出的**似然估计**。这其实量化了模型对于答案的置信度。不过，**模型置信并不自动意味着正确**。置信度高也只是可能拟合了学到的模式，但是这个模式也很可能是错误的。所以在工程中，还是需要验证器、外部工具或检索或者self-consistency 纠正这种“自信答错”。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791961676604_image.png)

如何查询特定 token 的 logit？

我们把文本过一遍model，然后就可以查询每个token的logit，把想要打分的token的logit拿出来就好了。

不过很多时候我们会更想比较完整答案而非单个token，所以会把token的概率相乘，从而计算**联合概率（joint probability）。**

写一个辅助函数来计算 next-token 概率：

```python
@torch.inference_mode()
def calc_next_token_probas(model, tokenizer, prompt, device, show=True):

    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    logits = model(token_ids.unsqueeze(0)).squeeze(0)                #A
    all_probas = torch.softmax(logits, dim=-1)                       #A

    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)   #B

    next_ids = token_ids[1:]    #C

    next_token_probas = all_probas[t_idx, next_ids]    #D

    prod_next_token_probas = torch.prod(next_token_probas)    #E
    if show:
        print("Next-token probabilities:", next_token_probas)
        print("Joint probability:", prod_next_token_probas)

    else:
        return next_token_probas, prod_next_token_probas

#A 和文本生成函数一样，先拿 logits 和概率
#B 选出我们要打分的位置（这里：全部）
#C 因为我们手里就有这段文本，所以真实的下一个 token 是已知的
#D 取出每个下一个 token 的概率
#E 序列的似然 = 这些概率分数的连乘
```

然后测试一下：

```python
torch.set_printoptions(precision=4, sci_mode=True)
calc_next_token_probas(
    model, tokenizer, device=device,
    prompt="The capital of Germany is Berlin"
)
```

```text
Next-token probabilities: tensor([6.5327e-05, 4.6289e-01, 1.6846e-02, 7.2266e-01, 1.6406e-01],
       device='cuda:0', dtype=torch.bfloat16)
Joint probability: tensor(6.0536e-08, device='cuda:0', dtype=torch.bfloat16)
```

```python
calc_next_token_probas(
        model, tokenizer, device=device,
        prompt="The capital of Germany is Bridge"
    )
```

```text
Next-token probabilities: tensor([6.5327e-05, 4.6289e-01, 1.6846e-02, 7.2266e-01, 2.7940e-07],
       device='cuda:0', dtype=torch.bfloat16)
Joint probability: tensor(1.0303e-13, device='cuda:0', dtype=torch.bfloat16)
```

### 1.5 token prob => log prob

因为连乘会有数值下溢的问题，所以我们对这些概率施加**对数缩放（logarithmic scaling）**。

```python
torch.set_printoptions(precision=4, sci_mode=False)
logits = torch.linspace(-2, 2, steps=7)
probas = torch.softmax(logits, dim=-1)
print(probas)
```

```text
tensor([0.0090, 0.0175, 0.0341, 0.0665, 0.1295, 0.2522, 0.4912])
```

```python
print(torch.log(probas))
```

```text
tensor([-4.7109, -4.0442, -3.3776, -2.7109, -2.0442, -1.3776, -0.7109])
```

或者直接 log_softmax：

```python
log_probas = torch.log_softmax(logits, dim=-1)
print(log_probas)
```

然后我们可以绘制一下 logits、softmax、log_softmax 的值：

```python
plt.figure(figsize=(9, 4))

#A
plt.subplot(1, 3, 1)
plt.bar(range(len(logits)), logits, color="C0", alpha=0.7)
plt.title("Logits")
plt.xlabel("Token index")
plt.ylabel("Value")
plt.grid(alpha=0.3)

#B
plt.subplot(1, 3, 2)
plt.bar(range(len(probas)), probas, color="C1", alpha=0.7)
plt.title("torch.softmax(logits)")
plt.xlabel("Token index")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(alpha=0.3)

#C
plt.subplot(1, 3, 3)
plt.bar(range(len(log_probas)), log_probas, color="C2", alpha=0.7)
plt.title("torch.log_softmax(logits)")
plt.xlabel("Token index")
plt.ylabel("Log-probability")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("logits_softmax_log_softmax.pdf")
plt.show()

#A 绘制 logits
#B 绘制 softmax 概率
#C 绘制 log-softmax 值
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791962415038_image.png)

如果我们认为**概率越大越好**，因为它表示模型对这个 token 赋予更多置信度；那么对 log 概率来说，**越接近零越好**，因为零对应概率 1，而非常负的值对应极小的概率。

用 log-logits 来计算联合概率还有一个好处就是 log 下可以转化成若干个 log 值累加。

然后我们写一个函数来计算下一个 token 的 log 概率：

```python
@torch.inference_mode()
def calc_next_token_logprobas(model, tokenizer, prompt, device, show=True):

    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    all_logprobas = torch.log_softmax(logits, dim=-1)    #A

    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    next_ids = token_ids[1:]
    next_token_logprobas = all_logprobas[t_idx, next_ids]

    sum_next_token_logprobas = torch.sum(next_token_logprobas)    #B

    if show:
        print("Next-token log-probabilities:", next_token_logprobas)
        print("Joint log-probability:", sum_next_token_logprobas)
    else:
        return next_token_logprobas, sum_next_token_logprobas

#A 我们现在用 log_softmax
#B 我们把连乘换成求和
```

```python
calc_next_token_logprobas(
        model, tokenizer, device=device,
        prompt="The capital of Germany is Berlin"
    )
```

```text
Next-token log-probabilities: tensor([-9.6250, -0.7695, -4.0938, -0.3262, -1.8047], device='cuda:0',
       dtype=torch.bfloat16)
Joint log-probability: tensor(-16.6250, device='cuda:0', dtype=torch.bfloat16)
```

```python
calc_next_token_logprobas(
    model, tokenizer, device=device,
    prompt="The capital of Germany is Bridge"
)
```

```text
Next-token log-probabilities: tensor([ -9.6250,  -0.7695,  -4.0938,  -0.3262, -15.0625], device='cuda:0',
       dtype=torch.bfloat16)
Joint log-probability: tensor(-29.8750, device='cuda:0', dtype=torch.bfloat16)
```

### 1.6 log-prob 打分

流程很简单，和上一节的流程非常像，主要做**两处修改**：

第一，**把 prompt 从分数计算中排除**，只对答案的 token 计算分数。第二，对 token log 概率取**平均**（而不是求和），这样比较**长度不同的两个序列**才公平。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791962805623_image.png)

比如下面这个示例 prompt 和 answer：

```python
example_prompt = "What is the capital of Germany?"
example_answer = " The capital of Germany is Berlin."

next_token_logprobas, sum_next_token_logprobas = calc_next_token_logprobas(
        model, tokenizer, device=device,
        prompt=example_prompt+example_answer,
        show=False
    )

print("Next-token logprobas:", next_token_logprobas)
print("Joint log-probability:", sum_next_token_logprobas)
```

```text
Next-token logprobas: tensor([-0.4512, -0.3281, -8.3125, -0.3828, -3.7969, -2.9375, -1.1641, -0.0021,
        -0.0232, -0.0031, -0.0093, -0.0767, -0.1602], device='cuda:0',
       dtype=torch.bfloat16)
Joint log-probability: tensor(-17.6250, device='cuda:0', dtype=torch.bfloat16)
```

可以看一下token数：

```python
print(len(tokenizer.encode(example_answer)))
```

```text
7
```

然后计算答案 token 的平均 log-prob：

```python
last_7 = next_token_logprobas[-7:]
print(last_7)
print(torch.mean(last_7))
```

```text
tensor([-1.1641, -0.0021, -0.0232, -0.0031, -0.0093, -0.0767, -0.1602],
       device='cuda:0', dtype=torch.bfloat16)
tensor(-0.2051, device='cuda:0', dtype=torch.bfloat16)
```

然后我们可以封装一个对 answer token 求平均 log-prob 的打分函数：

```python
@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):

    prompt_ids = tokenizer.encode(prompt)              #A
    answer_ids = tokenizer.encode(answer)              #A
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)   #B
    logprobs = torch.log_softmax(logits, dim=-1)       #B

    start = len(prompt_ids) - 1                        #C
    end = full_ids.shape[0] - 1                        #C

    #D
    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    #E
    return torch.mean(next_token_logps)

#A 分别编码 prompt 和答案的 token，以便后面拿到 prompt 的长度
#B 和前面 calc_next_token_logprobas 里一样
#C 答案 token 对应的位置区间
#D 和前面一样，只是改用 start 和 end
#E 对答案 token 的分数取平均
```

```python
score_1 = avg_logprob_answer(
    model, tokenizer,
    prompt="What is the capital of Germany?",
    answer=" The capital of Germany is Berlin.",
    device=device
)
print(score_1)
```

```text
tensor(-0.2051, device='cuda:0', dtype=torch.bfloat16)
```

```python
score_2 = avg_logprob_answer(
    model, tokenizer,
    prompt="What is the capital of Germany?",
    answer=" The capital of Germany is Bridge.",
    device=device
)
print(score_2)
```

```text
tensor(-3.8750, device='cuda:0', dtype=torch.bfloat16)
```

### 1.7 通过迭代反馈进行自我改进

自我改进是一种让 LLM **分析并精炼自己答案**的技术。如下图所示，LLM 先从对 prompt 的一个初始答案开始（就像常规使用 LLM 那样），然后它**批评**这个答案，再**改进**它。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791963195415_image.png)

先写一个基础 promt 和 answer：

```python
raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)

torch.manual_seed(123)
initial_response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.7,
    top_p=0.9
)
```

然后就得到了错误答案：

```text
 \boxed{18}
```

为了让 llm 来做critique，我们写一个 critique prompt：

```python
def make_critique_prompt(raw_prompt, draft):
    return (
        "You are a meticulous reviewer. Identify logical errors, missing "
        "steps, or arithmetic mistakes. If the answer seems correct, "
        "say so briefly. Then propose a concise plan to fix issues.\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Write a short critique and bullet-point fix plan "
        "(under ~120 words).\n"
        "Critique:"
    )

critique_prompt = make_critique_prompt(raw_prompt, initial_response)
torch.manual_seed(132)
critique = generate_text_stream_concat_flex(
    model, tokenizer, critique_prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.7,
    top_p=0.9,
)
```

上面这个批评提示让 LLM 写出了一份详尽的批评，其中**甚至包含了正确答案本身**：

```text
The question seems to have a logical error in its setup. The statement "Half the
value of $3x-9$ is $x+37$" is incorrect because half of $3x-9$ should be $(3x-
9)/2$, not $3x-9$. The equation should be $\frac{1}{2}(3x-9) = x + 37$.

Fix Plan:
1. Correct the equation to $\frac{1}{2}(3x-9) = x + 37$.
2. Multiply both sides by 2 to eliminate the fraction: $3x - 9 = 2(x + 37)$.
3. Distribute the 2 on the right side: $3x - 9 = 2x + 74$.
4. Subtract $2x$ from both sides: $x - 9 = 74$.
5. Add 9 to both sides: $x = 83$.
```

但**中间有明显的胡说**：原文那句 "half of 3x−9 should be (3x−9)/2, not 3x−9" 至少还在讲道理，本机这句 "the final answer is incorrect as it should be 18, not 36" 则完全莫名其妙，凭空捏出一个 36。批评里也没有指出"18 错在哪一步"。

>   所以这些方法也不总是可靠的。

最后我们需要精炼 prompt 来改进原答案：

```python
def make_refine_prompt(raw_prompt, draft, critique):
    return (
        "Revise the answer using the critique. Keep it concise and "
        "end with a final boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\n"
        "Revised answer:"
    )

refine_prompt = make_refine_prompt(raw_prompt, initial_response, critique)
torch.manual_seed(123)
revised_answer = generate_text_stream_concat_flex(
    model, tokenizer, refine_prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.7,
    top_p=0.9,
)
```

虽然 base model 的指令遵循能力一般般，但还是生成了正确答案：

```python
 \
boxed{18}

Explanation: \begin{align*}
\frac{1}{2}(3x - 9) &= x + 37 \\
3x - 9 &= 2(x + 37) \\
3x - 9 &= 2x + 74 \\
3x - 2x &= 74 + 9 \\
x &= 83
\end{align*}
```

### 1.8 self-refinement loop

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791963624156_image.png)

带可选打分的自我改进循环。模型先产生一个初始答案，然后批评它，再基于批评生成一份修订答案。两份答案都可以用本章的打分函数（例如 logprob 打分）来评估，并且**只有当修订答案的分数优于前一份时**才被接受。

```python
def self_refinement_loop(
    model,
    tokenizer,
    raw_prompt,
    device,
    iterations=2,
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=None,
    prompt_renderer=render_prompt,
    prompt_suffix="",
    verbose=False,
    temperature=0.7,
    top_p=0.9,
):
    steps = []

    #A 初始回复（草稿）
    prompt = prompt_renderer(raw_prompt) + prompt_suffix
    current_full = generate_text_stream_concat_flex(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_response_tokens,
        verbose=False,
        generate_func=generate_text_top_p_stream_cache,
        temperature=temperature,
        top_p=top_p,
    )

    current_extracted = extract_final_candidate(
        current_full, fallback="number_then_full"
    )
    if score_fn:
        current_score = score_fn(answer=current_full, prompt=prompt)
    else:
        current_score = 0.0

    #B 跑一轮或多轮
    for it in range(iterations):
        draft_before_full = current_full
        draft_before_extracted = current_extracted
        score_before = current_score

        #C 批评这个回复
        critique_prompt = make_critique_prompt(
            raw_prompt, draft_before_full
        )
        critique_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=critique_prompt,
            device=device,
            max_new_tokens=max_critique_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )

        #D 精炼这个回复
        refine_prompt = make_refine_prompt(
            raw_prompt, draft_before_full, critique_full
        )
        revised_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=refine_prompt,
            device=device,
            max_new_tokens=max_response_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )

        revised_extracted = extract_final_candidate(
            revised_full, fallback="number_then_full"
        )
        if score_fn:
            revised_score = score_fn(
                answer=revised_full, prompt=prompt   # 这里仍然用最初的 prompt
            )
        else:
            revised_score = 0.0

        #E 记录结果
        step = {
            "iteration": it + 1,
            "draft_full": draft_before_full,
            "draft_extracted": draft_before_extracted,
            "critique": critique_full,
            "revised_full": revised_full,
            "revised_extracted": revised_extracted,
            "score_before": score_before,
            "score_after": revised_score,
        }
        steps.append(step)

        if verbose:
            print(
                f"[Refinement {it+1}/{iterations}]"
                f"\nCurrent: {draft_before_extracted}"
                f"\nRevised: {revised_extracted}"
                f"\nScore before: {score_before:.3f}"
                f"\nScore after: {revised_score:.3f}"
                f"\n{'=' * 25}\n"
            )

        #F 如果修订版不差，就接受它
        if revised_score >= current_score:
            current_full = revised_full
            current_extracted = revised_extracted
            current_score = revised_score

    return {
        "final_full": current_full,
        "final_extracted": current_extracted,
        "steps": steps,
    }
#A 初始回复（草稿）
#B 跑一轮或多轮
#C 批评这个回复
#D 精炼这个回复
#E 记录结果
#F 如果修订版不差，就接受它
```

然后构造一个平均 log-prob 的打分器：

```python
from functools import partial

avg_logprob_score = partial(
    avg_logprob_answer,
    model=model,
    tokenizer=tokenizer,
    device=device
)
```

然后就可以在自我改进 loop 里面用这个函数了：

```python
torch.manual_seed(1)

results_logprob = self_refinement_loop(
    model=model,
    tokenizer=tokenizer,
    raw_prompt=raw_prompt,
    device=device,
    iterations=2,
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=avg_logprob_score,
    verbose=True,
    temperature=0.7,
    top_p=0.9,
)
```

```text
[Refinement 1/2]
Current: 10
Revised: 83
Score before: -0.855
Score after: -0.226
=========================
[Refinement 2/2]
Current: 83
Revised: 83
Score before: -0.226
Score after: -1.320
=========================
```





## 二、REFERENCE

[Build a Reasoning Model (From Scratch)，Sebastian Raschka](https://www.manning.com/books/build-a-reasoning-model-from-scratch?utm_source=raschka&utm_medium=affiliate&utm_campaign=book_raschka2&a_aid=raschka&a_bid=4c3c5398&chan=mm_github)

[[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)](https://github.com/rasbt/reasoning-from-scratch/tree/main)

