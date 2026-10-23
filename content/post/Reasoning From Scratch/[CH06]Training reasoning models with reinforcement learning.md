---
title: "[CH06]Training Reasoning Models With Reinforcement Learning"
description: "GRPO 训练模型推理能力"
date: 2026-10-23T18:23:12+08:00
lastmod: 2026-10-23T18:23:12+08:00
draft: false

categories:
  - Reasoning From Scratch
tags:
  - LLM
  - RLHF
  - RLVR

toc: true
math: true
mermaid: true
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1790353663827_image.png
---

<!--more-->





## 一、RL4LLM

提升 LLM 推理性能有两条正交的路：**推理时缩放**和**训练时缩放**，如下图所示。推理时缩放靠"每生成一个答案多花算力"来提高准确率；训练时缩放靠"在训练阶段多投入算力"来提高准确率。**本章讲的是后者**。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1792751896236_image.png)

二者是可以叠加的。

LLM 上常见的 RL 有两个：**推理训练**（reasoning training）和 **偏好调优**（preference tuning）。RL 通常用在指令微调之后，而偏好调优常常排在推理训练之后。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1792753399134_image.png)

DeepSeek 团队在 DeepSeek-R1 工作中证明了：**面向推理的 RL 也可以直接作用在预训练 base 模型上**，跳过指令微调和偏好调优

这样做得到的模型，在推理准确率上一般**弱于**走完完整流水线的模型，但它依然会表现出清晰、稳健的推理行为。更重要的是，直接在 base 模型上做推理训练，**避免了多个训练阶段的效应混在一起**，从而更容易把观察到的提升归因到"推理训练本身"。

### 1.1 RLHF

2022 年，InstructGPT 论文（https://arxiv.org/abs/2203.02155）提出了**基于人类反馈的强化学习**（reinforcement learning with human feedback，RLHF），一种用人类偏好标注来改变模型行为的训练方法。

>   事实上，RLHF 正是把 GPT-3 变成最初那个 ChatGPT 所用模型的关键成分——而 ChatGPT 可以说是 2022 年让 LLM 真正大众化的那个产品。

高层次地说，RLHF 是实现上一节提到的**偏好调优**阶段最流行的方法。在 RLHF 之前，LLM 主要靠预训练和（部分情况下的）监督微调来训练，这两者都建立在下一词预测目标之上。

RLHF 跳出了这种 token 级优化，**针对整个输出做优化**（在 RLHF 里，优化目标是"人类如何排序和评价 LLM 的输出"）。

为了更具体，看这个例子。假设提示词是："我想买一台既用来编程又用于日常使用的笔记本电脑，我应该考虑哪些因素？"

模型可能生成两个候选回复：

>   **回复 A**："你应该考虑 CPU、内存、存储、GPU、屏幕分辨率、电池续航、键盘手感、接口种类、散热设计和价格。"
>
>   **回复 B**："编程加日常使用的话，重点是一颗快的 CPU、至少 16 GB 内存、至少 256 GB 的 SSD。舒服的键盘和好的续航也很重要。GPU 只有在你打算在本地训练小 LLM 时才有意义。"

两个回复都提到了相关的点，但人类数据标注员很可能**偏好回复 B**，因为它更具体、更贴合提示词里说明的使用场景。

在 RLHF 中，这样的成对偏好会在大量提示词上收集起来，用来训练模型偏向那些人类一致评价更高的回复。

如下图所示，RLHF 有两个主要步骤：**(1) 训练一个奖励模型**（它本身也是一个 LLM），**(2) 用这个奖励模型给目标 LLM 的输出打分并微调它**。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1792754013695_image.png)

### 1.2 RLVR

RLHF 可能相当繁琐，因为它要求额外训练一个奖励模型，而这个模型往往也是一个大 LLM，（同样）训练起来很贵。

**可验证奖励强化学习**（RLVR）用一个简化设置替代了它：把 LLM 奖励模型换成**可验证奖励**（verifiable rewards）。可验证奖励是**确定性地、无需人类标注**计算出来的。例如，第 3 章引入的数学判分器就是一个针对数学题的可验证奖励生成器。

具体来说，给定一道数学题（和一份正确解答），数学判分器会自动检查生成解答的最终答案是否匹配真值，并给出对应奖励（正确为 1，错误为 0）。

于是，RLHF 的两步流水线在 RLVR 里**塌缩成单个训练循环**，它很像 RLHF 的第 2 步：模型生成回复，判分器给出奖励，奖励直接用于更新模型。

这个简化的 RLVR 训练流程如下图所示。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1792754275158_image.png)

RLVR 的流行很大程度上可以追溯到 2025 年 DeepSeek-R1 的成功（https://arxiv.org/abs/2501.12948）。它证明了：**不依赖人类偏好数据、也不依赖学出来的奖励模型，同样能拿到很强的推理性能**。

DeepSeek-R1 用**可自动验证的奖励**来训练推理行为，包括数学题的正确性检查（类似我们第 3 章的判分器），以及代码类任务的**编译与执行**。

>   代码执行它要求在**安全的执行环境**里训练 LLM。实践中这意味着把模型生成的代码放进隔离沙箱里编译和运行，使它不能不安全地访问宿主系统、私有文件或外部服务——这让整个设置比解数学题复杂得多。

不管怎样，验证数学任务（检查答案是否正确，一个二元标签）和验证代码编译（检查代码能否编译，也是二元标签）背后的想法是相似的，**RLVR 训练算法完全一致**。

总结一下，相比 RLHF，RLVR 有几个实际优势：

-   **不需要训练和维护一个独立的奖励模型**——那个模型往往在规模和成本上都和 base LLM 相当。
-   **需要可靠的验证信号**，这把 RLVR 限制在某些领域，例如数学和代码。
-   可验证奖励是**确定且可复现**的，同样的输出每次得到同样的结果，避免了人类偏好标注带来的噪声和不一致。
-   RLVR **天然可扩展**到大规模训练集：只要有参考解答，就能对任意数量的模型生成答案自动计算奖励，不需要额外的人工标注。





## 二、用 GRPO 实现 RLVR

我们要**实现 RLVR 来训练一个推理模型**。

### 2.1 加载预训练模型

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1792754892460_image.png)

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

```text
Using NVIDIA CUDA GPU
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
```

然后试一下温度缩放和top-p采样：

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

torch.manual_seed(0)
response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9
)
print(response)
```

```text
 47 47
```

### 2.2 加载 MATH 训练子集

接下来加载训练模型时要用到的数据集。

我们用的是从原始 MATH 数据集派生出来的一个**与测试集不重叠**的训练子集，也就是说 **MATH-500 的全部 500 道测试题都被显式排除在训练数据之外**。这可以防止数据泄漏，保证训练集和测试集干净分离。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1792755200810_image.png)

```python
import json
import requests
from pathlib import Path

def load_math_train(local_path="math_train.json", save_copy=True):
    local_path = Path(local_path)

    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "math_full_minus_math500/refs/heads/main/"
        "math_full_minus_math500.json"
    )
    backup_url = ( #A
        "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
        "MATH/math_full_minus_math500.json"
    )

    if local_path.exists():    #B
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            print("Using backup URL.")
            r = requests.get(backup_url, timeout=30)
            r.raise_for_status()

        data = r.json()

        if save_copy: #C
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data

#A Alternative URL in case the GitHub-hosted file is unavailable
#B Load from local file if already downloaded
#C Optionally cache a local copy for future runs
```

```python
math_train = load_math_train()
print("Dataset size:", len(math_train))
```

```text
Dataset size: 12000
```

然后可以拉一道看看：

```python
from pprint import pprint
pprint(math_train[4])
```

```text
{'answer': '6',
 'level': 'Level 3',
 'problem': 'Sam is hired for a 20-day period. On days that he works, he earns '
            '$\\$$60. For each day that he does not work, $\\$$30 is '
            'subtracted from his earnings. At the end of the 20-day period, he '
            'received $\\$$660. How many days did he not work?',
 'solution': 'Call $x$ the number of days Sam works and $y$ the number of days '
             'he does not. We can set up the following system of equations to '
             'represent the given information: \\begin{align*}\n'
             'x+y &= 20 \\\\\n'
             '60x - 30y &= 660 \\\\\n'
             '\\end{align*} The first equation represents the total number of '
             'days Sam works, and the second equation represents his total '
             'profit. Solving for $x$ in the first equation yields $x = 20 - '
             'y$. Substituting into the second equation gives $60(20-y) - 30y '
             '= 660$. Canceling a factor of $10$ and multiplying out gives '
             '$120 - 6y - 3y = 66$. This simplifies to $-9y = -54$, or $y = '
             '6$. Thus, Sam did not work for $\\boxed{6}$ days.',
 'type': 'Algebra',
 'unique_id': 4}
```

### 2.3 采样 rollouts

标准的 GRPO 的 loss 是含 KL散度项的，本章先暂时略去。那么流程可以简化如下图：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1792755471901_image.png)

第一步是用 LLM 生成多个 rollout。这里 "rollout" 指**模型针对某个提示词生成的一份完整答案**。

```python
from reasoning_from_scratch.qwen3 import KVCache
from reasoning_from_scratch.ch04 import top_p_filter


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
        )

    cache = KVCache(n_layers=model.cfg["n_layers"]) #A
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0: #B
            logits = logits / temperature

        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p) #C
        next_token = torch.multinomial(
            probas.cpu(), num_samples=1
        ).to(device)

        token_id = next_token.item()
        generated.append(token_id)

        if (
               tokenizer.eos_token_id is not None
               and token_id == tokenizer.eos_token_id
        ):
               break

        logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)

#A Cache past keys and values for efficient generation as introduced in chapter 2
#B Apply temperature scaling from chapter 4
#C Apply top_p filter from chapter 4
```

然后我们试一下：

```python
torch.manual_seed(0)

raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)

token_ids, prompt_len, answer_text = sample_response(
    model=model, 
    tokenizer=tokenizer,
    prompt=prompt,
    device=device,
    max_new_tokens=512,
    temperature=0.9,
    top_p=0.9
)

print(answer_text)
```

```text
 47<|endoftext|>
```

### 2.4 计算reward

在 CH03 的reward 的基础上加一个格式约束：

```python
from reasoning_from_scratch.ch03 import (
    extract_final_candidate, grade_answer
)

def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None #A
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)

#A fallback=None requires \boxed{} format to return an extracted answer
```

```python
rollout_rewards = []

for answer in rollouts:
    reward = reward_rlvr(answer_text=answer, ground_truth="83")
    print(f"Answer: {answer!r}")
    print(f"Reward: {reward}\n")
    rollout_rewards.append(reward)
```

```text
Answer: '\\boxed{83}'
Reward: 1.0

Answer: 'The correct answer is \\boxed{83}'
Reward: 1.0

Answer: 'The final answer is 83'
Reward: 0.0

Answer: 'We get \\boxed{38}'
Reward: 0.0

```

### 2.5 advantage

在强化学习的一些算法中，比如 PPO、GRPO 经常用 Advantage 来做奖励信号。

而 GRPO 相比于 PPO 的优势便是不需要 Critic 网络，它选择每次采样一组 rollout，然后每个 rollout 和组内平均值相比来做 advantage 构造。
$$
\text{advantage}_i = (r_i - \mu_r) / (\sigma_r + \epsilon)
$$
其中

- $r_i$ 表示第 $i$ 个 rollout 的奖励，
- $\mu_r$ 是这一组 rollout 的**平均**奖励，
- $\sigma_r$ 是对应的**标准差**，
- $\epsilon$ 是一个小常数，用于**数值稳定**，避免除零错误。

我们可以这样代码实现：

```python
rewards = torch.tensor(rollout_rewards, device=device)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
print(rewards)
print(advantages)
```

```text
tensor([1., 1., 0., 0.], device='cuda:0')
tensor([ 0.8659,  0.8659, -0.8659, -0.8659], device='cuda:0')
```

### 2.6 Scoring rollouts with sequence log-probabilities

```python
@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):

    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    return torch.mean(next_token_logps).item()

avg_logprob_val = avg_logprob_answer(
    model, tokenizer,
    prompt=prompt,
    answer=answer_text,
    device=device
)
print(avg_logprob_val)
```

```text
-2.203125
```

在 GRPO 中我们**不用**取平均的 token 级 logprobs。我们用的是**序列级 logprobs**（sequence-level logprobs）。

原因是：GRPO 给每个 rollout 分配**一个**奖励和**一个**优势，它作用于**整个生成的响应**。直觉上，logprob 应该反映"模型生成**整个序列**有多可能"，而不是"平均每个 token 有多可能"。

我们通过对所有生成 token 的 logprobs **求和**来计算序列级 logprobs。（这个和对应于"在当前模型参数下产出整个响应的对数似然"。）

相比之下，对 token 级 logprobs 取平均会**按序列长度做归一化**。这种归一化在"打分并比较不同长度的回复"时很有用，但在策略优化里它会**意外地重新缩放学习信号**，**导致长回复和短回复对更新的贡献不均衡**。

然后我们可以写一个序列级的实现：

```python
def sequence_logprob_draft(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    start = prompt_len - 1                        #A
    end = token_ids.shape[0] - 1                  #A

    t_idx = torch.arange(start, end, device=token_ids.device)
    next_tokens = token_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    return torch.sum(next_token_logps) #B

print(sequence_logprob_draft(model, token_ids, prompt_len))

#A Positions whose next-token probabilities we want to predict
#B Sum log-probabilities over the answer tokens
```

事实上，可以写一个 GPU 更高效的代码：

```python
def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(
        1, token_ids[1:].unsqueeze(-1)
    ).squeeze(-1)
    return torch.sum(selected[prompt_len - 1:])

print(sequence_logprob(model, token_ids, prompt_len))
```

>   Q：**`gather` 版本到底简化了什么？**
>
>   A：
>
>   对照一下就很清楚：draft 版要算 `start`/`end`、造 `torch.arange`、切 `next_tokens`，三行索引逻辑；`gather` 版一行搞定。它的等价性来自一个错位对齐的技巧——`logprobs[:-1]` 是"位置 0…L-2 的分布"，`token_ids[1:]` 是"位置 1…L-1 的实际 token"，两者天然一一对应；`selected[prompt_len - 1:]` 就是从"答案的第一个 token"开始取。

测一下：

```python
rollouts = [
    r"\boxed{83}",
    r"The correct answer is \boxed{83}",
    r"The final answer is 83",
    r"We get \boxed{38}",
]

rollout_logps = []

for text in rollouts:
    token_ids = tokenizer.encode(prompt + " " + text)
    logprob = sequence_logprob(
        model=model,
        token_ids=torch.tensor(token_ids, device=device),
        prompt_len=prompt_len,
    )

    print(f"Answer: {text}")
    print(f"Logprob: {logprob.item():.4f}\n")

    rollout_logps.append(logprob)
```

```text
Answer: \boxed{83}
Logprob: -8.0930

Answer: The correct answer is \boxed{83}
Logprob: -19.9793

Answer: The final answer is 83
Logprob: -16.7541

Answer: We get \boxed{38}
Logprob: -23.4721

```

### 2.7 GRPO loss

忽略 KL 散度项的话，GRPO 里用的策略梯度损失可以写成

>
> $$
> L_{PG} = -\frac{1}{N} \sum_{i=1}^{N} A_i \sum_{t=1}^{T_i} \log p_W\left(y_t^{(i)} \mid y_{<t}^{(i)}, x^{(i)}\right)
> $$

```python
logps = torch.stack(rollout_logps)
pg_loss = -(advantages.detach() * logps).mean()
print(logps)
print(pg_loss)
```

```text
tensor([ -8.0930, -19.9793, -16.7541, -23.4721], device='cuda:0',
       grad_fn=<StackBackward0>)
tensor(-2.6310, device='cuda:0', grad_fn=<NegBackward0>)
```

```python
def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        # Stage 1: generate rollouts
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        # Stage 2: compute rewards
        reward = reward_rlvr(text, example["answer"])

        # Stage 4: compute logprobs
        logp = sequence_logprob(model, token_ids, prompt_len)

        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    # Stage 2: collect all rewards
    rewards = torch.tensor(roll_rewards, device=device)

    # Stage 3: compute advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    # Stage 4: collect all logprobs
    logps = torch.stack(roll_logps)

    # Stage 5: compute policy gradient loss
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss  # In the next chapter we add a KL term here

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }
```

测一下：

```python
torch.manual_seed(123)

stats = compute_grpo_loss(
    model=model,
    tokenizer=tokenizer,
    example=math_train[4],
    device=device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9
)

pprint(stats)
```

```text
{'advantages': [0.0, 0.0],
 'loss': -0.0,
 'loss_tensor': tensor(-0., device='cuda:0', grad_fn=<NegBackward0>),
 'pg_loss': -0.0,
 'rewards': [0.0, 0.0],
 'samples': [{'gen_len': 4, 'reward': 0.0, 'text': ' 14<|endoftext|>'},
             {'gen_len': 256,
              'reward': 0.0,
              'text': ' 4\n'
                      "To solve the problem, let's break it down step by "
                      'step:\n'
                      '\n'
                      '1. **Define Variables:**\n'
                      '   - Let \\( x \\) be the number of days Sam works.\n'
                      '   - Then, the number of days he does not work is \\( '
                      '20 - x \\).\n'
                      '\n'
                      '2. **Set Up the Earnings Equation:**\n'
                      '   - For each day he works, he earns \\$60.\n'
                      '   - For each day he does not work, he loses \\$30.\n'
                      '   - His total earnings are \\$660.\n'
                      '\n'
                      '   The equation is:\n'
                      '   \\[\n'
                      '   60x - 30(20 - x) = 660\n'
                      '   \\]\n'
                      '\n'
                      '3. **Simplify the Equation:**\n'
                      '   \\[\n'
                      '   60x - 600 + 30x = 660\n'
                      '   \\]\n'
                      '   \\[\n'
                      '   90x - 600 = 660\n'
                      '   \\]\n'
                      '   \\[\n'
                      '   90x = 1260\n'
                      '   \\]\n'
                      '   \\[\n'
                      '   x = 14\n'
                      '   \\]\n'
                      '\n'
                      '4. **Calculate the Number of Days Not Worked:**\n'
                      '   \\[\n'
                      '   20 - x ='}]}
```



### 2.8 实现 GRPO Loop

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1792757913135_image.png)

按照上面的流程图我们可以写一个loop出来：

```python
import time

def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    device,
    steps=None,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=".",
    csv_log_path=None,

):
    if steps is None:
        steps = len(math_data)

    # Stage 1: initialize optimize
    # (the model was already initialized outside the function)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0

    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)

    try:
        # Stage 2: Iterate over training steps
        for step in range(steps):

            # Stage 3: Reset loss gradient
            # (it's best practice to do this at the beginning of each step)
            optimizer.zero_grad()

            current_step = step + 1
            example = math_data[step % len(math_data)]

            # Stage 4: calculate GRPO loss
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Stage 5: Backward pass to calculate loss gradients
            stats["loss_tensor"].backward()

            # Clip large gradients to improve training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Stage 6: Update model weights using loss gradients
            optimizer.step()

            # Stage 7: Collect rewards, losses, and response lengths
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            append_csv_metrics(
                csv_log_path,
                current_step,
                steps,
                stats["loss"],
                reward_avg,
                avg_response_len,
            )

            # Print step metrics
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )

            # Sample outputs (every 10 steps) to check if model
            # generates coherent text
            if current_step % 10 == 0:
                print(f"[Step {current_step}] sample outputs")
                for i, sample in enumerate(stats["samples"][:3]):
                    text = sample["text"].replace("\n", "\\n")
                    print(
                        f"  {i+1}) reward={sample['reward']:.3f} "
                        f"len={sample['gen_len']}: {text}"
                    )
                print()

            # Stage 8: Save model checkpoint
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")

    # Save a model checkpoint if we interrupt the training early
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model

    return model


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def append_csv_metrics(
    csv_log_path,
    step_idx,
    total_steps,
    loss,
    reward_avg,
    avg_response_len,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )
```

然后跑一下：

```python
device = get_device()
model.to(device)

torch.manual_seed(0)

train_rlvr_grpo(
    model=model,
    tokenizer=tokenizer,
    math_data=math_train,
    device=device,
    steps=50,
    num_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=5,
    checkpoint_dir=".",
)
```

```text
Using NVIDIA CUDA GPU
[Step 1/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.2
[Step 2/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.8
[Step 3/50] loss=9.1144 reward_avg=0.500 avg_resp_len=56.0
[Step 4/50] loss=2.9917 reward_avg=0.250 avg_resp_len=56.5
[Step 5/50] loss=2.9961 reward_avg=0.250 avg_resp_len=185.8
Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00005.pth
[Step 6/50] loss=7.7232 reward_avg=0.500 avg_resp_len=367.2
[Step 7/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=512.0
[Step 8/50] loss=4.3747 reward_avg=0.750 avg_resp_len=190.5
[Step 9/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=510.0
[Step 10/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=453.0
[Step 10] sample outputs
  1) reward=0.000 len=425:  To determine how much Tim should invest to reach \$60,000 in 5 years with a 7% annual interest rate compounded quarterly, we can use the formula for compound interest:\n\n\[\nA = P \left(1 + \frac{r}{n}\right)^{nt}\n\]\n\nWhere:\n- \( A \) is the amount of money accumulated after \( t \) years, including interest.\n- \( P \) is the principal amount (the initial amount of money).\n- \( r \) is the annual interest rate (decimal).\n- \( n \) is the number of times that interest is compounded per year.\n- \( t \) is the time the money is invested for in years.\n\nGiven:\n- \( A = \$60,000 \)\n- \( r = 7\% = 0.07 \)\n- \( n = 4 \) (since interest is compounded quarterly)\n- \( t = 5 \) years\n\nWe need to solve for \( P \):\n\n\[\n60,000 = P \left(1 + \frac{0.07}{4}\right)^{4 \times 5}\n\]\n\nSimplify the equation:\n\n\[\n60,000 = P \left(1 + 0.0175\right)^{20}\n\]\n\n\[\n60,000 = P \left(1.0175\right)^{20}\n\]\n\nCalculate \( (1.0175)^{20} \):\n\n\[\n(1.0175)^{20} \approx 1.4198\n\]\n\nNow, solve for \( P \):\n\n\[\nP = \frac{60,000}{1.4198} \approx 42,178.48\n\]\n\nTo the nearest dollar, Tim should invest:\n\n\[\n\boxed{42,\!178}\n\]<|endoftext|>
  2) reward=0.000 len=432:  To determine the amount Tim should invest to achieve a total of \$60,000 at the end of 5 years with quarterly compounding and an annual interest rate of 7%, we can use the compound interest formula:\n\n\[\nA = P \left(1 + \frac{r}{n}\right)^{nt}\n\]\n\nWhere:\n- \( A \) is the amount of money accumulated after \( t \) years, including interest.\n- \( P \) is the principal amount (the initial amount of money).\n- \( r \) is the annual interest rate (decimal).\n- \( n \) is the number of times that interest is compounded per year.\n- \( t \) is the time the money is invested for in years.\n\nGiven:\n- \( A = \$60,000 \)\n- \( r = 7\% = 0.07 \)\n- \( n = 4 \) (since the interest is compounded quarterly)\n- \( t = 5 \) years\n\nSubstitute the given values into the formula:\n\n\[\n60,000 = P \left(1 + \frac{0.07}{4}\right)^{4 \times 5}\n\]\n\nSimplify the equation:\n\n\[\n60,000 = P \left(1 + 0.0175\right)^{20}\n\]\n\n\[\n60,000 = P \left(1.0175\right)^{20}\n\]\n\nCalculate \( (1.0175)^{20} \):\n\n\[\n(1.0175)^{20} \approx 1.4196\n\]\n\nNow, solve for \( P \):\n\n\[\nP = \frac{60,000}{1.4196} \approx 42,256.52\n\]\n\nTo the nearest dollar, Tim should invest:\n\n\[\n\boxed{42,257}\n\]<|endoftext|>
  3) reward=0.000 len=473:  To determine how much Tim should invest to reach \$60,000 in 5 years with an annual interest rate of 7% compounded quarterly, we can use the **compound interest formula**:\n\n\[\nA = P \left(1 + \frac{r}{n}\right)^{nt}\n\]\n\nWhere:\n- \( A \) = the amount of money accumulated after \( t \) years, including interest.\n- \( P \) = the principal amount (the initial amount of money).\n- \( r \) = annual interest rate (in decimal).\n- \( n \) = number of times that interest is compounded per year.\n- \( t \) = time the money is invested for in years.\n\n**Given:**\n- \( A = \$60,000 \)\n- \( r = 7\% = 0.07 \)\n- \( n = 4 \) (since the interest is compounded quarterly)\n- \( t = 5 \) years\n\n**Step 1: Plug the values into the formula.**\n\n\[\n60,000 = P \left(1 + \frac{0.07}{4}\right)^{4 \times 5}\n\]\n\n**Step 2: Simplify the equation.**\n\n\[\n60,000 = P \left(1 + 0.0175\right)^{20}\n\]\n\n\[\n60,000 = P \left(1.0175\right)^{20}\n\]\n\n**Step 3: Calculate \( (1.0175)^{20} \).**\n\nUsing a calculator:\n\n\[\n(1.0175)^{20} \approx 1.4007\n\]\n\n**Step 4: Solve for \( P \).**\n\n\[\nP = \frac{60,000}{1.4007} \approx 42,847.27\n\]\n\n**Step 5: Round to the nearest dollar.**\n\n\[\nP \approx \$42,847\n\]\n\n**Final Answer:**\n\n\[\n\boxed{42847}\n\]<|endoftext|>

Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00010.pth
[Step 11/50] loss=-0.0000 reward_avg=1.000 avg_resp_len=307.5
[Step 12/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=376.0
[Step 13/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=190.8
[Step 14/50] loss=-0.0000 reward_avg=1.000 avg_resp_len=283.0
[Step 15/50] loss=-0.0000 reward_avg=1.000 avg_resp_len=188.2
Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00015.pth
[Step 16/50] loss=-4.0767 reward_avg=0.750 avg_resp_len=290.0
...
  2) reward=1.000 len=351:  To factor the expression \(58x^5 - 203x^{11}\), follow these steps:\n\n1. **Identify the Greatest Common Factor (GCF):**\n   - The coefficients are 58 and 203. The GCF of 58 and 203 is 29.\n   - The variable part is \(x^5\) and \(x^{11}\). The GCF of \(x^5\) and \(x^{11}\) is \(x^5\).\n   \n   Therefore, the GCF of the entire expression is \(29x^5\).\n\n2. **Factor out the GCF:**\n   \[\n   58x^5 - 203x^{11} = 29x^5 \left( \frac{58x^5}{29x^5} - \frac{203x^{11}}{29x^5} \right)\n   \]\n   \n   Simplify the terms inside the parentheses:\n   \[\n   29x^5 (2 - 7x^6)\n   \]\n\n3. **Final Factored Form:**\n   \[\n   \boxed{29x^5(2 - 7x^6)}\n   \]\n\n**Verification:**\nTo ensure the factorization is correct, multiply the factored form back with the original expression:\n\[\n29x^5(2 - 7x^6) = 58x^5 - 203x^{11}\n\]\nThis matches the original expression, confirming that the factorization is correct.<|endoftext|>
  3) reward=1.000 len=353:  To factor the expression \(58x^5 - 203x^{11}\), follow these steps:\n\n### Step 1: Identify the greatest common factor (GCF)\nFirst, find the GCF of the coefficients and the variable parts of the terms.\n\n- **Coefficients:** The coefficients are 58 and 203. The GCF of 58 and 203 is 29.\n- **Variable part:** The terms are \(x^5\) and \(x^{11}\). The GCF of \(x^5\) and \(x^{11}\) is \(x^5\).\n\nSo, the GCF of the entire expression is \(29x^5\).\n\n### Step 2: Factor out the GCF\nFactor out \(29x^5\) from each term:\n\n\[\n58x^5 - 203x^{11} = 29x^5 (2) - 29x^5 (7x^6)\n\]\n\n\[\n= 29x^5 (2 - 7x^6)\n\]\n\n### Step 3: Simplify the expression\nThe expression inside the parentheses can be simplified further:\n\n\[\n2 - 7x^6 = 2(1 - \frac{7}{2}x^6)\n\]\n\nHowever, this doesn't simplify the expression further. Therefore, the fully factored form of the original expression is:\n\n\[\n\boxed{29x^5 (2 - 7x^6)}\n\]\n\n### Final Answer:\n\boxed{29x^5 (2 - 7x^6)}<|endoftext|>

Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00050.pth
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

然后久违的测一下 MATH-500：

```bash
~$ uv run evaluate_math500_batched.py --dataset_size 500 --which_model base --checkpoint_path "qwen3-0.6B-rlvr-grpo-step00050.pth"
Using NVIDIA CUDA GPU
Model: base
Device: cuda
Batch size: 4
MATH-500: 328/500 | ETA: 30m 11s
MATH-500: 500/500 | ETA: 00s
Accuracy: 45.4% (227/500)
Total time: 85.2 min
Logs written to: math500_base-cuda-batched-bs4.jsonl
```

之前 base model 只获得了三十多的 acc，说明我们的训练是有效的。





## 三、REFERENCE

[Build a Reasoning Model (From Scratch)，Sebastian Raschka](https://www.manning.com/books/build-a-reasoning-model-from-scratch?utm_source=raschka&utm_medium=affiliate&utm_campaign=book_raschka2&a_aid=raschka&a_bid=4c3c5398&chan=mm_github)

[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch/tree/main)

