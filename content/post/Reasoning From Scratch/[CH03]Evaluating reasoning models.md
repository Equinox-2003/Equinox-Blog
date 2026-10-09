---
title: "[CH03]Evaluating Reasoning Models"
description: "评估推理模型"
date: 2026-10-08T16:50:20+08:00
lastmod: 2026-10-08T16:50:20+08:00
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

依旧不知道干啥，随便跑点代码。

基本就是 ai 写一些数据处理的API，然后过一下流程。





## 一、Building a math verifier

LLM 评估，常见的就是：

-   multiple choice
-   verifier
-   leaderboards
-   LLM as Judge

从更大的视角看，这四种方法可以分成两拨：

| 大类                                  | 包含的方法                    | 特点               |
| ------------------------------------- | ----------------------------- | ------------------ |
| **基于基准的评估**（benchmark-based） | multiple choice、**verifier** | 偏定量，可自动复现 |
| **基于判断的评估**（judgment-based）  | leaderboards、LLM as Judge    | 更依赖定性判断     |

对**推理模型**来说，比较适合verifier。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791534144182_image.png)

比如一些 math problem 的 bench，虽然解题可能需要长程推理，但是最后的测评只需要一个具体的答案。

流程：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791534205220_image.png)

---





## 二、Loading a pre-trained model to generate text

### 2.1 加载预训练模型

```python
from pathlib import Path
import torch

from reasoning_from_scratch.ch02 import (
    get_device
)
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B
)

def load_model_and_tokenizer(
    which_model, device, use_compile, local_dir="qwen3"
):
    if which_model == "base":
        download_qwen3_small(
        kind="base", tokenizer_only=False, out_dir=local_dir
    )
        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":
        download_qwen3_small(
            kind="reasoning", tokenizer_only=False, out_dir=local_dir
        )
        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )
    
    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")

    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    
    if use_compile: #A
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)
    
    return model, tokenizer

WHICH_MODEL = "base" #B
device = get_device()
# device = torch.device("cpu") #C

model, tokenizer = load_model_and_tokenizer(
    which_model=WHICH_MODEL,
    device=device,
    use_compile=False
)

```

```texT
Using NVIDIA CUDA GPU
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
```

### 2.2 生成模型输出

```python
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache
)

prompt = ( #A
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)

input_token_ids_tensor = torch.tensor( #B
    tokenizer.encode(prompt)
).unsqueeze(0).to(device)

all_token_ids = []

for token in generate_text_basic_stream_cache( #D
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=2048,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0) #E
    decoded_id = tokenizer.decode(token_id.tolist())
    print( #F
        decoded_id,
        end="",
        flush=True
    )
    all_token_ids.append(token_id)

all_tokens = tokenizer.decode(all_token_ids) #G
```

```text
 To find the value of \( a^2 + b^2 \) given that \( a + b = 3 \) and \( ab = \frac{13}{6} \), we can use the following algebraic identity:

\[
a^2 + b^2 = (a + b)^2 - 2ab
\]

**Step 1:** Substitute the given values into the equation.

\[
a^2 + b^2 = (3)^2 - 2 \left( \frac{13}{6} \right)
\]

**Step 2:** Calculate \( (3)^2 \).

\[
(3)^2 = 9
\]

**Step 3:** Calculate \( 2 \times \frac{13}{6} \).

\[
2 \times \frac{13}{6} = \frac{26}{6} = \frac{13}{3}
\]

**Step 4:** Subtract the second result from the first.
...
**Final Answer:**

\[
\boxed{\dfrac{14}{3}}
\]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

渲染一下：

```python
import re
from IPython.display import Markdown, display


def format_latex_for_vscode(text: str) -> str:
    # 1. 把整行公式 \[ ... \] 转换为 $$ ... $$
    text = re.sub(r"\\\[", "$$", text)
    text = re.sub(r"\\\]", "$$", text)
    # 2. 把行内公式 \( ... \) 转换为 $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)
    return text


# 转换后再渲染
display(Markdown(format_latex_for_vscode(all_tokens)))
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791534395670_image.png)

### 2.3  封装 text generation

```python
def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
):
    input_ids = torch.tensor( #A
        tokenizer.encode(prompt), device=device
        ).unsqueeze(0)
    
    generated_ids = []
    for token in generate_text_basic_stream_cache( #B
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())

        if verbose: #C
            print(
            tokenizer.decode(next_token_id.tolist()),
            end="",
            flush=True
        )
    return tokenizer.decode(generated_ids) #D

```

```python
generated_text = generate_text_stream_concat(
    model, tokenizer, prompt, device, 
    max_new_tokens=2048, 
    verbose=True    #A
)
```

会跟前面输出一样。

### 2.4 答案提取

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791534537080_image.png)

需要写个 api 提取答案，方便后续测评。

```python
def get_last_boxed(text):
    boxed_start_idx = text.rfind(r"\boxed") #A
    if boxed_start_idx == -1:
        return None

    current_idx = boxed_start_idx + len(r"\boxed")  # B

    #C
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    #D
    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    #E
    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:    # F
        return None 

    return text[content_start_idx:current_idx - 1]
```

```python
extracted_answer = get_last_boxed(model_answer)
print(extracted_answer)

```

```text
\dfrac{14}{3}
```

也可以选择渲染一下：

```python
from IPython.display import Math
display(Math(r"\dfrac{14}{3}"))

```

不过我们可以借助正则表达式来做答案提取：

```python
import re

RE_NUMBER = re.compile( #A
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

def extract_final_candidate(text, fallback="number_then_full"):

    result = ""        #B

    if text: #C
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")

        #D
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                result = m[-1] #E
            elif fallback == "number_then_full":

                result = text               #F
    return result

#A 用于从文本中抽取数值的正则表达式
#B 什么都没匹配到时的默认返回值
#C 如果存在 boxed 表达式，优先用它
#D 没有 boxed 表达式时，尝试回退策略
#E 使用最后一个数字
#F 连数字都没有时，返回整段文本
```

>   Q：为什么不让 LLM 来做提取
>
>   A：
>
>   因为任务机械、重复且简单（

### 2.5 标准化答案

因为抽取出来的格式太多了：`"\frac{14}{3}"`、`"14/3"`、`"$14/3$"`、`"(14)/(3)"`。要实现一套稳健的检查系统，首先得有一种**一致的比较方法**。

我们可以写一些函数来处理成指定格式：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791534969950_image.png)

```python
LATEX_FIXES = [ #A
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>") #B
SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",                 #C
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",                   #C
    "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",                             #C
}

def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()

    #D
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)

    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text) #D
    text = re.sub(r"\^\s*\\circ", "", text)           #E
    text = text.replace("°", "")                      #E

    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text) #F
    if match:
        text = match.group("x")

    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)               #G

    for pat, rep in LATEX_FIXES:           #H
        text = re.sub(pat, rep, text)

    def convert_superscripts(s, base=None):
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )
        if base is None:
            return converted
        return f"{base}**{converted}"

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)

    #I
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    #J
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    #K
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    #L
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )

    return text.replace("{", "").replace("}", "").strip().lower()

#A 需要替换的 LaTeX 写法（左：原始写法，右：替换成什么）
#B 剥掉 <|assistant|> 这类聊天特殊 token
#C 把 Unicode 上标转换成普通文本幂次的字典
#D 去掉选择题的前导标号（例如 "c. 3" -> 3）
#E 去掉角度度数标记
#F 如果整串被 "\text{...}" 包着，就把它解开
#G 去掉行内/行间数学包裹符号：\( \) \[ \]
#H LaTeX 规范化
#I 规范数字与根号表达式
#J 把 LaTeX 分数改写成除法形式
#K 处理幂次和带分数
#L 去掉数字里的千位分隔符
```

```python
print(normalize_text(extract_final_candidate(model_answer)))
```

```text
(14)/(3)
```

```python
print(normalize_text(r"\text{\[\frac{14}{3}\]}"))
```

```text
(14)/(3)
```

### 2.6 答案验证

显然需要把抽取的答案与 ground truth 做比对，我们用一个**符号数学引擎**来解析抽取并归一化后的答案。这里选用 SymPy 这个开源数学库（[https://sympy.org](https://sympy.org/) ）。

>   SymPy 已经有二十年历史了，是 Python 科学计算的标准配置之一。

```python
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError

def sympy_parser(expr):
    if expr is None or len(expr) > 2000: #A
        return None

    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations, #B
                #C
                spp.implicit_multiplication_application,
            ),

            evaluate=True, #D
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None

#A 避免在超长的垃圾回复上崩溃
#B 标准变换，例如处理括号
#C 允许省略乘号（例如 2y -> 2*y）
#D 解析时即求值，简单常数会被化简（例如 2+3 -> 5）
```

>   `sympy_parser` 里考虑的错误类型看起来多得有点夸张，但这些都是在全部 500 道 MATH-500 题目上评估模型时**真实遇到过的**错误，因为模型并不总能生成格式完美的输出。

看一下效果：

```python
print(sympy_parser(normalize_text(
    extract_final_candidate(model_answer)
)))
```

```text
14/3
```

```python
print(sympy_parser("28/6"))
```

```
14/3
```

然后写一个等价检查函数：

```python
from sympy import simplify

def equality_check(expr_gtruth, expr_pred):
    if expr_gtruth == expr_pred: #A
        return True

    #B
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)

    if gtruth is not None and pred is not None: #C
        try:
            return simplify(gtruth - pred) == 0 #D
        except (SympifyError, TypeError):
            pass

    return False

#A 先检查两个表达式是不是完全相同的字符串
#B 把两个表达式都解析成 SymPy 对象（解析失败则返回 None）
#C 若两边都解析成功，尝试符号比较
#D 差值为 0 则等价
```

测一下：

```python
print(equality_check(
    normalize_text("13/4."),
    normalize_text(r"(13)/(4)")
))
```

```text
True
```

```python
print(equality_check(
    normalize_text("0.5"),
    normalize_text(r"(1)/(2)")
))
```

```text
True
```

```python
print(equality_check(
    normalize_text("14/3"),
    normalize_text("15/3")
))
```

```text
False
```

但是如果是多个结果，比如元组这种，就不太行：

```python
print(equality_check(
    normalize_text("(14/3, 2/3)"),
    normalize_text("(14/3, 4/6)")
))
```

```text
False
```

### 2.7 Grading answers

写一个拆分元组的函数：

```python
def split_into_parts(text):
    result = [text]

    if text:         #A
        if (
                len(text) >= 2
                and text[0] in "([" and text[-1] in ")]"
                and "," in text[1:-1]
        ):
            items = [p.strip() for p in text[1:-1].split(",")] #B
            if all(items):
                result = items
    else: #C
        result = []

    return result

#A 检查 text 是否形如元组或列表，例如 "(a, b)" 或 "[a, b]"
#B 按括号内的逗号切分，并去掉空白
#C 如果 text 为空，返回空列表
```

```python
split_into_parts(normalize_text(r"(14/3, 2/3)"))
```

```text
['14/3', '2/3']
```

然后写一个比对预测答案和标准答案的函数：

```python
def grade_answer(pred_text, gt_text):
    result = False   #A
    if pred_text is not None and gt_text is not None: #B
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )
        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )

        if (gt_parts and pred_parts                #C
           and len(gt_parts) == len(pred_parts)): #C
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            ) #D

    return result         #E

#A 检查失败时的默认结果
#B 只有当两个输入都非空时才继续
#C 确保两边的有效子部分数量相同
#D 对每个子部分做数学等价性检查
#E 只有全部通过才返回 True
```

```python
grade_answer("14/3", r"\frac{14}{3}")
```

```text
True
```

```python
grade_answer(r"(14/3, 2/3)", "(14/3, 4/6)")
```

```text
True
```

跑几个样例：

```python
tests = [ #A
        ("check_1", "3/4", r"\frac{3}{4}", True),
        ("check_2", "(3)/(4)", r"3/4", True),
        ("check_3", r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
        ("check_4", r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
        ("check_5", "(1, 2)", r"(1,2)", True),
        ("check_6", "(2, 1)", "(1, 2)", False),
        ("check_7", "(1, 2, 3)", "(1, 2)", False),
        ("check_8", "0.5", "1/2", True),
        ("check_9", "0.3333333333", "1/3", False),
        ("check_10", "1,234/2", "617", True),
        ("check_11", r"\text{2/3}", "2/3", True),
        ("check_12", "50%", "1/2", False),
        ("check_13", r"2\cdot 3/4", "3/2", True),
        ("check_14", r"90^\circ", "90", True),
        ("check_15", r"\left(\frac{3}{4}\right)", "3/4", True),
        ("check_16", r"2²", "2**2", True),
    ]

def run_demos_table(tests):
    header = ("Test", "Expect", "Got", "Status")
    rows = []
    for name, pred, gtruth, expect in tests:
        got = grade_answer(pred, gtruth) #B
        status = "PASS" if got == expect else "FAIL"
        rows.append((name, str(expect), str(got), status))

    data = [header] + rows

    col_widths = [ #C
        max(len(row[i]) for row in data)
        for i in range(len(header))
    ]

    for row in data: #D
        line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(header))
        )
        print(line)

    passed = sum(r[3] == "PASS" for r in rows) #E
    print(f"\nPassed {passed}/{len(rows)}")    #E

#A 定义测试用例：(名称, 预测, 标准答案, 期望结果)
#B 运行等价性检查
#C 计算每列最大宽度以对齐表格
#D 逐行打印表格
#E 打印通过情况的汇总
```

```python
run_demos_table(tests)
```

```text
Test     | Expect | Got   | Status
check_1  | True   | True  | PASS  
check_2  | True   | True  | PASS  
check_3  | True   | True  | PASS  
check_4  | True   | True  | PASS  
check_5  | True   | True  | PASS  
check_6  | False  | False | PASS  
check_7  | False  | False | PASS  
check_8  | True   | True  | PASS  
check_9  | False  | False | PASS  
check_10 | True   | True  | PASS  
check_11 | True   | True  | PASS  
check_12 | False  | False | PASS  
check_13 | True   | True  | PASS  
check_14 | True   | True  | PASS  
check_15 | True   | True  | PASS  
check_16 | True   | True  | PASS  

Passed 16/16
```

### 2.8 加载 dataset

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791535915184_image.png)

因为对于这种数学问题，往往需要很久的推理，所以选了 MATH-500 这个dataset。

加载一下：

```python
import json
import requests

def load_math500_test(local_path="math500_test.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        if save_copy: # 保存一份本地副本
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data

math_data = load_math500_test()
print("Number of entries:", len(math_data))
```

```text
Number of entries: 500
```

可以看一个：

```python
from pprint import pprint
pprint(math_data[0])
```

```text
{'answer': '\\left( 3, \\frac{\\pi}{2} \\right)',
 'level': 2,
 'problem': 'Convert the point $(0,3)$ in rectangular coordinates to polar '
            'coordinates.  Enter your answer in the form $(r,\\theta),$ where '
            '$r > 0$ and $0 \\le \\theta < 2 \\pi.$',
 'solution': 'We have that $r = \\sqrt{0^2 + 3^2} = 3.$  Also, if we draw the '
             'line connecting the origin and $(0,3),$ this line makes an angle '
             'of $\\frac{\\pi}{2}$ with the positive $x$-axis.\n'
             '\n'
             '[asy]\n'
             'unitsize(0.8 cm);\n'
             '\n'
             'draw((-0.5,0)--(3.5,0));\n'
             'draw((0,-0.5)--(0,3.5));\n'
             'draw(arc((0,0),3,0,90),red,Arrow(6));\n'
             '\n'
             'dot((0,3), red);\n'
             'label("$(0,3)$", (0,3), W);\n'
             'dot((3,0), red);\n'
             '[/asy]\n'
             '\n'
             'Therefore, the polar coordinates are $\\boxed{\\left( 3, '
             '\\frac{\\pi}{2} \\right)}.$',
 'subject': 'Precalculus',
 'unique_id': 'test/precalculus/807.json'}
```

### 2.9 模型评估

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791536390261_image.png)

评估prompt：

```python
def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template
```

试个例题：

```python
prompt = (
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)
prompt_fmt = render_prompt(prompt)
print(prompt_fmt)
```

```text
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}

Question:
If $a+b=3$ and $ab=\tfrac{13}{6}$, what is the value of $a^2+b^2$?

Answer:
```

然后扔给LLM：

```python
generated_text = generate_text_stream_concat(
    model, tokenizer, prompt_fmt, device,
    max_new_tokens=2048,
    verbose=True
)
```

```text
 \boxed{10}
```

然后就答错了，base model 比较菜，当然也跟 prompt 有关，加一些CoT引导，说不定就对了。

写一个eval的demo：

```python
def mini_eval_demo(model, tokenizer, device):
    ex = { #A
        "problem": "Compute 1/2 + 1/6.",
        "answer": "2/3"
    }
    prompt = render_prompt(ex["problem"])     #B 1. 应用提示模板
    gen_text = generate_text_stream_concat( #C 2. 生成回复
        model, tokenizer, prompt, device,     #C
        max_new_tokens=64,                    #C
    )                                         #C
    pred_answer = extract_final_candidate(gen_text) #D 3. 抽取并归一化答案
    is_correct = grade_answer(                      #E 4. 打分
        pred_answer, ex["answer"]                   #E
    )                                               #E

    print(f"Device: {device}")
    print(f"Prediction: {pred_answer}")
    print(f"Ground truth: {ex['answer']}")
    print(f"Correct: {is_correct}")

#A 带 "problem" 和 "answer" 字段的测试样例
#B 1. 应用提示模板
#C 2. 生成回复
#D 3. 抽取并归一化答案
#E 4. 打分
```

```python
mini_eval_demo(model, tokenizer, device)
```

```text
Device: cuda
Prediction: 1/3
Ground truth: 2/3
Correct: False
```

然后写一下端到端测评：

```python
import time

def eta_progress_message( #A
    processed,
    total,
    start_time,
    show_eta=False,
    label="Progress",
):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)

    remaining = max(total - processed, 0)

    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0

    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"

    message = f"{progress} | ETA: {eta}"
    return message.ljust(pad_width)

def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):

    if out_path is None:
        dev_name = str(device).replace(":", "-")     #B
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:            #C
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])              #D 1. 应用提示模板
            gen_text = generate_text_stream_concat(             #E 2. 生成回复
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )

            extracted = extract_final_candidate(                #F 3. 抽取并归一化答案
                gen_text
            )
            is_correct = grade_answer(                          #G 4. 打分
                extracted, row["answer"]
            )
            num_correct += int(is_correct)

            record = {                                 #H
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)

            if verbose:                                #I
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected: {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )

    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc

#A 打印进度、可选附带 ETA（预计剩余时间）的辅助函数
#B 让文件名在 Windows 上也兼容
#C 保存结果以便查看
#D 1. 应用提示模板
#E 2. 生成回复
#F 3. 抽取并归一化答案
#G 4. 打分
#H 需要保存下来供查看的记录
#I 生成过程中打印回复
```

然后拿base model跑一下：

```python
print("Model:", WHICH_MODEL)
num_correct, num_examples, acc = evaluate_math500_stream(
    model, tokenizer, device,
    math_data=math_data[:10], #A
    max_new_tokens=2048,
    verbose=False              #B
)
#A 只在头 10 个样例上评估
#B 设为 True 可以在生成时实时看回复
```

```text
Model: base
MATH-500: 10/10 | ETA: 00s        
Accuracy: 30.0% (3/10)
Total time: 0.4 min
Logs written to: math500-cuda.jsonl
```

base model 还是太菜了。

写一个脚本，方便测各个model：

```python
import argparse
import json
import time
from pathlib import Path

import torch

from reasoning_from_scratch.qwen3_batched import (
    Qwen3Model,
    QWEN_CONFIG_06_B
)
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_math500_test,
    render_prompt,
    extract_final_candidate,
    grade_answer,
    eta_progress_message,
    load_tokenizer_only
)
from reasoning_from_scratch.qwen3_batched import (
    load_model_and_tokenizer,
)


def evaluate_math500_batched(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
    batch_size=4,
    show_eta=False,
):
    model.eval()

    # Default output path like the streaming variant
    if out_path is None:
        dev_name = str(device).replace(":", "-")  # Make filename compatible with Windows
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0

    start_time = time.time()

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    with open(out_path, "w", encoding="utf-8") as f:
        # Batched loop
        for start in range(0, num_examples, batch_size):
            batch = math_data[start:start + batch_size]

            prompts = [render_prompt(row["problem"]) for row in batch]

            # Encode and left-pad
            tokenized = [tokenizer.encode(p) for p in prompts]
            max_len = max(len(t) for t in tokenized)
            left_padded = [
                ([pad_id] * (max_len - len(t)) + t) if pad_id is not None else t
                for t in tokenized
            ]
            input_ids = torch.tensor(left_padded, device=device, dtype=torch.long)

            # Generate (batched)
            gen = generate_text_basic_batched_cache(
                model,
                token_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                pad_id=pad_id,
            )  # shape: (B, T_new)

            # Process each row
            B = gen.size(0)
            for b_idx in range(B):
                # Match the variable names used by evaluate_math500_stream
                i = start + b_idx + 1
                row = batch[b_idx]

                row_tokens = gen[b_idx]
                # Cut at first EOS if present
                if eos_id is not None:
                    eos_pos = (row_tokens == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        row_tokens = row_tokens[: eos_pos[0]]

                gen_text = tokenizer.decode(row_tokens.tolist())
                extracted = extract_final_candidate(gen_text)
                is_correct = grade_answer(extracted, row["answer"])
                num_correct += int(is_correct)

                record = {  # Record to be saved for inspection
                    "index": i,
                    "problem": row["problem"],
                    "gtruth_answer": row["answer"],
                    "generated_text": gen_text,
                    "extracted": extracted,
                    "correct": bool(is_correct),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                progress_msg = eta_progress_message(
                    processed=i,
                    total=num_examples,
                    start_time=start_time,
                    show_eta=show_eta,
                    label="MATH-500",
                )
                print(progress_msg, end="\r", flush=True)
                if verbose:  # Print responses during the generation process
                    print(
                        f"\n\n{'='*50}\n{progress_msg}\n"
                        f"{'='*50}\nExtracted: {extracted}\n"
                        f"Expected:  {row['answer']}\n"
                        f"Correct so far: {num_correct}\n{'-'*50}"
                    )

    # Print summary information
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', or any torch device string like 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning", "instruct"],
        help="Model variant to load",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10,
        help="Number of MATH-500 examples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a .pth checkpoint to load model weights from.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for batched generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample correctness while evaluating.",
    )
    parser.add_argument(
        "--disable_efficient_mode",
        action="store_true",
        help="Uses an alternative implementation of batched inference that is simpler but more memory and compute intense.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.disable_efficient_mode:
        from reasoning_from_scratch.qwen3_batched import (
            generate_text_basic_batched_cache,
        )
    else:
        from reasoning_from_scratch.qwen3_batched import (
            generate_text_basic_batched_cache_stop as generate_text_basic_batched_cache,
        )
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    which_model = args.which_model
    dataset_size = args.dataset_size
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size

    print("Model:", which_model)
    print("Device:", device)
    dev_name = str(device).replace(":", "-")
    print("Batch size:", batch_size)

    math_data = load_math500_test()[:dataset_size]
    if args.which_model == "instruct":
        which_model = "reasoning"
    else:
        which_model = args.which_model

    if args.checkpoint_path:
        # To load the saved RL checkpoint files from chapter 6
        tokenizer = load_tokenizer_only(which_model=which_model)
        model = Qwen3Model(QWEN_CONFIG_06_B)
        model.to(device)
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        if args.compile:
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            model = torch.compile(model)
    else:
        model, tokenizer = load_model_and_tokenizer(
            which_model=which_model,
            device=device,
            use_compile=args.compile
        )
    if args.which_model == "instruct":
        tokenizer.add_thinking = False

    out_path = f"math500_{which_model}-{dev_name}-batched-bs{batch_size}.jsonl"
    num_correct, num_examples, acc = evaluate_math500_batched(
        model=model,
        tokenizer=tokenizer,
        device=device,
        math_data=math_data,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        out_path=out_path,
        verbose=args.verbose,
        show_eta=True,
    )
```



然后拿 reasoning model 在单张 4090 24GB 上跑一下子集测试：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791537192258_image.png)

全量测试：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1791534825638_image.png)

其实效果也一般（

不过对于 0.6B 的模型而且simple prompt 来说貌似也可以了？

## 三、REFERENCE

[Build a Reasoning Model (From Scratch)，Sebastian Raschka](https://www.manning.com/books/build-a-reasoning-model-from-scratch?utm_source=raschka&utm_medium=affiliate&utm_campaign=book_raschka2&a_aid=raschka&a_bid=4c3c5398&chan=mm_github)

[[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)](https://github.com/rasbt/reasoning-from-scratch/tree/main)

