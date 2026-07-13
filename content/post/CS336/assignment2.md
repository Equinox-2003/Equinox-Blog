---
title: "Assignment2"
description: ""
date: 2026-06-12T14:48:27+08:00
lastmod: 2026-06-12T14:48:27+08:00
draft: false

categories:
  - CS336
tags:
  - LLM

toc: true
math: true
mermaid: true
---

<!--more-->



## 零、写在前面

做assignment2之前还是有必要听一下前面几个lecture或者看一下讲义的，慢慢积累。

至少需要了解这些内容：

[注意力替代方案与MoE](https://equinox.wiki/post/cs336/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9B%BF%E4%BB%A3%E6%96%B9%E6%A1%88%E4%B8%8Emoe/)

[GPUs and TPUs](https://equinox.wiki/post/cs336/gpus-and-tpus/)

[Benchmarking、Profiling 与 Triton Kernels](https://equinox.wiki/post/cs336/benchmarkingprofiling-%E4%B8%8E-triton-kernels/)

[并行训练优化、MoE扩展与系统瓶颈](https://equinox.wiki/post/cs336/%E5%B9%B6%E8%A1%8C%E8%AE%AD%E7%BB%83%E4%BC%98%E5%8C%96moe%E6%89%A9%E5%B1%95%E4%B8%8E%E7%B3%BB%E7%BB%9F%E7%93%B6%E9%A2%88/)

本章代码：[assignment2](https://github.com/Equinox-2003/CS336-Assignment/tree/main/assignment2)



## 一、 Profiling 

在作业的第一部分，我们将探索如何优化Transformer模型的性能，以最高效地利用GPU。我们将对模型进行性能分析，了解其在前向和后向传播过程中时间和内存的消耗情况，然后通过自定义GPU内核优化自注意力操作，使其比常规PyTorch实现更快。在作业的后续部分，我们将利用多个GPU，并理解如何在集群上训练模型。



### 1.1 End-to-End Benchmarking

要求写一个脚本，对你的模型中前向传播、后向传播和优化器步骤进行基本的端到端基准测试。具体来说，你的脚本应支持以下功能：

-   根据给定的超参数（例如层数）初始化一个模型。
-   生成一批随机数据。
-   运行w个热身步骤（在开始测量时间之前），然后测量执行 n 个步骤所需的时间（根据参数可以是仅前向、前向和后向、或前向、后向加优化器步骤）。你可以使用Python的timeit模块来计时。
-   在每一步之后调用 torch.cuda.synchronize()。

其实和之前benchmark那个lecture讲义上的例子差不多，只不过这里要求最好是让你的脚本通过命令行参数来启用这些变体，因为会很方便。

那么benchmark测试其实很容易，我们对随机初始化的模型，在保证做好cuda 同步的情况下，记录一下各种操作的运行时间即可。

对于脚本解析命令行参数，直接利用argparse库即可。

**库导入**

```python
from __future__ import annotations

import argparse
import csv
import statistics
import timeit
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
```

**讲义给的模型参数**

```python
MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "toy": {"d_model": 64, "d_ff": 256, "num_layers": 2, "num_heads": 4},
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

# 要求我们测试的三种操作
BenchmarkMode = Literal["forward", "forward_backward", "train_step"]
```

**一些数据容器**

```python
@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    vocab_size: int
    context_length: int
    batch_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    warmup_steps: int
    measurement_steps: int
    mode: BenchmarkMode
    device: str
    dtype: str
    compile_model: bool
    seed: int


@dataclass(frozen=True)
class StepTiming:
    forward_s: float
    backward_s: float
    optimizer_s: float
    total_s: float
    peak_memory_gib: float | None
```

**一些辅助函数**

```python
# 同步
def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

# 获取device
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device

# 获取dtype
def resolve_dtype(dtype_arg: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    return dtypes[dtype_arg]
```

**模型参数配置与命令行参数覆盖**

```python
# 合并默认配置和命令行覆盖
def build_config(args: argparse.Namespace, mode: BenchmarkMode) -> BenchmarkConfig:
    base_config = dict(MODEL_CONFIGS[args.model_size])
    for cli_name, config_name in [
        ("d_model", "d_model"),
        ("d_ff", "d_ff"),
        ("num_layers", "num_layers"),
        ("num_heads", "num_heads"),
    ]:
        value = getattr(args, cli_name)
        if value is not None:
            base_config[config_name] = value

    return BenchmarkConfig(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        d_model=base_config["d_model"],
        d_ff=base_config["d_ff"],
        num_layers=base_config["num_layers"],
        num_heads=base_config["num_heads"],
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        mode=mode,
        device=str(resolve_device(args.device)),
        dtype=args.dtype,
        compile_model=args.compile,
        seed=args.seed,
    )
```

**模型创建**

```python
# 创建模型
def make_model(config: BenchmarkConfig, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    ).to(device=device, dtype=dtype)

    if config.compile_model:
        model = torch.compile(model)
    return model
```

**随机输入与target**

```python
# random input and target
def make_batch(config: BenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed + 1)
    tokens = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
        generator=generator,
    )
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
        generator=generator,
    )
    return tokens, targets
```

**单步测试**

```python
def run_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    mode: BenchmarkMode,
    device: torch.device,
    measure_memory: bool,
) -> StepTiming:
    # 梯度清零
    optimizer.zero_grad(set_to_none=True)
	
    # 如果要记录memory的话，需要reset一下
    if measure_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
	
    # 前向传播是一定要做的	
    synchronize(device)
    step_start = timeit.default_timer()

    forward_start = timeit.default_timer()
    logits = model(tokens)
    loss = cross_entropy(logits, targets)
    synchronize(device)
    forward_s = timeit.default_timer() - forward_start
	
    # 根据需要来记录反向传播
    backward_s = 0.0
    optimizer_s = 0.0
    if mode in {"forward_backward", "train_step"}:
        backward_start = timeit.default_timer()
        loss.backward()
        synchronize(device)
        backward_s = timeit.default_timer() - backward_start

    if mode == "train_step":
        optimizer_start = timeit.default_timer()
        optimizer.step()
        synchronize(device)
        optimizer_s = timeit.default_timer() - optimizer_start
	
    # 统计一下返回
    total_s = timeit.default_timer() - step_start
    peak_memory_gib = None
    if measure_memory and device.type == "cuda":
        peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3

    return StepTiming(
        forward_s=forward_s,
        backward_s=backward_s,
        optimizer_s=optimizer_s,
        total_s=total_s,
        peak_memory_gib=peak_memory_gib,
    )
```



**benchmark**

```python
# (mean, std)
def summarize(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std

# benchmark
def benchmark(config: BenchmarkConfig) -> tuple[dict[str, object], list[StepTiming]]:
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)
    model = make_model(config, device, dtype)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    tokens, targets = make_batch(config, device)

    for _ in range(config.warmup_steps):
        run_one_step(
            model=model,
            optimizer=optimizer,
            tokens=tokens,
            targets=targets,
            mode=config.mode,
            device=device,
            measure_memory=False,
        )

    timings = [
        run_one_step(
            model=model,
            optimizer=optimizer,
            tokens=tokens,
            targets=targets,
            mode=config.mode,
            device=device,
            measure_memory=True,
        )
        for _ in range(config.measurement_steps)
    ]

    summary: dict[str, object] = asdict(config)
    for field in ["forward_s", "backward_s", "optimizer_s", "total_s"]:
        mean, std = summarize([getattr(timing, field) for timing in timings])
        summary[f"{field}_mean"] = mean
        summary[f"{field}_std"] = std

    peak_memory_values = [timing.peak_memory_gib for timing in timings if timing.peak_memory_gib is not None]
    summary["peak_memory_gib_max"] = max(peak_memory_values) if peak_memory_values else None
    summary["num_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return summary, timings

```

**控制台输出与csv保存**

```python
# 控制台输出
def print_summary(summary: dict[str, object]) -> None:
    print(
        f"{summary['model_size']} | mode={summary['mode']} | "
        f"layers={summary['num_layers']} d_model={summary['d_model']} "
        f"ctx={summary['context_length']} batch={summary['batch_size']} "
        f"device={summary['device']} dtype={summary['dtype']} compile={summary['compile_model']}"
    )
    print(f"parameters: {int(summary['num_parameters']):,}")
    print(
        "timing mean +/- std (seconds): "
        f"forward {summary['forward_s_mean']:.6f} +/- {summary['forward_s_std']:.6f}, "
        f"backward {summary['backward_s_mean']:.6f} +/- {summary['backward_s_std']:.6f}, "
        f"optimizer {summary['optimizer_s_mean']:.6f} +/- {summary['optimizer_s_std']:.6f}, "
        f"total {summary['total_s_mean']:.6f} +/- {summary['total_s_std']:.6f}"
    )
    if summary["peak_memory_gib_max"] is not None:
        print(f"peak allocated memory: {summary['peak_memory_gib_max']:.3f} GiB")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```

**解析命令行参数**

```python
# 解析命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CS336 basics Transformer end-to-end steps.")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="small")
    parser.add_argument("--all-model-sizes", action="store_true", help="Run every non-toy model size from the assignment table.")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step", "all"], default="all")
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', 'cuda', or a device like 'cuda:0'.")
    parser.add_argument("--dtype", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"], default="float32")
    parser.add_argument("--compile", action="store_true", help="Compile the model with torch.compile before benchmarking.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, help="Optional path for a CSV summary.")
    return parser.parse_args()
```

**主程序**

```python
def main() -> None:
    args = parse_args()
    model_sizes = ["small", "medium", "large", "xl", "10b"] if args.all_model_sizes else [args.model_size]
    modes: list[BenchmarkMode] = ["forward", "forward_backward", "train_step"] if args.mode == "all" else [args.mode]

    rows: list[dict[str, object]] = []
    for model_size in model_sizes:
        args.model_size = model_size
        for mode in modes:
            config = build_config(args, mode)
            summary, _ = benchmark(config)
            print_summary(summary)
            print()
            rows.append(summary)

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
        print(f"Wrote CSV summary to {args.output_csv}")


if __name__ == "__main__":
    main()

```

简单的记录了一下：

>   ```bash
>   uv run python -m cs336_systems.benchmarking_script --model-size small --mode all
>   ```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1781254196496_image.png)

可见backward还是很花时间的。



### 1.2 Nsight Systems Profiler

benchmark 不能告诉我们时间和内存具体花在哪里了，因此无法揭示具体的优化机会。为了了解程序在每个组件（例如，函数）上花费了多少时间，我们可以使用**性能分析器（Profiler）**。

标准的 Python 性能分析器（如 CProfile）无法分析 CUDA 内核，因为这些内核是在 GPU 上异步执行的。

幸运的是，NVIDIA 提供了一个可以通过命令行 nsys 使用的性能分析器。

```bash
uv run nsys profile -- python -m cs336_systems.benchmarking_script --model-size small --mode all
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1781256561737_image.png)

讲义还给了一个更全面的性能分析运行示例：

```bash
uv run nsys profile  --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python -m cs336_systems.benchmarking_script --model-size small --mode all
```

在这个例子中，--trace 指定要记录哪些 API，--pytorch 在模块调用和自动求导期间插入 NVTX 标签，--cudabacktrace 和 --python-backtrace 提供更好的回溯信息，以了解给定内核是从代码中的何处调用的，而 --gpu-metrics-devices 则指定要测量哪个 GPU 的利用率。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1781257810411_image.png)

为运行添加性能分析并非没有代价，它总体上会减慢你的运行速度。通常，只启用在特定运行中所关注的功能是值得的。具体来说，当不需要回溯信息时，你可能想要移除 --cudabacktrace=all 和 --python-backtrace=cuda，因为它们会带来过大的开销。

我们还可以利用 nvtx range来注解代码，从而在nsys-ui 中得到更好的可视化展示：

```python
def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for the basics attention function with NVTX ranges."""
    with nvtx.range("attention/scores_qk"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(K.shape[-1])

    if mask is not None:
        with nvtx.range("attention/causal_mask"):
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("attention/softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("attention/final_pv"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

```

然后只需要替换一下就好了：

```python
def maybe_patch_attention(annotate_attention: bool) -> None:
    if annotate_attention:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

```

我们也可以在benchmark那里加一下：

```python
with nvtx_range("phase/forward", device):
    logits = model(tokens)
    loss = cross_entropy(logits, targets)
    
with nvtx_range("phase/backward", device):
	loss.backward()

with nvtx_range("phase/optimizer", device):
	optimizer.step()
```

然后又加了点别的，跑了一下，本来想顺势回答一下讲义这里提的几个问题，然后发现太麻烦了，要跑好几个，就算了（

主要就是掌握一下profile这个技能。



### 1.3 Mixed Precision

```python
import torch

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)

```

```text
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
```

我们发现仅仅做1000次简单的加法运算，精度误差还是蛮大的。

因为FP16的精度显然是要比FP32差的，虽然FP32 accumulator + FP16 increment：比纯 FP16 accumulation 好很多，但仍受 FP16 表示 `0.01` 的误差影响



然后这一节给了三个问题，第三个是混合精度的benchmark，就把前面的改一下就行了，这里就不跑了。



**1、ToyModel autocast dtype**

```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
```

假设我们在 GPU 上训练该模型，并且模型参数最初使用 FP32。我们想使用带有 FP16 的自动转换混合精度。请说明以下各项的数据类型：

-   在 autocast 上下文中的模型参数？
-   第一个前馈层的输出 (ToyModel.fc1)？
-   层归一化的输出 (ToyModel.ln)？
-   模型预测的 logits？
-   损失值？
-   模型的梯度？

| component                               | dtype           |
| --------------------------------------- | --------------- |
| model parameters inside autocast        | `torch.float32` |
| output of `fc1`                         | `torch.float16` |
| output of `LayerNorm`                   | `torch.float32` |
| predicted logits after `fc2`            | `torch.float16` |
| loss, if using standard PyTorch CE loss | `torch.float32` |
| model gradients                         | `torch.float32` |

>   多数损失函数为了保证数值稳定性，会使用 torch.float32，autocast不支持



**2、LayerNorm 为什么特殊**

LayNorm 要进行像mean 和 variance 这样的规约操作，所以对于精度很敏感，即使BF16和FP32有同样的指数位宽，但是精度不敏感，FP16也不用说了，还是得用FP32来保证数值稳定性。



### 1.4 Profiling Memory

这一节就是关于内存性能分析，然后pytorch提供了很强大的memory profiler：

```python
...
# 基准测试脚本中的热身阶段

# 开始记录内存历史
torch.cuda.memory._record_memory_history(max_entries=1000000)

... # 你的基准测试脚本中你想要进行性能分析的部分

# 保存一个 pickle 文件，供 PyTorch 的在线工具加载
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# 停止记录历史
torch.cuda.memory._record_memory_history(enabled=None)
```

然后生成的 pickle 文件可以扔到：https://docs.pytorch.org/memory_viz



然后问题要求的参数太大了实在跑不动就算了。



## 二、Single-GPU Memory

下面就是探讨如何将 Tensor 分片到多个 GPU 上的技巧。但也有一些技巧甚至可以应用于单GPU训练。其中最常见的是**梯度检查点（gradient checkpointing）**（也称为**激活检查点（activation checkpointing）**）。



### 3.1 Autograd Residuals

为了进行反向传播，我们会保存在前向传播中产生的激活值。但默认情况下，保存的数目远比我们预想的要多。这些保存的Tensor 被称为 **Residuals**，或简称为 **saved tensors**。

我们以 RMSNorm 为例，添加一些 hook 来观察 Tensor 何时保存以及访问：

```python
import torch
from torch import nn

x = torch.randn((4, 512, 2560), requires_grad=True)

class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x

def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

ln = RMSNorm(x.shape[-1])

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
y.sum().backward()
```

```bash
$ uv run scripts/autograd_experiment.py
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=<MulBackward0 object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None

Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=<MulBackward0 object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
```

为什么看起来简单的 RMSNorm，会保存这么多 tensor。

```python
x.pow(2)
mean(...)
+ eps
rsqrt(...)
x * rms
weight * x
```

每一个小 op 都可能为了自己的 backward 保存中间 tensor。

比如：

```
x.pow(2)
```

backward 需要原始 `x`。

```
mean(...)
```

backward 需要知道 reduction shape。

```
rsqrt(...)
```

backward 需要 `rsqrt` 的输出或者输入。

```
x * rms
```

backward 需要乘法两边的输入。

```
weight * x
```

backward 也需要 `weight` 和 `x`。

所以虽然只写了一行：

```
rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

但 autograd 看见的是一串小计算图。每个节点都说：“我 backward 的时候要用某些东西，你 forward 时帮我存一下。”



**pack_hook / unpack_hook 是什么**

```python
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```

这不是改变模型行为，而是给 autograd 加“监控器”。

当 PyTorch 在 forward 中保存某个 tensor 给 backward 用时，会调用：

```
pack_hook(t)
```

当 backward 真的要用这个 tensor 时，会调用：

```
unpack_hook(t)
```



#### 3.1.1 Operator Fusion

原始 RMSNorm 是很多小 op：

```text
pow -> mean -> add -> rsqrt -> mul -> mul
```

如果用 `torch.compile` 或自定义 fused kernel，把 RMSNorm 融合成一个“大 op”：

```text
RMSNorm(x, weight) -> output
```

那么 PyTorch 不再把它看成一堆小 op，而是看成一个整体。

```python
...
ln = torch.compile(RMSNorm(x.shape[-1]))
	
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
y.sum().backward()
```

于是 backward 只需要保存更少的东西。torch.compile 后输出变成：

```text
Saving residual: shape=[4, 512, 2560]
Saving residual: shape=[2560]
Saving residual: shape=[4, 512, 1]
```

也就是说，它只保存：

1.  输入 activation `x`
2.  RMSNorm weight
3.  一个小得多的 `rms` / normalization statistic



### 3.2 Activation Checkpointing

训练时 autograd 会在 forward 保存大量 activation / residuals 给 backward 用，导致显存爆炸。那有没有办法少存一点？

答案就是 checkpointing：

>   forward 时不要保存所有中间 activation，只保存少数 checkpoint；等 backward 真需要中间 activation 时，再重新跑一遍局部 forward，把它们临时算回来。

​	这就是典型的 **用额外计算换显存**。

3.1 我们通过 Operator Fusion 降低了 residual 的显存占用，但即使这样，仍然不容乐观：

```python
import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock

# num_layers for this model is 32
d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                         positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length))

# Fuse as much torch.compile will allow
block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True)

# Now logs the number of bytes saved
total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):
        # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

...
# Run forward pass, saving for backward
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = block(x)
print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")    
```

```
Total size of saved tensors in single TransformerBlock: 3651.31 MiB
```

我们的 Attention 计算需要大量的显存占用，即使后面我们实现了 FlashAttention，显存占用仍然是随层数线性增长。



**Activation Checkpointing 的核心想法就是把 forward 的过程分段，每段保存输入，backward 的时候每段重新计算一下前向过程，然后再进行backward，用完立刻释放。**



PyTorch 提供了 checkpoint 接口：

```python
from torch.utils.checkpoint import checkpoint

y = checkpoint(function, x, use_reentrant=False)
```

这里 `function` 是你想 checkpoint 的一段计算，比如两个 Transformer blocks：

```python
def two_blocks(x):
    x = block(x)
    x = block(x)
    return x
```

然后：

```python
x = checkpoint(two_blocks, x, use_reentrant=False)
```

意思是：

>   forward 时执行 `two_blocks(x)`，但是不要保存 `two_blocks` 内部那些 activation，只保存输入 `x`。backward 时如果需要，就从 `x` 重新跑 `two_blocks`。



讲义给了一个示例很好地展现了checkpoint的效果：

```python
def four_blocks(x):
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    return x

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks(x)

print(f"Total size of saved tensors in four TransformerBlocks: {total_size_bytes / (1024**2):.2f} MiB")
```

```text
Total size of saved tensors in four TransformerBlocks: 14605.25 MiB
```

但是如果加了checkpoint：

```python
from torch.utils.checkpoint import checkpoint
def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
    # checkpoint throws out all the saved tensors until the backward pass
    # when getting to the checkpointed block in the backward pass,
    # it reruns a forward pass to produce the saved tensors,
    # then completes normal backward pass.
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)

print(f"Total size of saved tensors in four TransformerBlocks with checkpointing: {total_size_bytes / (1024**2):.2f} MiB")
```

```python
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None 
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, grad_fn=None 
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None 
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, grad_fn=<torch.autograd.function.CompiledFunctionBackward object at 0x7aa0657a19d0> 
Total size of saved tensors in four TransformerBlocks with checkpointing: 160.00 MiB
```

**但它不是免费午餐**。Checkpointing 没有让计算消失，只是让你不存中间结果。

 checkpoint 后 memory 被分成两类：

1.  **long-term checkpoint memory**：checkpoint 输入要一直留着，直到 backward 用它重算
2.  **short-term recomputation memory**：backward 到某个 checkpoint 区间时，会临时重新 forward 一遍这个区间，这时候又会短暂产生这个区间内部的 activation



所以 recomputation 的时候还是有一定的显存占用的，如果我们checkpoint 开的比较少，那么显存压力还是比较大。

我们可以进一步做  **recursive checkpointing**

普通 checkpoint 是一层：

```python
x = checkpoint(run_blocks_1_to_16, x)
x = checkpoint(run_blocks_17_to_32, x)
```

Recursive checkpointing 是 checkpoint 里面再 checkpoint：

```python
def run_blocks_1_to_16(x):
    x = checkpoint(run_blocks_1_to_8, x)
    x = checkpoint(run_blocks_9_to_16, x)
    return x

def run_blocks_1_to_8(x):
    x = checkpoint(run_blocks_1_to_4, x)
    x = checkpoint(run_blocks_5_to_8, x)
    return x
```

这会进一步减少 peak activation memory，但会让 recomputation 次数增加。当然，计算开销也会增加。



讲义问了两个问题：

**(a) 忽略 compute cost，怎么最小化 peak activation memory**

如果完全忽略计算代价，那策略就是：尽可能递归 checkpoint，也就是不要让很多 block 的 residuals 同时存在。

可以把 N 个 block 递归二分：

```
def run_range(blocks, lo, hi, x):
    if hi - lo == 1:
        return checkpoint(blocks[lo], x, use_reentrant=False)

    mid = (lo + hi) // 2
    x = checkpoint(lambda x: run_range(blocks, lo, mid, x), x, use_reentrant=False)
    x = checkpoint(lambda x: run_range(blocks, mid, hi, x), x, use_reentrant=False)
    return x
```

如果忽略 compute，最激进的做法可以把 peak activation memory 降到非常低，接近：**O(log N)**

或者在理想化讨论里，如果只考虑 block residuals、并允许极端重算，**可以接近只保留路径上的 checkpoints**。

但是 compute 会爆炸，因为 backward 过程中同一段 forward 会被多次重算。

这就是这问的 tradeoff：**最低显存来自递归 checkpoint，但代价是大量重复计算。**



**(b) 如果只能有一层 recomputation，不允许 nested checkpoint**

这问更实际。

“不允许 nested checkpoint” 意思是你只能做类似：

```
x = checkpoint(run_some_consecutive_blocks, x)
x = checkpoint(run_next_consecutive_blocks, x)
...
```

不能 checkpoint 里面再 checkpoint。

那你要选择 checkpoint block size，比如：

```
每 1 层 checkpoint
每 2 层 checkpoint
每 4 层 checkpoint
每 8 层 checkpoint
```

对于 xl, batch=4, seq=2048，你需要实际 profile peak memory，比较相邻大小。

直觉：

-   **checkpoint group 太大：重算时临时 activation 太大。**
-   **checkpoint group 太小：保存 checkpoint 边界太多。**
-   最佳值在中间，常常需要实测。



## 三、GPU Kernels

### 3.1 Optimizing Attention with FlashAttention-2

#### 3.1.1 Benchmarking PyTorch Attention

这一节其实很短，主要是让我们 benchmark 现在的 PyTorch attention，亲眼看到：

```text
seq_len 变大时，attention 的时间和显存会爆炸式增长
```

我们现在的实现是标准 **scaled dot-product attention**：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\text{mask}\left(\frac{QK^T}{\sqrt{d_k}}\right)\right)V \tag{1}
$$
4.1要求 benchmark 时：

```
batch_size = 8
没有 multi-head 维度
Q, K, V shape = [batch_size, seq_len, d_model]
```

那么：

```
Q: [B, T, D]
K: [B, T, D]
V: [B, T, D]
```

`QK^T` 结果是：

```
S: [B, T, T]
```

这里最关键的是：

```
T x T
```

也就是 attention score matrix 对 sequence length 是 **二次方增长**。



### 3.2 Benchmarking JIT-Compiled Attention

#### 3.2.1 Example - Weighted Sum

讲义这里是拿Weighted Sum作为FlashAttention之前的热身。

>   weight是一个d维向量，X是一个[n, D]的矩阵，二者通过广播机制做hadamard乘积

```python
def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)
```

然后给了个 triton 实现，略微不同于lecture的做法，这里为了节省一些手写指针寻址的操作，用了tl.make_block_ptr

```python
import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,          # 输入指针
    output_ptr,                 # 输出指针
    x_stride_row, x_stride_dim, # 步长告诉我们如何在张量的每个轴上移动一个元素
    weight_stride_dim,          # 很可能为 1
    output_stride_row,          # 很可能为 1
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # 块的形状必须在编译时已知
):
    # 每个实例将计算 x 中一个行块的加权和。
    # `tl.program_id` 让我们可以检查当前运行在哪个线程块中
    row_tile_idx = tl.program_id(0)

    # 块指针让我们可以从一个 N 维的内存区域中进行选择
    # 并移动我们的选择区域。
    # 块指针必须知道：
    # - 指向张量第一个元素的指针
    # - 张量的整体形状，以处理越界访问
    # - 每个维度的步长，以正确使用内存布局
    # - 起始块的 N 维坐标，即 "offsets"
    # - 一次加载/存储的块形状
    # - 从主要到次要轴的内存维度顺序
    #   (= np.argsort(strides))，用于优化, 在 >=Hopper 架构上需要
    #   以支持 TMA
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 初始化一个用于写入的缓冲区
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载当前的块指针
        # 由于 ROWS_TILE_SIZE 可能不能整除 NUM_ROWS，且 D_TILE_SIZE 可能不能整除 D，
        # 我们需要对两个维度都进行边界检查
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")   # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)

        # 计算行的加权和。
        output += tl.sum(row * weight[None, :], axis=1)

        # 将指针移动到下一个块。
        # 这些是 (行, 列) 的坐标增量
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))      # 在最后一个维度上移动 D_TILE_SIZE
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))  # 移动 D_TILE_SIZE

    # 将输出写入输出块指针（每行一个标量）。
    # 由于 ROWS_TILE_SIZE 可能不能整除 NUM_ROWS，我们需要进行边界检查
    tl.store(output_block_ptr, output, boundary_check=(0,))
```

其实就是分块卷积求和。

值得注意的是，为什么 output 用 float32？

这就和1.3一样了，这是为了 accumulation 稳定性。即使输入可能是 fp16/bf16，累加通常希望用 fp32。**乘法可以低精度，累加最好高精度。**

然后我们可以包装一下：

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # 缓存 x 和 weight 以便在反向传播中使用，
        # 我们只会收到关于输出张量的梯度，并
        # 需要计算关于 x 和 weight 的梯度。
        D, output_dims = x.shape[-1], x.shape[:-1]

        # 将输入张量重塑为 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "维度不匹配"
        assert x.is_cuda and weight.is_cuda, "期望输入 CUDA 张量"
        assert x.is_contiguous(), "我们的指针算术将假设 x 是连续的"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # 沿嵌入维度大约循环 16 次
        ctx.ROWS_TILE_SIZE = 16  # 每个线程一次处理 16 个批次元素
        ctx.input_shape = input_shape

        # 需要初始化一个空的结果张量。注意，这些元素不一定是 0！
        y = torch.empty(output_dims, device=x.device)

        # 在我们的 1D 网格中启动具有 n 个实例的内核。
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])
```



**但 Triton kernel 不是普通 PyTorch op。PyTorch 不知道它怎么 backward。**所以我们还要包装一下：

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ...
        return y

    @staticmethod
    def backward(ctx, grad_out):
        ...
        return grad_x, grad_weight
```

这样就可以得到一个函数：

```python
f_weightedsum = WeightedSumFunc.apply
```

调用时像普通 PyTorch 函数一样：

```python
y = f_weightedsum(x, weight)
```



然后讲义给了backward的triton实现：

```python
@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,                   # 输入
    grad_output_ptr,                     # 梯度输入
    grad_x_ptr, partial_grad_weight_ptr, # 梯度输出
    stride_xr, stride_xd,
    stride_wd, stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")  # (ROWS_TILE_SIZE,)

        # 计算 grad_x 的外积
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # 为得到 grad_weight 结果，尽可能多地归约行
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))  # 在第 0 维永远不会越界

        # 将指针沿 D 维度移动到下一个块
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
```



#### 3.2.2 FlashAttention-2 Forward Pass

##### 3.2.2.1 标准 Attention 到底哪里慢

标准 Attention 流程：

1. 计算 S = $QK^T$，把 S 写入 HBM。
2. 从 HBM 读出 S，计算 P = softmax(S)，把 P 写入 HBM。
3. 从 HBM 读出 P 和 V，计算 O = PV，把 O 写入 HBM。

这个 IO 搬运太多了，开销巨大。

- `S` 很大。
- `P` 也很大。
- softmax、mask、dropout 等操作经常是 memory-bound，也就是主要时间花在读写数据上。
- forward 要保存中间矩阵，backward 还会继续用这些中间矩阵，显存压力会进一步放大。

所以标准 Attention 的痛点就是：它不仅要算 N x N 的 attention，还要把 N x N 的中间矩阵反复搬进搬出 HBM。



##### 3.2.2.2 FlashAttention 的思想

所以 FlashAttention 的核心动机就是

-   能不能让大部分临时计算都发生在很快但很小的 SRAM 里，
-   并且不要把完整 N x N attention matrix 写回 HBM？

其实经过lecture的学习以及assignment2前面的部分，不难理解FlashAttention的思想：

-   **把 Q、K、V 分块搬到 SRAM 中计算，在线维护 softmax 的归一化统计量**
-   **直接累积输出 O，从而避免把完整 N x N attention matrix 写入 HBM。**

拆开看有四层含义：

1. 分块（tiling）：不要一次处理完整 `N x N`，而是处理 `Q_i` 和 `K_j, V_j` 的小块。
2. 在线 softmax：虽然只看到了一个局部 block，也能维护全局 softmax 所需的 row max 和 denominator。
3. 直接累积输出：边读 `K/V` block，边更新对应的 output block。
4. backward 重算：forward 不保存完整 attention matrix，backward 再按 block 重算局部 attention。

矩阵乘法可以通过分块加速计算，但是 Softmax 我们为了保证数值稳定性，往往需要对指数减去最大值，而分块只能处理局部最大值，所以我们还要在合并block的时候，更新最大值并且对指数进行调整，从算法思想来看，比较easy，但实际我们写triton算子的时候其实还是比较麻烦的。这个步骤，被称为 **Online-Softmax**。

然后因为分块重写了attention forward，我们还要手写backward，这个过程同样是 分块处理，并且为了降低显存开销，我们会加入前面用到的 重计算。



##### 3.2.2.3 FlashAttention 的 IO

讲义强烈建议去读原论文，这里我简单看了一下，原论文 Theorem 2 给出：
$$
FlashAttention \ HBM \ accesses: \Theta(N^2 d^2 / M)
$$
其中：

- `N` 是 sequence length。
- `d` 是 head dimension。
- `M` 是 SRAM size。

一个直观理解：

-   SRAM 越大，每次能放进去的 block 越大；
-   block 越大，重复从 HBM 搬运 Q/K/V 的次数越少；
-   所以 HBM access 和 1/M 成正相关。

论文指出，在典型设置下 `d = 64` 或 `128`，而 SRAM 约为 `100KB` 量级，`d^2` 比 `M` 小很多，因此 FlashAttention 的 HBM access 会明显少于标准实现。



##### 3.2.2.4 Memory complexity

原论文 Theorem 1 说明 FlashAttention：

$$
返回 exact \ O = softmax(QK^T)V \\
FLOPs: O(N^2 d) \\
additional \ memory: O(N)
$$
这里的 `O(N)` additional memory 主要就是每行的 softmax 统计量，例如 `m` 和 `l`。这和标准 Attention 保存 `N x N` 中间矩阵形成鲜明对比。

注意这个说法的边界：

- 它说的是 attention 模块额外中间存储的复杂度。
- 整个 Transformer 训练仍然还会保存其他层的 activation。
- **FlashAttention 没有把 Attention 的计算量从 `O(N^2 d)` 改成线性。**

##### 3.2.2.5 pytorch test

下面就是比较麻烦的内容了，手搓FlashAttention。不过为了避免一上来就写triton那么麻烦的东西，讲义让我们先写一个pytorch版本的标准自注意力。

经典的前向传播：

```python
def _attention_forward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用普通 PyTorch 计算 attention 输出和每一行的 logsumexp。

    这个函数主要作为参考实现，也用于自定义 backward 里的重算公式。
    """
    d = q.shape[-1]
    scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
    if is_causal:
        n_queries, n_keys = q.shape[-2], k.shape[-2]
        causal_mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
        scores = torch.where(causal_mask[None, :, :], scores, torch.tensor(-1e6, device=q.device, dtype=scores.dtype))

    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse[..., None])
    output = torch.bmm(probs, v)
    return output, lse
```

我们额外返回lse，保存激活值，用于后续 backward

包装一下：

```python
class FlashAttentionPytorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        output, lse = _attention_forward_pytorch(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output
```

然后根据公式写一下backward：
$$
L_i = \log\left(\sum_j \exp(S_{ij})\right)
$$

$$
\begin{align}
S &= QK^\top / \sqrt{d}\\
P_{ij} &= \exp(S_{ij} - L_i) \\
dV &= P^\top dO \\
dP &= dO V^\top \\
dS_{ij} &= P_{ij}(dP_{ij} - D_i) \\
dQ &= dS K / \sqrt{d} \\
dK &= dS^\top Q / \sqrt{d}
\end{align}
$$



```python
def _attention_backward_pytorch(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.bmm(q, k.transpose(1, 2)) * scale
    if is_causal:
        n_queries, n_keys = q.shape[-2], k.shape[-2]
        causal_mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
        scores = torch.where(causal_mask[None, :, :], scores, torch.tensor(-1e6, device=q.device, dtype=scores.dtype))

    probs = torch.exp(scores - lse[..., None])

    # D_i = rowsum(O_i * dO_i)，用于简化 softmax backward。
    d = torch.sum(output * grad_out, dim=-1)
    grad_v = torch.bmm(probs.transpose(1, 2), grad_out)
    grad_p = torch.bmm(grad_out, v.transpose(1, 2))
    grad_s = probs * (grad_p - d[..., None])
    grad_q = torch.bmm(grad_s, k) * scale
    grad_k = torch.bmm(grad_s.transpose(1, 2), q) * scale
    return grad_q, grad_k, grad_v
```

包装一下：

```python
class FlashAttentionPytorchFunction(torch.autograd.Function):
    """只使用 PyTorch ops 的 FlashAttention 风格 autograd.Function。
    并且只保存 Q/K/V/O/LSE，backward 时用公式重算梯度。
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        output, lse = _attention_forward_pytorch(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = _attention_backward_pytorch(grad_out, q, k, v, output, lse, ctx.is_causal)
        return grad_q, grad_k, grad_v, None
```

`adapters`

```python
def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttentionPytorchFunction
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1781430953519_image.png)



##### 3.2.2.6 Triton test

```python
@triton.jit
def _flash_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    stride_qb: tl.constexpr,
    stride_qq: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oq: tl.constexpr,
    stride_od: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lq: tl.constexpr,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    d_model: tl.constexpr,
    scale: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 一个 program instance 负责一个 batch 中的一块 query rows。
    batch_idx = tl.program_id(0)
    query_block_idx = tl.program_id(1)

    q_offsets = query_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_offsets = tl.arange(0, BLOCK_K)
    d_offsets = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + batch_idx * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd,
        mask=(q_offsets[:, None] < n_queries) & (d_offsets[None, :] < d_model),
        other=0.0,
    )

    # 在线 softmax 状态：m 是 running max，l 是 running denominator，acc 是未归一化输出累积。
    m = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

    for key_start in range(0, n_keys, BLOCK_K):
        cols = key_start + k_offsets
        k_tile = tl.load(
            k_ptr + batch_idx * stride_kb + cols[:, None] * stride_kk + d_offsets[None, :] * stride_kd,
            mask=(cols[:, None] < n_keys) & (d_offsets[None, :] < d_model),
            other=0.0,
        )
        v_tile = tl.load(
            v_ptr + batch_idx * stride_vb + cols[:, None] * stride_vk + d_offsets[None, :] * stride_vd,
            mask=(cols[:, None] < n_keys) & (d_offsets[None, :] < d_model),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k_tile)) * scale
        valid_mask = (q_offsets[:, None] < n_queries) & (cols[None, :] < n_keys)
        if is_causal:
            valid_mask = valid_mask & (q_offsets[:, None] >= cols[None, :])
        scores = tl.where(valid_mask, scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])

        # 注意对前面的指数进行调整
        alpha = tl.exp(m - m_new)
        l_new = alpha * l + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new
        l = l_new

    output = acc / l[:, None]
    lse = m + tl.log(l)

    tl.store(
        out_ptr + batch_idx * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od,
        output,
        mask=(q_offsets[:, None] < n_queries) & (d_offsets[None, :] < d_model),
    )
    tl.store(
        lse_ptr + batch_idx * stride_lb + q_offsets * stride_lq,
        lse,
        mask=q_offsets < n_queries,
    )

class FlashAttentionTritonFunction(torch.autograd.Function):
    """Triton forward + PyTorch 公式 backward 的 FlashAttention 实现。

    4.2.2 的重点是 forward tiling 和 online softmax。为了让当前 pytest 的 backward
    也能通过，这里 backward 使用和 PyTorch 参考版相同的重算公式。
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            output, lse = _attention_forward_pytorch(q, k, v, is_causal)
            ctx.save_for_backward(q, k, v, output, lse)
            ctx.is_causal = is_causal
            return output
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Expected q, k, v to have shape [batch, seq, d_model].")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0] or k.shape[1] != v.shape[1] or q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
            raise ValueError("Incompatible q, k, v shapes.")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch_size, n_queries, d_model = q.shape
        n_keys = k.shape[1]

        # 当前课程测试使用 D=64；这里选择 64 作为 hidden tile，足够覆盖常见 head dim。
        block_q = 16
        block_k = 32
        block_d = triton.next_power_of_2(d_model)
        output = torch.empty_like(q)
        lse = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
        grid = (batch_size, triton.cdiv(n_queries, block_q))

        _flash_attention_forward_kernel[grid](
            q,
            k,
            v,
            output,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            n_queries,
            n_keys,
            d_model,
            1.0 / math.sqrt(d_model),
            is_causal,
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
        )

        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = _attention_backward_pytorch(grad_out.contiguous(), q, k, v, output, lse, ctx.is_causal)
        return grad_q, grad_k, grad_v, None
```



![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1781431173831_image.png)



## 四、Distributed Data Parallel Training

然后这部分主要就是学习下怎么利用多gpu训练llm。

### 4.1 pytorch 实现单节点分布式通信

assignment给了一个示例代码：

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):  
    setup(rank, world_size) 
    data = torch.randint(0, 10, (3,)) 
    print(f"rank {rank} data (before all-reduce): {data}") 
    dist.all_reduce(data, async_op=False) 
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == '__main__':
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)
```

然后本地跑一下：

```text
rank 2 data (before all-reduce): tensor([2, 8, 9])
rank 1 data (before all-reduce): tensor([2, 0, 4])
rank 0 data (before all-reduce): tensor([0, 5, 0])
rank 3 data (before all-reduce): tensor([6, 9, 7])
rank 1 data (after all-reduce): tensor([10, 22, 20])
rank 0 data (after all-reduce): tensor([10, 22, 20])
rank 3 data (after all-reduce): tensor([10, 22, 20])
rank 2 data (after all-reduce): tensor([10, 22, 20])
```

其实就是做了一个 all_reduce 的操作，求和的结果送到了每个process上



-   `mp.spawn`
    -   生成 `nprocs` 个进程，每个进程都运行带有给定 `args` 参数的 `fn` 函数
    -   `fn` 函数会被以 `fn(rank, *args)` 的形式调用，其中 `rank` 是工作进程的索引（取值范围是 0 到 nprocs-1）
    -   distributed_demo 函数必须接受这个整数 rank 作为其第一个位置参数
    -   我们还会传入 world_size，它指的是工作进程的总数

**每个工作进程都属于一个进程组**，**该进程组通过 dist.init_process_group 初始化**。进程组代表多个工作进程，**这些进程将通过一个共享的主节点进行协调和通信。主节点由其 IP 地址和端口定义，并且主节点运行着 rank 为 0 的进程**。像 all-reduce 这样的集合通信操作会对进程组中的每个进程执行操作。

例子中使用 "gloo" 后端初始化了进程组，但还有其他后端可用：

"nccl" 后端将使用 NVIDIA NCCL 集合通信库，对于 CUDA 张量，通常会具有更高的性能。

不过 NCCL 只能在带有 GPU 的机器上使用，而 Gloo 可以在仅 CPU 的机器上运行。

在运行多 GPU 任务时，得确保不同的 rank 使用不同的 GPU。

一种实现方法是在 setup 函数中调用 torch.cuda.set_device(rank)，这样 tensor.to("cuda") 就会自动将其移动到指定的设备。

或者，也可以明确地创建一个按 rank 区分的设备字符串（例如，device = f"cuda:{rank}"），然后将这个设备字符串用作任何数据移动的目标设备（例如，tensor.to(f"cuda:{rank}")）。



#### 4.1.1 Best Practices for Benchmarking Distributed Applications

做实验看不同的 tensor 和 gpu 数量变化的时候 all_reduce 的时间和吞吐量怎么变化：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783751905056_image.png)

**左图：纵坐标表示All-reduce的时间**

-   all-reduce 通信开销同时受 tensor 大小和参与 GPU 数量影响；
-   大 tensor 更受带宽限制，小 tensor 更受固定启动/同步开销影响；GPU 数越多，梯度同步通常越贵。 

**右图：纵坐标：吞吐量**

-   吞吐量从 1 MiB 到 1024 MiB 明显上升，说明小 tensor 时很多时间花在固定开销上，而大 tensor 能更充分利用 GPU 间通信带宽（把很多小梯度分开 all-reduce 效率低； 把梯度合成较大的 bucket 再通信更高效） 
-   GPU 数越多，单次 all-reduce 通常越慢，同一个 tensor size 下，6 GPUs 通常比 4 GPUs 慢，4 GPUs 通常比 2 GPUs 慢。原因是更多 rank 参与 collective，会带来更多通信阶段和同步开销。从 2 GPUs 到 6 GPUs，GPU 数变成 3 倍，但 1024 MiB 时间只是：3.409 ms -> 5.039 m大约 1.48 倍，这说明NCCL 的 all-reduce 能利用单机多 GPU 的高速互联，并行化一部分通信



### 4.2 A Naïve Implementation of Distributed Data Parallel Training

naive 版本非常简单，广播一下参数，算完同步一下就好。让每个 rank 都拿到全局 batch 的平均梯度。随后每个 rank 以相同梯度执行相同的 optimizer.step()；因为初始参数和 optimizer state 一致，更新后模型仍保持一致。

`ddp.py`

```python
"""Minimal synchronous Distributed Data Parallel implementation"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class NaiveDistributedDataParallel(torch.nn.Module):
    """使用逐参数同步梯度的最小化数据并行容器。"""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("NaiveDistributedDataParallel requires an initialized process group.")

        self.module = module
        self._broadcast_initial_parameters()

    def _broadcast_initial_parameters(self) -> None:
        """将 rank 0 的初始参数复制到其他所有 rank。"""
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """将前向计算直接委托给被包装的模型。"""
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def synchronize_gradients(self) -> None:
        """对每个已产生的梯度分别求和，再计算所有 rank 的平均梯度。"""
        world_size = dist.get_world_size()
        for parameter in self.module.parameters():
            # 未参与本次图计算或被冻结的参数没有梯度，不应执行通信。
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=False)
            parameter.grad.div_(world_size)

```



`adapters.py`

```python
def get_ddp(module: torch.nn.Module) -> NaiveDistributedDataParallel:
    """
    返回一个负责参数广播和梯度同步的 DDP 容器。

    当前为第 5.2 节所需的朴素同步实现：反向传播结束后，
    对每个参数梯度单独执行 all-reduce。第 5.3 节可在此替换为
    扁平梯度或与反向计算重叠的实现。

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    return NaiveDistributedDataParallel(module)
```

Naive DDP 的问题

  - 每个参数张量单独发起一次 all_reduce，小张量很多时通信启动开销很大。
  - 必须等整个 backward 完成后才开始通信，通信时间完全暴露在训练关键路径上，无法与反向计算重叠。
  - 每个 rank 保存完整模型参数、梯度和 optimizer state，显存随模型规模增长很快。
  - 各 rank 必须执行相同通信顺序；任一 rank 较慢都会阻塞整个训练步骤。



![fa4e4acb520f37da4d7ab8d082e5d9f2.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783757122838_fa4e4acb520f37da4d7ab8d082e5d9f2.png)

-   单节点 2 GPU 上使用 NCCL backend 运行 naive DDP，每个 GPU 一个进程
-   xl模型配置：vocab size 10000、context length 512、d_model 2560、d_ff 10240、32 layers、32 heads，global batch size 为 4，因此每个 rank 处理 2 个样本。
-   测得平均每步训练时间约为 1070.7 ms，其中逐参数 all-reduce 梯度通信约为 60.8 ms，占总时间约 5.7%。
-   由于这个 naive 实现等 backward 全部完成后才开始通信，通信不能和反向计算重叠，所以这 5.7% 基本就是gradient all_reduce的开销。



### 4.3 Improving Upon the Minimal DDP Implementation

然后assignment对于前面的naive ddp 给了个减少通讯开销次数的方案：

-   把所有的梯度concatenate为一个tensor，然后all_reduce，可以用：

    ```python
    torch._utils._flatten_dense_tensors
    torch._utils._unflatten_dense_tensors
    ```



#### 4.3.1 Overlapping Computation with Communication of Individual Parameter Gradients

这部分就是想要让 backward 计算 和 通信尽可能重叠，然后要我们实现一个用于处理分布式数据并行训练的 Python 类。该类应封装任意 PyTorch nn.Module，并负责在训练前广播权重（使所有 rank 具有相同的初始参数）以及发起用于梯度平均的通信调用。实现以下公共接口：

    def __init__(self, module: torch.nn.Module)：给定一个已实例化、待并行化的 PyTorch nn.Module，构造一个将处理跨 rank 梯度同步的 DDP 容器。
    def forward(self, *inputs, **kwargs)：使用提供的定位参数和关键字参数调用被封装的模块的 forward 方法。
    def finish_gradient_synchronization(self)：当调用时，等待异步通信调用在 GPU 上完成。

要使用该类执行分布式训练，会将其传递给一个待封装的模块，然后在运行 optimizer.step 之前添加对 finish_gradient_synchronization 的调用，以确保依赖于梯度的优化步骤可以被安全地排入队列：

```python
model = ToyModel.to(device)
ddp_model = DDP(model)

for _ in range(train_steps):
    x, y = get_batch()
    logits = ddp_model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    optimizer.step()
```



`ddp.py`

```python
class OverlappedDistributedDataParallel(torch.nn.Module):
    """在反向传播期间异步同步逐参数梯度的 DDP 容器。"""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        _require_initialized_process_group()

        self.module = module
        self._world_size = dist.get_world_size()
        self._pending_works: list[Any] = []
        self._gradient_hook_handles: list[Any] = []
        _broadcast_initial_parameters(self.module)
        self._register_gradient_hooks()

    def _register_gradient_hooks(self) -> None:
        """在每个可训练参数的梯度累计完成后立即触发通信。"""
        for parameter in self.module.parameters():
            if parameter.requires_grad:
                handle = parameter.register_post_accumulate_grad_hook(self._queue_gradient_all_reduce)
                self._gradient_hook_handles.append(handle)

    @torch.no_grad()
    def _queue_gradient_all_reduce(self, parameter: torch.nn.Parameter) -> None:
        """缩放本地梯度并异步发起 all-reduce，不等待通信完成。"""
        gradient = parameter.grad
        if gradient is None:
            return

        # 先缩放本地梯度；all-reduce 求和后即为全局平均梯度。
        gradient.div_(self._world_size)
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
        self._pending_works.append(work)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """将前向计算直接委托给被包装的模型。"""
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """在 optimizer.step() 前等待所有尚未完成的梯度通信。"""
        for work in self._pending_works:
            work.wait()
        self._pending_works.clear()

```



## 五、Optimizer State Sharding

这个部分就是实现一下 optimizer state 分片，此时每个 rank 保留完整参数、完整梯度、**部分**Optimizer States。



```python
class ShardedOptimizer(torch.optim.Optimizer):
    """仅在参数所有者 rank 上保存 optimizer state 的包装器。"""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> None:
        # 1. 检查分布式环境是否就绪
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer requires an initialized process group.")

        self._optimizer_cls = optimizer_cls
        self._rank = dist.get_rank()	# 当前是第几号卡（比如 0, 1, 2, 3）
    	self._world_size = dist.get_world_size() 	# 总共有几张卡（比如 4）
        self._all_parameters: list[torch.Tensor] = []
        self._local_param_groups: list[dict[str, Any]] = []
        self._local_optimizer: torch.optim.Optimizer | None = None
        self._is_initializing = True

        # 2. 调用父类（原生Optimizer）的初始化
        # 注意：这里把所有参数都传给了父类。
        # 为什么？因为当你在外面调用 optimizer.zero_grad() 时，
        # 需要清空所有参数的梯度，所以父类必须知道所有的参数。
        super().__init__(params, kwargs)
        self._is_initializing = False
        
        # 3. 创建只属于当前卡的“本地优化器”
        self._create_local_optimizer()

    def _create_local_optimizer(self) -> None:
        """用当前 rank 拥有的非空参数组创建底层 optimizer。"""
        non_empty_groups = [group for group in self._local_param_groups if group["params"]]
        if non_empty_groups:
            self._local_optimizer = self._optimizer_cls(non_empty_groups)
            # 对外暴露的 state 也只包含当前 rank 的 optimizer state。
            self.state = self._local_optimizer.state

    def _make_local_group(self, param_group: dict[str, Any]) -> dict[str, Any]:
        """按全局参数顺序轮转分配参数，并保留原参数组的超参数。"""
        local_group = {key: value for key, value in param_group.items() if key != "params"}
        local_parameters: list[torch.Tensor] = []
        for parameter in param_group["params"]:
            owner_rank = len(self._all_parameters) % self._world_size
            self._all_parameters.append(parameter)
            if owner_rank == self._rank:
                local_parameters.append(parameter)
        local_group["params"] = local_parameters
        return local_group

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """添加完整参数组，并将其中参数分配给各个 rank。"""
        super().add_param_group(param_group)
        local_group = self._make_local_group(self.param_groups[-1])
        self._local_param_groups.append(local_group)

        if self._is_initializing or not local_group["params"]:
            return
        if self._local_optimizer is None:
            self._local_optimizer = self._optimizer_cls([local_group])
            self.state = self._local_optimizer.state
        else:
            self._local_optimizer.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None, **kwargs: Any) -> Any:
        """更新本 rank 的参数分片，再广播每个所有者更新后的参数。"""
        loss = None
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure=closure, **kwargs)
        elif closure is not None:
            # 没有本地参数时仍保持 Optimizer closure 的调用语义。
            with torch.enable_grad():
                loss = closure()

        for parameter_index, parameter in enumerate(self._all_parameters):
            owner_rank = parameter_index % self._world_size
            dist.broadcast(parameter, src=owner_rank)
        return loss

```



这个 `ShardedOptimizer` 是一个**包装器（Wrapper）**。它伪装成一个普通的 Optimizer，但背地里做了分工合作的事情。

**1. 初始化 `__init__`**

```python
def __init__(self, params, optimizer_cls, **kwargs):
    # 1. 检查分布式环境是否就绪
    # ...
    self._rank = dist.get_rank()             # 当前是第几号卡（比如 0, 1, 2, 3）
    self._world_size = dist.get_world_size() # 总共有几张卡（比如 4）
    
    # 2. 调用父类（原生Optimizer）的初始化
    # 注意：这里把所有参数都传给了父类。
    # 为什么？因为当你在外面调用 optimizer.zero_grad() 时，
    # 需要清空所有参数的梯度，所以父类必须知道所有的参数。
    super().__init__(params, kwargs)
    
    # 3. 创建只属于当前卡的“本地优化器”
    self._create_local_optimizer()
```

**2. 核心分发逻辑 `_make_local_group`**

这是决定“哪个参数归哪张卡管”的地方。代码使用了**轮询（Round-Robin）**的方式分配参数。
```python
def _make_local_group(self, param_group: dict[str, Any]) -> dict[str, Any]:
    local_group = {...} # 复制学习率等超参数
    local_parameters: list[torch.Tensor] = []
    
    # 遍历当前参数组里的每一个参数（Tensor）
    for parameter in param_group["params"]:
        # 【核心逻辑】：通过取模运算决定这个参数归谁管
        # len(self._all_parameters) 是当前参数的全局序号 (0, 1, 2...)
        owner_rank = len(self._all_parameters) % self._world_size
        self._all_parameters.append(parameter)
        
        # 如果这个参数的主人刚好是当前这张卡，就把它加入到 local_parameters 中
        if owner_rank == self._rank:
            local_parameters.append(parameter)
            
    local_group["params"] = local_parameters
    return local_group
```
**举个例子：** 假设有 2 张卡，模型有 4 个参数矩阵 [A, B, C, D]。
* A 是第0个，`0 % 2 = 0` -> 归卡 0 管。
* B 是第1个，`1 % 2 = 1` -> 归卡 1 管。
* C 是第2个，`2 % 2 = 0` -> 归卡 0 管。
* D 是第3个，`3 % 2 = 1` -> 归卡 1 管。

**3. 创建本地优化器 `_create_local_optimizer`**

```python
def _create_local_optimizer(self) -> None:
    non_empty_groups = [group for group in self._local_param_groups if group["params"]]
    if non_empty_groups:
        # 用刚才挑出来的、属于当前卡的参数，实例化真正的优化器（比如 Adam）
        self._local_optimizer = self._optimizer_cls(non_empty_groups)
        
        # 把包装器的 state 替换成本地优化器的 state。
        # 这样当前卡就只会在显存里保存它负责的那部分参数的动量/方差！
        self.state = self._local_optimizer.state
```

**4. 参数更新与同步 `step` (最关键的一步)**

当你在训练循环中调用 `optimizer.step()` 时，发生了什么？
```python
@torch.no_grad()
def step(self, closure=None, **kwargs):
    # 1. 本地更新
    if self._local_optimizer is not None:
        # 当前卡上的底层优化器开始工作。
        # 注意：卡 0 只更新了参数 A 和 C；卡 1 只更新了参数 B 和 D。
        # 此时，所有卡上的模型参数是不一致的！
        loss = self._local_optimizer.step(closure=closure, **kwargs)

    # 2. 全局同步 (Broadcast)
    # 遍历模型的所有参数
    for parameter_index, parameter in enumerate(self._all_parameters):
        # 算出这个参数刚刚是谁更新的（谁是 owner）
        owner_rank = parameter_index % self._world_size
        
        # 通信操作：把更新后的参数从 owner_rank 广播给所有的卡。
        # 如果当前卡是 owner，它就发送；如果不是，它就接收覆盖旧的值。
        dist.broadcast(parameter, src=owner_rank)
        
    # 当这个循环结束时，所有卡都集齐了拼图，拿到了最新、最完整的模型参数
    return loss
```



## 六、Fully-Sharded Data Parallel

然后就是实现一下 FSDP，把参数分片到各个rank上，降低显存开销。

    1. 初始化时先由 rank 0 广播参数，保证所有 rank 起点一致。
    2. 对 Linear 与 Embedding 的 weight：
       - 展平后按 rank 切分；
       - 空闲时参数仅保留本 rank 的 FP32 分片；
       - optimizer 因此只会为本地参数分片创建状态。
    
    3. 前向传播：
       - 当前层使用前执行 all_gather，临时拼出完整权重；
       - 当前层计算后立刻释放完整权重、恢复本地分片；
       - 当前层完成时异步预取第 i + 2 个分片层，实现通信与中间计算重叠。
    
    4. 反向传播：
       - 层反向前再次 all_gather 完整权重，以支持 grad_input 计算；
       - 完整梯度产生后执行异步 reduce_scatter，每个 rank 只保留自己的平均梯度分片；
       - Gloo/CPU 不支持原生 reduce_scatter 时，自动使用“all_reduce 后切片”的语义等价回退路径，保证测试可运行。
    
    5. RMSNorm、bias 等小参数不分片，但梯度会异步 all_reduce 平均，保持普通 DDP 语义。
    6. 支持 mixed precision：
       - master weight 与 optimizer 更新仍是 FP32；
       - compute_dtype=torch.float16 时，通信和层计算使用 FP16，降低带宽与计算开销；
       - 梯度在交给 optimizer 前恢复为 master weight 的 dtype。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class _ShardedWeight:
    """记录一个已分片权重在通信、还原和梯度同步时所需的元数据。"""

    parameter: nn.Parameter
    original_shape: torch.Size
    original_numel: int
    shard_numel: int
    master_shard: torch.Tensor
    prefetch_work: Any | None = None
    prefetch_input: torch.Tensor | None = None
    prefetch_outputs: list[torch.Tensor] | None = None


@dataclass
class _PendingShardedReduction:
    """保存尚未完成的 reduce-scatter 及其输入输出缓冲区的引用。"""

    work: Any
    weight: _ShardedWeight
    reduced_shard: torch.Tensor | None
    full_padded_gradient: torch.Tensor


class FullyShardedDataParallel(nn.Module):
    """对模型的 Linear / Embedding 权重进行全分片数据并行训练。

    module 中的参数对象不替换，因此外部仍可直接使用 torch.optim.AdamW(fsdp.parameters())；
    需要分片的权重空闲时是长度相同的一维 FP32 分片，前向和反向计算时临时
    替换为完整权重；
    
    Norm 等小层不分片，但在反向传播期间用 all-reduce 同步其梯度；
    
    compute_dtype 只影响通信与计算权重，master weight 和最终交给
    optimizer 的梯度始终保持原始（通常为 FP32）精度。
    """

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self._require_initialized_process_group()

        self.module = module
        self.compute_dtype = compute_dtype
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._sharded_weights: list[_ShardedWeight] = []
        self._weight_by_parameter_id: dict[int, _ShardedWeight] = {}
        self._sharded_modules: list[nn.Module] = []
        self._pending_sharded_reductions: list[_PendingShardedReduction] = []
        self._pending_replicated_reductions: list[tuple[Any, nn.Parameter]] = []
        self._hook_handles: list[Any] = []

        # 所有 rank 必须从同一组参数开始；之后只有分片参数的 owner 更新自己的部分。
        self._broadcast_initial_parameters()
        self._install_sharding_and_hooks()

    @staticmethod
    def _require_initialized_process_group() -> None:
        """在创建容器前验证分布式通信环境已初始化。"""
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("FullyShardedDataParallel requires an initialized process group.")

    @torch.no_grad()
    def _broadcast_initial_parameters(self) -> None:
        """将 rank 0 的初始参数广播给所有 rank，避免训练起点不一致。"""
        for parameter in self.module.parameters():
            dist.broadcast(parameter, src=0)

    @staticmethod
    def _is_shardable_module(module: nn.Module) -> bool:
        """仅分片大矩阵层；Norm 等小参数层保留复制副本以避免通信延迟。"""
        # 导入放在这里，避免仅使用 PyTorch 原生层时强制依赖作业 1 的包。
        from cs336_basics.model import Embedding as AssignmentEmbedding
        from cs336_basics.model import Linear as AssignmentLinear

        return isinstance(module, (AssignmentLinear, AssignmentEmbedding, nn.Linear, nn.Embedding))

    @torch.no_grad()
    def _install_sharding_and_hooks(self) -> None:
        """分片权重，并为前向、反向与梯度完成阶段注册hook。"""
        for module in self.module.modules():
            if not self._is_shardable_module(module):
                continue

            # 课程模型的 Linear / Embedding 都有 weight。原生 Linear 的 bias 很小，
            # 保持复制并通过后面的梯度 all-reduce 同步，避免改变模块的调用约定。
            parameter = getattr(module, "weight", None)
            if not isinstance(parameter, nn.Parameter):
                continue

            weight = self._weight_by_parameter_id.get(id(parameter))
            if weight is None:
                weight = self._shard_parameter(parameter)
                self._sharded_weights.append(weight)
                self._weight_by_parameter_id[id(parameter)] = weight

            layer_index = len(self._sharded_modules)
            self._sharded_modules.append(module)
            self._register_sharded_module_hooks(module, weight, layer_index)

        # 未分片的参数（例如 RMSNorm 与 nn.Linear.bias）仍需要 DDP 语义的梯度平均。
        for parameter in self.module.parameters():
            if id(parameter) not in self._weight_by_parameter_id and parameter.requires_grad:
                self._hook_handles.append(parameter.register_post_accumulate_grad_hook(self._queue_replicated_gradient_sync))

    @torch.no_grad()
    def _shard_parameter(self, parameter: nn.Parameter) -> _ShardedWeight:
        """把一个完整参数展平、补齐并切出当前 rank 的本地 master 分片。"""
        original_shape = parameter.shape
        original_numel = parameter.numel()
        shard_numel = (original_numel + self._world_size - 1) // self._world_size

        # collective 要求每个 rank 的输入形状相同；末尾补零只用于通信，从不参与计算。
        padded = torch.zeros(shard_numel * self._world_size, dtype=parameter.dtype, device=parameter.device)
        padded[:original_numel].copy_(parameter.detach().reshape(-1))
        master_shard = padded.narrow(0, self._rank * shard_numel, shard_numel).clone()

        # 将 Parameter 的 data 指向本地分片。optimizer 创建于 FSDP 之后时，看到的正是它。
        parameter.data = master_shard
        return _ShardedWeight(parameter, original_shape, original_numel, shard_numel, master_shard)

    def _register_sharded_module_hooks(self, module: nn.Module, weight: _ShardedWeight, layer_index: int) -> None:
        """注册在权重使用前 gather、使用后释放、反向时再 gather 的层级钩子。"""

        def forward_pre_hook(_: nn.Module, _inputs: tuple[Any, ...]) -> None:
            self._materialize_weight(weight)

        def forward_post_hook(_: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            # 本层输出已计算完成，完整权重不再被前向传播需要，立刻释放以降低峰值显存。
            self._restore_master_shard(weight)

            # 第 i 层完成后预取第 i+2 层：预取距离为 2，既让通信与中间两层计算重叠，
            # 又不会同时长期保留过多完整权重。
            prefetch_index = layer_index + 2
            if prefetch_index < len(self._sharded_modules):
                future_module = self._sharded_modules[prefetch_index]
                future_weight = self._weight_by_parameter_id[id(future_module.weight)]
                self._start_prefetch(future_weight)
            return output

        def backward_pre_hook(_: nn.Module, _grad_output: tuple[torch.Tensor, ...]) -> None:
            # 许多 Linear backward 需要完整权重计算 grad_input，因此反向前重新 all-gather。
            self._materialize_weight(weight)

        self._hook_handles.extend(
            [
                module.register_forward_pre_hook(forward_pre_hook),
                module.register_forward_hook(forward_post_hook),
                module.register_full_backward_pre_hook(backward_pre_hook),
                weight.parameter.register_post_accumulate_grad_hook(self._queue_sharded_gradient_reduce_scatter),
            ]
        )

    def _communication_weight(self, weight: _ShardedWeight) -> torch.Tensor:
        """返回通信用本地分片；mixed precision 时在通信前降低带宽占用。"""
        if self.compute_dtype is None:
            return weight.master_shard
        return weight.master_shard.to(self.compute_dtype)

    @torch.no_grad()
    def _start_prefetch(self, weight: _ShardedWeight) -> None:
        """异步预取一个未来层的完整权重；真正使用前由 ``_materialize_weight`` 等待。"""
        if weight.prefetch_work is not None:
            return

        communication_shard = self._communication_weight(weight)
        outputs = [torch.empty_like(communication_shard) for _ in range(self._world_size)]
        weight.prefetch_input = communication_shard
        weight.prefetch_outputs = outputs
        weight.prefetch_work = dist.all_gather(outputs, communication_shard, async_op=True)

    @torch.no_grad()
    def _materialize_weight(self, weight: _ShardedWeight) -> None:
        """等待或同步执行 all-gather，并把参数临时恢复为原始二维（或更高维）形状。"""
        if weight.prefetch_work is not None:
            weight.prefetch_work.wait()
            gathered = weight.prefetch_outputs
            weight.prefetch_work = None
            weight.prefetch_input = None
            weight.prefetch_outputs = None
            assert gathered is not None
        else:
            communication_shard = self._communication_weight(weight)
            gathered = [torch.empty_like(communication_shard) for _ in range(self._world_size)]
            dist.all_gather(gathered, communication_shard)

        # 删除补齐元素并还原原始形状；此张量的 dtype 是 compute_dtype 或 master dtype。
        full_weight = torch.cat(gathered, dim=0)[: weight.original_numel].view(weight.original_shape)
        weight.parameter.data = full_weight

    @torch.no_grad()
    def _restore_master_shard(self, weight: _ShardedWeight) -> None:
        """将参数 data 切回本地 FP32 分片，释放临时 all-gather 缓冲区的引用。"""
        weight.parameter.data = weight.master_shard

    @torch.no_grad()
    def _queue_sharded_gradient_reduce_scatter(self, parameter: nn.Parameter) -> None:
        """将完整梯度异步规约并仅保留当前 rank 对应的平均梯度分片。"""
        weight = self._weight_by_parameter_id[id(parameter)]
        gradient = parameter.grad
        if gradient is None:
            self._restore_master_shard(weight)
            return

        # 先转回 master dtype，再展平补齐，使 reduce-scatter 的每个输出分片大小一致。
        flat_gradient = gradient.detach().to(weight.master_shard.dtype).reshape(-1)
        padded_gradient = torch.zeros(
            weight.shard_numel * self._world_size,
            dtype=weight.master_shard.dtype,
            device=weight.master_shard.device,
        )
        padded_gradient[: weight.original_numel].copy_(flat_gradient)

        # 此时全量 gradient 不应继续绑定在 Parameter 上；恢复分片后 optimizer 才能安全读取它。
        parameter.grad = None
        self._restore_master_shard(weight)

        # NCCL 等后端支持真正的 reduce-scatter。Gloo 目前不提供此 collective，故使用
        # all-reduce 后切片作为仅 CPU 测试环境下的语义等价回退路径。
        if self._world_size == 1:
            self._pending_sharded_reductions.append(
                _PendingShardedReduction(_CompletedWork(), weight, None, padded_gradient)
            )
        elif dist.get_backend() != "gloo":
            reduced_shard = torch.empty_like(weight.master_shard)
            work = dist.reduce_scatter_tensor(reduced_shard, padded_gradient, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_sharded_reductions.append(
                _PendingShardedReduction(work, weight, reduced_shard, padded_gradient)
            )
        else:
            work = dist.all_reduce(padded_gradient, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_sharded_reductions.append(_PendingShardedReduction(work, weight, None, padded_gradient))

    @torch.no_grad()
    def _queue_replicated_gradient_sync(self, parameter: nn.Parameter) -> None:
        """对未分片参数异步 all-reduce，使其遵循普通 DDP 的平均梯度语义。"""
        if parameter.grad is None:
            return
        parameter.grad.div_(self._world_size)
        work = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._pending_replicated_reductions.append((work, parameter))

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """把前向计算委托给被包装模型；各层 hook 会按需管理权重。"""
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """在 ``optimizer.step`` 前等待通信，并把本地分片梯度写回各参数。"""
        for pending in self._pending_sharded_reductions:
            pending.work.wait()
            if pending.reduced_shard is None:
                # Gloo 回退：all-reduce 后仅取当前 rank 的那段，效果等同 reduce-scatter。
                local_gradient = pending.full_padded_gradient.narrow(
                    0, self._rank * pending.weight.shard_numel, pending.weight.shard_numel
                ).clone()
            else:
                local_gradient = pending.reduced_shard
            pending.weight.parameter.grad = local_gradient.div_(self._world_size).view_as(pending.weight.master_shard)
        self._pending_sharded_reductions.clear()

        for work, _parameter in self._pending_replicated_reductions:
            work.wait()
        self._pending_replicated_reductions.clear()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """为检查点或测试重建完整参数；不会改变训练时仍为分片的参数 data。"""
        full_params: dict[str, torch.Tensor] = {}
        for name, parameter in self.module.named_parameters():
            weight = self._weight_by_parameter_id.get(id(parameter))
            if weight is None:
                full_params[name] = parameter.detach().clone()
                continue

            local_shard = weight.master_shard
            shards = [torch.empty_like(local_shard) for _ in range(self._world_size)]
            dist.all_gather(shards, local_shard)
            full_params[name] = torch.cat(shards, dim=0)[: weight.original_numel].view(weight.original_shape).clone()
        return full_params


class _CompletedWork:
    """让单 rank 情况复用异步通信完成路径的极小适配对象。"""

    def wait(self) -> bool:
        return True

```



**1、两个数据类 (Dataclass)**

- `_ShardedWeight`：这就是那“1/4 字典”的档案袋。记录了这个参数原来有多大（`original_shape`），分片后有多大（`shard_numel`），以及你手里拿着的那个真实的 1/4 碎片（`master_shard`）。它还负责记录提前拉取别人碎片（Prefetch）的任务。
- `_PendingShardedReduction`：用来存放还没在后台同步完的“1/4 梯度”任务。



**2、初始化与分片 (`_shard_parameter`)**

```python
def _shard_parameter(self, parameter: nn.Parameter) -> _ShardedWeight:
```
- 切分的时候，不能整除就向上取整，然后padding
- 代码中用 `padded = torch.zeros(...)` 



**3、核心魔法：钩子 (Hooks) 机制**

FSDP 的运作全靠在每一层计算前后hook
在 `_register_sharded_module_hooks` 中，我们注册了 4 个 Hook：

1. **`forward_pre_hook` (前向计算前):**
   - 触发动作：`self._materialize_weight(weight)`。
   - 作用：这一层要开始算了，赶紧把 4 张卡的碎片收集过来（**All-Gather**），拼成完整的权重。
2. **`forward_post_hook` (前向计算后):**
   - 触发动作：`self._restore_master_shard(weight)`。
   - 作用：这一层算完了，完整权重没用了，赶紧扔掉（把指针切回本地小碎片），释放显存。同时，顺便提前去网络上拉取（Prefetch）后面要用到的层的参数，让计算和网络通信重叠，提高速度。
3. **`backward_pre_hook` (反向传播前):**
   - 触发动作：`self._materialize_weight(weight)`。
   - 作用：要算梯度了，同样需要完整的权重，再次通过 All-Gather 拼全。
4. **`post_accumulate_grad_hook` (梯度计算完后):**
   - 触发动作：`self._queue_sharded_gradient_reduce_scatter(parameter)`。
   - 作用：此时得到了这一层完整的梯度。但我们不需要完整梯度，只需要更新我们自己的 1/4。于是发起一次 **Reduce-Scatter** 通信，把所有卡的完整梯度加起来，然后把属于你的那 1/4 梯度留给你。算完后，再次把权重扔掉切回小碎片。



**4、小参数层的特殊处理**

代码里有 `_is_shardable_module` 这么个判断。

对于 `Linear` 和 `Embedding` 这种动辄几个 G 的大矩阵，我们进行分片。
但对于 `LayerNorm` 等只有几千个参数的小层，或者 `bias`，分片的收益太低了（省不了多少内存，反而增加了通信时间的开销）。
所以对这些小参数，代码选择**不分片（保持复制）**，只在最后算完梯度时做一个常规的梯度求平均（`_queue_replicated_gradient_sync` 中调用的 **All-Reduce**）。



**5、收尾工作 (`finish_gradient_synchronization`)**

在执行 `optimizer.step()` 之前，必须调用这个函数。
因为之前的网络通信（Reduce-Scatter 算梯度）都是**异步 (async_op=True)** 的（为了不阻塞 CPU）。这个函数的作用就是：

- 等待所有网络通信完成（`wait()`）。
- 将真正属于你的那 1/k 梯度正确地赋值给 `parameter.grad`。
- 之后，就可以放心地调用 `optimizer.step()` 更新你本地的 1/4 参数了。



## 七、Analyzing Parallelism Strategies

### 7.1 Communication Primitives

考虑 N 个设备，每对设备之间通过一条链路连接。假定设备带宽为 W B/s。那么如何实现 gather 和 reduce呢？

全局张量大小为 `S` bytes，每个 rank 初始持有 `S/N` bytes。环上共执行 `N-1` 轮，
每轮每卡发送 `S/N` bytes，因此：
$$
T_{AG}(S,N)=\frac{N-1}{N}\frac{S}{W}
$$
实现 **all-gather** 的一种常见方式是环形 **all-gather**，环上的每个节点每次向下一个节点传送自己上一轮得到的数据，这样除去初始各节点拥有的一片数据，一共需要传递 N - 1 次就能实现 **all-gather**。

然后分析一下**Ring reduce-scatter**

每卡起初持有完整的 `S` bytes 张量。将其分成 `N` 块并在环上转发、累加，最后每卡仅
保留一个已经规约的块。每轮发送量仍为 `S/N`，所以：
$$
T_{RS}(S,N)=\frac{N-1}{N}\frac{S}{W}
$$


标准 ring all-reduce 等于一次 reduce-scatter 加一次 all-gather：

$$
T_{AR}(S,N)=2\frac{N-1}{N}\frac{S}{W}.
$$


![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783834221603_image.png)

**答案：**
$$
T=\frac{(N-1)S}{W}
$$
证明：环上每个节点每轮都能新累加一块数据，那么只需要N - 1轮，环上节点就能够全部完成reduce。



### 7.2 Analyzing Data Parallel

DP 仅沿 batch 维切分数据。每个 rank 持有完整模型、处理 `B/N_DP` 个样本；前向没有通信。

反向时，每个 rank 得到自己的局部 `dW`，通过 all-reduce 求和（通常再除以`N_DP`）得到所有 rank 相同的完整梯度。

因此 DP 的**优点是通信模式简单**；**缺点是每张卡都复制参数、梯度和 optimizer state**。

给了个例子，后面要基于这些式子回答问题：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783836400266_image.png)

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783836183085_image.png)

**(a) 每个 rank 的反向 FLOPs**
$$
F_{bwd}^{DP}=\frac{12BDD_{FF}}{N_{DP}}
$$
全局反向仍有六次相同规模的矩阵乘法；DP 只把 batch 维平均切给 `N_DP` 个 rank，故
每卡计算量缩小为原来的 `1/N_DP`。

**(b) 每个 rank 的反向通信时间**

三个权重的总 FP16 大小为 `S_W=6DD_FF` bytes。对它们执行一次 ring all-reduce：

$$
T_{bwd,comm}^{DP}
=2\frac{N_{DP}-1}{N_{DP}}\frac{6DD_{FF}}{W}
=\frac{12(N_{DP}-1)DD_{FF}}{N_{DP}W}
$$
DP 前向没有 collective；反向只需要同步完整权重梯度。



**(c) 可保持 compute-bound 的最大 `N_DP`**

要求通信不超过反向计算：

$$
\frac{12(N_{DP}-1)DD_{FF}}{N_{DP}W}
\le
\frac{12BDD_{FF}}{N_{DP}C}.
$$
约去公共项可得：

$$
N_{DP}\le 1+\frac{BW}{C}
$$
**解释：**DP 的梯度通信不随 batch 变小，而每卡反向计算正比于 `B/N_DP`；因此较大的 global batch `B` 能支撑更多 DP rank。这也是 DP 常被称为“由 batch size 决定扩展性”的原因。



### 7.3 Analyzing Fully Sharded Data Parallel

FSDP 同样切分 batch，但还将权重、梯度和 optimizer state 沿 FSDP 轴切分：

- 前向前：对每个即将使用的权重做 all-gather；使用后释放完整权重。
- 反向前：再次 all-gather 权重，以计算输入梯度。
- 反向后：对完整局部梯度做 reduce-scatter，每卡只留下本地参数分片的梯度。

FSDP 以额外的 all-gather 为代价换取显存节省。实际系统会预取下一层权重，使通信尽可能与其他层计算重叠。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783871779746_image.png)

**(a) 前向和反向 FLOPs**

FSDP 仍只按 batch 切计算，因此每卡计算量与 DP 相同：

$$
F_{bwd}^{FSDP}=\frac{12BDD_{FF}}{N_{FSDP}}
\qquad
F_{fwd}^{FSDP}=\frac{6BDD_{FF}}{N_{FSDP}}
$$
权重存储是否分片不会改变完整 FFN 的数学计算；all-gather 后，每个 rank 仍对本地`B/N_FSDP` 个样本做完整层计算。



**(b) 前向和反向通信时间**

前向：三个权重都要 all-gather 一次：

$$
T_{fwd,comm}^{FSDP}
=\frac{N_{FSDP}-1}{N_{FSDP}}\frac{6DD_{FF}}{W}
$$
反向：先 all-gather 三个权重，再对三个完整梯度 reduce-scatter：

$$
T_{bwd,comm}^{FSDP}
=2\frac{N_{FSDP}-1}{N_{FSDP}}\frac{6DD_{FF}}{W}
=\frac{12(N_{FSDP}-1)DD_{FF}}{N_{FSDP}W}
$$
注意 FSDP 的反向通信时间恰好等于 DP 的梯度 all-reduce 时间；因为 all-reduce 本身就是 reduce-scatter 加 all-gather，而 FSDP 将这两个阶段分别用于“取权重”和“留梯度分片”。



**(c) 可保持 compute-bound 的最大 `N_FSDP`**

反向：

$$
\frac{12(N_{FSDP}-1)DD_{FF}}{N_{FSDP}W}
\le
\frac{12BDD_{FF}}{N_{FSDP}C}
\quad\Rightarrow\quad
N_{FSDP}\le 1+\frac{BW}{C}
$$
前向：

$$
\frac{6(N_{FSDP}-1)DD_{FF}}{N_{FSDP}W}
\le
\frac{6BDD_{FF}}{N_{FSDP}C}
\quad\Rightarrow\quad
N_{FSDP}\le 1+\frac{BW}{C}
$$
**两个上限相同：前向的通信和计算都恰好是反向对应量的一半。**FSDP 并没有改善纯吞吐扩展上限；其首要价值是降低模型状态显存，从而让本来装不下的模型能够训练。



### 7.4 Analyzing Tensor Parallel

TP 切的是张量维度而非 batch：每个 rank 处理完整 batch，但只计算部分隐藏通道。

- **列并行（column parallel）**：按输出维切 `W`。每卡都需要完整输入，得到输出通道的一部分；若后续操作需要完整输出，使用 all-gather。
- **行并行（row parallel）**：按输入维切 `W`，同时切输入通道；每卡产生完整输出的部分和，通过 all-reduce 相加。

本题的 FFN 选择 `W_1,W_2` 列并行，`W_3` 行并行。这样 `W_1,W_2` 的局部输出恰好能直接作为 `W_3` 的局部输入，避免了中间 `D_FF` 激活的 all-gather；仅在末尾 all-reduce大小为 `(B,D)` 的输出。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783873258117_image.png)

**(a) 反向传播方程**

每个 rank `i` 先拥有前向保存的 `x,x_1^(i),x_2^(i),z^(i)`，以及来自后续网络的完整`dy`。令 `N=N_TP`，则每个 rank 接收的反向输入相同，即 `dy^(i)=dy`，并且：

$$
dz^{(i)}=dy(W_3^{(i)})^T,
$$

$$
dx_2^{(i)}=dz^{(i)}*f(x_1^{(i)}),
\qquad
dx_1^{(i)}=dz^{(i)}*f'(x_1^{(i)})*x_2^{(i)},
$$

$$
dW_3^{(i)}=(z^{(i)})^Tdy,
$$

$$
dW_2^{(i)}=x^Tdx_2^{(i)},
\qquad
dW_1^{(i)}=x^Tdx_1^{(i)},
$$

$$
dx_{local}^{(i)}=dx_1^{(i)}(W_1^{(i)})^T
+dx_2^{(i)}(W_2^{(i)})^T,
$$

$$
dx=\operatorname{all\text{-}reduce}\bigl(\{dx_{local}^{(i)}\}_{i=0}^{N-1}\bigr)
$$

其中 `dz^(i), dx_1^(i), dx_2^(i)` 的形状均为 `(B,D_FF/N_TP)`，而`dx_local^(i)` 的形状为 `(B,D)`。

这里 `dW_k^(i)` 已经是对应权重分片的正确梯度，不需跨 TP rank 求和；只有 `dx` 是各输出通道分片贡献之和，必须 all-reduce。`dy` 在每个 TP rank 可直接使用，因为前向末尾的all-reduce 已使输出在每卡相同；其反向会把同一 `dy` 提供给各 rank。



**(b) 前向和反向 FLOPs**
$$
F_{fwd}^{TP}=\frac{6BDD_{FF}}{N_{TP}}
\qquad
F_{bwd}^{TP}=\frac{12BDD_{FF}}{N_{TP}}
$$
每个矩阵乘法都沿 `D_FF` 维切为 `1/N_TP`，而 batch 没有切分；所有局部矩阵乘法总量因此均缩小为原来的 `1/N_TP`。



**(c) 前向和反向通信时间**

前向末尾 all-reduce 的激活形状为 `(B,D)`，其 FP16 大小为 `2BD` bytes：
$$
T_{fwd,comm}^{TP}
=2\frac{N_{TP}-1}{N_{TP}}\frac{2BD}{W}
=\frac{4(N_{TP}-1)BD}{N_{TP}W}
$$
反向仅对同样形状的 `dx` 执行一次 all-reduce，因此：

$$
T_{bwd,comm}^{TP}=\frac{4(N_{TP}-1)BD}{N_{TP}W}
$$


**(d) 可保持 compute-bound 的最大 `N_TP`**

反向：

$$
\frac{4(N_{TP}-1)BD}{N_{TP}W}
\le
\frac{12BDD_{FF}}{N_{TP}C}
\quad\Rightarrow\quad
N_{TP}\le 1+\frac{3D_{FF}W}{C}
$$
前向：

$$
\frac{4(N_{TP}-1)BD}{N_{TP}W}
\le
\frac{6BDD_{FF}}{N_{TP}C}
\quad\Rightarrow\quad
N_{TP}\le 1+\frac{3D_{FF}W}{2C}
$$
前向更严格，因为前向 FLOPs 只有反向的一半，而两者通信量相同。还应注意 `B,D` 被约掉了：TP 的扩展性主要由每个 token 的模型宽度 `D_FF` 和机器的带宽/算力比决定，而不是 global batch size。



### 7.5 2D Parallelism (FSDP + TP)

Batch size 和 参数大小限制了我们能够扩展到多少设备，但是将批次大小扩展到超过某个点后，性能会开始下降，因为梯度噪声显著减小，丧失了 SGD 的隐式正则化特性。

这个点通常被称为“**临界批次大小**”。而 **scaling law** 通常告诉我们模型应该有多大。

这一节，考虑一个简化场景：有人把所有的任务参数（批次大小、模型大小、带宽、加速器速度）都交给了我们。我们的工作是选择一个 FSDP 和 TP 的配置，使得在尽可能多的设备上扩展时，仍然保持计算受限而非通信受限。

令总设备数为：

$$
N=N_{FSDP}N_{TP}
$$
FSDP 轴切 batch 和权重的另一维；TP 轴切 `D_FF` 通道。每个设备 `(i,j)` 持有：

$$
W_1^{(i,j)},W_2^{(i,j)}:\left(\frac{D}{N_{FSDP}},\frac{D_{FF}}{N_{TP}}\right)
$$

$$
W_3^{(i,j)}:\left(\frac{D_{FF}}{N_{TP}},\frac{D}{N_{FSDP}}\right)
$$

前向中，先沿 FSDP 轴 all-gather 出每个 TP rank 所需的完整 TP 分片权重；然后做 TP 局部计算；最后沿 TP 轴 all-reduce 输出激活。它结合了 FSDP 的省显存与 TP 的模型维度扩展能力。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783875343721_image.png)

**(a) 每设备前向 FLOPs**
$$
F_{fwd}^{2D}=\frac{6BDD_{FF}}{N_{FSDP}N_{TP}}
$$
FSDP 轴把 batch 除以 `N_FSDP`，TP 轴把每个矩阵乘法的 `D_FF` 维除以 `N_TP`，两者
相乘即为每卡计算量的缩放比例。



**(b) 前向通信时间（两轴可重叠）**

沿 FSDP 轴：三个权重的总 FP16 大小在一个 TP rank 内为 `6DD_FF/N_TP` bytes，故：

$$
T_{FSDP}
=\frac{6(N_{FSDP}-1)DD_{FF}}{N_{FSDP}N_{TP}W}
$$
沿 TP 轴：输出激活形状为 `(B/N_FSDP,D)`，其 all-reduce 时间为：

$$
T_{TP}
=\frac{4(N_{TP}-1)BD}{N_{TP}N_{FSDP}W}
$$
两条轴使用可独立重叠的通信资源时，临界路径取较大者：

$$
T_{fwd,comm}^{2D}
=\max\left(
\frac{6(N_{FSDP}-1)DD_{FF}}{N_{FSDP}N_{TP}W},
\frac{4(N_{TP}-1)BD}{N_{TP}N_{FSDP}W}
\right)
$$


**(c) 两轴通信可重叠时，最优总规模**

要使 `max(T_FSDP,T_TP)` 不超过计算时间 `6BDD_FF/(N_FSDP N_TP C)`，两项都必须各自
不超过计算时间：

$$
N_{FSDP}\le 1+\frac{BW}{C},
$$

$$
N_{TP}\le 1+\frac{3D_{FF}W}{2C}
$$

因此可以各自取最大，得到：

$$
N=N_{FSDP}N_{TP}
\le
\left(1+\frac{BW}{C}\right)
\left(1+\frac{3D_{FF}W}{2C}\right)
$$
这比单独 DP/FSDP 或单独 TP 的上限更大：FSDP 轴利用 batch 提供的计算，TP 轴利用模型中间维度提供的计算，且两条通信轴可以并行推进。



**(d) 两轴通信不能重叠时，最优总规模**

不能重叠时，临界路径通信是两项之和。设：

$$
\alpha=\frac{BW}{C},\qquad \beta=\frac{2B}{3D_{FF}}
$$
由 `T_FSDP + T_TP <= T_compute`，乘以公共正项并化简：

$$
(N_{FSDP}-1)+\beta(N_{TP}-1)\le\alpha
$$
令 `x=N_FSDP-1`，则边界上 `N_TP=1+(\alpha-x)/\beta`，待最大化的总设备数为：

$$
N(x)=(x+1)\left(1+\frac{\alpha-x}{\beta}\right)
$$
在内部可行域 `N_FSDP,N_TP >= 1` 中，对 `x` 求导并令其为零，得到连续最优解：

$$
N_{FSDP}^*=\frac{1+\alpha+\beta}{2},
\qquad
N_{TP}^*=\frac{1+\alpha+\beta}{2\beta}
$$
所以：

$$
N\le \frac{(1+\alpha+\beta)^2}{4\beta}
=\frac{3D_{FF}}{8B}
\left(1+\frac{BW}{C}+\frac{2B}{3D_{FF}}\right)^2
$$
这里按题目要求忽略了 `N_FSDP,N_TP` 必须为整数的限制。若该连续解使任一轴小于 1，实际最优点应落在边界（即只用另一轴）；通常讨论的大规模训练区间中，两者均大于等于 1。





































