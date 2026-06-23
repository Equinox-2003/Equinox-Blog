---
title: "多GPU并行训练"
description: ""
date: 2026-06-16T17:08:34+08:00
lastmod: 2026-06-16T17:08:34+08:00
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









## 一、多 GPU 并行训练

### 1.1 为什么需要多 GPU

之前讲的 warps、thread blocks、shared memory、tiling、fusion，这些都是 **单 GPU 内部并行**。核心问题是：**计算单元离数据很远，要减少 HBM 访问**。

本讲扩展到 **多 GPU/多节点并行**。核心问题变成：计算在多张 GPU 上，数据、参数、梯度和激活也分散在多张 GPU 上，要减少 GPU 之间、节点之间的数据传输。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1781603884558_image.png)

一个粗略的通信层次是：

| 层次         | 例子                           | 速度直觉             |
| ------------ | ------------------------------ | -------------------- |
| 单 GPU 内部  | registers / shared memory / L1 | 最快                 |
| 单 GPU 显存  | HBM                            | 很快，但比片上存储慢 |
| 单节点多 GPU | NVLink / NVSwitch / PCIe       | 比 HBM 慢            |
| 多节点       | InfiniBand / Ethernet          | 最慢                 |

因此本讲的统一主题是：

> orchestrate computation to avoid data transfer bottlenecks

为什么要多 GPU？

1. **放不下**：参数、optimizer state、梯度、激活可能超过单卡显存。
2. **想更快**：使用更多 GPU FLOPs，提高训练吞吐。

但增加 GPU 不会自动变快。只要通信成为瓶颈，更多 GPU 可能只是在更快地等待彼此。



### 1.2 Collective Operations：分布式通信原语

多 GPU 编程中，不直接手写每一对 GPU 的 send/recv，通常使用 collective operations。Collective 表示所有 ranks 共同参与某种通信模式。

基本术语：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1781625619952_image.png)

- **rank**：某个进程/设备的编号，例如 `0, 1, 2, 3`。
- **world size**：参与通信的 rank 总数，例如 `4`。
- **process group**：参与某组 collective 的进程集合；本讲默认所有 ranks 在同一个 group 里。

#### 1.2.1 Broadcast

Broadcast 是从一个源 rank 复制到所有 ranks。

输入：

```python
rank0 = tensor([0., 1, 2, 3])
```

输出：

```python
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
```

典型场景：rank 0 读取 checkpoint，然后广播初始参数到所有 ranks。



#### 1.2.2 Scatter 与 Gather

Scatter 把 rank 0 上的 tensor 拆开，发给各个 ranks。

```python
# input on rank 0
rank0 = tensor([0., 1, 2, 3])

# output
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])
```

Gather 是反方向：各 rank 的 pieces 聚合到 rank 0。

```python
# input
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# output on rank 0
rank0 = tensor([0., 1, 2, 3])
```

这两个操作本身不一定是训练最常用的，但它们是理解 `all_gather` 和 `reduce_scatter` 的台阶。



#### 1.2.3 Reduce

Reduce 从所有 ranks 收集值，并做一个 associative/commutative operation，例如 sum、min、max。

```python
# input
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# sum output on rank 0
rank0 = tensor([6.])
```

训练里最常见的是 sum 或 average gradients。



#### 1.2.4 All-gather

All-gather 是 gather 到所有 ranks，而不只是 rank 0。

```python
# input
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# output on every rank
tensor([0., 1, 2, 3])
```

用例：每个 rank 持有参数 shard，forward 前 all-gather 得到完整参数。



#### 1.2.5 Reduce-scatter

Reduce-scatter 可以理解为：

1. 先沿 rank 维度 reduce。
2. 再把 reduce 后的结果 scatter 到不同 ranks。

输入：

```python
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])
```

输出：

```python
rank0 = tensor([6.])   # 0 + 1 + 2 + 3
rank1 = tensor([10.])  # 1 + 2 + 3 + 4
rank2 = tensor([14.])  # 2 + 3 + 4 + 5
rank3 = tensor([18.])  # 3 + 4 + 5 + 6
```

用例：backward 后不同 data shards 产生梯度，把梯度求和后分片存储。这是 ZeRO/FSDP 背后的核心通信之一。



#### 1.2.6 All-reduce

All-reduce 是最常见的训练 collective。它可以看成：

```text
all-reduce = reduce-scatter + all-gather
```

输入：

```python
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])
```

输出到每个 rank：

```python
tensor([6., 10, 14, 18])
```

数据并行里，每个 rank 计算本地梯度 \(g_r\)，然后平均：

$$
g = \frac{1}{P}\sum_{r=0}^{P-1} g_r
$$

其中 \(P\) 是 world size。PyTorch 里可以用 `ReduceOp.AVG` 直接平均，也可以用 sum 后除以 \(P\)。



#### 1.2.7 All-to-all

All-to-all 是更一般的通信：每个 rank 给每个其他 rank 发一块数据。

```python
rank0 = tensor([0., 1, 2, 3])      # send to ranks 0,1,2,3
rank1 = tensor([4., 5, 6, 7])
rank2 = tensor([8., 9, 10, 11])
rank3 = tensor([12., 13, 14, 15])
```

输出：

```python
rank0 = tensor([0, 4, 8, 12])
rank1 = tensor([1, 5, 9, 13])
rank2 = tensor([2, 6, 10, 14])
rank3 = tensor([3, 7, 11, 15])
```

MoE 中常见 all-to-all：token 根据 router 被分发到不同 expert 所在的 rank。balanced split 时，它看起来像 transpose；不均衡时还要处理 variable splits。

记忆方式：

- **reduce：做 sum/min/max 等归约。**
- **scatter 是 gather 的反方向。**
- **all 表示目的地是所有 ranks。**



### 1.3 硬件互联与 NCCL

多 GPU 性能高度依赖互联拓扑。

#### 1.3.1 从家用机器到数据中心

经典家用机器里，GPU 往往通过 PCIe 连接。不同节点之间可能通过普通 Ethernet 通信，延迟更高、带宽更低。

数据中心训练集群通常更复杂：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1781603884558_image.png)

典型形态：

- 单节点：8 GPUs，通过 NVLink 连接到 NVSwitch。
- 多节点：节点之间通过 InfiniBand 或 RoCE 互联。
- 更大集群：多个 pod / rack 之间再通过更慢的网络层连接。

讲义给出的直觉数字：

- B200 NVLink 5.0 约 1.8 TB/s。
- B200 HBM 约 8 TB/s。
- InfiniBand 约 0.05 TB/s。

数值不重要，关键是比例：跨 GPU 比单 GPU HBM 慢，跨节点又比单节点互联慢。



#### 1.3.2 RDMA 与绕过 CPU

普通 Ethernet 通常要经过 CPU 和内核网络栈：

-   GPU/CPU memory -> kernel socket buffer -> TCP packet -> NIC ring buffer -> network

RDMA 允许一台机器直接读写另一台机器的内存，减少 CPU 参与。InfiniBand 支持 RDMA；RoCE 是在以太网上实现类似 RDMA 的方案。

训练系统里，能绕开 CPU 就少一次瓶颈。



#### 1.3.3 NCCL 的角色

NCCL 是 NVIDIA Collective Communication Library。它负责把 `all_reduce`、`all_gather` 等 collective operations 翻译成 GPU 间的高效通信。

NCCL 做的事包括：

- 检测拓扑：GPU 数量、NVLink、PCIe、交换机、节点间连接。
- 为 collective 选择通信路径和算法。
- 启动 GPU kernels 来搬运和规约数据。

在 PyTorch 中，GPU collective 通常通过 `backend="nccl"` 使用 NCCL。CPU collective 常用 `backend="gloo"`。





































