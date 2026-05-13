---
title: "Coding Attention Mechanisms"
description: ""
date: 2026-05-04T12:21:35+08:00
lastmod: 2026-05-04T12:21:35+08:00
draft: false

categories:
  - LLMs-from-scratch
tags:
  - LLM

toc: true
math: true
mermaid: true
---

<!--more-->



## 零、前言

原书这章主要就是搓了下注意力模块，总体还是非常easy的

本章代码：[ch03](https://github.com/Equinox-2003/LLMs-from-scratch-Practice/tree/main/ch03)



## 一、Self Attention Mechanism

### 1.1 Simple Attention Mechanism

作者没有一上来就手撕多头注意力机制或者说子注意力机制，而是用了一个思想相同，但是实现上简化的一个简单版本的注意力机制。

作者之间用输入向量之间两两做点积来进行注意力分数的计算。

```python
import torch
inputs = torch.tensor(
 [[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)
```

```python
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
```

```
tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

两个向量越相似，那么dot越大，这个直觉是有可取之处的。

那么所有的注意力分数直接用矩阵相乘计算即可：

```python
attn_scores = inputs @ inputs.T
attn_scores
```

```
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
```

接着做注意力权重的计算，只需对dim=-1做softmax

```python
attn_weights = torch.softmax(attn_scores, dim=-1)
attn_weights
```

```
tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
```



### 1.2 attention mechanism with trainable weights

上面那个简化版注意力机制有个问题就是，没有trainable weights，无法学习。

所以，作者总算是引入了经典注意力机制实现——**scaled dot-product attention**。

>   缩放点积注意力在标准注意力机制的基础上，在softmax之前，对注意力分数除以了特征维的平方根。
>
>   因为大模型意味着大数据，大数据意味着大运算。做这样一个缩放可以保证梯度计算的稳定性。

我们需要引入三个可训练的权重矩阵：$W_q, W_k, W_v$。

三个矩阵分别把输入向量映射为：**query、key、value**。

```python
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
```

```python
torch.manual_seed(123)
W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```

```python
q_2 = x_2 @ W_q
k_2 = x_2 @ W_k
v_2 = x_2 @ W_v
q_2
```

```
tensor([0.4306, 1.4551])
```

那么可以用矩阵运算求出所有的key，value

```python
keys = inputs @ W_k
values = inputs @ W_v
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
```

```
keys.shape: torch.Size([6, 2])
values.shape: torch.Size([6, 2])
```

每个输入的注意力分数就是输入向量去乘keys

```python
attn_scores_2 = q_2 @ keys.T
attn_scores_2
```

```
tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
```

然后尝试封装成 SelfAttention_v1 类

```python
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Parameter(torch.rand(d_in, d_out))
        self.W_k = nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        ks = x @ self.W_k
        qs = x @ self.W_q
        vs = x @ self.W_v
        attn_scores = qs @ ks.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

```

```python
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
sa_v1(inputs)
```

```
tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)
```

 

注意到上面的W_q, W_k, W_v都是用 nn.Parameters初始化的，我们其实可以直接用 nn.Linear来初始化，

>   Linear 层有着更优的权重初始化方式，这有助于我们的训练的稳定性以及效率。

```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        ks = self.W_k(x)
        qs = self.W_q(x)
        vs = self.W_v(x)
        attn_scores = qs @ ks.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
```

```python
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v2(inputs)
```

```
tensor([[0.2762, 0.7584],
        [0.2761, 0.7638],
        [0.2762, 0.7637],
        [0.2780, 0.7609],
        [0.2785, 0.7603],
        [0.2772, 0.7618]], grad_fn=<MmBackward0>)
```



### 1.3 mask attention

**mask attention** 的思想就是限制我们的model在计算注意力机制的时候只看前面已有的输入，不让看未来的输入。

```python
qs = sa_v2.W_q(inputs)
ks = sa_v2.W_k(inputs)
attn_scores = qs @ ks.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
attn_weights
```

```
tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
        [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
        [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
```

一个naive的想法是，创建一个下三角01掩码矩阵，和attn_weights做hadamard乘积，就可以mask掉future data

```python
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
```

```
tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])
```

```python
masked_simple = attn_weights * mask_simple # hadamard mul
masked_simple
```

```
tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<MulBackward0>)
```

我们还要进行的一个步骤就是归一化，让每个权重向量的和为1：

```python
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
masked_simple_norm
```

```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<DivBackward0>)
```

>   Q：这样做会有Information leakage吗
>
>   A：不会，因为我们重新做了归一化。
>
>   softmax计算的时候虽然分母涉及到了所有的数据，但是我们重新归一化相当于消去了那个分母。

在包装成类之前，还有一个可优化的地方便是，我们是先对 attention weights做hadamard乘积，然后重归一化。

但是我们可以利用softmax函数的特性简化这一步骤：

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
masked
```

```
tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
        [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
        [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
        [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
        [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
       grad_fn=<MaskedFillBackward0>)
```

```python
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
attn_weights
```

```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
```

**也就是在softmax之前，把需要mask掉的地方加上-inf，这样做指数幂就会变成0，softmax计算时就不会涉及到future data，并且还完成了归一化。**

我们还可以加入dropout来降低过拟合风险：

```python
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)	
example = torch.ones(6, 6)
dropout(example)
```

```
tensor([[2., 2., 0., 2., 2., 0.],
        [0., 0., 0., 2., 0., 2.],
        [2., 2., 2., 2., 0., 2.],
        [0., 2., 2., 0., 0., 2.],
        [0., 2., 0., 2., 0., 2.],
        [0., 2., 2., 2., 2., 0.]])
```

```python
torch.manual_seed(123)
dropout(attn_weights)
```

```
tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.7599, 0.6194, 0.6206, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.4921, 0.4925, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.3966, 0.0000, 0.3775, 0.0000, 0.0000],
        [0.0000, 0.3327, 0.3331, 0.3084, 0.3331, 0.0000]],
       grad_fn=<MulBackward0>)
```



然后考虑封装成类：

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        ks = self.W_k(x)
        qs = self.W_q(x)
        vs = self.W_v(x)

        attn_scores = qs @ ks.transpose(1, 2)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / ks.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ vs
        return context_vec

```

>   Q：什么是register_buffer
>
>   A：register_buffer 是 PyTorch 里 nn.Module 的一个方法，用来给模型注册一个 **不参与训练更新**、**但属于模型状态的一部分** 的 Tensor。
>
>   优点：会跟随model.to(device)，即当model迁移到另一个设备上时， register_buffer 创建的属性会跟着自动迁移。



```python
batch = torch.stack((inputs, inputs), dim=0)
batch.shape
```

```
torch.Size([2, 6, 3])
```

```python
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
```

```
context_vecs.shape: torch.Size([2, 6, 2])
```



### 1.4 Extending single-head attention to multi-head attention

单个的注意力模块可能只关注到某个维度的信息，而多头注意力，可以理解成多次做self attention，来捕捉多层次、多样化的特征（丰富上下文理解），增强模型的表达和泛化能力。

并且多头可以并行高效计算。

一个naive的实现就是直接开多个self attention 的module：

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
        dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
        [CausalAttention(
        d_in, d_out, context_length, dropout, qkv_bias
        )
        for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

```python
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, 2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

```
tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]],

        [[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
context_vecs.shape: torch.Size([2, 6, 4])
```



### 1.5 Implementing multi-head attention with weight splits

上面那个多头注意力的实现，实例化了多个自注意力模块，性能开销较大，下面整合为一个MultiHeadAttention类：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0, \
               "d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # Reduces the projection dim to match the desired output dim
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Uses a Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # Tensor shape: (b, num_tokens, d_out)
        ks = self.W_k(x)
        qs = self.W_q(x)
        vs = self.W_v(x)
        #We implicitly split the matrix by adding a num_heads dimension. 
        #Then we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
        ks = ks.view(b, num_tokens, self.num_heads, self.head_dim)
        vs = vs.view(b, num_tokens, self.num_heads, self.head_dim)
        qs = qs.view(b, num_tokens, self.num_heads, self.head_dim)
        # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        ks = ks.transpose(1, 2)
        qs = qs.transpose(1, 2)
        vs = vs.transpose(1, 2)
        # Computes dot product for each head
        attn_scores = qs @ ks.transpose(2, 3)
        # Masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Uses the mask to fill attention scores
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / ks.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ vs).transpose(1, 2)
        # Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        # Adds an optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec
```

-   这里的 d_out = num_heads * head_dim
-   然后我们对于计算出来的context_vec又经过了一次线性层投影，这是为了combine多头注意力























