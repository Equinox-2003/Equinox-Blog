---
title: "Assignment1"
description: ""
date: 2026-05-20T20:45:30+08:00
lastmod: 2026-05-20T20:45:30+08:00
draft: true

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

记录一下CS336 assignment1 的实验部分

保证所有test都过了，但是一些bonus太吃设备了，就不做了。

作业代码：



## 一、Byte-Pair Encoding

### 1.1 Unicode 与 UTF-8

-   Unicode 把字符映射到 code point。
-   UTF-8 把 Unicode 字符编码成字节序列。
-   byte-level tokenizer 的好处是永远不会出现 OOV（out of vocabulary），因为任意文本都可以表示成 0–255 的字节序列。



但也有一个问题就是，byte-level的tokenizer encode出来太长了，所以第一个实验就是去实现**BPE（Byte-Pair Encoding）**。



### 1.2 BPE 训练

#### 1.2.1 BPE思想

BPE的思想很简单：

1.  刚开始的vocabulary 就是 256个单字节，然后可能会有一些人为定义的特殊token。
2.  然后对于给定文本，统计token词频以及 相邻pair 词频
3.  每次选出一个词频最高的pair出来，构造pair为一个新的token
4.  然后更新一些用于维护的表（pair的词频，token词频等）
5.  因为vocab的size会一直变大，所以当达到目标size的时候就结束
6.  特殊 token，如 <|endoftext|>，要作为 hard boundary，不能跨越它合并。



#### 1.2.2 Problem (train_bpe): BPE Tokenizer Training (15 points)

这一部分需要实现adapters.py 中的run_train_bpe函数，该函数就是读取指定路径的文本然后在文本上面训练BPE。

大概说一下我的做法：

1.  首先就是初始化vocab为256个单字节token
2.  然后把给定的special_tokens加入vocab
3.  然后用讲义给定的正则表达式对输入文本进行分词
4.  然后就是对文本统计词频以及pair词频
5.  然后就该训练了，因为每次取最高频，我用了懒删除堆来实现
6.  然后每次从懒删除堆中取出一个best pair，对原来的一些表的更新就写的比较暴力了，应该可以多维护一些信息进一步优化这里我太困了就不做了



```python
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # validation
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")

    # -------------------------
    # 1. Initialize vocab
    # -------------------------
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # add special tokens to vocab
    cnt_id = 256
    token_set = set(vocab.values())
    for s in special_tokens:
        if len(vocab) >= vocab_size: break
        s2bytes = s.encode("utf-8")
        if s2bytes not in token_set:
            token_set.add(s2bytes)
            vocab[cnt_id] = s2bytes
            cnt_id += 1

    if len(vocab) >= vocab_size:
        return vocab, merges

    # -------------------------
    # 2. Load corpus
    # -------------------------
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read() # the whole file
    except FileNotFoundError:
        text = ""

    # -------------------------
    # 3. Pretokenization
    # -------------------------
    chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)
    # re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_freq = Counter()
    for chunk in chunks:
        for word in regex.findall(PAT, chunk):
            word_bytes = word.encode("utf-8")
            bytes_lst = [bytes([x]) for x in word_bytes]  #e.g. ['h', 'e', 'l', 'l', 'o']
            token_freq[tuple(bytes_lst)] += 1

    # -------------------------
    # 4. Initialize pair_freq
    # -------------------------
    pair_freq: Counter[tuple[bytes, bytes]] = Counter()
    for token, freq in token_freq.items():
        for l, r in pairwise(token):
            pair_freq[(l, r)] += freq

    # -------------------------
    # 5. Max heap with lazy deletion
    # -------------------------
    class MaxHeapItem:
        def __init__(self, cnt: int, p1: bytes, p2: bytes):
            self.cnt = cnt
            self.p1 = p1
            self.p2 = p2

        def __lt__(self, other):
            # Python heapq 是小根堆，这里反过来实现大根堆
            # tie-breaking: larger pair wins lexicographically
            return (self.cnt, self.p1, self.p2) > (other.cnt, other.p1, other.p2)

        def __eq__(self, other):
            return (self.cnt, self.p1, self.p2) == (other.cnt, other.p1, other.p2)

        @property
        def pair(self) -> tuple[bytes, bytes]:
            return self.p1, self.p2

        def __repr__(self):
            return f"({self.cnt}, ({self.p1}, {self.p2}))"

    # (cnt, p1, p2)
    lzheap: list[MaxHeapItem] = []
    for (l, r), c in pair_freq.items():
        heapq.heappush(lzheap, MaxHeapItem(c, l, r))

    # -------------------------
    # Helper functions
    # -------------------------
    def contains_pair(
        seq: tuple[bytes, ...],
        target_pair: tuple[bytes, bytes],
    ) -> bool:
        for i in range(len(seq) - 1):
            if seq[i] == target_pair[0] and seq[i + 1] == target_pair[1]:
                return True
        return False

    # new seq after merge
    def merge_seq_once(
        seq: tuple[bytes, ...],
        target_pair: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        """
        Left-to-right non-overlapping merge.

        Example:
            seq = (A, A, A), target = (A, A)
            result = (AA, A), not (AA, AA)
        """
        merged = []
        i = 0

        while i < len(seq):
            if i < len(seq) - 1 \
                and seq[i] == target_pair[0] \
                and seq[i + 1] == target_pair[1]:
                merged.append(seq[i] + seq[i + 1])
                i += 2
            else:
                merged.append(seq[i])
                i += 1

        return tuple(merged)

    # upd pair Counter
    def add_pair_count(pair: tuple[bytes, bytes], amount: int):
        pair_freq[pair] += amount
    def sub_pair_count(pair: tuple[bytes, bytes], amount: int):
        new_count = pair_freq[pair] - amount
        if new_count <= 0:
            del pair_freq[pair]
        else:
            pair_freq[pair] = new_count

    # -------------------------
    # 6. Main BPE loop
    # -------------------------
    while len(vocab) < vocab_size:
        best_item = None

        while lzheap:
            item = heapq.heappop(lzheap)
            pair = item.pair
            # check
            if pair in pair_freq and pair_freq[pair] == item.cnt:
                best_item = item
                break

        if best_item is None:
            break

        cur_pair = best_item.pair

        # record merge
        merges.append(cur_pair)
        # add new mapping
        new_token_bytes = cur_pair[0] + cur_pair[1]
        vocab[cnt_id] = new_token_bytes
        cnt_id += 1

        # all seqs containing cur_pair
        affected_seqs = [
            seq for seq in token_freq.keys()
            if contains_pair(seq, cur_pair)
        ]

        updated_pairs: set[tuple[bytes, bytes]] = set()
        new_seq_freq_delta: Counter[tuple[bytes, ...]] = Counter()

        for old_seq in affected_seqs:
            freq = token_freq[old_seq]

            # remove old sequence pair contributions
            for old_pair in pairwise(old_seq):
                updated_pairs.add(old_pair)
                sub_pair_count(old_pair, freq)

            # remove old sequence from token_freq
            del token_freq[old_seq]

            # merge old sequence
            new_seq = merge_seq_once(old_seq, cur_pair)

            # defer adding new sequence to token_freq
            new_seq_freq_delta[new_seq] += freq

        # add all new token sequences
        for new_seq, freq in new_seq_freq_delta.items():
            token_freq[new_seq] += freq

            # add new sequence pair contributions
            for new_pair in pairwise(new_seq):
                updated_pairs.add(new_pair)
                add_pair_count(new_pair, freq)

        # Push updated true counts into heap
        for pair in updated_pairs:
            if pair in pair_freq:
                heapq.heappush(
                    lzheap,
                    MaxHeapItem(pair_freq[pair], pair[0], pair[1]),
                )

    return (vocab, merges)

```

**测试结果：**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779344335143_image.png)



### 1.3 BPE Tokenizer

#### 1.3.1 Encoding and Decoding

tokenizer做encoding和BPE的训练非常相似：

1.  Pre-tokenize
2.  Apply the merges

然后tokenizer要能处理 Special tokens

**Memory considerations**

然后因为文本可能会很大，我们有必要将文本拆成可以放进内存的chunk。



decoding就是把token ID改回原文本

然后对于不能得到正确Unicode bytes的id，要将其替换成 U+FFFD。

然后讲义还特地说明了一下需要把 bytes.decode 的参数errors置成 'replace'



#### 1.3.2 Problem (tokenizer): Implementing the tokenizer (15 points)

>   这个实验需要在Linux下跑，我这里连的WSL

实现一个 Tokenizer类，然后讲义里面有要实现的接口

具体实现如下：

```python
import json
import ast
import regex
from itertools import pairwise
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.cache = {}

        token_set = set(self.vocab.values())
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0

        # append user-provided special tokens if they are not already in vocab
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in token_set:
                self.vocab[next_id] = token_bytes
                token_set.add(token_bytes)
                next_id += 1

        self.bytes2id = {bts: id for id, bts in self.vocab.items()}
        self.merge_priority = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] = None,
    ):
        def bytes_to_unicode():
            bs = (
                list(range(ord("!"), ord("~") + 1))
                + list(range(ord("¡"), ord("¬") + 1))
                + list(range(ord("®"), ord("ÿ") + 1))
            )
            cs = bs[:]
            n = 0

            for b in range(256):
                if b not in bs:
                    bs.append(b)
                    cs.append(256 + n)
                    n += 1

            cs = [chr(c) for c in cs]
            return dict(zip(bs, cs))

        byte_encoder = bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        def gpt2_str_to_bytes(s: str):
            return bytes(byte_decoder[c] for c in s)

        def to_bytes(x):
            if isinstance(x, bytes):
                return x

            if isinstance(x, list):
                return bytes(x)

            if isinstance(x, int):
                return bytes([x])

            if isinstance(x, str):
                if x.startswith("b'") or x.startswith('b"'):
                    try:
                        y = ast.literal_eval(x)
                        if isinstance(y, bytes):
                            return y
                    except Exception:
                        pass

                try:
                    return gpt2_str_to_bytes(x)
                except Exception:
                    pass

                try:
                    return x.encode("latin-1")
                except UnicodeEncodeError:
                    return x.encode("utf-8")

            raise ValueError(f"Cannot convert {x} to bytes")

        # load vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {}

        if isinstance(raw_vocab, dict):
            for k, v in raw_vocab.items():
                # format: {"0": [0], "1": [1], ...}
                if isinstance(k, str) and k.isdigit():
                    vocab[int(k)] = to_bytes(v)

                # GPT-2 format: {"token": id}
                elif isinstance(v, int):
                    vocab[v] = to_bytes(k)

                else:
                    raise ValueError("Unsupported vocab format")

        elif isinstance(raw_vocab, list):
            # format: [[0, [0]], [1, [1]], ...]
            for item in raw_vocab:
                if isinstance(item, list) and len(item) == 2:
                    idx, token = item
                    vocab[int(idx)] = to_bytes(token)
                else:
                    raise ValueError("Unsupported vocab format")

        else:
            raise ValueError("Unsupported vocab format")

        # load merges
        merges: list[tuple[bytes, bytes]] = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # GPT-2 merges.txt 第一行通常是版本声明
            if line.startswith("#"):
                continue

            try:
                item = ast.literal_eval(line)
                if len(item) != 2:
                    raise ValueError("Unsupported merges format")
                merges.append((to_bytes(item[0]), to_bytes(item[1])))
                continue
            except Exception:
                pass

            parts = line.split()

            if len(parts) != 2:
                raise ValueError("Unsupported merges format")

            merges.append((to_bytes(parts[0]), to_bytes(parts[1])))

        return cls(vocab, merges, special_tokens)

    def _bpe_merge(self, words: bytes) -> list[bytes]:
        if words in self.cache:
            return self.cache[words]

        wordsbytes = [bytes([x]) for x in words]
        merge_priority = self.merge_priority

        while len(wordsbytes) > 1:
            good_pairs = set(
                (l, r) for l, r in pairwise(wordsbytes)
                if (l, r) in merge_priority
            )

            if not good_pairs:
                break

            best_pair = min(good_pairs, key=lambda x: merge_priority[x])

            # O(1) space implementation
            i = 0
            for x in wordsbytes:
                wordsbytes[i] = x
                i += 1

                if i > 1 \
                    and wordsbytes[i - 2] == best_pair[0] \
                    and wordsbytes[i - 1] == best_pair[1]:
                    wordsbytes[i - 2] += wordsbytes[i - 1]
                    i -= 1

            del wordsbytes[i:]

        self.cache[words] = wordsbytes
        return wordsbytes

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        special_tokens = self.special_tokens
        bytes2id = self.bytes2id

        if special_tokens:
            special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
            special_pattern = "|".join(map(regex.escape, special_tokens_sorted))
            chunks = regex.split(f"({special_pattern})", text)
        else:
            chunks = [text]

        ids = []

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in special_tokens:
                ids.append(bytes2id[chunk.encode("utf-8")])
                continue

            for word in regex.findall(PAT, chunk):
                if not word:
                    continue

                merged_word = self._bpe_merge(word.encode("utf-8"))

                for s in merged_word:
                    ids.append(bytes2id[s])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        all2bytes = b"".join(self.vocab[id] for id in ids)
        return all2bytes.decode("utf-8", errors="replace")

```

-   初始化函数就是存一下 bytes2id 以及 id2bytes的映射，merges
-   encode 就是分词，然后做合并，合并那里可以O(1)空间实现，也算是算法题基本功了
    -   合并那里我开了个cache做优化，实测不加也能过
-   decode 直接映射就行

**测试结果**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779360181096_image.png)

>   XFAIL 是作者预期不通过，正常。



## 二、Transformer Language Model Architecture

>   一些前置知识：
>
>   [注意力机制](https://equinox.wiki/post/machinelearning/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/)
>
>   [Transformer](https://equinox.wiki/post/machinelearning/transformer/)

这一部分的总体目标是从零实现一个 decoder-only Transformer 语言模型，也就是类似 GPT/LLaMA 这类自回归语言模型的核心架构。从高层结构到每个基础模块，最后把它们组装成完整的 Transformer LM。

讲义先定义了语言模型的输入输出：

-   **输入**：一批 token ID，形状为

$$
\text{batch\_size} \times \text{sequence\_length}
$$

-   **输出**：每个位置对下一个 token 的预测 logits / 概率分布，形状为

$$
\text{batch\_size} \times \text{sequence\_length} \times \text{vocab\_size}
$$

模型的大致流程是：

1.  **Token Embedding**：把离散 token ID 映射成连续向量；
2.  **多个 Transformer Block**：进行上下文建模；
3.  **Final RMSNorm**：最后归一化；
4.  **LM Head / Output Projection**：映射到词表大小，得到 next-token logits；
5.  训练时用这些 logits 计算交叉熵，推理时用最后一个位置的分布生成下一个 token。



### 2.1 Basic Building Blocks: Linear and Embedding Modules

因为这个课nn.Linear 和 nn.Embedding这些东西ban了，所以要我们自己手搓，好在nn.Module、ModuleList这些东西还能用。

#### 2.1.1 Implementing the linear module (1 points)

手写一个Linear Module，值得注意的是：

1.  权重初始化要用讲义指定的方式
2.  然后讲义还特别强调了下：•construct and store your parameter as$ W$ (not $W^T$)，也就是说我们前向传播应该写成 x @ W

**实现：**

`Linear.py`

```python
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.rand(in_features, out_features, device=device, dtype=dtype))
        # no bias
        # approximate initializations given in handout
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x @ self.W.T
    
```

`adapters.py`

```python
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp = Linear(in_features=d_in, out_features=d_out, device=device, dtype=torch.float32)
    mlp.W = nn.Parameter(weights)
    return mlp(in_features)
```

**测试结果：**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779438297646_image.png)



#### 2.1.2 Implement the embedding module (1 points)

embedding就是查表，然后初始化方式按讲义指定的来就好。

`Embedding.py`

```python
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # also row main vector
        self.embeddings = nn.Parameter(torch.rand(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = 1
        nn.init.trunc_normal_(self.embeddings, std=std, a=-3*std, b=3*std)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
    
```

`adapters.py`

```python
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb = Embedding(vocab_size, d_model, device=device, dtype=torch.float32)
    emb.embeddings = nn.Parameter(weights)
    return emb(token_ids)

```



**测试结果：**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779438848819_image.png)



### 2.2 Pre-Norm Transformer Block

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779440900101_image.png)

最经典的Transformer结构是在子模块后面做norm，但有很多工作发现，我们在子模块之前就做norm，有利于提升Transformer 的训练稳定性。

一个直觉上的解释就是在子模块之前做norm，那么残差连接的数据流可以包含一些不经过任何norm的信息流，比较干净。

pre-norm Transformer是现在很多语言模型的标准，如 GPT-3、LLaMA、PaLM等。

#### 2.2.1 Root Mean Square Layer Normalization

$$
RMSNorm(a_i) = \frac{a_i}{RMS(a)}g_i
$$

标准的layernorm是经典的 减去均值，然后除以标准差。

讲义让我们实现另一种norm的方式 RMSnorm，见上式。

其中，
$$
\text{RMS}(a) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} a_i^2 + \epsilon}
$$


即，不减均值， 然后仅根据均方根来缩放。好处是计算更简单，现代LLM如 LLaMA 使用RMSNorm。

下面就要手搓这个模块了，讲义特地提醒，为了避免数值溢出，计算前要把输入转成 `float32`，然后再转回原 dtype。

`RMSnorm.py`

```python
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1E-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_type = x.dtype
        x = x.to(torch.float32)
        den = (x**2).mean(dim=-1, keepdim=True)
        x /= torch.sqrt(den + self.eps)
        # 注意是hadamard乘积
        return (self.g * x).to(x_type)
```

`adapters.py`

```python
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    norm = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=torch.float32)
    norm.g = nn.Parameter(weights)
    return norm(in_features)

```



**测试结果：**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779442667767_image.png)



#### 2.2.2 Position-Wise Feed-Forward Network

Attention is All you Need 那篇论文的 Feed Forward network 是两个线性层中间夹了一个ReLU，原始架构中，内层的feed forward层的维度一般是输入 * 4。

然后现代的LLM对原架构做了两个改变：

1.  **用SwiGLU代替 ReLU**

    -   e.g. Llama 3 [A. Grattafiori et al., 2024] and Qwen 2.5 [A. Yang et al., 2024
    -   **SwiGLU 就是将SiLU 结合 GLU（Gated Linear Unit）**

    -   e.g. PaLM [A. Chowdhery et al., 2022] and LLaMA [H. Touvron et al., 2023].

2.  省略有时会在线性层中用到的bias

    -   PaLM [A. Chowdhery et al., 2022] and LLaMA [H. Touvron et al., 2023].



**SiLU的定义：**
$$
SiLU(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$
**GLU 是一种门控机制；定义为：一个经过 sigmoid 函数的线性变换，与另一个线性变换之间的逐元素乘积：**
$$
GLU(x, W_1, W_2) = \sigma(W_1x) \odot W_2x
$$
-   $\odot$ 是逐元素相乘
-   门控线性单元被认为可以：通过为梯度提供一条线性路径，同时保留非线性能力，从而减少深层架构中的梯度消失问题。



总之，原始 FFN 通常是：
$$
\text{FFN}(x) = W_2 \text{ReLU}(W_1x)
$$

本章使用：
$$
\text{FFN}(x) = SwiGLU(x, W_1, W_2, W_3) = W_2(\text{SiLU}(W_1x) \odot W_3x)
$$
其中：
$$
x\in \mathbb{R}^{d_{\text{model}}}
$$

$$
W_1,W_3\in \mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}
$$

$$
W_2\in \mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}
$$
通常情况下：
$$
d_{\text{ff}}=\frac{8}{3}d_{\text{model}}
$$
在具体实现中，为了提高硬件效率，可以将这个维度四舍五入到接近的 64 的倍数。

Shazeer 首先提出将 **SiLU / Swish** 激活函数与 GLU 结合起来，并通过实验表明，在语言建模任务上，SwiGLU 的表现优于 ReLU 和没有门控机制的 SiLU 等基线方法。

讲义里面提到了一些关于这些组件的启发式解释，并且相关论文也提供了更多支持性证据，但最好还是保持一种经验主义视角：Shazeer 论文中有一句现在很有名的话：

    “我们并没有解释为什么这些架构看起来有效；我们把它们的成功归因于……”



#### 2.2.3 Implement the position-wise feed-forward network (2 points)

实现方面，就是把公式封装成module，然后讲义说了可以用torch.sigmoid

`SwiGLUFFN.py`

```python
import torch
from torch import nn
from Linear import Linear

# 注意运算都是 Hadamard 乘积
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)
        
    def forward(self, x: torch.Tensor):
        w1x = self.W1(x)
        w3x = self.W3(x)
        return self.W2(self._SiLU(w1x) * w3x)

    def _SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)
```

`adapters.py`

```python
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
    ffn.W1.W = nn.Parameter(w1_weight)
    ffn.W2.W = nn.Parameter(w2_weight)
    ffn.W3.W = nn.Parameter(w3_weight)
    return ffn(in_features)
```

**测试结果：**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779452731064_image.png)



### 2.3 Relative Positional Embeddings

讲义介绍了一种位置编码的实现方法：**旋转位置嵌入（Rotary Position Embeddings）**，通常称为 **RoPE**。

对于位于 token 位置 $i$ 的某个 query token：
$$
q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d
$$
我们会对它应用一个成对旋转矩阵 $R_i$，得到：
$$
q'^{(i)} = R_i q^{(i)} = R_i W_q x^{(i)}
$$
这里，$R_i$ 会把 embedding 元素中的成对分量：
$$
q^{(i)}_{2k-1:2k}
$$
看作二维向量，并将其旋转一个角度：
$$
\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}
$$
























