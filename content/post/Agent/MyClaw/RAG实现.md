---
title: "RAG实现"
description: "给模型装一个外挂"
date: 2026-06-27T18:33:55+08:00
lastmod: 2026-06-27T18:33:55+08:00
draft: true

categories:
  - MyClaw
tags:
  - LLM
  - Agent

toc: true
math: true
mermaid: true
---

<!--more-->



## 一、RAG

**RAG**，即 **Retrieval-Augmented Generation**，检索增强生成。

**为什么我们需要 RAG？**

1.  **解决“幻觉”问题：** 大模型遇到不懂的问题时，倾向于一本正经地胡说八道。RAG 给它提供了事实依据，**限制了它的发散**。

2.  **突破“知识更新”瓶颈：** 大模型的训练数据是有截止日期的（比如截止到2023年）。**重新训练模型成本极高，但有了 RAG，你只需要往数据库里扔最新的文档，模型就能回答最新信息。**

3.  **解决“数据隐私”问题：** 企业的内部数据（财报、员工手册、代码库）不能喂给公共大模型训练。通过 RAG，数据留在本地数据库，模型只是“阅读”检索出来的部分片段，保证了安全。

    >   当然，本质上还是有风险的。



### 1.1 工作机制

RAG 的工作流程分为两个完全独立的阶段：**数据准备阶段（Indexing）** 和 **检索生成阶段（Querying）**。

**阶段一：数据准备**

1.  **加载文档（Document Loading）：** 把各种格式的数据（PDF、Word、网页、TXT）读取进来。
2.  **文本切块（Chunking）：** 大模型一次读不了几十万字（上下文窗口限制），所以要把长文档切成一小块一小块的段落（Chunk），比如每 500 字一块。
3.  **向量化（Embedding）：**Embedding 模型把每一块文本转化成一串长长的数字（向量）。**语义越相近的文本，它们在多维空间中的距离就越近**。
4.  **存储（Vector Database）：** 把这些“文本块”和对应的“向量”存入向量数据库（如 Milvus, Pinecone, Chroma 等）。



**阶段二：检索与生成**

1.  **用户提问（Query）：** 用户问：“公司带薪年假有几天？”
2.  **问题向量化：** 把用户的这个问题，用同样的 Embedding 模型转化成向量。
3.  **相似度检索（Retrieval）：** 拿着“问题的向量”，去向量数据库里对比，找出距离最近（语义最相似）的 Top-K 个文本块（比如找到了员工手册里的年假规定片段）。
4.  **重排（Rerank）**：**向量相似不代表相关**，我们要的信息可能不是Top1的那个。所以还需要**Rerank模型**对召回的数据进行打分，进一步缩小范围。
5.  **组装提示词（Augmented）：** 把【用户的问题】和【检索出来的文本块】拼接在一起，写进 Prompt 里。
    -   *Prompt 示例：“请根据以下参考资料回答问题。参考资料：[检索出的年假规定片段]。用户问题：公司带薪年假有几天？”*
6.  **模型生成（Generation）：** 大模型阅读这个组装好的 Prompt，输出最终的准确答案。



### 1.2 简易实现

#### 1.1 Chunk

这里直接把准备好的文档按段落切分一下。

```python
from typing import List

def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r', encoding='utf-8') as file:
        content = file.read()

    return [chunk for chunk in content.split("\n\n")]

chunks = split_into_chunks("doc.md")

for i, chunk in enumerate(chunks):
    print(f"[{i}] {chunk}\n")
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1782656291822_image.png)



#### 1.2 Embedding

从 hugging-face 下载一个轻量的 中文 的 Embedding 模型，vocab_size 是768

```python
import os
from pathlib import Path

# HuggingFace downloads are often slow or unstable from China. Set this before loading models.
# You can override it before running the notebook, for example:
# $env:HF_ENDPOINT="https://huggingface.co"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str((Path.cwd() / "hf_cache").resolve()))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str((Path.cwd() / "hf_cache" / "sentence-transformers").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = os.environ.get("HF_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")

try:
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
        device="cpu",
    )
except Exception as exc:
    raise RuntimeError(
        "Failed to load the embedding model. If this is a HuggingFace network issue, "
        "run this cell again after setting HF_ENDPOINT=https://hf-mirror.com, or pre-download "
        "the model with: uv run hf download shibing624/text2vec-base-chinese"
    ) from exc

def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()

embedding = embed_chunk("原神启动")
print(EMBEDDING_MODEL_NAME)
print(len(embedding))
print(embedding[:8])
```

简单示例：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1782657545951_image.png)



然后暂存一下。

```python
embeddings = [embed_chunk(chunk) for chunk in chunks]
```



#### 1.3 Vector-DB

>   很棒的vector-db的视频：[【上集】向量数据库技术鉴赏](https://www.bilibili.com/video/BV11a4y1c7SW/?spm_id_from=333.337.search-card.all.click&vd_source=a7ce6b38365a0cb2ad96f0668de0bc51)

我们这里使用 **chroma** 来存储我们的 embedding。

```python
import chromadb

chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )

save_embeddings(chunks, embeddings)
```



#### 1.4 retrieve

我们需要对用户给定的输入，去检索 top-k 相似的embedding。

仍然是调库。

```python
def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]

query = "哆啦A梦使用的3个秘密道具分别是什么？"
retrieved_chunks = retrieve(query, 5)

for i, chunk in enumerate(retrieved_chunks):
    print(f"[{i}] {chunk}\n")
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1782658568589_image.png)



#### 1.5 rerank

重排不是重新从全库检索，而是在已经召回的 Top-k 结果里重新排序。

而`CrossEncoder` 是**一种专门用来判断“一对文本有多相关”的模型**。

它的输入不是单独一个句子，而是一对：(query, chunk)

例如：

```text
(
  "哆啦A梦使用的3个秘密道具分别是什么？",
  "三件秘密道具分别是复制斗篷、时间停止手表、精神与时光屋便携版。"
)
```

模型会直接输出一个相关性分数，比如：

```text
0.94
```



这里同样用huggingface上的小模型来做：

```python
import warnings
from sentence_transformers import CrossEncoder

RERANK_MODEL_NAME = os.environ.get("HF_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
cross_encoder = None

def get_cross_encoder():
    global cross_encoder
    if cross_encoder is None:
        cross_encoder = CrossEncoder(
            RERANK_MODEL_NAME,
            cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
            device="cpu",
        )
    return cross_encoder

def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    try:
        model = get_cross_encoder()
    except Exception as exc:
        warnings.warn(
            "Failed to load the rerank model, so this tutorial will skip reranking. "
            "The retrieval part still works. If this is a HuggingFace network issue, "
            "set HF_ENDPOINT=https://hf-mirror.com and rerun this cell. "
            f"Original error: {exc}"
        )
        return retrieved_chunks[:top_k]

    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = model.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]

reranked_chunks = rerank(query, retrieved_chunks, 3)

for i, chunk in enumerate(reranked_chunks):
    print(f"[{i}] {chunk}\n")
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1782659283623_image.png)

我们发现，top1的正是我们要的结果。



#### 1.6 generate

然后就可以让扔给模型生成了

```python
import os

from core.llm import call_llm_simple

def generate(query: str, chunks: List[str]) -> str:
    context = "\n\n".join(chunks)
    prompt = (
        "你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。\n\n"
        f"用户问题: {query}\n\n"
        "相关片段:\n"
        f"{context}\n\n"
        "请基于上述内容作答，不要编造信息。"
    )

    print(f"{prompt}\n\n---\n")
    return call_llm_simple(prompt)


answer = generate(query, reranked_chunks)
print(answer)

```



```markdown
你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: 哆啦A梦使用的3个秘密道具分别是什么？

相关片段:
三件秘密道具分别是：可以临时赋予超级战力的“复制斗篷”，能暂停时间五秒的“时间停止手表”，以及可在一分钟中完成一年修行的“精神与时光屋便携版”。大雄被推进精神屋内，在其中接受密集的训练，虽然只有几分钟现实时间，他却经历了整整一年的苦修。刚开始他依旧软弱，想放弃、想逃跑，但当他想起静香、父母，还有哆啦A梦那坚定的眼神时，他终于咬牙坚持了下来。出来之后，他的身体与精神都焕然一新，眼神中多了一份成熟与自信。

最终战在黑暗赛亚人的空中要塞前爆发，特兰克斯率先出击，释放全力与敌人正面对决。哆啦A梦则用任意门和道具支援，从各个方向制造混乱，尽量压制敌人的时空能力。但黑暗赛亚人太过强大，仅凭特兰克斯一人根本无法压制，更别说击败。就在特兰克斯即将被击倒之际，大雄披上复制斗篷、冲破恐惧从高空跃下。他的拳头燃烧着金色光焰，目标直指敌人心脏。

战后，未来世界开始恢复，植物重新生长，人类重建家园。特兰克斯告别时紧紧握住大雄的手，说：“你是我见过最特别的战士。”哆啦A梦也为大雄感到骄傲，说他终于真正成长了一次。三人站在山丘上，看着远方重新明亮的地平线，心中感受到从未有过的安宁。随后，哆啦A梦与大雄乘坐时光机返回了属于他们的那个年代，一切仿佛又恢复平静。

请基于上述内容作答，不要编造信息。

---

根据提供的片段，哆啦A梦使用的三件秘密道具分别是：

- **复制斗篷**（可以临时赋予超级战力）
- **时间停止手表**（能暂停时间五秒）
- **精神与时光屋便携版**（可在一分钟内完成一年的修行）
```







































