---
title: "BLIP2泛读"
description: ""
date: 2026-06-02T14:10:54+08:00
lastmod: 2026-06-02T14:10:54+08:00
draft: false

categories:
  - paper-reading
tags:
  - multimodality
  - LLM
  - NLP
  - CV

toc: true
math: true
mermaid: true
---

<!--more-->



## 零、写在前面

BLIP-2 认为之前的VLP工作，把下游任务SOTA刷的越来越高，但是每次都要从零训练，而且成本也越来越高。所以通过引入QFormer，只需要训练QFormer，就可以把现成的视觉模型和语言模型拼在一起协同工作。

QFormer把图像压缩成几个关键的可学习的query，然后把这些query投影到语义空间，拼接在prompt前面喂给llm，从而引导语言模型看图说话。

这几天看了一车多模态的论文，本来要读 Qwen-VL、LLaVA、MiniGPT的，但发现前面这些多模态工作还真挺有必要看一下的。这个BLIP-2 证明了在 frozen LLM 前拼接 soft visual prompts 后，LLM 原本的 instruction following 能力可以被视觉条件激活。它还不是完整的 visual instruction tuning 系统，但已经具备早期视觉对话能力。



## 一、标题

>   BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

标题中的关键词有四个：**BLIP-2**、**Bootstrapping**、**Frozen Image Encoders**、**Large Language Models**。

BLIP-2 延续了 BLIP 的命名，但“Bootstrapping”的含义发生了变化。BLIP 主要是通过 CapFilt 改造 noisy web captions，是一种 **data bootstrapping**。



BLIP-2 则强调从已有的 frozen unimodal models 出发，是一种 **model bootstrapping**。它不是从头训练一个端到端视觉语言大模型，而是利用视觉社区和 NLP 社区已经训练好的强模型，再用一个轻量模块把两者连接起来。

标题中的 “Frozen Image Encoders and Large Language Models” 直接点出论文的系统设计：视觉编码器冻结，语言模型冻结，中间训练一个 Querying Transformer，也就是 Q-Former。这个标题其实已经说明了论文最核心的问题：如果两端大模型都不更新，怎样才能完成有效的 vision-language alignment？



>   论文团队依旧是 Salesforce Research，太强了



## 二、摘要

摘要首先指出，vision-and-language pre-training 的成本越来越高，因为很多 SOTA 方法需要端到端训练大规模模型和大规模数据。BLIP-2 针对这一问题提出一种通用且高效的预训练策略：从现成的 frozen image encoders 和 frozen LLMs 中 bootstrapping 出视觉语言能力。

论文的关键模块是 **Querying Transformer (Q-Former)**。它是一个轻量桥接模块，用来弥合视觉模态和语言模态之间的 gap。Q-Former 经过两阶段预训练：

1. **第一阶段：** 从 frozen image encoder 中 bootstrap vision-language representation learning。
2. **第二阶段：** 从 frozen language model 中 bootstrap vision-to-language generative learning。

摘要强调，BLIP-2 在多种 vision-language tasks 上取得 SOTA，同时可训练参数显著少于已有方法。例如，BLIP-2 在 zero-shot VQAv2 上超过 Flamingo80B 8.7%，但可训练参数少 54 倍。摘要还指出，BLIP-2 展示了 zero-shot image-to-text generation 的 emerging capabilities，并且可以跟随自然语言指令。



## 三、引言

引言从 VLP 的快速发展写起：CLIP、ALBEF、BLIP、OFA、Flamingo、BEIT-3 等模型不断推动下游任务 SOTA，**但代价是越来越高的预训练成本**。许多方法需要端到端训练大模型和大规模图文数据，这在算力和数据上都很昂贵。

BLIP-2 的基本观察是：vision-language research 位于视觉和语言两个领域的交叉点，**因此应该能够利用两个领域已经训练好的 unimodal models**。	

- 预训练视觉模型提供高质量 visual representation；
- 预训练语言模型，尤其是 LLM，提供强语言生成和 zero-shot transfer 能力；
- **如果冻结这些 unimodal models，可以降低算力成本，并减少 catastrophic forgetting。**

但冻结模型也带来一个关键难题：**LLM 在单模态预训练时没有见过图像，所以如果 LLM 不更新，视觉特征必须被转换成它能理解的形式。** 论文指出，已有方法如 Frozen 和 Flamingo 主要依赖 image-to-text generation loss，但作者认为单靠这个目标不足以弥合 modality gap。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1780386142451_image.png)

为此，BLIP-2 提出 Q-Former。它用一组 learnable query vectors 从 frozen image encoder 中抽取对文本最有用的视觉特征，并作为 frozen image encoder 与 frozen LLM 之间的信息瓶颈。两阶段训练的设计对应两个不同目标：

1. **第一阶段让 Q-Former 学会从 frozen image encoder 中抽取与文本最相关的 visual representation。**
2. **第二阶段把 Q-Former 输出接到 frozen LLM，让这些视觉表示变成 LLM 可解释的 soft visual prompts。**

引言列出的主要贡献可以整理为三点：

1. **有效利用 frozen image encoder 与 frozen LLM。** 通过 Q-Former 和两阶段预训练弥合模态差距，在 VQA、captioning、retrieval 等任务上取得强结果。
2. **支持 instructed zero-shot image-to-text generation。** 依托 OPT、FlanT5 等 LLM，BLIP-2 可以根据自然语言 prompt 控制图像到文本生成，出现 visual knowledge reasoning、visual conversation 等能力。
3. **计算效率高。** 使用 frozen unimodal models 和轻量 Q-Former，BLIP-2 比现有 SOTA 需要少得多的可训练参数，例如在 zero-shot VQAv2 上超过 Flamingo80B，同时可训练参数少 54 倍。



## 四、Related work

相关工作主要分成两类。

### 4.1 End-to-end Vision-Language Pre-training

第一类是端到端 VLP。论文按架构归纳了几种路线：

1. **Dual-encoder architecture**，例如 CLIP、ALIGN；
2. **Fusion-encoder architecture**，例如 LXMERT、ALBEF；
3. **Encoder-decoder architecture**，例如 VL-T5、SimVLM、PaLI；
4. **Unified transformer architecture**，例如 BLIP、BEIT-3。

这些方法的预训练目标逐渐收敛到几类经典目标：image-text contrastive learning、image-text matching、masked language modeling 或 language modeling。

BLIP-2 对这类方法的批评是：**它们通常需要大规模端到端预训练，随着模型规模增长，计算成本极高；同时端到端模型不够灵活，难以及时吸收视觉和语言社区不断出现的新预训练模型。**



### 4.2 Modular Vision-Language Pre-training

第二类是模块化 VLP，即利用 off-the-shelf pretrained models 并在 VLP 中冻结它们。

- 有些方法冻结 image encoder，例如早期使用 frozen object detector 的 UNITER、OSCAR、VinVL，以及使用 frozen image encoder 做 CLIP pretraining 的 LiT。
- 有些方法冻结 language model，用 LLM 的知识做 vision-to-language generation，例如 Frozen、Flamingo、VisualGPT、MAPL、PnP-VQA、Img2Prompt 等。

这类方法的核心挑战是：**如何把视觉特征对齐到文本空间**。Frozen 直接把 image encoder 输出当作 soft prompts；Flamingo 在 LLM 中插入 cross-attention layers，让文本 token 可以访问视觉特征，并在十亿级图文数据上训练新增层。

**BLIP-2 与它们不同：它同时利用 frozen image encoder 和 frozen LLM，用 Q-Former 作为中间的 trainable bottleneck，而不是让 LLM 大量新增 cross-attention 层直接处理视觉序列。**



## 五、Method

BLIP-2 的方法可以概括为：**用 Q-Former 在 frozen image encoder 和 frozen LLM 之间建立一个可训练、可压缩、可对齐的信息瓶颈，并通过两阶段训练分别解决视觉语言表示学习和视觉到语言生成学习。**

### 5.1 Q-Former 架构

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1780389881654_image.png)

Q-Former 连接 frozen image encoder 与 frozen LLM，负责从视觉编码器输出中抽取固定数量的视觉特征。其设计要点包括：

1. **两套 transformer 子模块共享 self-attention layers。** 论文把它描述为 image transformer 和 text transformer：前者与 frozen image encoder 交互，后者可以作为 text encoder 或 text decoder。
2. **learnable query embeddings。** Q-Former 创建固定数量的可学习 query，作为 image transformer 的输入。
3. **query 与 image feature 通过 cross-attention 交互。** Cross-attention layers 每隔一个 transformer block 插入一次。
4. **query 之间通过 self-attention 交互。** Query 还可以根据不同预训练任务与 text tokens 通过 self-attention 交互。
5. **不同任务使用不同 attention mask。** 通过 mask 控制 query-text interaction，使同一个 Q-Former 可以服务 ITC、ITG、ITM 等目标。
6. **初始化来自 BERT-base。** Q-Former 使用 BERT_base 权重初始化，cross-attention layers 随机初始化。

参数上，Q-Former 约 188M parameters。实验中使用 32 个 query，每个 query 维度为 768。输出 query representation 记为 Z，尺寸为 32 x 768。相比 frozen image encoder 的输出，例如 ViT-L/14 的 257 x 1024，Q-Former 输出明显更小，因此它是一个显式 information bottleneck。

这个 bottleneck 的意义是：LLM 不需要直接处理庞大的视觉 patch 序列，而只接收 Q-Former 筛选后的少量视觉语义 token。Q-Former 的训练目标会迫使这些 query 抽取与文本最相关的信息，而不是保留所有图像细节。



### 5.2 第一阶段：从 Frozen Image Encoder 中学习视觉语言表示

第一阶段只连接 Q-Former 和 frozen image encoder，不接 LLM。目标是让 Q-Former 的 queries 学会抽取与文本最相关的视觉表示。论文借鉴 BLIP，联合优化三个目标：**ITC、ITG 和 ITM**。

值得注意的是，两个transformer 共用了self-attention 层，那么如果不加以限制直接去计算ITC、ITG、ITM 是不合理的，因为发生了信息泄露。

所以在 架构图的右侧，作者专门列出了三个任务所采用的mask矩阵。

>   注意 query 和 text 是拼接后一起送入模型的。



### 5.3 第二阶段：从 Frozen LLM 中学习视觉到语言生成

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1780392174927_image.png)

第二阶段把带有 frozen image encoder 的 Q-Former 接到 frozen LLM。具体做法是：

1. Q-Former 输出 **query embeddings Z**；

    >   注意是把query 的输出拿出来了

2. 用一个 fully-connected layer 把 Z 线性投影到 LLM text embedding 的维度；

3. 把投影后的 query embeddings prepend 到 input text embeddings 前面；

4. 这些投影后的 query embeddings 作为 **soft visual prompts**，让 LLM 在视觉表示条件下生成文本。

**由于 Q-Former 已经在第一阶段学会抽取 language-informative visual representation，第二阶段 LLM 不需要从原始视觉序列中学习对齐，只需要解释这些 soft prompts**。这降低了 LLM 学习 vision-language alignment 的负担，也缓解 catastrophic forgetting。

论文实验了两类 LLM：

- **Decoder-based LLMs：** 使用 OPT family，训练目标是 language modeling loss，让 frozen LLM 在视觉表示条件下生成文本。
- **Encoder-decoder-based LLMs：** 使用 FlanT5 family，训练目标是 prefix language modeling loss，把文本拆成 prefix 和 suffix。Prefix 与视觉表示一起输入 encoder，suffix 作为 decoder 生成目标。



### 5.4 预训练

BLIP-2 使用与 BLIP 相同的预训练数据，总计 129M images，包括 COCO、Visual Genome、CC3M、CC12M、SBU，以及 LAION400M 中的 115M images。

**论文继续采用 BLIP 的 CapFilt 方法为 web images 生成 synthetic captions**。具体做法是：

1. 用 BLIP_large captioning model 为每张 web image 生成 10 个 captions；
2. 将 synthetic captions 与原始 web caption 一起，用 CLIP ViT-L/14 计算 image-text similarity 进行排序；
3. 每张图保留 top-two captions；
4. 每个 pre-training step 随机采样其中一个 caption。

模型方面，frozen image encoder 实验了两种视觉模型：CLIP 的 ViT-L/14 和 EVA-CLIP 的 ViT-g/14。论文移除 ViT 最后一层，使用倒数第二层输出特征，因为这样效果略好。Frozen LLM 方面，decoder-only 使用 OPT，encoder-decoder 使用 FlanT5。

训练设置包括：

- 第一阶段训练 250k steps；
- 第二阶段训练 80k steps；
- 第一阶段 batch size 为 ViT-L 2320 / ViT-g 1680；
- 第二阶段 batch size 为 OPT 1920 / FlanT5 1520；
- frozen ViT 和 LLM 参数转成 FP16，FlanT5 使用 BFloat16；
- 最大模型 ViT-g + FlanT5-XXL 在单台 16 x A100 40G 机器上，第一阶段少于 6 天，第二阶段少于 3 天。



## 六、实验

![image-20260602175049651](D:\TyporaPics\image-20260602175049651.png)

>   论文贴了一车图出来（这些图都是instructed zero-shot image-to-text generation），效果拔群，匪夷所思……



### 6.1 Zero-shot 任务总览

论文的 zero-shot 总览表显示，BLIP-2 在多个任务上取得强结果，同时可训练参数很少。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1780394239456_image.png)

这个表服务于论文的核心叙事：BLIP-2 不是单纯更大，而是用更少 trainable parameters 更高效地利用 frozen models。



### 6.2 Instructed Zero-shot Image-to-Text Generation

**BLIP-2 可以把 text prompt 直接追加到 visual prompt 后，让 LLM 根据图像和自然语言指令生成文本**。论文展示了 visual knowledge reasoning、visual commonsense reasoning、visual conversation、personalized image-to-text generation 等示例。

这部分与后续 MiniGPT-4、LLaVA、Qwen-VL-Chat 有直接关系。BLIP-2 证明了在 frozen LLM 前拼接 soft visual prompts 后，LLM 原本的 instruction following 能力可以被视觉条件激活。它还不是完整的 visual instruction tuning 系统，但已经具备早期视觉对话能力。



### 6.3 Zero-shot VQA

在 zero-shot VQA 上，BLIP-2 取得了强结果。最强配置 ViT-g + FlanT5-XXL 的结果为：

| 模型                    | Trainable Params | Total Params | VQAv2 test-dev | OK-VQA test | GQA test-dev |
| ----------------------- | ---------------: | -----------: | -------------: | ----------: | -----------: |
| Flamingo80B             |            10.2B |          80B |           56.3 |        50.6 |            - |
| BLIP-2 ViT-g FlanT5-XXL |             108M |        12.1B |           65.0 |        45.9 |         44.7 |

BLIP-2 在 VQAv2 上比 Flamingo80B 高 8.7%，同时 trainable parameters 少 54 倍。OK-VQA 上 Flamingo80B 更强，论文认为原因可能是 OK-VQA 更依赖 open-world knowledge，而 Flamingo80B 使用 70B Chinchilla，语言知识更强；BLIP-2 的 FlanT5-XXL 为 11B，知识储备相对较少。

表中还支持一个重要观察：**更强 image encoder 和更强 LLM 都会提升性能。** 例如 ViT-g 优于 ViT-L；同一 LLM family 内更大模型更好；instruction-tuned FlanT5 在 VQA 上优于 unsupervised-trained OPT。这支持 BLIP-2 的 generic method 叙事：它可以随着视觉和语言 foundation models 的进步而自然受益。



### 6.4 第一阶段 Representation Learning 的作用

论文专门分析了 first-stage representation learning 对 second-stage generative learning 的作用。没有第一阶段时，Q-Former 只能依赖 vision-to-language generative learning 来弥合 modality gap，类似 Flamingo 的 Perceiver Resampler。结果显示，缺少第一阶段会显著降低 zero-shot VQA 性能；尤其 OPT 会出现 catastrophic forgetting，训练推进时性能大幅下降。

这个实验是 BLIP-2 最关键的因果证据之一。它说明 Q-Former 不是只需要接到 LLM 上做 caption generation 就够了，而是必须先通过 ITC/ITG/ITM 学会抽取与文本相关的视觉表示。换句话说，第一阶段让 Q-Former 成为一个更好的视觉信息瓶颈，第二阶段才更容易让 LLM 理解这些视觉 soft prompts。



### 6.5 Image Captioning

BLIP-2 在 COCO 上 fine-tune captioning，并在 COCO test set 与 NoCaps validation set 上评测。Fine-tuning 时：

- prompt 使用 `a photo of`；
- LLM 保持 frozen；
- 更新 Q-Former 和 image encoder；
- 使用 language modeling loss 生成 caption。

NoCaps zero-shot 结果显示，BLIP-2 ViT-g FlanT5-XL 达到 overall CIDEr 121.6 / SPICE 15.8，超过 BLIP 的 113.2 / 14.8，也优于 SimVLM 的 112.2。NoCaps 包含 out-of-domain images，因此这个结果说明 BLIP-2 不只是拟合 COCO caption 风格，也具备较强域外泛化能力。

COCO fine-tuned 上，BLIP-2 ViT-g OPT-2.7B 达到 CIDEr 145.8，接近或超过 OFA、SimVLM、Flamingo 等模型。这个结果说明 frozen LLM 配合 Q-Former 仍然能支持高质量 captioning。



### 6.6 Visual Question Answering Fine-tuning

VQA fine-tuning 时，BLIP-2 让 LLM 接收 Q-Former 输出和问题文本，并生成答案。为了让视觉特征更针对问题，论文还把 question tokens 输入 Q-Former，使 question 与 queries 通过 self-attention 交互，从而引导 Q-Former 的 cross-attention 更关注相关图像区域。

在 VQAv2 fine-tuned 上，BLIP-2 ViT-g OPT-6.7B 达到 82.19 test-dev / 82.30 test-std，在 open-ended generation models 中达到 SOTA，超过 ALBEF、BLIP、OFA、Flamingo80B 等。但 closed-ended classification models 中 BEIT-3 更高，达到 84.19 / 84.03。

这说明 BLIP-2 在生成式 VQA 上很强，但不同任务形式之间仍需区分。Open-ended generation 更符合 LLM 生成范式；closed-ended classification 可能在特定 benchmark 上更有优势。



### 6.7 Image-Text Retrieval

对于 image-text retrieval，论文不使用 LLM，而是直接 fine-tune 第一阶段预训练后的 Q-Former 和 image encoder。原因是 retrieval 不涉及语言生成，不需要 LLM 的 decoder 能力。

BLIP-2 在 Flickr30K zero-shot 和 COCO fine-tuned retrieval 上都取得强结果：

| 模型         | Flickr I2T R@1 | Flickr T2I R@1 | COCO I2T R@1 | COCO T2I R@1 |
| ------------ | -------------: | -------------: | -----------: | -----------: |
| BLIP         |           96.7 |           86.7 |         82.4 |         65.1 |
| BLIP-2 ViT-L |           96.9 |           88.6 |         83.5 |         66.3 |
| BLIP-2 ViT-g |           97.6 |           89.7 |         85.4 |         68.3 |

Retrieval ablation 显示，COCO fine-tuning objectives 中加入 ITG 会提升 retrieval：ITC + ITM 的 I2T R@1 / T2I R@1 为 84.5 / 67.2，加入 ITG 后提升到 85.4 / 68.3。这支持论文关于 ITG 的解释：生成目标迫使 queries 抽取对文本更相关的视觉特征，因此也能改善 retrieval 对齐。



## 七、limitation

论文的 limitation 部分主要指出三类问题。

### 7.1 In-context learning 不明显

虽然现代 LLM 可以通过 few-shot examples 做 in-context learning，但 BLIP-2 实验中没有观察到给 LLM 提供 in-context VQA examples 能提升 VQA 性能。作者认为原因在于预训练数据每个样本只包含单个 image-text pair，模型没有从训练中学习到单个序列中多个 image-text pairs 之间的关系。

Flamingo 也报告过类似观察，并使用包含多个 image-text pairs per sequence 的 interleaved image-text dataset M3W 来支持这种能力。BLIP-2 未来也希望构造类似数据。

**这个局限说明：Q-Former 能把单图视觉信息接入 LLM，但并不自动赋予模型多图 few-shot multimodal prompting 能力。**要实现真正的 multimodal in-context learning，训练数据结构必须包含 interleaved sequences。



### 7.2 Image-to-text generation 可能错误

BLIP-2 的图像到文本生成可能因为多种原因产生不满意结果：

1. LLM 本身知识不准确；
2. 模型激活了错误 reasoning path；
3. LLM 不具备关于新图像内容的最新知识；
4. 视觉特征不足以区分细节。

这类错误反映了 frozen LLM 的双刃剑特性：它提供强语言能力，但也带来已有知识、偏见和推理路径的局限。



### 7.3 继承 LLM 风险

由于使用 frozen LLM，BLIP-2 继承了 LLM 风险，例如输出 offensive language、传播 social bias、泄露 private information 等。论文建议的缓解方法包括使用更好的 instructions 引导生成，或者在过滤掉有害内容的数据上训练。

从后续 LVLM 视角看，这一局限非常重要。多模态模型不仅会产生文本风险，还可能把视觉输入中的人群、身份、场景与语言偏见结合，导致更复杂的安全问题。

