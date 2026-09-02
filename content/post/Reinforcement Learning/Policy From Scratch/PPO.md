---
title: "Policy-Gradient-Method-From-Scratch | PPO"
description: "从 Actor-Critic 到 Proximal Policy Optimization"
date: 2026-09-01T14:53:32+08:00
lastmod: 2026-09-02T12:00:00+08:00
draft: false

categories:
  - Reinforcement Learning
tags:
  - Deep Learning
  - Agent
  - Optimization
  - Heuristic

toc: true
math: true
mermaid: true
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1788003344827_image.png
---

<!--more-->

## 零、写在前面

前面已经实现了 REINFORCE 和 Actor-Critic：

- REINFORCE 使用完整 Episode 的 Monte-Carlo return，代码简单，但方差比较大；
- Actor-Critic 引入 Critic，并使用 TD error 或 GAE 来估计 Advantage，训练稳定了很多。

不过，A2C 通常在采集一批数据后只更新一次。如果希望对同一批 rollout 做多轮 minibatch 更新，就必须解决一个问题：**策略不能因为重复更新而在一次 update 中变化太大**。

PPO（Proximal Policy Optimization，近端策略优化）就是为了解决这个问题提出的。它仍然保留了 Actor-Critic 和 GAE 的结构，但使用旧策略和当前策略之间的 probability ratio，并通过 clipping 限制策略更新幅度。

源码：[PPO](https://github.com/Equinox-2003/Policy-Gradient-Method-From-Scratch/tree/main/PPO)

这也是我第一次完整地学习PPO理论，本文结合源码，依次解释 PPO 的目标函数、rollout、GAE、observation normalization、learning-rate decay、checkpoint，以及训练 CartPole 和 LunarLander 的完整流程。



## 一、PPO

### 1.1 为什么需要 PPO？

策略梯度的目标仍然是最大化策略产生的期望回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0],
\qquad
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}.
$$

这里的 $J(\theta)$ 才是强化学习真正想优化的目标：让策略在环境中产生的平均折扣回报越来越高。它不是一个可以像监督学习 loss 那样直接对固定数据求值的普通目标，因为策略参数改变以后，未来访问到的状态、动作和 reward 分布也会一起改变。

REINFORCE 和 A2C 都是在用采样轨迹估计 $\nabla_\theta J(\theta)$。因此，下面的 policy-gradient loss 不是另一个最终目标，而是对真实目标的可计算近似：

$$
L^{A2C}_{actor}(\theta)
=
-\mathbb{E}_t
\left[
\log\pi_\theta(a_t|s_t)\hat A_t
\right].
$$

当 $\hat A_t>0$ 时，最小化这个 loss 会提高动作 $a_t$ 的概率；当 $\hat A_t<0$ 时，会降低它的概率。这个更新方向是正确的，但它隐含了一个条件：样本应该由接近当前策略的策略采集，并且当前策略不能在同一批旧数据上变化得太远。

因此 PPO 的问题可以明确地写成：

> 在最大化真实回报 $J(\theta)$ 的方向上，如何安全地重复利用由 $\pi_{\theta_{old}}$ 采集的 rollout？

PPO 并没有把最终目标换成别的东西，而是为这个问题构造了一个更保守的 surrogate objective。后面的 ratio 和 clipping，都是为了让这个 surrogate objective 在固定旧数据上仍然能够可靠地近似策略改进。

从理论上，一次保守的策略更新可以写成：

$$
\max_\theta
\quad
\mathbb{E}_t
\left[
r_t(\theta)\hat A_t
\right]
\qquad
\text{subject to}
\qquad
\mathbb{E}_t
\left[
D_{KL}
\left(
\pi_{\theta_{old}}
\;\middle\|\;
\pi_\theta
\right)
\right]
\leq\delta.
$$

这个形式包含 PPO 的两个核心要求：

1. Advantage 决定策略应该往哪个方向变化；
2. 新策略和旧策略之间的距离不能太大。

TRPO 直接尝试求解类似的 KL 约束问题，PPO 则用更简单的 clipped objective 近似这个约束。因此需要区分三个层次：

| 层次 | 含义 |
| --- | --- |
| 强化学习目标 | 最大化真实期望回报 $J(\theta)$ |
| 策略 surrogate objective | 用旧 rollout 估计当前策略的改进 |
| 实际训练 loss | clipped Actor loss、Critic value loss 和 entropy bonus 的加权和 |

#### 1.1.1 为什么 A2C 的一次更新不够？

A2C 的一次更新可以理解为：用当前策略采集一批样本，计算 Advantage，更新一次 Actor 和 Critic，然后丢弃这批数据。

**一次更新结束时，当前策略还没有离采样策略太远，所以这批数据仍然比较接近 on-policy 数据。**如果想把同一批样本重复训练多个 epoch，第 2 个 epoch 使用的已经不是“当前策略产生的新鲜数据”，而是旧策略留下来的数据。

最直接的做法是反复使用 A2C loss：

~~~python
for epoch in range(update_epochs):
    actor_loss = -(log_probs * advantages).mean()
    optimizer.zero_grad()
    actor_loss.backward()
    optimizer.step()
~~~

这个写法没有记录新旧策略的差异。**只要某个样本的 Advantage 偏大，多个 epoch 就会不断放大它的影响，最后可能让策略在一次 update 中发生过大的变化。**

PPO 的第一个改动是用 probability ratio 修正旧数据和当前策略之间的分布差异；第二个改动是用 clipping 限制修正后的目标不会无限增长。



### 1.2 Probability ratio

#### 1.2.1 从旧数据估计新策略

设一批 rollout 数据满足：

$$
\mathcal{D}\sim\pi_{\theta_{old}}.
$$

我们想优化的是当前策略 $\pi_\theta$，但手里只有旧策略采集的数据。因此不能只计算当前策略给动作的 log probability，还需要知道当前策略相对于旧策略改变了多少。

对于 rollout 中的一条样本，定义：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}.
$$

当 $\theta=\theta_{old}$ 时，$r_t=1$，并且 surrogate objective 对 $\theta$ 的梯度正好给出普通的 on-policy policy-gradient 方向。也就是说，PPO 并没有在初始位置改变 A2C 的更新方向，而是在策略开始偏离旧策略以后，显式记录这种偏离。

#### 1.2.2 ratio 只是校正，不是稳定性约束

如果只使用：

$$
L^{CPI}(\theta)
=
\mathbb{E}_t
\left[
r_t(\theta)\hat A_t
\right],
$$

ratio 可以做 importance-sampling correction，但它本身没有上限。一个偶然得到正 Advantage 的动作，概率越大，$r_t\hat A_t$ 就越大；一个偶然得到负 Advantage 的动作，概率越低，目标也会继续改善。

所以 ratio 解决的是“旧策略采样的数据如何估计当前策略的目标”，但没有解决“当前策略变化太大时，如何阻止旧数据中的噪声被反复放大”。第二个问题由 clipped surrogate objective 处理。

一次 rollout 由旧策略 $\pi_{\theta_{old}}$ 采集。采样时记录动作在旧策略下的 log probability：

$$
\log\pi_{\theta_{old}}(a_t|s_t).
$$

更新时用当前策略重新计算同一个动作的 log probability，并构造 probability ratio：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}.
$$

当 $r_t=1$ 时，新旧策略给出的动作概率相同；当 $r_t>1$ 时，当前策略提高了这个动作的概率；当 $r_t<1$ 时，当前策略降低了这个动作的概率。

这个 ratio 来自 **重要性采样（importance sampling）**：

$$
\mathbb{E}_{a\sim\pi_\theta}[f(a)]
=
\mathbb{E}_{a\sim\pi_{old}}
\left[
\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}f(a)
\right].
$$

把 $f(a)$ 换成 Advantage，就得到 PPO 中的 $r_t\hat A_t$。

在离散动作空间中，代码中我们可以使用 Categorical(logits=logits) 构造策略分布，并通过 log probability 的差计算 ratio：

```python
ratios = torch.exp(
    log_probs - batch["old_log_probs"][minibatch]
)
```

这和下式完全等价：

$$
r_t(\theta)=
\exp\left(
\log\pi_\theta(a_t|s_t)
-\log\pi_{\theta_{old}}(a_t|s_t)
\right).
$$

old_log_probs 必须在采样时保存，并在整个 PPO update 期间保持不变。它不能在每个 epoch 重新计算，否则分母也会跟着当前策略变化，ratio 就失去了比较新旧策略的意义。



### 1.3 Clipped Surrogate Objective

#### 1.3.1 先看没有 clipping 的目标

没有 clipping 时，最自然的 surrogate objective 是：

$$
L^{CPI}(\theta)
=
\mathbb{E}_t
\left[
r_t(\theta)\hat A_t
\right].
$$

它的更新方向没有问题：

- 如果 $\hat A_t>0$，提高该动作的概率会让目标变大；
- 如果 $\hat A_t<0$，降低该动作的概率会让目标变大。

问题在于，它没有告诉优化器“什么时候应该停下来”。例如 $\hat A_t=3$ 时：

| ratio | $r_t\hat A_t$ |
| ---: | ---: |
| 1.0 | 3.0 |
| 1.2 | 3.6 |
| 1.8 | 5.4 |
| 2.5 | 7.5 |

只要继续提高好动作的概率，未裁剪目标就会继续增长。**如果这个较大的 Advantage 只是采样噪声，多个 epoch 会不断放大这条样本的影响。**

#### 1.3.2 clipping 的设计目标

PPO 希望保留正确的更新方向，但截断过于乐观的策略改进。令：

$$
s_t^{unclip}(\theta)=r_t(\theta)\hat A_t,
$$

$$
s_t^{clip}(\theta)
=
\operatorname{clip}
\left(
r_t(\theta),1-\epsilon,1+\epsilon
\right)
\hat A_t.
$$

在最大化形式下，PPO 使用：

$$
L^{CLIP}_{max}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
s_t^{unclip}(\theta),
s_t^{clip}(\theta)
\right)
\right].
$$

这里的 minimum 表达的是一个保守原则：

> 对每个样本，只承认未裁剪收益和裁剪收益中更保守的那个收益。

当策略变化还在合理范围内时，PPO 和普通 policy gradient 一样更新；当策略变化已经超出范围时，继续偏移不会再得到额外的目标奖励。

#### 1.3.3 为什么 Advantage 的正负号会决定 clipping 的一侧？

假设 $\epsilon=0.2$，ratio 的允许区间是 $[0.8,1.2]$。

当 $\hat A_t>0$ 时，该动作是好动作，正确的方向是提高它的概率，也就是让 ratio 变大。因此需要限制上界。

例如 $\hat A_t=3$、$r_t=1.8$：

$$
s_t^{unclip}=1.8\times3=5.4,
\qquad
s_t^{clip}=1.2\times3=3.6.
$$

最大化目标取 minimum 后只保留 3.6，不再奖励把 ratio 从 1.2 推到 1.8。

当 $\hat A_t<0$ 时，该动作是坏动作，正确的方向是降低它的概率，也就是让 ratio 变小。因此需要限制下界。

例如 $\hat A_t=-3$、$r_t=0.2$：

$$
s_t^{unclip}=0.2\times(-3)=-0.6,
\qquad
s_t^{clip}=0.8\times(-3)=-2.4.
$$

最大化目标取 minimum 后得到 -2.4，相当于不再奖励把 ratio 从 0.8 继续压到 0.2。

四种情况可以总结为：

| Advantage | ratio 的变化 | 期望的策略行为 | clipping 的效果 |
| --- | --- | --- | --- |
| $\hat A>0$ | $r$ 增大 | 提高好动作概率 | $r>1+\epsilon$ 后停止额外奖励 |
| $\hat A>0$ | $r$ 减小 | 不应降低好动作概率 | 保留这部分惩罚 |
| $\hat A<0$ | $r$ 减小 | 降低坏动作概率 | $r<1-\epsilon$ 后停止额外奖励 |
| $\hat A<0$ | $r$ 增大 | 不应提高坏动作概率 | 保留这部分惩罚 |

#### 1.3.4 为什么不是简单地 clamp ratio？

下面的写法看起来很接近 PPO，但实际上不完整：

~~~python
ratio = torch.clamp(
    ratio,
    1.0 - epsilon,
    1.0 + epsilon,
)
loss = -ratio * advantages
~~~

**PPO 需要同时计算 unclipped objective 和 clipped objective，再根据 Advantage 的正负关系选择更保守的目标。如果先把 ratio 永久替换成 clipped ratio，就丢失了“只截断过于乐观的改进、保留错误方向惩罚”的逻辑。**

#### 1.3.5 最大化公式和实现中的最小化 loss

论文或推导中经常先写最大化目标：

$$
\max_\theta
\quad
\mathbb{E}_t
\left[
\min
\left(
r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

PyTorch optimizer 默认执行梯度下降，所以我们实现时需要把它改写成最小化 loss：

$$
L^{CLIP}(\theta)
=
-\mathbb{E}_t
\left[
\min
\left(
r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

#### 1.3.6 clipping 不是严格的 KL 约束

clipping 不能保证所有状态、所有动作的 ratio 都严格位于 $[1-\epsilon,1+\epsilon]$，也不能严格保证 KL divergence 小于某个固定值。它提供的是一种局部的、样本级的保守更新：

> 当某个样本已经把新旧策略推得过远时，它不会继续以同一个方向推动目标函数。

$\epsilon$ 越小，更新越保守；$\epsilon$ 越大，允许的策略变化越大，但旧 rollout 被过度利用的风险也越高。



### 1.4 PPO 中的 Actor-Critic

#### 1.4.1 PPO 的联合优化目标

PPO 的 Actor 和 Critic 共同参与一次 update，但优化的是不同部分。令 $\theta$ 表示 Actor 参数，$\phi$ 表示 Critic 参数，$\hat A_t$ 和 $\hat G_t$ 表示 rollout 阶段计算并固定下来的 Advantage 和 value target，则当前实现对应的联合最小化目标为：

$$
\begin{aligned}
\mathcal{L}^{PPO}(\theta,\phi)
=&
-\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat A_t,
\operatorname{clip}
\left(
r_t(\theta),1-\epsilon,1+\epsilon
\right)
\hat A_t
\right)
\right]
\\\\
&+
c_V\frac{1}{2}\mathbb{E}_t
\left[
\left(
V_\phi(s_t)-\hat G_t
\right)^2
\right]
\\\\
&-
c_H\mathbb{E}_t
\left[
\mathcal{H}
\left(
\pi_\theta(\cdot|s_t)
\right)
\right].
\end{aligned}
$$

其中：

- 第一项是 clipped policy loss，更新 Actor；
- 第二项是 value loss，更新 Critic；
- 第三项是 entropy bonus，鼓励 Actor 保留探索；
- $c_V$ 对应 value_coef；
- $c_H$ 对应 entropy_coef。

这就是 PPO 在代码中真正执行的优化目标。需要注意，$J(\theta)$ 是强化学习的最终目标，而 $\mathcal{L}^{PPO}$ 是在固定 rollout 上对这个目标进行优化的可计算 surrogate loss。

当前代码把前两项组合起来，并在 Actor loss 中加入 entropy：

~~~python
actor_loss = -torch.minimum(
    unclipped,
    clipped,
).mean()
actor_loss -= self.entropy_coef * entropy.mean()

critic_loss = self.value_coef * 0.5 * value_loss
total_loss = actor_loss + critic_loss
~~~

GAE 产生的 advantages、returns，以及采样时保存的 old_log_probs，都会在一次 PPO update 内保持固定。只有当前策略重新计算的 log_probs、value 和 entropy 会随着参数更新而变化。

PPO 仍然是 Actor-Critic 方法：

- Actor 输出策略分布，决定动作概率；
- Critic 输出状态价值 $V(s)$，用于计算 TD error、GAE 和 value target。

当前实现使用两个独立的 MLP：

```text
observation
    ├── Actor  -> action logits -> Categorical -> action
    └── Critic -> state value V(s)
```

两个网络接收相同的 observation，但参数不共享。这样代码比较直观，也可以为 Actor 和 Critic 设置不同的 learning rate。



### 1.5 GAE

PPO 的稳定性不仅来自 clipping，也依赖 Advantage 的质量。当前实现使用 GAE（Generalized Advantage Estimation，广义优势估计）。

#### 1.5.1 为什么需要 Advantage？

$$
A(s,a)=Q(s,a)-V(s).
$$

Advantage 衡量当前动作相对于该状态平均水平的好坏：

- $A(s,a)>0$：提高该动作的概率；
- $A(s,a)<0$：降低该动作的概率；
- $A(s,a)\approx0$：这个动作没有提供太多额外信息。

状态价值 $V(s)$ 作为 baseline 可以降低梯度方差，同时不改变策略梯度的期望。

#### 1.5.2 TD error

一步 TD error 为：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

实际实现使用 bootstrap mask：

$$
\delta_t=r_t+\gamma m_tV(s_{t+1})-V(s_t).
$$

一步 TD 方差低，但比较依赖 Critic；如果 Critic 还没有学好，错误的 value 估计会直接影响 Actor。

#### 1.5.3 GAE 递推

GAE 把多个时间尺度的 TD error 进行指数衰减累积：

$$
\hat A_t^{GAE(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.
$$

代码从 rollout 尾部向前递推：

$$
\hat A_t
=
\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

其中 $\lambda$ 控制 bias-variance trade-off：

| gae_lambda | 特点 |
| --- | --- |
| 0 | 只使用一步 TD，方差低、偏差可能较大 |
| 接近 1 | 接近 Monte-Carlo，偏差低、方差较大 |
| 0.95 | 常用折中配置 |

当前 PPO 默认使用：

```python
gamma = 0.99
gae_lambda = 0.95
```

Critic 的 target 为：

$$
\hat G_t=V_{old}(s_t)+\hat A_t.
$$

它可以理解为 GAE 对应的 lambda-return，不一定等于完整 Episode 的 Monte-Carlo return，但通常能在偏差和方差之间取得较好的折中。



### 1.6 terminated、truncated 与两个 mask

Gymnasium 的 env.step() 返回：

```python
next_observation, reward, terminated, truncated, info
```

这两个结束信号不能简单地全部当作 terminal：

- terminated=True：环境真正结束，例如 LunarLander 坠毁或成功着陆，下一状态价值应为 0；
- truncated=True：通常是达到时间上限，状态本身不一定是终止状态，可以使用下一状态价值 bootstrap。

当前实现使用两个 mask：

| 情况 | bootstrap_mask | gae_mask | 含义 |
| --- | ---: | ---: | --- |
| 普通 transition | 1 | 1 | bootstrap，并继续累积 GAE |
| terminated | 0 | 0 | 不 bootstrap，也不跨边界累积 |
| truncated | 1 | 0 | bootstrap，但不连接重置后的新 Episode |

代码如下：

```python
if terminated:
    next_value = 0.0
    bootstrap_mask = 0.0
else:
    next_tensor = self._obs_tensor(next_observation, update=False)
    with torch.no_grad():
        next_value = float(self.policy.get_value(next_tensor).item())
    bootstrap_mask = 1.0

done = terminated or truncated
gae_masks.append(0.0 if done else 1.0)
```

两个 mask 的原因是：“是否可以使用下一状态价值”和“是否应该把下一条 transition 接到当前 GAE”是两个不同问题。对 truncated 状态，可以使用最后状态的 value，但不能把 reset 后的新 Episode 接到旧 Episode 的 GAE 后面。



### 1.7 Value loss、Entropy 与 Gradient Clipping

Critic 拟合 GAE 生成的 target：

$$
L_V(\phi)=
\frac{1}{2}\mathbb{E}_t
\left[
\left(V_\phi(s_t)-\hat G_t\right)^2
\right].
$$

代码中使用 value_coef 加权：

```python
value_loss = self._value_loss(
    values,
    batch["old_values"][minibatch],
    batch["returns"][minibatch],
)
critic_loss = self.value_coef * 0.5 * value_loss
```

当前实现默认关闭 value clipping。命令行参数默认为 0.0，在 main() 中转换为 None：

```python
value_clip_range = args.value_clip_range
if value_clip_range <= 0:
    value_clip_range = None
```

关闭时，Critic 直接拟合原始 reward 尺度的 return：

```python
unclipped_loss = F.mse_loss(values, returns, reduction="none")
if self.value_clip_range is None:
    return unclipped_loss.mean()
```

如果启用 value clipping，当前 value 会被限制在 old_values 加减 value_clip_range 的范围内，并取 unclipped loss 与 clipped loss 中较大的一个。

由于当前实现没有对 return 做归一化，LunarLander 的 return 仍然是原始尺度，value_clip_range=0.2 通常过小，会限制 Critic 的有效更新。因此训练命令中使用 --value-clip-range 0，即关闭 value clipping。

Entropy bonus 为：

$$
\mathcal{H}(\pi(\cdot|s))
=
-\sum_a\pi(a|s)\log\pi(a|s).
$$

Actor loss 为：

$$
L_{actor}
=
L^{CLIP}
-\beta\mathbb{E}_t
\left[\mathcal{H}(\pi_\theta(\cdot|s_t))\right].
$$

代码中的 entropy_coef 默认是 0.01：

```python
actor_loss = -torch.minimum(unclipped, clipped).mean()
actor_loss -= self.entropy_coef * entropy.mean()
```

Entropy 越大，策略越随机；Entropy 越小，策略越接近确定性。训练早期需要探索，训练后期则可以适当减小 entropy coefficient。

为了减少 LunarLander 中异常 rollout 对参数的冲击，代码还分别对 Actor 和 Critic 做 gradient clipping：

```python
torch.nn.utils.clip_grad_norm_(
    self.policy.actor.parameters(), self.max_grad_norm
)
torch.nn.utils.clip_grad_norm_(
    self.policy.critic.parameters(), self.max_grad_norm
)
```



### 1.8 PPO 的完整流程

#### 1.8.1 一批 rollout 的生命周期

PPO 最容易混淆的地方，是“采样时的旧策略”和“更新时的当前策略”同时存在。对一批数据来说，生命周期如下：

~~~text
采样：
π_old -> observation, action, old_log_prob, old_value, reward

目标计算：
reward + old_value + next_value
    -> TD error
    -> GAE advantage
    -> value target return

更新：
π_current(observation)
    -> new_log_prob, new_value, entropy
    -> ratio = exp(new_log_prob - old_log_prob)
    -> clipped Actor loss + Critic loss
    -> 多个 epoch 的 minibatch 更新

结束：
丢弃旧 rollout，使用更新后的策略重新采样
~~~

old_log_prob 固定了 denominator，使每个 epoch 都是在比较当前策略和同一个旧策略，而不是比较两个都在变化的策略。一次 update 结束后，旧 rollout 被丢弃，下一批数据由更新后的策略重新采集。

#### 1.8.2 PPO 和 A2C 的差别

| 步骤 | A2C | PPO |
| --- | --- | --- |
| 采样 | 得到 rollout | 得到 rollout，并保存 old_log_prob |
| Advantage | TD 或 GAE | TD 或 GAE |
| Actor 目标 | log probability 乘 Advantage | ratio 乘 Advantage，并做 clipping |
| 数据复用 | 通常一次更新 | 多个 epoch 的 minibatch 更新 |
| 更新保护 | 没有显式策略距离约束 | 使用 clipped surrogate objective |

因此 PPO 可以看成：

$$
\text{PPO}
=
\text{Actor-Critic}
+
\text{GAE}
+
\text{Importance Sampling Ratio}
+
\text{Clipped Policy Objective}.
$$

这也是为什么 PPO 的代码结构和 A2C 很像，但 update 部分多了 old_log_probs、ratio、clipped objective 和多轮 minibatch。

一次 PPO update 的流程为：

1. 使用当前策略采集固定长度 rollout；
2. 保存 observation、action、旧策略 old_log_prob、旧 value、reward 和 mask；
3. 根据 reward、value 和 next value 计算 TD error；
4. 使用 GAE 得到 advantages；
5. 计算 returns = old_values + advantages；
6. 对整批 advantages 做标准化；
7. 随机打乱 rollout，切分为 minibatch；
8. 重新计算当前策略下的 log_probs 和 entropy；
9. 用新旧 log probability 的差计算 ratio；
10. 计算 clipped actor loss、value loss 和 entropy bonus；
11. 对同一批数据做多个 epoch 更新；
12. 进入下一轮 rollout。

```text
rollout by old policy
        │
        ├── old_log_probs ─────────────┐
        ├── values, next_values        │
        └── rewards, masks             │
                │                      │
                └── GAE -> advantages  │
                         │              │
current policy -> log_probs -> ratio ──┤
                                      │
                              clipped objective
                                      │
                            Actor + Critic update
```



## 二、代码实现

### 2.1 PPO 网络：policy.py

当前项目的网络定义在 PPO/policy.py。Actor 和 Critic 是两个独立的 MLP，激活函数使用 Tanh：

```python
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNet(nn.Module):
    """Separate actor and critic networks used by the PPO agent."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_weights()
```

Actor 最后一层输出 logits，而不是先手动计算 Softmax。Categorical(logits=logits) 会在内部完成数值稳定的归一化。

网络采用 orthogonal initialization：

```python
def _initialize_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
    nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
```

Actor 输出层的 gain 较小，使初始 logits 接近，策略早期更接近均匀探索；Critic 输出层使用正常尺度，便于开始拟合状态价值。

环境交互时采样动作：

```python
def sample_actions(
    self, obs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values = self.forward(obs)
    distribution = Categorical(logits=logits)
    actions = distribution.sample()
    return (
        actions,
        distribution.log_prob(actions),
        distribution.entropy(),
        values,
    )
```

PPO 更新时不能重新采样动作，而要评估已经执行过的 action 在当前策略下的概率：

```python
def evaluate_actions(
    self, obs: torch.Tensor, actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values = self.forward(obs)
    distribution = Categorical(logits=logits)
    log_probs = distribution.log_prob(actions)
    entropy = distribution.entropy()
    return log_probs, entropy, values
```

评估阶段使用 greedy action：

```python
def get_action(self, obs: torch.Tensor) -> int:
    with torch.no_grad():
        logits, _ = self.forward(obs)
        return torch.argmax(logits).item()
```



### 2.2 RunningMeanStd

CartPole 和 LunarLander 的 observation 都是向量，但不同维度的尺度并不相同。项目使用轻量级的 online observation normalization：

```python
class RunningMeanStd:
    """Keep online mean and variance for observation normalization."""

    def __init__(self, obs_dim: int):
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, observations: Sequence[float]):
        batch = np.asarray(observations, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        if batch.ndim != 2:
            raise ValueError("observations must be (obs_dim,) or (batch, obs_dim)")

        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean += delta * batch_count / total_count

        mean_a = self.var * self.count
        mean_b = batch_var * batch_count
        correction = delta**2 * self.count * batch_count / total_count
        self.var = np.maximum(
            (mean_a + mean_b + correction) / total_count,
            1e-6,
        )
        self.count = total_count

    def normalize(self, observations: Sequence[float]) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        return (observations - self.mean) / np.sqrt(self.var + 1e-8)
```

归一化公式为：

$$
\tilde s_t=
\frac{s_t-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$

这里没有保存所有历史 observation，而是在线合并旧统计量和当前 batch 的均值、方差。mean_a、mean_b 和 correction 组合了旧统计量和当前 batch 的统计量，因此同时支持单个 observation 和一批 observation。

var 使用 np.maximum(..., 1e-6) 防止某一个特征暂时没有变化时除以过小的标准差。_obs_tensor() 最终会把归一化后的 observation 强制转换成 float32，和 PyTorch 网络的参数类型保持一致。

由于模型测试时也必须使用训练阶段的均值和方差，所以这些统计量会写进 checkpoint：

```python
def state_dict(self) -> Dict[str, object]:
    return {"mean": self.mean, "var": self.var, "count": self.count}

def load_state_dict(self, state: Dict[str, object]):
    self.mean = np.asarray(state["mean"], dtype=np.float64)
    self.var = np.asarray(state["var"], dtype=np.float64)
    self.count = float(state["count"])
```

训练时 _obs_tensor(observation, update=True) 先更新统计量再归一化；评估时使用 update=False，只读取训练阶段的统计量，不再改变输入尺度。



### 2.3 Agent 初始化

Agent 根据环境自动获得 observation 和 action 的维度，并创建网络：

```python
self.env = gym.make(env_id)
self.obs_dim = self.env.observation_space.shape[0]
self.action_dim = self.env.action_space.n
self.obs_rms = RunningMeanStd(self.obs_dim)

self.policy = ActorCriticNet(
    self.obs_dim,
    self.action_dim,
    hidden_dim=hidden_dim,
).to(DEVICE)
```

当前使用一个 Adam，但有两个 parameter groups：

```python
self.optimizer = optim.Adam(
    [
        {
            "params": self.policy.actor.parameters(),
            "lr": lr,
            "initial_lr": lr,
        },
        {
            "params": self.policy.critic.parameters(),
            "lr": critic_lr,
            "initial_lr": critic_lr,
        },
    ],
    eps=1e-5,
)
```

因此 Actor 和 Critic 可以使用不同的 learning rate，同时只需一次 zero_grad() 和 step()。默认 rollout 和 update 参数为：

```text
lr          = 2.5e-4
critic_lr   = 1e-3
rollout     = 2048 steps
epochs      = 10
minibatch   = 64
```

### 2.4 固定长度 rollout

PPO 按 environment steps 采样，而不是按完整 Episode 采样。默认每次采集 2048 个 transition：

```python
for _ in range(self.rollout_steps):
    if len(episode_rewards) >= max_episodes:
        break

    obs_tensor = self._obs_tensor(observation, update=True)
    with torch.no_grad():
        action, log_prob, _, value = self.policy.sample_action(obs_tensor)

    next_observation, reward, terminated, truncated, _ = self.env.step(
        action
    )
```

每个 transition 保存：

```python
observations = []
actions = []
log_probs = []
values = []
next_values = []
rewards = []
bootstrap_masks = []
gae_masks = []
```

一个 rollout 可以包含多个完整 Episode，也可能在某个 Episode 中间结束。末尾未结束的轨迹通过最后状态的 value bootstrap。

### 2.5 compute_gae

核心实现如下：

```python
def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    bootstrap_masks: Sequence[float],
    gae_masks: Sequence[float],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    bootstrap_masks = np.asarray(bootstrap_masks, dtype=np.float32)
    gae_masks = np.asarray(gae_masks, dtype=np.float32)

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step]
            + gamma * next_values[step] * bootstrap_masks[step]
            - values[step]
        )
        gae = delta + gamma * gae_lambda * gae_masks[step] * gae
        advantages[step] = gae

    return values + advantages, advantages
```

对应：

$$
\delta_t=r_t+\gamma m_tV(s_{t+1})-V(s_t),
$$

$$
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1},
$$

$$
\hat G_t=V(s_t)+\hat A_t.
$$

因为 $\hat A_t$ 依赖 $\hat A_{t+1}$，所以需要从 rollout 尾部向前计算。函数返回 returns 和 advantages。GAE 在 NumPy 中完成，旧 value 不会通过计算图反向传播。

### 2.6 标准化 Advantage 并准备 batch

对整个 rollout 的 Advantage 做标准化：

```python
advantages = torch.as_tensor(
    rollout["advantages"],
    dtype=torch.float32,
    device=DEVICE,
)
if advantages.numel() > 1:
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
```

$$
\tilde A_t=
\frac{\hat A_t-\operatorname{mean}(\hat A)}
{\operatorname{std}(\hat A)+\epsilon}.
$$

标准化不改变 Advantage 的正负关系，但可以减少不同 rollout 之间 policy gradient 的尺度变化。当前实现不标准化 returns，因为 Critic 直接预测环境原始 reward 尺度。

### 2.7 PPO update

对同一个 rollout 做多个 epoch，每个 epoch 随机打乱样本并切分 minibatch：

```python
for _ in range(self.update_epochs):
    indices = torch.randperm(batch_size, device=DEVICE)
    for start in range(0, batch_size, minibatch_size):
        minibatch = indices[start : start + minibatch_size]
        log_probs, entropy, values = self.policy.evaluate_actions(
            batch["observations"][minibatch],
            batch["actions"][minibatch],
        )
```

然后计算 ratio 和 actor loss：

```python
ratios = torch.exp(
    log_probs - batch["old_log_probs"][minibatch]
)
advantages = batch["advantages"][minibatch]
unclipped = ratios * advantages
clipped = torch.clamp(
    ratios,
    1.0 - self.clip_epsilon,
    1.0 + self.clip_epsilon,
) * advantages
actor_loss = -torch.minimum(unclipped, clipped).mean()
actor_loss -= self.entropy_coef * entropy.mean()
```

这里 log_probs 是当前策略重新计算的结果，old_log_probs 是采样时保存的常量。value loss 和优化步骤为：

```python
value_loss = self._value_loss(
    values,
    batch["old_values"][minibatch],
    batch["returns"][minibatch],
)
critic_loss = self.value_coef * 0.5 * value_loss

self.optimizer.zero_grad()
(actor_loss + critic_loss).backward()
torch.nn.utils.clip_grad_norm_(
    self.policy.actor.parameters(), self.max_grad_norm
)
torch.nn.utils.clip_grad_norm_(
    self.policy.critic.parameters(), self.max_grad_norm
)
self.optimizer.step()
```

rollout 仍然是 on-policy 数据，不能无限期重复使用。rollout_steps、update_epochs 和 clip_epsilon 共同决定数据复用率和稳定性：

| 参数 | 作用 |
| --- | --- |
| rollout_steps | 每次采集多少环境步 |
| update_epochs | 一批数据重复训练多少轮 |
| clip_epsilon | 允许新旧策略概率变化的范围 |

### 2.8 Learning-rate decay

当前实现按照总环境步数线性衰减两个 parameter group 的 learning rate：

```python
def _update_learning_rate(self, total_steps: int):
    if self.lr_decay_steps == 0:
        return

    progress = 1.0 - min(total_steps / self.lr_decay_steps, 1.0)
    scale = max(progress, self.min_lr_ratio)
    for group in self.optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale
```

默认：

```text
lr_decay_steps = 1_000_000
min_lr_ratio   = 0.1
```

即前 1,000,000 个环境步线性衰减，最低保持初始 learning rate 的 10%：

$$
lr_t=lr_0\cdot
\max\left(1-\frac{N_t}{N_{decay}},r_{min}\right).
$$

如果要关闭衰减，使用 --lr-decay-steps 0。学习率衰减可以减轻高分阶段的震荡，但不能代替正确的 GAE、mask 和 clipping。

### 2.9 Checkpoint、日志和评估

模型、observation statistics、曲线和日志按环境分目录保存：

```text
PPO/
├── agent.py
├── policy.py
├── TECHNICAL.md
├── checkpoints/
│   ├── CartPole-v1/
│   └── LunarLander-v3/
├── plots/
│   ├── CartPole-v1/
│   └── LunarLander-v3/
└── PPO_<env_id>.log
```

checkpoint 保存模型参数和 RunningMeanStd：

```python
torch.save(
    {
        "episode": episode,
        "avg_reward": avg_reward,
        "env_id": self.env_id,
        "model_state_dict": self.policy.state_dict(),
        "obs_rms": self.obs_rms.state_dict(),
    },
    self.checkpoint_dir / filename,
)
```

best_policy.pth 只有在完整的最近 100 轮窗口刷新最佳平均 reward 时才保存：

```python
last_100 = episode_rewards[
    max(0, episode_index - 99) : episode_index + 1
]
current_avg = float(np.mean(last_100))

if len(last_100) == 100 and current_avg > best_avg:
    best_avg = current_avg
    self._save_model(
        episode=episode_index + 1,
        avg_reward=current_avg,
        best=True,
    )
```

评估时不更新 observation statistics，并使用 greedy action：

```python
obs_tensor = self._obs_tensor(observation, update=False)
action = self.policy.get_action(obs_tensor)
```

因此评估输入尺度和训练一致，但不会污染训练阶段保存的统计量。



### 2.10 训练与测试——CartPole

```powershell
uv run .\agent.py `
    --env-id CartPole-v1 `
    --train `
    --episodes 2000 `
    --target-reward 475 `
    --eval-episodes 20 `
    --seed 42
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1788340000565_image.png)

```
开始训练 PPO（目标平均奖励: 475.0）...
09-02 17:04:10: Update    1 | Episode    97 | 近100轮均值:    21.1 | Actor Loss:  -0.0336 | Critic Loss:   4.5357 | Entropy:  0.684
09-02 17:04:29: Update   10 | Episode   238 | 近100轮均值:   169.0 | Actor Loss:  -0.0081 | Critic Loss:   0.2260 | Entropy:  0.506
09-02 17:04:48: Update   20 | Episode   279 | 近100轮均值:   355.7 | Actor Loss:  -0.0045 | Critic Loss:   0.0044 | Entropy:  0.425

达到目标平均奖励 475.0！vg-100=476.52！在 Episode 309 提前结束！

训练全部结束！

开始策略评估 (20 轮)...
评估轮次  1 | 总得分:   500.0
评估轮次  2 | 总得分:   500.0
评估轮次  3 | 总得分:   500.0
评估轮次  4 | 总得分:   500.0
评估轮次  5 | 总得分:   500.0
评估轮次  6 | 总得分:   500.0
评估轮次  7 | 总得分:   500.0
评估轮次  8 | 总得分:   500.0
评估轮次  9 | 总得分:   500.0
评估轮次 10 | 总得分:   500.0
评估轮次 11 | 总得分:   500.0
评估轮次 12 | 总得分:   500.0
评估轮次 13 | 总得分:   500.0
评估轮次 14 | 总得分:   500.0
评估轮次 15 | 总得分:   500.0
评估轮次 16 | 总得分:   500.0
评估轮次 17 | 总得分:   500.0
评估轮次 18 | 总得分:   500.0
评估轮次 19 | 总得分:   500.0
评估轮次 20 | 总得分:   500.0
评估平均奖励: 500.00
(Policy-RL) PS D:\gitRepo\Policy-Gradient-Method-From-Scratch\PPO> 
```

1min不到就训好了。



### 2.11 训练与测试——LunarLander

```powershell
uv run .\agent.py `
    --env-id CartPole-v1 `
    --train `
    --episodes 2000 `
    --target-reward 475 `
    --eval-episodes 20 `
    --seed 42
```

LunarLander 的 observation 是 8 维向量，动作空间有 4 个离散动作：

| Action | 含义 |
| ---: | --- |
| 0 | 不操作 |
| 1 | 启动左方向引擎 |
| 2 | 启动主引擎 |
| 3 | 启动右方向引擎 |

可以使用下面的配置从头训练：

```powershell
uv run .\agent.py `
    --env-id LunarLander-v3 `
    --train `
    --episodes 10000 `
    --target-reward 270 `
    --rollout-steps 2048 `
    --update-epochs 10 `
    --minibatch-size 64 `
    --lr 0.00025 `
    --critic-lr 0.001 `
    --entropy-coef 0.01 `
    --value-clip-range 0 `
    --lr-decay-steps 1000000 `
    --min-lr-ratio 0.1 `
    --eval-episodes 20 `
    --seed 42
```

这里显式关闭 value clipping，是因为当前 return 没有归一化；学习率衰减用于减小高分阶段的参数震荡。

训练过程中使用完整的最近 100 轮窗口：

$$
\operatorname{Avg100}_n
=\frac{1}{100}
\sum_{i=n-99}^{n}R_i.
$$

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1788340113158_image.png)

大概3000轮的时候早停了，然后测一下：

```powershell
uv run .\agent.py `
    --env-id LunarLander-v3 `
    --checkpoint ".\checkpoints\LunarLander-v3\best_policy.pth" `
    --eval-episodes 5 `
    --render `
    --seed 114514
```

```powershell
成功加载模型: best_policy.pth avg-100: 270.6495948380118

开始策略评估 (5 轮)...
评估轮次  1 | 总得分:   288.5
评估轮次  2 | 总得分:   288.8
评估轮次  3 | 总得分:   272.5
评估轮次  4 | 总得分:   272.2
评估轮次  5 | 总得分:   312.1
评估平均奖励: 286.82
```

相比之前的 A2C，效果又改进了一些，而且训练更稳定了。



## 三、REINFORCE、A2C 与 PPO 对比

| 维度 | REINFORCE | A2C / Actor-Critic | PPO |
| --- | --- | --- | --- |
| 回报估计 | Monte-Carlo return | TD / GAE | TD / GAE |
| 是否使用 Critic | 否 | 是 | 是 |
| 一批数据更新次数 | 通常一次 | 通常一次 | 多个 epoch |
| 是否保存旧策略概率 | 否 | 通常不需要 | 必须保存 |
| 方差 | 高 | 较低 | 较低 |
| 更新约束 | 无 | 无显式 clipping | clipped ratio |
| 样本利用率 | 低 | 中等 | 较高 |
| 实现复杂度 | 最低 | 中等 | 更高 |
| 高分阶段稳定性 | 较差 | 较好 | 通常更好 |

算法演化可以概括为：

```text
REINFORCE
    └── 加入 V(s) baseline 和 TD/GAE
            └── Actor-Critic / A2C
                    └── 保存 old policy + 多轮更新 + clipping
                            └── PPO
```

PPO 不是完全不同的一类策略梯度算法，它可以看成是在 Actor-Critic 的基础上增加了更保守的策略更新目标。

TRPO 通过 KL divergence 约束新旧策略，理论保证较强但实现复杂；PPO 用 clipped objective 近似实现“不要离旧策略太远”的思想，工程实现更简单。



## 四、一些问题

### 4.1 为什么 PPO 必须保存 old_log_probs？

因为 rollout 是由旧策略采集的。只有保存旧策略下的 log probability，更新时才能知道当前策略相对于采样策略变化了多少，并构造 ratio。

采样阶段保存 old_log_probs，更新阶段重新计算当前策略的 log_probs，二者之差再经过 exp 就是 probability ratio。

### 4.2 为什么不直接把 ratio 截断后计算 loss？

PPO 要同时比较 unclipped objective 和 clipped objective。只把 ratio 截断后计算一个 loss，会丢失 PPO 目标函数中针对 Advantage 正负号设计的 minimum 逻辑。

### 4.3 为什么 Advantage 要标准化，而 return 不标准化？

Advantage 是 Actor loss 中的样本权重，标准化可以控制 policy gradient 的尺度，不改变正负方向。

Return 是 Critic 的回归目标。当前实现让 Critic 直接预测环境原始 reward 尺度，因此不对 return 做标准化，同时默认关闭 value clipping。

### 4.4 为什么 rollout 不要求是完整 Episode？

PPO 按 environment steps 采集数据，固定长度 rollout 更容易控制 batch size。rollout 末尾如果 Episode 尚未结束，就用最后状态的 value bootstrap；遇到 terminated 时则把下一状态 value 置为零。

### 4.5 为什么 terminated 和 truncated 不能都置零？

terminated 代表任务真正结束，未来回报为零；truncated 只代表当前采样被时间上限截断，状态仍可能有未来价值。

因此：

```text
terminated: bootstrap_mask = 0, gae_mask = 0
truncated:  bootstrap_mask = 1, gae_mask = 0
```

如果把 truncated 当成 terminated，会丢失时间截断点之后的价值；如果让 GAE 跨 truncated 边界递推，又会错误连接 reset 后的新 Episode。

### 4.6 为什么 best checkpoint 根据最近 100 轮，而不是当前 Episode？

单个 Episode 的 reward 噪声很大。LunarLander 中偶然成功着陆可能得到高分，但策略整体仍不稳定。最近 100 轮平均 reward 更接近实际训练目标，所以只有完整的 100 轮窗口刷新最佳值时才保存 best_policy.pth。





## 五、总结

PPO 的关键并不只是把 A2C 的 loss 多跑几遍，而是必须保存旧策略信息，并用 clipped objective 控制新旧策略之间的变化：

1. rollout 由旧策略采集，保存 old_log_probs；
2. Critic 提供 value，GAE 生成 Advantage 和 return；
3. 当前策略重新计算 action log probability；
4. 用 ratio = exp(new_log_prob - old_log_prob) 比较策略变化；
5. 用 min(unclipped, clipped) 限制过大的策略更新；
6. 对同一批数据进行多个 epoch 的 minibatch 更新；
7. 用 observation normalization、gradient clipping 和 learning-rate decay 提高实际训练稳定性；
8. 保存模型时同时保存 RunningMeanStd，保证评估输入分布一致。

从 REINFORCE 到 A2C，再到 PPO，核心变化可以概括为：

> 用更低方差的 Advantage 指导策略，再用更保守的目标函数控制策略更新。

有空的话，再学习一下GRPO吧：）

