---
title: "Actor Critic"
description: ""
date: 2026-08-31T10:34:10+08:00
lastmod: 2026-08-31T10:34:10+08:00
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

之前的 REINFORCE 训 LunarLander 最佳最近 100 轮平均 reward 到190 就收敛了，这其实是因为 REINFORCE 本身的局限性，为了解决其本身的问题，引入了A2C 算法。

源码：[Actor-Critic](https://github.com/Equinox-2003/Policy-Gradient-Method-From-Scratch/tree/main/Actor-Critic)



## 一、Actor Critic

### 1.1 为什么需要 Actor-Critic？

策略网络表示为 `π_θ(a|s)`，目标是最大化期望折扣回报：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0],
\qquad
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}.
$$
REINFORCE 使用 `G_t` 直接乘在 `log π_θ(a_t|s_t)` 上。

REINFORCE 算法虽然简单直观，但存在两个重要缺陷：

1. **高方差**：每次梯度更新都依赖一个完整的 Episode 轨迹，噪声非常大，导致收敛极其不稳定。
2. **无价值引导**：REINFORCE 完全不使用状态价值信息，梯度方向完全由回报驱动，容易在非最优策略附近震荡。

**Actor-Critic 算法的核心目的**就是**解决方差问题**，让策略梯度更稳定、更高效地收敛。



### 1.2 Actor-Critic 核心思想

Actor-Critic 是 **REINFORCE 的重大改进**，它引入了一个**价值网络（Critic）**来降低梯度方差。

>   REINFORCE 通过采样轨迹，用 reward 来估计 V

**基本结构**

- **Actor**：策略网络（Policy Network），负责选择动作
- **Critic**：价值网络（Value Network），负责评估当前状态的好坏

Critic 用来估计 baseline，Actor 使用优势函数：

$$
A_t = G_t - V_\phi(s_t).
$$
因此 Actor 的损失为：

$$
L_{actor} = -\frac{1}{T}\sum_t \log \pi_\theta(a_t|s_t)\,\hat A_t
             - \beta \frac{1}{T}\sum_t \mathcal{H}(\pi_\theta(\cdot|s_t)),
$$
其中 `β` 是熵正则系数。熵项鼓励早期探索，避免策略过早退化为确定性策略。

>   Q：$- \beta \frac{1}{T}\sum_t \mathcal{H}(\pi_\theta(\cdot|s_t))$ 在做什么？
>
>   A：
>
>   该项指**在状态 $s_t$ 下，策略 $\pi_\theta$ 的条件熵（Conditional Entropy）**。
>
>   loss 的第一项自然是为了高回报，第二项则是提高策略随机性，避免陷入局部策略。



Critic 拟合回报目标：

$$
L_{critic} = c_v \cdot \operatorname{SmoothL1}(V_\phi(s_t), G_t).
$$


Actor-Critic 的策略梯度公式为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d_{\pi_\theta}} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s,a) \right]
$$

>   这里是直接对 REINFORCE 的公式进行代入得到的。



### 1.3 Actor-Critic 相比 REINFORCE 的优势

| 维度         | REINFORCE                  | Actor-Critic                     |
|--------------|---------------------------|---------------------------------|
| 方差         | 极高                      | 显著降低                        |
| 收敛稳定性   | 非常差                    | 好很多                          |
| 学习速度     | 慢                        | 快                              |
| 实现复杂度   | 极低                      | 稍高（需要 Value Network）     |
| 收敛速度     | 慢                        | 快                              |



### 1.4 GAE

**GAE（Generalized Advantage Estimation，广义优势估计）**是一种更稳定地估计优势函数 $A(s,a)$ 的方法。

**1. 为什么需要 Advantage？**

Actor 的更新通常写成：
$$
\nabla_\theta J(\theta) \approx \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \hat A_t \right]
$$
其中：
$$
A_t = Q(s_t,a_t)-V(s_t)
$$
直观上，$A_t$ 表示：

>   当前动作比“处于该状态时的平均表现”好多少。

如果 $A_t>0$，提高该动作的概率；如果 $A_t<0$，降低该动作的概率。



**2. 一步 TD 的问题**

最简单的 Actor-Critic 使用一步 TD 误差：
$$
\delta_t = r_t+\gamma V(s_{t+1})-V(s_t) 
$$
并令：
$$
\hat A_t = \delta_t 
$$
它的优点是方差低、更新快，但非常依赖 Critic 的准确性。如果 Critic 估计不准，Actor 得到的更新方向就可能有较大偏差。



**3. Monte-Carlo 回报的问题**

另一种方式是使用完整回报：
$$
\hat A_t = G_t - V(s_t) 
$$
其中 $G_t$ 是从当前时刻开始的完整折扣回报。

它对 Critic 的依赖较小，偏差低，但 LunarLander 这类长时间跨度环境中，完整回报会受到大量随机动作影响，导致方差很大，训练不稳定。



**4. GAE 的核心思想**

GAE 将多个时间尺度的 TD 误差加权组合：
$$
\hat A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} 
$$
其中：
$$
\delta_t = r_t+\gamma V(s_{t+1})-V(s_t) 
$$
也可以理解为：

-   只看一步 TD：更新快，但偏差可能较大；
-   看完整回报：偏差较小，但方差较大；
-   GAE：对 1-step、2-step、3-step……估计进行加权平均。

$\lambda$ 控制偏差-方差权衡：

| $\lambda$ | 特点                             |
| --------- | -------------------------------- |
| `0`       | 纯一步 TD，低方差、高偏差        |
| `1`       | 接近 Monte-Carlo，低偏差、高方差 |
| `0.95`    | 常用折中配置                     |



**5. 为什么 Actor-Critic 中引入 GAE？**

主要有四个原因：

1.  降低策略梯度方差，使 Actor 更新更稳定；
2.  减少 Critic 估计误差对 Actor 的直接影响；
3.  改善长时间跨度任务中的奖励分配；
4.  在 LunarLander 这类环境中通常比纯 Monte-Carlo 回报更容易收敛。

因此，GAE 的本质是：

>   用一组不同时间跨度的 TD 误差，构造一个更加平滑、稳定的 Advantage 估计。





### 1.5 算法流程

1. **采样轨迹**：与 REINFORCE 相同，采集一个 Episode，同时记录 `log_prob`、`value`、`entropy` 和 `reward`
2. **估计下一状态价值**：如果环境真正终止，则 `V(s_{t+1})=0`；如果只是时间截断，则使用 Critic 对下一状态进行 bootstrap
3. **优势估计**：先计算 TD residual，再使用 GAE 得到每个时间步的 `advantage`
4. **策略更新**（Actor）：使用 Advantage 作为权重更新策略梯度，并加入熵正则
5. **价值更新**（Critic）：让状态价值逼近 GAE 生成的 value target
6. **保存与评估**：记录最近 100 轮平均 reward，并保存当前最佳 checkpoint



### 1.5 代码实现

在之前 REINFORCE 的代码上修改。



#### 1.5.1 A2C 网络

```python
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and the scalar state-value estimate."""
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value

    def sample_action(
        self, obs: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return its log-probability, entropy and value."""
        logits, value = self.forward(obs)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return (
            action.item(),
            distribution.log_prob(action),
            distribution.entropy(),
            value,
        )

    def get_action(self, obs: torch.Tensor) -> int:
        """Return the greedy action for evaluation."""
        with torch.no_grad():
            logits, _ = self.forward(obs)
            return torch.argmax(logits).item()

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a detached value estimate for bootstrapping."""
        with torch.no_grad():
            _, value = self.forward(obs)
            return value

```



#### 1.5.2 RunningMeanStd

我们写一个轻量级的在线 observation normalization 模块，解决不同状态特征尺度不一致的问题，同时不需要保存完整轨迹或修改环境奖励。

```python
class RunningMeanStd:
    """Online observation normalization for continuous state features."""

    def __init__(self, obs_dim: int):
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, observation: Sequence[float]):
        batch = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        mean_a = self.var * self.count
        mean_b = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_var = (mean_a + mean_b + correction) / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count

    def normalize(self, observation: Sequence[float]) -> np.ndarray:
        return (
            np.asarray(observation, dtype=np.float32) - self.mean
        ) / np.sqrt(self.var + 1e-8)

    def state_dict(self) -> Dict[str, object]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Dict[str, object]):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
```



#### 1.5.3 Agent 初始化

在 `Agent` 中，首先根据环境自动获取 observation 和 action 的维度，然后分别创建 Actor 和 Critic 的优化器：

```python
class Agent:
    def __init__(
        self,
        env_id: str = "CartPole-v1",
        device: str = "cpu",
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: Optional[int] = 42,
    ):
        self.device = torch.device(device)
        self.env = gym.make(env_id)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.obs_rms = RunningMeanStd(self.obs_dim)

        self.policy = ActorCriticNet(
            self.obs_dim,
            self.action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.policy.actor.parameters(),
            lr=lr,
        )
        self.critic_optimizer = optim.Adam(
            self.policy.critic.parameters(),
            lr=critic_lr,
        )
```

这里 Actor 和 Critic 使用两个优化器：

- Actor 学习率为 `lr`，负责优化动作概率分布；
- Critic 学习率为 `critic_lr`，负责拟合状态价值；
- 一般 Critic 可以使用稍大的学习率，因为它需要尽快跟上当前策略产生的回报分布。

严格来说，经典 A2C 通常还会并行运行多个环境，然后同步汇总各个环境的 trajectory。我们这里就采用单环境、单 Episode 更新的形式，保留 Actor-Critic 和 GAE 的核心逻辑，代码更容易理解和调试。



#### 1.5.4 采样一条轨迹

训练时，Actor 根据当前状态采样动作，Critic 同时给出当前状态的价值估计：

```python
log_probs = []
values = []
next_values = []
entropies = []
bootstrap_masks = []
rewards = []

obs, _ = self.env.reset(seed=reset_seed)
done = False

while not done:
    obs_tensor = self._obs_tensor(obs, update=True)
    action, log_prob, entropy, value = self.policy.sample_action(obs_tensor)
    next_obs, reward, terminated, truncated, _ = self.env.step(action)

    if terminated:
        next_value = torch.zeros(
            (),
            dtype=torch.float32,
            device=self.device,
        )
        bootstrap_masks.append(0.0)
    else:
        next_obs_tensor = self._obs_tensor(
            next_obs,
            update=truncated,
        )
        next_value = self.policy.get_value(next_obs_tensor)
        bootstrap_masks.append(1.0)

    log_probs.append(log_prob)
    values.append(value)
    next_values.append(next_value)
    entropies.append(entropy)
    rewards.append(float(reward))

    obs = next_obs
    done = terminated or truncated
```

这里要区分 Gymnasium 返回的两个结束信号：

- `terminated=True`：环境真正终止，例如 LunarLander 坠毁或成功着陆，此时不能继续 bootstrap；
- `truncated=True`：达到时间上限，状态本身不一定是终止状态，此时仍然可以使用 `V(s_{t+1})` 进行 bootstrap。

因此代码通过 `bootstrap_masks` 控制是否保留下一状态的价值：

$$
m_t =
\begin{cases}
0, & \text{terminated} \\
1, & \text{truncated 或未结束}
\end{cases}
$$

`_obs_tensor()` 在训练时还会更新 `RunningMeanStd`，然后再把 observation 转成 PyTorch Tensor。评估时传入 `update=False`，只使用训练阶段已经保存好的统计量。




#### 1.5.5 使用 GAE 计算 Advantage

轨迹采样完成后，调用 `_compute_gae()`：

```python
def _compute_gae(
    self,
    rewards: Sequence[float],
    values: torch.Tensor,
    next_values: torch.Tensor,
    bootstrap_masks: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    reward_tensor = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=self.device,
    )
    mask_tensor = torch.as_tensor(
        bootstrap_masks,
        dtype=torch.float32,
        device=self.device,
    )

    deltas = (
        reward_tensor
        + self.gamma * next_values * mask_tensor
        - values.detach()
    )

    advantages = []
    gae = torch.zeros(
        (),
        dtype=torch.float32,
        device=self.device,
    )

    for delta, mask in zip(
        reversed(deltas),
        reversed(mask_tensor),
    ):
        gae = (
            delta
            + self.gamma * self.gae_lambda * mask * gae
        )
        advantages.append(gae)

    advantages.reverse()
    advantages_tensor = torch.stack(advantages).detach()
    returns = (
        advantages_tensor
        + values.detach()
    ).detach()
    return returns, advantages_tensor
```

代码先计算每个时间步的 TD residual：

$$
\delta_t = r_t + \gamma m_t V(s_{t+1}) - V(s_t)
$$

然后从后往前递推：

$$
\hat A_t = \delta_t + \gamma\lambda m_t \hat A_{t+1}
$$

最后得到：

$$
\hat G_t = \hat A_t + V(s_t)
$$

这里的 `returns` 就是 Critic 的训练目标。`values.detach()` 的作用是切断计算图，使得 Actor 的 loss 不会通过 advantage 反向影响 Critic；Actor 和 Critic 的优化目标保持独立。




#### 1.5.6 更新 Actor 和 Critic

得到 `returns` 和 `advantages` 后，分别构造两个 loss：

```python
advantages = self._normalize(advantages)

actor_loss = (
    -(log_probs_tensor * advantages).mean()
    - self.entropy_coef * entropies_tensor.mean()
)

critic_loss = self.value_coef * F.smooth_l1_loss(
    values_tensor,
    returns,
)

self.actor_optimizer.zero_grad()
actor_loss.backward()
torch.nn.utils.clip_grad_norm_(
    self.policy.actor.parameters(),
    self.max_grad_norm,
)
self.actor_optimizer.step()

self.critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(
    self.policy.critic.parameters(),
    self.max_grad_norm,
)
self.critic_optimizer.step()
```

Actor loss 为：

$$
L_{actor} = -\mathbb{E}[\log \pi_\theta(a_t|s_t)\hat A_t]
            - \beta\mathbb{E}[\mathcal{H}(\pi_\theta)]
$$

其中第一项负责提高高优势动作的概率，第二项是熵正则项，用来维持一定的探索能力。

Critic loss 使用 `SmoothL1Loss`：

$$
L_{critic}=c_v\operatorname{SmoothL1}(V_\phi(s_t),\hat G_t)
$$

相比 MSE，Smooth L1 在 value target 偶尔出现较大误差时更加稳定。最后使用 `clip_grad_norm_` 限制梯度范数，避免 LunarLander 中较大的回报波动造成梯度爆炸。

> Q：为什么要对 Advantage 做标准化？
>
> A：不同 Episode 的 reward 尺度差异可能很大。标准化后，Actor 的更新主要关注动作之间的相对优劣，而不是某一轮 reward 的绝对数值，可以减少梯度尺度波动。




#### 1.5.7 保存和加载模型

由于 observation 做过归一化，checkpoint 不能只保存神经网络参数，还需要保存 `RunningMeanStd` 的统计量：

```python
torch.save(
    {
        "episode": episode,
        "model_state_dict": self.policy.state_dict(),
        "actor_optimizer_state_dict": (
            self.actor_optimizer.state_dict()
        ),
        "critic_optimizer_state_dict": (
            self.critic_optimizer.state_dict()
        ),
        "obs_rms": self.obs_rms.state_dict(),
        "avg_reward": avg_reward,
        "env_id": self.env_id,
    },
    checkpoint_path,
)
```

CartPole 和 LunarLander 的 observation 分布不同，所以程序会把它们分别保存到：

```text
Actor-Critic/checkpoints/CartPole-v1/best_policy.pth
Actor-Critic/checkpoints/LunarLander-v3/best_policy.pth
```

加载时必须同时恢复网络参数和 observation normalization 参数，否则评估阶段的输入分布会发生变化：

```python
checkpoint = torch.load(
    model_path,
    map_location=self.device,
    weights_only=False,
)
self.policy.load_state_dict(checkpoint["model_state_dict"])
self.obs_rms.load_state_dict(checkpoint["obs_rms"])
```




#### 1.5.8 训练和测试

在 `Actor-Critic` 目录下训练 CartPole：

```text
uv run .\agent.py --env-id CartPole-v1 --train --episodes 2000 --target-reward 475
```

训练 LunarLander：

```text
uv run .\agent.py --env-id LunarLander-v3 --train --episodes 10000 --target-reward 250
```

测试最佳权重并打开渲染：

```text
uv run .\agent.py --env-id LunarLander-v3 --eval-episodes 5 --render
```



### 1.6 测试

```python
def main():
    parser = argparse.ArgumentParser(description="Train or evaluate Actor-Critic.")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--train", action="store_true", help="开始训练")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--target-reward", type=float, default=475.0)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent = Agent(
        env_id=args.env_id,
        device=device,
        hidden_dim=128,
        lr=3e-4 if "LunarLander" in args.env_id else 1e-3,
        critic_lr=1e-3,
        seed=args.seed,
    )

    try:
        if args.train:
            agent.train(
                target_reward=args.target_reward,
                max_episodes=args.episodes,
            )
        else:
            agent.load_model(args.checkpoint)

        agent.evaluate(num_episodes=args.eval_episodes, render=args.render)
    finally:
        agent.close()


if __name__ == "__main__":
    main()

```



#### 1.6.1 CartPole

```text
(Policy-RL) PS D:\gitRepo\Policy-Gradient-Method-From-Scratch\Actor-Critic> uv run .\agent.py --env-id CartPole-v1 --train --episodes 2000 --target-reward 475
开始训练 Actor-Critic（目标平均奖励: 475.0）...
09-01 13:34:59: Episode    1 | 单轮奖励:   14.0 | 近100轮均值:   14.0 | Actor Loss:  -0.0165 | Critic Loss:   2.8430
09-01 13:35:01: Episode   20 | 单轮奖励:   16.0 | 近100轮均值:   18.0 | Actor Loss:  -0.0052 | Critic Loss:   2.5541
09-01 13:35:03: Episode   40 | 单轮奖励:   22.0 | 近100轮均值:   19.7 | Actor Loss:  -0.0106 | Critic Loss:   2.7027
09-01 13:35:04: Episode   60 | 单轮奖励:   12.0 | 近100轮均值:   20.2 | Actor Loss:   0.0907 | Critic Loss:   1.2556
略...
09-01 13:37:37: Episode  500 | 单轮奖励:  500.0 | 近100轮均值:  457.6 | Actor Loss:   0.0367 | Critic Loss:   0.6167
09-01 13:37:52: Episode  520 | 单轮奖励:  500.0 | 近100轮均值:  473.2 | Actor Loss:  -0.0030 | Critic Loss:   0.0799

达到目标平均奖励 475.0！在 Episode 535 提前结束！

训练全部结束！

开始策略评估 (5 轮)...
评估轮次  1 | 总得分:  500.0
评估轮次  2 | 总得分:  500.0
评估轮次  3 | 总得分:  500.0
评估轮次  4 | 总得分:  500.0
评估轮次  5 | 总得分:  500.0
```



#### 1.6.2 LunarLander

```text

==================== 训练启动: 2026-08-31 21:47:01 ====================
08-31 21:47:01: Episode    1 | Reward: -146.1 | 100-Avg: -146.1 | Actor Loss:  -0.0132 | Critic Loss:  14.1854
08-31 21:47:04: Episode   20 | Reward: -129.0 | 100-Avg: -194.9 | Actor Loss:  -0.0056 | Critic Loss:  12.1218
08-31 21:47:06: Episode   40 | Reward: -206.5 | 100-Avg: -183.7 | Actor Loss:  -0.0073 | Critic Loss:  11.0019
08-31 21:47:09: Episode   60 | Reward:   -5.5 | 100-Avg: -172.9 | Actor Loss:  -0.0165 | Critic Loss:  11.1656
08-31 21:47:11: Episode   80 | Reward: -287.9 | 100-Avg: -175.4 | Actor Loss:  -0.0238 | Critic Loss:  22.3882
08-31 21:47:14: Episode  100 | Reward: -132.9 | 100-Avg: -172.9 | Actor Loss:  -0.0315 | Critic Loss:   5.2201
略...
08-31 22:57:57: Episode 9640 | Reward:  251.1 | 100-Avg:  202.3 | Actor Loss:   0.0272 | Critic Loss:   3.2213
08-31 22:58:04: Episode 9660 | Reward:  260.8 | 100-Avg:  202.8 | Actor Loss:  -0.0978 | Critic Loss:   2.9577
08-31 22:58:09: Episode 9680 | Reward:   39.7 | 100-Avg:  205.2 | Actor Loss:   0.1348 | Critic Loss:  10.5038
08-31 22:58:14: Episode 9700 | Reward:  277.9 | 100-Avg:  209.8 | Actor Loss:  -0.1363 | Critic Loss:   4.1678
08-31 22:58:21: Episode 9720 | Reward:  291.3 | 100-Avg:  227.0 | Actor Loss:  -0.0832 | Critic Loss:   1.8917
08-31 22:58:30: Episode 9740 | Reward:  236.2 | 100-Avg:  228.8 | Actor Loss:  -0.0541 | Critic Loss:   2.5453
08-31 22:58:36: Episode 9760 | Reward:  222.8 | 100-Avg:  241.9 | Actor Loss:  -0.0116 | Critic Loss:   1.4360

```

之前的 REINFORCE 最好才189+，效果提升还是很明显的。




