---
title: "[CH07]神经网络和Q学习"
description: "让 Q 函数从“查表”变成'计算'"
date: 2026-07-27T18:00:32+08:00
lastmod: 2026-07-27T18:00:32+08:00
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
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784273674617_image.png
---

<!--more-->



## 零、写在前面

上一章的 Q-learning 已经能在 3×4 GridWorld 中学出好策略。可是它把每一个状态—动作对都存为 Q 表中的一个独立格子：状态一多、状态是图像，或者动作与状态连续时，表就再也放不下。

本章做的是一件更根本的事：把表中的数 $Q(s,a)$，替换成由参数 $\theta$ 控制的函数 $Q_\theta(s,a)$，“如何让当前预测靠近 target”：

$$
\underbrace{Q(s,a)\leftarrow Q(s,a)+\alpha[T-Q(s,a)]}_{\text{第 6 章：直接改表中的一个格子}}
\quad\Longrightarrow\quad
\underbrace{\theta\leftarrow\theta-\eta\nabla_\theta
\bigl(T-Q_\theta(s,a)\bigr)^2}_{\text{第 7 章：通过梯度下降改网络参数}}.
$$
神经网络会让许多状态共享同一组参数，因此一次参数更新可能同时影响很多 $Q_\theta(s,a)$。这带来了函数近似和可扩展性，也带来了新的不稳定性。



## 一、Q 学习与神经网络

仍然从一条真实环境转移

$$
(S_t,A_t,R_{t+1},S_{t+1},\mathrm{done})
$$
构造 Q-learning target，但用神经网络去拟合当前动作的 Q 值。



### 1.1 神经网络的预处理：one-hot

本章 GridWorld 是 3×4，共 12 个格子。状态是坐标 $(y,x)$，我们将其转为独热编码：

```python
import numpy as np
def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :] # 由于是批量操作，因此添加了一个新轴
state = (2, 0)
x = one_hot(state)
print(x.shape) # (1, 12)
print(x) # [[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]
```



### 1.2 表示 Q 函数的神经网络

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785147718453_image.png)

我们实现第二种网络结构：

```python
class QNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(STATE_SIZE, 100)
        self.l2 = nn.Linear(100, ACTION_SIZE)

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.l1(state_vector))
        return self.l2(hidden)
```



### 1.3 神经网络和 Q 学习

Q 学习通过下面的式子更新 Q 函数：
$$
Q'(S_t,A_t)=Q(S_t,A_t)+\alpha
\left[
R_{t+1}+\gamma\max_aQ(S_{t+1},a)-Q(S_t,A_t)
\right].
\tag{7.3}
$$
这个式子沿着 $ R_{t+1}+\gamma\max_aQ_\theta(S_{t+1},a) $ 的方向更新 $ Q(S_t, A_t) $ 的值。

我们将其称为 target：
$$
T=R_{t+1}+\gamma\max_aQ_\theta(S_{t+1},a).
$$
因此 7.3 式可以写成下式：
$$
Q'(S_t,A_t)=Q(S_t,A_t)+\alpha
\left[
T-Q(S_t,A_t)
\right].
\tag{7.4}
$$


由于终止后没有未来回报可继续 bootstrap；但本次转移刚刚收到的即时奖励 $R_{t+1}$ 仍然保留。因此实际实现是：
$$
T=
\begin{cases}
R_{t+1},&\mathrm{done}=\mathrm{True},\\
R_{t+1}+\gamma\max_aQ_\theta(S_{t+1},a),&\mathrm{done}=\mathrm{False}.
\end{cases}
\tag{L7.1}
$$
由于现在不能执行“修改 $Q(s,a)$ 这个表格格子”，所以把 $T$ 当作回归标签，最小化

$$
L(\theta)=\bigl[T-Q_\theta(S_t,A_t)\bigr]^2.
\tag{L7.2}
$$
若将 $T$ 视作常数，梯度下降会使 $Q_\theta(S_t,A_t)$ 朝 $T$ 靠近：

$$
\nabla_\theta L
=-2\bigl[T-Q_\theta(S_t,A_t)\bigr]
\nabla_\theta Q_\theta(S_t,A_t).
$$


经过上面的探讨，我们进行 QLearningAgent 的实现：

```python
class QLearningAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = ACTION_SIZE
        self.qnet = QNet()
        self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state_vector: torch.Tensor) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))

        with torch.no_grad():
            qs = self.qnet(state_vector)
        return int(qs.argmax(dim=1).item())

    def compute_target(
        self,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> torch.Tensor:
        with torch.no_grad():
            if done:
                next_q = torch.zeros(
                    1,
                    dtype=next_state.dtype,
                    device=next_state.device,
                )
            else:
                next_q = self.qnet(next_state).max(dim=1).values
            return reward + self.gamma * next_q

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> float:
        target = self.compute_target(reward, next_state, done)

        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
```



然后写个训练函数：

```python
def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    env = GridWorld()
    agent = QLearningAgent()
    loss_history = []

    for _ in range(EPISODES):
        state = one_hot(env.reset())
        total_loss, cnt = 0.0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)

            loss = agent.update(state, action, float(reward), next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state

            if cnt >= MAX_STEPS_PER_EPISODE and not done:
                raise RuntimeError(
                    "An episode exceeded MAX_STEPS_PER_EPISODE without "
                    "reaching the goal."
                )

        loss_history.append(total_loss / cnt)

    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.plot(range(len(loss_history)), loss_history)
    plt.show()

    q_values = {}
    with torch.no_grad():
        for state in env.states():
            for action in env.action_space:
                q = agent.qnet(one_hot(state))[:, action]
                q_values[state, action] = float(q.item())
    env.render_q(q_values)


if __name__ == "__main__":
    main()

```

结果不是最优，但是效果还可以：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785151309279_image.png)

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785151325758_image.png)

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785151328936_image.png)



## 二、小结

只要懂了 贝尔曼方程、Q-learning 的原理，将其与神经网络结合就不算太困难。那么就可以进一步增加状态和行动规模。



