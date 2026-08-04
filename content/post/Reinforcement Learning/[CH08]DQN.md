---
title: "[CH08]DQN"
description: "用经验回放和目标网络稳定深度 Q 学习"
date: 2026-07-31T13:03:13+08:00
lastmod: 2026-07-31T13:03:13+08:00
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

其实直接把 Q 学习 和 神经网络结合是不太好的，二者一个依赖强相关数据，一个依赖随机数据，那么我们可以通过一些策略来解决这个问题。



## 一、OpenAI Gym

OpenAI Gym 是一个开源库，提供了各种强化学习任务（环境），如：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785746478895_image.png)

### 1.1 OpenAI Gym 的基础知识

先安装：

```shell
pip install gymnasium
```

以游戏 CartPole 为例：

![image-20260803170320049](D:\TyporaPics\image-20260803170320049.png)



CartPole 的状态是四维向量：

| 位置 | 物理量     |
| ---- | ---------- |
| 0    | 小车位置   |
| 1    | 小车速度   |
| 2    | 杆的角度   |
| 3    | 杆的角速度 |

动作空间为 **Discrete(2)**：0 向左施力，1 向右施力。杆保持直立时每步通常得到奖励 1，一个 episode 到达失败条件或时间上限后结束。

写个脚本演示下：

```python
import numpy as np
import gymnasium as gym
import time 

env = gym.make('CartPole-v1', render_mode='human')

state, info = env.reset()
done = False

while not done:
    # 渲染画面
    env.render()
    
    # 随机选择一个动作
    action = env.action_space.sample()
    
    # 与环境交互
    next_state, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    state = next_state
    
    # 稍微暂停一下，否则画面运行太快，一闪而过
    time.sleep(0.05) 

env.close()

```

基本 1s 就似了：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785747930840_image.png)



## 二、DQN 原理

### 2.1 经验回放：把在线轨迹变成随机 minibatch

使用神经网络成功解决监督学习问题的案例有很多。但在 2013 年 DQN 发表之前，几乎没有使用神经网络成功解决强化学习问题的案例。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785751757681_image.png)

神经网络解决监督学习问题通常采用**随机批量梯度下降**，而 Q 学习进行状态转移的数据却存在强相关性，这就导致如果直接用神经网络来做 Q 学习就很容易出现过拟合。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785751753786_image.png)

经验回放的流程很简单，一次经验可以写成
$$
E_t=(S_t,A_t,R_t,S_{t+1}).
$$
**然后把经验保存到“缓冲区”中。在更新 Q 函数时，从缓冲区中随机取出经验数据并使用。**

程序还需知道是否终止，因此 buffer 实际保存
$$
(s,a,r,s',d).
$$
其中 $d=1$ 表示没有后续价值可 bootstrap。Replay buffer 是有限 FIFO 队列：

~~~python
self.buffer = deque(maxlen=buffer_size)
~~~

容量满时最早经验自动丢弃。每次更新不取最新一条经验，而是随机抽取 $B$ 条：

~~~python
data = random.sample(self.buffer, self.batch_size)
~~~

它带来三件事：

1. **减弱相邻样本相关性**：随机混合后，一个 batch 内的样本通常不再是一段连续轨迹；
2. **重复利用经验**：一条环境交互数据可在后续多次被抽到，提高数据效率；
3. **稳定梯度估计**：用 minibatch 平均，而不是让一条新样本完全决定参数更新。

这不意味着数据变成严格 IID，也不意味着探索已经充分。buffer 中的数据仍来自过去的行为策略，分布会随训练改变。

> **值得注意的是，经验回放只适用于 off-policy。**
>
> Q-learning 的 target 是 $\max_aQ(s',a)$，它学习贪婪目标策略的价值；产生数据的行为策略可以是 epsilon-greedy，因此旧数据仍可用于更新。SARSA 的 target 含当前行为策略实际选择的下一动作，不能不加处理地把旧策略数据拿来回放。

### 

###  2.2 经验回放的实现

```python
from collections import deque
import random
import numpy as np
import gymnasium as gym
import time 

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done
```



写个脚本测一下：

```python

env = gym.make('CartPole-v1', render_mode='human')
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):
    state, info = env.reset()
    done = False

    while not done:
        env.render()

        action = 0        
        next_state, reward, terminated, truncated, info = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        done = terminated or truncated
        
        time.sleep(0.05) 

env.close()

state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.get_batch()
print(state_batch.shape)      # (32, 4)
print(action_batch.shape)     # (32,)
print(reward_batch.shape)     # (32,)
print(next_state_batch.shape) # (32, 4)
print(done_batch.shape)       # (32,)

```



### 2.3 目标网络：固定一段时间的监督标签

只有 replay 还不够。若 target 仍由正在更新的 $Q_\theta$ 计算，每一步参数更新都会改变监督标签。DQN 维护两张相同结构、职责不同的网络：

| 网络                              | 用途                                             | 会被反向传播更新吗？ |
| --------------------------------- | ------------------------------------------------ | -------------------- |
| **qnet**，$Q_\theta$              | 选 epsilon-greedy 行动；预测当前 $Q_\theta(s,a)$ | 会                   |
| **qnet_target**，$Q_{\bar\theta}$ | 计算下一状态最大 Q 值，构造 TD target            | 不会                 |

标准 DQN target 为：

$$
y_i=r_i+(1-d_i)\gamma\max_aQ_{\bar\theta}(s'_i,a).
\tag{8.1}
$$
目标网络不会每一步跟随 online network。经过 $C$ 个间隔后硬复制：

$$
\bar\theta\leftarrow\theta.
$$


### 2.4 目标网络的实现

```python
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam

import matplotlib.pyplot as plt
import gymnasium as gym

from replay_buffer import ReplayBuffer

STATE_SIZE = 4
ACTION_SIZE = 2


class QNet(nn.Module):
    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, states: Tensor) -> Tensor:
        hidden = F.relu(self.l1(states))
        hidden = F.relu(self.l2(hidden))
        return self.l3(hidden)


class DQNAgent:
    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        *,
        gamma: float = 0.98,
        lr: float = 0.0005,
        epsilon: float = 0.1,
        buffer_size: int = 10_000,
        batch_size: int = 32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.action_size = action_size
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.qnet = QNet(state_size, action_size).to(self.device)
        self.qnet_target = QNet(state_size, action_size).to(self.device)
        self.qnet_target.eval()
        for parameter in self.qnet_target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            qs = self.qnet(state_tensor)
        return int(qs.argmax(dim=1).item())

    def compute_target(self, rewards: Tensor, next_states: Tensor, dones: Tensor) -> Tensor:
        """补充：计算目标 Q 值"""
        with torch.no_grad():
            next_qs = self.qnet_target(next_states)
            max_next_qs = next_qs.max(dim=1).values
            target = rewards + (1.0 - dones.float()) * self.gamma * max_next_qs
        return target

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float | None:
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        all_qs = self.qnet(states)
        current_q = all_qs.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        target = self.compute_target(rewards, next_states, dones)
        loss = F.mse_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_qnet(self) -> None:
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.qnet_target.eval()

```

​	写个脚本跑300次：

```python
episodes = 300
sync_interval = 20

env = gym.make('CartPole-v1') 
# env = gym.make('CartPole-v1', render_mode='human') 
agent = DQNAgent(device= 'cuda' if torch.cuda.is_available() else 'cpu')

reward_history = [0] * episodes
for episode in range(episodes):
    state, _ = env.reset()  
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated  
        agent.update(state, action, reward, next_state, done)
            
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()
	reward_history.append(total_reward)

env.close()

figure, axis = plt.subplots()
axis.set_xlabel("Episode")
axis.set_ylabel("Total Reward")
axis.plot(range(len(reward_history)), reward_history)
figure.tight_layout()
plt.show()  

```

曲线很夸张：

![image-20260803202613519](D:\TyporaPics\image-20260803202613519.png)

因为强化学习的训练曲线变动是很大的，我们不能仅凭一次结果评判，所以进行100次实验取平均：

```python
iters = 100
episodes = 300
sync_interval = 20

env = gym.make('CartPole-v1') 
# env = gym.make('CartPole-v1', render_mode='human') 
agent = DQNAgent(device= 'cuda' if torch.cuda.is_available() else 'cpu')

avg_history = [0] * episodes
for i in range(iters):
    for episode in range(episodes):
        state, _ = env.reset()  
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated  
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()

        avg_history[episode] += total_reward

for i, x in enumerate(avg_history):
    avg_history[i] = x / 100

env.close()

figure, axis = plt.subplots()
axis.set_xlabel("Episode")
axis.set_ylabel("Total Reward")
axis.plot(range(len(avg_history)), avg_history)
figure.tight_layout()
plt.show()  

```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785762362847_image.png)



然后我们让训练后的 Agent 基于 ε-greedy 来采取行动：

```python
agent.epsilon = 0 # 贪婪策略
state = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    total_reward += reward
    env.render()
print('Total Reward:', total_reward)
```

```
Total Reward: 116
```



## 三、DQN 和 Atari

CartPole 是小型教学环境。DQN 出名的重要背景是 Atari。

>   Atari 是一家游戏开发商，其业务涉及街机游戏、游戏机和个人电脑行业。在强化学习领域，Atari 制作的怀旧游戏软件被称为“Atari”。

由于 Atari 的游戏比前面的 CartPole 复杂的多，训练往往需要一天，所以这里就不针对相关代码展开了。



### 3.1 Atari 的游戏环境

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785761174298_image.png)

>   这个游戏还是蛮经典的，高中的时候同桌经常用 apple watch 玩。

Pong 说明：原始屏幕为 $210\times160\times3$ RGB 图像，单帧通常无法看出球的运动方向。

若图像只显示球当前位置、不显示速度方向，仅凭该帧无法完全决定未来转移分布。对代理而言，这更接近**部分可观测 MDP（POMDP）**。

我们把连续四帧堆叠为状态：

$$
s_t=(x_{t-3},x_{t-2},x_{t-1},x_t).
$$
**由于帧间位置变化携带粗略的运动信息**，**使输入更接近 Markov 状态**。



### 3.2 预处理

DQN 的论文中还在重叠帧之前进行了一些例行处理。具体包括
$$
210\times160\times3
\rightarrow \text{裁剪}
\rightarrow \text{灰度化}
\rightarrow \text{缩放}
\rightarrow \text{归一化到 }[0,1]
\rightarrow 4\text{ 帧堆叠}.
$$


### 3.3 CNN

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785761429931_image.png)

随后使用：
$$
4\times84\times84
\rightarrow \text{Conv}(32)
\rightarrow \text{Conv}(64)
\rightarrow \text{Conv}(64)
\rightarrow \text{Linear}(512)
\rightarrow \text{Linear}(|\mathcal A|)
$$
的 **CNN**。



### 3.4 其他技巧

我们可以通过GPU/TPU加速训练；

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785761559601_image.png)

将 epsilon 在前 100 万步从 1.0 线性降到 0.1

奖励裁剪到 $[-1,1]$。



## 四、DQN 的扩展

### 4.1 Double DQN：缓解最大化过估计

标准 DQN 使用

$$
r+(1-d)\gamma\max_aQ_{\bar\theta}(s',a).
$$
Double DQN 将“选动作”和“评估动作”拆给两张网络：

$$
r+(1-d)\gamma Q_{\bar\theta}
\left(s',\arg\max_aQ_\theta(s',a)\right).
\tag{8.2}
$$
**online network 用 argmax 选动作编号，target network 只评估那个动作的数值，从而缓解 max 对噪声偏大的倾向。**

>   普通 DQN max算子放大了 估计值 Q 所携带的误差，通过选出 Q 值最大的行动，可以使得训练更稳定。



### 4.2 优先级经验回放

均匀回放对每条经验一视同仁。优先级经验回放依据 TD error：

$$
\delta_i=\left|
r_i+(1-d_i)\gamma\max_aQ_{\bar\theta}(s'_i,a)
-Q_\theta(s_i,a_i)\right|,
\qquad
p_i=\frac{\delta_i}{\sum_k\delta_k}.
\tag{8.3}
$$
**直觉是：目前预测得很不准的样本更值得复习。**
$$
\tilde p_i=(|\delta_i|+\varepsilon_p)^\alpha,\qquad
p_i=\frac{\tilde p_i}{\sum_k\tilde p_k},
$$


### 4.3 Dueling DQN

引入 advantage function：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s),\qquad
Q^\pi(s,a)=V^\pi(s)+A^\pi(s,a).
\tag{8.4}
$$
**Dueling 网络先共享特征提取，再分为 value 分支与 advantage 分支。它适合很多动作好坏相近的状态：先判断“状态整体好不好”可能比独立估计每个动作更有效。**

