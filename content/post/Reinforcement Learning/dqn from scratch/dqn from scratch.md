---
title: "Dqn From Scratch"
description: "别笑，你玩你也过不了第一关"
date: 2026-08-26T20:19:30+08:00
lastmod: 2026-08-26T20:19:30+08:00
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
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1787747613323_image.png
---

<!--more-->



## 零、写在前面

偶尔写点小玩具也是蛮有趣的。~~才不是因为我人工玩 flapy bird 过 7 个口就似了~~

源码：https://github.com/Equinox-2003/DQN-From-Scratch



## 一、FlappyBird Environment Setup

打开gymnasium.farama找一下 flappy bird 的环境：https://gymnasium.farama.org/environments/third_party_environments/

最后选了这个：https://github.com/markub3327/flappy-bird-gymnasium

因为他这个 observation 可以选择，一个是神秘传感器信号看不懂，一个是各种位置信息，相对来说简单一点。

>   有的环境直接给的RGB图像，会麻烦一点。

安装环境：

```shell
pip install flappy-bird-gymnasium
```

然后可以试着运行玩一下：

```shell
flappy_bird_gymnasium
```

>   我们会看到命令行有一些输出，含义这个环境的 github 仓库有写：
>
>   1.  option
>
>   -   the last pipe's horizontal position
>   -   the last top pipe's vertical position
>   -   the last bottom pipe's vertical position
>   -   the next pipe's horizontal position
>   -   the next top pipe's vertical position
>   -   the next bottom pipe's vertical position
>   -   the next next pipe's horizontal position
>   -   the next next top pipe's vertical position
>   -   the next next bottom pipe's vertical position
>   -   player's vertical position
>   -   player's vertical velocity
>   -   player's rotation
>
>   **Action space**
>
>   -   0 - **do nothing**
>   -   1 - **flap**
>
>   **Rewards**
>
>   -   +0.1 - **every frame it stays alive**
>   -   +1.0 - **successfully passing a pipe**
>   -   -1.0 - **dying**
>   -   −0.5 - **touch the top of the screen**

copy 一下官方的示例运行代码：

```python
import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)	# use_lidar 用不到

obs, _ = env.reset()
while True:
    # Next action: 
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()

```

也可以运行其他的游戏，比如 CartPole：

```python
env = gymnasium.make("CartPole-v1", render_mode="human")
```



## 二、Implement DQN with Pytorch

首先我们需要写一个 net 来做 Q-learning，输入是state，输出是 action：

![image-20260826211750140](D:\TyporaPics\image-20260826211750140.png)

简单搭建一个网络：

```python
# dqn.py

import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

可以跑一下：

```python
# dqn.py

if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)

```

```text
[ 0.0046,  0.1992],
[ 0.0690,  0.1556],
[-0.0180, -0.0007],
[ 0.0200, -0.0088],
[-0.0294, -0.0324],
[-0.2434, -0.2045],
[ 0.1420,  0.1645]], grad_fn=<AddmmBackward0>)
```



然后再写一个 agent 类用于后续包装我们的运行、训练等逻辑：

```python
# agent.py

import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def run(self, is_training=True, render=False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        obs, _ = env.reset()
        while True:
            # Next action: 
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing:
            next_obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break            
            
            # Checking if the player is still alive
            if terminated:
                break

        env.close()

```



## 三、Implement Experience Replay

在深度强化学习中，**经验回放（Experience Replay）** 是 DQN 能成功训练的最核心创新之一（DeepMind 2013/2015）。

如果不使用经验回放（即采用传统的“在线单步更新”），神经网络训练极易发散、崩溃。引入经验回放主要有以下 **三大核心原因**：

**1. 打破数据间的时间相关性（满足 I.I.D. 假设）**

* **传统在线学习的痛点**：
  强化学习采集的数据是一条**连续的时间序列**（比如小车向右倾斜，接下来几步大概率还是向右倾斜）。相邻两步的数据 $(s_t, a_t)$ 和 $(s_{t+1}, a_{t+1})$ 之间存在**极强的自相关性（Temporal Correlation）**。
* **对神经网络的危害**：
  现代深度学习的优化算法（如 SGD/Adam）是建立在**样本独立同分布（I.I.D.，Independent and Identically Distributed）** 的假设基础之上的。如果直接用强相关的数据连续喂给网络，网络很容易发生过拟合、震荡甚至权重爆炸。
* **经验回放的作用**：
  把经历过的转移元组 $(s_t, a_t, r_t, s_{t+1}, \text{done})$ 存入回放池（Buffer），训练时**随机均匀采样一个 Batch**。这打乱了时序关联，极大地消除了样本间的相关性，让数据分布更接近 I.I.D.。

**2. 提高数据利用效率（Sample Efficiency）**

* **传统在线学习的痛点**：
  每一条转移样本在被网络反向传播更新一次后就会被直接丢弃。
* **经验回放的作用**：
  在实际环境中（或复杂游戏中），某些关键事件（比如“顺利穿过一条水管”、“避开障碍物”）是非常稀少和宝贵的。
  通过经验回放池，**一条样本可以被反复多次采样并用于更新网络**，大幅提升了样本的利用效率，减少了与环境交互的总步数。

**3. 防止灾难性遗忘与平滑数据分布**

* **传统在线学习的痛点**：
  随着智能体探索的深入，它会进入新的状态空间区域。如果只拿最新收集到的经验来训练，神经网络会迅速“遗忘”之前在其他状态下学到的知识（**灾难性遗忘 Catastrophic Forgetting**）。
* **经验回放的作用**：
  回放池中保留了一定容量的历史记忆（既包含之前的策略行为，也包含当前的策略行为）。随机采样时，网络同时在学习“以前的状态”和“现在的状态”，**平滑了训练数据的分布漂移（Non-Stationary Distribution）**，使训练过程更加平稳。

>   Q：为什么 DQN 可以用经验回放，而某些算法不行？
>
>   A：因为 DQN 的本质是 **Q-Learning**，它是一种 **Off-Policy（离策略）** 算法：
>
>   * 它的更新目标是基于贝尔曼最优方程 $\max_{a'} Q(s', a')$，它计算目标值时**并不依赖于当初产生这个样本时所采用的具体策略**。
>   * 因此，即使样本是**过去几百步甚至旧版本的网络策略**采集来的，DQN 依然可以用它来正确更新当前的网络。
>   * （反之，如标准的 REINFORCE、A2C 等 **On-Policy** 算法，由于梯度推导必须基于当前策略产生的样本，就无法直接使用普通的经验回放池）。

下面写一个 ReplayMemor 类，其实就是一个 容量固定的deque：

```python
from collections import deque
import random

class ReplayMemory:
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
    
```



我们在之前的 agent 循环里面加入对应逻辑：

```python
class Agent:
    def run(self, is_training=True, render=False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        reward_per_episode = []

        if is_training:
            memory = ReplayMemory(10000)

        for episode in itertools.count():
            state, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                # Next action: 
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, truncated, info = env.step(action)

                # is done?
                done = terminated or truncated

                # acc reward
                episode_reward += reward

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                state = new_state

        reward_per_episode.append(episode_reward)

```



因为后面可能还要加入 epsilon-greedy 等策略，那么就需要设置一些超参数，这里开一个 yml 来存：

```yml
cartpole1:
  env_id: CartPole-v1
  replay_memory_size: 100000
  mini_batch_size: 32
  epsilon_init: 1
  epsilon_decay: 0.9995
  epsilon_min: 0.05
```



然后修改相应逻辑：

```python
import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from  experience_replay import ReplayMemory
import itertools
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_set[hyperparameter_set]
        
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        

    def run(self, is_training=True, render=False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        reward_per_episode = []

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

        for episode in itertools.count():
            state, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                # Next action: 
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, truncated, info = env.step(action)

                # is done?
                done = terminated or truncated

                # acc reward
                episode_reward += reward

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                state = new_state

        	reward_per_episode.append(episode_reward)

```



## 四、Implement Epsilon Greedy

Epsilon Greedy 就是要么随机决策，要么网络决策。

把 yml 配置读出来然后改几行就行了

```python
import itertools
import random
import flappy_bird_gymnasium
import gymnasium
import torch
import yaml
from dqn import DQN
from experience_replay import ReplayMemory

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_set[hyperparameter_set]
        
        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']

    def run(self, is_training=True, render=False):
        env = gymnasium.make(self.env_id, render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        reward_per_episode = []
        epsilon_history = []
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

        for episode in itertools.count():
            # state: numpy array
            state, _ = env.reset() 
            done = False
            episode_reward = 0.0

            while not done:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    # tensor for dqn
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = policy_dqn(state_tensor).argmax().item()

                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # 这里注意不要存 tensor，不然常驻显存跑几轮直接炸了 
                if is_training:
                    memory.append((state, action, reward, new_state, terminated))

                state = new_state

            reward_per_episode.append(episode_reward)

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

            if episode % 10 == 0:
                print(f"Episode {episode} | Reward: {episode_reward} | Epsilon: {epsilon:.4f}" if is_training else f"Episode {episode} | Reward: {episode_reward}")

if __name__ == '__main__':
    agent = Agent('cartpole1')
    agent.run(is_training=True, render=False)

```



## 五、Implement Target Network

dqn 引入目标网络的原因很简单，若 target 仍由正在更新的策略网络计算，每一步参数更新都会改变监督标签。

因此我们会额外创建一个 dqn 网络，作为目标网络，隔一段时间把策略网络的参数同步给目标网络。

```python
    def run(self, is_training=True, render=False):
		# ...
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        for episode in itertools.count():
		   # ...

            while not done:
			   # ...

                if is_training:
                    memory.append((state, action, reward, new_state, terminated))
                    step_count += 1

                    if len(memory) >= self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                    if step_count % self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

                state = new_state

			# ...

```

对应的参数加载：

```python
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_set[hyperparameter_set]
        
        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
```

hyperparameters.yml：

```python
cartpole1:
  env_id: CartPole-v1
  replay_memory_size: 100000
  mini_batch_size: 32
  epsilon_init: 1
  epsilon_decay: 0.9995
  epsilon_min: 0.05
  network_sync_rate: 75
  learning_rate_a: 0.001
  discount_factor_g: 0.99

```



我们还需要一个参数更新逻辑：

```python
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, rewards, new_states, terminations = zip(*mini_batch)

        # states: [B, state_dim]
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        # actions: [B, 1]
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        # rewards: [B, 1]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        # new_states: [B, state_dim]
        new_states_tensor = torch.tensor(new_states, dtype=torch.float32, device=device)
        # terminations: [B, 1] (True/False 转为 1.0/0.0)
        terminations_tensor = torch.tensor(terminations, dtype=torch.float32, device=device).unsqueeze(1)

        # 计算当前 Q 值: Q(s, a)
        # policy_dqn(states_tensor) 得到 [B, action_dim]
        # .gather(1, actions_tensor) 挑出实际执行的 action 对应的 Q 值，得到 [B, 1]
        current_q = policy_dqn(states_tensor).gather(1, actions_tensor)

        # 计算目标 Q 值: r + gamma * max_a' Q_target(s', a') * (1 - terminated)
        with torch.no_grad():
            # max(1)[0] 取出最大值，返回 [B]，再 unsqueeze(1) 变成 [B, 1]
            max_next_q = target_dqn(new_states_tensor).max(1)[0].unsqueeze(1)
            # 如果 terminated=True，target_q 就只等于 reward
            target_q = rewards_tensor + (1.0 - terminations_tensor) * self.discount_factor_g * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```



## 六、Test DQN

### 6.1 脚本完善

先从简单任务开始，CartPole 相比 flappy bird 训练的会快很多。

先补充一下超参数：

```yaml
cartpole1:
  env_id: CartPole-v1
  replay_memory_size: 100000
  mini_batch_size: 64
  epsilon_init: 1
  epsilon_decay: 0.9995
  epsilon_min: 0.01
  network_sync_rate: 100
  learning_rate_a: 0.001
  discount_factor_g: 0.99
  stop_on_reward: 500
  fc1_nodes: 128

flappybird1:
  env_id: FlappyBird-v0
  replay_memory_size: 100000
  mini_batch_size: 32
  epsilon_init: 1
  epsilon_decay: 0.99995
  epsilon_min: 0.05
  network_sync_rate: 10
  learning_rate_a: 0.0001
  discount_factor_g: 0.99
  stop_on_reward: 100000
  fc1_nodes: 512
  env_make_params:
    use_lidar: False

```

然后加入日志、绘图、命令行参数解析逻辑：

```python
import argparse
from datetime import datetime
import itertools
import os
import random
import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import yaml
from dqn import DQN
from experience_replay import ReplayMemory

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_set[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params', {})

        # Neural Network
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f'{start_time.strftime(DATE_FORMAT)}: Training starting...'
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions, hidden_dim=self.fc1_nodes).to(device)
        reward_per_episode = []

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions, hidden_dim=self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []

            best_reward = -float('inf')
        else:
            # 评估模式加载权重
            if not os.path.exists(self.MODEL_FILE):
                raise FileNotFoundError(f"Model checkpoint not found at {self.MODEL_FILE}. Please train first!")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                # 动作选择
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = policy_dqn(state_tensor).argmax().item()

                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                if is_training:
                    memory.append((state, action, reward, new_state, terminated))
                    step_count += 1

                    if len(memory) >= self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                    if step_count % self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

                state = new_state

            reward_per_episode.append(episode_reward)

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # 记录最高分并保存最佳模型
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    
                    self.save_graph(reward_per_episode, epsilon_history)

                # 达标停止条件
                if episode_reward >= self.stop_on_reward:
                    print(f"Goal reached! Solved in {episode} episodes with reward {episode_reward}!")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    self.save_graph(reward_per_episode, epsilon_history)
                    break
            else:
                print(f"Test Episode {episode} | Reward: {episode_reward}")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, rewards, new_states, terminations = zip(*mini_batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        new_states_tensor = torch.tensor(np.array(new_states), dtype=torch.float32, device=device)
        terminations_tensor = torch.tensor(terminations, dtype=torch.float32, device=device).unsqueeze(1)

        # 当前 Q 值
        current_q = policy_dqn(states_tensor).gather(1, actions_tensor)

        # 目标 Q 值
        with torch.no_grad():
            max_next_q = target_dqn(new_states_tensor).max(1)[0].unsqueeze(1)
            target_q = rewards_tensor + (1.0 - terminations_tensor) * self.discount_factor_g * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1, figsize=(10, 4))
        plt.clf() # 清除旧图像，避免重复重叠画图

        # 100 轮滑动平均奖励
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        plt.subplot(121)
        plt.title('Mean Rewards (Last 100)')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.plot(mean_rewards, label='Mean Reward')
        plt.grid(True)

        plt.subplot(122)
        plt.title('Epsilon Decay')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.plot(epsilon_history, color='orange', label='Epsilon')
        plt.grid(True)

        plt.tight_layout()
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Set of hyperparameters to run, defined in hyperparameters.yml')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
```



### 6.2 CartPole

训练一下：

```text
(Dqnenv) PS D:\gitRepo\DQN-From-Scratch> python agent.py cartpole1 --train                                                           
08-27 20:13:31: Training starting...
08-27 20:13:33: New best reward 15.0 (+nan%) at episode 0, saving model...
08-27 20:13:33: New best reward 26.0 (+73.3%) at episode 2, saving model...
08-27 20:13:33: New best reward 28.0 (+7.7%) at episode 3, saving model...
08-27 20:13:34: New best reward 36.0 (+28.6%) at episode 4, saving model...
08-27 20:13:34: New best reward 42.0 (+16.7%) at episode 13, saving model...
08-27 20:13:35: New best reward 55.0 (+31.0%) at episode 14, saving model...
08-27 20:13:38: New best reward 57.0 (+3.6%) at episode 81, saving model...
08-27 20:13:39: New best reward 91.0 (+59.6%) at episode 100, saving model...
08-27 20:13:46: New best reward 101.0 (+11.0%) at episode 234, saving model...
08-27 20:13:52: New best reward 106.0 (+5.0%) at episode 349, saving model...
08-27 20:14:04: New best reward 121.0 (+14.2%) at episode 492, saving model...
08-27 20:14:09: New best reward 141.0 (+16.5%) at episode 557, saving model...
08-27 20:14:16: New best reward 149.0 (+5.7%) at episode 620, saving model...
08-27 20:14:17: New best reward 169.0 (+13.4%) at episode 628, saving model...
08-27 20:14:27: New best reward 173.0 (+2.4%) at episode 711, saving model...
08-27 20:14:34: New best reward 176.0 (+1.7%) at episode 777, saving model...
08-27 20:14:36: New best reward 196.0 (+11.4%) at episode 787, saving model...
08-27 20:14:47: New best reward 238.0 (+21.4%) at episode 872, saving model...
08-27 20:15:12: New best reward 259.0 (+8.8%) at episode 1038, saving model...
08-27 20:15:31: New best reward 262.0 (+1.2%) at episode 1140, saving model...
08-27 20:15:34: New best reward 296.0 (+13.0%) at episode 1155, saving model...
08-27 20:15:40: New best reward 297.0 (+0.3%) at episode 1183, saving model...
08-27 20:15:45: New best reward 322.0 (+8.4%) at episode 1201, saving model...
08-27 20:15:55: New best reward 421.0 (+30.7%) at episode 1246, saving model...
08-27 20:19:46: New best reward 500.0 (+18.8%) at episode 2014, saving model...
Goal reached! Solved in 2014 episodes with reward 500.0!
```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1787833340922_image.png)

因为我们训练目标设置的比较低，然后停止条件也比较简单，所以很快就完成了训练，但是实际运行效果还是很不错的：

```text
(Dqnenv) PS D:\gitRepo\DQN-From-Scratch> python agent.py cartpole1        
Test Episode 0 | Reward: 500.0
Test Episode 1 | Reward: 500.0
Test Episode 2 | Reward: 500.0
Test Episode 3 | Reward: 500.0
...
```



### 6.3 FlappyBird

然后试一下难度更高的 FlappyBird，考虑到这个难度比较高，我们需要更大的模型，所以索性把网络加了一层：

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 加多一层
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

然后参数调一调：

```yaml
cartpole1:
  env_id: CartPole-v1
  replay_memory_size: 30000    
  mini_batch_size: 128     
  epsilon_init: 1.0
  epsilon_decay: 0.992         
  epsilon_min: 0.01
  network_sync_rate: 200   
  learning_rate_a: 0.0005     
  discount_factor_g: 0.99
  stop_on_reward: 490        
  fc1_nodes: 128

flappybird1:
  env_id: FlappyBird-v0
  replay_memory_size: 100000
  mini_batch_size: 128 
  epsilon_init: 1.0
  epsilon_decay: 0.999
  epsilon_min: 0.02
  network_sync_rate: 500       
  learning_rate_a: 0.0003    
  discount_factor_g: 0.99
  stop_on_reward: 1000          
  fc1_nodes: 256
  env_make_params:
    use_lidar: False
```



简单跑了一会：

```text
(Dqnenv) PS D:\gitRepo\DQN-From-Scratch> python agent.py flappybird1 --train
08-27 21:42:25: Training starting...
08-27 21:42:26: New best reward -7.5 (N/A) at episode 0, saving model...
08-27 21:42:27: New best reward -6.9 (+8.0%) at episode 1, saving model...
08-27 21:42:27: New best reward -5.7 (+17.4%) at episode 3, saving model...
08-27 21:42:29: New best reward -2.7 (+52.6%) at episode 17, saving model...
08-27 21:42:37: New best reward -2.1 (+22.2%) at episode 74, saving model...
08-27 21:42:38: New best reward -0.3 (+85.7%) at episode 79, saving model...
08-27 21:43:16: New best reward 3.9 (+1400.0%) at episode 337, saving model...
08-27 21:44:12: New best reward 6.0 (+53.8%) at episode 708, saving model...
08-27 21:44:48: New best reward 6.1 (+1.7%) at episode 935, saving model...
08-27 21:45:28: New best reward 6.6 (+8.2%) at episode 1179, saving model...
08-27 21:47:17: New best reward 6.8 (+3.0%) at episode 1860, saving model...
08-27 21:47:39: New best reward 8.4 (+23.5%) at episode 1986, saving model...
08-27 21:48:14: New best reward 9.5 (+13.1%) at episode 2176, saving model...
08-27 21:50:00: New best reward 12.9 (+35.8%) at episode 2781, saving model...
08-27 21:50:03: New best reward 15.4 (+19.4%) at episode 2791, saving model...
08-27 21:51:49: New best reward 20.3 (+31.8%) at episode 3298, saving model...
08-27 21:52:55: New best reward 22.4 (+10.3%) at episode 3568, saving model...
08-27 21:53:05: New best reward 27.1 (+21.0%) at episode 3595, saving model...
08-27 21:53:34: New best reward 31.9 (+17.7%) at episode 3683, saving model...
08-27 21:54:38: New best reward 34.4 (+7.8%) at episode 3874, saving model...
08-27 21:55:31: New best reward 37.1 (+7.8%) at episode 4015, saving model...
08-27 21:55:37: New best reward 40.9 (+10.2%) at episode 4024, saving model...
08-27 21:55:46: New best reward 45.9 (+12.2%) at episode 4043, saving model...
08-27 21:56:17: New best reward 64.4 (+40.3%) at episode 4114, saving model...
08-27 21:56:58: New best reward 96.9 (+50.5%) at episode 4196, saving model...
08-27 22:02:45: New best reward 115.9 (+19.6%) at episode 4917, saving model...

```

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1787839987074_image.png)

训练太慢了，就不再往下跑了。

实际跑出来效果其实一般，十把里面有7、8把都是上来就死，剩下的过三四个管子。

其实DQN 是有一些痛点的，后续有了很多 DQN 改进算法，能够在不显著增加计算量的情况下，提高训练稳定性，加速收敛。

等我们改进 DQN 后再进行更长时间的训练。



## 七、Double DQN

**Double DQN（Double Deep Q-Network，简称 DDQN）** 是由 DeepMind 的 Hado van Hasselt 等人在 2015 年提出的 DQN 经典改进算法（论文：《*Deep Reinforcement Learning with Double Q-learning*》）。

它的核心目的非常明确：**解决标准 DQN 在训练过程中对动作价值（Q 值）的“过估计（Overestimation Bias）”问题**，从而提升算法的稳定性和性能。

### 7.1 背景与动机：DQN 的“过估计”问题

#### 7.1.1 什么是过估计？
在标准的 DQN 中，计算目标 Q 值（Target Q-value）的公式为：
$$Y_t^{\text{DQN}} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$$
其中：

* $\theta^-$ 是目标网络（Target Network）的参数。
* $\max_{a'} Q(S_{t+1}, a'; \theta^-)$ 表示在下一状态 $S_{t+1}$ 下，选取目标网络估计出的最大 Q 值。



#### 7.1.2 为什么标准 DQN 会产生过估计？

由于神经网络在拟合 Q 函数时必然存在**估计误差（噪声）**，假设状态 $S_{t+1}$ 下所有动作的真实价值都差不多，但因为噪声的存在，某些动作的估计值可能偏高，某些动作偏低。

$\max$ 操作符会**贪婪地选取最大值**。根据统计学性质：
$$\mathbb{E}\left[ \max(X_1, X_2, \dots, X_n) \right] \ge \max\left( \mathbb{E}[X_1], \mathbb{E}[X_2], \dots, \mathbb{E}[X_n] \right)$$
这意味着：**最大值的期望总是大于等于期望的最大值**。

每次更新都选取被高估的那个动作，导致 target 值系统性偏大；随着贝尔曼迭代的自举（Bootstrapping），这种过估计会被不断传递并累积放大。



#### 7.1.3 过估计的危害
* **相对优劣被误导**：如果所有状态和动作的过估计是均匀的，问题还不算严重；但实验表明，过估计往往是**不均匀**的，导致 Agent 误以为某个较差的动作很好，从而学到次优策略。
* **训练震荡/发散**：Q 值脱离真实回报无限制膨胀，破坏训练稳定性。



### 7.2 核心思想：解耦动作选择与价值评估

Double DQN 的灵感来源于传统的 **Double Q-learning（van Hasselt, 2010）**，其核心思想是：

> **“将「选择哪个动作」与「评估这个动作有多好」这两个步骤解耦，由不同的网络分别负责。”**

* **动作选择（Action Selection）**：选出当前认为最好的动作。
* **动作评估（Action Evaluation）**：评估该动作的实际价值。

如果一个网络因为噪声高估了动作 A，另一个独立的网络大概率不会对动作 A 也产生相同的正向误差，从而大幅降低了过估计的概率。



### 7.3 算法原理

#### 7.3.1 传统 DQN vs Double DQN

| 算法           | 目标值计算公式（Target Q）                                   |
| :------------- | :----------------------------------------------------------- |
| **标准 DQN**   | $Y_t^{\text{DQN}} = R_{t+1} + \gamma Q\left(S_{t+1}, \color{red}{\arg\max_{a'} Q(S_{t+1}, a'; \theta^-)}; \color{red}{\theta^-}\right)$ |
| **Double DQN** | $Y_t^{\text{DDQN}} = R_{t+1} + \gamma Q\left(S_{t+1}, \color{blue}{\arg\max_{a'} Q(S_{t+1}, a'; \theta)}; \color{green}{\theta^-}\right)$ |



#### 7.3.2 Double DQN 的 Target 计算

Double DQN 将 Target 的计算拆分成两步：

1. **第一步：主网络（Online Network, 参数 $\theta$）负责【选动作】**
   $$a^* = \arg\max_{a'} Q(S_{t+1}, a'; \theta)$$
   *看当前正在更新的主网络认为在下一状态 $S_{t+1}$ 下，哪个动作 $a^*$ 最优。*

2. **第二步：目标网络（Target Network, 参数 $\theta^-$）负责【打分/评估】**
   $$Q_{\text{target\_val}} = Q(S_{t+1}, a^*; \theta^-)$$
   将选出的动作 $ a ^ * $ 送入目标网络，计算该动作的 Q 值。

3. **最终组合成目标值**：
   $$Y_t^{\text{DDQN}} = R_{t+1} + \gamma Q(S_{t+1}, a^*; \theta^-)$$

>   原论文给出了这样做更优的严格数学证明，这里就不展开说了，反正主要就是证明：Double DQN 的价值估计期望**永远不会超过真实最优价值**。它不仅彻底消除了单估计器的正向过估计（Positive Bias），甚至在动作选错时表现为轻微的**低估（Underestimation）**。



### 7.3.3 算法流程

1. 初始化主网络参数 $\theta$ 和目标网络参数 $\theta^- \leftarrow \theta$，初始化经验回放池 $\mathcal{D}$。
2. 在环境中交互，通过 $\epsilon$-greedy 策略采集样本 $(S_t, A_t, R_{t+1}, S_{t+1}, \text{done})$ 存入经验回放池。
3. 从经验池中随机采样一个批次（Batch）的数据。
4. **计算 DDQN 目标值**：
   * 用主网络预测下一状态的最大动作：$a^* = \arg\max_{a'} Q(S_{t+1}, a'; \theta)$
   * 用目标网络评估该动作价值并计算 Target：$Y = R_{t+1} + \gamma (1-\text{done}) Q(S_{t+1}, a^*; \theta^-)$
5. **计算 Loss 并更新主网络**：
   $$\mathcal{L}(\theta) = \mathbb{E}\left[ \left( Y - Q(S_t, A_t; \theta) \right)^2 \right]$$
   使用梯度下降更新主网络参数 $\theta$。
6. 定期同步参数给目标网络（硬更新 $\theta^- \leftarrow \theta$ 或软更新 $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$）。



### 7.3.4 代码改进

我们在之前的实现上加入 Double DQN 的逻辑。

我们加一个超参数 enable_double_dqn: True

然后在optimize 那里多写个 if-else 加上 double dqn 的逻辑：

```python
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, rewards, new_states, terminations = zip(*mini_batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        new_states_tensor = torch.tensor(np.array(new_states), dtype=torch.float32, device=device)
        terminations_tensor = torch.tensor(terminations, dtype=torch.float32, device=device).unsqueeze(1)

        # 当前 Q 值
        current_q = policy_dqn(states_tensor).gather(1, actions_tensor)

        # 目标 Q 值
        with torch.no_grad():
            if self.enable_double_dqn:
                # 1. 主网络 (policy_dqn) 在【下一状态】选出最优动作 a*
                best_actions = policy_dqn(new_states_tensor).argmax(dim=1, keepdim=True)
                
                # 2. 目标网络 (target_dqn) 在【下一状态】评估该动作 a* 的 Q 值
                next_q_values = target_dqn(new_states_tensor).gather(dim=1, index=best_actions)
                
                # 3. 计算 Double DQN 目标值
                target_q = rewards_tensor + (1.0 - terminations_tensor) * self.discount_factor_g * next_q_values
            else:
                # 标准 DQN
                max_next_q = target_dqn(new_states_tensor).max(1)[0].unsqueeze(1)
                target_q = rewards_tensor + (1.0 - terminations_tensor) * self.discount_factor_g * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=10.0) # 梯度裁剪防爆炸
        self.optimizer.step()
```



简单训一个 CartPole：

```text
(Dqnenv) PS D:\gitRepo\DQN-From-Scratch> python agent.py cartpole1 --train
08-28 01:28:24: Training starting...
08-28 01:28:26: New best reward 32.0 (N/A) at episode 0, saving model...
08-28 01:28:26: New best reward 39.0 (+21.9%) at episode 1, saving model...
08-28 01:28:27: New best reward 66.0 (+69.2%) at episode 9, saving model...
08-28 01:28:30: New best reward 113.0 (+71.2%) at episode 43, saving model...
08-28 01:28:32: New best reward 171.0 (+51.3%) at episode 59, saving model...
08-28 01:28:33: New best reward 223.0 (+30.4%) at episode 65, saving model...
08-28 01:28:36: New best reward 232.0 (+4.0%) at episode 81, saving model...
08-28 01:28:40: New best reward 269.0 (+15.9%) at episode 97, saving model...
08-28 01:29:03: New best reward 349.0 (+29.7%) at episode 242, saving model...
08-28 01:29:25: New best reward 402.0 (+15.2%) at episode 309, saving model...
08-28 01:29:28: New best reward 500.0 (+24.4%) at episode 312, saving model...
Goal reached! Solved in 316 episodes with 5-episode average reward: 500.00!
```

然后玩了一下发现有个问题，虽然游戏能持续很久，但是它的左右摇摆幅度过于大了，导致很容易暴毙。

同样放到 flappy bird 里面，训练出来的策略很可能是在两个管子中间的安全地带，不怎么操作，快到下边管子了猛地往上跳，我们不希望模型学到这样的策略，那么怎么解决呢？——**Dueling DQN**。



## 八、Dueling DQN

**Dueling DQN（Dueling Deep Q-Network）** 是由 DeepMind 的 Ziyu Wang 等人在 2016 年提出的经典改进算法（论文：《*Dueling Network Architectures for Deep Reinforcement Learning*》，获得 ICML 2016 最佳论文之一）。

如果说 **Double DQN 改进的是算法的“目标计算方式（Loss 目标值）”**，那么 **Dueling DQN 改进的则是“神经网络的内部骨架（网络结构 Architecture）”**。



### 8.1 核心动机：为什么需要 Dueling 架构？

在标准 DQN 中，神经网络直接输出每一个动作对应的 $Q(s, a)$[1]。然而在很多实际场景中存在一个普遍现象：

> **“在很多状态下，环境本身的优劣（状态价值 $V$）远比你选哪个具体动作（动作优势 $A$）重要得多；甚至在某些状态下，无论选什么动作，对结果几乎没有影响。”**

>   例子：
>
>   1. **开赛车游戏**：
>      * **直道无障碍时**：不管是微调方向盘、稍微靠左还是靠右，车子都能顺利向前开。此时状态本身很好（$V(s)$ 很高），动作的选择并不致命。
>      * **即将撞墙时（极度危险）**：此时**选什么动作至关重要**（左转能活，不转就死）。
>   2. **Flappy Bird（你刚才的代码环境）**：
>      * 小鸟在两根水管正中间正常下落时，无论这一帧跳不跳，都不会马上死。
>      * 只有当小鸟逼近下边缘或上边缘的临界瞬间，跳跃动作的优劣才真正体现出来。
>

**标准 DQN 的缺陷：**

在标准 DQN 中，要评估某个状态 $s$ 下的所有动作，必须**把每个动作都探索、更新一遍**，才能学好这些 $Q(s, a)$。如果某些动作很少被执行，它们对应的 $Q$ 值就很难被准确更新。

**Dueling DQN 的思路**：将 $Q$ 值拆解为两部分分别学习：
1. **状态价值 $V(s)$**：当前状态到底有多好（与具体动作无关）。
2. **动作优势 $A(s, a)$**：在当前状态下，选择动作 $a$ 相比于其他动作有多大的相对优势。

这样一来，**即便只执行了一个动作，网络也能更新对当前状态整体价值 $V(s)$ 的估计**，大幅提高了样本利用效率。



### 8.2  优势函数（Advantage Function）

在强化学习理论中，状态价值 $V^\pi(s)$、动作价值 $Q^\pi(s, a)$ 和**优势函数 $A^\pi(s, a)$** 的定义如下：

* 状态价值：$V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$
* 优势函数：
  $$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

由定义移项可得：
$$Q^\pi(s, a) = V^\pi(s) + A^\pi(s, a)$$

* $V(s)$ 反映了所处局势的基础分。
* $A(s, a)$ 反映了选该动作带来的“加分”或“扣分”。



### 8.3  架构设计与“不可辨识性”问题

#### 8.3.1 网络分流结构
Dueling DQN 的网络前半部分与标准 DQN 相同（共享特征提取层），随后**分叉为两条独立的流（Streams）**：
* **Value Stream（标量分支）**：输出一个一维标量 $V(s; \theta, \beta)$，表示状态价值。
* **Advantage Stream（向量分支）**：输出一个 $|\mathcal{A}|$ 维向量 $A(s, a; \theta, \alpha)$，表示每个动作的优势。

```text
               ┌───► 状态价值流 V(s) [维度: 1] ────┐
状态 State ──► 共享特征层                           ├──► 聚合计算 ──► Q(s, a) [维度: |A|]
               └───► 动作优势流 A(s, a) [维度: |A|] ┘
```



#### 8.3.2 朴素相加带来的“不可辨识性问题（Unidentifiability）”
如果直接写成 $Q(s, a) = V(s) + A(s, a)$，会遇到一个致命的数学问题——**解不唯一**：
* 假设给定一个 $Q$ 值，若让 $V(s)$ 增加一个常数 $c$，同时让所有的 $A(s, a)$ 都减去 $c$，最终相加得到的 $Q(s, a)$ **完全不变**。
* 这会导致神经网络在反向传播时陷入混乱：网络无法分辨到底该由 $V$ 来解释当前回报，还是由 $A$ 来解释。



#### 8.3.3 解决方案：中心化聚合公式

为了保证唯一确定性，原论文提出了两种约束方案：

##### 方案 A：最大值约束（理论推导形式）
强制最优动作的优势为 0（即 $\max_{a'} A(s, a') = 0$），此时 $V(s)$ 恰好等于最优动作的 $Q$ 值：
$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \max_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

##### 方案 B：均值中心化（实践中最常用的工程形式）
将最大值替换为**所有动作优势的平均值（Mean）**[1]：
$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

> **为什么工程上普遍选择方案 B（均值法）？**
> 1. 虽然方案 A 理论上更符合优势函数定义，但 $\max$ 操作会导致梯度更新只沿着某一个动作流动，波动剧烈。
> 2. 方案 B 减去均值后，保证了 $\sum_a \left(A(s, a) - \bar{A}\right) = 0$，**所有的动作优势在每次反向传播时都能平滑地获得梯度更新**，训练稳定性远高于最大值法[1]。



### 8.3.4 代码改进

先加个超参数 enable_dueling_dqn

只需要改进 DQN 类：

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super().__init__()

        self.enable_dueling_dqn = enable_dueling_dqn

        # 1. 共享特征提取层 (Shared Feature Layer)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        if enable_dueling_dqn:
            # 2. 状态价值流 (Value Stream) -> 输出标量 V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            # 3. 动作优势流 (Advantage Stream) -> 输出向量 A(s, a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 加多一层
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x): 
        if self.enable_dueling_dqn:
            # 提取共享特征
            features = self.feature_layer(x)
            
            # 分别计算 V(s) 和 A(s, a)
            values = self.value_stream(features)             # 形状: [batch_size, 1]
            advantages = self.advantage_stream(features)     # 形状: [batch_size, action_dim]
            
            # 4. Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
            # 广播机制，[batch_size, 1] 会自动扩展匹配 [batch_size, action_dim]
            q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
            
            return q_values

        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
```

创建的时候记得加一下：

```python
target_dqn = DQN(num_states, num_actions, hidden_dim=self.fc1_nodes, enable_dueling_dqn=self.enable_dueling_dqn).to(device)

```





## 九、 DQN、Double DQN、Dueling DQN

三者并不是相互替代的关系，而是**在不同维度上做出的正交改进**：

| 算法组件               | 改进维度                    | 核心解决的问题                                               | 相互兼容性       |
| :--------------------- | :-------------------------- | :----------------------------------------------------------- | :--------------- |
| **DQN (Nature 2015)**  | 基础算法框架                | 使用 Experience Replay 和 Target Network 稳定深度拟合        | 基础             |
| **Double DQN (2015)**  | **算法目标计算 (Loss)**     | 解决 $\max$ 算子引起的 **Q 值过估计（Overestimation）**      | 针对目标值计算   |
| **Dueling DQN (2016)** | **网络骨架 (Architecture)** | 显式分离局势优劣 $V(s)$ 与动作选择 $A(s, a)$，**提升学习效率**[1] | 针对网络内部结构 |

> **最佳实践**：
> 在实际工程中，通常会**同时开启 Double DQN 和 Dueling DQN**（即在 Dueling 结构的神经网络上，使用 Double DQN 的 Target 逻辑更新），这也是后续著名的 **Rainbow DQN** 的两大核心支柱。

### 9.1 Double DQN + Dueling DQN

#### 9.1.1 CartPole

训一下 CartPole：

```text
(Dqnenv) PS D:\gitRepo\DQN-From-Scratch> python agent.py cartpole1 --train
08-28 02:01:01: Training starting...
08-28 02:01:02: New best reward 11.0 (N/A) at episode 0, saving model...
08-28 02:01:03: New best reward 48.0 (+336.4%) at episode 2, saving model...
08-28 02:01:05: New best reward 60.0 (+25.0%) at episode 29, saving model...
08-28 02:01:09: New best reward 74.0 (+23.3%) at episode 54, saving model...
08-28 02:01:10: New best reward 85.0 (+14.9%) at episode 58, saving model...
08-28 02:01:12: New best reward 109.0 (+28.2%) at episode 65, saving model...
08-28 02:01:13: New best reward 128.0 (+17.4%) at episode 67, saving model...
08-28 02:01:14: New best reward 138.0 (+7.8%) at episode 72, saving model...
08-28 02:01:15: New best reward 147.0 (+6.5%) at episode 74, saving model...
08-28 02:01:22: New best reward 153.0 (+4.1%) at episode 106, saving model...
08-28 02:01:27: New best reward 173.0 (+13.1%) at episode 121, saving model...
08-28 02:01:32: New best reward 185.0 (+6.9%) at episode 133, saving model...
08-28 02:01:34: New best reward 283.0 (+53.0%) at episode 135, saving model...
08-28 02:01:46: New best reward 500.0 (+76.7%) at episode 160, saving model...
Goal reached! Solved in 325 episodes with 5-episode average reward: 500.00!
```

特别快就跑完了，并且跑出来的效果比之前那次好多了，没有明显摇摆。



#### 9.1.2 FlappyBird

训一下 FlappyBird：

```text
(Dqnenv) PS D:\gitRepo\DQN-From-Scratch> python agent.py flappybird1 --train                                                                   
08-28 02:05:12: Training starting...
08-28 02:05:14: New best reward -8.7 (N/A) at episode 0, saving model...
08-28 02:05:14: New best reward -6.9 (+20.7%) at episode 1, saving model...
08-28 02:05:15: New best reward -5.7 (+17.4%) at episode 3, saving model...
08-28 02:05:26: New best reward -5.1 (+10.5%) at episode 54, saving model...
08-28 02:05:37: New best reward -2.7 (+47.1%) at episode 109, saving model...
08-28 02:05:47: New best reward -2.1 (+22.2%) at episode 156, saving model...
08-28 02:06:08: New best reward -0.3 (+85.7%) at episode 255, saving model...
08-28 02:06:33: New best reward 0.9 (+400.0%) at episode 377, saving model...
08-28 02:06:35: New best reward 3.9 (+333.3%) at episode 383, saving model...
08-28 02:07:30: New best reward 4.1 (+5.1%) at episode 635, saving model...
08-28 02:08:10: New best reward 4.6 (+12.2%) at episode 816, saving model...
08-28 02:08:30: New best reward 6.0 (+30.4%) at episode 899, saving model...
08-28 02:09:00: New best reward 6.2 (+3.3%) at episode 1030, saving model...
08-28 02:09:11: New best reward 6.7 (+8.1%) at episode 1074, saving model...
08-28 02:10:44: New best reward 6.8 (+1.5%) at episode 1473, saving model...
08-28 02:10:58: New best reward 8.4 (+23.5%) at episode 1531, saving model...
08-28 02:12:16: New best reward 10.5 (+25.0%) at episode 1854, saving model...
08-28 02:13:01: New best reward 12.9 (+22.9%) at episode 2024, saving model...
08-28 02:13:22: New best reward 13.3 (+3.1%) at episode 2101, saving model...
08-28 02:14:59: New best reward 15.8 (+18.8%) at episode 2462, saving model...
08-28 02:16:47: New best reward 17.9 (+13.3%) at episode 2792, saving model...
08-28 02:16:48: New best reward 20.7 (+15.6%) at episode 2794, saving model...
08-28 02:16:55: New best reward 22.4 (+8.2%) at episode 2809, saving model...
08-28 02:17:40: New best reward 27.0 (+20.5%) at episode 2929, saving model...
08-28 02:17:43: New best reward 41.3 (+53.0%) at episode 2932, saving model...
08-28 02:23:04: New best reward 48.1 (+16.5%) at episode 3567, saving model...
08-28 02:24:38: New best reward 68.9 (+43.2%) at episode 3735, saving model...
08-28 02:37:04: New best reward 75.0 (+8.9%) at episode 4978, saving model...
08-28 02:37:41: New best reward 98.1 (+30.8%) at episode 5017, saving model...
08-28 02:38:16: New best reward 118.4 (+20.7%) at episode 5062, saving model...
08-28 02:43:53: New best reward 132.8 (+12.2%) at episode 5510, saving model...
08-28 03:40:51: New best reward 151.3 (+13.9%) at episode 9775, saving model...
08-28 03:57:15: New best reward 152.9 (+1.1%) at episode 10951, saving model...
08-28 05:50:26: New best reward 162.4 (+6.2%) at episode 19034, saving model...
```

测了几轮：

```
(Dqnenv) PS D:\gitRepo\DQN-From-Scratch> python agent.py flappybird1        
Test Episode 0 | Reward: 12.899999999999974
Test Episode 1 | Reward: 92.39999999999917
Test Episode 2 | Reward: 414.4000000000204
```

第三次跑了八十多根管子，效果还是很不错的。





















