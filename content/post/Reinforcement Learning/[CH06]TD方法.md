---
title: "[CH06]TD方法"
description: "用一步经验把价值向未来推进"
date: 2026-07-24T16:18:07+08:00
lastmod: 2026-07-24T16:18:07+08:00
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

**MC 方法已经让我们摆脱了环境模型**：不必知道 $p(s'\mid s,a)$，只要真实地走完一条轨迹，就能把完整回报 $G_t$ 当作样本更新价值。**但是它有一个根本等待：要知道 $G_t$，必须先走到回合终点。**

本讲的 TD（Temporal Difference，时间差分）方法只等一步。执行 $A_t$ 后，环境立刻给出奖励和下一状态；此时虽然还不知道完整回报 $G_t$，却已经能构造目标
$$
\underbrace{G_t}_{\text{MC：等待整条轨迹结束}}
\qquad\Longrightarrow\qquad
\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{\text{TD：真实的一步 + 对未来的当前估计}}.
$$
这里的 $V(S_{t+1})$ 不是未来的真实答案，而是当前估计。用一个估计去更新另一个估计，叫作**自举**（bootstrapping）。它带来偏置，也换来更快、更在线的更新能力。本章的 SARSA 与 Q-learning，都是把这一思想从 $V$ 推进到 $Q$ 后得到的控制算法。



## 一、使用 TD 方法评估策略

我们先暂不改进策略。策略 $\pi$ 固定，问题是：只凭交互样本，怎样逼近它的状态价值 $v_\pi(s)$？

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784967554897_image.png)

TD 介于 MC 与 DP 之间：

- 像 MC：不需要知道状态转移概率或奖励函数，只消费真实交互样本；
- 像 DP：只向前看一步，并把下一状态的估计价值接到当前目标上。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784968193032_image.png)



### 1.1.1 TD 方法的推导

先从收益的递推开始。
$$
\begin{aligned}
G_t
&=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots,\\
&=R_{t+1}+\gamma G_{t+1}.
\end{aligned}
$$
固定策略的价值定义为
$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
=\mathbb E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s].
\tag{6.3--6.4}
$$
从这里出发，有两条已学过的路。



**第一条：MC。** 把一次完整回报 $G_t$ 当作 $v_\pi(S_t)$ 的样本，以式 (6.5) 的指数移动平均形式更新：
$$
V(S_t)\leftarrow V(S_t)+\alpha\left[G_t-V(S_t)\right].
\tag{6.5}
$$
它的目标是真实走完后得到的 $G_t$。因此普通回合制 MC 必须等待终点。



**第二条：DP。** 若已知模型，则将上面的期望显式展开成贝尔曼方程：
$$
v_\pi(s)
=\sum_{a,s'}\pi(a\mid s)p(s'\mid s,a)
\left[r(s,a,s')+\gamma v_\pi(s')\right].
\tag{6.6}
$$
再将右侧真值换成当前估计，就得到 DP 的一次 backup（式 (6.7)）：

$$
V_{\text{new}}(s)
=\sum_{a,s'}\pi(a\mid s)p(s'\mid s,a)
\left[r(s,a,s')+\gamma V_{\text{old}}(s')\right].
\tag{6.7}
$$
TD 的关键变化是：不对所有 $a,s'$ 求期望，而是让环境实际产生一个样本

$$
(S_t,A_t,R_{t+1},S_{t+1}).
$$
由贝尔曼方程可写出

$$
v_\pi(s)
=\mathbb E_\pi\left[R_{t+1}+\gamma v_\pi(S_{t+1})\mid S_t=s\right].
\tag{6.8}
$$
于是用样本中的一步奖励 $R_{t+1}$ 代替期望中的奖励，同时用当前表中的 $V(S_{t+1})$ 代替未知的 $v_\pi(S_{t+1})$。一阶 TD（也叫 TD(0)）更新就是式 (6.9)：

$$
\boxed{
V(S_t)\leftarrow V(S_t)+\alpha
\left[
\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{\text{TD 目标}}
-V(S_t)
\right].
}
\tag{6.9}
$$
将括号中的差定义为 TD 误差：

$$
\delta_t
=R_{t+1}+\gamma V(S_{t+1})-V(S_t),
\qquad
V(S_t)\leftarrow V(S_t)+\alpha\delta_t.
\tag{L6.1}
$$
这里的“时间差分”正是 $V(S_{t+1})-V(S_t)$ 经奖励和折现修正后形成的差。它不是监督学习里“标签减预测”的真实误差，因为 TD 目标本身含有估计值；它是一个**可立即观测、可用于修正当前估计的误差信号**。



### 1.1.2 MC 方法和 TD 方法的比较

对于回合制问题，不能简单断言“TD 总是更好”。它们面对的是偏置—方差与更新时机的交换。

| 维度               | MC                           | TD(0)                              |
| ------------------ | ---------------------------- | ---------------------------------- |
| 更新目标           | $G_t$                        | $R_{t+1}+\gamma V(S_{t+1})$        |
| 是否等到终点       | 是                           | 否，每一步即可                     |
| 是否自举           | 否                           | 是                                 |
| 目标的偏置         | 完整回报样本对 $v_\pi$ 无偏  | 当前 $V(S_{t+1})$ 未准时会引入偏置 |
| 目标方差           | 长轨迹中随机性累积，通常较大 | 只看一步随机性，通常较小           |
| 连续性任务         | 普通 MC 不适用，因为没有终点 | 可直接使用                         |
| 对错误初始化的反应 | 要等回合结束才传播回报       | 回报可一跳一跳向前传播             |

“TD 方差通常较小”不等于“TD 一定更准确”。如果 $V(S_{t+1})$ 初始化很差，TD 目标也会跟着偏；MC 的完整回报虽波动更大，却不把这个估计偏差带入目标。实际中，TD 因为更新频繁、样本复用更及时，常常更快。

>   Q：为什么 TD 方法的误差累积比MC小？
>
>   A：
>
>   一个直觉上的理解就是下图，方向盘扰动对于行驶轨迹的影响：
>
>   ![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784968806395_image.png)



### 1.1.3 TD 方法的实现

把式子封装成类就行了：

```python
class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V
        self.V[state] += (target - self.V[state]) * self.alpha
```

然后跑一下：

```python
env = GridWorld()
agent = TdAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.eval(state, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_v(agent.V)

```

结果看上去近似正确：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784969747786_image.png)



## 二、SARSA

**策略评估之后是策略控制**：既要估计当前策略有多好，也要持续把策略改得更好。

无模型情况下，若只学 $V(s)$，要比较某状态各行动的好坏仍需要知道环境模型；而 $Q(s,a)$ 已把“先做行动 $a$”写进定义，直接贪婪化即可。记
$$
a^*(s)\in\arg\max_a Q(s,a).
$$
SARSA 是最直接的同策略 TD 控制算法。



### 2.1 同策略型的 SARSA

将 TD 状态价值更新中的两个 \(V\) 分别替换为当前状态—行动对和下一状态—行动对的 \(Q\)，得到教材式 (6.10)：

$$
\boxed{
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[
R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)
\right].
}
\tag{6.10}
$$
这条更新所需的一段数据是

$$
(S_t,A_t,R_{t+1},S_{t+1},A_{t+1}).
$$
SARSA 的名称正是这五个量首字母的串联：State, Action, Reward, next State, next Action。

所谓**同策略**（on-policy），是指：

$$
\text{行为策略}=\text{目标策略}=\pi.
$$
也就是说，实际拿来产生样本的 $A_{t+1}$，正是我们想评估和改进的那一个策略所选择的行动。为了既探索又利用，我们采用 $\varepsilon$-greedy。固定一个由随机并列打破选出的贪婪行动 $a^*(s)$ 后，完整概率式是：

$$
\pi'(a\mid s)
=\frac{\varepsilon}{|\mathcal A|}
+(1-\varepsilon)\,\mathbf 1[a=a^*(s)].
\tag{6.11}
$$
这等价于“以 $\varepsilon$ 的概率在所有行动中均匀随机采样，否则选择 $a^*(s)$”的程序描述。于是贪婪行动也会从随机分支额外获得 $\varepsilon/|\mathcal A|$ 的概率。

$\pi'$ 表示由当前 Q 值改进后的下一版策略；



### 2.2 SARSA 的实现

同样是把式子的逻辑包装成类：

```python
class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        next_q = 0 if done else self.Q[next_state, next_action]

        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)
```

然后写个循环测一下：

```python
env = GridWorld()
agent = SarsaAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done)

        if done:
            agent.update(next_state, None, None, None)
            break
        state = next_state

env.render_q(agent.Q)
```

结果：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784971273311_image.png)



![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784971277597_image.png)

多跑几次的话，结果基本都不同，但总体良好。

但策略中包含了 ε 的随机行动。由于策略的随机性，**因此可以看到行动都尽可能远离炸弹**。这样我们就完成了同策略型的 SARSA 的实现。



## 三、异策略型的 SARSA

同策略 SARSA 用一个策略同时完成采样和评价。异策略方法把这两件事拆开：

- 行为策略 $b$：负责走出去、覆盖不同状态—行动对，因而保留探索；
- 目标策略 $\pi$：负责定义我们真正要评估和改进的对象，通常取贪婪策略。



### 3.1 异策略型和重要性采样

![image-20260725172613262](D:\TyporaPics\image-20260725172613262.png)

| 策略           | 更新方式                           | 主要职责                        |
| -------------- | ---------------------------------- | ------------------------------- |
| 目标策略 $\pi$ | 对当前 $Q$ 完全贪婪                | 利用；定义 $Q^\pi$ 想逼近的对象 |
| 行为策略 $b$   | 对当前 $Q$ 做 $\varepsilon$-greedy | 探索；真实产生数据              |

如果仍沿用 SARSA 的样本 $A_{t+1}$，它现在是按 $b$ 而不是按 $\pi$ 抽到的。为把“来自 $b$ 的样本”校正为“若来自 $\pi$ 时的贡献”，我们做上一章的重要性采样比率：

$$
\rho
=\frac{\pi(A_{t+1}\mid S_{t+1})}
{b(A_{t+1}\mid S_{t+1})}.
$$
那么异策略 SARSA 更新写为：

$$
\boxed{
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[
\rho\left(R_{t+1}+\gamma Q(S_{t+1},A_{t+1})\right)
-Q(S_t,A_t)
\right].
}
\tag{6.13}
$$
这个 $\rho$ 校正的是下一行动 $A_{t+1}$ 的来源，而不是环境给出的 $S_{t+1}$。状态转移始终按同一个环境概率 $p$ 发生。



### 3.2 异策略型的 SARSA 的实现

仍然是封装一下：

```python
class SarsaOffPolicyAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0
            rho = 1
        else:
            next_q = self.Q[next_state, next_action]
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        target = rho * (reward + self.gamma * next_q)
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, 0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)
```

然后跑一下：

```python
env = GridWorld()
agent = SarsaOffPolicyAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done)

        if done:
            agent.update(next_state, None, None, None)
            break
        state = next_state

env.render_q(agent.Q)
```

结果：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784972580801_image.png)

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784972587245_image.png)

这个结果其实不太好，但是每次跑出来又不太一样，可改进的余地很大。



## 四、Q 学习

Q-learning 同样是异策略方法，却不再对采样到的下一行动 $A_{t+1}$ 做重要性采样。它的思路来自第 3、4 章的贝尔曼**最优**方程。

### 4.1 贝尔曼方程与 SARSA

对固定策略 $\pi$，Q 函数的贝尔曼方程为

$$
q_\pi(s,a)
=\sum_{s'}p(s'\mid s,a)
\left[
r(s,a,s')
+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')
\right].
\tag{L6.2}
$$
这个式子对两类不确定性取了期望：

1. 环境会迁移到哪个 $s'$；
2. 进入 $s'$ 后，策略会选择哪个 $a'$。

SARSA 是它的样本版：环境实际给出一个 $S_{t+1}$，策略实际抽出一个 $A_{t+1}$，于是用
$$
R_{t+1}+\gamma Q(S_{t+1},A_{t+1})
$$
近似原本的双重期望。这里下一行动是从策略中**采样**的，所以它所学的是该策略的 Q 函数。



### 4.2 贝尔曼最优方程与 Q 学习

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784973851173_image.png)

最优 Q 函数满足

$$
q_*(s,a)
=\sum_{s'}p(s'\mid s,a)
\left[
r(s,a,s')
+\gamma\max_{a'}q_*(s',a')
\right].
\tag{L6.3}
$$
**与上一式相比，下一行动不再按某个策略的概率求平均，而是直接取最大价值。**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784973882242_image.png)

对环境状态迁移仍做采样，就得到式 (6.14)：
$$
\boxed{
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[
R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)
\right].
}
\tag{6.14}
$$
这就是 Q-learning 的 TD 目标：

$$
\text{target}_{\mathrm{Q}}
=R_{t+1}+\gamma\max_a Q(S_{t+1},a).
$$

| 算法       | 下一状态使用什么                                   | 目标对应什么                                       |
| ---------- | -------------------------------------------------- | -------------------------------------------------- |
| SARSA      | 已按该步 $\varepsilon$-greedy 策略采样的 $A_{t+1}$ | 该步行为策略的 $q_\pi$；控制中 $\pi$ 随 Q 持续改变 |
| Q-learning | 表中所有下一行动 Q 值的最大值                      | 最优 Q 函数 $q_*$                                  |

因此 Q-learning 是异策略的：

- 行为策略 $b$ 可以是 $\varepsilon$-greedy，用于探索并生成转移；
- 隐含目标策略是对 $Q$ 完全贪婪的策略，用于构造 $\max$ 目标。

它不需要对 $A_{t+1}$ 做重要性采样，因为 TD 目标不使用“行为策略实际抽到的下一行动”。对于已经获得的 $(S_t,A_t,R_{t+1},S_{t+1})$，算法直接查看自己 Q 表中四个候选的下一行动值并取最大者。



### 4.3 Q 学习的实现

依旧封装：

```python
class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)
```

然后写测试脚本：

```python
env = GridWorld()
agent = QLearningAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_q(agent.Q)
```

结果：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784974087031_image.png)

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784974089087_image.png)

虽然结果每次都会发生变化，但在大多数情况下，我们能获得最优策略。（当然这次没有跑到最优）



## 五、分布模型与样本模型

我们回到“策略在程序里如何表达”的问题，讨论下**智能代理选行动的实现方式**。



### 5.1 分布模型与样本模型

分布模型显式保存每个状态下的行动概率。例如随机策略可以写成：

~~~python
random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
self.pi = defaultdict(lambda: random_actions)

action_probs = self.pi[state]
return np.random.choice(actions, p=probs)
~~~

这里程序持有完整的 $\pi(a\mid s)$，再据它采样一个行动。

样本模型不保存概率表，只要在调用时按正确规则生成一个样本即可：

~~~python
def get_action(self, state):
    return np.random.choice(4)
~~~

两者在“每个行动等概率”这个任务上表现等价，但信息保留不同。

| 实现方式 | 保存什么                                      | 优点                                      | 代价或限制                       |
| -------- | --------------------------------------------- | ----------------------------------------- | -------------------------------- |
| 分布模型 | 显式的 $\pi(\cdot\mid s)$ 或 $b(\cdot\mid s)$ | 可查询概率；异策略重要性采样直接可算      | 状态多时需维护更多对象           |
| 样本模型 | 只保存能够产生行动样本的规则                  | 代码短，适合直接执行 $\varepsilon$-greedy | 通常不能直接得到某行动的精确概率 |

例如异策略 SARSA 必须算 $\pi(A_{t+1}\mid S_{t+1})/b(A_{t+1}\mid S_{t+1})$，所以保留两张概率分布表很自然。Q-learning 不需要这个比率，便可以进一步简化。



### 5.2 样本模型版的 Q 学习

我们删去之前实现中的 pi 和 b（分别是目标策略和样本策略），只保存 Q 表

```python
class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
```

它仍实现 $\varepsilon$-greedy：

- 概率 $\varepsilon$：直接均匀随机探索；
- 概率 $1-\varepsilon$：返回当前最大 Q 值的行动。

而 **update** 与分布模型版 Q-learning 的 TD 核心完全相同：

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[R_{t+1}+\gamma\max_aQ(S_{t+1},a)-Q(S_t,A_t)\right].
$$
**这就是样本模型的实现。由于不需要保存概率分布， 因此其实现起来很简单。**

在后面的章节，我们将使用神经网络对 Q 学习进行扩展。



## 六、小结

我们完成了从“整段经历后才学习”到“每一步都能学习”的转折：

$$
\text{完整回报 }G_t
\quad\longrightarrow\quad
\text{一步奖励 }R_{t+1}
+\text{ 当前未来估计}.
$$
这使得无模型方法能够处理连续性任务，也让价值信息能沿着经验逐步传播。随后：

$$
\text{贝尔曼期望方程}
\rightarrow \text{SARSA},
\qquad
\text{贝尔曼最优方程}
\rightarrow \text{Q-learning}.
$$












