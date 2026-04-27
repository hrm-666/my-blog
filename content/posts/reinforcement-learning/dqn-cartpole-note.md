+++
title = "DQN 学习笔记：从 Q 表到 CartPole 平衡杆实验"
date = 2026-04-27T22:50:00+08:00
draft = false
description = "记录我学习 DQN 的完整过程：从 Q-learning 的局限，到 DQN 的公式理解、Replay Buffer、Target Network，再到用 PyTorch 训练 CartPole 平衡杆。"
summary = "这篇文章整理我学习 DQN 的全过程：为什么需要 DQN、它和 Q-learning 的关系、关键公式怎么理解，以及我如何用 PyTorch 训练 CartPole 并录制推理效果。"
categories = ["强化学习"]
tags = ["强化学习", "DQN", "CartPole", "PyTorch", "学习笔记"]
+++

## 前言

在学完 Q-learning 之后，我其实有一个很自然的问题：如果环境的状态不是一个有限编号，而是一串连续数值，那 Q 表还怎么建？

比如这次用到的 `CartPole-v1`，也就是经典的平衡杆问题。它的状态不是 FrozenLake 里那种 `0, 1, 2, ...` 的离散格子，而是四个连续变量：

```text
[小车位置, 小车速度, 杆子角度, 杆子角速度]
```

这就有点麻烦了。因为连续状态几乎是无限多的，我不可能真的开一张表，把每一种状态下每个动作的 Q 值都存下来。

于是 DQN 就顺理成章地出现了：

> 当 Q 表存不下时，用神经网络来近似 Q 函数。

这篇笔记主要记录我学习 DQN 的全过程，从背后的想法、公式理解，到最后用 PyTorch 训练一个 CartPole 平衡杆模型。

## 最终效果

先放一下训练后模型的推理效果。这里小车会根据当前状态选择向左或向右移动，让杆子尽量保持平衡。

![DQN 训练后的 CartPole 推理效果](/videos/dqn-cartpole-inference.gif)

这个 GIF 是用训练好的模型跑推理脚本录下来的。相比只看 reward 数字，能看到杆子真的被稳住，感觉会更直观一些。

## 从 Q-learning 到 DQN

Q-learning 的核心是学习动作价值函数：

```text
Q(s, a)
```

它表示：在状态 `s` 下采取动作 `a`，之后继续尽量做出较好的决策，未来大概能拿到多少累计回报。

在表格型 Q-learning 中，这个值直接存在 Q 表里：

```text
行：state
列：action
格子：Q(state, action)
```

但 CartPole 的状态是连续的，不适合直接查表。所以 DQN 做的事情其实没有改变强化学习目标，只是换了一种表达方式：

```text
Q-learning：用表存 Q 值
DQN：用神经网络估计 Q 值
```

也就是说，DQN 仍然是在学 `Q(s, a)`，不是在直接学习动作概率。它仍然是 value-based 方法。

## DQN 的网络到底输出什么

一开始我很容易想成：神经网络是不是直接输出“向左”或者“向右”？

但其实不是。

在 DQN 里，网络输入是当前状态，输出是当前状态下每个动作的 Q 值。CartPole 只有两个动作，所以网络输出可以理解成：

```text
[Q(s, 左), Q(s, 右)]
```

真正决策时，再从这两个值里选最大的那个：

```python
action = q_values.argmax(dim=1).item()
```

所以网络不是直接“给答案”，而是给每个动作打分。动作是我们对这些分数取 `argmax` 得到的。

我在代码里定义的 Q 网络是一个很朴素的全连接网络：

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
```

这里 `state_dim = 4`，对应 CartPole 的四个状态量；`action_dim = 2`，对应向左和向右两个动作。

最后一层输出 `action_dim` 个值，不是因为网络要输出动作编号，而是因为每个动作都需要一个 Q 值。

## DQN 的目标值怎么来

Q-learning 的核心更新公式是：

```text
Q(s,a) = Q(s,a) + alpha * (reward + gamma * max Q(s',a') - Q(s,a))
```

DQN 换成神经网络以后，核心思想还是一样的。只不过我们不再直接更新表格里的某个元素，而是让网络输出的当前 Q 值，朝着下面这个目标靠近：

```text
target = reward + gamma * max Q(next_state, a')
```

如果这一局已经结束，也就是 `done = True`，后面就没有未来收益了，所以目标值只剩当前奖励：

```text
target = reward
```

代码里我用了更统一的写法：

```python
target_q_values = rewards + gamma * next_q_values * (1 - dones)
```

当 `done = 1` 时，后面的未来价值会被乘成 `0`。

这一步我觉得很重要，因为它说明 DQN 并不是凭空造了一个监督学习标签，而是用环境反馈和当前网络估计一起构造了一个“临时目标”。

## 为什么 DQN 比表格 Q-learning 更不稳定

表格 Q-learning 里，我们更新的是表格里的一个格子。这个格子改了，别的格子不一定受影响。

但 DQN 不一样。DQN 更新的是神经网络参数。网络参数一变，很多状态下的 Q 值都会跟着变。

这带来几个问题：

- 神经网络近似本身会有误差；
- 目标值里又用了网络自己的估计；
- 连续交互产生的数据相关性很强；
- 当前网络一更新，目标也可能跟着晃。

所以 DQN 训练时很容易变成“自己给自己出题，自己又改答案”。为了让训练稳一点，经验回放和目标网络就变得很关键。

## Replay Buffer：不是单纯存数据

Replay Buffer 存的是智能体和环境交互得到的经验：

```text
(state, action, reward, next_state, done)
```

我的代码里是这样写的：

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
```

它的意义不只是“存起来以后再用”。更重要的是，每次训练时从 buffer 里随机采样一批经验，这样可以打破连续样本之间的时间相关性。

如果智能体刚刚连续经历了一串很相似的状态，然后马上拿这些相邻样本训练，网络更新方向就很容易偏。随机采样后，训练数据会更分散，也更稳定。

我现在对 Replay Buffer 的理解是：

> 它把“在线连续经历”变成了“随机小批量训练数据”。

这一步有点像把强化学习问题拉回到神经网络更擅长处理的 mini-batch 训练形式。

## Target Network：让目标别跟着一起乱跑

DQN 里通常有两个网络：

- `policy_net`：真正正在训练的网络；
- `target_net`：用来计算目标值的网络。

预测当前动作价值时，用的是 `policy_net`：

```python
q_values = policy_net(states).gather(1, actions)
```

计算下一状态的目标价值时，用的是 `target_net`：

```python
with torch.no_grad():
    next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones)
```

这样做的好处是：当前网络可以更新，但目标网络相对稳定一点。否则如果预测值和目标值都来自同一个正在剧烈变化的网络，训练会更抖。

我这份代码里使用的是 Soft Update：

```python
def soft_update(policy_net, target_net, tau):
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            tau * policy_param.data + (1.0 - tau) * target_param.data
        )
```

它不是每隔一段时间突然复制一次参数，而是每一步只让目标网络慢慢靠近当前网络：

```text
target = tau * policy + (1 - tau) * target
```

我这里设置的是：

```python
tau = 0.005
```

这个值比较小，意思是目标网络跟得很慢，目标值变化也更平滑。

## `.gather()` 这一句到底在干什么

这句代码一开始看起来很抽象：

```python
q_values = policy_net(states).gather(1, actions)
```

其实它在做的事情很具体。

网络对每个状态都会输出所有动作的 Q 值，比如：

```text
[Q(s, 左), Q(s, 右)]
```

但一条经验里实际发生过的动作只有一个。比如当时执行的是“右”，那这条样本只应该更新 `Q(s, 右)`，而不是把左右两个动作都拿来更新。

所以 `.gather(1, actions)` 的作用就是：

> 从网络输出的所有动作 Q 值里，挑出这批样本当时实际执行过的动作对应的 Q 值。

这也是 DQN 训练时非常关键的一步。

## `torch.no_grad()` 不能省

在计算 target 的时候，我用了：

```python
with torch.no_grad():
    next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
```

原因是 target 只是训练时的目标值，不应该参与反向传播。

如果不加 `no_grad()`，目标值那条分支也会被放进计算图里，梯度会沿着不该更新的方向传播。轻则浪费显存，重则让训练逻辑变得更乱。

所以这不是一个“优化小技巧”，而是概念上就应该这么做。

## 探索和利用：epsilon-greedy

训练时我还是用了和 Q-learning 类似的 `epsilon-greedy`。

```python
def select_action(state, policy_net, epsilon, action_dim, device):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state)

    return q_values.argmax(dim=1).item()
```

一开始 `epsilon = 1.0`，基本都在随机探索；之后逐渐衰减：

```python
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
```

这样做的原因很朴素：网络刚开始什么都不懂，如果一上来就完全贪心，很容易一直重复早期偶然选到的动作。先多探索，再慢慢转向利用，训练会更稳一些。

推理阶段就不需要继续探索了，直接选择 Q 值最大的动作：

```python
action = q_values.argmax(dim=1).item()
```

## 我的训练配置

这次实验的环境是：

```python
env_name = "CartPole-v1"
env = gym.make(env_name)
```

主要超参数如下：

```python
replay_buffer = ReplayBuffer(capacity=50000)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-4)

gamma = 0.99
batch_size = 128

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

num_episodes = 1000
tau = 0.005
```

我这里还加了梯度裁剪：

```python
torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
```

这是为了防止训练过程中梯度突然变得很大，让网络更新过猛。

loss 用的是 Huber loss：

```python
loss = F.smooth_l1_loss(q_values, target_q_values)
```

相比普通 MSE，Huber loss 对异常的大误差会稍微温和一些，训练时一般更稳。

## 一次训练回合发生了什么

把细节放在一起，一次训练大概是这样的：

1. 重置环境，得到初始状态；
2. 用 `epsilon-greedy` 选动作；
3. 执行动作，得到 `next_state`、`reward`、`done`；
4. 把经验存入 Replay Buffer；
5. 如果 buffer 里样本够了，就随机采样一批；
6. 用 `policy_net` 算当前动作的 Q 值；
7. 用 `target_net` 算目标 Q 值；
8. 计算 loss，反向传播，更新 `policy_net`；
9. 用 Soft Update 缓慢更新 `target_net`；
10. 一局结束后衰减 `epsilon`。

对应到代码里的主循环，大概就是：

```python
while not done:
    action = select_action(
        state,
        policy_net,
        epsilon,
        action_dim,
        device
    )

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    replay_buffer.push(state, action, reward, next_state, done)

    loss = optimize_model(
        policy_net,
        target_net,
        replay_buffer,
        optimizer,
        batch_size,
        gamma,
        device
    )

    soft_update(policy_net, target_net, tau)

    state = next_state
    total_reward += reward
```

这一段和之前 Q-learning 的精神其实非常像：

> 先试动作，再看环境反馈，然后修正自己对这个动作价值的估计。

只是这一次，修正的不是 Q 表里的某个格子，而是神经网络参数。

## 评估和保存模型

训练过程中，我没有只看单局 reward，而是每隔 20 轮做一次独立评估：

```python
if (episode + 1) % 20 == 0:
    eval_mean, eval_std = evaluate(policy_net, env_name, device, episodes=20)
```

评估时不再随机探索，而是直接用当前策略选择 Q 值最大的动作。

如果评估平均奖励更高，就保存当前最好模型：

```python
if eval_mean > best_eval_reward:
    best_eval_reward = eval_mean
    torch.save(policy_net.state_dict(), "best_dqn_cartpole.pth")
```

训练结束后，还会保存最后一版模型：

```python
torch.save(policy_net.state_dict(), "last_dqn_cartpole.pth")
```

这里我觉得比较重要的一点是：强化学习不能只看某一局的表现。CartPole 虽然比很多任务简单，但神经网络训练和环境过程仍然会有波动，所以看多局平均 reward 会更靠谱。

## 推理和录制 GIF

训练完成后，我写了一个单独的推理脚本，使用 `RecordVideo` 录制模型表现：

```python
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder=str(video_dir),
    episode_trigger=lambda episode_id: episode_id < 2,
    name_prefix="dqn_cartpole_inference"
)
```

推理时加载训练好的模型：

```python
policy_net = QNetwork(state_dim, action_dim).to(device)
model_path = Path(__file__).resolve().parent / "dqn_cartpole.pth"
policy_net.load_state_dict(torch.load(model_path, map_location=device))
policy_net.eval()
```

然后每一步都直接取 Q 值最大的动作：

```python
with torch.no_grad():
    q_values = policy_net(state_tensor)

action = q_values.argmax(dim=1).item()
```

我还在推理过程中打印了每一步的 Q 值、动作和 reward：

```python
print(
    f"episode={episode + 1}, step={step}, "
    f"Q values: {q_values.cpu().numpy()}, action: {action}, reward: {reward}"
)
```

这样做的好处是，不仅能看到小车最后能不能稳住杆子，还能看到网络每一步到底更偏向哪个动作。

## 配套代码

这篇文章对应的训练脚本、推理脚本和后续强化学习实验代码，我会逐步整理到这个仓库里：

- [rl-learning：强化学习代码与实验记录](https://github.com/hrm-666/rl-learning)

目前这篇文章主要对应 DQN 训练 CartPole 的部分。后面如果继续写 Double DQN、Dueling DQN 或者 PPO，我也会尽量把博客和代码放在一起，方便复盘。

## 这次实验里我最有感觉的点

第一，DQN 并不是一个和 Q-learning 完全割裂的新东西。它仍然在学 `Q(s, a)`，只是把“查表”换成了“网络估计”。

第二，神经网络不是直接输出动作，而是输出每个动作的价值。这个区别很关键，因为它决定了 DQN 仍然是 value-based 方法，而不是 policy-based 方法。

第三，Replay Buffer 和 Target Network 不是锦上添花，而是稳定训练的核心。没有它们，DQN 很容易因为数据相关性和目标漂移变得不稳定。

第四，`done`、`.gather()`、`torch.no_grad()` 这些看起来像代码细节的地方，其实都对应着强化学习里的概念。如果这些地方理解模糊，代码即使能跑，也很难知道自己到底在训练什么。

## 容易记错的地方

### DQN 学的是动作概率吗？

不是。DQN 学的是动作价值 `Q(s, a)`，属于 value-based 方法。

### 网络直接输出动作吗？

不是。网络输出每个动作的 Q 值，动作是通过 `argmax` 选出来的。

### Replay Buffer 只是为了存数据吗？

不是。它更重要的作用是打破样本之间的时间相关性，提高样本利用率，让训练更稳定。

### Target Network 可有可无吗？

不太行。Target Network 用来稳定目标值，是 DQN 能比较稳定训练的重要机制之一。

### `done` 只是表示一局结束吗？

不只是。它还决定目标值里是否应该保留未来收益。终止状态下，未来价值应该归零。

## 小结

这次 DQN + CartPole 实验对我来说，是从“表格强化学习”走向“深度强化学习”的第一步。

如果用一句话总结：

> DQN 还是在做 Q-learning 那件事，只不过它用神经网络代替 Q 表，用 Replay Buffer 和 Target Network 让训练尽量稳定。

后面如果继续往下学，我比较想接着看：

- Double DQN 如何缓解 Q 值高估；
- Dueling DQN 为什么要拆 value 和 advantage；
- Prioritized Replay 为什么要优先采样重要经验；
- DQN 和后面的 Actor-Critic 方法到底差在哪里。

这篇先作为我学习 DQN 的第一份完整记录。至少之后复习时，我能沿着“Q 表放不下 -> 网络估计 Q 值 -> 目标值构造 -> 稳定训练 -> 推理验证”这条线，把 DQN 重新串起来。
