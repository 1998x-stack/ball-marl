# 05 · Actions, temporal semantics and masks

## Primitive hybrid command
```python
action = {
  "move": [ux, uy],   # continuous [-1, 1]^2
  "split": 0,         # discrete 0/1
  "eject": 0,         # discrete 0/1
}
```
The interpreter maps joystick strength and angle to the configured game rule. Do not assume arbitrary speed throttling is supported in an official build.

## Command timing
At 5 Hz policy rate and 20 Hz physics, hold movement input for four substeps. Split is edge-triggered once at command submission; eject is either per-command or held based on a **versioned rule contract**. Simultaneous split and eject are legal only if the ruleset permits; never silently make them mutually exclusive.

## Hybrid policy
Baseline factorization pi(a|o)=pi(move|o)pi(split|o)pi(eject|o); upgraded autoregressive pi(move|o)pi(split|move,o)pi(eject|split,move,o). Use masked categorical distribution for illegal discrete commands; correctly implement transformed Gaussian log-density and joint PPO log-probabilities. Document whether simultaneous command evaluation uses pre- or post-split mass.

## Validation
When all split actions are illegal the mask leaves only no-split valid. NaN movement, invalid ranges and unexpected keys must follow documented reject/default behavior and log an error. Track attempted-invalid-action rate separately from executed commands.

## Hierarchical option research
Optional higher-level intentions: farm, pursue, escape, merge, feed, support, hazard-push. Do not hard-code tactics into first baseline. Evaluate whether options improve learning efficiency at matched environment interactions and simulator compute.


## V2: execution semantics and policy density
The action interpreter must return `requested, validated, executed, mask_at_request, reasons, generated_events`. Apply masks against the **pre-action** world state; simultaneous split/eject resolution order is an explicit ruleset choice. Separate control sampling from physics execution to prevent repeated split edges during four substeps.

For squashed Gaussian movement, sample `z~N(mu,sigma)`, set `u=tanh(z)`, and calculate `log p(u)=log N(z;mu,sigma)-sum log(1-tanh(z)^2)` with a numerically stable Jacobian. Combined PPO ratio uses sum of all head log-probabilities; entropy handling for transformed distributions needs an explicit estimator. When `split=1` is masked out, `split=0` remains valid; do not silently replace an illegal sampled action and still optimize its original log-prob.

### Edge cases
Zero movement vector, saturated joystick, repeated split key, split during merge cooldown, eject at mass threshold, simultaneous split/eject, death mid-repeat, respawn mid-repeat, and maximum fragment count all require golden traces. Track legal-action coverage and invalid request rate. Hierarchical options must have termination conditions and must be evaluated against primitive controls at equal compute.


## V3 · 混合动作控制、采样概率、动作执行

### 5.1 最小控制原语
```python
action = {
    "move": np.array([ux, uy], dtype=np.float32),  # Box([-1,-1],[1,1])
    "split": 0,    # Discrete(2)
    "eject": 0,    # Discrete(2)
}
```
分身与吐球若允许同帧触发，应按规则定义**完整联合动作**而非错误互斥。对不同模式允许额外 `ping/team_signal` 通讯动作，但第一阶段不实现避免动作空间膨胀；明确区别“信息存在”与“允许通信”。

### 5.2 物理时间和决策时间
20Hz 物理（dt=0.05s）与 5Hz 决策（4 substeps）是研究默认示例。移动向量在四个子步保持，split 在一个配置的边沿触发时刻执行一次，eject 可以按规则是单发或连续喷吐。一定要记录 `requested`, `validated`, `executed` 三类动作，否则训练日志无法重现。

### 5.3 动作合法性和 Mask
| 操作 | 合法条件 | 无法执行时 |
|---|---|---|
| move | 玩家活跃且向量 finite、范围合法 | strict 抛出错误；permissive 记录默认 |
| split | 活跃、质量阈值、分身上限、冷却条件 | split=1 屏蔽 |
| eject | 活跃、吐球后至少剩最小质量、资源条件 | eject=1 屏蔽 |
| no-op | 始终应当合法 | 返回无副作用 |
| dead/respawn | 仅 no-op（若玩家仍在 roster） | 记录 inactive |

注意 Mask 若在 `step` 前产生，必须和采取该动作的同一个环境状态版本对应；不能先改变质量再把动作解释为原本合法。

### 5.4 概率模型
基本独立条件头：
[
\pi(a|o)=\pi_m(u|o)\pi_s(s|o)\pi_e(e|o)
]
自回归版本：
[
\pi(a|o)=\pi_m(u|o)\pi_s(s|u,o)\pi_e(e|s,u,o).
]
两者都是有效选择；后者能表示依赖但需要更复杂的数据和 log-prob 处理。联合对数概率 `logp=logp_move+logp_split+logp_eject`。被 Mask 的触发动作不得被采样，合法 no-op 仍保留。

### 5.5 连续移动的 Tanh-Gaussian 细节
从正态分布采样 `z∼N(μ,σ)`，令 `u=tanh(z)`。变换后密度应减去 `sum(log(1-tanh(z)^2))` 的 Jacobian 项，并对饱和区采用稳定计算。策略输出的已采样动作直接在 [-1,1] 内，不要先取无界高斯再硬 clip 后仍使用原始高斯 logp。

### 5.6 PPO 与边沿触发
每个 Actor 决策产生一次 policy logp；动作在物理层重复 4 substeps，但**不能**把同一个 split 命令看作 4 次独立的 policy 采样。动作解释器的 `edge_trigger_once` 应防止连分被重复误触发；在真正执行了 0 个 split 的非法情形，行为应与 Mask/日志一致。

### 5.7 可选高层动作
`farm,pursue,escape,merge,feed,screen,encircle,split_attack` 可以作为 option `z`，每个 option 指定目标、前置条件、停止条件、最大持续步数。训练有两个独立实验：从头端到端 primitive action；高层 option policy + 低层 controller。要以相同环境交互次数及墙钟时间比较，而非只看曲线更快。

### 5.8 行为级回归测试
- 达到最大分身数无法再 split。
- 质量正好等于最小阈值时比较符被正确执行。
- 接近极限摇杆不产生 inf/nan。
- 连续同样动作两个决策 tick 的分身行为按规则生效。
- split 与 eject 同 tick 顺序固定。
- GRU 采样和 learner logp 复算误差小于声明容差。
- 自己死亡时 action_mask 仍含 no-op，且分身集合不变。
- 策略行动在 rollout 与 replay 中执行一致。

**关联**：[中文张量/动作](13-observation-action-tensor-contract.zh-CN.md) · [物理](02-physics-and-rules.md) · [训练](08-training.md)。
