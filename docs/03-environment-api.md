# 03 · Environment, agents and API contract

## Agent identity
Stable agent key `team_0/player_0` controls all of that player's fragments. Fragment creation/deletion does not change agent IDs. Death may be a temporary inactive/respawn state; episode termination depends on game mode and is distinct from player death.

Use PettingZoo ParallelEnv for simultaneous joint steps. A step advances one policy tick consisting of N physics substeps.

```python
observations, infos = env.reset(seed=42)
actions = {agent: policy(observations[agent]) for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
```

## Reset configuration
`reset(seed, options={ruleset_id, scenario_id, roster, spawn_seed, resource_seed, opponent_ids})`. Record all effective options and deterministic state hash.

## Episode semantics
- **Terminated**: game-defined terminal condition (e.g. match clock ends in a finite-horizon ranking game).
- **Truncated**: external rollout/time-budget interruption that is not a real terminal state. Bootstrap critic on truncation when appropriate.
- Respawn remains within an episode unless ruleset explicitly states otherwise.
- Maintain policy masks for temporarily inactive players and define zero-effect action behavior.

## Information contract
`infos`: non-sensitive debug metadata may contain global data during training but **never** be concatenated into Actor input. Expose global state only through an explicit training-only `state()` method, with leakage tests.

## Scenarios
Single-agent resource collection; predator/prey pursuit; split/counterattack; hazard interaction; 2v2 transfer; 5v5 coordination; ranked multi-team. Each has a fixed manifest and seed split.


## V2: precise lifecycle and compliance
`possible_agents` is the fixed roster for a match. `agents` contains agents currently participating according to the chosen API policy. In the recommended respawn mode, temporarily ball-less players remain in `agents`, receive inactive observations and may submit only no-op; they are **not terminated** until match end. For permanent elimination, remove the agent according to PettingZoo lifecycle conventions and return final transition data consistently.

```python
class BallParallelEnv(ParallelEnv):
    metadata = {"name": "ball_marl_research_v0", "render_modes": ["human","rgb_array"]}
    def reset(self, seed=None, options=None): ...
    def step(self, actions): ...
    def observation_space(self, agent): ...
    def action_space(self, agent): ...
    def state(self): ...  # critic only
```
At `step` boundary, reject unknown agent keys; missing active commands follow a documented policy (strict error in tests, no-op only in explicit permissive wrapper). `infos` exposes executed action, rejected reasons, reward breakdown and snapshot hash for research; do not route it into Actor tensors. `terminated` is true on a **ruleset-defined match end**, including a match clock if that clock is part of the actual game; `truncated` is reserved for external cutoff.

### Verification
Run `parallel_api_test`, seed determinism, observation/action `space.contains`, final-step dictionary shape, restart after close, and multi-episode memory-leak tests. Reference: PettingZoo Parallel API.


## V3 · PettingZoo 多智能体环境完整接口规范

### 3.1 API 主契约
环境以一个玩家为一个 Agent；分身是玩家实体数据，不是额外 Agent。推荐采用 `pettingzoo.utils.env.ParallelEnv`。相同时刻所有 Agent 提交动作，环境一次性更新 joint action。需要兼容 `reset(seed,options)→(observations,infos)` 和 `step(actions)→(observations,rewards,terminations,truncations,infos)`；空间应对同一 Agent 稳定返回同一个声明值。

```python
class BallMARLEnv(ParallelEnv):
    metadata = {"name": "ball_marl_research_v1", "render_modes": ["rgb_array", "human"]}
    def __init__(self, ruleset, scenario, render_mode=None):
        self.possible_agents = build_stable_roster(ruleset)
        self.agents = []
    def observation_space(self, agent): ...
    def action_space(self, agent): ...
    def reset(self, seed=None, options=None): ...
    def step(self, actions): ...
    def state(self): ...  # privileged critic-only accessor
    def render(self): ...
    def close(self): ...
```

### 3.2 `reset` 的输入输出
`seed` 作为根 SeedSequence，派生 `spawn/resource/hazard/physics/opponent` 独立流；`options` 包含 `ruleset_id, scenario_id, initial_world_override, team_roster, policy_timestep_limit`，仅白名单字段可改。reset 需清空历史对象、事件、GRU 状态建议标志、计数、统计；不得重用上局随机游标。返回每位活动 Player 的 observation dict 及 `info={scenario_hash,ruleset_hash,seed,initial_state_hash}`。

### 3.3 `step` 的事务定义
1. 检查 `actions.keys()` 是否与 `env.agents` 一致。
2. 拿**动作前快照**计算合法性和时间边沿。
3. 遍历全部 Agent 的动作验证，不因字典遍历顺序修改世界。
4. 一次性提交联合命令并推进固定数物理子步。
5. 结算事件、计分、重生和终局状态。
6. 使用最终状态产生合法观察及奖励；返回本步 step 的事件摘要。
7. 游戏终局时返回必要的最终键，再按 PettingZoo 合规要求清理 `agents`。

严格模式拒绝缺失/多余 Agent 键；宽松训练 wrapper 可将缺失动作映射 no-op，但必须记录 `action_defaulted=true`，以免静默产生策略偏差。

### 3.4 死亡、重生与长期 Agent 身份
建议模式 `RESPAWN_ENABLED`：所有选手贯穿整局保持 Agent ID；暂无分身时 observation 有 `alive=false, respawn_left` 和填零集合，动作仅接受 no-op，不把每次死亡标为 terminated。永久淘汰模式则按 API 规则移除完成的 Agent；不得在已终止玩家的 reward 字典中持续伪造有效动作。

### 3.5 游戏结束和 rollout 截断
若比赛 180 秒是模式本身定义的固定时长，那么 180 秒属于 `terminated=True`，**不是** `truncated=True`。如果实验训练器在 100 步时停止一次 rollout，而游戏还没结束，应由训练采样器切段并 bootstrap 价值，不应让环境误报真实 game over。若外部设定独立的 TimeLimit wrapper，必须保留截断原因用于 Critic bootstrap。

### 3.6 因果一致的调试信息
`infos[agent]` 推荐输出 `{requested_action,executed_action,mask_at_request,invalid_reason,reward_components,event_ids,pre_hash,post_hash}`。只能用于日志和测试；训练 Actor 输入由 `observation_builder` 构造，禁止将 debug info 或全局 hash 转为策略输入。集中式 Critic 使用单独 `state()` 函数。

### 3.7 兼容性测试清单
- reset 后返回字典键与 `agents` 一致，空间 `contains` 为真。
- seeded reset+动作序列在支持后端逐步 hash 一致。
- 空行动、额外 Agent、非法行动有确定性结果。
- 1/2/5 人队伍、更换队伍数量，AgentID 不重复。
- 含死亡/重生的长局保持状态机正确。
- `parallel_api_test` 通过，终局后的 `agents=[]` 行为正确。
- `render_mode=rgb_array` 图像形状稳定；无渲染模式下不能意外初始化图形窗口。
- close() 幂等、连续 100 局无明显内存/文件描述符泄漏。

### 3.8 场景注册契约
```yaml
scenario_id: feed_to_kill_v1
mode: research_2v2
world_seed: 8123
initial_balls: []    # 每个实体声明 owner / mass / x / y / velocity
initial_pellets: []
initial_hazards: []
max_policy_ticks: 600
success_metric: team_enemy_mass_consumed
allowed_ruleset_hash: null
```
正式案例中的列表不得为空，示意占位符需要具体样本填充。保存 `scenario_hash`，不可在同名 scenario 下隐式修改布局。

**关联**：[物理事务](02-physics-and-rules.md) · [状态](04-observations.md) · [动作](05-actions.md) · [工程验收](10-engineering-roadmap.md)。
