# 13｜状态、动作、张量与网络接口详细契约

## 1. 不可混淆的三层信息
`WorldState` 是引擎全知状态；`ActorObs_i` 只能由视野和历史产生；`CriticState` 仅训练期可用。隐藏敌人变化且未进入视野时，ActorObs 必须逐字节不变（包括排序、聚合、mask、历史缓存）。全局队伍排名若 UI 真实可见，可以单独按模式暴露，不可凭空推断。

## 2. 初版张量 schema（全部可配置）
| Key | Shape | dtype | 说明 |
|---|---|---|---|
| self_balls | [16, 12] | float32 | 每分身位置、速度、log质量、半径、merge剩余、合法性等 |
| self_mask | [16] | bool | 真实分身 |
| visible_balls | [64, 14] | float32 | 敌友可见球、相对坐标、质量、类型 |
| visible_mask | [64] | bool | 实体有效 |
| teammates | [4, 10] | float32 | 可见队友或规则允许的队伍信息 |
| teammate_mask | [4] | bool | 真实队友 |
| resources | [128, 8] | float32 | 可见食物/孢子/刺球 |
| resource_mask | [128] | bool | 真实资源 |
| local_grid | [8, 32, 32] | float32 | 食物、敌方、友方、刺球、边界等密度通道 |
| scalars | [24] | float32 | 当前剩余时间、局部危险、可见资源统计 |
| action_mask | [2, 2] | bool | split/eject 的 0/1 合法性 |

这些是**模型接口初始建议**，非环境固有上限。超额实体按危险优先+距离近邻保留，剩余进入密度网格，记录 `truncation_count` 和 `missed_threat_count`。

## 3. 特征定义
相对坐标 `(x_j-x_self)/view_radius`；速度 `v/v_ref`；质量 `log1p(m)/log1p(m_ref)`；冷却 `remaining/max_timer`。缺失值填 0 但必须由 mask 区分；last_seen_age 只能来自真实历史观察，不能从全局状态“补全”。对于敌人只统计可见分身质量，`estimated_total_mass` 应带 `estimate_valid` 和置信度。

## 4. 实体顺序与不变性
训练网络对非自身实体使用集合编码（Deep Sets：`rho(sum(mask_j*phi(x_j)))`）或注意力池化。若截断需要确定性优先级，则使用威胁分数、距离、实体稳定 ID 作为最后 tie-break；ID 不作为学习特征。测试打乱实体顺序后网络输出近似一致。

## 5. Actor/Critic
Actor：self encoder + entity Deep Sets + local CNN → fusion → optional GRU → 3 heads。Critic：全局实体编码 + team embedding → V(s,agent)；不与 Actor 共享能泄漏全局信息的 encoder。GRU 在 episode 结束时清零，玩家重生时是否清零应由配置明确，训练序列必须携带 mask。

## 6. 混合动作与概率
```python
action = {"move": np.array([ux, uy], dtype=np.float32),
          "split": 0, "eject": 0}
```
`move` 两维 [-1,1]，零向量的语义由规则定义。split/eject 是离散 0/1；两者可否同帧执行按规则配置。默认使用 `pi(move|obs)*pi(split|obs)*pi(eject|obs)`；自回归版本另行实验。PPO 联合 logp 是各头实际采样动作的 logp 之和；tanh 高斯需要 Jacobian 修正，mask 后必须重新归一化。

## 7. 时间语义
20Hz 物理、5Hz 决策时每个策略动作执行 4 个子步。移动保持；split 为一次边沿触发；eject 的按住/重复触发需显式配置。动作合法性按动作提交时刻的状态评估；不能在分身后偷偷重新解释同 tick 的 eject 资格。每次 step 记录 `requested_action`、`executed_action`、`mask`、`reject_reason`。

## 8. 测试
- 视野外敌人移动不影响 ActorObs。
- 分身列表重排不改变动作分布。
- 全部非法时离散动作仍有合法 no-op。
- 连续动作边界、NaN、Inf 处理有明确定义。
- GRU 截断与 episode 结束不混淆。
- 同时 split/eject 语义固定，非法动作不导致质量变化。
- Policy rollout 的 logp 与 learner 重新计算 logp 一致（容差内）。


## 9. 补充：输入字典与概率一致性
每个张量字段需发布字段名、顺序、dtype、单位、归一化区间、mask 和缺失值语义。玩家自身球体位置使用质量加权质心作相对坐标原点；敌人聚合质量仅可来自可见对象。保存 raw action、sample logp、mask、executed action；tanh Gaussian 的 Jacobian 和 masked categorical 重新归一化必须通过 rollout/learner logp 一致性测试。重生时 GRU 重置策略必须版本化。