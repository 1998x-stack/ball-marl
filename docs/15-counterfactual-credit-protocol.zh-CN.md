# 15｜反事实信用分配：可执行实验协议

## 1. 四类 estimand，严禁混用
- **MAPPO-GAE**：在当前策略与价值函数下的优势估计，服务于梯度更新。
- **Difference reward**：`D_i^H = G_H(actual) - G_H(intervene_i)`，依赖基准动作与反事实延续。
- **COMA advantage**：`Q(s,a)-E_{a_i'~pi_i}Q(s,(a_i',a_-i))`，依赖联合 Q 估计，连续动作要积分或采样。
- **Interaction effect**：`I_ij=G(ij)-G(i0,j)-G(i,j0)+G(i0,j0)`，刻画联合超额收益，非简单可加的个人奖金。

## 2. 反事实实验单元
`{snapshot_hash, tick, ruleset_hash, agent_id, actual_action, intervention, horizon_H, continuation_mode, opponent_policy_hash, rng_state, reward_definition}`。缺一不可。候选干预包括 no-op、延续上一步移动、从冻结策略抽样 K 次。对连续动作不能只比较“静止”基线。

## 3. Open-loop 与 Closed-loop
- **Open-loop**：固定后续所有智能体的原始动作序列，测局部动力学变化；可能产生在新状态下不合理的动作。
- **Closed-loop**：冻结策略参数，所有玩家在反事实观察下重新行动；测动态反馈，代价高。
- 两者都保持相同初始快照和可比较随机数流；当环境的随机调用次数随事件变化时，使用按事件键控的随机流或明确说明 common-random-numbers 失配风险。

## 4. 窗口与结果
`H={1,5,20,100}` 个策略步；指标至少包含团队原始回报、团队净质量变化、目标吞噬概率、队友死亡概率和最终排名。H 小时只能说明短期边际贡献；H 大时需报告方差、对手重规划、运行成本。

## 5. 团队协作专项场景
**Feed-to-kill**：A 的吐球使 B 的最大分身跨过吞噬阈值。干预 A 的 feed 后，B 是否仍能击杀？验证质量账本没有重复计分。

**Encirclement**：A、B 分别封锁逃跑方向，移除其中一个时目标有逃生路线。需验证不是仅由地图边界导致成功。

**Protection**：C 通过位置或质量转移使 D 存活。对照同一敌方策略，计算 D 的被吞噬风险变化。

**Delayed split trap**：分身 H=1 获利，H=100 受损，检查训练策略是否降低过度冒险。

**Role switch**：原核心失去质量后，另一队友接管；评估动态分工而非固定 agent ID。

## 6. 事件图与归因
事件表 `events(event_id,tick,type,actor,target,mass_delta,state_hash)`；因果候选边 `candidate_edges(source_event,target_event,reason,lag)`。候选边只表示可检验假说，不能因时间先后就宣布因果。每个事件有反事实证据 URI、基准敏感性和置信区间。

## 7. 训练集成边界
V1 共享团队奖励 MAPPO；V2 离线 D 分析；V3 用 D 训练辅助 Q/贡献预测头但 PPO 优势不变；V4 独立 COMA-like 算法；V5 长期交互贡献。绝不未经证明将 `GAE + D + COMA` 相加作为“更好优势”。

## 8. 评测统计与算力
按 scenario/seed/agent/baseline/H/continuation 分层报告均值、方差、bootstrap 区间；对配对干预使用配对差置信区间。报告环境物理步、反事实分支数、CPU/GPU 小时、wall time、每个可归因事件成本。抽样事件要报告选择偏差和覆盖率。
