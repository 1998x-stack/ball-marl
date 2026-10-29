# 12｜完整游戏规则契约与边界案例（中文）

> 状态：研究模拟器规范草案，不是《球球大作战》官方参数。**所有数字均须配置化和实验标定**。本章补足原文缺少的可执行判定、时间语义、异常场景。

## 1. 实体和身份
世界包含 Match、Team、Player、Ball、Pellet、EjectedMass、Virus、Boundary、Event。TeamID、PlayerID 在一局中固定；BallID 创建时分配且不复用；被吞噬的 Ball 进入 tombstone，不在同一物理步重复参与碰撞。玩家拥有零至 N 个分身；玩家失去最后分身不必立即结束比赛，应由模式规则决定重生。

## 2. 规则配置及来源
每条规则必须记录 `id, mode, value, unit, status(PROPOSED|MEASURED|UNKNOWN), source, build, confidence, calibration_test`。不得把经验值冒充官方值。

| 参数 | 单位 | 默认状态 | 必测边界 |
|---|---|---|---|
| map_width / map_height | world-unit | PROPOSED | 角落、边界夹击 |
| tick_dt / policy_repeat | 秒 / 次 | PROPOSED | 跨步事件顺序 |
| mass_radius_density | mass/area | PROPOSED | 极小/极大质量 |
| speed(m) | unit/s | UNKNOWN | 单球与多分身 |
| eat_ratio_lambda | 无量纲 | UNKNOWN | 等质量、临界比值 |
| overlap_beta | 无量纲 | UNKNOWN | 刚好相切、完全包含 |
| split_min_mass | mass | UNKNOWN | 等于阈值、阈值以下 |
| max_fragments | 个 | UNKNOWN | 15→16、16→拒绝 |
| split_impulse_curve | unit/s | UNKNOWN | 边界、不同质量 |
| split_mass_loss | mass | UNKNOWN | 总账核对 |
| merge_cooldown | 秒 | UNKNOWN | 计时与碰撞同刻 |
| eject_mass / eject_loss | mass | UNKNOWN | 质量不足、连续按住 |
| virus_trigger_rule | 条件 | UNKNOWN | 多球同时碰刺 |
| respawn_mass / invulnerability | mass/秒 | MODE-SPECIFIC | 同 tick 被击杀 |
| rank_tiebreak | 规则 | MODE-SPECIFIC | 质量相等 |

## 3. 单步物理事务
输入验证→按 PlayerID 排序排队→边沿操作生效→运动积分→边界约束→候选碰撞→稳定排序→逐个验证当前仍存活实体→原子吞噬/转移→刺球效果→计时器更新→质量账本结算→终局判定→事件输出。需要版本化说明：分身操作在移动前还是移动后发生；同帧多次操作是否接受；合并和吞噬哪个先判定。

**绝不使用容器遍历顺序作为隐含优先级**。同一 tick 两个捕食者竞争同一目标时，明确选择：先发生的连续碰撞时间、固定 tie-break 或同时结算方案，并以测试锁定。

## 4. 质量守恒与所有权
记录事件 `{source, sink, amount, reason, tick, team_from, team_to}`。玩家分身和合并属于同所有者内部转移；队友吞噬属于跨玩家同团队转移；敌方吞噬属于跨团队转移；食物刷新属于外源注入；吐球散失、衰减、过期属于质量损耗。逐步验证
`M_after = M_before + external_in - external_out`，
并分别对 TeamID 计算跨队流入流出，允许记录浮点误差阈值。

## 5. 战术规则与可测假设
- **分身追击**：必须检查分裂后最大单球是否仍满足吞噬阈值，且突进轨迹是否覆盖目标；击中后仍可能被第三方反吞。
- **中吐/交叉吐**：是摇杆与吐球的时序组合，不是独立魔法技能。用同一底层输入可重放的轨迹来定义战术标签。
- **扎刺/推刺**：刺球移动和触发由物理碰撞决定；不直接奖励“触发刺球”，只评价下游团队结果。
- **合球**：到时不一定自动合并，必须满足模式规定的空间条件；具体判定待标定。
- **队友保护**：用可重放的反事实比较“有/无保护行为”下被吞噬风险，不把站在旁边等同于贡献。

## 6. 规则黄金用例
| ID | 初始条件 | 输入 | 断言 |
|---|---|---|---|
| PHY-001 | 两球恰好相切 | 无 | 不发生非法吞噬 |
| PHY-002 | 两球质量刚好临界 | 接近 | 吞噬按边界比较符判定 |
| PHY-003 | 两敌同时可吞同目标 | 同时接触 | 结果唯一、确定 |
| SPL-001 | 达到最大分身数 | split | 动作被屏蔽且无新球 |
| SPL-002 | 质量低于最小阈值 | split | 质量不变 |
| SPL-003 | 边界附近分身 | split | 所有球位置合法 |
| MER-001 | 冷却差一 tick | 靠近 | 不提前合并 |
| EJT-001 | 质量不足 | eject | 不出现负质量 |
| EJT-002 | 友方接收 | eject+absorb | 团队总质量仅减少损耗 |
| VIR-001 | 同步多球触刺 | move | 事件顺序确定 |
| LIFE-001 | 玩家失去最后分身 | 无 | 由模式规则决定重生/结束 |
| RANK-001 | 两队同质量 | 终局 | 明确并列规则 |
| DET-001 | 同快照同动作同随机态 | replay | 逐 tick hash 相同 |

## 7. 官方对齐实验
以“模式+客户端版本+录屏帧率+输入时序”为样本单位。分别测速度-质量曲线、分身射程、吞噬临界比、合并等待时间、吐球损耗、刺球分裂数量；每个参数至少训练/验证两套轨迹，报告置信区间与拟合残差。禁止仅凭玩家攻略确定数值。


## 8. 补充：可执行状态机与规则校准
Player: ACTIVE→ZERO_BALLS→RESPAWN_WAIT→ACTIVE，或 ZERO_BALLS→ELIMINATED，由模式控制；Ball: CREATED→MOVING→MERGE_ELIGIBLE→MERGED/CONSUMED。吞噬后实体 tombstone，禁止同 tick 二次消费。高速分身采用子步或连续碰撞检测以避免穿透。官方对齐必须提供游戏版本、模式、录屏帧率、动作时序、测量置信区间与独立验证轨迹；没有证据时保持 UNKNOWN。