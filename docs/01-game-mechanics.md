# 01 · Mechanics and competitive strategy

## Entities
A **team** contains stable **player agents**; each player controls a dynamically changing set of **ball fragments**. Other entities: pellets, ejected mass, hazards/viruses, boundaries, respawn points and optional special-mode objects.

## Primitive interactions
1. **Movement** directs the player's controlled fragments within the ruleset.
2. **Consumption** transfers victim mass to an eligible larger ball when both mass-ratio and overlap requirements are met.
3. **Split** redistributes one or more fragments' mass and imparts an initial displacement/impulse; may increase reach while reducing per-fragment safety.
4. **Eject/feed** removes mass from a source and creates transferable matter; accounting must capture inefficiency.
5. **Merge/recombine** reunites eligible fragments after the configured timer and overlap conditions.
6. **Hazards** may disrupt concentrated balls and create smaller vulnerable fragments.
7. **Team interactions** may transfer mass to allies where allowed; distinguish friendly absorption from enemy elimination.

## Strategy taxonomy
| Strategy | Required state | Benefit | Exposure |
|---|---|---|---|
| Pellet farming | local resource map | low-risk growth | opportunity cost |
| Split pursuit | fragment mass, geometry, enemy velocity | intercept reach | counter-consumption |
| Escape | threat geometry, boundaries | survival | lost resources |
| Feed core | ally geometry, compatible mass | concentrated attack ability | transfer loss, exposed support |
| Two-core formation | ally strengths and separation | flanking and redundancy | slower concentration |
| Hazard push | hazard location and ejection | disrupt enemy | counterplay |
| Bait / encircle | team coordination | restrict escape | coordination failure |

## Strategic tradeoffs
Let total team mass be M and largest teammate share c=max_i(M_i)/M. Neither maximizing c nor minimizing c is globally optimal: concentrated attack power trades off independent mobility, spatial coverage and single-point failure.

Do not assume split is profitable from immediate consumption alone. Evaluate net outcome at horizons H, including whether the fragmented attacker is later consumed. Distinguish win probability, ranking, total mass, survival and team control as separate outcomes.

## Scenario acceptance examples
- Direct pursuit of a faster small ball does not imply capture.
- A split must be checked against third-party counterattack opportunities.
- Friendly mass transfer cannot count as newly created team mass.
- Hazard-triggered fragmentation is distinguishable from mass creation.
- A teammate that enables a kill need not be the agent recording the final consumption event.


## V2: tactics as testable decision problems
For each maneuver define preconditions, action sequence, success metric, counterplay and failure label. A split chase is viable only if (i) the projected split fragment can consume the target, (ii) the reachable trajectory intersects the target's feasible escape region, and (iii) the resulting fragment configuration does not create unacceptable counter-consumption exposure. Report success over matched start states rather than anecdotal clips.

### Team-level coordination taxonomy
**Resource allocation:** which player should hold mass; **spatial coordination:** two-sided interception, screen, escape corridor; **temporal coordination:** feed then split, split then merge, hazard timing; **role adaptation:** switch core/support/scout when mass ordering changes. Track total team mass and concentration ratio separately. Feeding can reduce immediate team mass yet increase future consumption odds.

### Tactical benchmark contract
Each scenario declares `scenario_id, setup_hash, legal_actions, scripted_baselines, win_condition, horizon, confounders`. For bait/encircle, verify that neither solo actor can secure the prey under the same enemy policy. For protection, compare the same threat with/without intervention. Include failed and successful trajectories; do not use event labels as reward without a controlled study.


## V3 · 游戏机制完整研究规格

### 1.1 典型一局的生命周期
`LOBBY → RESET → SPAWN → LIVE → SETTLEMENT → DONE`。LIVE 内包含采集、吞噬、分身、吐球、合并、刺球、死亡/重生以及计分。比赛时钟若是模式规则则到期直接终止；训练器超步数或外部取消则为截断。模式可以改变重生、排名、刺球类型、队伍数、资源刷新率，这些必须在 ruleset 中明示。

### 1.2 资产/实体语义
| 对象 | 关键状态 | 交互 |
|---|---|---|
| Player | ID、team、输入、可控球集、重生时间 | 一个 RL Agent |
| Ball | owner、mass、position、velocity、merge timer | 移动、吞噬、分裂、合并 |
| Pellet | mass、position、lifetime | 被吞噬，不属于玩家 |
| Ejected mass | donor、mass、velocity、age | 暂存并转移、可损耗 |
| Virus/hazard | position、size、触发条件 | 改变分身结构/动量 |
| Arena | 尺寸、边界类型、视野 | 限位/可见性 |
| Match | timer、mode、score ledger | 排名、终止、重生 |

### 1.3 动态能力不是总质量
吞噬资格通常应比较**可执行吞噬的单个分身质量与目标质量**，总质量只表示战略资源。分身可能提高射程、降低单球质量；吐球可能损失质量却让队友获得关键战力；合并恢复优势但损失空间覆盖。报告以下三种不同量：`team_total_mass`、`player_largest_ball`、`reachable_target_mass`。

### 1.4 战术操作与评判
| 战术 | 条件 | 原语动作序列 | 成功与失败 |
|---|---|---|---|
| 收集 | 安全区域中存在资源 | 移动→拾取 | 净质量/单位时间；被埋伏算失败 |
| 追击 | 可吞噬目标且速度/路径允许 | 转向→靠近→吞噬 | 目标被吃且无反杀 |
| 逃跑 | 存在可吞噬自身的威胁 | 远离→绕障→求援 | H 步存活，损失最小 |
| 二分突进 | 分裂后的球仍可吞目标 | 接近→分裂→收割 | H 步净收益为正 |
| 连分 | 每次分身后仍合法 | 多次分身→追击 | 违反上限/暴露过高为失败 |
| 中吐 | 多分身位置可向中心输送 | 调方向→吐球→接收 | 质量分布满足目标 |
| 交叉吐 | 分身几何关系满足 | 交叉移动+吐球 | 记录轨迹，不把名字当引擎原子 |
| 合球 | 已过冷却/重叠合法 | 靠近→等待→合并 | 单球质量恢复；计时结束不等于必然合并 |
| 扎刺/推刺 | 刺球存在且可影响 | 吐球/撞击→触发 | 是否产生可利用局势 |
| 诱敌包围 | 对手可被引入约束区域 | 诱敌→两翼封锁→主攻 | 对手无可行逃生路径 |
| 保护队友 | 队友受威胁 | 挡位/喂球/干扰 | 对照下队友存活概率提高 |

### 1.5 5v5 队伍组成
所有玩家共享终局队伍目标，但不一定共享当前最佳动作。质量集中度 `C=max_i M_i / Σ_i M_i` 仅是一个观测指标，并非奖励目标。定义主攻/辅助/侦查/诱敌/保护角色为**可变策略标签**，不是永久 player ID；当质量优势转移时必须允许角色重排。

### 1.6 决策中的博弈特征
玩家面对部分可观测对手，可能无法确认视野外球的质量和位置；对手可能诱敌、佯退、反分身。可训练的策略需根据历史推断风险而非访问隐藏世界状态。多队模式还涉及第三方介入和短期非正式联盟；不假设对手策略固定。

### 1.7 竞赛体系与版本差异
自由混战、组队、短时、存活与特殊模式应独立版本化；计分可能采用排名、质量、剩余生命或模式目标。文档不声明固定时长、队伍数、历史段位属于当下所有模式的官方事实。官方版本对齐必须提供特定 build 与可核实规则来源。

### 1.8 具体对照关卡
- **GM-01 追击**：目标可吞但更快；检查 AI 不做无效直线追逐。
- **GM-02 分身反杀**：短期吞噬盈利但敌方有更大单球；检查长时净收益。
- **GM-03 喂球跨阈值**：队友吸收后首次可吞目标；检查质量账本与协作。
- **GM-04 二人包围**：任一单独行动均无法捕获；计算协同效应。
- **GM-05 刺球风险**：大球触刺后被小球包围；检查危险预测。
- **GM-06 换核**：主力被削弱后次强队友应接管；评估动态分工。
每关至少 20 个保留种子、配对对手、失败重放与结果表；数量是初始测试设计，不构成统计功效保证。

**关联**：[游戏规则](12-rulebook-and-edge-cases.zh-CN.md) · [奖励](06-rewards.md) · [协作评测](09-evaluation.md)。
