# 04 · Observation and model specification

## Three distinct objects
- World state S: authoritative complete simulator state.
- Actor observation O_i=f_visible(S, history_i): only currently visible information and legally remembered history.
- Critic state C_i: optionally global training information; not accessible to deployed Actor.

## Entity-centric schema
Fragment fields: relative position and velocity, log1p mass, radius, owner relation, merge cooldown, split eligibility, visible mask. Player fields: relative position, **visible** aggregate mass, strongest **visible** fragment, visible count, last-seen age and confidence; never leak invisible fragments through aggregate features. Resource/hazard fields: type, position, motion, strength and visibility.

Add normalized arena boundaries, local 32x32 multi-channel occupancy/density grid, and optional last observed events. Use stable IDs only for tracking, not arbitrary enemy sorting features.

## Dynamic entities
Use masks and Deep Sets as the baseline. Attention/graph nodes are later variants. If capping entity count, retain urgent collision threats plus nearest neighbors and aggregate the remainder; log truncation fraction and missed high-threat frequency. Do not present hidden threat estimates as truth.

## Network progression
(1) MLP fixed-neighbor baseline; (2) Deep Sets + map CNN; (3) GRU; (4) entity attention/graph attention. Critic separate encoders if needed. Policy shared across homogeneous agents; conditional agent role embeddings must not contain oracle data.

## Audit
Counterfactual observation test: modify an invisible enemy state without affecting visible dynamics and assert Actor observation unchanged. Check permutation invariance/equivariance under entity reorder and team-name remapping.


## V2: feature governance and threat-aware selection
For each feature specify source (visible object, own internal state, public UI, inferred history), normalization, mask, missing semantics and test. Enemy total mass must not be inferred from hidden fragments as exact. Public scoreboard information is allowed only if the simulated game mode actually exposes it.

### Threat score for capped entities
Use a **non-oracle** selection score such as `w1 / (distance+eps) + w2 * can_consume_me + w3 * predicted_intercept_probability`; all terms derive from observed positions, observed masses and known mechanics. Preserve the top-K imminent threats, then nearest remaining entities; compress excluded entities into local density statistics. Track `threat_missed@K` using privileged evaluation labels only, never as Actor input.

### History
Store last-seen positions with age and uncertainty growth; do not refresh unseen entities from WorldState. Recurrent hidden state resets at episode boundary; respawn handling is a separately versioned policy. Validate observation under global translation/rotation only where arena symmetries permit. Test permutation invariance for sets, masking for padded entities, normalization under mass extremes, and no NaN from empty entity groups.


## V3 · Observation Schema、可见性与表征网络

### 4.1 三类观察视角
- `WorldState`：全量球、玩家、资源、计时和 RNG，物理唯一真相。
- `ActorObs_i`：玩家在当前模式合法可见的画面/UI、本人分身、合法历史记忆。
- `CriticState_i`：训练时可用的集中式世界特征，不随 Actor 一起导出。

不允许通过 `last_seen`、资源密度网格或队友总质量聚合将隐藏的真实敌人分身信息泄漏给 Actor。分身的总质量只有所有分身合法可见时才能精确统计。

### 4.2 完整数据字段和 shape
| 张量 | 建议 shape | 数据类型 | 字段及备注 |
|---|---|---|---|
| `self_balls` | [16,12] | float32 | 相对质心 dx/dy、vx/vy、log(m)、r、merge_left、split_ready、heading_x/y、mass_fraction、boundary_dist |
| `self_mask` | [16] | bool | 填充位 False |
| `visible_balls` | [64,14] | float32 | dx/dy、vx/vy、log(m)、r、owner_relation、can_eat、can_be_eaten、closest_dist、closing_speed、age、team_visible、threat_score、visible_flag |
| `visible_mask` | [64] | bool | 可见实体 |
| `teammates` | [4,10] | float32 | 相对位置、可见质量/估计值、活跃状态、分身数、通讯/公共 UI 信息 |
| `teammate_mask` | [4] | bool | 队友存在 |
| `resources` | [128,8] | float32 | dx/dy、类型、mass、velocity、visibility 等 |
| `resource_mask` | [128] | bool | 有效资源 |
| `local_grid` | [8,32,32] | float32 | 食物密度、敌方质量、友方质量、刺球、边界、逃生空间、运动趋势等 |
| `scalars` | [24] | float32 | 当前时间、剩余时间、自身最大球、自身质量、威胁等 |
| `split_mask/eject_mask` | [2] each | bool | 分别限制 no-op 和触发操作 |

字段设计必须形成机器可读 schema，不能出现上层写 `self_balls[:,3]` 代表质量、下层却改变字段顺序的破坏性修改。建议使用 `schema_version` 和 SHA256。

### 4.3 规范化策略
位置相对于自身质量加权质心除以 `view_radius`；速度除以 `speed_ref`；质量使用 `log1p(m)/log1p(m_ref)`；时间除以规则里的最大时间。所有参考尺度固定于 ruleset/obs schema，不从当前比赛动态最大值回归，否则会让同一位置/质量因对手变化而产生尺度漂移。

### 4.4 变长实体选择
观察对象可能超过上限 K：首先保留即将发生碰撞、可吞或威胁最大的 Top-K，再保留近邻，余下聚合进栅格密度。**截断规则不能用隐藏的未来真实轨迹**。评测阶段使用全局真相计算 `danger_missed@K`，但不让 Actor 读取该值。维护 `num_visible, num_selected, dropped_by_type`。

### 4.5 历史与不确定性
用固定长度记忆缓存最后看见的敌人位置、速度、时刻以及身份关联可靠性。历史推测值用独立 `is_estimate` mask 标记，并让估计方差随 `last_seen_age` 增加；未知实体不能凭空获知真实位置信息。GRU 在 Episode 真终止时清零，重生事件是否重置由版本固定并做消融。

### 4.6 网络设计与参数共享
Baseline 1：定长近邻 + MLP；Baseline 2：对球/资源分别使用 Deep Sets `pool_j mask_j φ(x_j)`，再接地图 CNN；Baseline 3：加 GRU(128)；研究项：多头 Attention/GNN，用 edge 特征表示距离、质量比、拦截时间。若队友角色异质，可给角色编码，但不按固定 PlayerID 进行隐式超参数特权。

### 4.7 实现伪代码
```python
def build_actor_obs(world, player_id, history, schema):
    visible = visibility_filter(world, player_id)
    self_balls = encode_self(world.player(player_id), schema)
    enemies = select_entities(visible.enemy_balls, schema.enemy_cap)
    allies = encode_visible_allies(visible.allies)
    grid = rasterize_visible_only(visible, schema.grid_shape)
    obs = pad_and_mask(self_balls, enemies, allies, grid)
    assert schema_contains(obs) and all_finite(obs)
    return obs, update_history_from_visible(history, visible)
```

### 4.8 必测性质
- 修改视野外敌人状态不改变当下 Actor 观察。
- 改变输入集合顺序不改变 Deep Sets 编码。
- 所有掩码都为 False 时无 NaN。
- 同场景旋转/镜像在对称规则下保持相应等变/不变性质。
- 特征中没有未授权绝对真相。
- `ActorObs` 不携带 target 的未来动作或评测标签。

**关联**：[张量详细规范](13-observation-action-tensor-contract.zh-CN.md) · [动作](05-actions.md) · [评测](09-evaluation.md)。
