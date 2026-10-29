# 02 · Physics engine and authoritative rules

## Model (proposed, not official)
For mass m and surface density rho, r=sqrt(m/(pi*rho)); normalize rho=1/pi for r=sqrt(m). A configurable monotone speed curve is v(m)=v_min+k/(m+m0)^alpha, with alpha>0. These equations are candidate models awaiting calibration.

For balls i,j, center separation d_ij=norm(x_i-x_j). Geometric circle intersection uses d_ij<r_i+r_j; consumption is **not equivalent** to contact. Candidate eligibility requires m_i>=lambda*m_j and sufficient penetration such as d_ij <= r_i-beta*r_j. lambda and beta are explicit ruleset parameters, **not asserted official constants**.

## Step ordering contract
1. Validate simultaneous player commands at policy boundary.
2. Convert edge-triggered split/eject commands into queued events.
3. For each physics substep: integrate motion and impulses, constrain boundaries, generate broad-phase contacts, deterministically sort candidate interactions, resolve eligibility and ownership, settle hazards and timers, emit events.
4. Aggregate measurements, compute rewards after state transition, produce observations and termination flags.

Define tie-breaks for simultaneous consumption, multi-victim contacts, allied transfer and same-tick elimination. Stable numeric entity IDs and contact ordering are mandatory. Prefer semi-implicit Euler initially; integration method, dt and collision tolerance are versioned.

## Mass ledger
For each substep record:
M_after - M_before = spawned_pellets + spawned_hazard_mass + respawn_grants - sinks - decay - expired_ejections.
Transfers between agents, splitting, merging and ordinary consumption reallocate mass rather than create it. If a hazard adds mass, explicitly record its grant. Assert both global and team-level reconciliations, accounting for cross-team consumption.

## Proposed baseline
20 Hz physics, 5 Hz policy, 16 maximum fragments per player, fixed arena and configurable 180-second episode. These are experiment settings rather than live-game measurements.

## Determinism
Snapshot positions, velocities, fragments, ownership, timers, pending inputs, random generator state, event counters and configuration hashes. Validate replay hashes at every policy tick, including tie cases; record platform/backend where floating-point behavior may differ.

## Calibration protocol
For each mechanic record source build and mode, reproducible video timestamps, input sequence, measured values with error bounds, fitting loss, held-out trajectories and decision to accept/reject. Until calibrated label every parameter as PROPOSED.


## V2: implementable physics pseudocode
```python
def advance(world, commands, n_substeps):
    validated = validate_against_pre_action_state(world, commands)
    for k in range(n_substeps):
        apply_due_inputs(world, validated, substep=k)
        integrate_motion(world, dt=world.rules.dt)
        enforce_bounds(world)
        pairs = broadphase_spatial_hash(world.active_balls)
        contacts = sorted(narrowphase(pairs), key=stable_contact_key)
        for contact in contacts:
            if world.entities_alive(contact.ids):
                resolve_atomically(world, contact)
        process_hazards_and_timers(world)
        settle_mass_ledger(world)
        assert_world_invariants(world)
    return world
```
Spatial hashing cell width should be calibrated to the largest relevant interaction radius; large circles may occupy many cells. Deduplicate pairs before narrow phase. For N balls, naive all-pairs costs O(N²); grid broad phase is workload dependent, not a guaranteed O(N) algorithm. Use continuous collision detection or substepping if tunneling appears at high split impulse.

### Numeric and collision contracts
Positions/velocities use float64 for reference backend, float32 optional only after tolerance validation. Define strict/weak inequality at every threshold, stable tie-break for simultaneous prey capture, minimum positive mass, finite coordinates, deterministic event order and clamped boundary projection. A mass transfer is an atomic debit/credit with source/sink ledger entries. A ball marked consumed cannot collide again within the tick.

### Replay serialization
Canonical snapshot includes sorted entity records, RNG streams, pending input queue, timers, next entity ID, ruleset digest and floating-point backend metadata. Canonical hash uses stable serialization and specified float encoding. Replaying a different backend may be tolerance-equal without bitwise identity; never claim bitwise reproducibility across unsupported platforms.


## V3 · 物理与规则的可执行细化

### 2.1 单位与状态向量
坐标 `(x,y)` 以 world units 记录；时间用秒；质量用 simulator mass units；半径由 `r=sqrt(m/(πρ))` 定义；速度 `v_x,v_y` 用 units/s。浮点数必须 finite，质量严格大于零，世界时间 `tick` 为整数、时间由 `tick*dt` 推得而不逐次累加浮点误差。可复现参考引擎优先 float64，快照明确字节序和格式。

### 2.2 运动学与速度模型
研究假设（非官方）：`vmax(m)=v_min+k/(m+m0)^α`，`α>0`。输入方向 `u` 归一化后形成目标速度；定义惯性模式 A（立即对齐）和模式 B（受最大加速度约束），不能同一 ruleset 中隐式切换。半隐式 Euler：`v_{t+1}=clip_speed(v_t + a_t dt)`，`x_{t+1}=x_t+v_{t+1}dt`。分身的冲量项和摩擦衰减单独处理。输入 `u=(0,0)` 时是否惯性滑动由 ruleset 决定。

### 2.3 碰撞几何、吞噬和仲裁
候选碰撞首先满足 `distance<=r_i+r_j`；真正吞噬还须 `m_pred >= λ m_prey` 且 `distance<=r_pred−β*r_prey`。若右侧为负则吞噬条件必失败；不能因浮点 roundoff 把同质量同中心当合法。等于阈值的比较符为独立配置。处理多个捕食者争夺同猎物采用：计算连续首次接触时刻（若支持 CCD），否则选择距离/质量/稳定 ID 组成的稳定键；禁止同 tick 重复消费已 tombstone 的猎物。

### 2.4 空间索引、复杂度与穿透
参考版可从 O(N²) all-pairs 开始锁定正确性，然后引入 uniform-grid broadphase；格宽与最大半径相关，大球跨多格，需要对 pair 去重。基准记录 `num_balls, candidate_pairs, true_contacts, substeps, wall_ms`。高速分身造成 tunneling 时，采用自适应子步或 swept-circle CCD；加入 1×、2×、4× 时间分辨率的结果一致性试验。

### 2.5 分身与吐球的原子事务
```text
split_request → validate(source_ball, mass, cap, cooldown)
→ debit source mass → allocate new BallID → set impulses
→ collision participation on configured tick boundary → ledger
```
分裂前后总质量差等于显式损耗；质量不足或达到上限则无副作用并记录拒绝原因。eject 类似，先扣源质量，再生成漂移实体；源球不得扣到负值。队友吞噬孢子属于同团队所有权转移，不得计入新生成质量。

### 2.6 合并、刺球、重生与资源刷新
合并必须同时满足 owner 相同、计时器结束、接触/重叠条件和模式规则。刺球效果先检查触发资格，再以事务拆分目标（可能发生质量源/汇，需要独立账本）。重生注入的初始质量必须登记外源流入。资源刷新使用可重播的独立 RNG 流 `rng_spawn`，不允许碰撞分支改变资源随机序列。

### 2.7 单次 policy tick 的事务处理
```python
def policy_tick(world, actions):
    snapshot = world.snapshot()
    commands = validate_and_queue(snapshot, actions)
    for substep in range(world.rules.substeps):
        advance_motion_and_impulses(world, commands, substep)
        resolve_bounds(world)
        contacts = sorted(generate_contacts(world), key=contact_order)
        for c in contacts:
            if c.entities_still_eligible(world):
                apply_atomic_collision(world, c)
        run_hazard_and_merge_transitions(world)
        record_external_mass_flows(world)
        audit_mass_ledger(world)
    terminal = evaluate_game_outcome(world)
    return transition_record(world, commands, terminal)
```
注意：若“移动前分裂”或“吞噬前合并”有不同规则，必须分别写成 ruleset 变体和黄金回放。伪代码的顺序即规范，不得在实现中悄悄改变。

### 2.8 双层质量账本
`global_before + external_in − external_out = global_after`；队伍级 `team_before + cross_team_in − cross_team_out + external_in − external_out = team_after`。事件至少包含 `tick,event_id,source_entity,target_entity,team_from,team_to,mass_before,mass_after,delta,reason`。对总质量差设置绝对+相对容差，并把残差写入训练日志；训练中出现超限就 fail fast。

### 2.9 Snapshot 与回放
完整快照：所有 ball/player/resource/hazard、速度、计时器、pending events、EntityID 计数器、各 RNG 流、ruleset hash、backend fingerprint。排序序列化后 SHA256。相同 build/平台保证逐 tick bitwise；跨平台允许在单独浮点容差测试中报告“数值近似相等”，不可宣称完全确定。

### 2.10 性质测试
P01 无负质量；P02 合并与分裂质量闭合；P03 镜像配置下状态镜像等价（规则无破坏对称时）；P04 重排实体内部存储不改变结果；P05 同 tick 双吃只记一次；P06 分身不得离开合法边界；P07 大冲量无穿透；P08 同快照重放一致；P09 单步质量账本残差达标；P10 资源/重生流入有来源记录。

**关联**：[详细规则](12-rulebook-and-edge-cases.zh-CN.md) · [环境 API](03-environment-api.md) · [回放与 CI](16-experiment-reproducibility-and-ci.zh-CN.md)。
