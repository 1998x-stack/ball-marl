# 00 · Scope, assumptions, and evidence register

## Research object
A 2D synchronous multi-player ball-consumption game **inspired by** Ball Battle. This repository is not an official implementation and must not imply proprietary source-code access or exact official mechanics.

## Evidence classes
- **Verified**: source-linked publication or reproducible measurement against a declared game build.
- **Proposed**: research rule selected for the simulator.
- **Unknown**: requires empirical measurement, including split impulse, mass transfer efficiency, speed curve, collision penetration, merge timers, maximum fragments, virus effects, and ranking formula.
- **Version-specific**: may change by mode or update.

Every numerical mechanic in a shipped ruleset must have a row in a `rule_registry` containing parameter name, value, units, evidence class, source/build, calibration procedure, and version.

## Modes
Start with a two-team five-player benchmark; extend to free-for-all, multi-team ranking, finite-life survival and special modes only under separately versioned rulesets. Never mix rewards or results from distinct modes without explicit normalization.

## Research hypotheses
H1: variable-sized entity representations outperform fixed ordered truncation under changing populations.
H2: potential-based mass shaping improves sample efficiency without changing the stated objective under its assumptions.
H3: offline simulator counterfactuals recover meaningful event contributors but may be costly and sensitive to baseline choice.
H4: league training improves held-out opponent robustness compared with latest-only self-play.
H5: zero-shot teammate tests expose brittle co-adaptation absent from standard self-play win rates.

## Explicit limitations
Counterfactual results are conditional on intervention definition and rollout policy, not unique causal ground truth. Leaderboard performance on a custom simulator is not proof of official-game skill.


## V2: requirements traceability and governance
Requirement IDs use `REQ-<domain>-NNN`. Each requirement maps to one design section, a concrete automated test, a status and an evidence artifact. Modes are explicitly separated: `research_5v5`, `research_multiteam`, and `calibrated_<build>`; never aggregate their leaderboard results. The research simulator is **not** a verified reproduction of the commercial game.

### Evidence review procedure
Every gameplay constant must include provenance, unit, mode/build, sample count, uncertainty and owner. Unverified parameters fail the `official_parity` gate but may pass the `research_ruleset` gate. Rule changes invalidate dependent golden replays, checkpoints and comparisons. Maintain a change-impact table for physics → observations → rewards → training → evaluation.

### Decision record template
`ADR-ID; context; options; decision; rejected alternatives; invariants; measurable risk; rollback; owner; date`. Major choices (per-player agent identity, simultaneous semantics, deterministic collision resolution, reward ownership, time-limit terminal semantics) each require an ADR. Acceptance: every REQ-P0 maps to a passing test or an explicitly tracked blocker.


## V3 · 需求定义、证据等级与研究边界（详细规格）

### 0.1 产品目标与明确非目标
Ball-MARL 的首要产物是可复现的研究基准，而不是未经验证地克隆商业客户端。首阶段提供纯 Python 参考引擎、同步多人环境、基础智能体、可重复的比赛记录；第二阶段提供分布式训练、联赛和反事实分叉。**非目标**：官方数值一比一复刻、绕过官方客户端机制、实时操纵线上比赛、对商业服务作任何未经允许的接入。

| 需求编号 | 必须交付 | 验收证据 |
|---|---|---|
| REQ-RULE-001 | 版本化游戏规则 | YAML + schema 校验 + 参数证据状态 |
| REQ-PHY-001 | 确定性参考物理引擎 | 同环境同输入每 tick 状态哈希一致 |
| REQ-PHY-002 | 完整质量事件账本 | 全局账本残差在规定容差内 |
| REQ-ENV-001 | PettingZoo ParallelEnv | parallel_api_test 通过 |
| REQ-OBS-001 | Actor 不可获得隐藏信息 | hidden-state perturbation 测试 |
| REQ-ACT-001 | 混合动作与合法性 | action_space.contains、非法输入测试 |
| REQ-REW-001 | 终局目标与 PBRS 分离 | 回报分解与 telescoping 测试 |
| REQ-ALG-001 | 可运行随机/规则/IPPO/MAPPO 基线 | 多随机种子测试报告 |
| REQ-CREDIT-001 | 反事实估计可复现 | 同快照同干预输出一致 |
| REQ-LEAGUE-001 | 历史策略与跨对手评估 | 冻结策略池及比赛表 |
| REQ-EVAL-001 | 分组、配对与统计消融 | 原始逐场记录和置信区间 |

### 0.2 研究对象的形式化
世界状态为 `s_t`，参与者按玩家而非分身定义 `i∈{1…N}`，第 i 名玩家控制变长分身集合 `B_i(t)`。联合动作 `a_t=(a_{1,t},…,a_{N,t})`，转移 `P(s_{t+1}|s_t,a_t)`；局部观察 `o_{i,t}=O_i(s_t,h_{i,t})`。团队收益 `R_k`，训练网络共享参数不改变环境智能体身份。先固定有限时域 2 队×5 人 research 模式，其他模式以单独 ruleset 进行。

### 0.3 假设、证据与决策注册
| 类别 | 例子 | 允许在文档如何陈述 |
|---|---|---|
| VERIFIED-PAPER | MAPPO 在公开基准具有强基线性能 | 附论文及适用场景 |
| VERIFIED-API | ParallelEnv 的 step/reset 语义 | 附官网版本 |
| RESEARCH-ASSUMPTION | 20Hz 物理、5Hz 决策 | 作为默认配置而非官方事实 |
| MEASURED-GAME | 指定客户端 build 的分身速度 | 附采样/残差/置信区间 |
| UNKNOWN | 吞噬阈值、合并机制、玩家视野 | 不得编造，先用可配置假设 |
| DESIGN-DECISION | 玩家级 Agent/质量账本 | ADR、测试与兼容性声明 |

必须为每条会影响行为的规则保留 `{id,unit,mode,build,status,source,confidence,calibration_id}`。一个模式的测量不能自动泛化到另一个模式。任何阈值变化都要求递增 ruleset 版本。

### 0.4 规范性语言
**MUST** 是测试必过；**SHOULD** 是推荐并记录例外；**MAY** 是实验项。程序入口通过声明式配置选择策略，禁止用隐藏的环境变量改变动力学。策略训练使用训练集，超参选择用验证集，最终论文结论只使用封存的测试集。

### 0.5 定义与术语
- **Mass**：物理守恒账本里的标量；不是积分或排名。
- **Ball fragment**：玩家控制的一枚球，玩家 ID 不因分裂变化。
- **Consume**：原子性所有权/质量转移，不等同普通碰撞。
- **Feed**：吐球并可能被队友拾取；不是质量凭空增长。
- **Potential shaping**：按 `γΦ(s')−Φ(s)` 计算、受终止条件约束的奖励变换。
- **Counterfactual baseline**：指定替代动作及其后续策略，而不是唯一真相。
- **Episode termination**：真实比赛规则结束；**truncation**：采样器/资源限制截断。
- **Deterministic reference backend**：限定执行平台/版本的逐 tick 可复现引擎。

### 0.6 文档交付与风险登记
每章至少给出：接口、状态机、约束、测试、开放问题、依赖、验收标准。P0 风险包括高速穿透、吞噬重复结算、全局信息泄露、PBRS 边界错误、训练/评测随机种子混用、反事实随机流分叉。对于未知机制采取“锁定模拟规则 + 测量计划”的方式推进，而非无限等待官方常数。

### 0.7 阶段验收
S00：所有 P0 规则有配置及显式状态（PROPOSED/MEASURED/UNKNOWN）、且不再存在隐式默认值；S01：确定性+质量账本；S02：环境 API 与观察隔离；S03：规则对照；S04：MAPPO；S05–S07：自博弈、信用、消融。文档签收仅代表接口定义明确，不代表代码或训练已经交付。

**关联**：[物理](02-physics-and-rules.md) · [API](03-environment-api.md) · [测试路线](10-engineering-roadmap.md) · [中文实验](16-experiment-reproducibility-and-ci.zh-CN.md)。
