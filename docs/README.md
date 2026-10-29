# Ball-MARL · Research and Engineering Documentation

Status: **design specification, not an implemented or experimentally validated simulator**.

## Goals
A reproducible, research-grade, multi-agent ball-consumption environment with team cooperation, split attacks, feeding, counterfactual credit analysis, and out-of-distribution evaluation.

## Reading order
1. [Scope and evidence](00-scope-and-evidence.md)
2. [Game mechanics](01-game-mechanics.md)
3. [Physics and rules contract](02-physics-and-rules.md)
4. [Environment and API](03-environment-api.md)
5. [Observations and networks](04-observations.md)
6. [Actions and timing](05-actions.md)
7. [Rewards](06-rewards.md)
8. [Credit assignment](07-credit-assignment.md)
9. [Training and league](08-training.md)
10. [Evaluation and ablations](09-evaluation.md)
11. [Engineering, tests, milestones](10-engineering-roadmap.md)
12. [References](11-references.md)

## Non-negotiable research rules
- **No invented official parameters.** The default implementation is an explicitly named research ruleset; official parity is a separately measured claim.
- **One agent controls a player, not each split ball.**
- **Actors receive only legal visible observations.** Centralized critics may see global training information.
- **Final objective is game outcome;** dense shaping is separately versioned and audited.
- **Counterfactual attribution is not automatically a valid PPO advantage.**
- **Deterministic event logs and reproducible snapshots precede causal claims.**
- **Report wall time and simulator calls as well as environment steps.**

## Design status
All numeric defaults are proposed experimental values, subject to calibration. Track ruleset, observation, action, reward, engine, and training configuration hashes in every experiment.


## 深入设计（中文，优先阅读）
- [12｜游戏规则契约、物理边界、黄金测试](12-rulebook-and-edge-cases.zh-CN.md)
- [13｜状态张量、动作概率、时间与可见性契约](13-observation-action-tensor-contract.zh-CN.md)
- [14｜奖励数学证明、漏洞分析与单元性质测试](14-reward-math-and-exploits.zh-CN.md)
- [15｜反事实信用分配实验协议](15-counterfactual-credit-protocol.zh-CN.md)
- [16｜实验复现、数据格式、统计与 CI](16-experiment-reproducibility-and-ci.zh-CN.md)

## 阅读及落地建议
中文 12–16 章细化英文 01–10 章的工程边界。若有冲突，以明确标注的研究假设和可测试的契约为准，提交 Issue 统一修订；不可默默选用某个版本。先执行 S00 规则标定与 S01 引擎验证，再训练 S04 MAPPO。所有内容目前为设计文档，不代表已经通过真实游戏标定或运行实验。
