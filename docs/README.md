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
