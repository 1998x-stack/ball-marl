# 10 · Implementation roadmap, architecture and acceptance

## Suggested directories
```text
ball-marl/
  docs/
  configs/rulesets/
  engine/{world,physics,collision,events,snapshots}/
  envs/{pettingzoo,scenarios}/
  observations/{visibility,entities,maps}/
  actions/{interpreter,masks}/
  rewards/{objective,potential,ledger}/
  algorithms/{ippo,mappo,coma}/
  credit/{counterfactual,attribution}/
  league/{pool,matchmaking}/
  evaluation/{tournaments,ablations,generalization}/
  tests/{unit,property,integration,regression}/
```

## Stage gates
| Stage | Deliverable | Exit gate |
|---|---|---|
| S00 | mode/rule/observation/action/reward specifications | every parameter classified proposed/measured/unknown; collision tie cases specified |
| S01 | deterministic engine, snapshots and mass/event ledger | mass reconciliation and replay hash tests pass |
| S02 | PettingZoo ParallelEnv and scenario manifests | API compliance, reset/step seeds, actor visibility contract |
| S03 | scripted bots/IPPO | scripted benchmark reproducible with CI |
| S04 | shared-reward MAPPO | multiple seeds, stronger than chosen scripted baselines with intervals |
| S05 | historical opponent league | held-out opponents and no catastrophic forgetting |
| S06 | controlled counterfactual replay | repeatable matched scenarios and cost report |
| S07 | full ablation and zero-shot test | reproducible report with manifest and artifacts |

## Critical automated checks
- Mass ledger global and per-team reconciliation including respawn grants, transfer loss and consumed balls.
- Contact-order permutation and simultaneous-consumption deterministic tie-break.
- Fragment cap, split cooldown, eject minimum mass and merge timer.
- Fixed agent IDs through fragment changes and death/respawn.
- Repeated seeded simulation produces identical per-tick hashes on supported backend.
- Hidden-object perturbation leaves Actor observations unchanged.
- Permutation invariance of set encoder and mask correctness.
- PPO log-prob and action distribution finite under all legal masks.
- PBRS discounted telescoping, terminal potential zero, truncation bootstrap.
- Counterfactual branches share initial snapshot and RNG but do not contaminate original trajectory.
- Replay artifact hashes and configuration version embedding.

## Milestone acceptance evidence
For each milestone require source commit, exact CLI command, CI result, sample replay, metrics JSON, known limitations and sign-off. **Documentation completion does not imply these stages were implemented.**
