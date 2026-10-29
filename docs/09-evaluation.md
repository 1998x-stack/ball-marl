# 09 · Tournament, causal scenarios and ablations

## Primary metrics
Champion fraction, average rank, head-to-head win rate, rank uncertainty, per-mode success. Secondary: team mass trajectories, survival and respawn, clean consumption, split net gain by horizon, transfer efficiency, positional control and invalid actions. Always report dispersion.

## Paired protocol
Use paired initial conditions, shuffled spawn positions, matched environment RNG, both sides of team assignment, and multiple independent training seeds. Present uncertainty intervals and minimum meaningful effect sizes; do not claim improvement from overlapping or low-powered tests.

## Cooperation benchmark
- Feed-to-enable: one agent must transfer enough mass for ally to gain consumption eligibility.
- Two-agent encirclement: neither isolated player can secure the target.
- Rescue: team must reduce ally loss risk, accepting an opportunity cost.
- Dynamic role swap: team's former core loses mass; another player becomes primary.
- Delayed split trap: immediate split yields positive mass but poor H-step net return.
- Hazard-assisted attack: hazard timing changes enemy fragmentation and counterattack risk.

Include at least one matched non-cooperating policy baseline and replayable scenario snapshots for each. Report whether the objective is genuinely achievable without coordination before using a test as evidence of cooperation.

## Ablation matrix
Observation: fixed MLP vs Deep Sets vs attention vs GRU.
Action: move only vs split vs eject vs masking vs high-level options.
Reward: terminal vs naive mass vs PBRS vs PBRS+aux.
Credit: shared MAPPO vs auxiliary counterfactual vs COMA-like independent method.
League: current-only vs history vs history+exploiters.

Do not change model size or simulator budget invisibly between ablations; explicitly report any mismatches.

## Zero-shot generalization
Hold out team policies, adversaries, map sizes, food distributions, start mass regimes, and team population sizes. Evaluate teammate replacement and role permutation. Shared actors may still overfit a fixed roster.

## Reproducibility gate
Every published plot references manifest, raw match results, code commit, checkpoint digest, ruleset and CI status. Disclose failed seeds and simulator exploits.


## V2: preregistered tournament and statistics
Define unit of independence: independent training seed for algorithm comparisons; match seeds are nested repeated measurements, not substitutes for training seeds. Use side-swapped paired matches, same spawn/resource RNG and a held-out opponent pool. Report raw wins/draws/losses, average rank, confidence intervals, and paired effect sizes; do not infer algorithm superiority from overlapping single-run learning curves.

### Scenario scoring
Feed-to-kill: measure team consumption probability and transferred net mass under intervention. Encirclement: compare joint vs isolated interventions; ensure map boundary is not the true cause. Delayed split: compare H=1 and H=100 net return. Rescue: survival probability and opportunity cost. Zero-shot: unseen teammates, opponents, maps, team sizes and mass distributions.

### Failure taxonomy
`RULE_BUG, OBS_LEAK, ACTION_INVALID, REWARD_HACK, COLLISION_NONDET, TRAIN_DIVERGENCE, OPPONENT_OVERFIT, CREDIT_UNSTABLE`. Every published failure gets a replay hash, event log and reproduction command. All figures must specify confidence method, denominator, ruleset and checkpoint selection policy.
