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
