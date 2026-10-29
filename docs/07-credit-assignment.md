# 07 · Credit assignment: estimands before algorithms

## Three different questions
**Policy-gradient credit**: whether sampled action improved expected return relative to a valid baseline. Standard shared-reward MAPPO uses per-agent GAE with centralized V, not necessarily identical advantages.

**Simulator counterfactual contribution**: finite-horizon return change after intervening on agent i's action while defining how subsequent participants act.

**Synergy**: whether joint interventions yield more return than the sum of isolated interventions. These objects are not interchangeable and are not guaranteed to sum to observed team return.

## Baseline MAPPO
All five teammates share objective team reward; a centralized value V(s,i) gives GAE for each agent. Use recurrent state masks when deaths, respawns, and rollouts reset. Do not feed post-action events into a pre-action Actor.

## Difference rewards
D_i^H = G_H(s_t, a_t; followup) - G_H(s_t, (a_i^0,a_-i); followup).
Specify H in policy steps and baseline a_i^0:
- no-op (sometimes misleading);
- continue previous movement;
- action sampled from frozen policy (averaged across samples).
Use identical starting snapshot and matched random-number streams. Compare **open-loop** continuation (fixed other commands) and **closed-loop** continuation (all agents replan). Both are conditional measurements, not unique ground-truth causal responsibility.

## COMA-style alternative
Train joint Q(s,a). For discrete agent actions, A_i = Q(s,a) - sum_{a_i'} pi_i(a_i'|o_i) Q(s,(a_i',a_-i)). Continuous movement requires integration/Monte Carlo or an appropriate model. This is an **algorithmic alternative** to baseline MAPPO, not simply an extra reward added to GAE.

## Event attribution ledger
Capture mass_transfer, split_attack, enemy_consumed, hazard_triggered, teammate_saved_candidate, encirclement_candidate with timestamps, actor/target IDs, snapshots, mass deltas and provenance. Candidate contributors are hypotheses; do not mark them proven merely by co-occurrence.

For pair synergy I_ij = G(a_i,a_j) - G(a_i^0,a_j) - G(a_i,a_j^0) + G(a_i^0,a_j^0), holding all other conditions consistently fixed. Include uncertainty and baseline sensitivity.

## Research protocol
V1 standard MAPPO; V2 offline counterfactual reports; V3 auxiliary counterfactual value estimation with unchanged primary PPO objective; V4 independent COMA-style comparison; V5 long-horizon team synergy. For each, log intervention coverage, simulation cost, baseline stability, policy performance, confidence intervals and failure modes.

## Failure modes
Do not count teammate feeding as new team mass; do not double-count overlapping events; do not simply sum individual differences as a budget-balanced allocation; do not claim do-operator causal identification from an unvalidated simulator; do not use hindsight contributor labels in the Actor.
