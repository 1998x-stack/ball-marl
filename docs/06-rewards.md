# 06 · Reward specification and invariants

## Primary objective
The benchmark objective is the **team ranking under a declared ruleset**. For N>1 teams and no ties, terminal reward R_T = 1 - 2*(rank-1)/(N-1). Ties must be resolved by a published tie policy (shared mean rank or fixed rule); no hidden arbitrary tiebreaker. For two teams use +1/-1 and optionally 0 for draw.

## Potential-based reward shaping (PBRS)
Let M_team be the sum of all active and, if modeled, recoverable team-owned mass under a documented ownership definition. Proposed potential Phi(s)=c*log(1+M_team/M_ref). Per transition F(s,s')=gamma*Phi(s')-Phi(s). For terminal states set Phi(s_terminal)=0; truncated rollouts are **not** terminal and require value bootstrapping. Discounted telescoping makes additive PBRS differ by a state-dependent constant from the original discounted objective under the relevant conditions; it does not validate arbitrary mass-growth bonuses or changing gamma mid-run.

The simulator must include every transfer, sink, respawn grant and expiration in a mass ledger. Friendly feeding must not become a new team-growth event. Negative potential changes are permitted; do not clamp them.

## Three separated channels
1. `objective_reward`: terminal ranking, always reported unchanged.
2. `shaping_reward`: versioned PBRS; beta/c chosen prior to a run.
3. `aux_targets`: outcome predictions (future mass delta, enemy consumed, split net outcome, ally support need) used in supervised auxiliary losses, **not** automatically converted into reward.

Do not introduce unconditional per-timestep survival bonuses or fixed kill bonuses in the main benchmark: both can change behavior and obscure win-rate evaluation.

## Reward auditing
Assert: (i) team internal mass transfers alone do not increase total mass; (ii) shaped return telescopes on fixed trajectories, including terminal handling; (iii) ranks and tie rewards are correct; (iv) termination/truncation distinct; (v) no exploitable repeated eject/merge loop creates positive objective rewards; (vi) objective and shaping are separately logged.

## Ablations
R0 terminal only; R1 terminal+naive delta mass (deliberately non-invariant comparator); R2 terminal+PBRS; R3 PBRS+auxiliary predictions. Report fixed-seed win rate, return components, strategy changes, simulator calls and uncertainty.
