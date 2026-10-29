# 01 · Mechanics and competitive strategy

## Entities
A **team** contains stable **player agents**; each player controls a dynamically changing set of **ball fragments**. Other entities: pellets, ejected mass, hazards/viruses, boundaries, respawn points and optional special-mode objects.

## Primitive interactions
1. **Movement** directs the player's controlled fragments within the ruleset.
2. **Consumption** transfers victim mass to an eligible larger ball when both mass-ratio and overlap requirements are met.
3. **Split** redistributes one or more fragments' mass and imparts an initial displacement/impulse; may increase reach while reducing per-fragment safety.
4. **Eject/feed** removes mass from a source and creates transferable matter; accounting must capture inefficiency.
5. **Merge/recombine** reunites eligible fragments after the configured timer and overlap conditions.
6. **Hazards** may disrupt concentrated balls and create smaller vulnerable fragments.
7. **Team interactions** may transfer mass to allies where allowed; distinguish friendly absorption from enemy elimination.

## Strategy taxonomy
| Strategy | Required state | Benefit | Exposure |
|---|---|---|---|
| Pellet farming | local resource map | low-risk growth | opportunity cost |
| Split pursuit | fragment mass, geometry, enemy velocity | intercept reach | counter-consumption |
| Escape | threat geometry, boundaries | survival | lost resources |
| Feed core | ally geometry, compatible mass | concentrated attack ability | transfer loss, exposed support |
| Two-core formation | ally strengths and separation | flanking and redundancy | slower concentration |
| Hazard push | hazard location and ejection | disrupt enemy | counterplay |
| Bait / encircle | team coordination | restrict escape | coordination failure |

## Strategic tradeoffs
Let total team mass be M and largest teammate share c=max_i(M_i)/M. Neither maximizing c nor minimizing c is globally optimal: concentrated attack power trades off independent mobility, spatial coverage and single-point failure.

Do not assume split is profitable from immediate consumption alone. Evaluate net outcome at horizons H, including whether the fragmented attacker is later consumed. Distinguish win probability, ranking, total mass, survival and team control as separate outcomes.

## Scenario acceptance examples
- Direct pursuit of a faster small ball does not imply capture.
- A split must be checked against third-party counterattack opportunities.
- Friendly mass transfer cannot count as newly created team mass.
- Hazard-triggered fragmentation is distinguishable from mass creation.
- A teammate that enables a kill need not be the agent recording the final consumption event.


## V2: tactics as testable decision problems
For each maneuver define preconditions, action sequence, success metric, counterplay and failure label. A split chase is viable only if (i) the projected split fragment can consume the target, (ii) the reachable trajectory intersects the target's feasible escape region, and (iii) the resulting fragment configuration does not create unacceptable counter-consumption exposure. Report success over matched start states rather than anecdotal clips.

### Team-level coordination taxonomy
**Resource allocation:** which player should hold mass; **spatial coordination:** two-sided interception, screen, escape corridor; **temporal coordination:** feed then split, split then merge, hazard timing; **role adaptation:** switch core/support/scout when mass ordering changes. Track total team mass and concentration ratio separately. Feeding can reduce immediate team mass yet increase future consumption odds.

### Tactical benchmark contract
Each scenario declares `scenario_id, setup_hash, legal_actions, scripted_baselines, win_condition, horizon, confounders`. For bait/encircle, verify that neither solo actor can secure the prey under the same enemy policy. For protection, compare the same threat with/without intervention. Include failed and successful trajectories; do not use event labels as reward without a controlled study.
