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
