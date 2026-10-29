# Implementation status · 2026-10-29

## Implemented and locally verified

- Configuration loading and proposed reference ruleset.
- Synchronous deterministic world reset/step, controlled ball fragments, split/eject/merge/consumption, neutral pellets, stable entity identities, snapshot/restore and mass ledger.
- Gymnasium/PettingZoo-compatible surface API with optional dependency fallback; **official PettingZoo compliance not yet run**.
- Fixed-size locally visible observation and action masks. Central critic has privileged global state.
- Shared-parameter MAPPO training against scripted opponents; hybrid tanh-Gaussian + categorical stochastic actor; clipped PPO + GAE.
- Model save/load, deterministic inference, held-out evaluation, event traces, open-loop mass counterfactual for a single agent.
- Smoke and regression tests via `python -m pytest -q`.

## Not implemented / research limitations

- Official gameplay calibration, hazards/virus, respawn, third-party mass transfer, network synchronization, advanced gameplay modes and continuous collision detection.
- Full hierarchical skills, graph encoder, recurrent policy, multi-population self-play, COMA, Shapley aggregation, closed-loop counterfactual and validated causal inference.
- Official PettingZoo validation due to missing optional local packages. Compatibility layer is not a substitute for API certification.
- Statistically significant RL improvement over established baselines; smoke learning is for integration only.

## Next implementation gates

1. Validate env with real installed PettingZoo/Gymnasium `parallel_api_test`.
2. Upgrade physics (hazard/virus, fragment steering, split impulse, CCD), and add golden replays.
3. Test 5v5 scaling and optional dynamic player handling.
4. Add historical opponent self-play pool and checkpoint evaluator.
5. Add data-centric matched interventions, confidence intervals and multiple training seeds.
6. External source-backed calibration per ruleset and ablation report.
