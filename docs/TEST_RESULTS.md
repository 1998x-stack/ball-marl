# Verified local test evidence · 2026-10-29

## Environment

- Platform: local Linux container, CPU-only PyTorch 2.10.0+cpu, NumPy 2.3.5, PyYAML 6.0.3, pytest 9.0.2.
- PettingZoo and Gymnasium were **not installed** locally; the official `parallel_api_test` was **skipped**, not passed.
- Ruleset: `research-v0.1`, explicitly **proposed**, not commercially calibrated.

## Automated suite

`python -m pytest -q`

**Result: 18 passed, 1 skipped** (optional PettingZoo compliance). Tested mass ledger, deterministic snapshot replay, split cap, ejection loss, collisions, invalid inputs, bounds, visibility leakage, PBRS telescoping, true termination, action spaces, actor log-probability recomputation, MAPPO train/checkpoint/load/inference/evaluation and counterfactual fork isolation.

## Full pipeline executed

```bash
python -m training.cli train --episodes 4 --steps 30 --seed 2026 --out runs/verified
python -m training.cli evaluate --checkpoint runs/verified/policy.pt --episodes 6 --seed 3210 --trace-dir runs/verified/eval
python -m training.cli infer --checkpoint runs/verified/policy.pt
python -m training.cli credit --seed 100
```

- Four training episodes completed with finite PPO loss and checkpoint saved.
- Held-out evaluation: **0 wins out of 6**, 0 draws, mean team mass margin **-27.5167** against the scripted pellet-gathering opponent.
- Inference loaded the trained checkpoint and produced bounded movement and valid discrete actions.
- Counterfactual open-loop forking produced finite outputs and preserved original snapshot.
- Additional 5v5 heuristic simulation: 3 independently seeded episodes × 25 policy steps; all completed with mass ledger closure.

**Important:** this is engineering smoke coverage, not statistical proof of algorithm performance. The naive policy is not strong. Numerical results are valid for the stated research ruleset and seeds only.

## Outstanding acceptance blockers

- Official PettingZoo Parallel API compliance requires installing optional packages and running tests on a supported environment.
- Real client physics calibration, contact CCD, hazards, respawn and multi-team are absent.
- Actor–learner distributed scaling, recurrent MAPPO, full self-play league and long-horizon counterfactual attribution are not implemented.
- Independent training-seed confidence intervals, paired team-side swaps and zero-shot teammate benchmarks remain to implement.
