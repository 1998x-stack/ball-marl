# Ball-MARL · runnable research prototype

A **non-official** multi-agent ball-consumption research environment with deterministic snapshots, mass accounting, per-player hybrid actions, optional PettingZoo ParallelEnv compatibility, local shared-parameter MAPPO training, model inference, scripted opponents, held-out evaluation, event traces and open-loop difference-reward analysis.

This is a **minimal end-to-end proof of pipeline**, not a faithful recreation of the commercial game's mechanics and **not** a validated competitive agent. No claim of PettingZoo compliance is made until official test dependencies are installed and run.

## Requirements and installation

- Python 3.10+, NumPy, PyTorch CPU/GPU and PyYAML; pytest for tests.
- Optional: `pip install -e '.[pettingzoo]'` to use Gymnasium/PettingZoo space types and run their official compliance tests.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m training.cli train --episodes 5 --steps 40 --out runs/demo
python -m training.cli infer --checkpoint runs/demo/policy.pt
python -m training.cli evaluate --checkpoint runs/demo/policy.pt --episodes 10 --seed 9000 --trace-dir runs/demo/eval
python -m training.cli credit --seed 9000
```

## Dataflow

`specifications/game_rules.yaml → engine/world.py → observations/entities.py → environments/pettingzoo_env.py → algorithms/mappo.py → training/learner.py → policy.pt → evaluation/tournament.py`.

- **Environment:** one Agent per player (not per fragment), simultaneous hybrid `move/split/eject` commands, visibility-filtered actor observations and privileged critic state, finite-horizon team ranking reward with PBRS, no respawn or virus in reference ruleset.
- **Training:** player-team-0 shared stochastic actor, team-0 centralized V-critic, GAE and PPO clip objective, scripted team-1 opponents. No claim of population self-play or zero-shot coordination yet.
- **Inference:** checkpoint-loaded deterministic actor action for one observation.
- **Evaluation:** frozen policy, disjoint test seeds, per-match outcomes + event traces; no statistical superiority claims from smoke tests.
- **Counterfactual:** simulator forks a snapshot and substitutes one agent's action for a fixed, open-loop horizon. It is *not* a COMA gradient nor a unique causal attribution.

## Restrictions and intentionally missing modules

`engine/world.py` uses an O(n²) reference collision kernel with documented deterministic ID ordering, a research-only mass/speed curve, and no asynchronous spawn, server-authoritative networking, virus splitting, advanced player-controlled fragment steering, render GUI or multi-policy league training. Spatial hash, CCD, scenario registry, rewards generalization, PettingZoo full compliance, multi-team ranks, real client calibration and an advanced self-play tournament system remain staged work.

The version identifiers and evidence statuses in `specifications/` are **PROPOSED**, not measured official rules.

## Baseline regression gates

- Mass conservation including external pellet spawning and ejection loss.
- Snapshot hash equality under identical actions and RNG state.
- Split cap, illegal input rejection and positive mass invariant.
- Enemy out-of-view perturbation leaves actor observation unchanged.
- Terminal outcome distinct from external rollout truncation.
- PBRS telescoping with zero terminal potential.
- PPO log-probability recomputation, train/save/load/inference/evaluation and replay trace generation.

## Useful paths

- `runs/demo/policy.pt` — trained checkpoint (local file; do not commit as source code).
- `runs/demo/manifest.json` — rules and training run summary.
- `runs/demo/train.jsonl` — step-level PPO summaries by episode.
- `runs/demo/eval/match_000.json` — event trace + state hashes.

**Research honesty:** an agent trained for 5–40 episodes generally performs worse than hand-coded pellet collection. Report that baseline rather than treating a successful training loop as policy quality.

## Engineering V0.2 hardening — production readiness is NOT claimed

This research prototype now includes strict rule validation and SHA256 ruleset fingerprints, arena-relative spawn layouts, deterministic geometric collision precedence, incremental event hash chains, corruption-rejecting checkpoints, non-overwriting seeded training, per-match independent RNG, side-swapped paired evaluation, and 5v5 CPU reference benchmarking.

```bash
python -m pytest -q
python -m scripts.release_gate --out runs/gate-001
python -m training.cli benchmark --players-per-team 5 --episodes 3 --steps 50
python -m training.cli evaluate --checkpoint runs/demo/policy.pt --episodes 20 --side-swaps
```

Historical V0.1 checkpoints lack checksum sidecars and are rejected by default. Migrate only a **trusted** legacy checkpoint explicitly:

```bash
python -m training.cli migrate-checkpoint --source OLD.pt --target NEW.pt
```

A SHA256 sidecar is a corruption detector, not a publisher signature. Checkpoints use restricted PyTorch `weights_only=True` loading. Current local tests: **54 passed, 1 skipped (official optional PettingZoo dependencies absent locally)**. Hosted CI must independently confirm its results. The model is not yet competitively strong; implementation does not claim commercial-game rules parity.

See [production-readiness and blockers](docs/PRODUCTION_READINESS.md).


### V0.2 collision performance and correctness
The default reference-engine execution path now uses deterministic spatial hashing; O(N²) all-pairs remains an explicit correctness oracle. On the same local CPU and a five-versus-five workload with 160 pellets and tracemalloc enabled: reference ~3.59 policy steps/sec, grid ~6.69 policy steps/sec, both with the same mass residual. Randomized contact equivalence and paired physical-trajectory tests passed locally. This is a limited measurement, not a throughput SLA or official-game fidelity claim.
