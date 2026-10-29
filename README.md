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


## Engineering V0.2 hardening (production-readiness is NOT claimed)

The reference prototype now includes stricter rule validation with a SHA256
ruleset digest, arena-relative spawns, event hash chaining, deterministic
geometric collision precedence, checkpoint integrity and atomic persistence,
per-match RNG isolation, side-swapped paired tournaments with cluster bootstrap,
5v5 performance measurement and a one-command release gate.

```bash
python -m pytest -q
python -m scripts.release_gate --out runs/gate-local-001
python -m training.cli benchmark --players-per-team 5 --episodes 3 --steps 50
python -m training.cli evaluate --checkpoint runs/demo/policy.pt --episodes 20 --side-swaps
```

**Compatibility:** historical V0.1 checkpoints have no checksum sidecar and
are rejected by default. Explicitly migrate a trusted artifact with
`python -m training.cli migrate-checkpoint --source OLD --target NEW`. The
legacy path uses PyTorch `weights_only=True` and never loads arbitrary Python
objects. Checksums detect accidental corruption, **not authenticity**.

See [production-readiness matrix](docs/PRODUCTION_READINESS.md) for verified
gates and the major open blockers to industrial use. Official PettingZoo
compliance is tested in CI with the optional dependency installed; local
fallback tests alone do not establish that compatibility.

## Play the local browser game (v0.3)

The human-playable **ORBIT.ARENA** front end now uses FastAPI/WebSocket to drive
**the existing `engine.World` through `BallMARLEnv`** — no duplicated browser
physics or oracle-controlled opponent scores.

```bash
python -m pip install -e '.[game,game-test,test]'
python -m game_server --host 127.0.0.1 --port 8000
# visit http://127.0.0.1:8000
```

Alternative local Docker build: `docker compose -f compose.game.yml up --build`.
Choose 1v1, 2v2 or 5v5. Use WASD/mouse/touch to steer, Space to split,
E to eject, P to pause, R to restart. Loguru JSONL logs are written under
`runs/web/logs` (override `BALL_MARL_LOG_DIR`).

See [game architecture, controls, security boundaries and testing](docs/WEB_ARCADE.md).
This is a **local, unauthenticated research simulation**, not public multiplayer
and not an official game implementation. Existing training API remains unchanged.
