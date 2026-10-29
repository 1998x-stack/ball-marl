# Ball-MARL industrial readiness and engineering acceptance · V0.2

**Status: hardened research prototype, not production-ready or officially calibrated game mechanics.** The simulator is explicitly a proposed research ruleset.

## Verified locally (2026-10-29)

- 54 pytest tests passed; 1 optional official PettingZoo compliance test skipped because Gymnasium/PettingZoo are not installed in the local container.
- Seeded 2v2 MAPPO trained, model saved/loaded, legacy checkpoint migrated via restricted `weights_only=True` loader, inference and side-swapped paired evaluation executed.
- Six independently seeded *match outcomes* (three seed clusters x two sides): 0 wins by the tiny trained checkpoint. This establishes only pipeline health, not agent competence.
- 5v5 CPU reference benchmark with tracemalloc: approximately 14.23 policy steps/s and P95 92.03 ms for the measured workload; global mass residual about 3.46e-14. Benchmark metrics are hardware/workload-dependent, not performance SLAs.
- Wheel built and inspected; `specifications/*.yaml` included. Hosted GitHub Actions and official PettingZoo test outcomes are **not yet confirmed**.

## Module-level quality gates

| Module | Implemented hardening | Remaining blocker |
|---|---|---|
| Rules | strict parameter schema, bounds, SHA256 rules digest | official parameter calibration |
| Engine | geometry-based collision priority, safe spawn layout, event hash chain, snapshot tampering rejection, mass ledger | high-speed CCD, spatial broadphase, virus and respawn |
| Environment | player-level simultaneous steps, audited mask enforcement, RGB render | official PettingZoo compliance verification |
| Observation | fog-of-war actor observation and equal-visibility scripted opponent | entity GNN/attention and richer game mechanics |
| Action | bounded hybrid policy and stable legal-action masks | complex temporal tactics and options |
| Reward | terminal rank + separately logged potential-based shaping | mode-specific objectives and adversarial reward exploit sweep |
| MAPPO | actual shared actor, centralized V, GAE/PPO, gradient update | recurrent MAPPO, hyperparameter sweep, distributed actor-learner |
| Artifacts | atomic save, SHA256 corruption check, versioned model format, explicit legacy migration | signing/trusted registry, atomic multi-file experiment transaction |
| Evaluation | held-out seeds, paired side swaps, cluster bootstrap; per-match RNG isolation | multi-training-seed stats, unknown teammate and opponent evaluation |
| Operations | CI matrix, install-wheel check, release-gate script, performance instrument | persistence, external observability, resource quotas, crash recovery |

## Reproduction

```bash
python -m pip install -e '.[test,pettingzoo]'
python -m pytest -q
python -m scripts.release_gate --out runs/release-gate-001
python -m training.cli train --episodes 50 --steps 100 --out runs/train-001
python -m training.cli infer --checkpoint runs/train-001/policy.pt
python -m training.cli evaluate --checkpoint runs/train-001/policy.pt --episodes 30 --seed 9000 --side-swaps
python -m training.cli benchmark --players-per-team 5 --episodes 5 --steps 100
```

Never reuse an existing training run directory without intentional retention/migration. Legacy V0.1 checkpoints without SHA256 must be explicitly migrated from a **trusted source**:

```bash
python -m training.cli migrate-checkpoint --source OLD.pt --target NEW.pt
```

SHA256 detects corruption, **not** cryptographic authenticity. Checkpoints are loaded using `weights_only=True`, never unrestricted pickle loading.

## Production blockers, ordered

**P0:** Official PettingZoo compliance executed in CI; replay invariants under more scenarios, complete ruleset/physics calibration, independent training-seed reproducibility, full-match evaluation and systematic reward exploit tests.

**P1:** Durable structured event/metric storage, post-crash resume and checkpoint lifecycle, role-based access to artifacts, external telemetry, dependency/SBOM security scan, bounded job concurrency and quotas, isolation of arbitrary user input.

**P2:** Scalable spatial collision index with exact oracle parity tests, GPU/multi-node Actor–Learner, league self-play and anti-forgetting policies, closed-loop counterfactual event attribution, zero-shot teammate generalization.

**P3:** Capacity testing under representative loads, service SLOs, upgrade/rollback procedures, multi-host recovery drills and long-run operational monitoring.

## Engineering truthfulness

Do not label this project 'production certified' based on CI green alone. Distinguish (a) deterministic simulation and save/load correctness, (b) statistical algorithm performance and (c) production operations readiness. All three require separate release evidence.


### V0.2 collision performance and correctness
The default reference-engine execution path now uses deterministic spatial hashing; O(N²) all-pairs remains an explicit correctness oracle. On the same local CPU and a five-versus-five workload with 160 pellets and tracemalloc enabled: reference ~3.59 policy steps/sec, grid ~6.69 policy steps/sec, both with the same mass residual. Randomized contact equivalence and paired physical-trajectory tests passed locally. This is a limited measurement, not a throughput SLA or official-game fidelity claim.
