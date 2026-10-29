# 08 · Training, curriculum and league design

## Stage sequence
S1 stationary pellet navigation; S2 moving prey/predator; S3 split attack/counterattack; S4 hazards; S5 2v2 feeding/protection; S6 5v5; S7 multi-team rankings; S8 self-play generalization. Each stage retains held-out validation seeds; only advance on documented objective metrics and stability.

## Baseline model
Shared-parameter IPPO then MAPPO; MLP or Deep Sets encoders; optional recurrent GRU. Starting hyperparameters for experiments: gamma 0.99/0.995/0.999 sweep, GAE lambda .95, clip .2, learning rate 3e-4, PPO epochs 5, gradient clipping, value normalization, masked actions, physics 20Hz and policy 5Hz. Report **effective discounted horizon**; gamma .99 at 5Hz is short for full matches.

## League
Maintain main policy, historical snapshots, rule bots and targeted exploiters. Example sampling mix 30/30/20/20 is a tunable hypothesis, not empirically optimal. Snapshot based on held-out tournaments, behavioral diversity, ruleset validation, and absence of known exploit abuse. Prefer opponent mixture and no forgotten historical benchmarks.

## Experiment record
`{engine_hash,ruleset_hash,obs_hash,action_hash,reward_hash,scenario_manifest,seed,opponent_pool_version,policy_checkpoint,optimizer,compute_budget}`.
Record environment steps, physics steps, wall clock, CPU/GPU hours, FPS, invalid action rate, loss statistics, population diversity and exploit detections.

## Training separation
Use training environments for learning, fixed validation scenarios for hyperparameter selection, and untouched test seeds/policies for final report. Prevent checkpoint selection on test opponents.
