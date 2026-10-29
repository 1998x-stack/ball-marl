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


## V2: rollout and optimization protocol
Collect tuples `(obs_t, global_state_t, action_t, legal_mask_t, logp_t, value_t, reward_t, terminated_t, truncated_t, recurrent_state_t, opponent_id)`. Compute GAE with bootstrap for truncation and zero bootstrap for genuine termination. Use time-major sequences for recurrent PPO and preserve episode boundaries; mask padded timesteps in losses. Advantage normalization is computed over valid samples only.

### Curriculum gate
Each stage must publish a frozen evaluation manifest and promotion threshold (e.g. beat a scripted baseline on held-out seeds with a lower confidence bound above a predeclared threshold). Do not promote based on training return. Preserve previous-stage tests to detect forgetting.

### League robustness
Opponent sampling is versioned; sample historical policies and targeted exploiters, freeze opponent weights during each match, and avoid leaking opponent IDs if they are not available in deployment. Use paired matches with side swaps. Log policy diversity, exploit rates, stale-policy fraction and pool composition. Checkpoint actor, critic, optimizer, normalizer, RNG and league snapshot together.
