# 03 · Environment, agents and API contract

## Agent identity
Stable agent key `team_0/player_0` controls all of that player's fragments. Fragment creation/deletion does not change agent IDs. Death may be a temporary inactive/respawn state; episode termination depends on game mode and is distinct from player death.

Use PettingZoo ParallelEnv for simultaneous joint steps. A step advances one policy tick consisting of N physics substeps.

```python
observations, infos = env.reset(seed=42)
actions = {agent: policy(observations[agent]) for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
```

## Reset configuration
`reset(seed, options={ruleset_id, scenario_id, roster, spawn_seed, resource_seed, opponent_ids})`. Record all effective options and deterministic state hash.

## Episode semantics
- **Terminated**: game-defined terminal condition (e.g. match clock ends in a finite-horizon ranking game).
- **Truncated**: external rollout/time-budget interruption that is not a real terminal state. Bootstrap critic on truncation when appropriate.
- Respawn remains within an episode unless ruleset explicitly states otherwise.
- Maintain policy masks for temporarily inactive players and define zero-effect action behavior.

## Information contract
`infos`: non-sensitive debug metadata may contain global data during training but **never** be concatenated into Actor input. Expose global state only through an explicit training-only `state()` method, with leakage tests.

## Scenarios
Single-agent resource collection; predator/prey pursuit; split/counterattack; hazard interaction; 2v2 transfer; 5v5 coordination; ranked multi-team. Each has a fixed manifest and seed split.
