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


## V2: precise lifecycle and compliance
`possible_agents` is the fixed roster for a match. `agents` contains agents currently participating according to the chosen API policy. In the recommended respawn mode, temporarily ball-less players remain in `agents`, receive inactive observations and may submit only no-op; they are **not terminated** until match end. For permanent elimination, remove the agent according to PettingZoo lifecycle conventions and return final transition data consistently.

```python
class BallParallelEnv(ParallelEnv):
    metadata = {"name": "ball_marl_research_v0", "render_modes": ["human","rgb_array"]}
    def reset(self, seed=None, options=None): ...
    def step(self, actions): ...
    def observation_space(self, agent): ...
    def action_space(self, agent): ...
    def state(self): ...  # critic only
```
At `step` boundary, reject unknown agent keys; missing active commands follow a documented policy (strict error in tests, no-op only in explicit permissive wrapper). `infos` exposes executed action, rejected reasons, reward breakdown and snapshot hash for research; do not route it into Actor tensors. `terminated` is true on a **ruleset-defined match end**, including a match clock if that clock is part of the actual game; `truncated` is reserved for external cutoff.

### Verification
Run `parallel_api_test`, seed determinism, observation/action `space.contains`, final-step dictionary shape, restart after close, and multi-episode memory-leak tests. Reference: PettingZoo Parallel API.
