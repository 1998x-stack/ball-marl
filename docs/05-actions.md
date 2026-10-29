# 05 · Actions, temporal semantics and masks

## Primitive hybrid command
```python
action = {
  "move": [ux, uy],   # continuous [-1, 1]^2
  "split": 0,         # discrete 0/1
  "eject": 0,         # discrete 0/1
}
```
The interpreter maps joystick strength and angle to the configured game rule. Do not assume arbitrary speed throttling is supported in an official build.

## Command timing
At 5 Hz policy rate and 20 Hz physics, hold movement input for four substeps. Split is edge-triggered once at command submission; eject is either per-command or held based on a **versioned rule contract**. Simultaneous split and eject are legal only if the ruleset permits; never silently make them mutually exclusive.

## Hybrid policy
Baseline factorization pi(a|o)=pi(move|o)pi(split|o)pi(eject|o); upgraded autoregressive pi(move|o)pi(split|move,o)pi(eject|split,move,o). Use masked categorical distribution for illegal discrete commands; correctly implement transformed Gaussian log-density and joint PPO log-probabilities. Document whether simultaneous command evaluation uses pre- or post-split mass.

## Validation
When all split actions are illegal the mask leaves only no-split valid. NaN movement, invalid ranges and unexpected keys must follow documented reject/default behavior and log an error. Track attempted-invalid-action rate separately from executed commands.

## Hierarchical option research
Optional higher-level intentions: farm, pursue, escape, merge, feed, support, hazard-push. Do not hard-code tactics into first baseline. Evaluate whether options improve learning efficiency at matched environment interactions and simulator compute.
