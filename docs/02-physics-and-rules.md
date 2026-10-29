# 02 · Physics engine and authoritative rules

## Model (proposed, not official)
For mass m and surface density rho, r=sqrt(m/(pi*rho)); normalize rho=1/pi for r=sqrt(m). A configurable monotone speed curve is v(m)=v_min+k/(m+m0)^alpha, with alpha>0. These equations are candidate models awaiting calibration.

For balls i,j, center separation d_ij=norm(x_i-x_j). Geometric circle intersection uses d_ij<r_i+r_j; consumption is **not equivalent** to contact. Candidate eligibility requires m_i>=lambda*m_j and sufficient penetration such as d_ij <= r_i-beta*r_j. lambda and beta are explicit ruleset parameters, **not asserted official constants**.

## Step ordering contract
1. Validate simultaneous player commands at policy boundary.
2. Convert edge-triggered split/eject commands into queued events.
3. For each physics substep: integrate motion and impulses, constrain boundaries, generate broad-phase contacts, deterministically sort candidate interactions, resolve eligibility and ownership, settle hazards and timers, emit events.
4. Aggregate measurements, compute rewards after state transition, produce observations and termination flags.

Define tie-breaks for simultaneous consumption, multi-victim contacts, allied transfer and same-tick elimination. Stable numeric entity IDs and contact ordering are mandatory. Prefer semi-implicit Euler initially; integration method, dt and collision tolerance are versioned.

## Mass ledger
For each substep record:
M_after - M_before = spawned_pellets + spawned_hazard_mass + respawn_grants - sinks - decay - expired_ejections.
Transfers between agents, splitting, merging and ordinary consumption reallocate mass rather than create it. If a hazard adds mass, explicitly record its grant. Assert both global and team-level reconciliations, accounting for cross-team consumption.

## Proposed baseline
20 Hz physics, 5 Hz policy, 16 maximum fragments per player, fixed arena and configurable 180-second episode. These are experiment settings rather than live-game measurements.

## Determinism
Snapshot positions, velocities, fragments, ownership, timers, pending inputs, random generator state, event counters and configuration hashes. Validate replay hashes at every policy tick, including tie cases; record platform/backend where floating-point behavior may differ.

## Calibration protocol
For each mechanic record source build and mode, reproducible video timestamps, input sequence, measured values with error bounds, fitting loss, held-out trajectories and decision to accept/reject. Until calibrated label every parameter as PROPOSED.
