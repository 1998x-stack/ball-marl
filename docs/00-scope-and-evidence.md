# 00 · Scope, assumptions, and evidence register

## Research object
A 2D synchronous multi-player ball-consumption game **inspired by** Ball Battle. This repository is not an official implementation and must not imply proprietary source-code access or exact official mechanics.

## Evidence classes
- **Verified**: source-linked publication or reproducible measurement against a declared game build.
- **Proposed**: research rule selected for the simulator.
- **Unknown**: requires empirical measurement, including split impulse, mass transfer efficiency, speed curve, collision penetration, merge timers, maximum fragments, virus effects, and ranking formula.
- **Version-specific**: may change by mode or update.

Every numerical mechanic in a shipped ruleset must have a row in a `rule_registry` containing parameter name, value, units, evidence class, source/build, calibration procedure, and version.

## Modes
Start with a two-team five-player benchmark; extend to free-for-all, multi-team ranking, finite-life survival and special modes only under separately versioned rulesets. Never mix rewards or results from distinct modes without explicit normalization.

## Research hypotheses
H1: variable-sized entity representations outperform fixed ordered truncation under changing populations.
H2: potential-based mass shaping improves sample efficiency without changing the stated objective under its assumptions.
H3: offline simulator counterfactuals recover meaningful event contributors but may be costly and sensitive to baseline choice.
H4: league training improves held-out opponent robustness compared with latest-only self-play.
H5: zero-shot teammate tests expose brittle co-adaptation absent from standard self-play win rates.

## Explicit limitations
Counterfactual results are conditional on intervention definition and rollout policy, not unique causal ground truth. Leaderboard performance on a custom simulator is not proof of official-game skill.


## V2: requirements traceability and governance
Requirement IDs use `REQ-<domain>-NNN`. Each requirement maps to one design section, a concrete automated test, a status and an evidence artifact. Modes are explicitly separated: `research_5v5`, `research_multiteam`, and `calibrated_<build>`; never aggregate their leaderboard results. The research simulator is **not** a verified reproduction of the commercial game.

### Evidence review procedure
Every gameplay constant must include provenance, unit, mode/build, sample count, uncertainty and owner. Unverified parameters fail the `official_parity` gate but may pass the `research_ruleset` gate. Rule changes invalidate dependent golden replays, checkpoints and comparisons. Maintain a change-impact table for physics → observations → rewards → training → evaluation.

### Decision record template
`ADR-ID; context; options; decision; rejected alternatives; invariants; measurable risk; rollback; owner; date`. Major choices (per-player agent identity, simultaneous semantics, deterministic collision resolution, reward ownership, time-limit terminal semantics) each require an ADR. Acceptance: every REQ-P0 maps to a passing test or an explicitly tracked blocker.
