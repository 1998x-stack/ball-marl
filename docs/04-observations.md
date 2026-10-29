# 04 · Observation and model specification

## Three distinct objects
- World state S: authoritative complete simulator state.
- Actor observation O_i=f_visible(S, history_i): only currently visible information and legally remembered history.
- Critic state C_i: optionally global training information; not accessible to deployed Actor.

## Entity-centric schema
Fragment fields: relative position and velocity, log1p mass, radius, owner relation, merge cooldown, split eligibility, visible mask. Player fields: relative position, **visible** aggregate mass, strongest **visible** fragment, visible count, last-seen age and confidence; never leak invisible fragments through aggregate features. Resource/hazard fields: type, position, motion, strength and visibility.

Add normalized arena boundaries, local 32x32 multi-channel occupancy/density grid, and optional last observed events. Use stable IDs only for tracking, not arbitrary enemy sorting features.

## Dynamic entities
Use masks and Deep Sets as the baseline. Attention/graph nodes are later variants. If capping entity count, retain urgent collision threats plus nearest neighbors and aggregate the remainder; log truncation fraction and missed high-threat frequency. Do not present hidden threat estimates as truth.

## Network progression
(1) MLP fixed-neighbor baseline; (2) Deep Sets + map CNN; (3) GRU; (4) entity attention/graph attention. Critic separate encoders if needed. Policy shared across homogeneous agents; conditional agent role embeddings must not contain oracle data.

## Audit
Counterfactual observation test: modify an invisible enemy state without affecting visible dynamics and assert Actor observation unchanged. Check permutation invariance/equivariance under entity reorder and team-name remapping.
