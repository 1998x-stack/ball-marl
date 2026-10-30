# ORBIT.ARENA V0.4 — Multi-human rooms, model opponents and verified replays

> Research demonstration, **not** production multiplayer. Single-process FastAPI/Uvicorn, localhost-only by default. Actual game mechanics remain configurable research assumptions, not verified commercial rules. The server owns all physics, scoring, entity ownership and fog of war.

## Quick start

```bash
python -m pip install -e '.[game,game-test]'
python -m game_server --host 127.0.0.1 --port 8000
# Browser: http://127.0.0.1:8000/
```

- **单人体验**: original `进入竞技场` uses `/ws/play`, unchanged.
- **创建多人房间**: choose 1v1 / 2v2 / 5v5 and bot option, then `创建多人房间`; share the 16-hex-character room code (never share a seat token or replay token).
- **加入**: another browser enters the code and chooses blue/red/automatic seat. Players own exactly one fixed PlayerID; split balls belong to their player.
- **断线**: the room retains the seat, switches to an AI driver and permits reconnecting with the original seat token **while that room is still alive**. The browser keeps the seat token in **per-tab sessionStorage** and displays a manual "恢复上次房间" control after refresh. Clearing browser storage or opening an unrelated tab may lose this capability; tokens are not accounts and a stale room expires.
- **房主**: only the creator can pause/restart. A restart creates a new replay artifact and clears old controls.
- **终局回放**: select `观看比赛回放`. The server replays the actual recorded joint actions on the same engine and compares each tick's world hash. Tampering or incompatible rules cause failure. Fog of war applies to the selected player viewpoint, including during replays.

## Install and train a compatible MAPPO opponent

The server lists only checkpoints installed by the operator under `BALL_MARL_CHECKPOINT_DIR` (default `runs/web/checkpoints`). Client-provided filesystem paths and model uploads are unsupported. The checkpoint must contain a valid `.sha256` sidecar, match the **entire game ruleset digest**, and have correct observation/critic dimensions. `MAPPO.load` uses restricted `torch.load(weights_only=True)`.

```bash
# Generate an example ruleset-compatible checkpoint; only a smoke training baseline.
python -m game_server.train_bot --mode duel --id demo-duel --episodes 3 --rollout-steps 40
# Alternatively, after pip install:
ball-marl-train-bot --mode duel --id demo-duel --episodes 3 --rollout-steps 40

# List policies available to rooms
curl http://127.0.0.1:8000/api/bots
```

The training adapter uses the exact `GameSession` ruleset, independent of the shorter rollout horizon. **Short smoke training does not imply competitive performance**; it is provided to demonstrate real Checkpoint inference, not to claim learned teamwork. Mode-specific (1v1, 2v2, 5v5) checkpoints need separate training because player count and critic dimension/rules digest differ.

## HTTP/WebSocket API V2

| Endpoint | Purpose | Access |
|---|---|---|
| `GET /api/health` | liveness and legacy connection count | local |
| `GET /api/modes` | supported modes | local |
| `GET /api/bots` | installed checkpoint IDs | local |
| `POST /api/rooms` | `{mode,seed?,bot?}` creates room, returns owner seat token + replay token | local, capacity-bound |
| `GET /api/rooms/{room_id}` | non-secret room occupancy | local |
| `POST /api/rooms/{room_id}/join` | `{team?:0\|1}` reserves next seat and returns its token | room code is join capability |
| `WS /ws/rooms/{room_id}` | authenticate first with `{type:"auth",token:"..."}`, then stream viewpoint-scoped frames and per-seat actions | bearer seat token in WS frame |
| `GET /api/replays/{id}/frames?viewer=...&start=0&limit=100` | verified playback, bounded page | `X-Replay-Token`, *finalized only* |

Protocol messages share the legacy `input`, `pause`, `restart`, `ping`. The room streams extras `room_id,players_online,is_host,bot_type,replay_id,replay_token`. Room ID is a join code, seat token is sent as the first authenticated WS data frame (never as a URL query parameter), replay token grants finalized playback. Do not include private tokens in logs/screenshots.

## Isolation and failure handling

1. Room participants can control only the seat assigned to their token; two simultaneous clients cannot claim the same seat. The room owner is the only player who can pause/restart.
2. A room ticker computes one joint action per policy step, mixing human controls, deterministic scripted bots or a frozen checkpoint actor. It then calls `BallMARLEnv.step` *once*. No duplicate physics in JavaScript.
3. The browser receives its own local player state and friendly balls; enemy totals are withheld until terminal. Event visibility is evaluated at event time and never recomputed retroactively.
4. Disconnects switch a seat to a bot until reconnect. The room continues while at least one WebSocket subscriber is present; idle rooms expire after 60 seconds when manager cleanup runs. Sessions and tokens exist only in one Uvicorn process.
5. Input size (4096 bytes), numeric finiteness, strict command schema, input-message frequency and same-origin checks are enforced. Capacity is limited to 16 rooms. This is **not** a hardened Internet-facing authentication system.
6. Loguru handles lifecycle and events, with JSONL rotation/retention. Replay journals are separate append/flush JSONL files and have an opaque hashed access token in metadata; they are **not signed** or transactional database records. Crashes can leave incomplete journals; finalized replay access fails closed for incomplete logs.
7. Replay validation includes initial state/rules hash, per-tick monotonically increasing IDs, recomputation of the world digest using recorded joint actions. The hashes detect accidental/tampered changes but are not a cryptographic signature.
8. WS sends are bounded by a timeout; no public network/TLS, multi-worker coordination, durable room service, public account identity, account throttling or cloud matchmaking.

## Testing and verification

```bash
python -m pytest -q
node --check game_server/static/app.js
python -m pip wheel --no-deps --no-build-isolation -w dist .
```

Specific suites: `tests/test_multiplayer_rooms.py`, `tests/test_arcade_training.py`, and the original engine/game integration suites. Add full Playwright network exercise when available in an environment permitting Chromium↔localhost. A passing smoke test does not establish game balance, policy convergence or multi-machine fault tolerance.
