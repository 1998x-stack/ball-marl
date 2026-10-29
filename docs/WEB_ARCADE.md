# Web Arcade · Human-playable Ball-MARL (v0.3)

> This local game uses the **same deterministic research engine as training**, but its rules are NOT official 《球球大作战》 rules. It is a standalone human-vs-rule-based-bot UI, not an internet multiplayer service or a trained MAPPO opponent.

## Quick start

Requirements: Python 3.10+, a modern browser, NumPy/PyYAML and the optional `game` extras. From the repo root:

```bash
python -m pip install -e '.[game,test,game-test]'
python -m game_server --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/ in your browser
```

Equivalent installed console command: `ball-marl-play --port 8000`.

**Docker (optional):** `docker compose -f compose.game.yml up --build`; open http://127.0.0.1:8000/. Docker builds a lightweight gameplay image without PyTorch. Docker image execution has to be verified separately on a machine with Docker daemon.

## Features

- Three playable local modes: **1v1**, **2v2**, **5v5**. You control `team_0/player_0`; remaining players are deterministic locally-observing scripted bots.
- Canvas renderer with adaptive sizing, camera-follow, interpolation, mini-map, live team mass and player state, recent events and a match result overlay.
- Controls: **WASD or arrow keys**, mouse direction over canvas, touch joystick; **Space** split, **E** eject, **P** pause, **R** replay same seed.
- Match duration: 450 policy ticks @5Hz = 90 seconds, with 20Hz physics. Human death enters spectator state until match end or restart. There is no respawn in this research ruleset.
- `Loguru` JSONL logs include `match_started`, `engine_event`, `match_finished`, `pause_toggled`, rejected client inputs and disconnections. Logs default to `runs/web/logs/game-*.jsonl`, rotated at 10 MB and retained for 14 days. Override with `BALL_MARL_LOG_DIR`.
- Runs entirely on localhost: no CDN, npm compilation or external API required.

## Architecture and trust boundaries

```text
Browser (Canvas/UI, untrusted input) ── WebSocket ── FastAPI /ws/play
                                                │
                                                ▼
                                      GameSession (one per socket)
                                                │
                                      BallMARLEnv.step(joint_action)
                                                │
                                           engine.World
                                                │
                           snapshots / events / mass ledger (Loguru)
                                                │
                           visibility-filtered frame → browser
```

Server is authoritative: the UI never computes actual physical state, collision, mass or victory. Per-match RNGs isolate bots and seeded resets. Each session has a fresh World and cannot act as another player. The client sees all **friendly** balls and only nearby non-friendly entities. Opponent **total** mass is hidden until terminal; during play the panel shows observed enemy mass marked `*` (a visible partial sum). Match results reveal exact mass for both teams.

## Protocol v1

- WebSocket URL `/ws/play?mode=duel|squad|arena[&seed=<uint32>]`.
- Server frame: `type=frame`, `session_id`, `tick`, `hz`, `seed`, `player`, `arena`, `time_left`, `score`, `visible_balls`, `abilities`, `event_feed`, `paused`, `finished`, optional `outcome`.
- Client input: `{ "type":"input", "move":[0.5,-0.2], "split":false, "eject":false }`. Movement stays active until next message; `split/eject` are **rising-edge actions** and execute at most once per held key. Accepted values: finite floats in `[-1,1]`; booleans for abilities.
- Controls: `{ "type":"pause" }`, `{ "type":"restart", "new_seed":true|false }`, `{ "type":"ping", "client_time": 123.4 }`.
- Invalid messages receive `{ "type":"error", "message":"..." }`; oversized frames are rejected, unknown modes/seeds/origins are rejected before handshake. No HTTP endpoint accepts player actions.
- Restart with `new_seed:false` deterministically restores initial match, including NPC RNG; restarting with `true` chooses new cryptographic seed.

## Security and operational limits

1. **Local-only, unauthenticated.** CLI binds `127.0.0.1` by default and Docker publishes to localhost. Do not expose on the public internet without identity, TLS, rate limits at an edge gateway, and origin policy suitable for deployed domains.
2. Only one human-controlled agent per WebSocket; this is **not network multiplayer**. Up to 16 local matches per process; one Uvicorn worker only because session registry is in-memory.
3. Input schema, byte size, rate and origin are bounded. Each WebSocket is fully isolated; clients cannot choose arbitrary owners, ruleset values or overwrite other sessions.
4. Match and input history are ephemeral in RAM; JSONL event logs can be persisted; no external database is required. Logs are not guaranteed to provide crash-consistent durable replay archives.
5. Environment/world physics runs synchronously on the event loop and is tuned to small local matches. For remote/high-load operation, move simulation to a bounded worker pool and introduce admission control and dedicated state storage.

## Tests, build and verification

```bash
python -m pytest -q
node --check game_server/static/app.js  # optional static JS syntax check
python -m pip wheel --no-deps --no-build-isolation -w dist .
# Optional graphical smoke (requires Playwright and Chromium):
python -m scripts.browser_game_smoke --out runs/screenshots
```

The browser smoke uses **actual engine frames** with a simulated WebSocket transport to work in sandboxes that block Chromium network navigation; it tests UI behavior, but is not proof of browser-to-server network connectivity. The FastAPI `TestClient` integration tests independently exercise the real in-process WebSocket endpoint, including pause/restart/error and cleanup.

### Open production gaps

Server-side collision/physics are research approximations; no CCD for high-speed split, bot policies are scripted and weak, no anti-cheat/authenticated public multiplayer, no persistent replay service, no long-run concurrent production load tests, no container image security attestation. This game is for **local development, demonstration and exploratory experimentation**, not production deployment.
