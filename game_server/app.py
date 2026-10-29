"""FastAPI local arcade UI. Each WebSocket owns an isolated engine instance.

Run with: python -m game_server [--host 127.0.0.1] [--port 8000]
Network exposure beyond localhost requires separate authentication/TLS/reverse-proxy work.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from loguru import logger
from starlette.websockets import WebSocketState

from game_server.session import GameSession, InputError, MAX_CLIENT_MESSAGE_BYTES, MODES, POLICY_HZ

STATIC_DIR = Path(__file__).resolve().parent / 'static'
MAX_SESSIONS = 16


def configure_logging(log_dir: str | Path | None = None) -> int:
    """Add an isolated structured file sink; caller removes it on shutdown."""
    location = Path(log_dir or os.getenv('BALL_MARL_LOG_DIR', 'runs/web/logs'))
    location.mkdir(parents=True, exist_ok=True)
    return logger.add(
        str(location / 'game-{time:YYYY-MM-DD}.jsonl'), level='INFO', serialize=True,
        rotation='10 MB', retention='14 days', compression='gz', enqueue=True,
        backtrace=False, diagnose=False,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    sink_id = configure_logging()
    logger.info('game_server_started')
    try:
        yield
    finally:
        logger.info('game_server_stopped')
        await logger.complete()
        logger.remove(sink_id)


app = FastAPI(title='Ball-MARL Arcade', version='0.3.0', docs_url=None,
              redoc_url=None, openapi_url=None, lifespan=lifespan)
app.state.connections = set()


@app.get('/')
async def index():
    return FileResponse(STATIC_DIR / 'index.html', headers={'Cache-Control': 'no-store'})


@app.get('/assets/app.js')
async def javascript():
    return FileResponse(STATIC_DIR / 'app.js', media_type='application/javascript',
                        headers={'Cache-Control': 'no-store'})


@app.get('/assets/styles.css')
async def styles():
    return FileResponse(STATIC_DIR / 'styles.css', media_type='text/css',
                        headers={'Cache-Control': 'no-store'})


@app.get('/api/health')
async def health():
    return {'status': 'ok', 'service': 'ball-marl-arcade', 'protocol': 1,
            'active_sessions': len(app.state.connections)}


@app.get('/api/modes')
async def modes():
    return {'modes': [{'id': key, 'players_per_team': value} for key, value in MODES.items()],
            'policy_hz': POLICY_HZ, 'max_sessions': MAX_SESSIONS,
            'disclaimer': 'Research simulator, not official game mechanics.'}


def _valid_origin(websocket: WebSocket) -> bool:
    # Browser WebSockets send an Origin. Require an exact same-host origin.
    # Non-browser local clients are allowed without Origin for tests and tooling.
    origin = websocket.headers.get('origin')
    if origin is None:
        return True
    from urllib.parse import urlparse
    parsed = urlparse(origin)
    return parsed.scheme in ('http', 'https') and parsed.netloc == websocket.headers.get('host')


def _control_message(session: GameSession, raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not isinstance(raw.get('type'), str):
        raise InputError('message must be a JSON object with a type')
    kind = raw['type']
    if kind == 'input':
        session.set_input(raw)
        return None
    if kind == 'pause':
        if set(raw) != {'type'}:
            raise InputError('pause accepts no additional fields')
        session.paused = not session.paused if not session.finished else session.paused
        logger.bind(match_id=session.session_id, paused=session.paused).info('pause_toggled')
        return session.frame()
    if kind == 'restart':
        if set(raw) - {'type', 'new_seed'}:
            raise InputError('restart accepts only new_seed')
        new_seed = raw.get('new_seed', False)
        if type(new_seed) is not bool:
            raise InputError('new_seed must be boolean')
        session.restart(new_seed=new_seed)
        return session.frame()
    if kind == 'ping':
        if set(raw) != {'type', 'client_time'}:
            raise InputError('ping accepts type and client_time')
        stamp = raw.get('client_time')
        if type(stamp) not in (int, float) or not math.isfinite(stamp):
            raise InputError('client_time must be finite number')
        return {'type': 'pong', 'client_time': stamp}
    raise InputError(f'unknown message type: {kind[:40]}')


@app.websocket('/ws/play')
async def play(websocket: WebSocket):
    if not _valid_origin(websocket):
        await websocket.close(code=1008, reason='origin mismatch')
        return
    mode = websocket.query_params.get('mode', 'squad')
    if mode not in MODES:
        await websocket.close(code=1008, reason='invalid mode')
        return
    # Client-supplied deterministic seeds are allowed for replay debugging.
    raw_seed = websocket.query_params.get('seed')
    if raw_seed is not None:
        if not raw_seed.isascii() or not raw_seed.isdecimal() or len(raw_seed) > 10:
            await websocket.close(code=1008, reason='invalid seed')
            return
        seed = int(raw_seed)
        if seed >= 2**32:
            await websocket.close(code=1008, reason='invalid seed')
            return
    else:
        seed = None
    if len(app.state.connections) >= MAX_SESSIONS:
        await websocket.close(code=1013, reason='server capacity reached')
        return
    session = GameSession(mode, seed)
    await websocket.accept()
    app.state.connections.add(session.session_id)
    send_lock = asyncio.Lock()

    async def emit(data: dict[str, Any]):
        async with send_lock:
            await websocket.send_json(data)

    async def receiver():
        # Ignore > 60 incoming commands/second; preserve the most recent control.
        # Ping and pause messages should not be individually rate limited.
        recent_inputs: list[float] = []
        while True:
            text = await websocket.receive_text()
            if len(text.encode('utf-8')) > MAX_CLIENT_MESSAGE_BYTES:
                await websocket.close(code=1009, reason='message too large')
                return
            try:
                raw = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                if isinstance(raw, dict) and raw.get('type') == 'input':
                    now = time.monotonic()
                    recent_inputs = [t for t in recent_inputs if now - t < 1]
                    if len(recent_inputs) >= 60:
                        continue
                    recent_inputs.append(now)
                response = _control_message(session, raw)
                if response is not None:
                    await emit(response)
            except (ValueError, TypeError, KeyError) as error:
                await emit({'type': 'error', 'message': str(error)[:180]})
                logger.bind(match_id=session.session_id, reason=str(error)[:180]).warning('rejected_client_input')

    async def ticker():
        await emit(session.frame())
        target = time.monotonic()
        while True:
            target += 1 / POLICY_HZ
            await asyncio.sleep(max(0, target - time.monotonic()))
            if session.finished:
                # Stop transmitting unchanged game-over screens indefinitely; keep
                # listening for restart until the websocket disconnects.
                continue
            start = time.monotonic()
            frame = session.step()
            await emit(frame)
            lag = time.monotonic() - start
            if lag > 1 / POLICY_HZ:
                logger.bind(match_id=session.session_id, step_ms=round(lag*1000, 2)).warning('slow_game_tick')
                target = time.monotonic()

    try:
        tasks = [asyncio.create_task(receiver()), asyncio.create_task(ticker())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in pending:
            with suppress(asyncio.CancelledError, WebSocketDisconnect):
                await task
        for task in done:
            with suppress(WebSocketDisconnect, RuntimeError):
                task.result()
    except WebSocketDisconnect:
        pass
    finally:
        app.state.connections.discard(session.session_id)
        logger.bind(match_id=session.session_id, tick=session.env.policy_steps).info('match_disconnected')
