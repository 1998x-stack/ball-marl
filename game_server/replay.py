"""Durable, verifiable joint-action match journals.

Replay is the same deterministic engine and its actual executed actions, not
video frames or a second physics implementation. The replay token is intentionally
separate from any player websocket seat token.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any

import numpy as np

from engine.rules import rules_digest
from game_server.session import GameSession

REPLAY_ID = re.compile(r'^[0-9a-f]{24}$')
MAX_JOURNAL_BYTES = 8 * 1024 * 1024


def _jsonline(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'


def action_to_json(action: dict[str, Any]) -> dict[str, Any]:
    return {'move': [float(v) for v in action['move']],
            'split': int(action['split']), 'eject': int(action['eject'])}


class ReplayIntegrityError(ValueError):
    pass


class ReplayRecorder:
    def __init__(self, directory: str | Path, session: GameSession):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.id = secrets.token_hex(12)
        self.token = secrets.token_urlsafe(32)
        self.path = self.directory / f'{self.id}.jsonl'
        self.meta = self.directory / f'{self.id}.meta.json'
        self._stream = self.path.open('x', encoding='utf-8')
        self._closed = False
        header = {'kind': 'header', 'schema': 1, 'mode': session.mode,
                  'seed': session.seed, 'rules_digest': rules_digest(session.rules),
                  'initial_hash': session.env.world.digest()}
        self._stream.write(_jsonline(header))
        self._stream.flush()
        # Never write a bearer token in plaintext to a replay file.
        self.meta.write_text(_jsonline({'token_sha256': hashlib.sha256(self.token.encode()).hexdigest()}),
                             encoding='utf-8')

    def append(self, actions: dict[str, dict[str, Any]], digest: str) -> None:
        if self._closed:
            raise RuntimeError('journal already closed')
        tick = self._last_tick + 1 if hasattr(self, '_last_tick') else 1
        self._stream.write(_jsonline({'kind': 'tick', 'tick': tick,
                                     'actions': {p: action_to_json(a) for p, a in sorted(actions.items())},
                                     'hash': digest}))
        self._stream.flush()
        self._last_tick = tick

    def close(self, reason: str = 'closed') -> None:
        if not self._closed:
            self._stream.write(_jsonline({'kind': 'end', 'reason': reason}))
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True


class ReplayStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def authorize(self, replay_id: str, token: str) -> None:
        if not REPLAY_ID.fullmatch(replay_id) or not isinstance(token, str):
            raise PermissionError('invalid replay')
        metadata = self.directory / f'{replay_id}.meta.json'
        if not metadata.is_file():
            raise FileNotFoundError('replay not found')
        actual = hashlib.sha256(token.encode()).hexdigest()
        expected = json.loads(metadata.read_text(encoding='utf-8'))['token_sha256']
        if not secrets.compare_digest(actual, expected):
            raise PermissionError('invalid replay token')

    def read(self, replay_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not REPLAY_ID.fullmatch(replay_id):
            raise ValueError('bad replay id')
        file = self.directory / f'{replay_id}.jsonl'
        if file.stat().st_size > MAX_JOURNAL_BYTES:
            raise ReplayIntegrityError('replay exceeds maximum size')
        rows = []
        with file.open(encoding='utf-8') as handle:
            for line in handle:
                if len(line) > 65536:
                    raise ReplayIntegrityError('oversized tick record')
                rows.append(json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ReplayIntegrityError(x))))
        if not rows or rows[0].get('kind') != 'header' or rows[0].get('schema') != 1:
            raise ReplayIntegrityError('missing or invalid replay header')
        header = rows[0]
        ticks = []
        last_tick = 0
        for index, row in enumerate(rows[1:], start=1):
            if row.get('kind') == 'end':
                if index != len(rows)-1:
                    raise ReplayIntegrityError('replay has records after end marker')
                break
            if row.get('kind') != 'tick' or row.get('tick') != last_tick + 1:
                raise ReplayIntegrityError('tick number gap or corrupt replay')
            ticks.append(row)
            last_tick += 1
        return header, ticks

    def frames(self, replay_id: str, viewer: str, *, start: int = 0, limit: int = 100,
               include_frames: bool = True) -> dict[str, Any]:
        if not isinstance(start, int) or not 0 <= start <= 1000 or not 1 <= limit <= 100:
            raise ValueError('replay range invalid')
        header, ticks = self.read(replay_id)
        session = GameSession(header['mode'], header['seed'], session_id=replay_id)
        if viewer not in session.env.possible_agents:
            raise ValueError('viewer not in replay roster')
        if rules_digest(session.rules) != header['rules_digest']:
            raise ReplayIntegrityError('ruleset differs from replay')
        if session.env.world.digest() != header['initial_hash']:
            raise ReplayIntegrityError('initial world mismatch')
        frames = []
        if include_frames and start == 0:
            frames.append(session.frame(viewer, event_feed=[]))
        for row in ticks:
            try:
                actions = {p: {'move': np.asarray(a['move'], dtype=np.float32),
                               'split': int(a['split']), 'eject': int(a['eject'])}
                           for p, a in row['actions'].items()}
                session.env.step(actions)
            except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                raise ReplayIntegrityError(f'invalid actions at tick {row["tick"]}') from exc
            if session.env.world.digest() != row.get('hash'):
                raise ReplayIntegrityError(f'world hash mismatch at tick {row["tick"]}')
            if not session.env.agents:
                session.finished = True
            if include_frames and start <= row['tick'] < start + limit:
                frames.append(session.frame(viewer, event_feed=[]))
        return {'replay_id': replay_id, 'verified': True, 'total_ticks': len(ticks),
                'start': start, 'frames': frames}
