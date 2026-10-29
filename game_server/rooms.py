"""Multi-human local rooms with authoritative, single-ticker simulation.

Each seat has a capability token. Disconnected players are temporarily taken
by scripted/selected checkpoint AI and can reconnect without losing identity.
All state mutation happens synchronously on one event loop, before awaits.
"""
from __future__ import annotations

from collections import deque
import asyncio
from dataclasses import dataclass, field
import secrets
import time
from typing import Any

import numpy as np
from loguru import logger

from game_server.session import GameSession, ControlState, InputError, MODES, bot_action, validate_input
from game_server.policy import PolicyRegistry
from game_server.replay import ReplayRecorder, ReplayStore

MAX_ROOMS = 16
ROOM_IDLE_SECONDS = 60


class RoomError(ValueError):
    pass


@dataclass
class Seat:
    player: str
    token: str
    connected: bool = False
    joined_at: float = field(default_factory=time.monotonic)
    controls: ControlState = field(default_factory=ControlState.empty)


class Room:
    def __init__(self, mode: str, seed: int | None, registry: PolicyRegistry,
                 replays: ReplayStore, *, bot_id: str | None = None):
        self.id = secrets.token_hex(8)
        self.match = GameSession(mode, seed, session_id=self.id)
        self.registry = registry
        self.replays = replays
        self.bot_id = bot_id
        self.bot_rngs = {p: np.random.default_rng(np.random.SeedSequence([self.match.seed, i, 742]))
                         for i, p in enumerate(self.match.env.possible_agents)}
        self.bot_model = None
        if bot_id is not None:
            sample = self.match.env.observe(self.match.human_id)
            self.bot_model = registry.load(bot_id, self.match.rules,
                                           len(sample['features']), len(self.match.env.state(0)))
        self.seats: dict[str, Seat] = {}
        self.completed = False
        self.subscribers: dict[str, Any] = {}
        self.owner_token = self.join(team=0)
        self.event_history = deque(maxlen=24)
        self.event_feed_by_player = {p: deque(maxlen=8) for p in self.match.env.possible_agents}
        self.invalid_by_player: dict[str, list[str]] = {}
        self.recorder = ReplayRecorder(replays.directory, self.match)
        self.last_active = time.monotonic()
        self.lock = asyncio.Lock()
        self.task: asyncio.Task | None = None

    def join(self, team: int | None = None) -> str:
        if team is not None and (type(team) is not int or team not in (0, 1)):
            raise RoomError('team must be 0 or 1')
        if self.completed:
            raise RoomError('room has closed')
        players = self.match.env.possible_agents
        chosen = next((p for p in players
                       if (team is None or p.startswith(f'team_{team}/'))
                       and p not in {s.player for s in self.seats.values()}), None)
        if chosen is None:
            raise RoomError('team/room is full')
        token = secrets.token_urlsafe(32)
        self.seats[token] = Seat(player=chosen, token=token)
        self.last_active = time.monotonic()
        logger.bind(room_id=self.id, player=chosen).info('room_seat_reserved')
        return token

    def seat(self, token: str) -> Seat:
        if not isinstance(token, str) or token not in self.seats:
            raise PermissionError('invalid room token')
        return self.seats[token]

    def connect(self, token: str) -> Seat:
        seat = self.seat(token)
        if seat.connected:
            raise RoomError('seat is already connected')
        seat.connected = True
        seat.controls = ControlState.empty()
        self.last_active = time.monotonic()
        logger.bind(room_id=self.id, player=seat.player).info('room_player_connected')
        return seat

    def disconnect(self, token: str) -> None:
        seat = self.seat(token)
        seat.connected = False
        seat.controls = ControlState.empty()
        self.last_active = time.monotonic()
        logger.bind(room_id=self.id, player=seat.player).info('room_player_disconnected')

    def input(self, token: str, payload: dict[str, Any]) -> None:
        seat = self.seat(token)
        if not seat.connected:
            raise RoomError('player is offline')
        move, split, eject = validate_input(payload)
        seat.controls.update(move, split, eject)

    def set_paused(self, token: str) -> None:
        if token != self.owner_token:
            raise PermissionError('only room owner can pause')
        if not self.match.finished:
            self.match.paused = not self.match.paused
        logger.bind(room_id=self.id, paused=self.match.paused).info('room_pause_toggled')

    def restart(self, token: str, *, new_seed: bool = False) -> None:
        if token != self.owner_token:
            raise PermissionError('only room owner can restart')
        self.recorder.close('restarted')
        self.match.restart(new_seed=new_seed)
        self.bot_rngs = {p: np.random.default_rng(np.random.SeedSequence([self.match.seed, i, 742]))
                         for i, p in enumerate(self.match.env.possible_agents)}
        self.recorder = ReplayRecorder(self.replays.directory, self.match)
        self.event_history.clear()
        for feed in self.event_feed_by_player.values(): feed.clear()
        self.invalid_by_player.clear()
        for seat in self.seats.values():
            seat.controls = ControlState.empty()
        logger.bind(room_id=self.id, replay_id=self.recorder.id).info('room_restarted')

    def _bot(self, player: str) -> dict[str, Any]:
        if self.bot_model is not None:
            # The inference input is the same partial actor observation as training.
            return self.bot_model.choose(self.match.env.observe(player), deterministic=True)[0]
        return bot_action(self.match.env.world, player, self.bot_rngs[player])

    def tick(self) -> None:
        if self.match.paused or self.match.finished:
            return
        env = self.match.env
        connected = {s.player: s for s in self.seats.values() if s.connected}
        actions = {p: connected[p].controls.take() if p in connected else self._bot(p)
                   for p in env.agents}
        _, _, done, _, info = env.step(actions)
        for p in env.possible_agents:
            self.invalid_by_player[p] = info[p]['invalid_actions']
        events = info[env.possible_agents[0]]['events']
        # Visibility is evaluated on the tick the event occurred. Recomputing
        # visibility on later frames would retroactively reveal hidden actions.
        world = env.world
        visible_by_player = {}
        for p in world.players:
            center = (world.centroid(p) if world.owned(p)
                      else self.match._viewer_centers.get(p, np.zeros(2)))
            visible_by_player[p] = {b.id for b in world.balls
                                    if self.match._visible(b.x, b.y, center=center)}
        for event in events:
            self.event_history.append(event.copy())
            logger.bind(room_id=self.id, event_type=event['type'], tick=event['tick']).info('room_engine_event')
            for p, visible in visible_by_player.items():
                if (event.get('actor') == p or event.get('target') == p
                        or event.get('source') in visible or event.get('eater') in visible):
                    self.event_feed_by_player[p].append({
                        'type': event['type'], 'actor': event.get('actor'),
                        'target': event.get('target'), 'tick': event['tick'],
                        'mass': round(float(event.get('mass', 0)), 2)})
        self.recorder.append(actions, env.world.digest())
        if all(done.values()):
            self.match.finished = True
            self.recorder.close('finished')
            logger.bind(room_id=self.id, replay_id=self.recorder.id,
                        seed=self.match.seed).info('room_finished')

    def frame(self, token: str) -> dict[str, Any]:
        seat = self.seat(token)
        p = seat.player
        w = self.match.env.world
        events = list(self.event_feed_by_player[p])
        online = sum(s.connected for s in self.seats.values())
        result = self.match.frame(p, bot_count=len(w.players)-online,
                                  event_feed=events[-8:],
                                  invalid_actions=self.invalid_by_player.get(p, []))
        result.update({'room_id': self.id, 'players_online': online,
                       'replay_id': self.recorder.id, 'replay_token': self.recorder.token, 'is_host': token == self.owner_token,
                       'bot_type': self.bot_id or 'scripted'})
        return result

    def public(self) -> dict[str, Any]:
        return {'room_id': self.id, 'mode': self.match.mode,
                'players_online': sum(s.connected for s in self.seats.values()),
                'seats_reserved': len(self.seats), 'capacity': len(self.match.env.possible_agents),
                'tick': self.match.env.policy_steps, 'finished': self.match.finished,
                'bot_type': self.bot_id or 'scripted'}

    def close(self) -> None:
        if not self.completed:
            self.completed = True
            self.recorder.close('room_closed')
            if self.task is not None and not self.task.done():
                self.task.cancel()


class RoomManager:
    def __init__(self, *, replay_dir: str, checkpoint_dir: str, max_rooms: int = MAX_ROOMS):
        self.rooms: dict[str, Room] = {}
        self.replays = ReplayStore(replay_dir)
        self.registry = PolicyRegistry(checkpoint_dir)
        self.max_rooms = max_rooms

    def prune(self) -> None:
        now = time.monotonic()
        for room_id, room in list(self.rooms.items()):
            if not any(s.connected for s in room.seats.values()) and now-room.last_active > ROOM_IDLE_SECONDS:
                room.close()
                del self.rooms[room_id]

    def create(self, mode: str, seed: int | None, bot_id: str | None = None) -> Room:
        self.prune()
        if len(self.rooms) >= self.max_rooms:
            raise RoomError('room capacity reached')
        if mode not in MODES:
            raise RoomError('invalid mode')
        room = Room(mode, seed, self.registry, self.replays, bot_id=bot_id)
        self.rooms[room.id] = room
        logger.bind(room_id=room.id, mode=mode, bot_id=bot_id or 'scripted').info('room_created')
        return room

    def get(self, room_id: str) -> Room:
        self.prune()
        if room_id not in self.rooms:
            raise KeyError('room not found')
        return self.rooms[room_id]

    def close(self) -> None:
        for room in self.rooms.values():
            room.close()
        self.rooms.clear()
