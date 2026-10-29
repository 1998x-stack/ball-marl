"""Deterministic per-connection matches. Physics is exclusively engine-owned.

The server never forks the rules implementation: it submits a joint action to
BallMARLEnv and derives all visible state from env.world.
"""
from __future__ import annotations

from collections import deque
import math
import secrets
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from engine.rules import rules_from_yaml
from environments.pettingzoo_env import BallMARLEnv

MODES = {"duel": 1, "squad": 2, "arena": 5}
MAX_CLIENT_MESSAGE_BYTES = 4096
MAX_EVENTS_PER_FRAME = 8
POLICY_HZ = 5


class InputError(ValueError):
    """Client-supplied action does not meet the protocol."""


def _unit(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-9:
        return np.zeros(2, dtype=np.float32)
    return (vector / magnitude).astype(np.float32)


def validate_input(payload: dict[str, Any]) -> tuple[np.ndarray, bool, bool]:
    if not isinstance(payload, dict) or set(payload) != {"type", "move", "split", "eject"}:
        raise InputError("input requires exactly type, move, split and eject")
    if payload["type"] != "input":
        raise InputError("expected input message")
    move = payload["move"]
    if not isinstance(move, list) or len(move) != 2:
        raise InputError("move must be a two-dimensional array")
    if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or abs(n) > 1 for n in move):
        raise InputError("move components must be finite numbers in [-1,1]")
    for name in ("split", "eject"):
        if type(payload[name]) is not bool:
            raise InputError(f"{name} must be boolean")
    return np.asarray(move, dtype=np.float32), payload["split"], payload["eject"]


@dataclass
class ControlState:
    move: np.ndarray
    split_held: bool = False
    eject_held: bool = False
    pending_split: bool = False
    pending_eject: bool = False

    @classmethod
    def empty(cls) -> ControlState:
        return cls(move=np.zeros(2, dtype=np.float32))

    def update(self, move: np.ndarray, split: bool, eject: bool) -> None:
        self.move = move.copy()
        if split and not self.split_held:
            self.pending_split = True
        if eject and not self.eject_held:
            self.pending_eject = True
        self.split_held = split
        self.eject_held = eject

    def take(self) -> dict[str, Any]:
        result = {"move": self.move.copy(), "split": int(self.pending_split), "eject": int(self.pending_eject)}
        self.pending_split = self.pending_eject = False
        return result


def bot_action(world, player: str, rng: np.random.Generator) -> dict[str, Any]:
    """Locally observable heuristic bot: forage, avoid predators, chase prey.

    This is not a trained policy and must not use global hidden objects.
    """
    owned = world.owned(player)
    if not owned:
        return {"move": np.zeros(2, dtype=np.float32), "split": 0, "eject": 0}
    center = world.centroid(player)
    strongest = max(b.mass for b in owned)
    view = world.rules["view_radius"]
    best: tuple[float, np.ndarray] | None = None
    dangers: list[tuple[float, np.ndarray]] = []
    for ball in world.balls:
        if ball.owner == player:
            continue
        if ball.owner != "pellet" and ball.owner.split("/")[0] == player.split("/")[0]:
            continue
        delta = np.array([ball.x, ball.y], dtype=np.float64) - center
        distance = float(np.linalg.norm(delta))
        if distance > view:
            continue
        if ball.owner != "pellet" and ball.mass >= strongest * world.rules["eat_ratio"]:
            if distance < view * .65:
                dangers.append((distance, -delta))
            continue
        if ball.owner == "pellet":
            score = 3.5 / (distance + 1)
        elif strongest >= ball.mass * world.rules["eat_ratio"]:
            score = (7.0 + math.sqrt(ball.mass)) / (distance + 3)
        else:
            continue
        if best is None or score > best[0]:
            best = (score, delta)
    if dangers:
        direction = _unit(sum((v / max(d, 1) for d, v in dangers), np.zeros(2)))
    elif best is not None:
        direction = _unit(best[1])
    else:
        # Center-seeking exploration on empty maps; bounded random dither.
        direction = _unit(np.array([world.rules['width']/2, world.rules['height']/2]) - center
                          + rng.uniform(-20, 20, 2))
    # Baseline bots deliberately avoid hidden split advantage and are simple controls.
    return {"move": direction, "split": 0, "eject": 0}


class GameSession:
    """A single WebSocket owns a single game session. No shared mutable World."""

    def __init__(self, mode: str = "squad", seed: int | None = None, *, session_id: str | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown match mode: {mode}")
        self.mode = mode
        self.session_id = session_id or secrets.token_hex(8)
        self.seed = secrets.randbits(32) if seed is None else _check_seed(seed)
        self.rules = rules_from_yaml(
            players_per_team=MODES[mode], pellet_count=140 if mode != 'arena' else 200,
            max_policy_steps=450,  # 90 seconds at 5 policy Hz
            view_radius=42.0,
        )
        self.env = BallMARLEnv(rules=self.rules, seed=self.seed)
        self.human_id = "team_0/player_0"
        self.controls = ControlState.empty()
        self.paused = False
        self.finished = False
        self.events_visible = deque(maxlen=MAX_EVENTS_PER_FRAME)
        self.rngs = {}
        self.last_invalid = []
        self.last_center = np.zeros(2, dtype=np.float64)
        self._init_match(self.seed)

    def _init_match(self, seed: int) -> None:
        self.seed = seed
        self.env.reset(seed=self.seed)
        self.last_center = self.env.world.centroid(self.human_id).copy()
        self.controls = ControlState.empty()
        self.paused = False
        self.finished = False
        self.events_visible.clear()
        self.last_invalid = []
        self.rngs = {p: np.random.default_rng(np.random.SeedSequence([seed, i, 523]))
                     for i, p in enumerate(self.env.world.players) if p != self.human_id}
        logger.bind(match_id=self.session_id, mode=self.mode, seed=seed).info("match_started")

    def restart(self, *, new_seed: bool = False) -> None:
        self._init_match(secrets.randbits(32) if new_seed else self.seed)

    def set_input(self, payload: dict[str, Any]) -> None:
        move, split, eject = validate_input(payload)
        self.controls.update(move, split, eject)

    def _visible(self, x: float, y: float, *, center: np.ndarray | None = None) -> bool:
        if center is None:
            center = self.env.world.centroid(self.human_id)
        return math.hypot(x - center[0], y - center[1]) <= self.rules["view_radius"]

    def _record_events(self, events: list[dict[str, Any]]) -> None:
        w = self.env.world
        center = w.centroid(self.human_id)
        visible_ids = {b.id for b in w.balls if self._visible(b.x, b.y, center=center)}
        for event in events:
            actor = event.get("actor")
            target = event.get("target")
            # Full engine event audit stays server-side; client feed is visibility-filtered.
            logger.bind(match_id=self.session_id, player=actor, kind=event['type'],
                        event_id=event.get('event_id'), tick=event.get('tick')).info('engine_event')
            own_or_visible = actor == self.human_id or target == self.human_id
            # For consumed entities, visual LOS is not always recoverable after step;
            # only surface non-human events if their acting ball is still visible.
            own_or_visible |= event.get("eater") in visible_ids or event.get("source") in visible_ids
            if own_or_visible:
                public = {"type": event["type"], "actor": actor, "target": target,
                          "tick": event["tick"], "mass": round(float(event.get("mass", 0)), 2)}
                self.events_visible.append(public)

    def step(self) -> dict[str, Any]:
        if self.paused or self.finished:
            return self.frame()
        w = self.env.world
        actions = {p: self.controls.take() if p == self.human_id else bot_action(w, p, self.rngs[p])
                   for p in self.env.agents}
        _, _, done, _, infos = self.env.step(actions)
        info = infos[self.human_id]
        self.last_invalid = info["invalid_actions"]
        self._record_events(info["events"])
        if all(done.values()):
            self.finished = True
            logger.bind(match_id=self.session_id, seed=self.seed,
                        mass_blue=w.team_mass(0), mass_red=w.team_mass(1)).info("match_finished")
        return self.frame()

    def frame(self) -> dict[str, Any]:
        w = self.env.world
        own_balls = w.owned(self.human_id)
        if own_balls:
            self.last_center = w.centroid(self.human_id).copy()
        center = self.last_center
        # The displayed viewport and minimap obey the same radius as Actor visibility;
        # friendly positions are always available as a public team UI convention.
        balls = []
        visible_enemy_mass = 0.
        for ball in sorted(w.balls, key=lambda b: b.id):
            friendly = ball.owner.startswith('team_0/')
            visible = friendly or self._visible(ball.x, ball.y, center=center)
            if not visible:
                continue
            if not friendly and ball.owner != 'pellet':
                visible_enemy_mass += ball.mass
            balls.append({"id": ball.id, "owner": ball.owner,
                          "x": round(ball.x, 4), "y": round(ball.y, 4),
                          "r": round(ball.radius, 4), "mass": round(ball.mass, 3),
                          "vx": round(ball.vx, 4), "vy": round(ball.vy, 4)})
        own_mass = w.team_mass(0)
        red_mass = w.team_mass(1)
        remaining = max(0, self.rules['max_policy_steps'] - self.env.policy_steps)
        outcome = None
        if self.finished:
            outcome = {"blue_mass": round(own_mass, 2), "red_mass": round(red_mass, 2),
                       "winner": 'blue' if own_mass > red_mass else 'red' if red_mass > own_mass else 'draw'}
        masks = w.action_masks(self.human_id)
        return {"type": "frame", "session_id": self.session_id, "seed": self.seed,
                "mode": self.mode, "tick": self.env.policy_steps, "hz": POLICY_HZ,
                "paused": self.paused, "finished": self.finished,
                "arena": {"width": self.rules['width'], "height": self.rules['height'],
                          "view_radius": self.rules['view_radius']},
                "player": {"id": self.human_id, "x": round(float(center[0]), 4),
                           "y": round(float(center[1]), 4), "mass": round(w.mass(self.human_id), 2),
                           "alive": bool(own_balls), "fragments": len(own_balls)},
                "score": {"blue_mass": round(own_mass, 2),
                          "red_visible_mass": round(visible_enemy_mass, 2),
                          "red_mass": round(red_mass, 2) if self.finished else None},
                "time_left": round(remaining / POLICY_HZ, 1),
                "duration": round(self.rules['max_policy_steps'] / POLICY_HZ, 1),
                "bots": len(self.env.possible_agents) - 1,
                "visible_balls": balls,
                "event_feed": list(self.events_visible),
                "invalid_actions": list(self.last_invalid),
                "abilities": {"split": bool(masks['split'][1]), "eject": bool(masks['eject'][1])},
                "outcome": outcome}


def _check_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    return seed
