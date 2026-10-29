"""Deterministic reference research ruleset (NOT official game mechanics)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import yaml

DEFAULT_RULES = Path(__file__).resolve().parents[1] / 'specifications' / 'game_rules.yaml'


def rules_from_yaml(path=None, **overrides):
    with open(path or DEFAULT_RULES, encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    cfg.update(overrides)
    assert cfg['teams'] == 2, 'research-v0 supports two teams only'
    assert cfg['players_per_team'] >= 1
    assert 0 < cfg['eject_efficiency'] <= 1
    assert cfg['substeps'] >= 1 and cfg['physics_dt'] > 0
    return cfg


@dataclass
class Ball:
    id: int
    owner: str  # stable player ID; pellets use "pellet"
    x: float
    y: float
    mass: float
    vx: float = 0.0
    vy: float = 0.0
    merge_at: int = 0

    @property
    def radius(self):
        return math.sqrt(self.mass) * .35


class World:
    def __init__(self, rules=None):
        self.rules = dict(rules or rules_from_yaml())
        self.reset(0)

    def reset(self, seed):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.tick = 0
        self.next_id = 0
        self.balls = []
        self.players = [f'team_{team}/player_{i}' for team in range(2)
                        for i in range(self.rules['players_per_team'])]
        self.events = []
        self.external_in = 0.0
        self.external_out = 0.0
        for i, player in enumerate(self.players):
            team = i // self.rules['players_per_team']
            idx = i % self.rules['players_per_team']
            x = (25 if team == 0 else 75) + 5 * idx
            y = 25 + 13 * idx
            self.add(player, x, y, self.rules['initial_mass'], external=True)
        for _ in range(self.rules['pellet_count']):
            self.spawn_pellet()
        return self

    def add(self, owner, x, y, mass, *, external=False, merge_at=0):
        if not (math.isfinite(mass) and mass > 0):
            raise ValueError('mass must be finite and positive')
        b = Ball(self.next_id, owner, float(x), float(y), float(mass), merge_at=merge_at)
        self.next_id += 1
        self.balls.append(b)
        if external:
            self.external_in += mass
        return b

    def spawn_pellet(self):
        return self.add('pellet', self.rng.uniform(2, self.rules['width'] - 2),
                        self.rng.uniform(2, self.rules['height'] - 2),
                        self.rules['pellet_mass'], external=True)

    def owned(self, player):
        return sorted([b for b in self.balls if b.owner == player], key=lambda b: b.id)

    def mass(self, owner):
        return sum(b.mass for b in self.balls if b.owner == owner)

    def team_mass(self, team):
        return sum(b.mass for b in self.balls if b.owner.startswith(f'team_{team}/'))

    def total_mass(self):
        return sum(b.mass for b in self.balls)

    def centroid(self, owner):
        balls = self.owned(owner)
        if not balls:
            return np.zeros(2, dtype=np.float64)
        return np.average(np.array([[b.x, b.y] for b in balls]), axis=0,
                          weights=[b.mass for b in balls])

    def _log(self, kind, **details):
        self.events.append({'tick': self.tick, 'type': kind, **details})

    def snapshot(self):
        return {'seed': self.seed, 'tick': self.tick, 'next_id': self.next_id,
                'balls': [asdict(b) for b in sorted(self.balls, key=lambda x: x.id)],
                'rng_state': copy.deepcopy(self.rng.bit_generator.state),
                'external_in': self.external_in, 'external_out': self.external_out,
                'events': copy.deepcopy(self.events), 'rules': copy.deepcopy(self.rules)}

    @classmethod
    def from_snapshot(cls, state):
        world = cls(state['rules'])
        world.seed = state['seed']; world.tick = state['tick']
        world.next_id = state['next_id']
        world.balls = [Ball(**b) for b in state['balls']]
        world.rng = np.random.default_rng()
        world.rng.bit_generator.state = copy.deepcopy(state['rng_state'])
        world.external_in = state['external_in']
        world.external_out = state['external_out']
        world.events = copy.deepcopy(state['events'])
        return world

    def digest(self):
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _validate(self, actions):
        if set(actions) != set(self.players):
            raise ValueError('joint actions must contain every player exactly once')
        normalized = {}
        for p in self.players:
            a = actions[p]
            v = np.asarray(a['move'], dtype=float)
            if v.shape != (2,) or not np.isfinite(v).all() or (np.abs(v) > 1).any():
                raise ValueError('move outside [-1,1]^2 or nonfinite')
            if a['split'] not in (0, 1) or a['eject'] not in (0, 1):
                raise ValueError('split/eject must be binary')
            normalized[p] = {'move': v, 'split': int(a['split']), 'eject': int(a['eject'])}
        return normalized

    def action_masks(self, player):
        owned = self.owned(player)
        split_ok = (len(owned) < self.rules['max_fragments'] and
                    any(b.mass >= self.rules['split_min_mass'] for b in owned))
        eject_ok = any(b.mass > self.rules['eject_min_mass'] + self.rules['eject_mass'] for b in owned)
        return {'split': np.array([True, split_ok]), 'eject': np.array([True, eject_ok])}

    def _apply_split_eject(self, player, command):
        balls = self.owned(player)
        if not balls:
            return
        direction = command['move'].astype(float)
        length = np.linalg.norm(direction)
        direction = direction / length if length > 1e-12 else np.array([1., 0.])
        if command['split'] and self.action_masks(player)['split'][1]:
            target = max(balls, key=lambda b: (b.mass, -b.id))
            if target.mass >= self.rules['split_min_mass']:
                half = target.mass / 2
                target.mass = half
                target.merge_at = self.tick + self.rules['merge_delay_ticks']
                offset = target.radius * 1.1
                new = self.add(player, np.clip(target.x + offset * direction[0], 0, self.rules['width']),
                               np.clip(target.y + offset * direction[1], 0, self.rules['height']),
                               half, merge_at=target.merge_at)
                new.vx = float(direction[0] * self.rules['split_impulse'])
                new.vy = float(direction[1] * self.rules['split_impulse'])
                self._log('split', actor=player, source=target.id, new=new.id, mass=half)
        if command['eject'] and self.action_masks(player)['eject'][1]:
            target = max(self.owned(player), key=lambda b: (b.mass, -b.id))
            dm = self.rules['eject_mass']
            target.mass -= dm
            output_mass = dm * self.rules['eject_efficiency']
            self.external_out += dm - output_mass
            dist = target.radius + math.sqrt(output_mass) * .35 + .2
            pellet = self.add('pellet', np.clip(target.x + dist * direction[0], 0, self.rules['width']),
                              np.clip(target.y + dist * direction[1], 0, self.rules['height']), output_mass)
            self._log('eject', actor=player, source=target.id, pellet=pellet.id, mass=output_mass)

    def _move(self, command_by_owner):
        dt = self.rules['physics_dt']
        for b in sorted(self.balls, key=lambda x: x.id):
            if b.owner == 'pellet':
                continue
            d = command_by_owner[b.owner]['move']
            length = np.linalg.norm(d)
            direction = d / length if length > 1e-12 else np.zeros(2)
            speed = self.rules['speed_min'] + self.rules['speed_scale'] / max(b.mass, 1) ** self.rules['speed_exponent']
            b.vx = float(direction[0] * speed + b.vx * .2)
            b.vy = float(direction[1] * speed + b.vy * .2)
            b.x = float(np.clip(b.x + dt * b.vx, 0., self.rules['width']))
            b.y = float(np.clip(b.y + dt * b.vy, 0., self.rules['height']))

    def _collide(self):
        # Deterministic O(n^2) reference implementation, optimized backend later.
        alive = {b.id: b for b in self.balls}
        ordered = sorted(self.balls, key=lambda b: b.id)
        for eater in ordered:
            if eater.id not in alive or eater.owner == 'pellet':
                continue
            for prey in ordered:
                if prey.id not in alive or prey.id == eater.id or prey.owner == eater.owner:
                    continue
                # Research-v0 excludes consuming friendly players; ejected matter is neutral pellet.
                if prey.owner != 'pellet' and eater.owner.split('/')[0] == prey.owner.split('/')[0]:
                    continue
                if eater.mass < self.rules['eat_ratio'] * prey.mass:
                    continue
                distance = math.hypot(eater.x - prey.x, eater.y - prey.y)
                threshold = eater.radius - self.rules['penetration'] * prey.radius
                if distance <= threshold:
                    eater.mass += prey.mass
                    del alive[prey.id]
                    self._log('consume', actor=eater.owner, target=prey.owner,
                              eater=eater.id, prey=prey.id, mass=prey.mass)
        self.balls = sorted(alive.values(), key=lambda b: b.id)

    def _merge(self):
        for owner in self.players:
            balls = self.owned(owner)
            for i, a in enumerate(balls):
                if a not in self.balls or a.merge_at > self.tick:
                    continue
                for b in balls[i + 1:]:
                    if b not in self.balls or b.merge_at > self.tick:
                        continue
                    if math.hypot(a.x - b.x, a.y - b.y) <= max(a.radius, b.radius):
                        a.mass += b.mass
                        self.balls.remove(b)
                        self._log('merge', actor=owner, a=a.id, b=b.id)

    def step(self, actions):
        normalized = self._validate(actions)
        old_mass = self.total_mass()
        old_in, old_out = self.external_in, self.external_out
        for player in self.players:
            self._apply_split_eject(player, normalized[player])
        for _ in range(self.rules['substeps']):
            self._move(normalized)
            self._collide()
            self._merge()
            self.tick += 1
        delta = self.total_mass() - old_mass
        ledger_delta = (self.external_in - old_in) - (self.external_out - old_out)
        if abs(delta - ledger_delta) > 1e-7 * max(1, old_mass):
            raise AssertionError(f'mass ledger: world {delta}, ledger {ledger_delta}')
        for b in self.balls:
            if b.mass <= 0 or not all(math.isfinite(x) for x in [b.mass, b.x, b.y]):
                raise AssertionError('invalid ball state')
        return self.events
