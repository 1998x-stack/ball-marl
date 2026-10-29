"""Deterministic O(N²) reference collision kernel; penetration outranks player ID.

Pairs are generated from pre-resolution state; newly enabled capture chains
become eligible in the next physics substep.
"""
from __future__ import annotations
import math

def consumption_candidates(balls, rules):
    ordered = sorted(balls, key=lambda b: b.id)
    candidates = []
    for predator in ordered:
        if predator.owner == 'pellet':
            continue
        for prey in ordered:
            if predator.id == prey.id or predator.owner == prey.owner:
                continue
            if (prey.owner != 'pellet'
                    and predator.owner.split('/')[0] == prey.owner.split('/')[0]):
                continue
            if predator.mass < rules['eat_ratio'] * prey.mass:
                continue
            limit = predator.radius - rules['penetration'] * prey.radius
            margin = limit - math.hypot(predator.x-prey.x, predator.y-prey.y)
            if margin >= 0:
                candidates.append(((-margin, -predator.mass, predator.id, prey.id),
                                   predator.id, prey.id))
    candidates.sort(key=lambda x: x[0])
    return candidates

def resolve_consumption(world):
    alive = {b.id: b for b in world.balls}
    for _, eater_id, prey_id in consumption_candidates(world.balls, world.rules):
        predator, prey = alive.get(eater_id), alive.get(prey_id)
        if predator is None or prey is None:
            continue
        predator.mass += prey.mass
        del alive[prey_id]
        world._log('consume', actor=predator.owner, target=prey.owner,
                   eater=predator.id, prey=prey.id, mass=prey.mass)
    world.balls = sorted(alive.values(), key=lambda b: b.id)
