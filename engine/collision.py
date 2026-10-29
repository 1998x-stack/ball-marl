"""Deterministic reference and spatial-grid collision backends.

Grid broadphase is an acceleration and must produce exactly the same ordered
eligible contacts as the O(N²) oracle. Candidate state is sampled before mass
transfer; newly unlocked captures wait until the next physics substep.
"""
from __future__ import annotations
import math
from collections import defaultdict

def _grid_pairs(balls, cell_size):
    """Unordered pairs whose radius-AABBs share at least one grid cell."""
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError('cell_size must be positive and finite')
    cells=defaultdict(list)
    ordered=sorted(balls,key=lambda z:z.id)
    for b in ordered:
        r=b.radius
        x_min=math.floor((b.x-r)/cell_size)
        x_max=math.floor((b.x+r)/cell_size)
        y_min=math.floor((b.y-r)/cell_size)
        y_max=math.floor((b.y+r)/cell_size)
        if (x_max-x_min+1)*(y_max-y_min+1)>10000:
            return [(a,z) for i,a in enumerate(ordered) for z in ordered[i+1:]]
        for ix in range(x_min,x_max+1):
            for iy in range(y_min,y_max+1):
                cells[(ix,iy)].append(b.id)
    by_id={b.id:b for b in balls}
    pairs=set()
    for ids in cells.values():
        for idx,a in enumerate(ids):
            for b in ids[idx+1:]:
                pairs.add((min(a,b),max(a,b)))
    return [(by_id[a],by_id[b]) for a,b in sorted(pairs)]

def _reference_pairs(balls):
    ordered=sorted(balls,key=lambda b:b.id)
    return [(a,b) for i,a in enumerate(ordered) for b in ordered[i+1:]]

def consumption_candidates(balls,rules,backend='reference',cell_size=8.):
    if backend=='reference':
        pairs=_reference_pairs(balls)
    elif backend=='grid':
        pairs=_grid_pairs(balls,cell_size)
    else:
        raise ValueError('unsupported collision backend')
    candidates=[]
    for a,b in pairs:
        for predator,prey in ((a,b),(b,a)):
            if predator.owner=='pellet' or predator.owner==prey.owner:
                continue
            if (prey.owner!='pellet'
                    and predator.owner.split('/')[0]==prey.owner.split('/')[0]):
                continue
            if predator.mass<rules['eat_ratio']*prey.mass:
                continue
            limit=predator.radius-rules['penetration']*prey.radius
            margin=limit-math.hypot(predator.x-prey.x,predator.y-prey.y)
            if margin>=0:
                candidates.append(((-margin,-predator.mass,predator.id,prey.id),
                                   predator.id,prey.id))
    candidates.sort(key=lambda x:x[0])
    return candidates

def resolve_consumption(world):
    alive={b.id:b for b in world.balls}
    backend=getattr(world,'collision_backend','reference')
    for _,eater_id,prey_id in consumption_candidates(
            world.balls,world.rules,backend=backend):
        predator,prey=alive.get(eater_id),alive.get(prey_id)
        if predator is None or prey is None:
            continue
        predator.mass+=prey.mass
        del alive[prey_id]
        world._log('consume',actor=predator.owner,target=prey.owner,
                   eater=predator.id,prey=prey.id,mass=prey.mass)
    world.balls=sorted(alive.values(),key=lambda b:b.id)
