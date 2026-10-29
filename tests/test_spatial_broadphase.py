"""Grid broadphase must exactly match the reference collision oracle."""
from __future__ import annotations
import numpy as np
from engine.rules import rules_from_yaml
from engine.world import World
from engine.collision import consumption_candidates

def test_spatial_broadphase_matches_exact_oracle_over_random_worlds():
    rules=rules_from_yaml(players_per_team=5,pellet_count=80)
    for seed in range(40):
        world=World(rules,collision_backend='grid')
        world.reset(seed)
        rng=np.random.default_rng(seed+903)
        for b in world.balls:
            if b.owner!='pellet':
                b.x=float(rng.uniform(0,rules['width']))
                b.y=float(rng.uniform(0,rules['height']))
                b.mass=float(rng.uniform(1,140))
        expected=consumption_candidates(world.balls,rules,backend='reference')
        actual=consumption_candidates(world.balls,rules,backend='grid')
        assert expected==actual, f'collision mismatch seed={seed}'

def test_grid_and_reference_world_produce_equal_physical_trajectories():
    rules=rules_from_yaml(players_per_team=3,pellet_count=40,max_policy_steps=15)
    slow=World(rules,collision_backend='reference')
    fast=World(rules,collision_backend='grid')
    slow.reset(222);fast.reset(222)
    rng=np.random.default_rng(2)
    for _ in range(15):
        joint={}
        for p in slow.players:
            mask=slow.action_masks(p)
            joint[p]={'move':rng.uniform(-1,1,2).astype(np.float32),
                      'split':int(mask['split'][1] and rng.random()<.15),
                      'eject':int(mask['eject'][1] and rng.random()<.1)}
        slow.step(joint);fast.step(joint)
        a=slow.snapshot();b=fast.snapshot()
        a.pop('collision_backend');b.pop('collision_backend')
        assert a==b

def test_grid_boundary_and_large_ball_oracle_parity():
    rules=rules_from_yaml(players_per_team=1,pellet_count=0)
    w=World(rules)
    a=w.owned(w.players[0])[0]
    b=w.owned(w.players[1])[0]
    a.x=7.99;b.x=8.01;a.y=b.y=8.
    a.mass=500;b.mass=1
    for size in (.3,1,2,8,100):
        assert consumption_candidates(w.balls,rules,'grid',size)==consumption_candidates(w.balls,rules,'reference')
