"""Zero-oracle actor features; padded fixed-entity features for baseline."""
from __future__ import annotations
import math
import numpy as np

OTHER_SLOTS = 10
PELLET_SLOTS = 8
OBS_DIM = 6 + 5 * OTHER_SLOTS + 3 * PELLET_SLOTS + 4


def actor_features(world, player):
    own = world.owned(player)
    center = world.centroid(player)
    view = world.rules['view_radius']
    scale = max(world.rules['width'], world.rules['height'])
    total = sum(b.mass for b in own)
    largest = max([b.mass for b in own], default=0.0)
    features = [center[0] / world.rules['width'], center[1] / world.rules['height'],
                math.log1p(total) / 8, math.log1p(largest) / 8,
                len(own) / max(1, world.rules['max_fragments']), float(bool(own))]
    others = []
    pellets = []
    for b in world.balls:
        if b.owner == player:
            continue
        delta = np.array([b.x,b.y]) - center
        distance = float(np.linalg.norm(delta))
        if distance > view:
            continue
        if b.owner == 'pellet':
            pellets.append((distance,b.id,[delta[0]/view,delta[1]/view,math.log1p(b.mass)/8]))
        else:
            relation = 1. if b.owner.split('/')[0] == player.split('/')[0] else -1.
            others.append((distance,b.id,[delta[0]/view,delta[1]/view,
                                         math.log1p(b.mass)/8,relation,1.]))
    others.sort(key=lambda t:(t[0],t[1]))
    pellets.sort(key=lambda t:(t[0],t[1]))
    for _,_,entry in others[:OTHER_SLOTS]: features.extend(entry)
    features.extend([0.] * (5 * (OTHER_SLOTS-min(len(others),OTHER_SLOTS))))
    for _,_,entry in pellets[:PELLET_SLOTS]: features.extend(entry)
    features.extend([0.] * (3 * (PELLET_SLOTS-min(len(pellets),PELLET_SLOTS))))
    features.extend([min(len(others)/OTHER_SLOTS,1.),min(len(pellets)/PELLET_SLOTS,1.),
                     world.tick/(world.rules['substeps']*world.rules['max_policy_steps']),
                     min(largest/max(world.rules['initial_mass'],1),10.) /10])
    out = np.asarray(features,dtype=np.float32)
    assert out.shape == (OBS_DIM,) and np.isfinite(out).all()
    return out


def critic_features(world, team):
    vec=[]
    for p in world.players:
        center=world.centroid(p)
        balls=world.owned(p)
        vec.extend([center[0]/world.rules['width'], center[1]/world.rules['height'],
                    math.log1p(sum(b.mass for b in balls))/8,
                    math.log1p(max([b.mass for b in balls],default=0.))/8,
                    len(balls)/max(1,world.rules['max_fragments'])])
    vec.extend([world.tick/(world.rules['substeps']*world.rules['max_policy_steps']),
                math.log1p(world.team_mass(0))/8,math.log1p(world.team_mass(1))/8,
                float(team)])
    return np.asarray(vec,dtype=np.float32)
