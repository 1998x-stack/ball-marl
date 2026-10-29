import copy
import math
import numpy as np
import pytest
from engine.world import World,rules_from_yaml
from actions.masks import safe_action
from environments.pettingzoo_env import BallMARLEnv


def zero_actions(w):
    return {p:{'move':np.zeros(2,dtype=np.float32),'split':0,'eject':0} for p in w.players}


def test_initial_mass_and_ledger():
    w=World()
    assert w.total_mass()==pytest.approx(w.external_in-w.external_out)
    for _ in range(5): w.step(zero_actions(w))
    assert w.total_mass()==pytest.approx(w.external_in-w.external_out)


def test_split_creates_child_and_conserves_mass():
    w=World(rules_from_yaml(pellet_count=0))
    old=w.total_mass(); p=w.players[0]
    a=zero_actions(w);a[p]['split']=1
    w.step(a)
    assert len(w.owned(p))==2
    assert w.total_mass()==pytest.approx(old)
    assert any(e['type']=='split' for e in w.events)


def test_max_fragments_mask():
    w=World(rules_from_yaml(pellet_count=0,max_fragments=1))
    assert not w.action_masks(w.players[0])['split'][1]
    e=BallMARLEnv(rules=w.rules)
    a=safe_action(e,e.agents[0],[0,0],split=1)
    assert a['split']==0


def test_eject_mass_accounting():
    w=World(rules_from_yaml(pellet_count=0))
    before=w.total_mass()
    a=zero_actions(w);a[w.players[0]]['eject']=1
    w.step(a)
    assert w.total_mass() == pytest.approx(before-w.rules['eject_mass']*(1-w.rules['eject_efficiency']))
    assert w.external_out==pytest.approx(w.rules['eject_mass']*(1-w.rules['eject_efficiency']))


def test_snapshot_replay_identical():
    w=World()
    state=w.snapshot()
    clone=World.from_snapshot(state)
    assert w.digest()==clone.digest()
    actions=zero_actions(w)
    for i in range(12):
        if i in (2,5): actions[w.players[0]]['split']=1
        else: actions[w.players[0]]['split']=0
        w.step(actions);clone.step(actions)
        assert w.digest()==clone.digest()


def test_invalid_joint_actions():
    w=World()
    with pytest.raises(ValueError):w.step({})
    actions=zero_actions(w)
    actions[w.players[0]]['move']=np.array([np.nan,0.])
    with pytest.raises(ValueError):w.step(actions)


def test_consumption_tombstones_once():
    w=World(rules_from_yaml(pellet_count=0))
    p=w.players[0]
    w.add('pellet',w.owned(p)[0].x,w.owned(p)[0].y,1.,external=True)
    old=w.mass(p)
    w.step(zero_actions(w))
    assert w.mass(p)==pytest.approx(old+1)
    assert len([e for e in w.events if e['type']=='consume' and e['target']=='pellet'])==1


def test_bounds_after_high_speed():
    w=World(rules_from_yaml(pellet_count=0,split_impulse=300.0))
    for b in w.owned(w.players[0]):b.x=99.99;b.y=99.99
    a=zero_actions(w);a[w.players[0]]['move']=np.ones(2,dtype=np.float32);a[w.players[0]]['split']=1
    w.step(a)
    assert all(0<=b.x<=w.rules['width'] and 0<=b.y<=w.rules['height'] for b in w.balls)


def test_no_negative_mass_under_random_actions():
    e=BallMARLEnv()
    rng=np.random.default_rng(33)
    for seed in range(5):
        e.reset(seed=seed)
        for _ in range(30):
            if not e.agents:break
            a={p:safe_action(e,p,rng.uniform(-1,1,size=2),int(rng.integers(2)),int(rng.integers(2))) for p in e.agents}
            e.step(a)
            assert all(b.mass>0 and math.isfinite(b.mass) for b in e.world.balls)
            assert e.world.total_mass()==pytest.approx(e.world.external_in-e.world.external_out)
