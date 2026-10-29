"""Reference collision determinism, integrity and measurable benchmark."""
from __future__ import annotations
import copy
import pytest
from engine.collision import consumption_candidates
from engine.rules import rules_from_yaml
from engine.world import World
from evaluation.benchmark import run_benchmark

def _competition(reverse=False):
    w=World(rules_from_yaml(players_per_team=1,pellet_count=0))
    a=w.owned(w.players[0])[0]
    b=w.owned(w.players[1])[0]
    a.mass=b.mass=25
    a.x,a.y=40.,50.
    b.x,b.y=41.,50.
    w.add('pellet',41.,50.,1.,external=True)
    if reverse: w.balls.reverse()
    return w

def test_geometric_collision_priority_not_player_id():
    w=_competition()
    candidates=consumption_candidates(w.balls,w.rules)
    assert candidates[0][1]==w.owned(w.players[1])[0].id
    assert candidates[0][2]==2
    w._collide()
    assert w.mass(w.players[1])==pytest.approx(26.)
    assert w.mass(w.players[0])==pytest.approx(25.)

def test_collision_order_independent_of_input_list_order():
    a,b=_competition(),_competition(reverse=True)
    a._collide(); b._collide()
    assert a.digest()==b.digest()

def test_event_hash_chain_rejects_tampering():
    w=_competition()
    w._collide()
    snap=w.snapshot()
    corrupted=copy.deepcopy(snap)
    corrupted['events'][0]['mass']+=1
    with pytest.raises(ValueError,match='event chain integrity'):
        World.from_snapshot(corrupted)
    assert World.from_snapshot(snap).digest()==w.digest()
    assert snap['next_event_id']==len(snap['events'])

def test_state_digest_does_not_serialize_event_history():
    w=_competition()
    w._collide()
    short=w.snapshot(include_events=False)
    assert 'events' not in short and len(short['event_chain_hash'])==64

def test_benchmark_reports_real_invariants():
    r=run_benchmark(episodes=1,steps=4,players_per_team=2,pellet_count=3)
    assert r['policy_steps']==4
    assert r['physics_steps']==16
    assert r['steps_per_second']>0
    assert r['step_latency_ms_p95']>=0
    assert r['max_global_mass_residual']<1e-6
