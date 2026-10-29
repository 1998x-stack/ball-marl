"""Fail-closed rules/checkpoints, adversarial inputs and reproducible evaluation."""
from __future__ import annotations
import copy
import json
import numpy as np
import pytest
from engine.rules import rules_from_yaml, rules_digest, validate_rules
from engine.world import World
from environments.pettingzoo_env import BallMARLEnv
from algorithms.mappo import MAPPO
from evaluation.tournament import evaluate
from training.learner import train, scripted_action

@pytest.mark.parametrize('override',[
    {'discount_gamma':0},{'discount_gamma':1.1},
    {'physics_dt':float('nan')},{'eject_efficiency':1.1},
    {'max_fragments':0},{'players_per_team':0},
    {'players_per_team':33},{'teams':3},
    {'pellet_count':-1},{'substeps':0},
    {'eat_ratio':1},{'width':-1},
    {'max_policy_steps':'100'},{'status':'VERIFIED_OFFICIAL'},
    {'not_a_real_parameter':True},
])
def test_bad_rules_fail_closed(override):
    with pytest.raises(ValueError):
        rules_from_yaml(**override)

def test_rules_digest_stable_but_changes_on_override():
    a=rules_from_yaml()
    b=dict(reversed(list(a.items())))
    assert rules_digest(a)==rules_digest(b)
    assert rules_digest(a)!=rules_digest(rules_from_yaml(pellet_count=3))
    with pytest.raises(ValueError):
        validate_rules({k:v for k,v in a.items() if k!='width'})

def test_snapshot_rules_tamper_detection():
    w=World()
    bad=copy.deepcopy(w.snapshot())
    bad['rules']['eat_ratio']*=2
    with pytest.raises(ValueError,match='integrity'):
        World.from_snapshot(bad)

def test_spawn_with_small_arena_and_large_roster():
    w=World(rules_from_yaml(width=20.0,height=24.0,
                            players_per_team=10,pellet_count=0))
    assert len(w.players)==20
    assert all(0<=b.x<=20 and 0<=b.y<=24 for b in w.balls)

def test_scripted_opponent_cannot_see_hidden_pellets():
    env=BallMARLEnv(rules_from_yaml(pellet_count=0))
    env.reset(seed=3)
    p=env.possible_agents[0]
    for b in env.world.balls:
        if b.owner==p: b.x=b.y=0.
    env.world.add('pellet',99.,99.,2.,external=True)
    a=scripted_action(env,p,rng=np.random.default_rng(919))
    env.world.balls[-1].x=90.
    env.world.balls[-1].y=99.
    b=scripted_action(env,p,rng=np.random.default_rng(919))
    np.testing.assert_array_equal(a['move'],b['move'])

def test_masked_trigger_no_effect_and_strict_mode():
    cfg=rules_from_yaml(max_fragments=1,pellet_count=0)
    env=BallMARLEnv(cfg)
    env.reset(seed=4)
    actions={p:{'move':np.zeros(2,dtype=np.float32),'split':1,'eject':0} for p in env.agents}
    before=env.world.digest()
    _,_,_,_,info=env.step(actions)
    assert 'split' in info[env.possible_agents[0]]['invalid_actions']
    assert all(len(env.world.owned(p))==1 for p in env.possible_agents)
    assert env.world.digest()!=before
    strict=BallMARLEnv(cfg,strict_action_masks=True)
    strict.reset(seed=4)
    with pytest.raises(ValueError,match='illegal'):
        strict.step(actions)

def test_deterministic_renderer():
    env=BallMARLEnv(render_mode='rgb_array')
    env.reset(seed=10)
    a,b=env.render(),env.render()
    assert a.shape==(256,256,3) and a.dtype==np.uint8
    np.testing.assert_array_equal(a,b)
    a[0,0]=0
    assert not np.array_equal(a,env.render())

def test_checkpoint_corruption_rejected_and_run_not_overwritten(tmp_path):
    train(episodes=1,max_steps=3,seed=91,out=tmp_path)
    checkpoint=tmp_path/'policy.pt'
    sidecar=tmp_path/'policy.pt.sha256'
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert len(sidecar.read_text().strip())==64
    assert manifest['rules_hash']==rules_digest(manifest['rules'])
    with pytest.raises(FileExistsError):
        train(episodes=1,max_steps=3,seed=91,out=tmp_path)
    with checkpoint.open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum mismatch'):
        MAPPO.load(checkpoint)
    sidecar.unlink()
    with pytest.raises(ValueError,match='lacks checksum'):
        MAPPO.load(checkpoint)

def test_paired_tournament_reproducible_and_side_balanced(tmp_path):
    a=evaluate(None,episodes=3,max_steps=4,seed=900,side_swaps=True,
               trace_dir=tmp_path/'trace_a')
    b=evaluate(None,episodes=3,max_steps=4,seed=900,side_swaps=True,
               trace_dir=tmp_path/'trace_b')
    assert a==b
    assert a['seed_clusters']==3 and len(a['matches'])==6
    assert [x['trained_side'] for x in a['matches']]==[0,1]*3
    assert len(list((tmp_path/'trace_a').glob('match_*.json')))==6
    lo,hi=a['score_95pct_cluster_bootstrap_ci']
    assert 0<=lo<=hi<=1
    with pytest.raises(ValueError):
        evaluate(None,episodes=0)

def test_randomized_5v5_soak_mass_and_snapshot():
    cfg=rules_from_yaml(players_per_team=5,pellet_count=20,max_policy_steps=20)
    for seed in range(5):
        env=BallMARLEnv(cfg)
        env.reset(seed=seed)
        mirror=BallMARLEnv(cfg)
        mirror.reset(seed=seed)
        rng=np.random.default_rng(seed*17+10)
        while env.agents:
            cmds={}
            for p in env.agents:
                m=env.world.action_masks(p)
                cmds[p]={'move':rng.uniform(-1,1,2).astype(np.float32),
                         'split':int(m['split'][1] and rng.random()<.15),
                         'eject':int(m['eject'][1] and rng.random()<.1)}
            env.step(cmds)
            mirror.step(cmds)
            assert env.world.digest()==mirror.world.digest()
            assert env.world.total_mass()==pytest.approx(env.world.external_in-env.world.external_out)

def test_snapshot_entity_corruption_rejected():
    w=World(rules_from_yaml(pellet_count=0))
    bad=w.snapshot()
    bad['balls'][0]['mass']=-10
    with pytest.raises(ValueError,match='invalid entity'):
        World.from_snapshot(bad)
    bad2=w.snapshot()
    bad2['balls'][1]['id']=bad2['balls'][0]['id']
    with pytest.raises(ValueError,match='entity ID'):
        World.from_snapshot(bad2)

def test_one_cluster_cannot_support_bootstrap_interval():
    result=evaluate(None,episodes=1,max_steps=2,seed=1)
    assert result['score_95pct_cluster_bootstrap_ci'] is None

def test_full_5v5_train_infer_evaluate_pipeline(tmp_path):
    report, agent = train(episodes=1, max_steps=4, seed=307,
                          out=tmp_path, players_per_team=5, epochs=1)
    assert report[0]['batch_size'] == 20
    model, meta = MAPPO.load(tmp_path / 'policy.pt')
    assert meta['rules']['players_per_team'] == 5
    result = evaluate(tmp_path / 'policy.pt', episodes=2, seed=191, side_swaps=True)
    assert result['seed_clusters'] == 2 and len(result['matches']) == 4
    assert all(x['policy_steps'] == 4 for x in result['matches'])

def test_training_against_frozen_checkpoint_and_rules_guard(tmp_path):
    base=tmp_path/'base'
    train(episodes=1,max_steps=4,seed=200,out=base,epochs=1)
    checkpoint=base/'policy.pt'
    follower=tmp_path/'frozen'
    report,_=train(episodes=1,max_steps=4,seed=201,out=follower,epochs=1,
                   opponent_checkpoint=checkpoint)
    assert report[0]['batch_size']==8
    meta=json.loads((follower/'manifest.json').read_text())
    assert meta['mode']=='frozen_policy'
    assert meta['opponent_checkpoint_sha256'] is not None
    with pytest.raises(ValueError,match='ruleset differs'):
        train(episodes=1,max_steps=5,seed=202,out=tmp_path/'bad',epochs=1,
              opponent_checkpoint=checkpoint)
