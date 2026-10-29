import json
import numpy as np
import pytest
import torch
from pathlib import Path
from algorithms.mappo import MAPPO
from environments.pettingzoo_env import BallMARLEnv
from training.learner import train
from evaluation.tournament import evaluate
from credit_assignment.difference_rewards import counterfactual_mass_delta
from league.opponent_pool import register


def test_logprob_roundtrip_finite():
    env=BallMARLEnv()
    obs,_=env.reset(seed=7)
    o=obs[env.agents[0]]
    model=MAPPO(len(o['features']),len(env.state(0)))
    action,old_logp=model.choose(o)
    feat=torch.tensor(o['features']).unsqueeze(0)
    sm=torch.tensor(o['split_mask'].astype(bool)).unsqueeze(0)
    em=torch.tensor(o['eject_mask'].astype(bool)).unsqueeze(0)
    move=torch.tensor(action['move']).unsqueeze(0)
    new_lp=model.actor.logprob(feat,sm,em,move,torch.tensor([action['split']]),torch.tensor([action['eject']]))
    assert np.isfinite(old_logp)
    assert new_lp.item()==pytest.approx(old_logp,abs=1e-5)


def test_train_infer_evaluate_and_register(tmp_path):
    report,model=train(episodes=2,max_steps=8,out=tmp_path,seed=101,epochs=2)
    assert len(report)==2
    assert report[0]['batch_size']==16
    assert np.isfinite(report[-1]['loss'])
    saved=tmp_path/'policy.pt'
    loaded,meta=MAPPO.load(saved)
    assert meta['episodes']==2
    env=BallMARLEnv(rules=meta['rules'])
    o,_=env.reset(seed=23)
    action,_=loaded.choose(o[env.agents[0]],deterministic=True)
    assert env.action_space(env.agents[0]).contains(action)
    ev=evaluate(saved,episodes=2,seed=500,trace_dir=tmp_path/'traces')
    assert 0<=ev['win_rate']<=1 and len(ev['matches'])==2
    assert len(list((tmp_path/'traces').glob('match_*.json')))==2
    pool=register(tmp_path/'pool.json',saved,'smoke')
    assert len(pool)==1
    assert len(register(tmp_path/'pool.json',saved,'smoke'))==1
    assert json.loads((tmp_path/'manifest.json').read_text())['mode']=='scripted_opponent'


def test_counterfactual_snapshot_unchanged():
    env=BallMARLEnv()
    env.reset(seed=2)
    p=env.agents[0]
    actions={q:{'move':np.zeros(2,dtype=np.float32),'split':0,'eject':0} for q in env.agents}
    actions[p]['split']=1
    before=env.world.digest()
    result=counterfactual_mass_delta(env.world,actions,p,horizon=2)
    assert result['horizon']==2 and np.isfinite(result['difference'])
    assert env.world.digest()==before
