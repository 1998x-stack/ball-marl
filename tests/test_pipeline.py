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


def test_batched_deterministic_actions_match_individual_inference():
    env = BallMARLEnv()
    observations, _ = env.reset(seed=17)
    model = MAPPO(len(observations[env.agents[0]]['features']), len(env.state(0)))

    batched = model.choose_many(observations, deterministic=True)

    assert set(batched) == set(env.agents)
    for player, observation in observations.items():
        individual_action, individual_logp = model.choose(observation, deterministic=True)
        batched_action, batched_logp = batched[player]
        np.testing.assert_allclose(batched_action['move'], individual_action['move'], atol=1e-6)
        assert batched_action['split'] == individual_action['split']
        assert batched_action['eject'] == individual_action['eject']
        assert batched_logp == pytest.approx(individual_logp, abs=1e-5)


def test_new_policy_prefers_no_eject_when_action_is_legal():
    env = BallMARLEnv()
    observations, _ = env.reset(seed=23)
    observation = observations[env.agents[0]]
    model = MAPPO(len(observation['features']), len(env.state(0)))
    features = torch.as_tensor(observation['features']).unsqueeze(0)
    split_mask = torch.as_tensor(observation['split_mask'].astype(bool)).unsqueeze(0)
    eject_mask = torch.as_tensor(observation['eject_mask'].astype(bool)).unsqueeze(0)

    _, _, eject_distribution = model.actor.distribution(features, split_mask, eject_mask)

    assert eject_mask[0, 1]
    assert eject_distribution.probs[0, 1].item() < 0.05


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
    assert meta['initial_eject_logits'] == [2.0, -2.0]


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
