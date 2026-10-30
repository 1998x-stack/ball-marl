import numpy as np
import pytest
from environments.pettingzoo_env import BallMARLEnv
from rewards.objective import potential,reward_components
from engine.world import rules_from_yaml
from training.learner import scripted_action


def test_observation_and_action_spaces():
    env=BallMARLEnv()
    obs,_=env.reset(seed=123)
    assert all(env.observation_space(p).contains(obs[p]) for p in env.agents)
    assert len(env.state(0))==5*len(env.possible_agents)+4
    acts={p:scripted_action(env,p) for p in env.agents}
    assert all(env.action_space(p).contains(acts[p]) for p in env.agents)
    env.step(acts)


def test_joint_step_computes_one_shared_state_hash(monkeypatch):
    env = BallMARLEnv(rules_from_yaml(players_per_team=5, pellet_count=0))
    original_digest = env.world.digest
    calls = 0

    def counted_digest():
        nonlocal calls
        calls += 1
        return original_digest()

    monkeypatch.setattr(env.world, 'digest', counted_digest)
    actions = {player: scripted_action(env, player) for player in env.agents}
    _, _, _, _, infos = env.step(actions)

    assert calls == 1
    assert {info['state_hash'] for info in infos.values()} == {original_digest()}


def test_actor_does_not_observe_hidden_enemy():
    env=BallMARLEnv(rules_from_yaml(pellet_count=0))
    env.reset(seed=9)
    enemy=env.world.owned(env.possible_agents[-1])[0]
    enemy.x=100; enemy.y=100
    env.world.owned(env.possible_agents[0])[0].x=0
    env.world.owned(env.possible_agents[0])[0].y=0
    before=env.observe(env.possible_agents[0])['features'].copy()
    enemy.x=99;enemy.mass=40
    after=env.observe(env.possible_agents[0])['features']
    np.testing.assert_array_equal(before,after)


def test_end_of_match_is_termination_not_truncation():
    env=BallMARLEnv(rules_from_yaml(max_policy_steps=2,pellet_count=0))
    obs,_=env.reset(seed=8)
    for i in range(2):
        a={p:scripted_action(env,p) for p in env.agents}
        obs,reward,terminated,truncated,infos=env.step(a)
    assert all(terminated.values()) and not any(truncated.values())
    assert not env.agents
    with pytest.raises(RuntimeError):env.step(a)


def test_pbrs_telescoping_terminal_zero():
    gamma=.99; beta=.05
    masses=[10.,15.,13.,24.]
    shaping=[]
    for i in range(len(masses)-1):
        x=reward_components({0:masses[i],1:10.}, {0:masses[i+1],1:10.}, gamma,beta,
                            terminated=i==len(masses)-2)
        shaping.append(x[0]['shaping'])
    discounted=sum(gamma**i*r for i,r in enumerate(shaping))
    assert discounted==pytest.approx(-beta*potential(masses[0]))


def test_rank_reward_win_loss_tie():
    a=reward_components({0:1,1:1},{0:4,1:2},.99,0,True)
    b=reward_components({0:1,1:1},{0:2,1:4},.99,0,True)
    c=reward_components({0:1,1:1},{0:4,1:4},.99,0,True)
    assert (a[0]['objective'],a[1]['objective'])==(1.,-1.)
    assert (b[0]['objective'],b[1]['objective'])==(-1.,1.)
    assert (c[0]['objective'],c[1]['objective'])==(0.,0.)


def test_agent_identity_constant_during_split():
    env=BallMARLEnv(rules_from_yaml(pellet_count=0))
    env.reset(seed=2)
    ids=env.agents.copy()
    actions={p:scripted_action(env,p) for p in ids}
    actions[ids[0]]['split']=1
    env.step(actions)
    assert env.agents==ids
    assert len(env.world.owned(ids[0]))==2
