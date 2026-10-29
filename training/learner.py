"""End-to-end local MAPPO rollout / update / checkpoint pipeline."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from algorithms.mappo import MAPPO
from environments.pettingzoo_env import BallMARLEnv
from actions.masks import safe_action


def scripted_action(env,player,kind='heuristic',rng=None):
    rng=rng or np.random.default_rng(0)
    center=env.world.centroid(player)
    candidates=[b for b in env.world.balls if b.owner=='pellet']
    if kind=='random' or not candidates:
        direction=rng.uniform(-1,1,2)
    else:
        target=min(candidates,key=lambda b: (b.x-center[0])**2+(b.y-center[1])**2)
        direction=np.array([target.x-center[0],target.y-center[1]])
        direction=direction/(np.linalg.norm(direction)+1e-9)
    return safe_action(env,player,direction,0,0)


def train(episodes=5,max_steps=40,out='runs/smoke',seed=42,epochs=3):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    cfg={'max_policy_steps':int(max_steps)}
    env=BallMARLEnv(rules=__import__('engine.world',fromlist=['rules_from_yaml']).rules_from_yaml(**cfg),seed=seed)
    obs,_=env.reset(seed=seed)
    learner=MAPPO(len(obs[env.agents[0]]['features']),len(env.state(0)),gamma=env.rules['discount_gamma'],epochs=epochs)
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed+999)
    summary=[]
    for ep in range(episodes):
        obs,_=env.reset(seed=seed+ep)
        traces={p:[] for p in env.possible_agents if p.startswith('team_0/')}
        terminal_info=None
        while env.agents:
            global_state=env.state(0)
            with torch.no_grad():
                current_value=float(learner.critic(torch.tensor(global_state,dtype=torch.float32).unsqueeze(0)).item())
            joint={}
            logps={}
            for p in env.agents:
                if p.startswith('team_0/'):
                    joint[p],logps[p]=learner.choose(obs[p])
                else:
                    joint[p]=scripted_action(env,p,rng=rng)
            old_obs=obs
            obs,rewards,terminations,truncations,infos=env.step(joint)
            terminal_info=infos
            done=all(terminations.values())
            next_state=env.state(0)
            with torch.no_grad():
                next_value=0. if done else float(learner.critic(torch.tensor(next_state,dtype=torch.float32).unsqueeze(0)).item())
            for p in traces:
                a=joint[p]
                traces[p].append({'obs':old_obs[p]['features'].copy(),'state':global_state.copy(),
                  'split_mask':old_obs[p]['split_mask'].copy(),'eject_mask':old_obs[p]['eject_mask'].copy(),
                  'move':a['move'].copy(),'split':a['split'],'eject':a['eject'],
                  'logp':logps[p],'value':current_value,'next_value':next_value,
                  'reward':float(rewards[p]),'done':done})
        metrics=learner.update(list(traces.values()))
        m=terminal_info[env.possible_agents[0]]['mass_by_team']
        result={'episode':ep,'seed':seed+ep,'team_mass':m,'rank_reward':np.sign(m[0]-m[1]).item(),
                'policy_steps':env.policy_steps,**metrics}
        summary.append(result)
        with (output/'train.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(result,ensure_ascii=False)+'\n')
    metadata={'seed':seed,'episodes':episodes,'rules':env.rules,'mode':'scripted_opponent',
              'note':'Proof-of-pipeline only, not verified commercial mechanics.'}
    learner.save(output/'policy.pt',metadata)
    (output/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary,learner
