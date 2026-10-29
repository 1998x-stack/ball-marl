"""Held-out seed evaluation; separate train/eval records and objectives."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from algorithms.mappo import MAPPO
from environments.pettingzoo_env import BallMARLEnv
from engine.world import rules_from_yaml
from training.learner import scripted_action


def evaluate(policy_path=None,episodes=4,max_steps=None,seed=1000,enemy='heuristic',trace_dir=None):
    agent, metadata = MAPPO.load(policy_path) if policy_path else (None,{})
    cfg=dict(metadata.get('rules') or rules_from_yaml())
    if max_steps is not None: cfg['max_policy_steps']=max_steps
    env=BallMARLEnv(cfg)
    output=[]
    rng=np.random.default_rng(seed+7)
    if trace_dir: Path(trace_dir).mkdir(parents=True,exist_ok=True)
    for i in range(episodes):
        obs,_=env.reset(seed=seed+i)
        history=[]
        while env.agents:
            actions={}
            for p in env.agents:
                if p.startswith('team_0/') and agent:
                    actions[p],_=agent.choose(obs[p],deterministic=True)
                else:
                    actions[p]=scripted_action(env,p,kind=enemy if p.startswith('team_1/') else 'random',rng=rng)
            obs,rewards,terms,truncs,infos=env.step(actions)
            history.append({'tick':env.world.tick,'hash':env.world.digest(),
                            'mass':[env.world.team_mass(t) for t in (0,1)],
                            'events':infos[env.possible_agents[0]]['events']})
        m=[env.world.team_mass(t) for t in (0,1)]
        output.append({'seed':seed+i,'team_mass':m,'win':float(m[0]>m[1]),
                       'draw':float(m[0]==m[1]),'policy_steps':env.policy_steps})
        if trace_dir:
            (Path(trace_dir)/f'match_{i:03d}.json').write_text(json.dumps(history,indent=2))
    return {'matches':output,'win_rate':sum(x['win'] for x in output)/len(output),
            'draw_rate':sum(x['draw'] for x in output)/len(output),
            'mean_mass_margin':sum(x['team_mass'][0]-x['team_mass'][1] for x in output)/len(output),
            'ruleset':cfg['version']}
