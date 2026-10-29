"""Seed-controlled tournaments with optional paired team-side swaps.

Side-swapped matches from one seed form a cluster; they are not independent.
This is a research benchmark, not evidence of official-game skill.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from algorithms.mappo import MAPPO
from engine.rules import rules_digest, rules_from_yaml
from environments.pettingzoo_env import BallMARLEnv
from training.learner import scripted_action
from training.artifacts import atomic_write_json

def _score(mass, trained_side):
    mine, theirs = mass[trained_side], mass[1-trained_side]
    return float(mine > theirs), float(mine == theirs)

def _cluster_bootstrap_ci(cluster_scores, seed, bootstrap_samples=1000):
    if not cluster_scores:
        raise ValueError('no tournament clusters')
    rng = np.random.default_rng(seed+876543)
    a = np.asarray(cluster_scores, dtype=np.float64)
    if len(a) < 2:
        return None
    indices = rng.integers(0,len(a),size=(bootstrap_samples,len(a)))
    samples = a[indices].mean(axis=1)
    return [float(x) for x in np.quantile(samples,[.025,.975])]

def evaluate(policy_path=None,episodes=4,max_steps=None,seed=1000,
             enemy='heuristic',trace_dir=None,side_swaps=False,
             allow_legacy_checkpoint=False):
    if not isinstance(episodes,int) or episodes < 1:
        raise ValueError('episodes must be >=1')
    if not isinstance(seed,int) or seed < 0:
        raise ValueError('seed must be nonnegative')
    if max_steps is not None and (not isinstance(max_steps,int) or max_steps < 1):
        raise ValueError('max_steps must be >=1')
    if enemy not in ('heuristic','random'):
        raise ValueError('unknown enemy policy')
    agent,metadata=MAPPO.load(policy_path,allow_legacy=allow_legacy_checkpoint) if policy_path else (None,{})
    cfg=dict(metadata.get('rules') or rules_from_yaml())
    if metadata.get('rules_hash') and metadata['rules_hash'] != rules_digest(cfg):
        raise ValueError('checkpoint metadata rules hash mismatch')
    if max_steps is not None:
        cfg['max_policy_steps']=max_steps
    env=BallMARLEnv(cfg)
    if agent is not None:
        if agent.actor.body[0].in_features != env.observation_space(env.possible_agents[0]).spaces['features'].shape[0]:
            raise ValueError('observation dimension differs from checkpoint')
        if agent.critic.model[0].in_features != len(env.state(0)):
            raise ValueError('critic state dimension differs from checkpoint')
    trace_path=Path(trace_dir) if trace_dir else None
    if trace_path:
        trace_path.mkdir(parents=True,exist_ok=True)
    output=[]
    clusters=[]
    for i in range(episodes):
        results=[]
        for trained_side in ((0,1) if side_swaps else (0,)):
            match_seed=seed+i
            rng=np.random.default_rng(np.random.SeedSequence([match_seed,trained_side,17]))
            obs,_=env.reset(seed=match_seed)
            history=[]
            while env.agents:
                actions={}
                for p in env.agents:
                    team=int(p.split('/')[0].split('_')[1])
                    if team==trained_side and agent is not None:
                        actions[p],_=agent.choose(obs[p],deterministic=True)
                    else:
                        kind=enemy if team != trained_side else 'random'
                        actions[p]=scripted_action(env,p,kind=kind,rng=rng)
                obs,rewards,terms,truncs,infos=env.step(actions)
                if trace_path:
                    history.append({'tick':env.world.tick,'hash':env.world.digest(),
                                    'mass':[env.world.team_mass(t) for t in (0,1)],
                                    'events':infos[env.possible_agents[0]]['events']})
            mass=[env.world.team_mass(t) for t in (0,1)]
            win,draw=_score(mass,trained_side)
            row={'seed':match_seed,'trained_side':trained_side,'team_mass':mass,
                 'win':win,'draw':draw,'score':win+.5*draw,
                 'mass_margin':mass[trained_side]-mass[1-trained_side],
                 'policy_steps':env.policy_steps}
            output.append(row)
            results.append(row['score'])
            if trace_path:
                suffix=f'_side{trained_side}' if side_swaps else ''
                atomic_write_json(trace_path/f'match_{i:03d}{suffix}.json',history)
        clusters.append(float(np.mean(results)))
    return {'matches':output,'seed_clusters':episodes,'paired_side_swaps':bool(side_swaps),
            'win_rate':float(np.mean([x['win'] for x in output])),
            'draw_rate':float(np.mean([x['draw'] for x in output])),
            'mean_match_score':float(np.mean([x['score'] for x in output])),
            'score_95pct_cluster_bootstrap_ci':_cluster_bootstrap_ci(clusters,seed),
            'mean_mass_margin':float(np.mean([x['mass_margin'] for x in output])),
            'ruleset':cfg['version'],'evaluation_rules_hash':rules_digest(cfg),
            'opponent':enemy,
            'provenance':'research rules, not calibrated official physics'}
