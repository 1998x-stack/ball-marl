"""End-to-end local MAPPO rollout / update / checkpoint pipeline."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from algorithms.mappo import MAPPO
from environments.pettingzoo_env import BallMARLEnv
from actions.masks import safe_action
from engine.rules import rules_digest, rules_from_yaml
from training.artifacts import atomic_write_json, sha256_file


def scripted_action(env,player,kind='heuristic',rng=None):
    if rng is None:
        rng=np.random.default_rng(0)
    center=env.world.centroid(player)
    # Scripted competitors follow the same local visibility rule as the actor.
    # Prior implementation searched the entire world, unfairly granting oracle vision.
    view=env.rules['view_radius']
    candidates=[b for b in env.world.balls if b.owner=='pellet'
                and np.linalg.norm(np.array([b.x,b.y])-center) <= view]
    if kind=='random' or not candidates:
        direction=rng.uniform(-1,1,2)
    else:
        target=min(candidates,key=lambda b: (b.x-center[0])**2+(b.y-center[1])**2)
        direction=np.array([target.x-center[0],target.y-center[1]])
        direction=direction/(np.linalg.norm(direction)+1e-9)
    return safe_action(env,player,direction,0,0)


def train(episodes=5,max_steps=40,out='runs/smoke',seed=42,epochs=3,players_per_team=2,
          opponent_checkpoint=None, preset_rules=None, rollout_steps=None):
    if episodes < 1 or max_steps < 1 or epochs < 1 or seed < 0 or players_per_team < 1:
        raise ValueError('episodes, max_steps and epochs must be >=1 and seed nonnegative')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    cfg={'max_policy_steps':int(max_steps),'players_per_team':int(players_per_team)}
    if rollout_steps is not None and (type(rollout_steps) is not int or rollout_steps < 1):
        raise ValueError('rollout_steps must be positive integer')
    env=BallMARLEnv(rules=preset_rules if preset_rules is not None else rules_from_yaml(**cfg),seed=seed)
    obs,_=env.reset(seed=seed)
    learner=MAPPO(len(obs[env.agents[0]]['features']),len(env.state(0)),gamma=env.rules['discount_gamma'],epochs=epochs)
    frozen_opponent=None
    if opponent_checkpoint is not None:
        frozen_opponent, opponent_meta=MAPPO.load(opponent_checkpoint)
        if opponent_meta.get('rules_hash') != rules_digest(env.rules):
            raise ValueError('frozen opponent ruleset differs from training ruleset')
        if frozen_opponent.actor.body[0].in_features != len(obs[env.agents[0]]['features']):
            raise ValueError('frozen opponent observation dimension mismatch')
        frozen_opponent.actor.eval()
    output=Path(out)
    output.mkdir(parents=True,exist_ok=True)
    if any((output / x).exists() for x in ('policy.pt','policy.pt.sha256','train.jsonl','manifest.json')):
        raise FileExistsError(f'run directory contains existing artifacts: {output}; choose a new --out')
    summary=[]
    for ep in range(episodes):
        obs,_=env.reset(seed=seed+ep)
        rng=np.random.default_rng(np.random.SeedSequence([seed,ep,999]))
        traces={p:[] for p in env.possible_agents if p.startswith('team_0/')}
        terminal_info=None
        while env.agents and (rollout_steps is None or env.policy_steps < rollout_steps):
            global_state=env.state(0)
            with torch.no_grad():
                current_value=float(learner.critic(torch.tensor(global_state,dtype=torch.float32).unsqueeze(0)).item())
            joint={}
            logps={}
            for p in env.agents:
                if p.startswith('team_0/'):
                    joint[p],logps[p]=learner.choose(obs[p])
                elif frozen_opponent is not None:
                    joint[p],_=frozen_opponent.choose(obs[p],deterministic=True)
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
    metadata={'seed':seed,'episodes':episodes,'rules':env.rules,
              'rules_hash':rules_digest(env.rules),
              'observation_dim':len(obs[env.possible_agents[0]]['features']),
              'critic_state_dim':len(env.state(0)),
              'training_algorithm':'shared_actor_central_critic_ppo',
              'engine_collision_backend':env.world.collision_backend,
              'torch_version':str(torch.__version__), 'numpy_version':np.__version__,
              'mode':'frozen_policy' if frozen_opponent is not None else 'scripted_opponent',
              'rollout_steps':rollout_steps, 'preset_rules':preset_rules is not None,
              'opponent_checkpoint_sha256':sha256_file(opponent_checkpoint) if frozen_opponent is not None else None,
              'note':'Proof-of-pipeline only, not verified commercial mechanics.'}
    learner.save(output/'policy.pt',metadata)
    metadata['checkpoint_sha256']=sha256_file(output/'policy.pt')
    atomic_write_json(output/'manifest.json',metadata)
    return summary,learner
