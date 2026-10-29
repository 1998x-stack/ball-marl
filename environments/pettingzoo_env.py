"""Simultaneous-action environment; optional PettingZoo/Gymnasium dependencies."""
from __future__ import annotations
import numpy as np
from engine.world import World, rules_from_yaml
from environments.compat import ParallelEnv, spaces
from observations.entities import OBS_DIM, actor_features, critic_features
from rewards.objective import reward_components


class BallMARLEnv(ParallelEnv):
    metadata = {'name':'ball_marl_research_v0', 'render_modes': []}
    def __init__(self, rules=None, seed=0):
        self.rules=dict(rules or rules_from_yaml())
        self.world=World(self.rules)
        self.possible_agents=self.world.players.copy()
        self.agents=[]
        self._obs_spaces={p:spaces.Dict({
            'features':spaces.Box(-1e6,1e6,shape=(OBS_DIM,),dtype=np.float32),
            'split_mask':spaces.MultiBinary(2), 'eject_mask':spaces.MultiBinary(2)}) for p in self.possible_agents}
        self._act_spaces={p:spaces.Dict({
            'move':spaces.Box(-1.,1.,shape=(2,),dtype=np.float32),
            'split':spaces.Discrete(2),'eject':spaces.Discrete(2)}) for p in self.possible_agents}
        self.reset(seed=seed)

    def observation_space(self,agent): return self._obs_spaces[agent]
    def action_space(self,agent): return self._act_spaces[agent]
    def state(self,team=0): return critic_features(self.world,team)

    def observe(self,p):
        masks=self.world.action_masks(p)
        return {'features':actor_features(self.world,p),
                'split_mask':masks['split'].astype(np.int8),
                'eject_mask':masks['eject'].astype(np.int8)}

    def reset(self,seed=None,options=None):
        options=options or {}
        if options:
            raise ValueError('initial overrides not yet supported in research-v0')
        self.world.reset(seed if seed is not None else self.world.seed)
        self.agents=self.possible_agents.copy()
        self.policy_steps=0
        self.last_info={}
        obs={p:self.observe(p) for p in self.agents}
        infos={p:{'state_hash':self.world.digest(), 'ruleset':self.rules['version']} for p in self.agents}
        return obs,infos

    def step(self,actions):
        if not self.agents:
            raise RuntimeError('step called after episode ended; reset required')
        if set(actions) != set(self.agents):
            raise ValueError('joint actions must match current agents')
        for p,a in actions.items():
            if not self.action_space(p).contains(a):
                raise ValueError(f'invalid action: {p}')
            masks=self.world.action_masks(p)
            if a['split'] and not masks['split'][1]: raise ValueError(f'illegal split: {p}')
            if a['eject'] and not masks['eject'][1]: raise ValueError(f'illegal eject: {p}')
        pre={t:self.world.team_mass(t) for t in (0,1)}
        event_count=len(self.world.events)
        self.world.step(actions)
        self.policy_steps+=1
        terminated=self.policy_steps>=self.rules['max_policy_steps']
        post={t:self.world.team_mass(t) for t in (0,1)}
        components=reward_components(pre,post,self.rules['discount_gamma'],self.rules['shaping_beta'],terminated)
        current=self.agents.copy()
        obs={p:self.observe(p) for p in current}
        rewards={p:components[int(p.split('/')[0].split('_')[1])]['total'] for p in current}
        terminations={p:terminated for p in current}
        truncations={p:False for p in current}
        infos={p:{'reward_components':components[int(p.split('/')[0].split('_')[1])],
                  'mass_by_team':post.copy(),'events':self.world.events[event_count:],
                  'state_hash':self.world.digest()} for p in current}
        if terminated:
            self.agents=[]
        self.last_info=infos
        return obs,rewards,terminations,truncations,infos

    def close(self): pass
