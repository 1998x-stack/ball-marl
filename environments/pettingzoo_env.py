"""Simultaneous-action environment; optional PettingZoo/Gymnasium dependencies."""
from __future__ import annotations
import numpy as np
from engine.world import World, rules_from_yaml
from environments.compat import ParallelEnv, spaces
from observations.entities import OBS_DIM, actor_features, critic_features
from rewards.objective import reward_components


class BallMARLEnv(ParallelEnv):
    metadata = {'name':'ball_marl_research_v0', 'render_modes': ['rgb_array']}
    def __init__(self, rules=None, seed=0, render_mode=None, strict_action_masks=False,
                 collision_backend='grid'):
        from engine.rules import validate_rules
        self.rules=validate_rules(rules if rules is not None else rules_from_yaml())
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('render_mode must be None or rgb_array')
        self.render_mode=render_mode
        self.strict_action_masks=bool(strict_action_masks)
        self.world=World(self.rules,collision_backend=collision_backend)
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
        # PettingZoo allows advisory reset options. They are inert metadata,
        # not implicit overrides of physics or gameplay rules.
        if options is None:
            options={}
        if not isinstance(options,dict):
            raise ValueError('reset options must be a dict or None')
        unsupported=set(options) & {'ruleset','scenario_id','initial_world'}
        if unsupported:
            raise NotImplementedError(f'gameplay reset overrides not implemented: {sorted(unsupported)}')
        ignored_options=sorted(options)
        self.world.reset(seed if seed is not None else self.world.seed)
        self.agents=self.possible_agents.copy()
        self.policy_steps=0
        self.last_info={}
        obs={p:self.observe(p) for p in self.agents}
        infos={p:{'state_hash':self.world.digest(), 'ruleset':self.rules['version'],
                  'ignored_reset_options':ignored_options} for p in self.agents}
        return obs,infos

    def step(self,actions):
        if not self.agents:
            raise RuntimeError('step called after episode ended; reset required')
        if set(actions) != set(self.agents):
            raise ValueError('joint actions must match current agents')
        # Valid Gymnasium-space actions can still be game-illegal button presses.
        # Mask them to audited no-ops by default, strict mode raises.
        executed = {}
        invalid = {}
        for p in self.agents:
            a = actions[p]
            if not self.action_space(p).contains(a):
                raise ValueError(f'invalid action: {p}')
            masks = self.world.action_masks(p)
            rejected = []
            split = int(a['split'])
            eject = int(a['eject'])
            if split and not masks['split'][1]:
                rejected.append('split')
                split = 0
            if eject and not masks['eject'][1]:
                rejected.append('eject')
                eject = 0
            if rejected and self.strict_action_masks:
                raise ValueError(f'illegal actions for {p}: {rejected}')
            invalid[p] = rejected
            executed[p] = {'move': np.asarray(a['move'], dtype=np.float32).copy(),
                           'split': split, 'eject': eject}
        pre={t:self.world.team_mass(t) for t in (0,1)}
        event_count=len(self.world.events)
        self.world.step(executed)
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
                  'state_hash':self.world.digest(),
                  'invalid_actions':invalid[p],
                  'executed_action':executed[p]} for p in current}
        if terminated:
            self.agents=[]
        self.last_info=infos
        return obs,rewards,terminations,truncations,infos

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        height, width = 256, 256
        img = np.full((height, width, 3), 242, dtype=np.uint8)
        for b in sorted(self.world.balls, key=lambda x: x.id):
            if b.owner == 'pellet':
                color = (130, 170, 110)
            elif b.owner.startswith('team_0/'):
                color = (55, 110, 218)
            else:
                color = (224, 84, 77)
            px = int(np.clip(b.x / self.rules['width'] * (width-1), 0, width-1))
            py = int(np.clip(b.y / self.rules['height'] * (height-1), 0, height-1))
            radius = max(1, int(b.radius / self.rules['width'] * width))
            ys, xs = np.ogrid[max(0,py-radius):min(height,py+radius+1),
                              max(0,px-radius):min(width,px+radius+1)]
            patch = img[max(0,py-radius):min(height,py+radius+1),
                        max(0,px-radius):min(width,px+radius+1)]
            patch[(xs-px)**2+(ys-py)**2 <= radius**2] = color
        return img

    def close(self):
        return None
