"""Simulator-fork causal *conditional* difference; not a PPO advantage."""
import numpy as np
from engine.world import World


def counterfactual_mass_delta(world,actions,player,baseline=None,horizon=1):
    if horizon < 1: raise ValueError('horizon must be >=1')
    baseline=baseline or {'move':np.zeros(2,dtype=np.float32),'split':0,'eject':0}
    team=int(player.split('/')[0].split('_')[1])
    original=World.from_snapshot(world.snapshot())
    fork=World.from_snapshot(world.snapshot())
    altered={p:{'move':np.asarray(a['move']).copy(),'split':a['split'],'eject':a['eject']}
             for p,a in actions.items()}
    altered[player]=baseline
    old_mass=world.team_mass(team)
    for _ in range(horizon):
        original.step(actions)
        fork.step(altered)
    original_gain=original.team_mass(team)-old_mass
    fork_gain=fork.team_mass(team)-old_mass
    return {'actual_gain':original_gain,'counterfactual_gain':fork_gain,
            'difference':original_gain-fork_gain,'horizon':horizon,
            'continuation':'open_loop_fixed_actions', 'baseline':str(baseline)}
