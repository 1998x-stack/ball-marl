"""Input masking for independent split and eject categorical heads."""
import numpy as np

def safe_action(env, player, move, split=0, eject=0):
    mask=env.world.action_masks(player)
    move=np.clip(np.asarray(move,dtype=np.float32),-1,1)
    return {'move':move, 'split':int(bool(split) and mask['split'][1]),
            'eject':int(bool(eject) and mask['eject'][1])}
