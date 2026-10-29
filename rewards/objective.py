"""Team ranking reward and policy-invariant potential shaping (with terminal zero)."""
import math

def potential(mass):
    return math.log1p(mass)

def reward_components(pre_mass, post_mass, gamma, beta, terminated):
    objective = {0:0., 1:0.}
    if terminated:
        a,b=post_mass[0],post_mass[1]
        objective = {0:float(a>b)-float(a<b),1:float(b>a)-float(b<a)}
    shaping = {team:beta*(gamma*(0. if terminated else potential(post_mass[team]))-potential(pre_mass[team]))
               for team in (0,1)}
    return {team:{'objective':objective[team], 'shaping':shaping[team],
                  'total':objective[team]+shaping[team]} for team in (0,1)}
