"""Reproducible reference CPU benchmark; no host-independent performance claim."""
from __future__ import annotations
import argparse
import json
import statistics
import time
import tracemalloc
import numpy as np
from actions.masks import safe_action
from engine.rules import rules_from_yaml, rules_digest
from environments.pettingzoo_env import BallMARLEnv

def run_benchmark(*, players_per_team=5, episodes=4, steps=40,
                  pellet_count=32, seed=2026):
    if min(players_per_team,episodes,steps)<1 or pellet_count<0 or seed<0:
        raise ValueError('invalid benchmark dimensions')
    rules=rules_from_yaml(players_per_team=int(players_per_team),
                          max_policy_steps=int(steps),pellet_count=int(pellet_count))
    env=BallMARLEnv(rules)
    latencies=[]
    max_residual=0.
    total_steps=0
    event_count=0
    started=time.perf_counter()
    tracemalloc.start()
    try:
        for ep in range(episodes):
            obs,_=env.reset(seed=seed+ep)
            rng=np.random.default_rng(np.random.SeedSequence([seed,ep,77]))
            while env.agents:
                joint={}
                for p in env.agents:
                    joint[p]=safe_action(env,p,rng.uniform(-1,1,size=2).astype(np.float32),
                                         split=int(rng.random()<.12),
                                         eject=int(rng.random()<.06))
                t0=time.perf_counter()
                obs,_,_,_,infos=env.step(joint)
                latencies.append((time.perf_counter()-t0)*1000)
                total_steps+=1
                event_count+=len(infos[env.possible_agents[0]]['events'])
                residual=abs(env.world.total_mass()-env.world.external_in+env.world.external_out)
                max_residual=max(max_residual,residual)
        _,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        env.close()
    elapsed=time.perf_counter()-started
    if not total_steps:
        raise AssertionError('no benchmark steps executed')
    return {'rules_hash':rules_digest(rules),'players_total':2*players_per_team,
            'episodes':episodes,'policy_steps':total_steps,
            'physics_steps':total_steps*rules['substeps'],
            'elapsed_seconds':elapsed,'steps_per_second':total_steps/elapsed,
            'step_latency_ms_p50':statistics.median(latencies),
            'step_latency_ms_p95':float(np.quantile(latencies,.95)),
            'peak_python_tracemalloc_bytes':peak,
            'max_global_mass_residual':max_residual,
            'events_observed':event_count,'backend':'reference_cpu_all_pairs',
            'note':'Host-specific benchmark; no cross-device throughput claim.'}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--players-per-team',type=int,default=5)
    parser.add_argument('--episodes',type=int,default=4)
    parser.add_argument('--steps',type=int,default=40)
    parser.add_argument('--pellets',type=int,default=32)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args(argv)
    print(json.dumps(run_benchmark(players_per_team=args.players_per_team,
                                   episodes=args.episodes,steps=args.steps,
                                   pellet_count=args.pellets,seed=args.seed),indent=2))
if __name__=='__main__': main()
