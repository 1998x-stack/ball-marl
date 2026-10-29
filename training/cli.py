"""CLI: train -> infer -> evaluate -> counterfactual (research mode)."""
import argparse
import json
import numpy as np
from algorithms.mappo import MAPPO
from environments.pettingzoo_env import BallMARLEnv
from training.learner import train,scripted_action
from evaluation.tournament import evaluate
from credit_assignment.difference_rewards import counterfactual_mass_delta
from engine.rules import rules_digest
from evaluation.benchmark import run_benchmark


def main(argv=None):
    parser=argparse.ArgumentParser(prog='ball-marl')
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--episodes',type=int,default=5)
    t.add_argument('--steps',type=int,default=40);t.add_argument('--seed',type=int,default=42)
    t.add_argument('--out',default='runs/smoke')
    t.add_argument('--players-per-team',type=int,default=2)
    e=sub.add_parser('evaluate');e.add_argument('--checkpoint');e.add_argument('--episodes',type=int,default=5)
    e.add_argument('--seed',type=int,default=1000);e.add_argument('--steps',type=int)
    e.add_argument('--trace-dir')
    e.add_argument('--side-swaps',action='store_true')
    e.add_argument('--allow-legacy-checkpoint',action='store_true')
    inf=sub.add_parser('infer');inf.add_argument('--checkpoint',required=True);inf.add_argument('--seed',type=int,default=100)
    inf.add_argument('--allow-legacy-checkpoint',action='store_true')
    cf=sub.add_parser('credit');cf.add_argument('--seed',type=int,default=100)
    b=sub.add_parser('benchmark');b.add_argument('--episodes',type=int,default=2)
    b.add_argument('--steps',type=int,default=20);b.add_argument('--players-per-team',type=int,default=5)
    m=sub.add_parser('migrate-checkpoint');m.add_argument('--source',required=True);m.add_argument('--target',required=True)
    args=parser.parse_args(argv)
    if args.command=='train':
        report,_=train(episodes=args.episodes,max_steps=args.steps,out=args.out,seed=args.seed,
                       players_per_team=args.players_per_team)
        print(json.dumps(report[-1],indent=2))
    elif args.command=='evaluate':
        print(json.dumps(evaluate(args.checkpoint,episodes=args.episodes,max_steps=args.steps,
                                  seed=args.seed,trace_dir=args.trace_dir,side_swaps=args.side_swaps,
                                  allow_legacy_checkpoint=args.allow_legacy_checkpoint),indent=2))
    elif args.command=='infer':
        model,meta=MAPPO.load(args.checkpoint,allow_legacy=args.allow_legacy_checkpoint)
        env=BallMARLEnv(rules=meta['rules']);obs,_=env.reset(seed=args.seed)
        p=env.agents[0]
        action,logp=model.choose(obs[p],deterministic=True)
        print(json.dumps({'agent':p,'move':action['move'].tolist(),'split':action['split'],
                          'eject':action['eject'],'logp':logp},indent=2))
    elif args.command=='benchmark':
        print(json.dumps(run_benchmark(episodes=args.episodes,steps=args.steps,
                                      players_per_team=args.players_per_team),indent=2))
    elif args.command=='migrate-checkpoint':
        from pathlib import Path
        if Path(args.source).resolve() == Path(args.target).resolve():
            raise ValueError('migration target must not overwrite source')
        if Path(args.target).exists():
            raise FileExistsError('migration target already exists')
        model,meta=MAPPO.load(args.source,allow_legacy=True)
        if 'rules' in meta:
            meta['rules_hash']=rules_digest(meta['rules'])
        model.save(args.target,meta)
        print(json.dumps({'target':args.target,'schema':2,'source':args.source},indent=2))
    elif args.command=='credit':
        env=BallMARLEnv();obs,_=env.reset(seed=args.seed)
        actions={p:scripted_action(env,p) for p in env.agents}
        p=env.agents[0]
        print(json.dumps(counterfactual_mass_delta(env.world,actions,p,horizon=2),indent=2))

if __name__ == '__main__': main()
