"""CLI: train -> infer -> evaluate -> counterfactual (research mode)."""
import argparse
import json
import numpy as np
from algorithms.mappo import MAPPO
from environments.pettingzoo_env import BallMARLEnv
from training.learner import train,scripted_action
from evaluation.tournament import evaluate
from credit_assignment.difference_rewards import counterfactual_mass_delta


def main(argv=None):
    parser=argparse.ArgumentParser(prog='ball-marl')
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--episodes',type=int,default=5)
    t.add_argument('--steps',type=int,default=40);t.add_argument('--seed',type=int,default=42)
    t.add_argument('--out',default='runs/smoke')
    e=sub.add_parser('evaluate');e.add_argument('--checkpoint');e.add_argument('--episodes',type=int,default=5)
    e.add_argument('--seed',type=int,default=1000);e.add_argument('--steps',type=int)
    e.add_argument('--trace-dir')
    inf=sub.add_parser('infer');inf.add_argument('--checkpoint',required=True);inf.add_argument('--seed',type=int,default=100)
    cf=sub.add_parser('credit');cf.add_argument('--seed',type=int,default=100)
    args=parser.parse_args(argv)
    if args.command=='train':
        report,_=train(episodes=args.episodes,max_steps=args.steps,out=args.out,seed=args.seed)
        print(json.dumps(report[-1],indent=2))
    elif args.command=='evaluate':
        print(json.dumps(evaluate(args.checkpoint,episodes=args.episodes,max_steps=args.steps,
                                  seed=args.seed,trace_dir=args.trace_dir),indent=2))
    elif args.command=='infer':
        model,meta=MAPPO.load(args.checkpoint)
        env=BallMARLEnv(rules=meta['rules']);obs,_=env.reset(seed=args.seed)
        p=env.agents[0]
        action,logp=model.choose(obs[p],deterministic=True)
        print(json.dumps({'agent':p,'move':action['move'].tolist(),'split':action['split'],
                          'eject':action['eject'],'logp':logp},indent=2))
    elif args.command=='credit':
        env=BallMARLEnv();obs,_=env.reset(seed=args.seed)
        actions={p:scripted_action(env,p) for p in env.agents}
        p=env.agents[0]
        print(json.dumps(counterfactual_mass_delta(env.world,actions,p,horizon=2),indent=2))

if __name__ == '__main__': main()
