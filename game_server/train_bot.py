"""Train a ruleset-compatible checkpoint for local Arcade bot inference.

Example:
 python -m scripts.train_arcade_bot --mode duel --id my-duel-v1 --episodes 3 --rollout-steps 40
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from game_server.session import GameSession, MODES
from training.learner import train


def main(argv: list[str] | None = None) -> Path:
    parser=argparse.ArgumentParser(description='Train a checkpoint matching the live Arcade ruleset')
    parser.add_argument('--mode', choices=MODES, default='duel')
    parser.add_argument('--id', default='local-duel-v1')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--rollout-steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--outdir', type=Path, default=Path('runs/web/checkpoints'))
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}',args.id):
        parser.error('--id must contain only A-Z/a-z/0-9/_/- (up to 48 chars)')
    if args.episodes < 1 or args.rollout_steps < 1 or args.seed < 0:
        parser.error('episodes/rollout steps must be positive and seed nonnegative')
    # Reuse exactly the engine rules from the player-facing game, not a parallel config.
    game=GameSession(args.mode, seed=args.seed)
    output=args.outdir / args.id
    summary,_=train(episodes=args.episodes, max_steps=game.rules['max_policy_steps'],
                    out=output, seed=args.seed, epochs=args.epochs,
                    players_per_team=MODES[args.mode],preset_rules=game.rules,
                    rollout_steps=args.rollout_steps)
    print(f'Arcade checkpoint ready: {output / "policy.pt"} ({len(summary)} training rollouts)')
    print('Research baseline only: improve training/league and evaluate independently before ranking.')
    return output


if __name__=='__main__':
    main()
