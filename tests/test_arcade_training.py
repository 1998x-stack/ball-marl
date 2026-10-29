from __future__ import annotations
from pathlib import Path
from game_server.policy import PolicyRegistry
from game_server.session import GameSession
from game_server.train_bot import main


def test_arcade_bot_train_and_load_ruleset(tmp_path):
    out=main(['--mode','duel','--id','trained','--episodes','1','--rollout-steps','5',
              '--epochs','1','--seed','77','--outdir',str(tmp_path)])
    assert (out/'policy.pt.sha256').exists()
    session=GameSession('duel', 77)
    m=PolicyRegistry(tmp_path).load('trained',session.rules,
                                      len(session.env.observe(session.human_id)['features']),
                                      len(session.env.state(0)))
    action,_=m.choose(session.env.observe(session.human_id),deterministic=True)
    assert session.env.action_space(session.human_id).contains(action)
