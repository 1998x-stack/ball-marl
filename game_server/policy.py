"""Safely configured local policy inference. No user-controlled filesystem path."""
from __future__ import annotations

from pathlib import Path
import re

from algorithms.mappo import MAPPO
from engine.rules import rules_digest

POLICY_ID = re.compile(r'^[a-zA-Z0-9_-]{1,48}$')


class PolicyRegistry:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()

    def list_ids(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(p.name for p in self.directory.iterdir()
                      if p.is_dir() and POLICY_ID.fullmatch(p.name)
                      and (p / 'policy.pt').is_file() and (p / 'policy.pt.sha256').is_file())[:128]

    def load(self, policy_id: str, rules: dict, obs_dim: int, state_dim: int) -> MAPPO:
        if not POLICY_ID.fullmatch(policy_id):
            raise ValueError('invalid policy ID')
        checkpoint = (self.directory / policy_id / 'policy.pt').resolve()
        if not checkpoint.is_relative_to(self.directory) or not checkpoint.is_file():
            raise ValueError('policy not found in configured registry')
        model, metadata = MAPPO.load(checkpoint)
        if metadata.get('rules_hash') != rules_digest(rules):
            raise ValueError('checkpoint ruleset hash mismatch')
        if model.actor.body[0].in_features != obs_dim or model.critic.model[0].in_features != state_dim:
            raise ValueError('checkpoint observation/critic dimensions mismatch')
        model.actor.eval()
        model.critic.eval()
        return model
