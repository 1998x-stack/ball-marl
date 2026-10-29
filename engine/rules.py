"""Schema validation and canonical hashing for the research (non-official) ruleset."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import yaml

DEFAULT_RULES = Path(__file__).resolve().parents[1] / 'specifications' / 'game_rules.yaml'
POSITIVE = {'width','height','initial_mass','pellet_mass','split_min_mass','eject_mass',
            'eject_min_mass','speed_scale','eat_ratio','physics_dt','view_radius','discount_gamma'}
NONNEGATIVE = {'split_impulse','speed_min','speed_exponent','penetration','shaping_beta'}
INTEGERS = {'teams','players_per_team','pellet_count','max_fragments',
            'merge_delay_ticks','substeps','max_policy_steps'}

def validate_rules(rules: Mapping[str, Any]) -> dict:
    if not isinstance(rules, Mapping):
        raise ValueError('rules must be a mapping')
    with DEFAULT_RULES.open(encoding='utf-8') as f:
        expected = yaml.safe_load(f)
    missing = sorted(set(expected) - set(rules))
    unknown = sorted(set(rules) - set(expected))
    if missing or unknown:
        raise ValueError(f'rules schema mismatch: missing={missing}, unknown={unknown}')
    cfg = dict(rules)
    if not isinstance(cfg['version'], str) or not cfg['version']:
        raise ValueError('rules version must be nonempty')
    if cfg['status'] != 'PROPOSED_NOT_OFFICIAL':
        raise ValueError('rules must explicitly declare PROPOSED_NOT_OFFICIAL status')
    for key in INTEGERS:
        x = cfg[key]
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError(f'{key} must be an integer')
        if x < (0 if key in {'pellet_count','merge_delay_ticks'} else 1):
            raise ValueError(f'{key} outside valid range')
    if cfg['teams'] != 2:
        raise ValueError('research-v0 supports exactly two teams')
    if cfg['players_per_team'] > 32 or cfg['max_fragments'] > 128 or cfg['pellet_count'] > 100000:
        raise ValueError('rules exceed supported reference-backend resource limits')
    for key in POSITIVE | NONNEGATIVE | {'eject_efficiency'}:
        x = cfg[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(f'{key} must be a finite numeric value')
        if key in POSITIVE and x <= 0:
            raise ValueError(f'{key} must be > 0')
        if key in NONNEGATIVE and x < 0:
            raise ValueError(f'{key} must be >= 0')
    if not (0 < cfg['eject_efficiency'] <= 1):
        raise ValueError('eject_efficiency must be in (0, 1]')
    if not (0 < cfg['discount_gamma'] <= 1):
        raise ValueError('discount_gamma must be in (0, 1]')
    if cfg['eat_ratio'] <= 1:
        raise ValueError('eat_ratio must be > 1 to prevent symmetric consumption')
    if cfg['eject_min_mass'] + cfg['eject_mass'] >= cfg['initial_mass'] * 1000:
        raise ValueError('ejection thresholds outside supported scale')
    if cfg['width'] <= 4 or cfg['height'] <= 4:
        raise ValueError('arena too small to hold spawned pellets')
    return cfg

def rules_from_yaml(path=None, **overrides):
    with Path(path or DEFAULT_RULES).open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    return validate_rules(cfg)

def rules_digest(rules: Mapping[str, Any]) -> str:
    valid = validate_rules(rules)
    canonical = json.dumps(valid, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
