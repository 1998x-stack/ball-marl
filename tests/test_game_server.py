"""Web game integration, security, browser contract and authoritative engine tests."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
pytest.importorskip('loguru')
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from game_server.app import app, _control_message, _valid_origin
from game_server.session import GameSession, InputError, MODES, validate_input, ControlState


def test_game_modes_have_correct_roster():
    for mode, count in MODES.items():
        session = GameSession(mode, 42)
        assert session.env.world.rules['players_per_team'] == count
        assert len(session.env.world.players) == 2 * count
        assert session.frame()['bots'] == 2 * count - 1
        assert session.frame()['session_id'] == session.session_id
        assert session.frame()['score']['red_mass'] is None  # prevent oracle score leakage


def test_input_contract_rejects_untrusted_numeric_and_extra_fields():
    good = {'type': 'input', 'move': [0.0, 1.0], 'split': False, 'eject': True}
    value, split, eject = validate_input(good)
    assert value.tolist() == [0.0, 1.0] and not split and eject
    bad = [
        {**good, 'move': [float('nan'), 0]},
        {**good, 'move': [float('inf'), 0]},
        {**good, 'move': [True, 0]},
        {**good, 'move': [1.1, 0]},
        {**good, 'move': '0,1'},
        {**good, 'split': 1},
        {**good, 'eject': 0},
        {**good, 'hidden': True},
        {'type': 'input', 'move': [0, 0]},
    ]
    for item in bad:
        with pytest.raises(InputError):
            validate_input(item)


def test_button_press_has_exactly_one_rising_edge_and_is_not_lost():
    controls = ControlState.empty()
    controls.update(np.array([.4, -.8], np.float32), True, False)
    controls.update(np.array([0., 1.], np.float32), False, False)
    one = controls.take()
    two = controls.take()
    assert one['split'] == 1 and two['split'] == 0
    assert one['move'].tolist() == [0., 1.]
    controls.update(np.array([0., 0.], np.float32), True, True)
    assert controls.take()['split'] == 1
    assert controls.take()['eject'] == 0
    controls.update(np.array([0., 0.], np.float32), True, True)
    assert controls.take()['split'] == 0  # held button no repeated split


def test_live_engine_step_mass_and_visibility():
    session = GameSession('duel', 11)
    w = session.env.world
    removed = sum(b.mass for b in w.balls if b.owner == 'pellet')
    w.balls = [b for b in w.balls if b.owner != 'pellet']
    w.external_out += removed
    far = w.add('pellet', 90, 90, 1, external=True)
    near = w.add('pellet', 26, 52, 1, external=True)
    first = session.frame()
    ids = {b['id'] for b in first['visible_balls']}
    assert near.id in ids and far.id not in ids
    session.set_input({'type': 'input', 'move': [0.1, 0.4], 'split': False, 'eject': False})
    frame = session.step()
    assert frame['tick'] == 1
    assert w.total_mass() == pytest.approx(w.external_in-w.external_out)
    assert frame['player']['x'] != first['player']['x'] or frame['player']['y'] != first['player']['y']
    assert frame['score']['red_mass'] is None


def test_paused_and_same_seed_restart_reproducibility():
    session = GameSession('squad', 123)
    pristine = session.env.world.digest()
    session.step()
    session.paused = True
    t = session.env.policy_steps
    session.step()
    assert session.env.policy_steps == t
    session.restart()
    assert session.seed == 123 and session.env.world.digest() == pristine
    assert session.env.policy_steps == 0 and not session.paused
    session2 = GameSession('squad', 123)
    for _ in range(8):
        result_a, result_b = session.step(), session2.step()
        result_a.pop('session_id'); result_b.pop('session_id')
        assert result_a == result_b
    assert session.env.world.digest() == session2.env.world.digest()


def test_game_over_winner_and_no_oracle_during_play():
    session = GameSession('duel', 7)
    session.env.rules['max_policy_steps'] = 2
    session.rules['max_policy_steps'] = 2
    # Engine and env use the same mutable rules mapping: run genuine terminal.
    for _ in range(2):
        end = session.step()
    assert session.finished and end['outcome']
    assert end['score']['red_mass'] is not None
    assert end['outcome']['winner'] in ('blue', 'red', 'draw')
    assert session.step() == end


def test_inactive_human_observes_last_position_not_top_left():
    session = GameSession('duel', 42)
    center = session.env.world.centroid(session.human_id)
    human = session.env.world.owned(session.human_id)[0]
    session.env.world.balls.remove(human)
    frame = session.frame()
    assert not frame['player']['alive']
    assert frame['player']['x'] == pytest.approx(center[0])
    assert frame['player']['y'] == pytest.approx(center[1])


def test_cannot_mutate_match_rules_with_controls():
    session = GameSession('duel', 1)
    for payload in ({'type': 'restart', 'new_seed': 'false'},
                    {'type': 'restart', 'mode': 'arena'},
                    {'type': 'pause', 'extra': 1},
                    {'type': 'ping', 'client_time': float('nan')},
                    {'type': 'script'}, []):
        with pytest.raises(InputError):
            _control_message(session, payload)
    assert _control_message(session, {'type': 'pause'})['paused']
    assert _control_message(session, {'type': 'pause'})['paused'] is False
    assert _control_message(session, {'type': 'ping', 'client_time': 37.5})['client_time'] == 37.5


def test_http_assets_health_and_entrypage(tmp_path, monkeypatch):
    monkeypatch.setenv('BALL_MARL_LOG_DIR', str(tmp_path))
    with TestClient(app) as client:
        root = client.get('/')
        assert root.status_code == 200 and 'ORBIT' in root.text and 'arenaCanvas' in root.text
        assert client.get('/assets/app.js').status_code == 200
        assert client.get('/assets/styles.css').status_code == 200
        assert 'no-store' == root.headers['cache-control']
        assert client.get('/api/health').json()['status'] == 'ok'
        assert client.get('/api/modes').json()['policy_hz'] == 5
        assert len(client.get('/api/modes').json()['modes']) == 3
    # Loguru writes structured JSONL to a real file with rotation-ready sink.
    log_files = list(tmp_path.glob('game-*.jsonl'))
    assert log_files
    assert any('game_server_started' in line for p in log_files for line in p.read_text().splitlines())
    assert all('"record"' in line for p in log_files for line in p.read_text().splitlines())


def test_websocket_connect_error_pause_reset_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('BALL_MARL_LOG_DIR', str(tmp_path))
    with TestClient(app) as client:
        count = client.get('/api/health').json()['active_sessions']
        with client.websocket_connect('/ws/play?mode=duel&seed=21') as ws:
            first = ws.receive_json()
            assert first['tick'] == 0 and first['seed'] == 21
            assert client.get('/api/health').json()['active_sessions'] == count+1
            ws.send_json({'type': 'input', 'move':[float('nan'), 0.0], 'split':False, 'eject':False})
            err = ws.receive_json()
            assert err['type']=='error' and 'nonfinite' in err['message']
            ws.send_json({'type': 'pause'})
            paused = ws.receive_json()
            assert paused['paused']
            ws.send_json({'type': 'restart', 'new_seed':False})
            restarted = ws.receive_json()
            assert restarted['tick'] == 0 and restarted['seed'] == 21
            assert not restarted['paused']
        # disconnected room is removed; cleanup may be scheduled very briefly.
        for _ in range(20):
            if client.get('/api/health').json()['active_sessions'] == count:
                break
            time.sleep(.01)
        assert client.get('/api/health').json()['active_sessions'] == count


def test_websocket_bad_mode_and_cross_site_origin(tmp_path, monkeypatch):
    monkeypatch.setenv('BALL_MARL_LOG_DIR', str(tmp_path))
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect('/ws/play?mode=secret'):
                pass
        assert e.value.code == 1008
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect('/ws/play?mode=duel', headers={'origin':'https://untrusted.test'}):
                pass
        assert e.value.code == 1008
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/ws/play?mode=duel&seed=-20'):
                pass


def test_game_mode_stress_snapshot_accounting():
    for mode in ('duel','squad','arena'):
        session=GameSession(mode, 404)
        for tick in range(15):
            session.set_input({'type':'input','move':[.2, .5],
                               'split':tick%7==0,'eject':tick%5==0})
            result=session.step()
            assert all(math.isfinite(b['mass']) and b['mass']>0 for b in result['visible_balls'])
            assert session.env.world.total_mass() == pytest.approx(
                session.env.world.external_in-session.env.world.external_out, rel=1e-7)
            assert result['tick']==tick+1
