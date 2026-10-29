"""Room lifecycle, capability isolation, replay integrity and checkpoint opponents."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from algorithms.mappo import MAPPO
from engine.rules import rules_digest
from game_server.app import app
from game_server.rooms import RoomError, RoomManager
from game_server.replay import ReplayIntegrityError


def manager(tmp_path):
    return RoomManager(replay_dir=str(tmp_path/'replays'), checkpoint_dir=str(tmp_path/'models'))


def test_room_seats_and_no_double_control(tmp_path):
    m=manager(tmp_path)
    room=m.create('squad', 18)
    assert room.seat(room.owner_token).player == 'team_0/player_0'
    a=room.join(team=1)
    b=room.join(team=0)
    assert room.seat(a).player == 'team_1/player_0'
    assert room.seat(b).player == 'team_0/player_1'
    with pytest.raises(RoomError, match='full'):
        room.join(team=0)
    room.connect(a)
    with pytest.raises(RoomError, match='already connected'):
        room.connect(a)
    with pytest.raises(PermissionError):
        room.set_paused(a)
    with pytest.raises(PermissionError):
        room.restart(a)
    room.disconnect(a)
    room.connect(a)
    assert room.seat(a).connected
    with pytest.raises(PermissionError):
        room.seat('wrong token')
    m.close()


def test_scoped_views_and_reconnect_bot_takeover(tmp_path):
    m=manager(tmp_path)
    room=m.create('duel', 33)
    red=room.join(team=1)
    room.connect(room.owner_token)
    room.connect(red)
    blue_frame=room.frame(room.owner_token)
    red_frame=room.frame(red)
    assert blue_frame['player']['id'] == 'team_0/player_0'
    assert red_frame['player']['id'] == 'team_1/player_0'
    assert blue_frame['score']['red_mass'] is None
    assert red_frame['score']['red_mass'] is None
    assert red_frame['score']['blue_mass'] == pytest.approx(room.match.env.world.team_mass(1))
    assert red_frame['score']['blue_mass'] != room.match.env.world.team_mass(0) or room.match.env.world.team_mass(0)==room.match.env.world.team_mass(1)
    room.input(red, {'type':'input','move':[0.5,0.2],'split':False,'eject':False})
    room.tick()
    room.disconnect(red)
    room.tick()  # disconnected player automatically handed to bot
    assert room.match.env.policy_steps==2
    room.connect(red)
    assert room.seat(red).controls.take()['split'] == 0
    m.close()


def test_replay_verification_replay_is_engine_authoritative(tmp_path):
    m=manager(tmp_path)
    room=m.create('duel', 123)
    room.connect(room.owner_token)
    for _ in range(8):
        room.tick()
    record = room.recorder
    record.close('test_finalized')
    store = m.replays
    store.authorize(record.id, record.token)
    with pytest.raises(PermissionError):
        store.authorize(record.id, 'invalid')
    result=store.frames(record.id, 'team_0/player_0', start=0, limit=30)
    assert result['verified'] and result['total_ticks']==8 and len(result['frames'])==9
    assert result['frames'][-1]['tick']==8
    assert len(store.frames(record.id,'team_1/player_0',start=3,limit=2)['frames'])==2
    # Tampering with a signed-by-hash physics state is detected, not trusted.
    path=record.path
    rows=path.read_text().splitlines()
    item=json.loads(rows[2]);item['hash']='0'*64
    rows[2]=json.dumps(item)
    path.write_text('\n'.join(rows)+'\n')
    with pytest.raises(ReplayIntegrityError, match='hash mismatch'):
        store.frames(record.id, 'team_0/player_0')
    m.close()


def test_policy_loader_compatibility_and_rejects_path_traversal(tmp_path):
    m=manager(tmp_path)
    baseline=m.create('duel', 1)
    env=baseline.match.env
    dims=len(env.observe(env.possible_agents[0])['features']),len(env.state(0))
    policy=MAPPO(*dims)
    path=tmp_path/'models'/'compatible'/'policy.pt'
    policy.save(path,metadata={'rules_hash':rules_digest(baseline.match.rules)})
    assert 'compatible' in m.registry.list_ids()
    model=m.registry.load('compatible',baseline.match.rules,*dims)
    assert not model.actor.training
    with pytest.raises(ValueError):
        m.registry.load('../../etc/passwd',baseline.match.rules,*dims)
    with pytest.raises(ValueError,match='ruleset'):
        m.registry.load('compatible',{**baseline.match.rules,'width':200},*dims)
    checkpoint_room=m.create('duel',3,bot_id='compatible')
    checkpoint_room.connect(checkpoint_room.owner_token)
    checkpoint_room.tick()
    assert checkpoint_room.match.env.policy_steps==1
    m.close()


def test_http_room_create_join_and_ws_security(tmp_path,monkeypatch):
    monkeypatch.setenv('BALL_MARL_REPLAY_DIR',str(tmp_path/'r'))
    monkeypatch.setenv('BALL_MARL_CHECKPOINT_DIR',str(tmp_path/'p'))
    monkeypatch.setenv('BALL_MARL_LOG_DIR',str(tmp_path/'log'))
    with TestClient(app) as client:
        assert 'scripted' in client.get('/api/bots').json()['available']
        assert client.post('/api/rooms',json={'mode':'unknown'}).status_code==422
        assert client.post('/api/rooms',json={'mode':'duel','seed':True}).status_code==422
        response=client.post('/api/rooms',json={'mode':'duel','seed':123})
        assert response.status_code==201
        created=response.json()
        room_id=created['room_id']
        assert client.get('/api/rooms/'+room_id).status_code==200
        assert client.post('/api/rooms/'+room_id+'/join',json={'team':0}).status_code==409
        second=client.post('/api/rooms/'+room_id+'/join',json={'team':1}).json()
        assert second['player']=='team_1/player_0'
        assert client.post('/api/rooms/'+room_id+'/join',json={}).status_code==409
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/ws/rooms/'+room_id) as unknown:
                unknown.send_json({'type':'auth','token':'wrong'})
                unknown.receive_json()
        with client.websocket_connect('/ws/rooms/'+room_id) as ws:
            ws.send_json({'type':'auth','token':created['token']})
            frame=ws.receive_json()
            assert frame['player']['id']==created['player']
            assert frame['replay_id']==created['replay_id']
            ws.send_json({'type':'input','move':[0,0],'split':False,'eject':False})
            ws.send_json({'type':'pause'})
            for _ in range(12):
                result=ws.receive_json()
                if result.get('paused'):
                    break
            assert result['paused']
            with client.websocket_connect('/ws/rooms/'+room_id) as red:
                red.send_json({'type':'auth','token':second['token']})
                status=red.receive_json()
                assert status['player']['id']=='team_1/player_0'
                red.send_json({'type':'restart','new_seed':False})
                errors=[]
                for _ in range(10):
                    msg=red.receive_json()
                    if msg.get('type')=='error':
                        errors.append(msg)
                        break
                assert errors and 'owner' in errors[0]['message']
        # Journal still live; replay endpoint must reject early requests.
        replay=client.get('/api/replays/'+created['replay_id']+'/frames',
                          params={'viewer':created['player']},
                          headers={'X-Replay-Token':created['replay_token']})
        assert replay.status_code==409
        assert client.get('/api/replays/'+created['replay_id']+'/frames',
                          params={'viewer':created['player']}).status_code==404
