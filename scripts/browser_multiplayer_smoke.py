"""Optional browser UI smoke, using actual engine frames and mocked browser transport.

Chromium HTTP access is blocked in some CI/sandbox environments. Live network
WebSocket and HTTP routes are independently covered by TestClient and network
integration checks; this script checks browser controls and drawing only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from game_server.session import GameSession


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/screenshots'))
    args=parser.parse_args(argv)
    from playwright.sync_api import sync_playwright
    static=Path(__file__).resolve().parents[1]/'game_server'/'static'
    html=(static/'index.html').read_text(encoding='utf-8')
    html=html.replace('<link rel="stylesheet" href="/assets/styles.css">','')
    html=html.replace('<script src="/assets/app.js" defer></script>','')
    session=GameSession('squad', 99)
    frames=[session.frame()]
    for _ in range(2):
        frames.append(session.step())
    session.finished=True
    final=session.frame()
    room={'room_id':'abcdef1234567890','mode':'squad','token':'test-seat-only',
          'replay_id':'abcdefabcdefabcdefabcdef', 'replay_token':'test-replay-only',
          'player':'team_0/player_0'}
    frames[-1]=final.copy()
    final.update(room_id=room['room_id'],players_online=1,is_host=True,
                 bot_type='scripted',replay_id=room['replay_id'],replay_token=room['replay_token'])
    args.out.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',
                                  args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1440,'height':900})
        errors=[]
        page.on('pageerror',lambda e: errors.append(str(e)))
        page.set_content(html)
        page.evaluate('''([room,final,frames])=>{
          window.fetch=async(url,options)=>({ok:true,status:200,json:async()=>{
            if(url.includes('/api/bots'))return {available:['scripted']};
            if(url.includes('/api/replays/'))return {verified:true,total_ticks:2,frames};
            return room;
          }});
          class MockSocket {static OPEN=1;constructor(url){this.url=url;this.readyState=1;
            setTimeout(()=>{this.onopen?.();this.onmessage?.({data:JSON.stringify(final)});},20)}
            send(raw){window.__sent=(window.__sent||[]).concat([JSON.parse(raw)]);}
            close(){this.readyState=3;}}
          window.WebSocket=MockSocket;
        }''',[room,final,frames])
        page.add_style_tag(content=(static/'styles.css').read_text(encoding='utf-8'))
        page.add_script_tag(content=(static/'app.js').read_text(encoding='utf-8'))
        page.locator('#createRoomButton').click()
        page.wait_for_timeout(120)
        assert page.locator('#connectionText').inner_text()=='本地引擎在线'
        assert page.locator('#roomLivebar').is_visible()
        assert page.locator('#roomLiveInfo').inner_text().startswith('房间 ABCDEF')
        assert page.locator('#resultOverlay').is_visible()
        page.screenshot(path=str(args.out/'room-finished.png'),full_page=True)
        page.locator('#resultReplayButton').click()
        page.wait_for_timeout(130)
        assert page.locator('#replayBar').is_visible()
        assert page.locator('#replayRange').get_attribute('max')=='2'
        assert page.locator('#replayTime').inner_text()=='0 / 2'
        page.locator('#replayRange').fill('2')
        assert page.locator('#replayTime').inner_text()=='2 / 2'
        page.screenshot(path=str(args.out/'room-replay.png'),full_page=True)
        page.locator('#replayExit').click()
        assert page.locator('#resultOverlay').is_visible()
        assert not errors,errors
        assert any(msg['type']=='auth' for msg in page.evaluate('window.__sent||[]'))
        browser.close()
    print(f'Browser room and replay smoke PASS: {args.out}')


if __name__=='__main__':main()
