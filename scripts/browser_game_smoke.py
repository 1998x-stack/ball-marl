"""Optional browser rendering smoke without network permissions.

Uses frames from the real World but substitutes a WebSocket transport because
some sandboxes block Chromium navigation. NOT an end-to-end network E2E test.
Install playwright + chromium separately before running.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

from game_server.session import GameSession

ROOT=Path(__file__).resolve().parents[1]


def run(outdir: Path, browser_bin: str = '/usr/bin/chromium'):
    from playwright.sync_api import sync_playwright
    outdir.mkdir(parents=True, exist_ok=True)
    static=ROOT/'game_server'/'static'
    page_html=(static/'index.html').read_text(encoding='utf-8')
    page_html=re.sub(r'<link rel="stylesheet" href="/assets/styles.css">','',page_html)
    page_html=re.sub(r'<script src="/assets/app.js" defer></script>','',page_html)
    session=GameSession('squad', seed=123)
    frames=[session.frame()]
    for _ in range(15):frames.append(session.step())
    fake='''
      const CANNED_FRAMES = '''+json.dumps(frames,ensure_ascii=False)+''';
      class MockWebSocket {
        static OPEN=1;
        constructor() {
          this.readyState=0;this.current=0;this.paused=false;
          setTimeout(()=>{this.readyState=1;this.onopen?.();this.deliver();
            this.timer=setInterval(()=>{if(!this.paused){this.current=Math.min(this.current+1,CANNED_FRAMES.length-1);this.deliver();}},200);
          },20);
        }
        deliver(){const frame={...CANNED_FRAMES[this.current],paused:this.paused};this.onmessage?.({data:JSON.stringify(frame)});}
        send(text){const m=JSON.parse(text);
          if(m.type==='pause'){this.paused=!this.paused;this.deliver();}
          if(m.type==='restart'){this.current=0;this.paused=false;this.deliver();}
        }
        close(){clearInterval(this.timer);this.readyState=3;this.onclose?.();}
      }
      window.WebSocket=MockWebSocket;
    '''
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True,executable_path=browser_bin,args=['--no-sandbox'])
        errors=[]
        desktop=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
        desktop.on('pageerror',lambda e:errors.append(str(e)))
        desktop.set_content(page_html,wait_until='load')
        desktop.add_style_tag(content=(static/'styles.css').read_text(encoding='utf-8'))
        desktop.add_script_tag(content=fake)
        desktop.add_script_tag(content=(static/'app.js').read_text(encoding='utf-8'))
        desktop.screenshot(path=str(outdir/'desktop-start.png'),full_page=True)
        desktop.locator('#playButton').click()
        desktop.wait_for_function("document.getElementById('startOverlay').classList.contains('hidden')",timeout=10000)
        desktop.wait_for_timeout(600)
        desktop.screenshot(path=str(outdir/'desktop-playing.png'),full_page=True)
        assert desktop.locator('#blueScore').text_content() != '—'
        desktop.keyboard.press('p')
        assert desktop.locator('#pauseText').text_content()=='继续'
        desktop.keyboard.press('p')
        assert desktop.locator('#pauseText').text_content()=='暂停'
        desktop.keyboard.press('r')
        assert desktop.locator('#frameTicker').text_content()=='TICK 0000'
        mobile=browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1)
        mobile.on('pageerror',lambda e:errors.append(str(e)))
        mobile.set_content(page_html,wait_until='load')
        mobile.add_style_tag(content=(static/'styles.css').read_text(encoding='utf-8'))
        mobile.add_script_tag(content=fake)
        mobile.add_script_tag(content=(static/'app.js').read_text(encoding='utf-8'))
        mobile.locator('#playButton').click()
        mobile.wait_for_function("document.getElementById('startOverlay').classList.contains('hidden')")
        mobile.wait_for_timeout(400)
        mobile.screenshot(path=str(outdir/'mobile-playing.png'),full_page=True)
        assert mobile.locator('#joystick').is_visible()
        browser.close()
        if errors:raise AssertionError('browser JavaScript exceptions: '+repr(errors))
        print('Browser smoke PASS (mocked transport, real engine frames); screenshots:',outdir)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('runs/screenshots'))
    parser.add_argument('--browser-bin',default='/usr/bin/chromium')
    args=parser.parse_args()
    run(args.out,args.browser_bin)
