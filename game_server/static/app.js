/* ORBIT.ARENA — server-authoritative client. No simulation or scoring here. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const canvas = $('arenaCanvas');
  const ctx = canvas.getContext('2d', { alpha: false });
  const state = {
    mode: 'squad', socket: null, frame: null, previous: null, receivedAt: 0,
    selected: false, keys: new Set(), mouse: {active:false,x:0,y:0},
    touch: {active:false,x:0,y:0}, sendTimer: null, sound:false, audio:null,
    cam: {x:25,y:25}, lastEvents: new Set(), toastTimeout:null,
    wasAlive:true, frameCounter:0, initializedCamera:false, room:null, replayMode:false, replayFrames:[], replayIndex:0, replayTimer:null
  };
  const MAX_DPR = 2;
  const colors = {blue:'#7aafff', red:'#ff787d', lime:'#c6f27b', bg:'#0d1920'};
  const clamp = (n,a,b) => Math.max(a, Math.min(b,n));
  const lerp = (a,b,t) => a+(b-a)*t;
  const fmt = value => Math.round(value||0).toLocaleString('en-US');
  const playerTeam = owner => {if(owner==='pellet')return -1;const own=state.frame?.player?.id?.split('/')[0]||'team_0';return owner.startsWith(own+'/')?0:1;};
  const now = () => performance.now();

  function toast(text) {
    const box = $('canvasToast');
    box.textContent = text;
    box.classList.add('visible');
    clearTimeout(state.toastTimeout);
    state.toastTimeout = setTimeout(() => box.classList.remove('visible'), 2200);
  }
  function connectionStatus(online, label) {
    $('connectionDot').classList.toggle('online', online);
    $('connectionText').textContent = label;
    $('canvasStatus').textContent = online ? 'GAME ENGINE LIVE' : 'STANDBY';
    document.querySelector('.footer-status-dot').classList.toggle('online', online);
    $('pauseButton').disabled = !online;
    $('restartButton').disabled = !online;
  }
  function playTone(freq=380, duration=.075, type='sine') {
    if (!state.sound) return;
    try {
      state.audio ??= new (window.AudioContext || window.webkitAudioContext)();
      const osc=state.audio.createOscillator(), gain=state.audio.createGain();
      osc.type=type; osc.frequency.value=freq;
      gain.gain.setValueAtTime(.035,state.audio.currentTime);
      gain.gain.exponentialRampToValueAtTime(.0001,state.audio.currentTime+duration);
      osc.connect(gain); gain.connect(state.audio.destination);
      osc.start(); osc.stop(state.audio.currentTime+duration);
    } catch { /* Audio unavailable; gameplay is unaffected. */ }
  }
  function closeSocket() {
    stopReplay();
    if (state.socket) {
      const old = state.socket;
      state.socket = null;
      old.onclose = old.onmessage = old.onerror = old.onopen = null;
      old.close(1000, 'replaced');
    }
    if (state.sendTimer !== null) {clearInterval(state.sendTimer); state.sendTimer = null;}
    state.keys.clear(); state.mouse.active=false; state.touch.active=false;
  }
  function connect(room=null) {
    state.room=room;
    if(room){try{sessionStorage.setItem('ball-marl-room-v1',JSON.stringify(room));}catch{/* local storage disabled */}}
    closeSocket();
    $('playButton').disabled = true;
    $('playButton').querySelector('span').textContent = '正在连接引擎…';
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const endpoint=room?`/ws/rooms/${room.room_id}`:
      `/ws/play?mode=${encodeURIComponent(state.mode)}`;
    const ws = new WebSocket(`${scheme}://${location.host}${endpoint}`);
    state.socket = ws;
    ws.onopen = () => {
      if(room)ws.send(JSON.stringify({type:'auth',token:room.token}));
      state.frame=state.previous=null; state.lastEvents.clear(); state.wasAlive=true;
      state.initializedCamera=false;
      $('startOverlay').classList.add('hidden'); $('resultOverlay').classList.add('hidden');
      $('playButton').disabled = false;
      $('playButton').querySelector('span').textContent = '进入竞技场';
      connectionStatus(true, '本地引擎在线');
      state.sendTimer=setInterval(sendInput, 80);
      sendInput();
      toast(room?'房间连接成功 · 分享房间码与朋友对战':'连接成功 · 吞噬资源，获得质量');
    };
    ws.onmessage = event => {
      let message;
      try {message=JSON.parse(event.data);} catch {return;}
      if (message.type === 'frame') consumeFrame(message);
      else if (message.type === 'error') toast('操作未执行：'+message.message);
      else if (message.type === 'pong') { /* reserved for network RTT */ }
    };
    ws.onerror = () => toast('网络异常，无法连接本地游戏服务');
    ws.onclose = () => {
      if (state.socket !== ws) return;
      closeSocket();
      connectionStatus(false, '连接已断开');
      $('startOverlay').classList.remove('hidden');
      $('resultOverlay').classList.add('hidden');
      $('playButton').disabled=false;
      $('playButton').querySelector('span').textContent='重新进入竞技场';
    };
  }
  function send(message) {
    if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify(message));
  }
  function movement() {
    let dx=0,dy=0;
    if (state.keys.has('w') || state.keys.has('arrowup')) dy-=1;
    if (state.keys.has('s') || state.keys.has('arrowdown')) dy+=1;
    if (state.keys.has('a') || state.keys.has('arrowleft')) dx-=1;
    if (state.keys.has('d') || state.keys.has('arrowright')) dx+=1;
    if (dx||dy) { const m=Math.hypot(dx,dy); return [dx/m,dy/m]; }
    if (state.touch.active) return [state.touch.x, state.touch.y];
    if (state.mouse.active) {
      const rect=canvas.getBoundingClientRect();
      const x=state.mouse.x-(rect.left+rect.width/2);
      const y=state.mouse.y-(rect.top+rect.height/2);
      const m=Math.hypot(x,y);
      return m>28 ? [x/m,y/m] : [0,0];
    }
    return [0,0];
  }
  function sendInput() {
    if (!state.frame || state.frame.paused || state.frame.finished || state.replayMode) return;
    send({type:'input',move:movement(),split:state.keys.has(' ') || state.keys.has('split-button'),
      eject:state.keys.has('e') || state.keys.has('eject-button')});
  }
  function pressAbility(kind) {
    if (!state.frame?.abilities?.[kind] || state.frame.finished) return;
    const key=kind==='split' ? 'split-button' : 'eject-button';
    state.keys.add(key); sendInput(); playTone(kind==='split'?570:360,.07);
    setTimeout(() => {state.keys.delete(key);sendInput();}, 110);
  }
  function updateEvents(events) {
    for (const event of events||[]) {
      const key=`${event.tick}:${event.type}:${event.actor}:${event.target}`;
      if (state.lastEvents.has(key)) continue;
      state.lastEvents.add(key);
      if (state.lastEvents.size>200) state.lastEvents=new Set(Array.from(state.lastEvents).slice(-100));
      const dict={consume:['✦','吞噬'], split:['✳','分身'], eject:['◎','吐球'], merge:['◉','合体']};
      const [glyph,name]=dict[event.type]||['·',event.type];
      const activity=$('activityFeed');
      if (activity.querySelector('.empty-activity')) activity.replaceChildren();
      const row=document.createElement('div');row.className='event-item';
      const icon=document.createElement('span');icon.className='event-symbol';icon.textContent=glyph;
      const msg=document.createElement('span');
      msg.textContent=`${event.actor === state.frame?.player?.id ? '你' : event.actor && playerTeam(event.actor)===0 ? '队友' : '对手'} · ${name}${event.mass ? ' '+fmt(event.mass) : ''}`;
      const step=document.createElement('small');step.textContent=`T${String(event.tick).padStart(3,'0')}`;
      row.append(icon,msg,step);activity.prepend(row);
      while(activity.children.length>5)activity.lastChild.remove();
      if (event.actor===state.frame?.player?.id && event.type==='consume')playTone(520,.055);
    }
  }
  function consumeFrame(frame) {
    if (state.frame && frame.tick < state.frame.tick) {
      state.previous=null; state.lastEvents.clear(); state.initializedCamera=false;
      $('activityFeed').replaceChildren();
    } else {
      state.previous=state.frame;
    }
    state.frame=frame;
    state.receivedAt=now();
    if (!state.initializedCamera) {
      state.cam.x=frame.player.x;state.cam.y=frame.player.y;state.initializedCamera=true;
    }
    $('modeLabel').textContent={duel:'1V1 DUEL',squad:'2V2 SQUAD',arena:'5V5 ARENA'}[frame.mode]||frame.mode;
    $('matchId').textContent=frame.session_id.slice(0,8).toUpperCase();
    $('frameTicker').textContent=`TICK ${String(frame.tick).padStart(4,'0')}`;
    $('coordinateText').textContent=`X: ${frame.player.x.toFixed(1)} / Y: ${frame.player.y.toFixed(1)}`;
    $('footerSeed').textContent=`SEED ${frame.seed}`;
    $('timeValue').textContent=`${String(Math.floor(frame.time_left/60)).padStart(2,'0')}:${String(Math.floor(frame.time_left%60)).padStart(2,'0')}`;
    const percent=clamp(frame.time_left/frame.duration*100,0,100);
    $('timeBar').style.width=percent+'%';$('timePercent').textContent=Math.round(percent)+'%';
    $('blueScore').textContent=fmt(frame.score.blue_mass);
    $('redScore').textContent=frame.finished?fmt(frame.score.red_mass):fmt(frame.score.red_visible_mass)+'*';
    const b=frame.score.blue_mass,r=frame.score.red_visible_mass;
    const ratio=100*b/Math.max(b+r,1);
    $('blueBar').style.width=`${ratio}%`;$('redBar').style.width=`${100-ratio}%`;
    $('blueTeamCount').textContent=`${frame.bots>0 ? `${Math.max(1,frame.bots/2+0.5)} 人小队`:'玩家'} · LIVE`;
    $('playerMass').textContent=fmt(frame.player.mass);
    $('fragmentCount').textContent=String(frame.player.fragments).padStart(2,'0');
    $('botCount').textContent=String(frame.bots);
    $('resultReplayButton').hidden=!frame.room_id;
    if(frame.room_id){
      $('roomLivebar').hidden=false;
      $('roomLiveInfo').textContent=`房间 ${frame.room_id.toUpperCase()} · ${frame.players_online} 人在线 · AI: ${frame.bot_type}`;
      $('showReplayButton').hidden=!frame.finished;
      if(state.room){state.room.replay_id=frame.replay_id;state.room.replay_token=frame.replay_token;}
    }else{$('roomLivebar').hidden=true;}
    $('pauseButton').disabled=!!frame.room_id&&!frame.is_host;
    $('restartButton').disabled=!!frame.room_id&&!frame.is_host;
    $('aliveStatus').textContent=frame.player.alive?'ALIVE':'ELIMINATED';
    $('aliveStatus').style.color=frame.player.alive?'var(--lime)':'var(--red)';
    $('splitButton').disabled=(!frame.abilities.split || frame.finished);
    $('ejectButton').disabled=(!frame.abilities.eject || frame.finished);
    $('pauseText').textContent=frame.paused?'继续':'暂停';
    $('pauseIcon').textContent=frame.paused?'▶':'Ⅱ';
    if (frame.paused)$('canvasStatus').textContent='PAUSED';
    else if (frame.finished)$('canvasStatus').textContent='MATCH COMPLETE';
    else $('canvasStatus').textContent='GAME ENGINE LIVE';
    updateEvents(frame.event_feed);
    if (!frame.player.alive && state.wasAlive)toast('你已被吞噬 · 可以继续观战或重开');
    state.wasAlive=frame.player.alive;
    if(frame.finished && !state.replayMode && !$('resultOverlay').dataset.shown) showResult(frame.outcome);
  }
  function showResult(outcome){
    if(!outcome)return;
    $('resultOverlay').dataset.shown='1';
    $('resultOverlay').classList.remove('hidden');
    $('resultIcon').textContent=outcome.winner==='blue'?'✦':outcome.winner==='red'?'◇':'◎';
    $('resultPretitle').textContent='MATCH COMPLETE';
    $('resultTitle').textContent=outcome.winner==='blue'?'蓝队胜利':outcome.winner==='red'?'挑战失败':'平局';
    $('resultDetail').textContent=outcome.winner==='blue'?'出色的质量掌控，成功赢得这场比赛。':
      outcome.winner==='red'?'再试一次。提升质量、观察敌人，寻找吞噬机会。':'双方表现相当，再开一局决定胜负。';
    $('resultBlue').textContent=fmt(outcome.blue_mass);
    $('resultRed').textContent=fmt(outcome.red_mass);
    playTone(outcome.winner==='blue'?730:260,.18);
  }

  function resizeCanvas() {
    const rect=canvas.getBoundingClientRect();
    const dpr=Math.min(window.devicePixelRatio||1,MAX_DPR);
    const width=Math.max(1,Math.round(rect.width*dpr));
    const height=Math.max(1,Math.round(rect.height*dpr));
    if(canvas.width!==width || canvas.height!==height){canvas.width=width;canvas.height=height;}
    ctx.setTransform(dpr,0,0,dpr,0,0);
    return {w:rect.width,h:rect.height};
  }
  function circle(x,y,r,fill){ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);ctx.fillStyle=fill;ctx.fill();}
  function drawBackground(w,h,t,scale,ox,oy,worldW,worldH){
    ctx.fillStyle='#0a151b';ctx.fillRect(0,0,w,h);
    const gradient=ctx.createRadialGradient(w*.5,h*.45,10,w*.5,h*.5,Math.max(w,h)*.7);
    gradient.addColorStop(0,'#192b2b');gradient.addColorStop(1,'#0b171c');
    ctx.fillStyle=gradient;ctx.fillRect(0,0,w,h);
    // Grid in physical world coordinates (not screen space): camera-safe.
    ctx.beginPath();ctx.strokeStyle='#78b79512';ctx.lineWidth=1;
    for(let x=0;x<=worldW;x+=5){const px=(x-ox)*scale+w/2;if(px>=0&&px<=w){ctx.moveTo(Math.round(px)+.5,0);ctx.lineTo(Math.round(px)+.5,h);}}
    for(let y=0;y<=worldH;y+=5){const py=(y-oy)*scale+h/2;if(py>=0&&py<=h){ctx.moveTo(0,Math.round(py)+.5);ctx.lineTo(w,Math.round(py)+.5);}}
    ctx.stroke();
    // Fine dotted pattern and precision ticks.
    for(let px=12;px<w;px+=28)for(let py=12;py<h;py+=28){ctx.fillStyle='#9dc5b815';ctx.fillRect(px,py,1,1);}
    const left=(0-ox)*scale+w/2,top=(0-oy)*scale+h/2;
    const width=worldW*scale,height=worldH*scale;
    ctx.fillStyle='#040b0ec0';
    if(left>0)ctx.fillRect(0,0,left,h);
    if(top>0)ctx.fillRect(0,0,w,top);
    if(left+width<w)ctx.fillRect(left+width,0,w-left-width,h);
    if(top+height<h)ctx.fillRect(0,top+height,w,h-top-height);
    ctx.strokeStyle='#a1dfa765';ctx.lineWidth=1.5;ctx.strokeRect(left,top,width,height);
    // subtle view horizon
    if(state.frame){const rad=state.frame.arena.view_radius*scale;
      ctx.beginPath();ctx.arc(w/2,h/2,rad,0,Math.PI*2);
      ctx.setLineDash([6,10]);ctx.strokeStyle='#a0c9ad1c';ctx.stroke();ctx.setLineDash([]);}
  }
  function drawBall(ball,x,y,scale,isHuman,time){
    const r=Math.max(ball.owner==='pellet'?2.3:7,ball.r*scale);
    if(x+r<-30||x-r>canvas.clientWidth+30||y+r<-30||y-r>canvas.clientHeight+30)return;
    if(ball.owner==='pellet'){
      ctx.shadowColor='#bef97a';ctx.shadowBlur=8;
      circle(x,y,r, '#bbec79');ctx.shadowBlur=0;
      circle(x-r*.25,y-r*.25,Math.max(.7,r*.33),'#f3ffd4');
      return;
    }
    const team=playerTeam(ball.owner);const base=team===0?colors.blue:colors.red;
    ctx.shadowColor=base;ctx.shadowBlur=isHuman?27:15;
    const grad=ctx.createRadialGradient(x-r*.4,y-r*.48,r*.1,x,y,r);
    if(team===0){grad.addColorStop(0,'#d3eaff');grad.addColorStop(.32,'#7cbbff');grad.addColorStop(.83,'#3d77be');grad.addColorStop(1,'#1e4269');}
    else{grad.addColorStop(0,'#ffd5d4');grad.addColorStop(.36,'#ff8b8d');grad.addColorStop(.83,'#c04657');grad.addColorStop(1,'#692637');}
    circle(x,y,r,grad);ctx.shadowBlur=0;
    ctx.strokeStyle=isHuman?'#e3fbff':'#ffffff44';ctx.lineWidth=isHuman?2:1;
    ctx.beginPath();ctx.arc(x,y,r-.8,0,Math.PI*2);ctx.stroke();
    ctx.strokeStyle='#ffffff35';ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(x,y,r*.72,Math.PI*1.14,Math.PI*1.83);ctx.stroke();
    if(isHuman){const pulse=Math.sin(time/470)*.055+1;
      ctx.strokeStyle='#d3fcacaa';ctx.lineWidth=1.2;ctx.setLineDash([4,5]);
      ctx.beginPath();ctx.arc(x,y,(r+8)*pulse,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);}
    if(r>14){ctx.font=`800 ${clamp(r*.38,11,18)}px system-ui`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#fff';ctx.shadowBlur=0;
      ctx.fillText(isHuman?'YOU':team===0?'ALLY':'ENEMY',x,y-1);}
    if(r>19){ctx.font='700 10px system-ui';ctx.fillStyle='#e8fff0b8';ctx.fillText(fmt(ball.mass),x,y+14);}
  }
  function draw(time) {
    const {w,h}=resizeCanvas();
    const data=state.frame;
    const scale=Math.min(w/87,h/66); // World units/pixel, independent of retina DPR.
    let tx=data?data.player.x:50,ty=data?data.player.y:50;
    if(data && state.previous && !data.paused && !data.finished){
      const a=clamp((time-state.receivedAt)/(1000/(data.hz||5)),0,1);
      tx=lerp(state.previous.player.x,data.player.x,a);
      ty=lerp(state.previous.player.y,data.player.y,a);
    }
    // Camera remains centered on last location after elimination.
    if(data?.player?.alive){state.cam.x=lerp(state.cam.x,tx,.085);state.cam.y=lerp(state.cam.y,ty,.085);}
    const ox=state.initializedCamera?state.cam.x:50,oy=state.initializedCamera?state.cam.y:50;
    drawBackground(w,h,time,scale,ox,oy,data?.arena.width??100,data?.arena.height??100);
    if(!data){
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='800 15px system-ui';ctx.fillStyle='#c6f27b92';
      ctx.fillText('◌  SYSTEM READY / AWAITING MATCH',w/2,h/2);
    } else {
      const old=new Map((state.previous?.visible_balls||[]).map(b=>[b.id,b]));
      const alpha=data.paused?1:clamp((time-state.receivedAt)/(1000/(data.hz||5)),0,1);
      const ordered=[...data.visible_balls].sort((a,b)=>
        (a.owner==='pellet'?0:1)-(b.owner==='pellet'?0:1)||a.mass-b.mass);
      for(const ball of ordered){const prev=old.get(ball.id);
        const bx=prev?lerp(prev.x,ball.x,alpha):ball.x;
        const by=prev?lerp(prev.y,ball.y,alpha):ball.y;
        drawBall(ball,(bx-ox)*scale+w/2,(by-oy)*scale+h/2,scale,
          ball.owner===data.player.id,time);
      }
      if(data.paused){ctx.fillStyle='#061017aa';ctx.fillRect(0,0,w,h);
        ctx.textAlign='center';ctx.fillStyle='#d1f1b3';ctx.font='900 32px system-ui';ctx.fillText('PAUSED',w/2,h/2);}
      // The minimap is intentionally own-team + visible-entity only.
      const mw=112,mh=112,mx=w-mw-22,my=h-mh-51;
      if(w>640){
        ctx.fillStyle='#09141bd9';ctx.fillRect(mx-7,my-7,mw+14,mh+14);
        ctx.strokeStyle='#466c6177';ctx.strokeRect(mx-7,my-7,mw+14,mh+14);
        for(const b of data.visible_balls){if(b.owner==='pellet')continue;
          const color=playerTeam(b.owner)===0?colors.blue:colors.red;
          circle(mx+b.x/data.arena.width*mw,my+b.y/data.arena.height*mh,Math.max(1.8,b.r),color);
        }
        const px=mx+data.player.x/data.arena.width*mw;
        const py=my+data.player.y/data.arena.height*mh;
        ctx.beginPath();ctx.arc(px,py,data.arena.view_radius/data.arena.width*mw,0,Math.PI*2);
        ctx.strokeStyle='#c6f27b66';ctx.lineWidth=.8;ctx.stroke();
      }
    }
    // subtle corner targeting reticle in all states
    ctx.strokeStyle='#89b6a833';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(w/2-10,h/2);ctx.lineTo(w/2-4,h/2);
    ctx.moveTo(w/2+4,h/2);ctx.lineTo(w/2+10,h/2);
    ctx.moveTo(w/2,h/2-10);ctx.lineTo(w/2,h/2-4);
    ctx.moveTo(w/2,h/2+4);ctx.lineTo(w/2,h/2+10);ctx.stroke();
    requestAnimationFrame(draw);
  }
  function resetSameSeed(){if(state.replayMode || (state.frame?.room_id&&!state.frame?.is_host))return;send({type:'restart',new_seed:false});$('resultOverlay').classList.add('hidden');$('resultOverlay').dataset.shown='';state.lastEvents.clear();$('activityFeed').replaceChildren();toast('重新开始 · 使用相同随机种子');}
  function resetNewSeed(){if(state.replayMode || (state.frame?.room_id&&!state.frame?.is_host))return;send({type:'restart',new_seed:true});$('resultOverlay').classList.add('hidden');$('resultOverlay').dataset.shown='';state.lastEvents.clear();$('activityFeed').replaceChildren();toast('新比赛已开始');}
  function togglePause(){if(!state.replayMode&&state.frame&&!state.frame.finished&&(!state.frame.room_id||state.frame.is_host))send({type:'pause'});}
  async function apiRequest(endpoint, payload) {
    const response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await response.json();
    if(!response.ok)throw new Error(data.detail||`HTTP ${response.status}`);
    return data;
  }
  function roomError(error){$('roomHint').textContent='操作失败：'+error.message;toast(error.message);}
  async function createRoom(){
    try{
      $('createRoomButton').disabled=true;
      const room=await apiRequest('/api/rooms',{mode:state.mode,bot:$('botSelector').value});
      $('roomHint').textContent=`已创建房间 ${room.room_id}，可分享给其他玩家。`;
      $('roomCodeInput').value=room.room_id;
      connect(room);
    }catch(e){roomError(e);}finally{$('createRoomButton').disabled=false;}
  }
  async function joinRoom(){
    try{
      const id=$('roomCodeInput').value.trim().toLowerCase();
      if(!/^[a-f0-9]{16}$/.test(id))throw new Error('请输入有效的 16 位房间码');
      const choice=$('teamChoice').value;
      const room=await apiRequest(`/api/rooms/${id}/join`,choice===''?{}:{team:Number(choice)});
      state.mode=room.mode;
      connect(room);
    }catch(e){roomError(e);}
  }
  async function loadBots(){
    try{
      const response=await fetch('/api/bots');if(!response.ok)return;
      const available=(await response.json()).available||[];
      for(const name of available.filter(id=>id!=='scripted')){
        const opt=document.createElement('option');opt.value=name;opt.textContent=`MAPPO · ${name}`;$('botSelector').append(opt);
      }
    }catch{ /* Backend may be disconnected before starting. */ }
  }
  function stopReplay(){
    if(state.replayTimer!==null){clearInterval(state.replayTimer);state.replayTimer=null;}
    state.replayMode=false;state.replayFrames=[];
    $('replayBar').hidden=true;
  }
  function displayReplayFrame(index){
    if(!state.replayFrames.length)return;
    state.replayIndex=clamp(index,0,state.replayFrames.length-1);
    $('replayRange').value=state.replayIndex;
    $('replayTime').textContent=`${state.replayIndex} / ${state.replayFrames.length-1}`;
    consumeFrame(state.replayFrames[state.replayIndex]);
  }
  async function showReplay(){
    if(!state.room?.replay_id||!state.frame?.finished)return;
    try{
      $('showReplayButton').disabled=true;
      const replay_id=state.room.replay_id;
      const liveFrame=state.frame;
      const viewer=state.frame.player.id;
      const headers={'X-Replay-Token':state.room.replay_token};
      let frames=[];let count=0;
      do{
        const response=await fetch(`/api/replays/${replay_id}/frames?viewer=${encodeURIComponent(viewer)}&start=${count}&limit=100`,{headers});
        const data=await response.json();if(!response.ok)throw new Error(data.detail||'回放校验失败');
        if(!data.verified)throw new Error('回放尚未验证');
        frames.push(...data.frames.filter((f,i)=>!frames.length||f.tick!==frames[frames.length-1].tick));
        count=frames.at(-1)?.tick+1||0;
        if(count>data.total_ticks)break;
      }while(count<=state.frame.tick);
      state.replayFrames=frames.map(frame=>({...frame,room_id:liveFrame.room_id,
        replay_id:liveFrame.replay_id,replay_token:liveFrame.replay_token,
        players_online:liveFrame.players_online,is_host:liveFrame.is_host,bot_type:liveFrame.bot_type}));
      frames=state.replayFrames;
      state.replayMode=true;
      $('resultOverlay').classList.add('hidden');
      $('replayBar').hidden=false;
      $('replayRange').max=String(frames.length-1);
      displayReplayFrame(0);
      $('replayPlayPause').textContent='播放';
      toast(`回放校验通过 · ${frames.length-1} 个策略步`);
    }catch(e){toast('回放不可用：'+e.message);}finally{$('showReplayButton').disabled=false;}
  }
  function toggleReplay(){
    if(state.replayTimer!==null){clearInterval(state.replayTimer);state.replayTimer=null;$('replayPlayPause').textContent='播放';return;}
    $('replayPlayPause').textContent='暂停';
    state.replayTimer=setInterval(()=>{
      if(state.replayIndex>=state.replayFrames.length-1){toggleReplay();return;}
      displayReplayFrame(state.replayIndex+1);
    },200);
  }
  $('createRoomButton').addEventListener('click',createRoom);
  $('joinRoomButton').addEventListener('click',joinRoom);
  $('copyRoomButton').addEventListener('click',()=>{
    if(state.room?.room_id)navigator.clipboard?.writeText(state.room.room_id).then(()=>toast('房间码已复制'),()=>toast(state.room.room_id));
  });
  $('showReplayButton').addEventListener('click',showReplay);
  $('resultReplayButton').addEventListener('click',showReplay);
  $('replayPlayPause').addEventListener('click',toggleReplay);
  $('replayRange').addEventListener('input',e=>{if(state.replayTimer!==null)toggleReplay();displayReplayFrame(Number(e.target.value));});
  $('replayExit').addEventListener('click',()=>{const last=state.replayFrames.at(-1);stopReplay();if(last){$('resultOverlay').dataset.shown='';consumeFrame(last);}else $('startOverlay').classList.remove('hidden');});
  try{
    const last=JSON.parse(sessionStorage.getItem('ball-marl-room-v1')||'null');
    if(last && /^[a-f0-9]{16}$/.test(last.room_id) && typeof last.token==='string'){
      $('rejoinRoomButton').hidden=false;
      $('rejoinRoomButton').textContent=`恢复上次房间 · ${last.room_id.toUpperCase()}`;
      $('rejoinRoomButton').addEventListener('click',()=>connect(last));
    }
  }catch{/* ignore invalid browser state */}
  loadBots();
  document.querySelectorAll('.mode-option').forEach(button=>button.addEventListener('click',()=>{
    state.mode=button.dataset.mode;document.querySelectorAll('.mode-option').forEach(b=>b.classList.toggle('selected',b===button));
  }));
  $('playButton').addEventListener('click',()=>connect());
  $('pauseButton').addEventListener('click',togglePause);
  $('restartButton').addEventListener('click',resetSameSeed);
  $('splitButton').addEventListener('click',()=>pressAbility('split'));
  $('ejectButton').addEventListener('click',()=>pressAbility('eject'));
  $('playAgainButton').addEventListener('click',resetNewSeed);
  $('changeModeButton').addEventListener('click',()=>{
    closeSocket();state.frame=null;$('resultOverlay').classList.add('hidden');$('startOverlay').classList.remove('hidden');connectionStatus(false,'未连接');
  });
  $('menuButton').addEventListener('click',()=>$('helpOverlay').classList.remove('hidden'));
  $('helpClose').addEventListener('click',()=>$('helpOverlay').classList.add('hidden'));
  $('helpGotIt').addEventListener('click',()=>$('helpOverlay').classList.add('hidden'));
  $('soundButton').addEventListener('click',()=>{state.sound=!state.sound;$('soundButton').style.color=state.sound?'var(--lime)':'';toast(state.sound?'音效已打开':'音效已关闭');});
  document.addEventListener('keydown',e=>{
    if(e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement)return;
    const key=e.key.toLowerCase();
    if([' ','arrowup','arrowdown','arrowleft','arrowright'].includes(key))e.preventDefault();
    if(!e.repeat && key==='p'){togglePause();return;}
    if(!e.repeat && key==='r' && state.frame && !state.replayMode){resetSameSeed();return;}
    if(!e.repeat && key==='escape')$('helpOverlay').classList.add('hidden');
    state.keys.add(key);sendInput();
  });
  document.addEventListener('keyup',e=>{state.keys.delete(e.key.toLowerCase());sendInput();});
  window.addEventListener('blur',()=>{state.keys.clear();state.mouse.active=false;state.touch.active=false;sendInput();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){state.keys.clear();state.mouse.active=false;sendInput();}});
  canvas.addEventListener('pointermove',e=>{if(e.pointerType==='mouse'){state.mouse.active=true;state.mouse.x=e.clientX;state.mouse.y=e.clientY;}});
  canvas.addEventListener('pointerleave',()=>{state.mouse.active=false;sendInput();});
  const joystick=$('joystick'),knob=$('joystickKnob');
  const joy=e=>{const rect=joystick.getBoundingClientRect();let dx=(e.clientX-rect.left-rect.width/2)/(rect.width*.38),dy=(e.clientY-rect.top-rect.height/2)/(rect.height*.38);
    const n=Math.max(1,Math.hypot(dx,dy));dx/=n;dy/=n;state.touch.active=true;state.touch.x=dx;state.touch.y=dy;
    knob.style.transform=`translate(${dx*29}px,${dy*29}px)`;sendInput();};
  joystick.addEventListener('pointerdown',e=>{joystick.setPointerCapture(e.pointerId);joy(e);});
  joystick.addEventListener('pointermove',e=>{if(state.touch.active)joy(e);});
  const resetJoy=()=>{state.touch.active=false;knob.style.transform='';sendInput();};
  joystick.addEventListener('pointerup',resetJoy);joystick.addEventListener('pointercancel',resetJoy);
  for(const [id,kind] of [['mobileSplit','split'],['mobileEject','eject']]){
    $(id).addEventListener('pointerdown',e=>{e.preventDefault();pressAbility(kind);});
  }
  connectionStatus(false,'未连接');
  requestAnimationFrame(draw);
})();
