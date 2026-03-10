/* ── marked / hljs init ── */
marked.setOptions({ breaks: true, gfm: true });

/* ══════════════════════════════════════════
   State
══════════════════════════════════════════ */
let ws = null;
let ttsEnabled   = true;
let isBusy       = false;
let realtimeMode = false;

// streaming assistant bubble state
let asBubble = null, asText = "", asCursor = null, asWrap = null;
let typingEl = null;

// realtime audio state
let rtStream=null, rtAudioCtx=null, rtAnalyser=null;
let rtMediaRec=null, rtChunks=[];
let rtSilenceTimer=null, rtSpeechDetected=false;
let rtScanId=null;
const RT_SILENCE_MS  = 900;
const RT_ENERGY_THR  = 0.014;

// PTT state
let pttMediaRec=null, pttChunks=[], pttTimerIv=null;

// waveform analyser (shared PTT/RT)
let wfAnalyser=null, wfAnimId=null;

/* ══════════════════════════════════════════
   Helpers
══════════════════════════════════════════ */
function $(id) { return document.getElementById(id); }

function nowStr() {
  return new Date().toLocaleTimeString('zh-CN', {hour:'2-digit', minute:'2-digit'});
}

function setStatus(text, color='green') {
  $('status-text').textContent = text;
  $('status-dot').className = 'status-dot ' + color;
}

function setWaveLabel(t) { $('wave-label').textContent = t; }

// safe base64 encode (chunked, no stack overflow)
function toB64(buffer) {
  const bytes = new Uint8Array(buffer);
  let s = '', chunk = 8192;
  for (let i=0; i<bytes.length; i+=chunk)
    s += String.fromCharCode(...bytes.subarray(i, i+chunk));
  return btoa(s);
}

function copyText(btn, text, label='复制') {
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✓ 已复制';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = label; btn.classList.remove('copied'); }, 1500);
  });
}

function scrollBottom() {
  const chat = $('chat');
  requestAnimationFrame(() => chat.scrollTop = chat.scrollHeight);
}

function hideWelcome() {
  const w = $('welcome-screen');
  if (w) w.style.display = 'none';
}

function addInfo(text, isError=false) {
  hideWelcome();
  const d = document.createElement('div');
  d.className = 'info' + (isError ? ' err' : '');
  d.textContent = text;
  $('chat').appendChild(d);
  scrollBottom();
}

/* ══════════════════════════════════════════
   WebSocket
══════════════════════════════════════════ */
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => setStatus('就绪', 'green');
  ws.onclose = () => { setStatus('重连中...', 'red'); setTimeout(connect, 2000); };
  ws.onerror = () => setStatus('连接错误', 'red');
  ws.onmessage = onWsMessage;
}

function onWsMessage(e) {
  const msg = JSON.parse(e.data);
  switch (msg.type) {
    case 'token':
      appendToken(msg.text); break;
    case 'done':
      finishAssistant();
      setStatus('就绪', 'green');
      isBusy = false;
      // realtime 模式等 tts_done 再开麦，避免 TTS 回声循环
      if (realtimeMode && ttsEnabled === false) setTimeout(rtStartListening, 300);
      break;
    case 'tts_done':
      // TTS 全部播完，实时模式现在可以安全开麦
      if (realtimeMode) setTimeout(rtStartListening, 300);
      break;
    case 'user':
      removeTyping(); addUserMsg(msg.text); showTyping(); break;
    case 'asr':
      $('text-input').value = msg.text;
      if (realtimeMode) rtSetState('processing', msg.text);
      else setWaveLabel('识别完成');
      break;
    case 'error':
      removeTyping();
      addInfo('❌ ' + msg.text, true);
      setStatus('就绪', 'green');
      isBusy = false;
      if (realtimeMode) setTimeout(rtStartListening, 800);
      break;
    case 'status': setStatus(msg.text, msg.color || 'green'); break;
    case 'cleared':
      $('chat').innerHTML = '';
      { const w2 = $('welcome-screen'); if(w2) w2.style.display = ''; }
      break;
    case 'models_updated': onModelsUpdated(msg.options); break;
  }
}

function wsSend(obj) {
  if (ws && ws.readyState === 1)
    ws.send(JSON.stringify(obj));
}

/* ══════════════════════════════════════════
   Chat DOM helpers
══════════════════════════════════════════ */
function addUserMsg(text) {
  hideWelcome();
  const chat = $('chat');
  const row  = document.createElement('div');
  row.className = 'msg user';

  const av = document.createElement('div');
  av.className = 'avatar'; av.textContent = '👤';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.textContent = text;

  const cb = document.createElement('button');
  cb.className = 'copy-btn'; cb.textContent = '复制';
  cb.onclick = () => copyText(cb, text);
  bub.appendChild(cb);

  const ts = document.createElement('span');
  ts.className = 'ts'; ts.textContent = nowStr();

  body.appendChild(bub); body.appendChild(ts);
  row.appendChild(av); row.appendChild(body);
  chat.appendChild(row);
  scrollBottom();
}

function showTyping() {
  removeTyping();
  const chat = $('chat');
  const row = document.createElement('div');
  row.className = 'msg assistant'; row.id = 'typing-row';
  row.innerHTML = `<div class="avatar">🔧</div>
    <div class="msg-body">
      <div class="typing-dots">
        <div class="td"></div><div class="td"></div><div class="td"></div>
      </div>
    </div>`;
  chat.appendChild(row);
  typingEl = row;
  scrollBottom();
}

function removeTyping() {
  if (typingEl) { typingEl.remove(); typingEl = null; }
}

function startAssistant() {
  removeTyping();
  asText = '';
  const chat = $('chat');
  const row = document.createElement('div');
  row.className = 'msg assistant';

  const av = document.createElement('div');
  av.className = 'avatar'; av.textContent = '🔧';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const bub = document.createElement('div');
  bub.className = 'bubble';

  asCursor = document.createElement('span');
  asCursor.className = 'cursor';
  bub.appendChild(asCursor);

  const ts = document.createElement('span');
  ts.className = 'ts'; ts.textContent = nowStr();

  body.appendChild(bub); body.appendChild(ts);
  row.appendChild(av); row.appendChild(body);
  chat.appendChild(row);

  asBubble = bub; asWrap = body;
  scrollBottom();
}

function appendToken(tok) {
  if (!asBubble) startAssistant();
  asText += tok;
  asBubble.textContent = asText;
  asBubble.appendChild(asCursor);
  scrollBottom();
}

function wrapCodeBlocks(bubble) {
  bubble.querySelectorAll('pre').forEach(pre => {
    if (pre.parentElement.classList.contains('code-wrap')) return;
    const code = pre.querySelector('code');
    const lang = code ? (code.className.replace('language-','').split(' ')[0] || 'code') : 'code';
    hljs.highlightElement(code);

    const wrap = document.createElement('div');
    wrap.className = 'code-wrap';

    const hdr = document.createElement('div');
    hdr.className = 'code-header';

    const lbl = document.createElement('span');
    lbl.className = 'code-lang'; lbl.textContent = lang;

    const cp = document.createElement('button');
    cp.className = 'code-copy'; cp.textContent = '复制代码';
    cp.onclick = () => copyText(cp, code.innerText, '复制代码');

    hdr.appendChild(lbl); hdr.appendChild(cp);
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(hdr); wrap.appendChild(pre);
  });
}

function finishAssistant() {
  if (!asBubble) return;
  asBubble.innerHTML = marked.parse(asText);
  wrapCodeBlocks(asBubble);

  const cb = document.createElement('button');
  cb.className = 'copy-btn'; cb.textContent = '复制';
  cb.onclick = () => copyText(cb, asBubble.innerText);
  asBubble.appendChild(cb);

  asBubble = null; asCursor = null;
  scrollBottom();
}

/* ══════════════════════════════════════════
   Send text
══════════════════════════════════════════ */
function sendText() {
  const inp = $('text-input');
  const text = inp.value.trim();
  if (!text || isBusy) return;
  isBusy = true;
  inp.value = ''; inp.style.height = '40px';
  wsSend({type:'text', text});
}

$('btn-send').onclick = sendText;
$('text-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendText(); }
});
$('text-input').addEventListener('input', function() {
  this.style.height = '40px';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

/* ══════════════════════════════════════════
   Waveform
══════════════════════════════════════════ */
function startWaveform(analyserNode) {
  wfAnalyser = analyserNode;
  const canvas = $('waveform');
  const ctx = canvas.getContext('2d');
  function draw() {
    wfAnimId = requestAnimationFrame(draw);
    const w = canvas.width  = canvas.offsetWidth;
    const h = canvas.height = canvas.offsetHeight;
    const buf = new Uint8Array(wfAnalyser.frequencyBinCount);
    wfAnalyser.getByteTimeDomainData(buf);
    ctx.clearRect(0,0,w,h);
    const g = ctx.createLinearGradient(0,0,w,0);
    g.addColorStop(0,   'rgba(91,124,246,0.3)');
    g.addColorStop(0.5, 'rgba(91,124,246,0.9)');
    g.addColorStop(1,   'rgba(91,124,246,0.3)');
    ctx.beginPath(); ctx.strokeStyle = g; ctx.lineWidth = 1.5;
    const step = w / buf.length;
    buf.forEach((v,i) => {
      const x = i * step;
      const y = ((v/128)-1) * h*0.4 + h/2;
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
  }
  draw();
}

function stopWaveform() {
  if (wfAnimId) cancelAnimationFrame(wfAnimId);
  wfAnimId = null; wfAnalyser = null;
  const canvas = $('waveform');
  canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
}

/* ══════════════════════════════════════════
   PTT Recording (manual)
══════════════════════════════════════════ */
const micBtn = $('btn-mic');

async function pttStart() {
  if (pttMediaRec || isBusy || realtimeMode) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1}});
  } catch(err) {
    addInfo('❌ 麦克风权限拒绝: ' + err.message, true); return;
  }
  pttChunks = [];
  pttMediaRec = new MediaRecorder(stream);
  pttMediaRec.ondataavailable = e => { if(e.data.size>0) pttChunks.push(e.data); };
  pttMediaRec.onstop = async () => {
    clearInterval(pttTimerIv);
    $('rec-timer').textContent = '';
    stopWaveform();
    stream.getTracks().forEach(t=>t.stop());
    micBtn.classList.remove('recording');
    if (!pttChunks.length) { pttMediaRec=null; return; }
    const blob = new Blob(pttChunks, {type: pttMediaRec.mimeType||'audio/webm'});
    const b64 = toB64(await blob.arrayBuffer());
    isBusy = true;
    wsSend({type:'audio', data:b64});
    setWaveLabel('识别中'); setStatus('识别中...','yellow');
    pttMediaRec = null;
  };

  pttMediaRec.start(100);
  micBtn.classList.add('recording');
  setWaveLabel('录音中');

  // timer
  const t0 = Date.now();
  pttTimerIv = setInterval(()=>{
    $('rec-timer').textContent = ((Date.now()-t0)/1000).toFixed(1)+'s';
  }, 100);

  // waveform
  const actx = new AudioContext();
  const an = actx.createAnalyser(); an.fftSize = 512;
  actx.createMediaStreamSource(stream).connect(an);
  startWaveform(an);
}

function pttStop() {
  if (pttMediaRec && pttMediaRec.state==='recording') pttMediaRec.stop();
}

micBtn.addEventListener('mousedown', e=>{ e.preventDefault(); pttStart(); });
micBtn.addEventListener('touchstart', e=>{ e.preventDefault(); pttStart(); }, {passive:false});
micBtn.addEventListener('mouseup',   pttStop);
micBtn.addEventListener('mouseleave',pttStop);
micBtn.addEventListener('touchend',  pttStop);
micBtn.addEventListener('touchcancel',pttStop);

document.addEventListener('keydown', e=>{
  if (e.code==='Space' && document.activeElement?.tagName!=='TEXTAREA'
      && document.activeElement?.tagName!=='INPUT' && !e.repeat) {
    e.preventDefault(); pttStart();
  }
});
document.addEventListener('keyup', e=>{ if(e.code==='Space') pttStop(); });

/* ══════════════════════════════════════════
   Realtime Mode
══════════════════════════════════════════ */
const rtOverlay   = $('rt-overlay');
const rtOrb       = $('rt-orb');
const rtRings     = document.querySelectorAll('.rt-ring');
const rtStateText = $('rt-state-text');
const rtTransText = $('rt-transcript');
const rtWaveCanvas= $('rt-wave-canvas');

function rtSetState(state, caption='') {
  const stateMap = {
    listening:  {icon:'🎙', label:'等待说话...'},
    speaking:   {icon:'🎤', label:'正在说话'},
    processing: {icon:'⚙️', label:'处理中...'},
  };
  const s = stateMap[state] || stateMap.listening;
  rtOrb.className = 'rt-orb ' + state;
  rtRings.forEach(r => { r.className = 'rt-ring ' + state; });
  rtStateText.className = 'rt-state-text ' + state;
  rtStateText.textContent = s.label;
  rtOrb.textContent = s.icon;
  rtTransText.textContent = caption;

  // wave bar sync
  const wbar = $('bottom-bar');
  if (state==='listening'||state==='speaking') wbar.classList.add('rt-on');
  else wbar.classList.remove('rt-on');

  // sb-btn state
  const btn = $('btn-rt');
  btn.className = 'sb-btn ' + (
    state==='speaking' ? 'active-red' :
    state==='processing' ? 'active' : 'active-green'
  );
}

async function rtOpen() {
  if (rtStream) return;
  try {
    rtStream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, sampleRate:16000}});
  } catch(e) {
    addInfo('❌ 麦克风权限拒绝: '+e.message, true); return false;
  }
  rtAudioCtx = new AudioContext({sampleRate:16000});
  rtAnalyser = rtAudioCtx.createAnalyser(); rtAnalyser.fftSize=256;
  rtAudioCtx.createMediaStreamSource(rtStream).connect(rtAnalyser);
  startWaveform(rtAnalyser);

  // also drive overlay wave canvas
  rtDrawOverlayWave();
  return true;
}

function rtClose() {
  realtimeMode = false;
  clearTimeout(rtSilenceTimer);
  cancelAnimationFrame(rtScanId);
  if (rtMediaRec && rtMediaRec.state!=='inactive') rtMediaRec.stop();
  if (rtStream) { rtStream.getTracks().forEach(t=>t.stop()); rtStream=null; }
  if (rtAudioCtx) { rtAudioCtx.close(); rtAudioCtx=null; }
  rtAnalyser=null; stopWaveform();

  rtOverlay.classList.remove('show');
  const btn = $('btn-rt');
  btn.className = 'sb-btn';
  $('bottom-bar').classList.remove('rt-on');
  setWaveLabel('待机');
  setStatus('就绪','green');
  isBusy = false;
}

function rtStartListening() {
  if (!realtimeMode || !rtStream) return;
  rtSpeechDetected = false; rtChunks = [];
  isBusy = false;
  rtSetState('listening');
  setWaveLabel('监听中');

  rtMediaRec = new MediaRecorder(rtStream);
  rtMediaRec.ondataavailable = e=>{ if(e.data.size>0) rtChunks.push(e.data); };
  rtMediaRec.start(50);

  rtEnergyScan();
}

function rtEnergyScan() {
  if (!realtimeMode) return;
  const buf = new Uint8Array(rtAnalyser.frequencyBinCount);
  rtAnalyser.getByteTimeDomainData(buf);
  let sum=0;
  buf.forEach(v=>{ const d=(v-128)/128; sum+=d*d; });
  const rms = Math.sqrt(sum/buf.length);

  if (rms > RT_ENERGY_THR) {
    if (!rtSpeechDetected) {
      rtSpeechDetected = true;
      rtSetState('speaking'); setWaveLabel('说话中');
    }
    clearTimeout(rtSilenceTimer);
    rtSilenceTimer = setTimeout(rtOnSilence, RT_SILENCE_MS);
  }
  rtScanId = requestAnimationFrame(rtEnergyScan);
}

async function rtOnSilence() {
  if (!realtimeMode || !rtSpeechDetected) return;
  cancelAnimationFrame(rtScanId);
  rtSetState('processing'); setWaveLabel('识别中');
  setStatus('识别中...','yellow');
  isBusy = true;

  await new Promise(res=>{ rtMediaRec.onstop=res; rtMediaRec.stop(); });
  if (!rtChunks.length) { isBusy=false; rtStartListening(); return; }

  const blob = new Blob(rtChunks, {type: rtMediaRec.mimeType||'audio/webm'});
  const b64 = toB64(await blob.arrayBuffer());
  wsSend({type:'audio_rt', data:b64});
}

// overlay wave canvas
let _rtWaveId=null;
function rtDrawOverlayWave() {
  if (!rtAudioCtx) return;
  const canvas = rtWaveCanvas;
  const ctx = canvas.getContext('2d');
  function draw() {
    _rtWaveId = requestAnimationFrame(draw);
    const w = canvas.width  = canvas.offsetWidth;
    const h = canvas.height = canvas.offsetHeight;
    if (!rtAnalyser) return;
    const buf = new Uint8Array(rtAnalyser.frequencyBinCount);
    rtAnalyser.getByteTimeDomainData(buf);
    ctx.clearRect(0,0,w,h);
    const g = ctx.createLinearGradient(0,0,w,0);
    g.addColorStop(0,'rgba(45,212,170,0.2)');
    g.addColorStop(0.5,'rgba(45,212,170,0.8)');
    g.addColorStop(1,'rgba(45,212,170,0.2)');
    ctx.beginPath(); ctx.strokeStyle=g; ctx.lineWidth=1.5;
    const step = w/buf.length;
    buf.forEach((v,i)=>{
      const x=i*step, y=((v/128)-1)*h*0.45+h/2;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
  }
  draw();
}

// toggle button
$('btn-rt').onclick = async () => {
  realtimeMode = !realtimeMode;
  if (realtimeMode) {
    const ok = await rtOpen();
    if (!ok) { realtimeMode=false; return; }
    rtOverlay.classList.add('show');
    rtStartListening();
  } else {
    cancelAnimationFrame(_rtWaveId);
    rtClose();
  }
};

$('rt-close').onclick = () => {
  realtimeMode = false;
  cancelAnimationFrame(_rtWaveId);
  rtClose();
};

/* ══════════════════════════════════════════
   Sidebar controls
══════════════════════════════════════════ */
$('btn-tts').onclick = () => {
  ttsEnabled = !ttsEnabled;
  const btn = $('btn-tts');
  btn.textContent  = ttsEnabled ? '🔊' : '🔇';
  btn.dataset.tip  = ttsEnabled ? 'TTS 已开启，点击关闭' : 'TTS 已关闭，点击开启';
  btn.className    = 'sb-btn' + (ttsEnabled ? ' active' : ' muted');
  wsSend({type:'tts_toggle', enabled:ttsEnabled});
};

$('btn-clear').onclick = () => {
  wsSend({type:'clear'});
  $('chat').innerHTML = '';
  const w = $('welcome-screen');
  if (w) w.style.display = '';
};

$('model-select').onchange = e => {
  wsSend({type:'model_switch', key:e.target.value});
  setStatus('已切换: ' + e.target.options[e.target.selectedIndex].text, 'green');
};

/* ── Engine bar popup menus ── */
function makeEngPopup(triggerId, popupId) {
  const trigger = $(triggerId);
  const popup   = $(popupId);
  if (!trigger || !popup) return;
  trigger.onclick = e => {
    e.stopPropagation();
    const rect = trigger.getBoundingClientRect();
    popup.style.bottom = (window.innerHeight - rect.top + 4) + 'px';
    popup.classList.toggle('show');
    document.querySelectorAll('.eng-popup').forEach(p => { if(p!==popup) p.classList.remove('show'); });
  };
}
makeEngPopup('eng-asr', 'popup-asr');
makeEngPopup('eng-tts', 'popup-tts-engine');

document.addEventListener('click', () => {
  document.querySelectorAll('.eng-popup').forEach(p => p.classList.remove('show'));
});

// ASR switch
document.querySelectorAll('[data-asr]').forEach(btn => {
  btn.onclick = e => {
    e.stopPropagation();
    const engine = btn.dataset.asr;
    document.querySelectorAll('[data-asr]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const label = btn.textContent.replace(/^[^\s]+\s/, '').replace('（DashScope）','').replace('（本地）','').trim();
    $('eng-asr-val').textContent = label;
    wsSend({type:'asr_switch', engine});
    setStatus('ASR: ' + engine, 'green');
    $('popup-asr').classList.remove('show');
  };
});

// TTS engine switch
document.querySelectorAll('[data-tts-engine]').forEach(btn => {
  btn.onclick = e => {
    e.stopPropagation();
    const engine = btn.dataset.ttsEngine;
    document.querySelectorAll('[data-tts-engine]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const label = {'edge':'Edge','qwen3tts':'Qwen3','pyttsx3':'Local'}[engine] || engine;
    $('eng-tts-val').textContent = label;
    wsSend({type:'tts_engine_switch', engine});
    setStatus('TTS: ' + engine, 'green');
    $('popup-tts-engine').classList.remove('show');
  };
});

/* ══════════════════════════════════════════
   Init
══════════════════════════════════════════ */

/* ══ Settings Panel ══ */
$('btn-settings').onclick = () => $('settings-panel').classList.add('show');
$('sp-close').onclick     = () => $('settings-panel').classList.remove('show');
document.querySelector('.sp-backdrop').addEventListener('click', () => $('settings-panel').classList.remove('show'));

$('sp-save').onclick = () => {
  const fieldMap = {
    'sp-qwen-key':'qwen_api_key','sp-openai-key':'openai_api_key',
    'sp-deepseek-key':'deepseek_api_key','sp-groq-key':'groq_api_key',
    'sp-siliconflow-key':'siliconflow_api_key',
  };
  const keys = {};
  for (const [id, field] of Object.entries(fieldMap)) {
    const v = $(id).value.trim(); if (v) keys[field] = v;
  }
  const voice = $('sp-tts-voice')?.value;
  if (voice) keys.tts_voice = voice;
  wsSend({type:'settings_save', keys});
  $('sp-hint').textContent = '保存中...';
};

function onModelsUpdated(options) {
  const sel = $('model-select');
  if (!sel) return;
  sel.innerHTML = options.map(o=>`<option value="${o.key}">${o.label}</option>`).join('');
  $('sp-hint').textContent = '✓ 已保存';
  setTimeout(()=>{ $('sp-hint').textContent=''; }, 3000);
}

connect();
