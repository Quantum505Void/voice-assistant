/* ── marked / hljs init ── */
marked.setOptions({ breaks: true, gfm: true });

/* ══════════════════════════════════════════
   State
══════════════════════════════════════════ */
let ws = null;
let ttsEnabled   = true;
let isBusy       = false;
let realtimeMode = false;

let asBubble = null, asText = '', asCursor = null;
let typingEl = null;

let rtStream = null, rtAudioCtx = null, rtAnalyser = null;
let rtMediaRec = null, rtChunks = [];
let rtSilenceTimer = null, rtSpeechDetected = false;
let rtScanId = null;
const RT_SILENCE_MS = 900;
const RT_ENERGY_THR = 0.014;

let pttMediaRec = null, pttChunks = [], pttTimerIv = null;
let wfAnalyser = null, wfAnimId = null;
let _rtWaveId = null;

const HISTORY_KEY = 'xiaoxin_chat_history';
const MAX_HISTORY = 100;
let _userScrolled = false; // 用户是否主动上翻

/* ══════════════════════════════════════════
   Helpers
══════════════════════════════════════════ */
const $ = id => document.getElementById(id);

function nowStr() {
  return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function setStatus(text, color = 'green') {
  $('status-text').textContent = text;
  $('status-dot').className = 'status-dot ' + color;
}

/* 智能滚动：监听用户上翻 */
function initScrollBehavior() {
  const chat = $('chat');
  const btn  = $('scroll-bottom-btn');
  chat.addEventListener('scroll', () => {
    const atBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 60;
    _userScrolled = !atBottom;
    if (btn) btn.classList.toggle('show', _userScrolled);
  });
  if (btn) btn.addEventListener('click', () => {
    _userScrolled = false;
    scrollBottom(true);
  });
}

function setWaveLabel(t) {
  const el = $('wave-label');
  if (el) el.textContent = t;
}

function toB64(buffer) {
  const bytes = new Uint8Array(buffer);
  let s = '', chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk)
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return btoa(s);
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function copyText(btn, text, label = '复制') {
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✓ 已复制';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = label; btn.classList.remove('copied'); }, 1500);
  });
}

function scrollBottom(force = false) {
  const chat = $('chat');
  if (force || !_userScrolled) {
    requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
  }
}

/* Toast 通知 */
function showToast(text, type = 'info', durationMs = 2800) {
  let container = $('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = text;
  container.appendChild(t);
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => t.remove(), 300);
  }, durationMs);
}

function hideWelcome() {
  const w = $('welcome-screen');
  if (w) w.style.display = 'none';
}

function showWelcome() {
  const w = $('welcome-screen');
  if (w) w.style.display = '';
  $('chat').innerHTML = '';
}

function addInfo(text, isError = false) {
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
  ws.onopen  = () => setStatus('就绪', 'green');
  ws.onclose = () => { setStatus('重连中…', 'red'); setTimeout(connect, 2000); };
  ws.onerror = () => setStatus('连接错误', 'red');
  ws.onmessage = onWsMessage;
}

function wsSend(obj) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

function onWsMessage(e) {
  const msg = JSON.parse(e.data);
  switch (msg.type) {
    case 'token':
      appendToken(msg.text); break;

    case 'done':
      finishAssistant();
      saveHistory();
      setStatus('就绪', 'green');
      isBusy = false;
      if (realtimeMode && !ttsEnabled) setTimeout(rtStartListening, 300);
      break;

    case 'tts_done':
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

    case 'status':
      setStatus(msg.text, msg.color || 'green');
      if (msg.toast) showToast(msg.text, msg.color === 'green' ? 'success' : 'info');
      break;

    case 'cleared':
      showWelcome();
      localStorage.removeItem(HISTORY_KEY);
      break;

    case 'models_updated':
      onModelsUpdated(msg.options); break;
  }
}

/* ══════════════════════════════════════════
   Chat DOM helpers
══════════════════════════════════════════ */
function addUserMsg(text) {
  hideWelcome();
  const row = document.createElement('div');
  row.className = 'msg user';
  row.innerHTML = `
    <div class="avatar">👤</div>
    <div class="msg-body">
      <div class="bubble">${escapeHtml(text)}<button class="copy-btn">复制</button></div>
      <span class="ts">${nowStr()}</span>
    </div>`;
  row.querySelector('.copy-btn').onclick = function () { copyText(this, text); };
  $('chat').appendChild(row);
  scrollBottom();
}

function showTyping() {
  removeTyping();
  const row = document.createElement('div');
  row.className = 'msg assistant'; row.id = 'typing-row';
  row.innerHTML = `<div class="avatar">🔧</div>
    <div class="msg-body"><div class="typing-dots">
      <div class="td"></div><div class="td"></div><div class="td"></div>
    </div></div>`;
  $('chat').appendChild(row);
  typingEl = row;
  scrollBottom();
}

function removeTyping() {
  if (typingEl) { typingEl.remove(); typingEl = null; }
}

function startAssistant() {
  removeTyping();
  asText = '';
  const row = document.createElement('div');
  row.className = 'msg assistant';

  const av  = document.createElement('div');
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
  $('chat').appendChild(row);
  asBubble = bub;
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
    const lang = code
      ? (code.className.replace('language-', '').split(' ')[0] || 'code')
      : 'code';
    if (code) hljs.highlightElement(code);

    const wrap = document.createElement('div'); wrap.className = 'code-wrap';
    const hdr  = document.createElement('div'); hdr.className  = 'code-header';
    const lbl  = document.createElement('span'); lbl.className = 'code-lang'; lbl.textContent = lang;
    const cp   = document.createElement('button'); cp.className = 'code-copy'; cp.textContent = '复制代码';
    cp.onclick = () => copyText(cp, (code || pre).innerText, '复制代码');

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
  const inp  = $('text-input');
  const text = inp.value.trim();
  if (!text || isBusy) return;
  isBusy = true;
  inp.value = ''; inp.style.height = '40px';
  wsSend({ type: 'text', text });
}

$('btn-send').onclick = sendText;
$('text-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendText(); }
});
$('text-input').addEventListener('input', function () {
  this.style.height = '40px';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

/* ══════════════════════════════════════════
   Waveform canvas
══════════════════════════════════════════ */
function startWaveform(analyserNode) {
  wfAnalyser = analyserNode;
  const canvas = $('waveform');
  const ctx    = canvas.getContext('2d');
  function draw() {
    wfAnimId = requestAnimationFrame(draw);
    const w = canvas.width  = canvas.offsetWidth;
    const h = canvas.height = canvas.offsetHeight;
    const buf = new Uint8Array(wfAnalyser.frequencyBinCount);
    wfAnalyser.getByteTimeDomainData(buf);
    ctx.clearRect(0, 0, w, h);
    const g = ctx.createLinearGradient(0, 0, w, 0);
    g.addColorStop(0,   'rgba(91,124,246,0.25)');
    g.addColorStop(0.5, 'rgba(108,143,255,0.9)');
    g.addColorStop(1,   'rgba(91,124,246,0.25)');
    ctx.beginPath(); ctx.strokeStyle = g; ctx.lineWidth = 1.5;
    const step = w / buf.length;
    buf.forEach((v, i) => {
      const x = i * step, y = ((v / 128) - 1) * h * 0.4 + h / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  draw();
}

function stopWaveform() {
  if (wfAnimId) cancelAnimationFrame(wfAnimId);
  wfAnimId = null; wfAnalyser = null;
  const c = $('waveform');
  if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

/* ══════════════════════════════════════════
   PTT (push-to-talk)
══════════════════════════════════════════ */
const micBtn = $('btn-mic');

async function pttStart() {
  if (pttMediaRec || isBusy || realtimeMode) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  } catch (err) {
    addInfo('❌ 麦克风权限拒绝: ' + err.message, true); return;
  }
  pttChunks = [];
  pttMediaRec = new MediaRecorder(stream);
  pttMediaRec.ondataavailable = e => { if (e.data.size > 0) pttChunks.push(e.data); };
  pttMediaRec.onstop = async () => {
    clearInterval(pttTimerIv);
    $('rec-timer').textContent = '';
    stopWaveform();
    stream.getTracks().forEach(t => t.stop());
    micBtn.classList.remove('recording');
    $('bottom-bar').classList.remove('recording');
    if (!pttChunks.length) { pttMediaRec = null; return; }
    const blob = new Blob(pttChunks, { type: pttMediaRec.mimeType || 'audio/webm' });
    const b64  = toB64(await blob.arrayBuffer());
    isBusy = true;
    wsSend({ type: 'audio', data: b64 });
    setWaveLabel('识别中'); setStatus('识别中…', 'yellow');
    pttMediaRec = null;
  };
  pttMediaRec.start(100);
  micBtn.classList.add('recording');
  $('bottom-bar').classList.add('recording');
  setWaveLabel('录音中');

  const t0 = Date.now();
  pttTimerIv = setInterval(() => {
    $('rec-timer').textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's';
  }, 100);

  const actx = new AudioContext();
  const an   = actx.createAnalyser(); an.fftSize = 512;
  actx.createMediaStreamSource(stream).connect(an);
  startWaveform(an);
}

function pttStop() {
  if (pttMediaRec && pttMediaRec.state === 'recording') pttMediaRec.stop();
}

micBtn.addEventListener('mousedown',  e => { e.preventDefault(); pttStart(); });
micBtn.addEventListener('touchstart', e => { e.preventDefault(); pttStart(); }, { passive: false });
micBtn.addEventListener('mouseup',    pttStop);
micBtn.addEventListener('mouseleave', pttStop);
micBtn.addEventListener('touchend',   pttStop);
micBtn.addEventListener('touchcancel',pttStop);

document.addEventListener('keydown', e => {
  if (e.code === 'Space' &&
      document.activeElement?.tagName !== 'TEXTAREA' &&
      document.activeElement?.tagName !== 'INPUT' && !e.repeat) {
    e.preventDefault(); pttStart();
  }
});
document.addEventListener('keyup', e => { if (e.code === 'Space') pttStop(); });

/* ══════════════════════════════════════════
   Realtime Mode
══════════════════════════════════════════ */
const rtOverlay   = $('rt-overlay');
const rtOrb       = $('rt-orb');
const rtRings     = document.querySelectorAll('.rt-ring');
const rtStateText = $('rt-state-text');
const rtTransText = $('rt-transcript');
const rtWaveCanvas= $('rt-wave-canvas');

function rtSetState(state, caption = '') {
  const stateMap = {
    listening:  { icon: '🎙', label: '等待说话…' },
    speaking:   { icon: '🎤', label: '正在说话'   },
    processing: { icon: '⚙️', label: '处理中…'    },
  };
  const s = stateMap[state] || stateMap.listening;
  rtOrb.className        = 'rt-orb ' + state;
  rtStateText.className  = 'rt-state-text ' + state;
  rtStateText.textContent = s.label;
  rtOrb.textContent      = s.icon;
  rtTransText.textContent = caption;
  rtRings.forEach(r => r.className = 'rt-ring ' + state);

  const wbar = $('bottom-bar');
  (state === 'listening' || state === 'speaking')
    ? wbar.classList.add('rt-on')
    : wbar.classList.remove('rt-on');

  $('btn-rt').className = 'sb-btn ' + (
    state === 'speaking'   ? 'active-red'   :
    state === 'processing' ? 'active'        : 'active-green'
  );
}

async function rtOpen() {
  if (rtStream) return true;
  try {
    rtStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16000 }
    });
  } catch (e) {
    addInfo('❌ 麦克风权限拒绝: ' + e.message, true); return false;
  }
  rtAudioCtx = new AudioContext({ sampleRate: 16000 });
  rtAnalyser = rtAudioCtx.createAnalyser(); rtAnalyser.fftSize = 256;
  rtAudioCtx.createMediaStreamSource(rtStream).connect(rtAnalyser);
  startWaveform(rtAnalyser);
  rtDrawOverlayWave();
  return true;
}

function rtClose() {
  realtimeMode = false;
  clearTimeout(rtSilenceTimer);
  cancelAnimationFrame(rtScanId);
  cancelAnimationFrame(_rtWaveId);
  if (rtMediaRec && rtMediaRec.state !== 'inactive') rtMediaRec.stop();
  if (rtStream) { rtStream.getTracks().forEach(t => t.stop()); rtStream = null; }
  if (rtAudioCtx) { rtAudioCtx.close(); rtAudioCtx = null; }
  rtAnalyser = null; stopWaveform();
  rtOverlay.classList.remove('show');
  $('btn-rt').className = 'sb-btn';
  $('bottom-bar').classList.remove('rt-on');
  setWaveLabel('待机');
  setStatus('就绪', 'green');
  isBusy = false;
}

function rtStartListening() {
  if (!realtimeMode || !rtStream) return;
  rtSpeechDetected = false; rtChunks = [];
  isBusy = false;
  rtSetState('listening');
  setWaveLabel('监听中');
  rtMediaRec = new MediaRecorder(rtStream);
  rtMediaRec.ondataavailable = e => { if (e.data.size > 0) rtChunks.push(e.data); };
  rtMediaRec.start(50);
  rtEnergyScan();
}

function rtEnergyScan() {
  if (!realtimeMode) return;
  const buf = new Uint8Array(rtAnalyser.frequencyBinCount);
  rtAnalyser.getByteTimeDomainData(buf);
  let sum = 0;
  buf.forEach(v => { const d = (v - 128) / 128; sum += d * d; });
  const rms = Math.sqrt(sum / buf.length);
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
  setStatus('识别中…', 'yellow');
  isBusy = true;
  await new Promise(res => { rtMediaRec.onstop = res; rtMediaRec.stop(); });
  if (!rtChunks.length) { isBusy = false; rtStartListening(); return; }
  const blob = new Blob(rtChunks, { type: rtMediaRec.mimeType || 'audio/webm' });
  const b64  = toB64(await blob.arrayBuffer());
  wsSend({ type: 'audio_rt', data: b64 });
}

function rtDrawOverlayWave() {
  if (!rtAudioCtx) return;
  const canvas = rtWaveCanvas;
  const ctx    = canvas.getContext('2d');
  function draw() {
    _rtWaveId = requestAnimationFrame(draw);
    const w = canvas.width  = canvas.offsetWidth;
    const h = canvas.height = canvas.offsetHeight;
    if (!rtAnalyser) return;
    const buf = new Uint8Array(rtAnalyser.frequencyBinCount);
    rtAnalyser.getByteTimeDomainData(buf);
    ctx.clearRect(0, 0, w, h);
    const g = ctx.createLinearGradient(0, 0, w, 0);
    g.addColorStop(0,   'rgba(45,212,170,0.15)');
    g.addColorStop(0.5, 'rgba(45,212,170,0.85)');
    g.addColorStop(1,   'rgba(45,212,170,0.15)');
    ctx.beginPath(); ctx.strokeStyle = g; ctx.lineWidth = 1.5;
    const step = w / buf.length;
    buf.forEach((v, i) => {
      const x = i * step, y = ((v / 128) - 1) * h * 0.45 + h / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  draw();
}

async function enterRealtime() {
  realtimeMode = true;
  const ok = await rtOpen();
  if (!ok) { realtimeMode = false; return; }
  rtOverlay.classList.add('show');
  rtStartListening();
}

$('btn-rt').onclick   = async () => realtimeMode ? rtClose() : enterRealtime();
$('rt-close').onclick = () => rtClose();

/* ══════════════════════════════════════════
   Sidebar controls
══════════════════════════════════════════ */
$('btn-tts').onclick = () => {
  ttsEnabled = !ttsEnabled;
  const btn = $('btn-tts');
  btn.textContent = ttsEnabled ? '🔊' : '🔇';
  btn.dataset.tip = ttsEnabled ? 'TTS 开启中，点击关闭' : 'TTS 已关闭，点击开启';
  btn.className   = 'sb-btn' + (ttsEnabled ? ' active' : ' muted');
  wsSend({ type: 'tts_toggle', enabled: ttsEnabled });
};

$('btn-clear').onclick  = () => wsSend({ type: 'clear' });
$('btn-export').onclick = exportChat;

/* ══════════════════════════════════════════
   Header chip popups (LLM / ASR / TTS)
══════════════════════════════════════════ */
function closeAllPopups() {
  document.querySelectorAll('.hdr-popup').forEach(p => p.classList.remove('show'));
  document.querySelectorAll('.hdr-chip').forEach(c => c.classList.remove('open'));
}

function togglePopup(popupId, triggerEl) {
  const popup = document.getElementById(popupId);
  if (!popup) return;
  const wasOpen = popup.classList.contains('show');
  closeAllPopups();
  if (!wasOpen) {
    const rect = triggerEl.getBoundingClientRect();
    popup.style.top   = (rect.bottom + 6) + 'px';
    popup.style.right = (window.innerWidth - rect.right) + 'px';
    popup.style.left  = 'auto';
    popup.classList.add('show');
    triggerEl.classList.add('open');
  }
}

document.getElementById('chip-llm-btn').onclick = e => { e.stopPropagation(); togglePopup('popup-llm', e.currentTarget); };
document.getElementById('chip-asr-btn').onclick = e => { e.stopPropagation(); togglePopup('popup-asr', e.currentTarget); };
document.getElementById('chip-tts-btn').onclick = e => { e.stopPropagation(); togglePopup('popup-tts', e.currentTarget); };

document.addEventListener('click', closeAllPopups);
document.querySelectorAll('.hdr-popup').forEach(p => p.addEventListener('click', e => e.stopPropagation()));

// LLM popup items (data-model)
function bindModelItems() {
  document.querySelectorAll('[data-model]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const key   = btn.dataset.model;
      const label = btn.querySelector('b')?.textContent || key;
      document.querySelectorAll('[data-model]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('chip-llm-label').textContent = label.replace(/^.*·\s*/, '');
      wsSend({ type: 'model_switch', key });
      showToast('🧠 已切换: ' + label, 'success');
      closeAllPopups();
    });
  });
  // mark first as active by default
  const first = document.querySelector('[data-model]');
  if (first && !document.querySelector('[data-model].active')) {
    first.classList.add('active');
  }
}

// ASR popup items
document.querySelectorAll('[data-asr]').forEach(btn => {
  btn.addEventListener('click', e => {
    e.stopPropagation();
    const engine = btn.dataset.asr;
    document.querySelectorAll('[data-asr]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const label = btn.querySelector('b')?.textContent || engine;
    document.getElementById('chip-asr-label').textContent = label;
    wsSend({ type: 'asr_switch', engine });
    closeAllPopups();
  });
});

// TTS popup items
document.querySelectorAll('[data-tts-engine]').forEach(btn => {
  btn.addEventListener('click', e => {
    e.stopPropagation();
    const engine = btn.dataset.ttsEngine;
    document.querySelectorAll('[data-tts-engine]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const label = btn.querySelector('b')?.textContent || engine;
    document.getElementById('chip-tts-label').textContent = label;
    wsSend({ type: 'tts_engine_switch', engine });
    closeAllPopups();
  });
});

/* ══════════════════════════════════════════
   Settings panel
══════════════════════════════════════════ */
function openSettings() {
  $('settings-panel').classList.add('show');
  fetch('/config').then(r => r.json()).then(c => {
    const map = {
      'sp-qwen-key':          c.qwen_api_key,
      'sp-openai-key':        c.openai_api_key,
      'sp-deepseek-key':      c.deepseek_api_key,
      'sp-groq-key':          c.groq_api_key,
      'sp-siliconflow-key':   c.siliconflow_api_key,
      'sp-moonshot-key':      c.moonshot_api_key,
      'sp-baichuan-key':      c.baichuan_api_key,
      'sp-zhipu-key':         c.zhipu_api_key,
      'sp-minimax-key':       c.minimax_api_key,
      'sp-anthropic-key':     c.anthropic_api_key,
      'sp-gemini-key':        c.gemini_api_key,
      'sp-mistral-key':       c.mistral_api_key,
      'sp-azure-key':         c.azure_tts_key,
      'sp-system-prompt':     c.system_prompt,
      'sp-ollama-url':        c.ollama_url,
      'sp-ollama-model':      c.ollama_model,
      'sp-ollama-asr-model':  c.ollama_asr_model,
      'sp-ollama-tts-model':  c.ollama_tts_model,
      'sp-azure-region':      c.azure_tts_region,
    };
    for (const [id, val] of Object.entries(map)) {
      const el = $(id); if (el && val) el.value = val;
    }
    if (c.tts_voice)        { const el = $('sp-tts-voice');        if (el) el.value = c.tts_voice; }
    if (c.openai_tts_voice) { const el = $('sp-openai-tts-voice'); if (el) el.value = c.openai_tts_voice; }
    if (c.whisper_model)    { const el = $('sp-whisper-size');     if (el) el.value = c.whisper_model; }
    const asrR = document.querySelector(`input[name="asr-engine"][value="${c.asr_engine}"]`);
    if (asrR) asrR.checked = true;
    const ttsR = document.querySelector(`input[name="tts-engine"][value="${c.tts_engine}"]`);
    if (ttsR) ttsR.checked = true;
  }).catch(() => {});
}
function closeSettings() { $('settings-panel').classList.remove('show'); }

$('btn-settings').onclick = openSettings;
$('sp-close').onclick     = closeSettings;
document.querySelector('.sp-backdrop')?.addEventListener('click', closeSettings);

// Tab switching
document.querySelectorAll('.sp-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.sp-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.sp-tab-pane').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const pane = $('tab-' + tab.dataset.tab);
    if (pane) pane.classList.add('active');
  });
});

$('sp-save').onclick = () => {
  const fieldMap = {
    // 国产 LLM
    'sp-qwen-key':        'qwen_api_key',
    'sp-deepseek-key':    'deepseek_api_key',
    'sp-moonshot-key':    'moonshot_api_key',
    'sp-zhipu-key':       'zhipu_api_key',
    'sp-baichuan-key':    'baichuan_api_key',
    'sp-minimax-key':     'minimax_api_key',
    'sp-siliconflow-key': 'siliconflow_api_key',
    // 国外 LLM
    'sp-openai-key':      'openai_api_key',
    'sp-anthropic-key':   'anthropic_api_key',
    'sp-gemini-key':      'gemini_api_key',
    'sp-mistral-key':     'mistral_api_key',
    'sp-groq-key':        'groq_api_key',
    'sp-azure-key':       'azure_tts_key',
  };
  const keys = {};
  for (const [id, field] of Object.entries(fieldMap)) {
    const v = $(id)?.value.trim();
    if (v) keys[field] = v;
  }
  // TTS options
  const ttsVoice = $('sp-tts-voice')?.value;
  if (ttsVoice) keys.tts_voice = ttsVoice;
  const openaiTtsVoice = $('sp-openai-tts-voice')?.value;
  if (openaiTtsVoice) keys.openai_tts_voice = openaiTtsVoice;
  const ollamaTtsModel = $('sp-ollama-tts-model')?.value.trim();
  if (ollamaTtsModel) keys.ollama_tts_model = ollamaTtsModel;
  const azureRegion = $('sp-azure-region')?.value.trim();
  if (azureRegion) keys.azure_tts_region = azureRegion;
  // ASR options
  const ollamaAsrModel = $('sp-ollama-asr-model')?.value.trim();
  if (ollamaAsrModel) keys.ollama_asr_model = ollamaAsrModel;
  const whisperSize = $('sp-whisper-size')?.value;
  if (whisperSize) keys.whisper_model = whisperSize;
  // ASR engine
  const asrEngine = document.querySelector('input[name="asr-engine"]:checked')?.value;
  if (asrEngine) keys.asr_engine = asrEngine;
  // TTS engine
  const ttsEngine = document.querySelector('input[name="tts-engine"]:checked')?.value;
  if (ttsEngine) keys.tts_engine = ttsEngine;
  // LLM settings
  const sp = $('sp-system-prompt')?.value.trim();
  if (sp) keys.system_prompt = sp;
  const ollamaUrl = $('sp-ollama-url')?.value.trim();
  if (ollamaUrl) keys.ollama_url = ollamaUrl;
  const ollamaModel = $('sp-ollama-model')?.value.trim();
  if (ollamaModel) keys.ollama_model = ollamaModel;

  wsSend({ type: 'settings_save', keys });
  $('sp-hint').textContent = '保存中…';
};

function onModelsUpdated(options) {
  const popup = document.getElementById('popup-llm');
  if (!popup) return;
  const title = popup.querySelector('.popup-title');
  popup.innerHTML = '';
  if (title) popup.appendChild(title);
  options.forEach(o => {
    const btn = document.createElement('button');
    btn.className = 'popup-item';
    btn.dataset.model = o.key;
    btn.innerHTML = `<span class="pi-info"><b>${o.label}</b></span>`;
    popup.appendChild(btn);
  });
  bindModelItems();
  showToast('✓ 设置已保存', 'success');
  $('sp-hint').textContent = '✓ 已保存';
  setTimeout(() => { $('sp-hint').textContent = ''; }, 3000);
}

/* ══════════════════════════════════════════
   Chat History persistence (localStorage)
══════════════════════════════════════════ */
function saveHistory() {
  const msgs = $('chat').querySelectorAll('.msg');
  if (!msgs.length) { localStorage.removeItem(HISTORY_KEY); return; }
  const records = [];
  msgs.forEach(row => {
    const isUser = row.classList.contains('user');
    const bubble = row.querySelector('.bubble');
    if (!bubble) return;
    const clone = bubble.cloneNode(true);
    clone.querySelectorAll('.copy-btn,.code-copy').forEach(el => el.remove());
    records.push({
      role: isUser ? 'user' : 'assistant',
      text: clone.innerText.trim(),
      ts:   row.querySelector('.ts')?.textContent || '',
      html: isUser ? null : clone.innerHTML,  // clone 已去除按钮
    });
  });
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(records.slice(-MAX_HISTORY)));
  } catch (e) { /* quota exceeded — skip */ }
}

function loadHistory() {
  let records;
  try { records = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return; }
  if (!records.length) return;

  hideWelcome();
  records.forEach(r => {
    const row = document.createElement('div');
    row.className = 'msg ' + r.role;
    const av = document.createElement('div');
    av.className = 'avatar';
    av.textContent = r.role === 'user' ? '👤' : '🔧';
    const body = document.createElement('div');
    body.className = 'msg-body';
    if (r.role === 'user') body.style.alignItems = 'flex-end';
    const bub = document.createElement('div');
    bub.className = 'bubble';
    if (r.role === 'assistant' && r.html) {
      bub.innerHTML = r.html;
    } else {
      bub.textContent = r.text;
    }
    const cb = document.createElement('button');
    cb.className = 'copy-btn'; cb.textContent = '复制';
    cb.onclick = () => copyText(cb, r.text);
    bub.appendChild(cb);
    const ts = document.createElement('span');
    ts.className = 'ts'; ts.textContent = r.ts + ' (历史)';
    body.appendChild(bub); body.appendChild(ts);
    row.appendChild(av); row.appendChild(body);
    $('chat').appendChild(row);
  });
  const sep = document.createElement('div');
  sep.className = 'info'; sep.textContent = '── 以上为历史记录，当前会话开始 ──';
  $('chat').appendChild(sep);
  scrollBottom();
}

/* ══════════════════════════════════════════
   Export chat
══════════════════════════════════════════ */
function exportChat() {
  const msgs = $('chat').querySelectorAll('.msg');
  if (!msgs.length) { addInfo('没有对话可导出'); return; }
  let md = `# 小新语音助手 对话记录\n\n导出时间：${new Date().toLocaleString('zh-CN')}\n\n---\n\n`;
  msgs.forEach(row => {
    const isUser = row.classList.contains('user');
    const bubble = row.querySelector('.bubble');
    if (!bubble) return;
    const clone = bubble.cloneNode(true);
    clone.querySelectorAll('.copy-btn,.code-copy').forEach(el => el.remove());
    const text = clone.innerText.trim();
    const ts   = row.querySelector('.ts')?.textContent || '';
    md += isUser
      ? `**👤 用户** \`${ts}\`\n\n${text}\n\n`
      : `**🔧 小新** \`${ts}\`\n\n${text}\n\n`;
    md += '---\n\n';
  });
  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `xiaoxin-chat-${new Date().toISOString().slice(0,10)}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ══════════════════════════════════════════
   Welcome screen shortcuts
══════════════════════════════════════════ */
$('chip-ptt')?.addEventListener('click', () => pttStart());
$('chip-space')?.addEventListener('click', () => pttStart());
$('chip-rt')?.addEventListener('click',  enterRealtime);

/* ══════════════════════════════════════════
   Keyboard shortcuts (single listener)
══════════════════════════════════════════ */
function openHelp()  { $('help-panel')?.classList.add('show'); }
function closeHelp() { $('help-panel')?.classList.remove('show'); }

document.addEventListener('DOMContentLoaded', () => {
  $('help-close')?.addEventListener('click', closeHelp);
  document.querySelector('.help-backdrop')?.addEventListener('click', closeHelp);
});

document.addEventListener('keydown', e => {
  const tag = document.activeElement?.tagName;
  const inInput = tag === 'TEXTAREA' || tag === 'INPUT';

  if (e.code === 'Space' && !inInput && !e.repeat) { e.preventDefault(); pttStart(); return; }
  if (e.code === 'Space' && !inInput) return; // already handled above

  if (e.key === '?' && !inInput) { e.preventDefault(); openHelp(); return; }

  if (e.key === 'Escape') {
    if ($('help-panel')?.classList.contains('show'))     { closeHelp(); return; }
    if ($('settings-panel')?.classList.contains('show')) { closeSettings(); return; }
    if (realtimeMode) { rtClose(); return; }
    closeAllPopups(); return;
  }

  if (e.key === 'l' && e.ctrlKey && !inInput) { e.preventDefault(); wsSend({ type: 'clear' }); return; }
  if (e.key === 'e' && e.ctrlKey && !inInput) { e.preventDefault(); exportChat(); return; }
  if (e.key === ',' && e.ctrlKey)             { e.preventDefault(); openSettings(); return; }
});

/* ══════════════════════════════════════════
   Re-send on double-click user message
══════════════════════════════════════════ */
$('chat').addEventListener('dblclick', e => {
  const row = e.target.closest('.msg.user');
  if (!row || isBusy) return;
  const bubble = row.querySelector('.bubble');
  if (!bubble) return;
  const clone = bubble.cloneNode(true);
  clone.querySelectorAll('.copy-btn').forEach(el => el.remove());
  const text = clone.innerText.trim();
  if (!text) return;
  isBusy = true;
  wsSend({ type: 'text', text });
});

/* ══════════════════════════════════════════
   Init
══════════════════════════════════════════ */
bindModelItems();
initScrollBehavior();
loadHistory();
connect();
