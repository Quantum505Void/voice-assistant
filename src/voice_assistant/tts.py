"""TTS module — edge-tts (node) → CosyVoice/Qwen3 → pyttsx3 fallback.

Hierarchy (auto mode):
  1. edge-tts via node-edge-tts (OpenClaw 自带, XiaoxiaoNeural, 无 API key)
  2. Qwen3-TTS (dashscope, 如果有 api_key)
  3. pyttsx3 → espeak-ng (最差, 仅兜底)
"""
from __future__ import annotations
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path


# node-edge-tts 随 OpenClaw 安装的路径
_EDGE_TTS_MOD = Path.home() / ".npm-global/lib/node_modules/openclaw/node_modules/node-edge-tts"

_EDGE_TTS_JS = """
const {EdgeTTS} = require('%s');
const fs = require('fs');
const text = process.argv[2];
const out  = process.argv[3];
const voice = process.argv[4] || 'zh-CN-XiaoxiaoNeural';
const rate  = process.argv[5] || '+5%%';
const tts = new EdgeTTS({voice, lang:'zh-CN', rate});
tts.ttsPromise(text, out)
  .then(()=>{ console.log('OK'); process.exit(0); })
  .catch(e=>{ console.error(e.message); process.exit(1); });
""".strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[。！？\.!?\n])', text)
    return [p.strip() for p in parts if p.strip()]


class TTS:
    def __init__(self, engine: str = "auto", api_key: str = ""):
        self.engine_name = engine
        self.api_key     = api_key
        self._engine: str | None = None
        self._lock       = threading.Lock()
        self._pyttsx3    = None
        self._js_path: str | None = None  # temp .js file for edge-tts
        self._aborted    = threading.Event()   # 打断标志
        self._play_proc: subprocess.Popen | None = None  # 当前播放进程

    # ── Public ────────────────────────────────────────────────────────────
    def speak(self, text: str):
        text = text.strip()
        if not text or self._aborted.is_set():
            return
        with self._lock:
            self._init_engine()
            if self._engine == "edge":
                self._speak_edge(text)
            elif self._engine == "qwen3tts":
                self._speak_qwen3tts(text)
            else:
                self._speak_pyttsx3(text)

    def abort(self):
        """打断当前 TTS 播放"""
        self._aborted.set()
        proc = self._play_proc
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    def reset_abort(self):
        """新一轮对话开始前重置打断标志"""
        self._aborted.clear()

    # ── Init ──────────────────────────────────────────────────────────────
    def _init_engine(self):
        if self._engine is not None:
            return

        if self.engine_name in ("auto", "edge"):
            if self._can_edge():
                self._engine = "edge"
                return
            if self.engine_name == "edge":
                raise RuntimeError("node-edge-tts not available")

        if self.engine_name in ("auto", "qwen3tts", "cosyvoice") and self.api_key:
            try:
                import dashscope  # noqa
                self._engine = "qwen3tts"
                return
            except ImportError:
                if self.engine_name in ("qwen3tts", "cosyvoice"):
                    raise

        self._init_pyttsx3()
        self._engine = "pyttsx3"

    def _can_edge(self) -> bool:
        return _EDGE_TTS_MOD.exists()

    # ── Edge TTS (node) ───────────────────────────────────────────────────
    def _get_js(self) -> str:
        if self._js_path and Path(self._js_path).exists():
            return self._js_path
        script = _EDGE_TTS_JS % str(_EDGE_TTS_MOD)
        fd, path = tempfile.mkstemp(suffix=".js", prefix="edge_tts_")
        with os.fdopen(fd, "w") as f:
            f.write(script)
        self._js_path = path
        return path

    def _speak_edge(self, text: str, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+5%"):
        if self._aborted.is_set():
            return
        js = self._get_js()
        fd, out_mp3 = tempfile.mkstemp(suffix=".mp3", prefix="tts_")
        os.close(fd)
        try:
            result = subprocess.run(
                ["node", js, text, out_mp3, voice, rate],
                capture_output=True, text=True, timeout=20,
            )
            if result.returncode != 0:
                print(f"[TTS] edge-tts error: {result.stderr.strip()}")
                return
            if self._aborted.is_set():
                return
            # play — 用 Popen 支持 abort kill
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", out_mp3],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._play_proc = proc
            proc.wait()
            self._play_proc = None
        finally:
            try:
                os.unlink(out_mp3)
            except OSError:
                pass
                pass

    # ── Qwen3 TTS ─────────────────────────────────────────────────────────
    def _speak_qwen3tts(self, text: str):
        """Use qwen3-tts-instruct-flash via dashscope streaming."""
        import dashscope
        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
        dashscope.api_key = self.api_key

        chunks: list[bytes] = []
        done = threading.Event()

        class CB:
            def on_open(self): pass
            def on_data(s, data, *_): chunks.append(data)
            def on_error(s, e): print(f"[TTS] qwen3tts err: {e}")
            def on_close(s): done.set()
            def on_complete(s): done.set()
            def on_event(s, e): pass

        syn = SpeechSynthesizer(
            model    = "qwen3-tts-instruct-flash",
            voice    = "Cherry",
            callback = CB(),
            format   = AudioFormat.MP3_22050HZ_MONO_256KBPS,
        )
        syn.streaming_call(text)
        syn.streaming_complete()
        done.wait(timeout=20)

        audio = b"".join(chunks)
        if not audio:
            self._speak_edge(text)  # fallback
            return

        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        with os.fdopen(fd, "wb") as f:
            f.write(audio)
        try:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp], capture_output=True, timeout=60)
        finally:
            try: os.unlink(tmp)
            except: pass

    # ── pyttsx3 ───────────────────────────────────────────────────────────
    def _init_pyttsx3(self):
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", 175)
        for v in eng.getProperty("voices"):
            if "zh" in v.id.lower() or "chinese" in v.name.lower():
                eng.setProperty("voice", v.id)
                break
        self._pyttsx3 = eng

    def _speak_pyttsx3(self, text: str):
        self._pyttsx3.say(text)
        self._pyttsx3.runAndWait()
