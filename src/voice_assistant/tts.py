"""TTS module — 多引擎语音合成

支持引擎 (engine 参数):
  auto        优先 edge → qwen3tts → pyttsx3
  edge        Microsoft Edge-TTS (免费, 需 Node.js)
  qwen3tts    阿里云 Qwen3-TTS (需 DashScope API Key)
  openai_tts  OpenAI TTS API (需 openai_api_key)
  azure_tts   Microsoft Azure TTS (需 azure_tts_key)
  ollama_tts  Ollama 本地 TTS 模型 (如 kokoro, orpheus)
  pyttsx3     系统本地 TTS (无需联网)
"""
from __future__ import annotations
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path


_EDGE_TTS_MOD = Path.home() / ".npm-global/lib/node_modules/openclaw/node_modules/node-edge-tts"

_EDGE_TTS_JS = """
const {EdgeTTS} = require('%s');
const fs = require('fs');
const text  = process.argv[2];
const out   = process.argv[3];
const voice = process.argv[4] || 'zh-CN-XiaoxiaoNeural';
const rate  = process.argv[5] || '+5%%';
const tts   = new EdgeTTS({voice, lang:'zh-CN', rate});
tts.ttsPromise(text, out)
  .then(()=>{ console.log('OK'); process.exit(0); })
  .catch(e=>{ console.error(e.message); process.exit(1); });
""".strip()


class TTS:
    def __init__(
        self,
        engine: str = "auto",
        api_key: str = "",
        voice: str = "zh-CN-XiaoxiaoNeural",
        # OpenAI TTS
        openai_api_key: str = "",
        openai_voice: str = "nova",
        openai_tts_model: str = "tts-1-hd",
        # Azure TTS
        azure_key: str = "",
        azure_region: str = "eastus",
        azure_voice: str = "zh-CN-XiaoxiaoNeural",
        # Ollama TTS
        ollama_url: str = "http://localhost:11434",
        ollama_tts_model: str = "kokoro",
    ):
        self.engine_name     = engine
        self.api_key         = api_key          # DashScope
        self.voice           = voice            # Edge-TTS voice
        self.openai_api_key  = openai_api_key
        self.openai_voice    = openai_voice
        self.openai_tts_model= openai_tts_model
        self.azure_key       = azure_key
        self.azure_region    = azure_region
        self.azure_voice     = azure_voice
        self.ollama_url      = ollama_url.rstrip("/")
        self.ollama_tts_model= ollama_tts_model

        self._engine: str | None = None
        self._lock       = threading.Lock()
        self._pyttsx3    = None
        self._js_path: str | None = None
        self._aborted    = threading.Event()
        self._play_proc: subprocess.Popen | None = None

    # ── Public ────────────────────────────────────────────────────────────
    def speak(self, text: str):
        text = text.strip()
        if not text or self._aborted.is_set():
            return
        with self._lock:
            self._init_engine()
            dispatch = {
                "edge":       self._speak_edge,
                "qwen3tts":   self._speak_qwen3tts,
                "openai_tts": self._speak_openai_tts,
                "azure_tts":  self._speak_azure_tts,
                "ollama_tts": self._speak_ollama_tts,
            }
            fn = dispatch.get(self._engine)
            if fn:
                fn(text)
            else:
                self._speak_pyttsx3(text)

    def abort(self):
        self._aborted.set()
        proc = self._play_proc
        if proc and proc.poll() is None:
            try: proc.kill()
            except: pass

    def reset_abort(self):
        self._aborted.clear()

    # ── Init ──────────────────────────────────────────────────────────────
    def _init_engine(self):
        if self._engine is not None:
            return
        name = self.engine_name

        if name == "openai_tts":
            if not self.openai_api_key:
                raise RuntimeError("openai_api_key required for openai_tts")
            self._engine = "openai_tts"; return

        if name == "azure_tts":
            if not self.azure_key:
                raise RuntimeError("azure_tts_key required for azure_tts")
            self._engine = "azure_tts"; return

        if name == "ollama_tts":
            self._engine = "ollama_tts"; return

        if name in ("auto", "edge"):
            if _EDGE_TTS_MOD.exists():
                self._engine = "edge"; return
            if name == "edge":
                raise RuntimeError("node-edge-tts not available")

        if name in ("auto", "qwen3tts", "cosyvoice") and self.api_key:
            try:
                import dashscope  # noqa
                self._engine = "qwen3tts"; return
            except ImportError:
                if name in ("qwen3tts", "cosyvoice"):
                    raise

        if name in ("auto", "openai_tts") and self.openai_api_key:
            self._engine = "openai_tts"; return

        self._init_pyttsx3()
        self._engine = "pyttsx3"

    # ── Edge TTS ──────────────────────────────────────────────────────────
    def _get_js(self) -> str:
        if self._js_path and Path(self._js_path).exists():
            return self._js_path
        script = _EDGE_TTS_JS % str(_EDGE_TTS_MOD)
        fd, path = tempfile.mkstemp(suffix=".js", prefix="edge_tts_")
        with os.fdopen(fd, "w") as f:
            f.write(script)
        self._js_path = path
        return path

    def _speak_edge(self, text: str, voice: str | None = None, rate: str = "+5%"):
        if self._aborted.is_set(): return
        voice = voice or self.voice or "zh-CN-XiaoxiaoNeural"
        js = self._get_js()
        fd, out_mp3 = tempfile.mkstemp(suffix=".mp3", prefix="tts_")
        os.close(fd)
        try:
            r = subprocess.run(
                ["node", js, text, out_mp3, voice, rate],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode != 0:
                print(f"[TTS] edge-tts error: {r.stderr.strip()}"); return
            if self._aborted.is_set(): return
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", out_mp3],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._play_proc = proc
            proc.wait()
            self._play_proc = None
        finally:
            try: os.unlink(out_mp3)
            except: pass

    # ── Qwen3 TTS ─────────────────────────────────────────────────────────
    def _speak_qwen3tts(self, text: str):
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
            model="qwen3-tts-instruct-flash", voice="Cherry",
            callback=CB(),
            format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
        )
        syn.streaming_call(text); syn.streaming_complete()
        done.wait(timeout=20)
        audio = b"".join(chunks)
        if not audio:
            self._speak_edge(text); return
        self._play_bytes(audio, ".mp3")

    # ── OpenAI TTS ────────────────────────────────────────────────────────
    def _speak_openai_tts(self, text: str):
        """OpenAI /v1/audio/speech endpoint."""
        import httpx
        try:
            r = httpx.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                json={
                    "model": self.openai_tts_model,
                    "input": text,
                    "voice": self.openai_voice,
                    "response_format": "mp3",
                },
                timeout=30,
            )
            r.raise_for_status()
            self._play_bytes(r.content, ".mp3")
        except Exception as e:
            print(f"[TTS] openai_tts error: {e}, fallback edge")
            self._speak_edge(text)

    # ── Azure TTS ─────────────────────────────────────────────────────────
    def _speak_azure_tts(self, text: str):
        """Azure Cognitive Services TTS REST API."""
        import httpx
        token_url = f"https://{self.azure_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        try:
            tok_r = httpx.post(token_url,
                               headers={"Ocp-Apim-Subscription-Key": self.azure_key},
                               timeout=10)
            tok_r.raise_for_status()
            token = tok_r.text

            ssml = (
                f'<speak version="1.0" xml:lang="zh-CN">'
                f'<voice name="{self.azure_voice}">{text}</voice></speak>'
            )
            tts_url = (f"https://{self.azure_region}.tts.speech.microsoft.com"
                       "/cognitiveservices/v1")
            r = httpx.post(
                tts_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3",
                },
                content=ssml.encode("utf-8"),
                timeout=30,
            )
            r.raise_for_status()
            self._play_bytes(r.content, ".mp3")
        except Exception as e:
            print(f"[TTS] azure_tts error: {e}, fallback edge")
            self._speak_edge(text)

    # ── Ollama TTS ────────────────────────────────────────────────────────
    def _speak_ollama_tts(self, text: str):
        """Ollama /api/tts endpoint (kokoro / orpheus / etc.)."""
        import httpx, base64
        try:
            r = httpx.post(
                f"{self.ollama_url}/api/tts",
                json={"model": self.ollama_tts_model, "input": text},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            # 返回 base64 encoded audio
            audio_b64 = data.get("audio") or data.get("data", "")
            if not audio_b64:
                raise ValueError("empty audio response")
            audio_bytes = base64.b64decode(audio_b64)
            self._play_bytes(audio_bytes, ".wav")
        except Exception as e:
            print(f"[TTS] ollama_tts error: {e}, fallback edge")
            self._speak_edge(text)

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

    # ── Util ──────────────────────────────────────────────────────────────
    def _play_bytes(self, data: bytes, suffix: str):
        """Write bytes to tmp file and play with ffplay."""
        if self._aborted.is_set(): return
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", tmp],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._play_proc = proc
            proc.wait()
            self._play_proc = None
        finally:
            try: os.unlink(tmp)
            except: pass
