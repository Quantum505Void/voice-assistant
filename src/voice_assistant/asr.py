"""ASR module — 多引擎语音识别

支持引擎 (engine 参数):
  auto            优先 paraformer → whisper
  paraformer      DashScope Paraformer-realtime-v2 (中文最准)
  whisper         本地 faster-whisper
  ollama          Ollama 本地 ASR 模型 (如 whisper)
  openai_whisper  OpenAI Whisper API (需 openai_api_key)
"""
from __future__ import annotations
import io
import os
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional


class ASR:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        language: str = "zh",
        api_key: str = "",
        engine: str = "auto",
        # 额外参数
        ollama_url: str = "http://localhost:11434",
        ollama_asr_model: str = "whisper",
        openai_api_key: str = "",
    ):
        self.model_size      = model_size
        self.device          = device
        self.language        = language if language != "auto" else None
        self.api_key         = api_key       # DashScope
        self.engine_name     = engine
        self.ollama_url      = ollama_url.rstrip("/")
        self.ollama_asr_model= ollama_asr_model
        self.openai_api_key  = openai_api_key
        self._engine: str | None = None
        self._whisper        = None

    # ── Public ────────────────────────────────────────────────────────────
    def _load(self):
        self._init_engine()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 16kHz mono array → text."""
        self._init_engine()
        dispatch = {
            "paraformer":     self._transcribe_paraformer,
            "ollama":         self._transcribe_ollama,
            "openai_whisper": self._transcribe_openai_whisper,
        }
        fn = dispatch.get(self._engine)
        if fn:
            return fn(audio)
        return self._transcribe_whisper(audio)

    # ── Init ──────────────────────────────────────────────────────────────
    def _init_engine(self):
        if self._engine is not None:
            return

        name = self.engine_name

        # 显式指定引擎
        if name == "ollama":
            self._engine = "ollama"; return
        if name == "openai_whisper":
            if not self.openai_api_key:
                raise RuntimeError("openai_api_key required for openai_whisper ASR")
            self._engine = "openai_whisper"; return
        if name == "whisper":
            self._load_whisper(); self._engine = "whisper"; return

        # paraformer / auto
        if name in ("auto", "paraformer") and self.api_key:
            try:
                import dashscope
                dashscope.api_key = self.api_key
                from dashscope.audio.asr import Recognition  # noqa
                self._engine = "paraformer"; return
            except ImportError:
                if name == "paraformer":
                    raise RuntimeError("dashscope not installed")

        # fallback → faster-whisper
        self._load_whisper()
        self._engine = "whisper"

    def _load_whisper(self):
        if self._whisper is not None:
            return
        from faster_whisper import WhisperModel
        compute = "float16" if self.device == "cuda" else "int8"
        self._whisper = WhisperModel(
            self.model_size, device=self.device, compute_type=compute)

    # ── Paraformer ────────────────────────────────────────────────────────
    def _transcribe_paraformer(self, audio: np.ndarray) -> str:
        from dashscope.audio.asr import Recognition
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                sf.write(f, audio, 16000, format="WAV", subtype="PCM_16")
            result = Recognition(
                model="paraformer-realtime-v2", format="wav",
                sample_rate=16000, callback=None,
                language_hints=["zh", "en"],
            ).call(tmp_wav)
        finally:
            try: os.unlink(tmp_wav)
            except: pass

        if result and result.get("status_code") == 200:
            sentences = result.get("output", {}).get("sentence", [])
            return "".join(s.get("text", "") for s in sentences).strip()

        reason = result.get("message") if result else "None"
        print(f"[ASR] paraformer failed ({reason}), fallback whisper")
        self._load_whisper()
        return self._transcribe_whisper(audio)

    # ── Ollama ASR ────────────────────────────────────────────────────────
    def _transcribe_ollama(self, audio: np.ndarray) -> str:
        """Call Ollama /api/audio/transcriptions (whisper model)."""
        import httpx, base64
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                sf.write(f, audio, 16000, format="WAV", subtype="PCM_16")
            with open(tmp_wav, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
        finally:
            try: os.unlink(tmp_wav)
            except: pass

        try:
            r = httpx.post(
                f"{self.ollama_url}/api/audio/transcriptions",
                json={
                    "model": self.ollama_asr_model,
                    "audio": audio_b64,
                    "language": self.language or "zh",
                },
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("text", "").strip()
        except Exception as e:
            print(f"[ASR] ollama asr error: {e}, fallback whisper")
            self._load_whisper()
            return self._transcribe_whisper(audio)

    # ── OpenAI Whisper API ─────────────────────────────────────────────────
    def _transcribe_openai_whisper(self, audio: np.ndarray) -> str:
        """OpenAI /v1/audio/transcriptions endpoint."""
        import httpx
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                sf.write(f, audio, 16000, format="WAV", subtype="PCM_16")
            with open(tmp_wav, "rb") as f:
                wav_bytes = f.read()
        finally:
            try: os.unlink(tmp_wav)
            except: pass

        try:
            r = httpx.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": "whisper-1",
                      "language": self.language or "zh",
                      "response_format": "text"},
                timeout=30,
            )
            r.raise_for_status()
            return r.text.strip()
        except Exception as e:
            print(f"[ASR] openai whisper error: {e}, fallback local whisper")
            self._load_whisper()
            return self._transcribe_whisper(audio)

    # ── Local Whisper ─────────────────────────────────────────────────────
    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        self._load_whisper()
        segments, _ = self._whisper.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            temperature=0,
            condition_on_previous_text=False,
            initial_prompt="以下是普通话内容：",
        )
        return "".join(seg.text for seg in segments).strip()
