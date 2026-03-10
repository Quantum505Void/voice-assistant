"""ASR module — Paraformer (DashScope) → faster-whisper fallback.

Priority (auto):
  1. paraformer-realtime-v2 via DashScope (中文最准, 低延迟, 已有 key)
  2. faster-whisper small/base (本地, 无网络依赖)
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
        engine: str = "auto",   # auto / paraformer / whisper
    ):
        self.model_size = model_size
        self.device     = device
        self.language   = language if language != "auto" else None
        self.api_key    = api_key
        self.engine_name = engine
        self._engine: str | None = None
        self._whisper   = None

    # ── Public ────────────────────────────────────────────────────────────
    def _load(self):
        """Eagerly initialize engine (called from web.py at startup)."""
        self._init_engine()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 16kHz mono array → text."""
        self._init_engine()
        if self._engine == "paraformer":
            return self._transcribe_paraformer(audio)
        return self._transcribe_whisper(audio)

    # ── Init ──────────────────────────────────────────────────────────────
    def _init_engine(self):
        if self._engine is not None:
            return

        if self.engine_name in ("auto", "paraformer") and self.api_key:
            try:
                import dashscope
                dashscope.api_key = self.api_key
                from dashscope.audio.asr import Recognition  # noqa
                self._engine = "paraformer"
                self._dashscope = dashscope
                return
            except ImportError:
                if self.engine_name == "paraformer":
                    raise RuntimeError("dashscope not installed")

        # fallback → faster-whisper
        self._load_whisper()
        self._engine = "whisper"

    def _load_whisper(self):
        if self._whisper is not None:
            return
        from faster_whisper import WhisperModel
        compute = "float16" if self.device == "cuda" else "int8"
        self._whisper = WhisperModel(self.model_size, device=self.device, compute_type=compute)

    # ── Paraformer ────────────────────────────────────────────────────────
    def _transcribe_paraformer(self, audio: np.ndarray) -> str:
        from dashscope.audio.asr import Recognition

        # write to tmp wav
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                sf.write(f, audio, 16000, format="WAV", subtype="PCM_16")
            result = Recognition(
                model       = "paraformer-realtime-v2",
                format      = "wav",
                sample_rate = 16000,
                callback    = None,
                language_hints=["zh", "en"],
            ).call(tmp_wav)
        finally:
            try: os.unlink(tmp_wav)
            except: pass

        if result.get("status_code") == 200:
            sentences = result.get("output", {}).get("sentence", [])
            return "".join(s.get("text", "") for s in sentences).strip()

        # quota/network error → fallback to whisper
        print(f"[ASR] paraformer failed ({result.get('message')}), fallback whisper")
        self._load_whisper()
        return self._transcribe_whisper(audio)

    # ── Whisper ───────────────────────────────────────────────────────────
    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        self._load_whisper()
        segments, _ = self._whisper.transcribe(
            audio,
            language    = self.language,
            beam_size   = 5,
            vad_filter  = True,
            temperature = 0,          # deterministic, 避免幻觉
            condition_on_previous_text = False,
            initial_prompt = "以下是普通话内容：",  # 提示中文
        )
        return "".join(seg.text for seg in segments).strip()
