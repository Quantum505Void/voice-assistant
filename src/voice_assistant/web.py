"""Web UI server for voice assistant — modern refactored version."""
from __future__ import annotations
import asyncio
import base64
import json
import os
import queue
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from voice_assistant import load_config
from voice_assistant.asr import ASR
from voice_assistant.llm import LLMClient
from voice_assistant.tts import TTS

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
STATIC_DIR   = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

# ── Config ────────────────────────────────────────────────────────────────
cfg = load_config()

# ── Model registry ────────────────────────────────────────────────────────
def _build_model_registry(c) -> dict:
    models = {}

    # ── 本地 Ollama ──────────────────────────────────────────────────────
    models["local"] = {
        "label": f"本地 · {c.ollama_model}",
        "base_url": c.ollama_url, "model": c.ollama_model,
        "api_key": "", "backend": "ollama",
    }

    # ── 国产云端 ─────────────────────────────────────────────────────────
    # 通义千问 (Qwen)
    if c.qwen_api_key:
        for mid, label in [
            ("qwen3-235b-a22b",   "Qwen3-235B"),
            ("qwen3-30b-a3b",     "Qwen3-30B"),
            ("qwen3-max",         "Qwen3-Max"),
            ("qwen3.5-plus",      "Qwen3.5-Plus"),
            ("qwen3.5-flash",     "Qwen3.5-Flash"),
            ("qwen-plus",         "Qwen-Plus"),
            ("qwen-turbo",        "Qwen-Turbo"),
        ]:
            models[mid] = {"label": f"🌏 千问 · {label}",
                           "base_url": c.qwen_api_url, "model": mid,
                           "api_key": c.qwen_api_key, "backend": "openai"}

    # DeepSeek
    if c.deepseek_api_key:
        for mid, label in [
            ("deepseek-chat",      "DeepSeek-V3"),
            ("deepseek-reasoner",  "DeepSeek-R1"),
        ]:
            models[mid] = {"label": f"🌏 DeepSeek · {label}",
                           "base_url": "https://api.deepseek.com", "model": mid,
                           "api_key": c.deepseek_api_key, "backend": "openai"}

    # Moonshot / Kimi
    if c.moonshot_api_key:
        for mid, label in [
            ("moonshot-v1-128k", "Kimi-128K"),
            ("moonshot-v1-32k",  "Kimi-32K"),
            ("moonshot-v1-8k",   "Kimi-8K"),
        ]:
            models[mid] = {"label": f"🌏 Kimi · {label}",
                           "base_url": "https://api.moonshot.cn/v1", "model": mid,
                           "api_key": c.moonshot_api_key, "backend": "openai"}

    # 智谱 GLM
    if c.zhipu_api_key:
        for mid, label in [
            ("glm-4-plus",    "GLM-4-Plus"),
            ("glm-4-flash",   "GLM-4-Flash"),
            ("glm-z1-plus",   "GLM-Z1-Plus"),
        ]:
            models[mid] = {"label": f"🌏 智谱 · {label}",
                           "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": mid,
                           "api_key": c.zhipu_api_key, "backend": "openai"}

    # 百川
    if c.baichuan_api_key:
        for mid, label in [
            ("Baichuan4-Air",   "百川4-Air"),
            ("Baichuan4-Turbo", "百川4-Turbo"),
        ]:
            models[mid] = {"label": f"🌏 百川 · {label}",
                           "base_url": "https://api.baichuan-ai.com/v1", "model": mid,
                           "api_key": c.baichuan_api_key, "backend": "openai"}

    # MiniMax
    if c.minimax_api_key:
        for mid, label in [
            ("MiniMax-Text-01", "MiniMax-Text-01"),
            ("abab6.5s-chat",   "ABAB-6.5s"),
        ]:
            models[mid] = {"label": f"🌏 MiniMax · {label}",
                           "base_url": "https://api.minimax.chat/v1", "model": mid,
                           "api_key": c.minimax_api_key, "backend": "openai"}

    # 硅基流动
    if c.siliconflow_api_key:
        for mid, label in [
            ("Qwen/Qwen3-8B",             "Qwen3-8B"),
            ("deepseek-ai/DeepSeek-V3",   "DeepSeek-V3"),
            ("deepseek-ai/DeepSeek-R1",   "DeepSeek-R1"),
            ("THUDM/glm-4-9b-chat",       "GLM-4-9B"),
            ("meta-llama/Llama-3.3-70B-Instruct", "Llama3.3-70B"),
        ]:
            models[mid] = {"label": f"🌏 硅基 · {label}",
                           "base_url": "https://api.siliconflow.cn/v1", "model": mid,
                           "api_key": c.siliconflow_api_key, "backend": "openai"}

    # ── 国外云端 ─────────────────────────────────────────────────────────
    # OpenAI
    if c.openai_api_key:
        for mid, label in [
            ("gpt-4o",         "GPT-4o"),
            ("gpt-4o-mini",    "GPT-4o-mini"),
            ("gpt-4.1",        "GPT-4.1"),
            ("gpt-4.1-mini",   "GPT-4.1-mini"),
            ("o4-mini",        "o4-mini"),
        ]:
            models[mid] = {"label": f"🌐 OpenAI · {label}",
                           "base_url": "https://api.openai.com/v1", "model": mid,
                           "api_key": c.openai_api_key, "backend": "openai"}

    # Anthropic Claude (via OpenRouter openai-compat)
    if c.anthropic_api_key:
        for mid, label in [
            ("claude-opus-4-5",    "Claude Opus 4.5"),
            ("claude-sonnet-4-5",  "Claude Sonnet 4.5"),
            ("claude-haiku-3-5",   "Claude Haiku 3.5"),
        ]:
            models["claude/" + mid] = {"label": f"🌐 Claude · {label}",
                                       "base_url": "https://api.anthropic.com/v1",
                                       "model": mid,
                                       "api_key": c.anthropic_api_key,
                                       "backend": "anthropic"}

    # Gemini (OpenAI-compatible endpoint)
    if c.gemini_api_key:
        for mid, label in [
            ("gemini-2.5-pro",    "Gemini 2.5 Pro"),
            ("gemini-2.5-flash",  "Gemini 2.5 Flash"),
            ("gemini-2.0-flash",  "Gemini 2.0 Flash"),
        ]:
            models["gemini/" + mid] = {
                "label": f"🌐 Gemini · {label}",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "model": mid, "api_key": c.gemini_api_key, "backend": "openai",
            }

    # Mistral
    if c.mistral_api_key:
        for mid, label in [
            ("mistral-large-latest",  "Mistral-Large"),
            ("mistral-small-latest",  "Mistral-Small"),
            ("codestral-latest",      "Codestral"),
        ]:
            models[mid] = {"label": f"🌐 Mistral · {label}",
                           "base_url": "https://api.mistral.ai/v1", "model": mid,
                           "api_key": c.mistral_api_key, "backend": "openai"}

    # Groq
    if c.groq_api_key:
        for mid, label in [
            ("llama-3.3-70b-versatile", "Llama3.3-70B"),
            ("llama-3.1-8b-instant",    "Llama3.1-8B"),
            ("gemma2-9b-it",            "Gemma2-9B"),
            ("qwen-qwq-32b",            "QwQ-32B"),
        ]:
            models[mid] = {"label": f"🌐 Groq · {label}",
                           "base_url": "https://api.groq.com/openai/v1", "model": mid,
                           "api_key": c.groq_api_key, "backend": "openai"}

    return models

MODEL_REGISTRY = _build_model_registry(cfg)
DEFAULT_MODEL  = "local"

# ── Global service singletons ─────────────────────────────────────────────
_asr:        Optional[ASR]    = None
_llm_pool:   dict[str, LLMClient] = {}
_tts_engine: Optional[TTS]   = None
_tts_queue:  Optional[queue.Queue] = None
_tts_thread: Optional[threading.Thread] = None
_tts_enabled: dict[int, bool] = {}   # conn_id → bool
_abort_flags: dict[int, bool] = {}   # conn_id → abort requested

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="小新语音助手")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Service init ──────────────────────────────────────────────────────────
def _tts_worker(q: queue.Queue, tts: TTS):
    while True:
        item = q.get()
        if item is None: break
        try:   tts.speak(item)
        except Exception: pass
        finally: q.task_done()

def _get_asr() -> ASR:
    global _asr
    if _asr is None:
        _asr = ASR(
            model_size       = cfg.whisper_model,
            device           = cfg.device,
            language         = cfg.language,
            api_key          = cfg.asr_api_key,
            engine           = cfg.asr_engine,
            ollama_url       = cfg.ollama_url,
            ollama_asr_model = cfg.ollama_asr_model,
            openai_api_key   = cfg.openai_api_key,
        )
        _asr._load()
    return _asr

def _get_llm(key: str = DEFAULT_MODEL) -> LLMClient:
    if key not in _llm_pool:
        info = MODEL_REGISTRY.get(key, MODEL_REGISTRY[DEFAULT_MODEL])
        _llm_pool[key] = LLMClient(
            base_url=info["base_url"], model=info["model"],
            system_prompt=cfg.system_prompt,
            api_key=info["api_key"], backend=info["backend"],
        )
    return _llm_pool[key]

def _get_tts() -> TTS:
    global _tts_engine, _tts_queue, _tts_thread
    if _tts_engine is None:
        _tts_engine = TTS(
            engine           = cfg.tts_engine,
            api_key          = cfg.tts_api_key,
            voice            = cfg.tts_voice,
            openai_api_key   = cfg.openai_api_key,
            openai_voice     = cfg.openai_tts_voice,
            openai_tts_model = cfg.openai_tts_model,
            azure_key        = cfg.azure_tts_key,
            azure_region     = cfg.azure_tts_region,
            azure_voice      = cfg.azure_tts_voice,
            ollama_url       = cfg.ollama_url,
            ollama_tts_model = cfg.ollama_tts_model,
        )
        _tts_queue = queue.Queue()
        _tts_thread = threading.Thread(
            target=_tts_worker, args=(_tts_queue, _tts_engine), daemon=True)
        _tts_thread.start()
    return _tts_engine

def _enqueue_tts(text: str):
    pass  # replaced by ws-based TTS; kept for compat

def _reset_asr():
    global _asr
    _asr = None

def _abort_tts(cid: int = -1):
    """打断当前 TTS 播放（不销毁引擎）"""
    if _tts_engine:
        _tts_engine.reset_abort()   # 先 reset 让 synthesize 不被旧 abort 卡住
        _tts_engine.abort()
    if cid >= 0:
        _abort_flags[cid] = True

def _reset_tts():
    global _tts_engine, _tts_queue, _tts_thread
    if _tts_engine:
        _tts_engine.abort()
    _tts_engine = None; _tts_queue = None; _tts_thread = None

def strip_markdown(text: str) -> str:
    """清理 markdown 标记和 emoji，返回适合 TTS 朗读的纯文本"""
    # 去除 markdown 格式
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    text = re.sub(r'`+([^`]*)`+', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    # 去除 emoji（只删 emoji 范围，不碰中文）
    text = re.sub(
        r'[\U0001F600-\U0001F64F'   # Emoticons
        r'\U0001F300-\U0001F5FF'    # Misc Symbols and Pictographs
        r'\U0001F680-\U0001F6FF'    # Transport and Map
        r'\U0001F1E0-\U0001F1FF'    # Flags
        r'\U0001F900-\U0001F9FF'    # Supplemental Symbols
        r'\u2702-\u27B0'            # Dingbats
        r'\u2600-\u26FF'            # Misc symbols
        r'\uFE0F\u200D\u20E3]+',    # Variation selectors, ZWJ, combining
        '', text, flags=re.UNICODE
    )
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# ── HTML rendering ────────────────────────────────────────────────────────
def _model_icon(key: str) -> str:
    if key == "local":          return "🖥️"
    if key.startswith("qwen"):  return "🇨🇳"
    if key.startswith("gpt"):   return "🤖"
    if key.startswith("deepseek"): return "🐋"
    if key.startswith("llama") or key.startswith("gemma"): return "⚡"
    if "Qwen" in key or "deepseek-ai" in key: return "🌊"
    return "🧠"

def _model_badge(key: str) -> str:
    if key == "local": return '<span class="pi-badge local">本地</span>'
    if key.startswith("gpt"): return '<span class="pi-badge cloud">OpenAI</span>'
    if key.startswith("deepseek"): return '<span class="pi-badge cloud">DeepSeek</span>'
    if key.startswith("llama") or key.startswith("gemma"): return '<span class="pi-badge free">Groq</span>'
    if "Qwen" in key or "deepseek-ai" in key: return '<span class="pi-badge local">硅基</span>'
    return '<span class="pi-badge cloud">云端</span>'

def _render_html() -> str:
    tmpl = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    # ASR label
    asr_label = "Paraformer" if cfg.asr_engine in ("auto","paraformer") and cfg.asr_api_key \
                else f"Whisper-{cfg.whisper_model}"
    # TTS label
    tts_label = "Edge-TTS" if cfg.tts_engine in ("auto","edge") \
                else ("Qwen3-TTS" if cfg.tts_engine == "qwen3tts" else "本地TTS")
    # model options — rendered as popup-item buttons
    opts = "".join(
        f'<button class="popup-item" data-model="{k}">'
        f'<span class="pi-icon">{_model_icon(k)}</span>'
        f'<span class="pi-info"><b>{v["label"]}</b></span>'
        f'{_model_badge(k)}'
        f'</button>'
        for k, v in MODEL_REGISTRY.items()
    )
    tmpl = tmpl.replace("{{asr_label}}", asr_label)
    tmpl = tmpl.replace("{{tts_label}}", tts_label)
    tmpl = tmpl.replace("{{model_options}}", opts)
    return tmpl

HTML = _render_html()

# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

@app.get("/health")
async def health():
    ok = _get_llm("local").check_connection()
    return {"ollama": ok, "model": cfg.ollama_model, "qwen_available": bool(cfg.qwen_api_key)}

@app.get("/config")
async def get_config():
    """返回当前配置供设置面板预填（API key 只返回脱敏后的值）"""
    def mask(v: str) -> str:
        if not v: return ""
        return v[:4] + "…" + v[-4:] if len(v) > 10 else "****"
    return {
        # LLM keys
        "qwen_api_key":        mask(cfg.qwen_api_key),
        "openai_api_key":      mask(cfg.openai_api_key),
        "deepseek_api_key":    mask(cfg.deepseek_api_key),
        "groq_api_key":        mask(cfg.groq_api_key),
        "siliconflow_api_key": mask(cfg.siliconflow_api_key),
        "moonshot_api_key":    mask(cfg.moonshot_api_key),
        "baichuan_api_key":    mask(cfg.baichuan_api_key),
        "zhipu_api_key":       mask(cfg.zhipu_api_key),
        "minimax_api_key":     mask(cfg.minimax_api_key),
        "anthropic_api_key":   mask(cfg.anthropic_api_key),
        "gemini_api_key":      mask(cfg.gemini_api_key),
        "mistral_api_key":     mask(cfg.mistral_api_key),
        # ASR
        "asr_engine":          cfg.asr_engine,
        "asr_api_key":         mask(cfg.asr_api_key),
        "xunfei_app_id":       mask(cfg.xunfei_app_id),
        "xunfei_api_key":      mask(cfg.xunfei_api_key),
        # TTS
        "tts_engine":          cfg.tts_engine,
        "tts_voice":           cfg.tts_voice,
        "tts_api_key":         mask(cfg.tts_api_key),
        "azure_tts_key":       mask(cfg.azure_tts_key),
        "azure_tts_region":    cfg.azure_tts_region,
        "azure_tts_voice":     cfg.azure_tts_voice,
        "openai_tts_voice":    cfg.openai_tts_voice,
        "openai_tts_model":    cfg.openai_tts_model,
        # Ollama
        "system_prompt":       cfg.system_prompt,
        "ollama_url":          cfg.ollama_url,
        "ollama_model":        cfg.ollama_model,
        "ollama_asr_model":    cfg.ollama_asr_model,
        "ollama_tts_model":    cfg.ollama_tts_model,
        "whisper_model":       cfg.whisper_model,
    }

@app.get("/models")
async def list_models():
    return {k: v["label"] for k, v in MODEL_REGISTRY.items()}

# ── WebSocket ─────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    cid = id(websocket)
    _tts_enabled[cid] = True
    _abort_flags[cid] = False
    asr = _get_asr()
    model_key = DEFAULT_MODEL
    global MODEL_REGISTRY, HTML

    async def send(obj: dict):
        await websocket.send_text(json.dumps(obj, ensure_ascii=False))

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "text":
                text = msg.get("text", "").strip()
                if not text: continue
                _abort_tts(cid)
                _abort_flags[cid] = False
                await send({"type": "user", "text": text})
                await _stream_llm(websocket, _get_llm(model_key), text, _tts_enabled.get(cid, True), cid)

            elif mtype == "abort":
                # 用户主动打断：停 LLM+TTS，通知前端
                _abort_tts(cid)
                _abort_flags[cid] = True
                await send({"type": "tts", "state": "stop"})
                await send({"type": "status", "text": "就绪", "color": "green"})
                isBusy_reset = {"type": "done"}   # 让前端解锁 isBusy
                await send(isBusy_reset)

            elif mtype in ("audio", "audio_rt"):
                try:
                    _abort_tts(cid)
                    _abort_flags[cid] = False
                    text = await _process_audio(asr, msg["data"], msg.get("mime", "audio/webm"))
                    if not text:
                        await send({"type": "error", "text": "未识别到语音，请重试"}); continue
                    if text == "__silent__":
                        await send({"type": "error", "text": "音量太低，请靠近麦克风重试"}); continue
                    await send({"type": "asr", "text": text})
                    await send({"type": "user", "text": text})
                    await _stream_llm(websocket, _get_llm(model_key), text, _tts_enabled.get(cid, True), cid)
                except Exception as e:
                    err_str = str(e)
                    # 不把完整 ffmpeg 堆栈发给前端
                    if "ffmpeg" in err_str or len(err_str) > 100:
                        short = "音频处理失败，请靠近麦克风重试"
                    else:
                        short = err_str[:80]
                    await send({"type": "error", "text": short})

            elif mtype == "tts_toggle":
                _tts_enabled[cid] = msg.get("enabled", True)

            elif mtype == "asr_switch":
                engine = msg.get("engine", "auto")
                cfg.asr_engine = engine
                _reset_asr()
                asr = _get_asr()
                await send({"type": "status", "text": f"ASR 已切换: {engine}", "color": "green"})

            elif mtype == "tts_engine_switch":
                engine = msg.get("engine", "edge")
                cfg.tts_engine = engine
                _reset_tts()
                await send({"type": "status", "text": f"TTS 已切换: {engine}", "color": "green"})

            elif mtype == "model_switch":
                key = msg.get("key", DEFAULT_MODEL)
                if key in MODEL_REGISTRY:
                    model_key = key
                    _get_llm(key).clear_history()
                    await send({"type": "status", "text": f'已切换: {MODEL_REGISTRY[key]["label"]}', "color": "green"})
                    await send({"type": "cleared"})

            elif mtype == "clear":
                _get_llm(model_key).clear_history()
                await send({"type": "cleared"})

            elif mtype == "settings_save":
                import yaml as _yaml
                keys = msg.get("keys", {})
                # 所有可写字段（API keys + 配置）
                writable_fields = (
                    # LLM keys
                    "openai_api_key", "deepseek_api_key", "groq_api_key",
                    "siliconflow_api_key", "qwen_api_key",
                    "moonshot_api_key", "baichuan_api_key", "zhipu_api_key",
                    "minimax_api_key", "anthropic_api_key", "gemini_api_key",
                    "mistral_api_key", "cohere_api_key",
                    # ASR keys
                    "asr_api_key",
                    "xunfei_app_id", "xunfei_api_key", "xunfei_api_secret",
                    # TTS keys
                    "tts_api_key", "azure_tts_key",
                    # Ollama
                    "ollama_url", "ollama_model", "ollama_asr_model", "ollama_tts_model",
                    # OpenAI TTS
                    "openai_tts_voice", "openai_tts_model",
                    # Azure TTS
                    "azure_tts_region", "azure_tts_voice",
                    # Other
                    "system_prompt", "whisper_model",
                )
                for field in writable_fields:
                    if field in keys and keys[field]:
                        setattr(cfg, field, keys[field])

                # ASR engine
                if "asr_engine" in keys and keys["asr_engine"]:
                    cfg.asr_engine = keys["asr_engine"]
                    _reset_asr()
                    asr = _get_asr()
                # TTS engine
                if "tts_engine" in keys and keys["tts_engine"]:
                    cfg.tts_engine = keys["tts_engine"]
                    _reset_tts()
                # TTS voice 热更新
                if "tts_voice" in keys and keys["tts_voice"]:
                    cfg.tts_voice = keys["tts_voice"]
                    if _tts_engine:
                        _tts_engine.voice = keys["tts_voice"]

                MODEL_REGISTRY = _build_model_registry(cfg)
                _llm_pool.clear()
                HTML = _render_html()
                # persist to config.yaml
                config_path = Path(__file__).parent.parent.parent / "config.yaml"
                if config_path.exists():
                    with open(config_path) as _f:
                        _raw = _yaml.safe_load(_f) or {}
                    _raw.update({k: v for k, v in keys.items() if v})
                    with open(config_path, "w") as _f:
                        _yaml.dump(_raw, _f, allow_unicode=True)
                await send({"type": "status", "text": "设置已保存", "color": "green"})
                await send({"type": "models_updated", "options": [
                    {"key": k, "label": v["label"]} for k, v in MODEL_REGISTRY.items()
                ]})

    except WebSocketDisconnect:
        _tts_enabled.pop(cid, None)
        _abort_flags.pop(cid, None)

# ── Audio processing ──────────────────────────────────────────────────────
async def _process_audio(asr: ASR, b64_data: str, mime: str = "audio/webm") -> str:
    import soundfile as sf
    raw_bytes = base64.b64decode(b64_data)

    # 根据 mime 选后缀和 ffmpeg 输入格式
    if "ogg" in mime:
        ext, fmt = ".ogg", "ogg"
    elif "mp4" in mime or "m4a" in mime:
        ext, fmt = ".mp4", "mp4"
    elif "wav" in mime:
        ext, fmt = ".wav", "wav"
    else:
        ext, fmt = ".webm", "matroska"  # audio/webm 或 audio/webm;codecs=opus (Matroska容器)

    print(f"[ASR] recv mime={mime} bytes={len(raw_bytes)} fmt={fmt}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(raw_bytes); tmp_in = f.name
    tmp_out = tmp_in + ".wav"
    try:
        # -f 强制指定输入格式，避免 ffmpeg 按头部猜格式失败
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-f", fmt, "-i", tmp_in,
            "-ar", "16000", "-ac", "1", "-f", "wav", tmp_out,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            # fallback: 不指定 -f，让 ffmpeg 自动探测
            proc2 = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", tmp_in,
                "-ar", "16000", "-ac", "1", "-f", "wav", tmp_out,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr2 = await proc2.communicate()
            if proc2.returncode != 0:
                raise RuntimeError(f"ffmpeg failed (code {proc2.returncode}): {stderr2.decode()[-400:]}")

        import os as _os
        wav_size = _os.path.getsize(tmp_out) if _os.path.exists(tmp_out) else 0
        if wav_size < 100:
            raise RuntimeError(f"ffmpeg produced empty wav ({wav_size} bytes)")

        audio_np, sr = sf.read(tmp_out, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        rms = float((audio_np**2).mean()**.5)
        print(f"[ASR] mime={mime} ext={ext} wav={wav_size}B samples={len(audio_np)} rms={rms:.4f}")

        if rms < 0.005:
            print(f"[ASR] rms too low ({rms:.4f}), skipping")
            return "__silent__"

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, asr.transcribe, audio_np)
        result = (result or "").strip()
        print(f"[ASR] result={repr(result)}")
        return result
    finally:
        for p in (tmp_in, tmp_out):
            try: os.unlink(p)
            except: pass

# ── LLM streaming ─────────────────────────────────────────────────────────
async def _stream_llm(websocket: WebSocket, llm: LLMClient, text: str, tts_on: bool, cid: int = -1):
    async def send(obj):
        await websocket.send_text(json.dumps(obj, ensure_ascii=False))

    def is_aborted():
        return _abort_flags.get(cid, False)

    # 新一轮对话，重置打断标志
    _get_tts()   # 确保 _tts_engine 已初始化
    if _tts_engine:
        _tts_engine.reset_abort()

    await send({"type": "status", "text": "思考中...", "color": "yellow"})
    if tts_on:
        await send({"type": "tts", "state": "start"})

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def run():
        def on_tok(tok): loop.call_soon_threadsafe(q.put_nowait, tok)
        try:
            result = llm.chat_stream(text, on_token=on_tok)
        except Exception as e:
            print(f"[LLM] stream error: {e}")
            loop.call_soon_threadsafe(q.put_nowait, f"\n[错误: {e}]")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=run, daemon=True).start()

    buf = ""
    sentence_idx = 0
    full_reply = ""
    while True:
        tok = await q.get()
        if tok is None: break
        if is_aborted(): break          # LLM token 生产中被打断
        await send({"type": "token", "text": tok})
        full_reply += tok
        if tts_on:
            buf += tok
            if re.search(r'[。！？\.!?\n]', buf):
                s = strip_markdown(buf)
                if s:
                    if is_aborted(): break
                    if sentence_idx == 0:
                        await send({"type": "tts", "state": "sentence_start"})
                    print(f"[TTS] → {repr(s[:40])}", flush=True)
                    await _send_tts_chunk(websocket, s, cid)
                    sentence_idx += 1
                buf = ""

    tail = strip_markdown(buf)
    if tts_on and tail:
        if not is_aborted():
            print(f"[TTS] tail → {repr(tail[:40])}", flush=True)
            await _send_tts_chunk(websocket, tail, cid)

    await send({"type": "done"})
    if tts_on:
        await send({"type": "tts", "state": "stop"})
    await send({"type": "tts_done"})


async def _send_tts_chunk(websocket: WebSocket, text: str, cid: int = -1):
    """Synthesize text and stream audio chunks to browser via WebSocket."""
    _get_tts()   # 确保 engine 已初始化
    if not _tts_engine or not text.strip():
        return
    if _abort_flags.get(cid, False):
        return
    # 确保 _engine 已设置（lazy init）
    if _tts_engine._engine is None:
        try:
            _tts_engine._init_engine()
        except Exception as e:
            print(f"[TTS] init engine failed: {e}")
            return
    engine = _tts_engine._engine
    print(f"[TTS] engine={engine} → {repr(text[:30])}", flush=True)

    try:
        if engine == "edge":
            await _send_tts_edge_stream(websocket, text, cid)
        else:
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(None, _tts_engine.synthesize, text)
            if audio_bytes and not _abort_flags.get(cid, False):
                b64 = base64.b64encode(audio_bytes).decode()
                await websocket.send_text(json.dumps(
                    {"type": "tts_audio", "data": b64}, ensure_ascii=False))
    except Exception as e:
        print(f"[TTS] send chunk error: {e}")


async def _send_tts_edge_stream(websocket: WebSocket, text: str, cid: int = -1):
    """edge_tts 流式：合成到一定大小就发一帧，降低首帧延迟"""
    import edge_tts
    voice = getattr(_tts_engine, 'voice', None) or "zh-CN-XiaoxiaoNeural"
    communicate = edge_tts.Communicate(text, voice=voice, rate="+5%")
    buf = bytearray()
    CHUNK_SIZE = 8 * 1024  # 8KB 发一次，约 0.5s 音频
    async for chunk in communicate.stream():
        if _abort_flags.get(cid, False):
            return
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
            if len(buf) >= CHUNK_SIZE:
                b64 = base64.b64encode(bytes(buf)).decode()
                await websocket.send_text(json.dumps(
                    {"type": "tts_audio", "data": b64}, ensure_ascii=False))
                buf.clear()
    if buf and not _abort_flags.get(cid, False):
        b64 = base64.b64encode(bytes(buf)).decode()
        await websocket.send_text(json.dumps(
            {"type": "tts_audio", "data": b64}, ensure_ascii=False))
