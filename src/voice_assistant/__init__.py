"""Voice Assistant - Config loader."""
from pathlib import Path
import yaml
from pydantic_settings import BaseSettings
from pydantic import Field


CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


class Config(BaseSettings):
    whisper_model: str = "small"
    device: str = "cpu"
    language: str = "zh"
    asr_engine: str = "auto"    # auto / paraformer / whisper / ollama / openai_whisper / funasr
    asr_api_key: str = ""       # DashScope key for Paraformer ASR
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_asr_model: str = "whisper"    # Ollama ASR model (e.g. whisper)
    ollama_tts_model: str = ""           # Ollama TTS model (e.g. kokoro, orpheus)
    # TTS
    tts_engine: str = "auto"   # auto / edge / qwen3tts / openai_tts / azure_tts / ollama_tts / pyttsx3
    tts_api_key: str = ""      # DashScope key for Qwen3 TTS
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    # OpenAI TTS
    openai_tts_voice: str = "nova"        # alloy/echo/fable/onyx/nova/shimmer
    openai_tts_model: str = "tts-1-hd"
    # Azure TTS
    azure_tts_key: str = ""
    azure_tts_region: str = "eastus"
    azure_tts_voice: str = "zh-CN-XiaoxiaoNeural"
    # 讯飞 FunASR / TTS
    xunfei_app_id: str = ""
    xunfei_api_key: str = ""
    xunfei_api_secret: str = ""
    # VAD
    vad_aggressiveness: int = 2
    silence_duration: float = 1.2
    sample_rate: int = 16000
    system_prompt: str = "你是一个聪明友好的语音助手，叫小新。回答简洁自然，适合语音播报。"
    # ── LLM API Keys ────────────────────────────────────────────
    qwen_api_key: str = ""
    qwen_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    groq_api_key: str = ""
    siliconflow_api_key: str = ""
    # 国产
    moonshot_api_key: str = ""    # Kimi
    baichuan_api_key: str = ""    # 百川
    zhipu_api_key: str = ""       # 智谱 GLM
    minimax_api_key: str = ""     # MiniMax
    # 国外
    anthropic_api_key: str = ""   # Claude
    gemini_api_key: str = ""      # Gemini (via OpenAI compat)
    mistral_api_key: str = ""     # Mistral
    cohere_api_key: str = ""      # Cohere

    model_config = {"env_prefix": "VA_", "extra": "ignore"}


def load_config() -> Config:
    data = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return Config(**data)
