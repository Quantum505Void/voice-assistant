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
    asr_engine: str = "auto"    # auto / paraformer / whisper
    asr_api_key: str = ""       # DashScope key for Paraformer ASR
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # 远程千问
    qwen_api_key: str = ""
    qwen_api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"
    # TTS
    tts_engine: str = "auto"
    tts_api_key: str = ""   # DashScope key for CosyVoice TTS
    vad_aggressiveness: int = 2
    silence_duration: float = 1.2
    sample_rate: int = 16000
    system_prompt: str = "你是一个聪明友好的语音助手，叫小新。回答简洁自然，适合语音播报。"
    # Multi-provider LLM API keys
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    groq_api_key: str = ""
    siliconflow_api_key: str = ""
    # TTS voice
    tts_voice: str = "zh-CN-XiaoxiaoNeural"

    model_config = {"env_prefix": "VA_"}


def load_config() -> Config:
    data = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return Config(**data)
