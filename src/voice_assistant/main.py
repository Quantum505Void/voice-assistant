"""Main entry point for voice assistant."""
from __future__ import annotations
import signal
import sys
import threading
import re

from voice_assistant import load_config
from voice_assistant.asr import ASR
from voice_assistant.llm import LLMClient
from voice_assistant.tts import TTS
from voice_assistant.audio import AudioRecorder
from voice_assistant.ui import UI, console


def strip_markdown(text: str) -> str:
    """Remove markdown symbols unsuitable for TTS."""
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    text = re.sub(r'`+([^`]*)`+', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text.strip()


def main():
    cfg = load_config()
    ui = UI()
    ui.print_banner()

    # Check Ollama
    ui.show_info(f"正在连接 Ollama ({cfg.ollama_url})...")
    llm = LLMClient(cfg.ollama_url, cfg.ollama_model, cfg.system_prompt)
    if not llm.check_connection():
        ui.show_error(f"无法连接 Ollama，请先运行: ollama serve")
        sys.exit(1)
    ui.show_info(f"✅ Ollama 已连接，模型: {cfg.ollama_model}")

    # Load ASR
    ui.show_info(f"加载 Whisper ({cfg.whisper_model}) ...")
    asr = ASR(cfg.whisper_model, cfg.device, cfg.language)
    asr._load()
    ui.show_info("✅ Whisper 已加载")

    # Init TTS
    tts = TTS(cfg.tts_engine)
    audio = AudioRecorder(cfg.sample_rate, cfg.vad_aggressiveness, cfg.silence_duration)

    def handle_exit(sig, frame):
        console.print("\n\n[dim]再见！[/dim]")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    console.print("\n[bold]准备好了！开始说话吧 👂[/bold]\n")

    while True:
        try:
            # Record
            ui.set_state("listening")
            audio_data = audio.record_until_silence()

            if audio_data is None or len(audio_data) < 1000:
                continue

            # ASR
            ui.set_state("thinking")
            text = asr.transcribe(audio_data)
            if not text or len(text.strip()) < 2:
                ui.show_info("（未识别到有效语音，继续聆听）")
                continue

            ui.show_user(text)

            # LLM stream + TTS pipeline
            ui.show_assistant_start()
            sentence_buf = ""
            full_response = ""

            def on_token(token: str):
                nonlocal sentence_buf, full_response
                sentence_buf += token
                full_response += token
                ui.show_token(token)
                # 凑够一句话就 TTS
                if re.search(r'[。！？\.!?\n]', sentence_buf):
                    to_speak = strip_markdown(sentence_buf)
                    sentence_buf = ""
                    if to_speak:
                        tts.speak(to_speak)

            ui.set_state("speaking")
            try:
                llm.chat_stream(text, on_token=on_token)
            except Exception as e:
                ui.show_error(f"LLM 错误: {e}")
                continue

            ui.show_assistant_end()

            # Speak remaining buffer
            if sentence_buf.strip():
                tts.speak(strip_markdown(sentence_buf))

            ui.set_state("idle")

        except KeyboardInterrupt:
            handle_exit(None, None)
        except Exception as e:
            ui.show_error(str(e))
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
