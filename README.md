# 小新语音助手 🔧

一个现代化的本地语音助手，支持多模型、多引擎，带精美 Web UI。

![UI Screenshot](docs/screenshot.png)

## 特性

- 🎤 **双输入模式**：PTT 按住录音 / Space 快捷键 + 实时对话模式
- 🔊 **多 TTS 引擎**：Edge TTS（免费）/ Qwen3-TTS / pyttsx3 本地
- 🧠 **多 LLM 支持**：本地 Ollama + 通义千问 / OpenAI / DeepSeek / Groq / 硅基流动
- 🎙 **ASR 识别**：Paraformer（DashScope）/ Whisper（本地）
- ⚙️ **设置面板**：界面内配置 API Key，无需重启
- 💬 **流式输出**：Token 级 streaming，实时显示
- 🌗 **现代 UI**：Aurora 背景、glassmorphism、neon 效果

## 快速开始

### 1. 安装依赖

```bash
# 需要 Python 3.10+、uv、Node.js（Edge TTS）
pip install uv
uv sync
npm install -g node-edge-tts   # Edge TTS
```

### 2. 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入你的 API Key
```

或者启动后在 ⚙️ 设置面板里直接填写。

### 3. 启动

```bash
uv run voice-assistant
# 访问 http://localhost:8765
```

## 支持的 LLM

| Provider | 模型 | 需要 |
|---|---|---|
| 本地 | Ollama（任意模型）| 安装 Ollama |
| 通义千问 | qwen3-max, qwen3.5-plus, qwen3.5-flash | DashScope API Key |
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini | OpenAI API Key |
| DeepSeek | deepseek-chat (V3), deepseek-reasoner (R1) | DeepSeek API Key |
| Groq | llama-3.3-70b, gemma2-9b | Groq API Key |
| 硅基流动 | Qwen3-8B, DeepSeek-V3 | SiliconFlow API Key |

## 支持的 ASR

| 引擎 | 说明 |
|---|---|
| Paraformer | DashScope，高精度中文识别，需要 API Key |
| Whisper | 本地运行，无需联网 |

## 支持的 TTS

| 引擎 | 说明 |
|---|---|
| Edge TTS | 微软 Edge 云端 TTS，免费，多声音 |
| Qwen3-TTS | 阿里云高质量 TTS，需要 API Key |
| pyttsx3 | 系统本地 TTS，无需联网 |

## 键盘快捷键

| 按键 | 功能 |
|---|---|
| `Space` | 按住 PTT 录音 |
| `Enter` | 发送文字 |
| `Shift+Enter` | 文字换行 |

## 项目结构

```
src/voice_assistant/
├── web.py          # FastAPI 服务 + WebSocket
├── asr.py          # 语音识别（Paraformer / Whisper）
├── llm.py          # LLM 客户端（流式）
├── tts.py          # TTS 引擎（Edge / Qwen3 / pyttsx3）
├── templates/
│   └── index.html  # Web UI 模板
└── static/
    ├── style.css   # 样式
    └── app.js      # 前端逻辑
```

## License

MIT
