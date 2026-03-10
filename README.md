# 小新语音助手 🔧

> 现代化本地语音 AI 助手 · 多模型 · 多引擎 · 精美 Web UI

一个运行在本机的语音助手，支持实时对话、多 LLM 切换、多 ASR/TTS 引擎，带 Aurora 风格 Web 界面。

---

## ✨ 功能亮点

- 🎤 **双录音模式**：PTT 按住录音（鼠标 / Space 键）+ 实时对话（VAD 自动检测停顿）
- 🧠 **多 LLM**：Ollama 本地 + 通义千问 / OpenAI / DeepSeek / Groq / 硅基流动，界面内一键切换
- 🎙 **多 ASR 引擎**：Paraformer（DashScope 云端高精度）/ Whisper（本地离线）
- 🔊 **多 TTS 引擎**：Edge TTS 免费多声音 / Qwen3-TTS 高表现力 / pyttsx3 完全离线
- 💬 **流式输出**：Token 级 streaming，Markdown 渲染 + 代码高亮
- 💾 **对话持久化**：localStorage 保存历史，刷新不丢失，支持导出 Markdown
- ⚙️ **设置面板**：界面内填写 API Key，热切换引擎，无需重启
- 🌌 **现代 UI**：Aurora 动态背景 · glassmorphism · 实时音频波形可视化

---

## 🚀 快速开始

### 依赖

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（Python 包管理）
- Node.js（Edge TTS 所需）
- [Ollama](https://ollama.com)（本地模型）+ `ollama pull qwen2.5:7b`
- ffmpeg + ffplay（音频播放）

### 安装

```bash
git clone https://github.com/Quantum505Void/voice-assistant.git
cd voice-assistant
uv sync
```

### 配置

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，或启动后在 ⚙️ 设置面板里填写（推荐）。

### 启动

```bash
# 前台运行
uv run python -m uvicorn src.voice_assistant.web:app --host 0.0.0.0 --port 8765

# 后台运行
nohup uv run python -m uvicorn src.voice_assistant.web:app --host 0.0.0.0 --port 8765 > /tmp/va.log 2>&1 &

# 访问
open http://localhost:8765
```

---

## 🧠 支持的 LLM

| Provider | 模型 | 需要 |
|---|---|---|
| 本地 Ollama | 任意模型（默认 qwen2.5:7b）| 安装 Ollama |
| 通义千问 | qwen3-max · qwen3.5-plus · qwen3.5-flash · qwen-plus · qwen-turbo | DashScope API Key |
| OpenAI | gpt-4o · gpt-4o-mini · gpt-4.1 · gpt-4.1-mini | OpenAI API Key |
| DeepSeek | deepseek-chat (V3) · deepseek-reasoner (R1) | DeepSeek API Key |
| Groq | llama-3.3-70b · gemma2-9b | Groq API Key（免费） |
| 硅基流动 | Qwen3-8B · DeepSeek-V3 | SiliconFlow API Key |

## 🎙 支持的 ASR

| 引擎 | 特点 | 需要 |
|---|---|---|
| Paraformer | DashScope 云端 · 高精度中文 · 低延迟 | DashScope API Key |
| Whisper | 本地离线 · 多语言 · 无需联网 | 首次自动下载模型 |

## 🔊 支持的 TTS

| 引擎 | 特点 | 需要 |
|---|---|---|
| Edge TTS | 微软云端 · 免费 · 多声音（晓晓/云希等）| Node.js（随 OpenClaw 内置）|
| Qwen3-TTS | 阿里云 · 高表现力 · 情感自然 | DashScope API Key |
| pyttsx3 | 系统本地 · 完全离线 | 无 |

---

## ⌨️ 快捷键

| 按键 | 功能 |
|---|---|
| `Space` | 按住 PTT 录音 |
| `Enter` | 发送消息 |
| `Shift+Enter` | 换行 |
| `?` | 显示快捷键帮助 |
| `Esc` | 关闭弹窗 / 退出实时模式 |
| `Ctrl+L` | 清空对话 |
| `Ctrl+E` | 导出对话 Markdown |
| `Ctrl+,` | 打开设置面板 |

---

## 📁 项目结构

```
voice-assistant/
├── config.yaml              # 运行配置（gitignore，含 API key）
├── config.example.yaml      # 配置模板
├── pyproject.toml           # uv 项目依赖
└── src/voice_assistant/
    ├── web.py               # FastAPI 服务 + WebSocket 处理
    ├── asr.py               # ASR（Paraformer / Whisper）
    ├── llm.py               # LLM 客户端（流式 streaming）
    ├── tts.py               # TTS（Edge-TTS / Qwen3 / pyttsx3）
    ├── audio.py             # 音频采集 + VAD
    ├── main.py              # CLI 终端模式入口
    ├── templates/
    │   └── index.html       # Web UI 模板
    └── static/
        ├── style.css        # Aurora 风格样式
        └── app.js           # 前端 WebSocket 逻辑
```

---

## License

MIT
