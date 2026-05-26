"""Rich terminal UI for voice assistant."""
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import print as rprint
import time


console = Console()

STATES = {
    "idle":      ("[dim]等待中...[/dim]", "grey50"),
    "listening": ("🎤 [bold green]正在聆听...[/bold green]", "green"),
    "thinking":  ("🤔 [bold yellow]思考中...[/bold yellow]", "yellow"),
    "speaking":  ("🔊 [bold cyan]正在播报...[/bold cyan]", "cyan"),
    "error":     ("❌ [bold red]出错了[/bold red]", "red"),
}


class UI:
    def __init__(self):
        self.history: list[tuple[str, str]] = []  # (role, text)
        self._current_state = "idle"
        self._llm_buffer = ""

    def print_banner(self):
        console.print(Panel.fit(
            "[bold cyan]🔧 小新语音助手[/bold cyan]\n"
            "[dim]ASR: faster-whisper  |  LLM: Ollama  |  TTS: 本地合成[/dim]\n"
            "[dim]Ctrl+C 退出  |  说话后停顿即触发[/dim]",
            border_style="cyan"
        ))

    def set_state(self, state: str):
        self._current_state = state
        label, _ = STATES.get(state, STATES["idle"])
        console.print(f"  {label}")

    def show_user(self, text: str):
        self.history.append(("user", text))
        console.print(f"\n[bold blue]👤 你：[/bold blue]{text}")

    def show_assistant_start(self):
        console.print("[bold green]🤖 小新：[/bold green]", end="")
        self._llm_buffer = ""

    def show_token(self, token: str):
        self._llm_buffer += token
        console.print(token, end="", highlight=False)

    def show_assistant_end(self):
        console.print()  # newline
        self.history.append(("assistant", self._llm_buffer))

    def show_error(self, msg: str):
        console.print(f"\n[bold red]❌ 错误：[/bold red]{msg}")

    def show_info(self, msg: str):
        console.print(f"[dim]{msg}[/dim]")
