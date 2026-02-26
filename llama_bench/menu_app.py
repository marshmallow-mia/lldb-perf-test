"""Interactive Textual TUI menu for configuring and launching llama-bench."""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Iterable
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.validation import ValidationResult, Validator
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    Rule,
    Select,
    Static,
    Switch,
)


# ---------------------------------------------------------------------------
# Path discovery helpers
# ---------------------------------------------------------------------------

def _find_llama_server() -> str:
    """Return path to llama-server if found on PATH, else empty string."""
    found = shutil.which("llama-server")
    return found or ""


def _find_first_gguf() -> str:
    """Return first .gguf found by scanning common model directories generically."""
    search_dirs: list[str] = [
        os.path.expanduser("~/models"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.local/share/models"),
        "/opt/models",
        "./models",
        "/models",
    ]
    # Scan /mnt/* and /media/* generically — never hardcode specific mount points
    search_dirs += sorted(glob.glob("/mnt/*/"))
    search_dirs += sorted(glob.glob("/media/*/"))

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        # Shallow scan first (faster)
        hits = sorted(glob.glob(os.path.join(d, "*.gguf")))
        if hits:
            return hits[0]
        # Recursive scan
        hits = sorted(glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True))
        if hits:
            return hits[0]
    return ""


_DEFAULT_SERVER = _find_llama_server()
_DEFAULT_MODEL = _find_first_gguf()


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path.home() / ".config" / "llama-bench" / "menu_config.json"


def _load_saved_config() -> dict:
    """Load the last-used menu config from disk, returning {} on any failure."""
    try:
        if _CONFIG_PATH.exists():
            data = json.loads(_CONFIG_PATH.read_text())
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_config(mode: str, values: dict) -> None:
    """Persist the current menu config to disk, silently ignoring errors."""
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(
            json.dumps({"mode": mode, "values": values}, indent=2)
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Path autocompletion
# ---------------------------------------------------------------------------

class PathSuggester(Suggester):
    """Inline tab-completion for filesystem paths."""

    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None
        try:
            expanded = os.path.expanduser(value)
            if value.endswith(os.sep):
                matches = sorted(glob.glob(os.path.join(expanded, "*")))
            else:
                matches = sorted(glob.glob(expanded + "*"))
            if matches:
                result = matches[0]
                if value.startswith("~"):
                    home = os.path.expanduser("~")
                    if result.startswith(home):
                        result = "~" + result[len(home):]
                return result
        except Exception:
            return None
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class PathExistsValidator(Validator):
    """Validates that the value is a non-empty path that exists on disk."""

    def __init__(self, allow_empty: bool = False) -> None:
        super().__init__()
        self._allow_empty = allow_empty

    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            if self._allow_empty:
                return self.success()
            return self.failure("Path is required")
        expanded = os.path.expanduser(value.strip())
        if os.path.exists(expanded):
            return self.success()
        return self.failure(f"Path not found: {value}")


# ---------------------------------------------------------------------------
# Option descriptor
# ---------------------------------------------------------------------------

@dataclass
class Opt:
    key: str
    label: str
    kind: str          # "text" | "path" | "int" | "float" | "bool" | "select"
    default: Any = ""
    placeholder: str = ""
    help: str = ""
    choices: list[str] = field(default_factory=list)
    required: bool = False   # show error when empty on Run
    is_path: bool = False    # path-exists validation + PathSuggester


# ---------------------------------------------------------------------------
# Option definitions (common + mode-specific)
# ---------------------------------------------------------------------------

_COMMON_OPTS: list[Opt] = [
    Opt("server", "llama-server binary", "path",
        default=_DEFAULT_SERVER,
        placeholder="Enter path to llama-server binary",
        help="Path to the llama-server executable. Build from llama.cpp; must be v8133+.",
        is_path=True, required=True),
    Opt("model", "Model file (.gguf)", "path",
        default=_DEFAULT_MODEL,
        placeholder="Enter path to .gguf model file",
        help="Path to the GGUF model file. Any quantisation (Q4_K_M, Q5_K_M, Q8_0…) works.",
        is_path=True, required=True),
    Opt("host", "Host", "text",
        default="0.0.0.0", placeholder="0.0.0.0",
        help="Server bind address. Use 0.0.0.0 for all interfaces or 127.0.0.1 for local only."),
    Opt("port", "Port", "int",
        default="5001", placeholder="5001",
        help="Server port. Range: 1024–65535. Default 5001 avoids conflicts with common services."),
    Opt("np", "Parallel slots", "int",
        default="1", placeholder="1",
        help="Parallel request slots (-np). 1 = sequential; 2–4 for concurrent client testing. Higher values increase VRAM."),
    Opt("ctx", "Context size", "int",
        default="49152", placeholder="49152",
        help="Max context window in tokens. Powers of 2 or multiples of 4096. ≥ prompt size (~14k). More ctx = more VRAM."),
    Opt("n_gpu_layers", "GPU layers (ngl)", "int",
        default="45", placeholder="45",
        help="Layers to offload to GPU. Max depends on model (e.g. 35B ≈ 65 layers). -1 = all. Lower if VRAM is tight."),
    Opt("batch_size", "Batch size", "int",
        default="1536", placeholder="1536",
        help="Logical batch size. Range: 512–4096. Larger = better throughput but more VRAM. Must be ≥ ubatch."),
    Opt("ubatch_size", "Micro-batch size", "int",
        default="512", placeholder="512",
        help="Physical (micro) batch size. Range: 128–1024. Must be ≤ batch size. 256–512 is a safe default."),
    Opt("flash_attn", "Flash Attention", "bool",
        default=True,
        help="Flash Attention — leave ON. Cuts VRAM and speeds up prefill. Only disable if you hit a driver bug."),
    Opt("kv_unified", "Unified KV cache", "bool",
        default=True,
        help="Unified KV cache pool — leave ON. Shares memory across slots. Only disable for debugging."),
    Opt("cont_batching", "Continuous batching", "bool",
        default=True,
        help="Continuous batching — leave ON. Required for multi-slot (np>1) efficiency."),
    Opt("cache_type_k", "KV-cache type K", "select",
        default="q8_0", choices=["q8_0", "q4_0", "f16", "f32"],
        help="K-cache quantisation. q8_0 = best quality/size balance. q4_0 saves ~50% VRAM with small accuracy cost."),
    Opt("cache_type_v", "KV-cache type V", "select",
        default="q8_0", choices=["q8_0", "q4_0", "f16", "f32"],
        help="V-cache quantisation. q8_0 = default. Match K type for consistency. f16 = lossless but 2× VRAM."),
    Opt("split_mode", "GPU split mode", "select",
        default="none", choices=["none", "layer", "row"],
        help="GPU split strategy. none = single GPU. layer = split layers across GPUs (recommended multi-GPU). row = tensor-parallel."),
    Opt("threads", "CPU threads", "int",
        default="8", placeholder="8",
        help="CPU inference threads. Tip: set to physical core count, not logical (no hyperthreading). Max useful: ~16."),
    Opt("output", "Output file", "path",
        default="", placeholder="results/bench_<timestamp>.jsonl (auto)",
        help="JSONL output path. Leave blank for an auto-generated timestamped file in results/.",
        is_path=False, required=False),
    Opt("sudo", "Use sudo", "bool",
        default=True,
        help="Launch llama-server with sudo. Required on most Linux setups for GPU memory access."),
]

_BENCH_OPTS: list[Opt] = [
    Opt("goal", "Optimization goal", "select",
        default="general",
        choices=["reverse_engineering", "coding", "chatting", "rag_research", "general", "custom"],
        help="Goal preset that weights the 4 benchmark phases. Choose 'custom' to set individual weights."),
    Opt("w_max_context", "  Weight: max context", "int",
        default="5", placeholder="5",
        help="Points (0–20) for max-context phase. Only used when goal=custom. All 4 weights must sum to 20."),
    Opt("w_fastest_response", "  Weight: fastest response", "int",
        default="5", placeholder="5",
        help="Points (0–20) for fastest-response phase. Only used when goal=custom."),
    Opt("w_throughput", "  Weight: throughput", "int",
        default="5", placeholder="5",
        help="Points (0–20) for throughput phase. Only used when goal=custom."),
    Opt("w_long_context_rag", "  Weight: long-context RAG", "int",
        default="5", placeholder="5",
        help="Points (0–20) for long-context RAG phase. Only used when goal=custom."),
    Opt("ctx_min", "Context min", "int",
        default="16384", placeholder="16384",
        help="Minimum context for the sweep. Must be > prompt size (~13k tokens). Recommended: 16384."),
    Opt("ctx_step", "Context step", "int",
        default="8192", placeholder="8192",
        help="Step between context candidates. Powers of 2 (4096, 8192, 16384). Smaller = more configs tested."),
    Opt("max_retries", "Max OOM retries", "int",
        default="5", placeholder="5",
        help="Max OOM retries per context candidate. Range: 3–10. Each retry reduces ngl by --ngl-step (default 4)."),
    Opt("n_followups", "Follow-up prompts", "int",
        default="4", placeholder="4",
        help="Follow-up prompts to measure warm TTFT. Range: 1–8. More = better stats, longer run time."),
    Opt("max_tokens", "Max tokens", "int",
        default="512", placeholder="512",
        help="Max tokens generated per request. Range: 128–2048. Controls decode length and throughput measurement."),
]

_SEARCH_OPTS: list[Opt] = [
    Opt("np_tests", "np values", "text",
        default="1", placeholder="1-2 or 1,2",
        help="Parallel slot values to test. Format: '1,2' or range '1-4'. More slots = more VRAM, test multi-client perf."),
    Opt("ctx_tests", "ctx values", "text",
        default="49152", placeholder="49152,98304",
        help="Context sizes to test. Comma-separated or range. Must be > prompt size (~13k). Example: 16384,32768,49152."),
    Opt("ngl_tests", "ngl values", "text",
        default="45", placeholder="40,45,47",
        help="GPU layer counts to test. Comma-separated. Use values near your GPU's max. Example: 40,43,45."),
    Opt("max_configs", "Max configs", "int",
        default="50", placeholder="50",
        help="Max configs in Phase 1. Range: 10–200. Higher = more thorough search but longer run time."),
]

_EXPLORE_OPTS: list[Opt] = [
    Opt("ctx_min", "Context min", "int",
        default="16384", placeholder="16384",
        help="Minimum context for the sweep. Must be > prompt size (~13k tokens). Recommended: 16384."),
    Opt("ctx_step", "Context step", "int",
        default="8192", placeholder="8192",
        help="Step between context candidates. Powers of 2 (4096, 8192, 16384). Smaller = more configs explored."),
    Opt("ngl_tests", "ngl values", "text",
        default="", placeholder="37,41,45 (optional)",
        help="GPU layers to sweep. Comma-separated. Blank = use the ngl set in Common options. Example: 40,43,45."),
    Opt("batch_tests", "batch values", "text",
        default="", placeholder="1024,1536 (optional)",
        help="Batch sizes to sweep. Comma-separated. Blank = use the batch size set in Common options."),
    Opt("n_followups", "Follow-up prompts", "int",
        default="2", placeholder="2",
        help="Follow-ups per run. Range: 1–4. Keep low (1–2) for fast indefinite sweeps; higher for better warm TTFT stats."),
    Opt("max_tokens", "Max tokens", "int",
        default="512", placeholder="512",
        help="Max tokens generated per request. Range: 128–2048. Controls decode length and throughput measurement."),
]


# ---------------------------------------------------------------------------
# Field-type badge labels
# ---------------------------------------------------------------------------

_KIND_BADGE: dict[str, str] = {
    "path":   "PATH",
    "int":    "INT",
    "float":  "FLOAT",
    "text":   "TEXT",
    "bool":   "BOOL",
    "select": "LIST",
}


def _badge_for(opt: Opt) -> str:
    base = _KIND_BADGE.get(opt.kind, opt.kind.upper())
    if opt.required:
        return f"[bold]{base}*[/bold]"
    return base


# ---------------------------------------------------------------------------
# File picker modal
# ---------------------------------------------------------------------------

class _GgufTree(DirectoryTree):
    """DirectoryTree that shows only directories + .gguf files."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [p for p in paths if p.is_dir() or p.suffix == ".gguf"]


class _ExeTree(DirectoryTree):
    """DirectoryTree that shows directories + executable files."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        result = []
        for p in paths:
            if p.is_dir():
                result.append(p)
            elif os.access(p, os.X_OK):
                result.append(p)
        return result


class FilePickerModal(ModalScreen):  # type: ignore[type-arg]
    """Full-screen file browser modal. Returns selected path string or None."""

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cancel", show=True),
    ]

    CSS = """
    FilePickerModal {
        align: center middle;
    }
    #fp-container {
        width: 80%;
        height: 80%;
        background: $surface;
        border: solid $accent;
        padding: 0;
    }
    #fp-title {
        background: $accent;
        color: $text;
        text-style: bold;
        width: 100%;
        height: 3;
        content-align: center middle;
        padding: 0 2;
    }
    #fp-path-bar {
        height: 3;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $panel-lighten-2;
    }
    #fp-path-bar Label {
        width: auto;
        content-align: left middle;
        color: $text-muted;
        padding: 0 1 0 0;
    }
    #fp-path-input {
        width: 1fr;
        height: 3;
        background: $surface;
        border: none;
    }
    #fp-tree {
        height: 1fr;
        border: none;
        padding: 0 1;
    }
    #fp-btn-bar {
        height: auto;
        padding: 1 2;
        background: $panel;
        align: right middle;
    }
    #fp-btn-select {
        min-width: 12;
        margin-right: 1;
    }
    #fp-btn-cancel {
        min-width: 12;
    }
    #fp-hint {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $panel;
    }
    """

    def __init__(self, start_path: str = "", gguf_only: bool = False) -> None:
        super().__init__()
        # Resolve start path to a directory
        if start_path:
            p = Path(os.path.expanduser(start_path))
            if p.is_file():
                p = p.parent
            if p.is_dir():
                self._start = p
            else:
                self._start = Path.home()
        else:
            self._start = Path.home()
        self._gguf_only = gguf_only
        self._selected: Path | None = None

    def compose(self) -> ComposeResult:
        title = "Browse — .gguf files" if self._gguf_only else "Browse — executable files"
        with Vertical(id="fp-container"):
            yield Static(title, id="fp-title")
            with Horizontal(id="fp-path-bar"):
                yield Label("Dir:")
                yield Input(str(self._start), id="fp-path-input")
            if self._gguf_only:
                yield _GgufTree(self._start, id="fp-tree")
            else:
                yield _ExeTree(self._start, id="fp-tree")
            yield Static(
                "↑↓ navigate  ·  Enter select  ·  Esc cancel",
                id="fp-hint",
            )
            with Horizontal(id="fp-btn-bar"):
                yield Button("✓  Select", id="fp-btn-select", variant="success")
                yield Button("✕  Cancel", id="fp-btn-cancel", variant="error")

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        self._selected = event.path
        # Update path input to show current selection
        self.query_one("#fp-path-input", Input).value = str(event.path)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Navigate to typed directory."""
        p = Path(os.path.expanduser(event.value))
        if p.is_dir():
            tree = self.query_one("#fp-tree", DirectoryTree)
            tree.path = p

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "fp-btn-select":
            if self._selected is not None:
                self.dismiss(str(self._selected))
            else:
                # If nothing selected but input has a valid path, use it
                val = self.query_one("#fp-path-input", Input).value.strip()
                if val and Path(os.path.expanduser(val)).exists():
                    self.dismiss(val)
        elif event.button.id == "fp-btn-cancel":
            self.dismiss(None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# OptionRow widget — one label + one input/switch/select per row
# ---------------------------------------------------------------------------

class OptionRow(Horizontal):
    """Single-line row: [Label 24ch][Control 1fr][Browse?][Undo][Badge]."""

    DEFAULT_CSS = """
    OptionRow {
        height: auto;
        width: 100%;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    /* Label */
    OptionRow Label.opt-label {
        width: 24;
        padding: 1 1 0 0;
        color: $text-muted;
        text-align: right;
        content-align: right top;
    }

    /* Text / path / int / float inputs */
    OptionRow Input {
        width: 1fr;
        height: 3;
        background: $surface;
        color: $text;
        border: solid $panel-lighten-2;
        padding: 0 1;
    }
    OptionRow Input:focus {
        border: solid $accent;
        background: $surface-lighten-1;
    }
    OptionRow Input.-invalid {
        border: solid $error;
        color: $error;
    }
    OptionRow Input.error {
        border: solid $error;
        color: $error;
    }
    OptionRow Input.goal-locked {
        color: $text-muted;
        background: $panel-darken-1;
        border: solid $panel;
        opacity: 0.6;
    }

    /* Switch */
    OptionRow Switch {
        width: auto;
        margin: 1 0 0 0;
    }

    /* Select */
    OptionRow Select {
        width: 1fr;
        border: solid $panel-lighten-2;
        height: 3;
    }
    OptionRow Select:focus {
        border: solid $accent;
    }

    /* Undo button */
    OptionRow Button.undo-btn {
        min-width: 5;
        width: 5;
        height: 3;
        margin: 0 0 0 1;
        border: solid $panel-lighten-2;
        background: $panel;
        color: $text-muted;
    }
    OptionRow Button.undo-btn:hover {
        border: solid $warning;
        color: $warning;
    }
    OptionRow Button.undo-btn.dirty {
        border: solid $warning;
        color: $warning;
    }

    /* Browse button — path fields only */
    OptionRow Button.browse-btn {
        min-width: 5;
        width: 5;
        height: 3;
        margin: 0 0 0 1;
        border: solid $panel-lighten-2;
        background: $panel;
        color: $text-muted;
    }
    OptionRow Button.browse-btn:hover {
        border: solid $accent;
        color: $accent;
    }

    /* Type badge */
    OptionRow Static.type-badge {
        width: 8;
        height: 3;
        margin: 0 0 0 1;
        color: $text-muted;
        background: $panel;
        border: solid $panel-lighten-2;
        content-align: center middle;
    }
    OptionRow Static.type-badge.required {
        color: $error;
        border: solid $error;
    }
    OptionRow Static.type-badge.path-ok {
        color: $success;
        border: solid $success;
    }
    OptionRow Static.type-badge.path-bad {
        color: $error;
        border: solid $error;
    }
    """

    def __init__(self, opt: Opt, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.opt = opt
        self._touched: set[str] = set()

    def compose(self) -> ComposeResult:
        opt = self.opt
        yield Label(opt.label, classes="opt-label")

        if opt.kind == "bool":
            yield Switch(value=bool(opt.default), id=f"opt_{opt.key}")
        elif opt.kind == "select":
            options = [(c, c) for c in opt.choices]
            default_idx = opt.choices.index(opt.default) if opt.default in opt.choices else 0
            yield Select(
                options,
                value=opt.choices[default_idx],
                id=f"opt_{opt.key}",
                allow_blank=False,
            )
            yield Button("↺", id=f"undo_{opt.key}", classes="undo-btn", tooltip="Revert to default")
        else:
            # text / int / float / path
            validators = []
            if opt.is_path:
                validators.append(PathExistsValidator(allow_empty=not opt.required))

            suggester = PathSuggester() if opt.is_path else None

            inp = Input(
                value=str(opt.default) if opt.default else "",
                placeholder=opt.placeholder,
                suggester=suggester,
                validators=validators if validators else None,
                validate_on=["blur", "submitted"] if validators else None,
                valid_empty=not opt.required,
                id=f"opt_{opt.key}",
                tooltip=opt.help if opt.help else None,
            )
            yield inp

            if opt.is_path:
                yield Button("…", id=f"browse_{opt.key}", classes="browse-btn", tooltip="Browse filesystem")

            yield Button("↺", id=f"undo_{opt.key}", classes="undo-btn", tooltip="Revert to default")

        # Type / status badge — rightmost column
        badge_classes = "type-badge"
        badge_text = _badge_for(opt)
        if opt.required:
            badge_classes += " required"
        yield Static(badge_text, id=f"badge_{opt.key}", classes=badge_classes)

    def on_mount(self) -> None:
        """Validate required/path fields immediately — errors visible from open."""
        opt = self.opt
        if opt.kind == "bool":
            return
        # For select fields: defer the value assignment until after the full
        # compose/mount/render cycle so SelectCurrent's #label Static exists.
        # (Textual 8 quirk: watch_value fires before SelectCurrent children
        # are mounted, so select_current.query_one('#label') raises NoMatches
        # and the update is silently dropped.)
        if opt.kind == "select":
            def _set_select_value() -> None:
                try:
                    sel = self.query_one(f"#opt_{opt.key}", Select)
                    if opt.default in opt.choices:
                        sel.value = opt.default
                except Exception:
                    pass
            self.call_after_refresh(_set_select_value)
            return
        # Pre-touch required fields and path fields so errors show immediately
        if opt.required or opt.is_path:
            self._touched.add(opt.key)
            try:
                inp = self.query_one(f"#opt_{opt.key}", Input)
                self._validate_input(inp, inp.value)
            except Exception:
                pass

    def on_input_changed(self, event: Input.Changed) -> None:
        """Live validation: once user starts typing."""
        self._touched.add(self.opt.key)
        self._validate_input(event.input, event.value)
        self._update_undo_dirty(event.value)

    def on_input_blur(self, event: Input.Blur) -> None:  # type: ignore[override]
        """Validate when focus leaves a field."""
        inp = event.input
        self._touched.add(self.opt.key)
        self._validate_input(inp, inp.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        # Undo: revert to default
        if btn_id == f"undo_{self.opt.key}":
            event.stop()
            self._revert_to_default()
            return

        # Browse: open file picker modal
        if btn_id == f"browse_{self.opt.key}":
            event.stop()
            opt = self.opt
            try:
                inp = self.query_one(f"#opt_{opt.key}", Input)
                current_val = inp.value
            except Exception:
                current_val = str(opt.default) if opt.default else ""

            gguf_only = opt.key == "model" or "gguf" in opt.placeholder.lower()

            def _on_picked(path: str | None) -> None:
                if path is not None:
                    try:
                        inp2 = self.query_one(f"#opt_{opt.key}", Input)
                        inp2.value = path
                        # Remove Textual's own --invalid CSS class that may have
                        # been set from the initial empty-value validation on mount.
                        # The class name in Textual's BEM system is '--invalid'.
                        inp2.remove_class("--invalid")
                        self._touched.add(opt.key)
                        self._validate_input(inp2, path)
                        self._update_undo_dirty(path)
                    except Exception:
                        pass

            self.app.push_screen(
                FilePickerModal(start_path=current_val, gguf_only=gguf_only),
                _on_picked,
            )
            return

    def _revert_to_default(self) -> None:
        opt = self.opt
        default_str = str(opt.default) if opt.default else ""
        if opt.kind == "bool":
            try:
                self.query_one(f"#opt_{opt.key}", Switch).value = bool(opt.default)
            except Exception:
                pass
        elif opt.kind == "select":
            try:
                sel = self.query_one(f"#opt_{opt.key}", Select)
                if opt.default in opt.choices:
                    sel.value = opt.default
            except Exception:
                pass
            self._mark_undo_clean()
        else:
            try:
                inp = self.query_one(f"#opt_{opt.key}", Input)
                inp.value = default_str
                self._touched.add(opt.key)
                self._validate_input(inp, default_str)
                self._update_undo_dirty(default_str)
            except Exception:
                pass

    def _update_undo_dirty(self, current_value: str) -> None:
        """Mark undo button orange when value differs from default."""
        default_str = str(self.opt.default) if self.opt.default else ""
        try:
            btn = self.query_one(f"#undo_{self.opt.key}", Button)
            if current_value != default_str:
                btn.add_class("dirty")
            else:
                btn.remove_class("dirty")
        except Exception:
            pass

    def _mark_undo_clean(self) -> None:
        try:
            self.query_one(f"#undo_{self.opt.key}", Button).remove_class("dirty")
        except Exception:
            pass

    def _validate_input(self, inp: Input, value: str) -> None:
        """Manage the 'error' CSS class and update the type badge."""
        opt = self.opt
        if opt.key not in self._touched:
            return

        has_error = False

        if opt.is_path:
            if not value.strip():
                has_error = opt.required
            else:
                expanded = os.path.expanduser(value.strip())
                has_error = not os.path.exists(expanded)
        elif opt.required and not value.strip():
            has_error = True

        if has_error:
            inp.add_class("error")
        else:
            inp.remove_class("error")

        self._update_badge(value, has_error)

    def _update_badge(self, value: str, has_error: bool) -> None:
        opt = self.opt
        try:
            badge = self.query_one(f"#badge_{opt.key}", Static)
        except Exception:
            return

        badge.remove_class("required", "path-ok", "path-bad")

        if has_error:
            badge.add_class("path-bad")
        elif opt.required and value.strip():
            badge.add_class("path-ok")
        elif opt.required:
            badge.add_class("required")


# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------

MODES = ["bench", "search", "explore"]
MODE_DESCRIPTIONS = {
    "bench":   "4-preset characterization bench — finds optimal server config for your use-case goal",
    "search":  "Staged two-phase parameter search over a config space",
    "explore": "Continuous multi-objective explorer — runs indefinitely",
}


# ---------------------------------------------------------------------------
# Main menu app
# ---------------------------------------------------------------------------

class MenuApp(App):
    """Interactive configuration menu for llama-bench."""

    TITLE = "llama-bench"
    SUB_TITLE = "Configure and launch"

    CSS = """
    Screen {
        background: $surface-darken-1;
    }

    /* ── Mode selector bar ── */
    #mode-bar {
        height: auto;
        padding: 1 2;
        background: $panel;
    }
    #mode-bar Label {
        margin-right: 2;
        color: $text-muted;
        content-align: left middle;
    }
    .mode-btn {
        margin-right: 0;
        border: solid $panel-lighten-2;
        background: $panel;
        color: $text-muted;
        min-width: 12;
    }
    .mode-btn:hover {
        background: $panel-lighten-1;
        color: $text;
        border: solid $accent;
    }
    .mode-btn.active {
        background: $accent;
        color: $text;
        border: solid $accent;
        text-style: bold;
    }

    #mode-desc {
        padding: 0 2;
        color: $text-muted;
        height: 1;
        margin-bottom: 0;
    }

    /* ── Field legend bar ── */
    #legend-bar {
        height: 1;
        padding: 0 2;
        background: $panel-darken-1;
        margin-bottom: 1;
    }
    #legend-bar Label {
        color: $text-muted;
        margin-right: 2;
        width: auto;
    }

    /* ── Scrollable form ── */
    #opts-scroll {
        height: 1fr;
        padding: 0 2;
    }

    .section-rule {
        margin: 1 0 0 0;
        color: $panel-lighten-2;
    }
    .section-label {
        padding: 0 0 1 0;
        color: $accent;
        text-style: bold;
    }

    #advanced-toggle {
        padding: 0 1;
        color: $text-muted;
        height: 1;
        margin: 1 0;
    }
    #advanced-toggle:hover {
        color: $text;
    }

    #advanced-section {
        display: none;
    }
    #advanced-section.visible {
        display: block;
    }

    /* Weight rows are always visible — locked/greyed for builtin goals, editable for custom */
    /* Goal section label */
    .goal-section-label {
        padding: 0 0 0 1;
        color: $accent-darken-1;
        text-style: italic;
        height: 1;
        margin: 0 0 1 0;
    }

    /* ── Bottom button bar ── */
    #btn-bar {
        height: auto;
        padding: 1 2;
        background: $panel;
        align: right middle;
    }
    #btn-run {
        margin-right: 1;
        min-width: 12;
    }
    #btn-quit {
        min-width: 12;
    }
    """

    BINDINGS = [
        Binding("q", "quit_app", "Quit", show=True),
        Binding("f5", "toggle_advanced", "Advanced", show=True),
        Binding("ctrl+r", "run_bench", "Run", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._mode = "bench"
        self._show_advanced = False
        saved = _load_saved_config()
        if saved.get("mode") in MODES:
            self._mode = saved["mode"]
        self._saved_values: dict = saved.get("values", {})

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        # Mode selector bar
        with Horizontal(id="mode-bar"):
            yield Label("Mode:")
            for m in MODES:
                cls = "mode-btn active" if m == self._mode else "mode-btn"
                yield Button(m, id=f"mode-{m}", classes=cls)

        yield Static(MODE_DESCRIPTIONS[self._mode], id="mode-desc")

        # Legend — explain badges and buttons
        with Horizontal(id="legend-bar"):
            yield Label(
                "[bold]TYPE*[/bold]=required  TYPE=optional  "
                "[ … ]=browse file  [ ↺ ]=undo  (orange ↺ = modified)"
            )

        # Scrollable option form
        with ScrollableContainer(id="opts-scroll"):
            yield Rule(classes="section-rule")
            yield Static("Common options", classes="section-label")
            for opt in _COMMON_OPTS:
                yield OptionRow(self._apply_saved(opt), id=f"row_{opt.key}")

            yield Rule(classes="section-rule")
            yield Static("Mode options", id="mode-section-label", classes="section-label")
            yield from self._mode_rows()

            yield Static(
                "  [F5] Show advanced options",
                id="advanced-toggle",
            )
            with Vertical(id="advanced-section"):
                yield Rule(classes="section-rule")
                yield Static("Advanced options", classes="section-label")
                yield from self._advanced_rows()

        # Run / Quit buttons
        with Horizontal(id="btn-bar"):
            yield Button("▶  Run", id="btn-run", variant="success")
            yield Button("✕  Quit", id="btn-quit", variant="error")

        yield Footer()

    def on_mount(self) -> None:
        """Apply initial goal-lock state after the UI is fully mounted."""
        if self._mode == "bench":
            self._apply_goal_lock()

    def _apply_goal_lock(self) -> None:
        """Lock/unlock weight inputs AND show/hide rows based on the current goal."""
        from llama_bench.presets import BUILTIN_GOALS
        _WEIGHT_KEYS = [
            "w_max_context", "w_fastest_response", "w_throughput", "w_long_context_rag"
        ]
        _WEIGHT_ATTRS = [
            "max_context", "fastest_response", "throughput", "long_context_rag"
        ]
        try:
            goal = str(self.query_one("#opt_goal", Select).value)
        except Exception:
            return
        is_custom = (goal == "custom")
        for key, attr in zip(_WEIGHT_KEYS, _WEIGHT_ATTRS):
            try:
                inp = self.query_one(f"#opt_{key}", Input)
                if is_custom:
                    inp.disabled = False
                    inp.remove_class("goal-locked")
                else:
                    preset = BUILTIN_GOALS.get(goal)
                    if preset is not None:
                        inp.value = str(getattr(preset, attr))
                    inp.disabled = True
                    inp.add_class("goal-locked")
            except Exception:
                pass


    def _mode_rows(self) -> list:
        opts = {
            "bench": _BENCH_OPTS,
            "search": _SEARCH_OPTS,
            "explore": _EXPLORE_OPTS,
        }[self._mode]
        _WEIGHT_KEYS = {"w_max_context", "w_fastest_response", "w_throughput", "w_long_context_rag"}
        rows: list = []
        if self._mode == "bench":
            rows.append(Static(
                "  Goal & scoring weights",
                id="goal-section-label",
                classes="goal-section-label",
            ))
        for opt in opts:
            extra_classes = "weight-row" if opt.key in _WEIGHT_KEYS else ""
            row = OptionRow(self._apply_saved(opt), id=f"row_{opt.key}",
                            classes=extra_classes if extra_classes else None)
            rows.append(row)
        return rows

    def _advanced_rows(self) -> list[OptionRow]:
        adv = [
            Opt("vk_devices", "VK devices", "text",
                default="", placeholder="0,1 (auto-detect if blank)",
                help="Vulkan GPU indices (GGML_VK_VISIBLE_DEVICES). Blank = auto-discover via vulkaninfo. Example: 0,1 for two GPUs."),
            Opt("split_mode_adv", "Split mode", "select",
                default="none", choices=["none", "layer", "row"],
                help="GPU split strategy. none = single GPU. layer = pipeline-parallel (recommended). row = tensor-parallel (experimental)."),
            Opt("cache_reuse", "Cache reuse (tokens)", "int",
                default="512", placeholder="512",
                help="Min shared prefix tokens to trigger KV-cache reuse. Range: 0–2048. 512 is a good default; 0 disables reuse."),
            Opt("model_draft", "Draft model (.gguf)", "path",
                default="", placeholder="Path to draft model for speculative decoding",
                help="Optional small draft model for speculative decoding. Must match the main model's vocabulary.",
                is_path=True, required=False),
            Opt("max_ttft", "Max TTFT (s)", "float",
                default="60.0", placeholder="60.0",
                help="Max acceptable time-to-first-token. Configs exceeding this are skipped. Range: 5–120 s."),
            Opt("min_tps", "Min throughput (tok/s)", "float",
                default="1.0", placeholder="1.0",
                help="Min acceptable decode throughput. Configs below this are skipped. Range: 0.5–10. Keep low (1.0) for initial sweeps."),
            Opt("threads_batch", "Batch threads", "int",
                default="8", placeholder="8",
                help="CPU threads for batch processing. Set to physical core count. Can match --threads or be set higher for prefill."),
        ]
        return [OptionRow(self._apply_saved(opt), id=f"row_adv_{opt.key}") for opt in adv]

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id

        if btn_id == "btn-quit":
            self.action_quit_app()
            return

        if btn_id == "btn-run":
            self.action_run_bench()
            return

        # Mode buttons
        for m in MODES:
            if btn_id == f"mode-{m}":
                self._switch_mode(m)
                return

    def _switch_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode

        # Update button styles
        for m in MODES:
            btn = self.query_one(f"#mode-{m}", Button)
            if m == mode:
                btn.add_class("active")
            else:
                btn.remove_class("active")

        # Update description
        self.query_one("#mode-desc", Static).update(MODE_DESCRIPTIONS[mode])

        # Rebuild mode rows: remove old, mount new
        existing_keys = {opt.key for opts in [_BENCH_OPTS, _SEARCH_OPTS, _EXPLORE_OPTS] for opt in opts}
        for key in existing_keys:
            try:
                row = self.query_one(f"#row_{key}")
                row.remove()
            except Exception:
                pass
        # Remove goal section label if present
        try:
            self.query_one("#goal-section-label").remove()
        except Exception:
            pass

        # Insert new rows before the advanced-toggle static
        _WEIGHT_KEYS = {"w_max_context", "w_fastest_response", "w_throughput", "w_long_context_rag"}
        anchor = self.query_one("#advanced-toggle")
        if mode == "bench":
            label = Static(
                "  Goal & scoring weights",
                id="goal-section-label",
                classes="goal-section-label",
            )
            anchor.mount(label, before=anchor)
        for opt in {
            "bench": _BENCH_OPTS,
            "search": _SEARCH_OPTS,
            "explore": _EXPLORE_OPTS,
        }[mode]:
            extra_classes = "weight-row" if opt.key in _WEIGHT_KEYS else None
            new_row = OptionRow(self._apply_saved(opt), id=f"row_{opt.key}",
                                classes=extra_classes)
            anchor.mount(new_row, before=anchor)
        if mode == "bench":
            self._apply_goal_lock()

    def action_toggle_advanced(self) -> None:
        self._show_advanced = not self._show_advanced
        adv = self.query_one("#advanced-section")
        toggle = self.query_one("#advanced-toggle", Static)
        if self._show_advanced:
            adv.add_class("visible")
            toggle.update("  [F5] Hide advanced options")
        else:
            adv.remove_class("visible")
            toggle.update("  [F5] Show advanced options")

    def action_quit_app(self) -> None:
        self.exit(None)

    def action_run_bench(self) -> None:
        args = self._collect_args()
        if args is None:
            return  # validation failed, errors already shown
        # Persist config before exiting
        values = self._gather_current_values()
        _save_config(self._mode, values)
        self.exit(args)

    def on_select_changed(self, event: Select.Changed) -> None:
        """When the goal dropdown changes, fill weight fields from the builtin preset.

        For builtin goals: write the canonical weights into the four int inputs and
        disable them so the user sees (but cannot accidentally edit) the preset values.
        For 'custom': re-enable all weight inputs so the user can type freely.
        """
        if event.select.id != "opt_goal":
            return
        from llama_bench.presets import BUILTIN_GOALS
        goal = str(event.value)
        _WEIGHT_KEYS = [
            "w_max_context", "w_fastest_response", "w_throughput", "w_long_context_rag"
        ]
        _WEIGHT_ATTRS = [
            "max_context", "fastest_response", "throughput", "long_context_rag"
        ]
        is_custom = (goal == "custom")
        for key, attr in zip(_WEIGHT_KEYS, _WEIGHT_ATTRS):
            try:
                inp = self.query_one(f"#opt_{key}", Input)
                if is_custom:
                    inp.disabled = False
                    inp.remove_class("goal-locked")
                else:
                    preset = BUILTIN_GOALS.get(goal)
                    if preset is not None:
                        inp.value = str(getattr(preset, attr))
                    inp.disabled = True
                    inp.add_class("goal-locked")
            except Exception:
                pass
    def _apply_saved(self, opt: Opt) -> Opt:
        """Return a copy of *opt* with default overridden from saved config if present."""
        saved = self._saved_values.get(opt.key)
        if saved is None:
            return opt
        # For bool fields the saved value is "true"/"false"
        if opt.kind == "bool":
            return _dc_replace(opt, default=(saved == "true"))
        if opt.kind == "select" and saved in opt.choices:
            return _dc_replace(opt, default=saved)
        if opt.kind not in ("bool", "select"):
            return _dc_replace(opt, default=str(saved))
        return opt

    def _gather_current_values(self) -> dict:
        """Read current widget values for all visible options (used to save config)."""
        all_opts = _COMMON_OPTS + {
            "bench": _BENCH_OPTS,
            "search": _SEARCH_OPTS,
            "explore": _EXPLORE_OPTS,
        }[self._mode]
        values: dict = {}
        for opt in all_opts:
            widget_id = f"opt_{opt.key}"
            try:
                if opt.kind == "bool":
                    sw = self.query_one(f"#{widget_id}", Switch)
                    values[opt.key] = "true" if sw.value else "false"
                elif opt.kind == "select":
                    sel = self.query_one(f"#{widget_id}", Select)
                    values[opt.key] = str(sel.value)
                else:
                    inp = self.query_one(f"#{widget_id}", Input)
                    values[opt.key] = inp.value.strip()
            except Exception:
                pass
        return values

    # ------------------------------------------------------------------
    # Argument collection
    # ------------------------------------------------------------------

    def _collect_args(self) -> list[str] | None:
        """Collect current widget values and return a CLI arg list, or None on error."""
        all_opts = _COMMON_OPTS + {
            "bench": _BENCH_OPTS,
            "search": _SEARCH_OPTS,
            "explore": _EXPLORE_OPTS,
        }[self._mode]

        # Gather values; force-touch all rows so errors become visible on Run
        values: dict[str, str] = {}
        has_error = False

        for opt in all_opts:
            widget_id = f"opt_{opt.key}"
            try:
                if opt.kind == "bool":
                    sw = self.query_one(f"#{widget_id}", Switch)
                    values[opt.key] = "true" if sw.value else "false"
                elif opt.kind == "select":
                    sel = self.query_one(f"#{widget_id}", Select)
                    values[opt.key] = str(sel.value)
                else:
                    inp = self.query_one(f"#{widget_id}", Input)
                    values[opt.key] = inp.value.strip()

                    # Force-touch the row so errors are now visible
                    try:
                        row = self.query_one(f"#row_{opt.key}", OptionRow)
                        row._touched.add(opt.key)
                    except Exception:
                        pass

                    # Check required / path validity
                    if opt.required and not values[opt.key]:
                        inp.add_class("error")
                        has_error = True
                    elif opt.is_path and values[opt.key]:
                        expanded = os.path.expanduser(values[opt.key])
                        if not os.path.exists(expanded):
                            inp.add_class("error")
                            has_error = True
            except Exception:
                pass

        if has_error:
            self.notify("Fix highlighted fields before running.", severity="error")
            return None

        # Build CLI args
        mode = self._mode
        args = [mode]

        flag_map = {
            "server": "--server",
            "model": "--model",
            "host": "--host",
            "port": "--port",
            "np": "--np",
            "ctx": "--ctx",
            "n_gpu_layers": "--n-gpu-layers",
            "batch_size": "--batch-size",
            "ubatch_size": "--ubatch-size",
            "flash_attn": "--flash-attn",
            "kv_unified": "--kv-unified",
            "cont_batching": "--cont-batching",
            "cache_type_k": "--cache-type-k",
            "cache_type_v": "--cache-type-v",
            "split_mode": "--split-mode",
            "threads": "--threads",
            "output": "--output",
            "sudo": "--sudo",
            # bench
            "ctx_min": "--ctx-min",
            "ctx_step": "--ctx-step",
            "max_retries": "--max-retries",
            "n_followups": "--n-followups",
            "max_tokens": "--max-tokens",
            "goal": "--goal",
            "w_max_context": "--w-max-context",
            "w_fastest_response": "--w-fastest-response",
            "w_throughput": "--w-throughput",
            "w_long_context_rag": "--w-long-context-rag",
            # search
            "np_tests": "--np-tests",
            "ctx_tests": "--ctx-tests",
            "ngl_tests": "--ngl-tests",
            "max_configs": "--max-configs",
            # explore
            "batch_tests": "--batch-tests",
        }

        bool_flags = {"flash_attn", "kv_unified", "cont_batching", "sudo"}
        weight_flags = {"w_max_context", "w_fastest_response", "w_throughput", "w_long_context_rag"}

        # Validate custom goal weights sum to 20; skip weight flags for builtin goals
        if mode == "bench" and values.get("goal") == "custom":
            weight_keys = ["w_max_context", "w_fastest_response", "w_throughput", "w_long_context_rag"]
            try:
                total = sum(int(values.get(k, "0") or "0") for k in weight_keys)
            except ValueError:
                self.notify("Custom goal weights must be integers.", severity="error")
                return None
            if total != 20:
                self.notify(f"Custom goal weights must sum to 20 (currently {total}).", severity="error")
                return None

        for key, flag in flag_map.items():
            if key not in values:
                continue
            # Skip weight flags for builtin goals — CLI derives them from --goal
            if key in weight_flags and values.get("goal") != "custom":
                continue
            val = values[key]
            if not val:
                continue
            if key in bool_flags:
                if val == "true":
                    args.append(flag)
                else:
                    args.append(f"--no-{flag[2:]}")
            else:
                args.extend([flag, val])

        return args


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_menu() -> None:
    """Launch the interactive menu and execute the chosen command, or exit."""
    app = MenuApp()
    result = app.run()

    if result is None:
        # User pressed Quit
        sys.exit(0)

    # result is a list of CLI args starting with the subcommand
    import click
    from llama_bench.cli import main
    try:
        main(result, standalone_mode=False)
    except click.exceptions.Exit as e:
        sys.exit(e.code)
    except click.exceptions.Abort:
        sys.exit(1)
    except Exception as exc:
        from rich.console import Console
        Console(stderr=True).print(f"[red]Error:[/] {exc}")
        sys.exit(1)
