# llama-bench

**llama-bench** is a command-line adaptive tuning and testing tool for
[llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server`.  It runs a
realistic **reverse-engineering workload**, automatically iterates server
configurations, handles VRAM OOM errors gracefully, and emits ranked results so
you can find the best combination of context size, GPU layers, and batch parameters
for your hardware.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Full CLI Reference](#full-cli-reference)
6. [Profile Files](#profile-files)
7. [How It Works](#how-it-works)
8. [Output Format](#output-format)
9. [Security Note](#security-note)
10. [Version Compatibility](#version-compatibility)

---

## Overview

`llama-bench` was built to answer the question: *what combination of
`llama-server` flags gives the best latency and context capacity for my model,
GPU, and workload?*

It has a **single unified tuning mode** (`bench`) that:

1. Iterates candidate context sizes from `--ctx` down to `--ctx-min` in steps of
   `--ctx-step`, adjusting GPU layers and batch size when VRAM OOM is detected.
2. Runs each valid configuration against the built-in **reverse-engineering
   workload** — a ~1200-token expert system prompt plus a shared binary-analysis
   context (drawn from the repository prompt files: `README.md`, `agent.md`,
   `artifacts.md`), followed by follow-up questions that re-send the shared
   prefix to exercise KV-cache prefix reuse.
3. Collects TTFT, end-to-end latency, and tok/s for every run, then produces a
   ranked JSONL result file and a `summary.json` with the best configurations.

Key features:

- **Single tuning mode**: No separate bench/search modes. One unified `bench`
  command for the reverse-engineering use case.
- **Adaptive error handling**: VRAM OOM automatically reduces `--n-gpu-layers` and
  `--batch-size` within configurable bounds before retrying.  Memory-fit
  heuristics trigger either a proportional or big-step (halve) context reduction
  depending on how far memory is from fitting.
- **Reverse-engineering workload**: prompt corpus loaded from `README.md`,
  `agent.md`, and `artifacts.md` (never `solution.md`), exercising
  KV-cache prefix reuse across multi-turn conversations.
- **Usability threshold**: A configuration is *usable* when
  `tokens/s >= --min-tokens-per-sec` (default **4.0**).  TTFT is always recorded
  and used for ranking, but TTFT gating is **disabled by default** (enable with
  `--max-ttft-s`).
- **Engine selection** (`--engine`): choose `vulkan` (default, GPU via Vulkan) or
  `cpu`.  Pass `--vk-devices` to restrict which Vulkan device indices are visible
  to the server.  If `--vk-devices` is omitted, `GGML_VK_VISIBLE_DEVICES` is not
  set (all devices visible).
- **Engine mismatch detection**: if `--engine vulkan` was requested but the server
  binary does not emit any Vulkan backend lines, a **red warning** is displayed in
  the TUI and printed to stdout.
- **Rich TUI**: live display with four panels — *To-Dos* (remaining configs),
  *Current State* (config under test), *Progress* bar with ETA, and *Errors /
  Warnings* (critical errors in red, warnings in yellow).  Disable with
  `--no-tui` for CI or logging use.
- **Verbose logging** (`-v` / `-vv`): INFO or DEBUG output to stderr *and* a
  timestamped log file under the results directory.
- **Report generation** (`report`): JSONL to Markdown with ranked tables.

---

## Installation

```bash
# From the repository root
pip install -e .

# With dev dependencies (pytest, pytest-mock)
pip install -e ".[dev]"
```

The CLI entry point `llama-bench` is installed automatically.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python >= 3.10 | |
| `llama-server` binary | Build from [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| A `.gguf` model file | Any GGUF-format model |
| Vulkan drivers + `vulkaninfo` | Optional; enables GPU discovery display |
| `sudo` access | Required when `--sudo` is set (default) |

---

## Quick Start

### Run the adaptive tuner (Vulkan GPU, all devices)

```bash
llama-bench bench --model /path/to/model.gguf
```

### Run with a specific Vulkan device

```bash
llama-bench bench \
  --model /path/to/model.gguf \
  --engine vulkan \
  --vk-devices 0
```

### Run on CPU only

```bash
llama-bench bench \
  --model /path/to/model.gguf \
  --engine cpu \
  --n-gpu-layers 0
```

### Override the binary path and a few flags

```bash
llama-bench bench \
  --server /opt/llama.cpp/llama-server \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --n-gpu-layers 48 \
  --ctx 65536 \
  --flash-attn \
  --output results/codellama_bench.jsonl
```

### Control the tuning sweep bounds

```bash
llama-bench bench \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --ctx 98304 \
  --ctx-min 16384 \
  --ctx-step 16384 \
  --ngl-step 4 \
  --max-retries 3
```

### Add TTFT gating (optional)

```bash
llama-bench bench \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --max-ttft-s 30.0
```

### Run with verbose logging (no TUI)

Use `-v` for INFO-level logs and `-vv` for DEBUG-level logs.  Both are written
to stderr **and** to a timestamped log file under `results/`:

```bash
# INFO-level verbose, plain output (no TUI)
llama-bench bench \
  -v --no-tui \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --server /opt/llama.cpp/llama-server
```

### Generate a report

```bash
llama-bench report results/bench_*.jsonl --output results/report.md
```

---

## Full CLI Reference

### `llama-bench bench`

Adaptive configuration tuner: sweep context sizes and GPU-layer values,
apply error handling on OOM, and emit ranked results.

| Flag | Default | Description |
|------|---------|-------------|
| `--server`, `-s` | `./llama-server` | Path to llama-server binary |
| `--model`, `-m` | *(required)* | Path to model `.gguf` file |
| `--host` | `0.0.0.0` | Server bind address |
| `--port`, `-p` | `5001` | Server port |
| `--np` | `1` | Number of parallel request slots |
| `--ctx`, `-c` | `49152` | Starting (maximum) context size (tokens) |
| `--n-gpu-layers`, `-ngl` | `45` | Layers to offload to GPU |
| `--flash-attn` / `--no-flash-attn` | `True` | Flash Attention |
| `--batch-size` | `1536` | Logical batch size |
| `--ubatch-size` | `512` | Micro (physical) batch size |
| `--cache-type-k` | `q8_0` | K cache quantisation type |
| `--cache-type-v` | `q8_0` | V cache quantisation type |
| `--kv-unified` / `--no-kv-unified` | `True` | Unified KV cache pool |
| `--cache-reuse` | `512` | Min prefix tokens for cache reuse |
| `--cont-batching` / `--no-cont-batching` | `True` | Continuous batching |
| `--threads` | `8` | CPU inference threads |
| `--threads-batch` | `8` | CPU batch-processing threads |
| `--split-mode` | `none` | GPU split mode (`none`/`layer`/`row`) |
| `--engine` | `vulkan` | Inference engine: `vulkan` or `cpu` |
| `--vk-devices` | *(all devices)* | Comma-separated Vulkan device indices for `GGML_VK_VISIBLE_DEVICES`. If omitted, env var is not set. |
| `--sudo` / `--no-sudo` | `True` | Launch server with sudo |
| `--ctx-min` | `8192` | Minimum context size to try |
| `--ctx-max` | -- | Maximum context size (defaults to `--ctx`) |
| `--ctx-step` | `8192` | Context step between candidates |
| `--ngl-step` | `4` | GPU-layers reduction step on OOM |
| `--batch-step` | `256` | Batch-size reduction step on OOM |
| `--max-retries` | `5` | Max retry attempts per candidate on OOM |
| `--max-ttft-s` | *(disabled)* | Max acceptable TTFT (seconds). Disabled by default; TTFT recorded but does not gate usability. |
| `--min-tokens-per-sec` | `4.0` | Minimum decode throughput (tokens/s) |
| `--ctx-pct-threshold` | `0.90` | Fraction of max usable ctx for secondary throughput ranking |
| `--output`, `-o` | auto | JSONL output path (default: `results/bench_<timestamp>.jsonl`) |
| `--summary` | auto | Summary JSON path (default: `results/summary.json`) |
| `--no-tui` | `False` | Disable TUI, use plain output |
| `-v` / `--verbose` | off | Verbose logs to stderr + file (`-vv` for debug) |

### `llama-bench report`

```
llama-bench report <file1.jsonl> [file2.jsonl ...] [--output report.md] [--top-n 10]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | derived from input | Markdown output path |
| `--top-n` | `10` | Top results to include in table |

---

## Profile Files

Profile files (YAML) capture a named search space and benchmark configuration.
See [`profiles/reverse_engineering.yaml`](profiles/reverse_engineering.yaml) for an example.

```yaml
name: reverse_engineering
search_space:
  np: [1, 2]
  ctx: [49152, 98304]
  n_gpu_layers: [40, 45, 47]
  ...
benchmark:
  n_followups: 4
objective: minimize_e2e_latency
```

*Profile loading via CLI is on the roadmap; currently use `--ctx`, `--ctx-min`, etc.*

---

## How It Works

### Reverse-Engineering Workload

The benchmark uses a realistic reverse-engineering workload to stress-test
the server under representative conditions:

- **System prompt** (~1200 tokens): Expert-level reverse engineer persona with
  deep knowledge of assembly, binary analysis, heap exploitation, and debugger
  tooling (IDA Pro, Ghidra, lldb, Frida).
- **Corpus context**: Loaded from the repository prompt files — `README.md`,
  `agent.md`, and `artifacts.md` — and injected as a shared prefix in every
  prompt turn.  `solution.md` is intentionally excluded from the prompt
  corpus to avoid leaking answers.
- **Follow-up questions**: Each run includes an initial analysis request followed
  by follow-up questions (vulnerability identification, exploit PoC, debugger
  scripting, etc.) that re-send the same shared prefix, exercising KV-cache
  prefix reuse.

### Adaptive Tuning Loop

1. **Candidate generation**: Starting from `--ctx` (or `--ctx-max`) down to
   `--ctx-min` in `--ctx-step` increments, the tuner generates candidate
   configurations.
2. **Error handling**: When a run fails with VRAM OOM, the tuner automatically
   reduces `--n-gpu-layers` by `--ngl-step` and retries, up to `--max-retries`
   times.  If ngl hits the minimum, `--batch-size` is reduced by `--batch-step`
   before retrying.
3. **ctx big-step ladder**: When a memory-fit failure line reports that free memory
   is less than 50% of projected usage, the context is halved (big step) for the
   next retry instead of using the configured `--ctx-step`.  This avoids wasting
   many retries when the model clearly cannot fit.
4. **Early stopping**: A configuration is abandoned if it fails, causes OOM beyond
   the retry budget, or falls below `--min-tokens-per-sec`.
5. **Multi-objective selection**: Results are ranked by max usable context, then
   by throughput near that context level.  The `recommended` config balances
   both objectives.

### Engine Selection and Mismatch Detection

Pass `--engine vulkan` (default) to target Vulkan GPU acceleration, or
`--engine cpu` for CPU-only inference.

When `--engine vulkan` is active, llama-bench checks the server startup log for
Vulkan backend lines (e.g. `ggml_vulkan: ...`).  If none are found — meaning the
server binary was not built with Vulkan support or defaulted to a different
backend — an **engine mismatch** is flagged:

- In TUI mode: a **red** error entry is shown in the Errors panel.
- In `--no-tui` mode: a red warning is printed to the console.
- The `engine_mismatch` field is set to `true` in the JSONL result records.

The run continues regardless; the warning is informational.

### Usability Definition

A configuration is *usable* when **all** of the following hold:

- `tokens/s >= --min-tokens-per-sec` (default **4.0**)
- `ttft_s <= --max-ttft-s` — only checked when `--max-ttft-s` is explicitly set;
  **disabled by default** so TTFT alone never causes a rejection

TTFT is always recorded and contributes to ranking between equal-context configs.

### Terminal UI Panels

The TUI (disabled by `--no-tui`) shows these panels:

| Panel | Content |
|-------|---------|
| **To-Dos** | Remaining candidate configurations to test |
| **Current State** | Configuration currently under test |
| **Progress** | Progress bar with estimated time remaining |
| **Errors / Warnings** | Errors in **red**, warnings in yellow |
| **Results** | Rolling table of completed runs (tok/s, status, reason) |

### KV-Cache Prefix Reuse Probing

Follow-up messages re-send the same shared prefix (corpus context), allowing
`--cache-reuse` to kick in if the server has stored the prefix in its KV cache.
This makes the benchmark sensitive to the `--cache-reuse` threshold and exercises
cache-hit performance.

### Metrics Collection

- **Client metrics**: TTFT, end-to-end latency, streaming tok/s — measured by the
  client using `requests` with `stream=True`.
- **Server metrics**: prompt eval time, decode time, decode tok/s — parsed from the
  `llama-server` stderr log after each run.

---

## Output Format

Results are saved as **JSONL** (one JSON object per line) plus a `summary.json`.

Each JSONL line contains:

```json
{
  "run_id": "run_20240101T120000_abcd1234",
  "timestamp": "2024-01-01T12:00:00+00:00",
  "config_hash": "abcd1234",
  "success": true,
  "failure_reason": null,
  "engine_mismatch": false,
  "log_file": "results/llama_bench_20240101T120000.log",
  "client": {
    "ttft_ms": 142.3,
    "end_to_end_latency_ms": 8450.1,
    "streaming_tok_per_s": 28.4,
    "total_tokens": 240,
    "is_streaming": true
  },
  "server": {
    "prompt_eval_time_ms": 1230.5,
    "prompt_eval_count": 512,
    "eval_time_ms": 5430.2,
    "eval_count": 240,
    "prompt_tok_per_s": 415.7,
    "decode_tok_per_s": 44.2,
    "total_time_ms": 6660.7,
    "has_memory_info": true,
    "memory_used_mb": 14234.6
  }
}
```

The `summary.json` contains the multi-objective selection: `max_ctx_result`,
`top_throughput`, and `recommended` configuration.

---

## Security Note

By default, `llama-bench` launches `llama-server` with `sudo` (`--sudo` flag).
This is because GPU memory operations may require elevated privileges on some
systems.

- Use `--no-sudo` if your user already has the necessary permissions.
- Never run untrusted binaries with sudo.
- The sudo command runs the server binary directly; no shell expansion is used.

### Running `llama-bench` itself under sudo

`llama-bench` is installed as a console script inside the active virtual
environment (e.g. `~/.local/share/.../venv/bin/llama-bench`).  When you invoke
`sudo llama-bench`, sudo's restricted `PATH` will often **not** include the
venv's `bin/` directory, causing a *"command not found"* error.

**Recommended workaround — preserve PATH:**

```bash
sudo env PATH="$PATH" llama-bench bench --server /absolute/path/to/llama-server ...
```

**Best practice:** only the `llama-server` binary needs elevated privileges.
`llama-bench` itself does not need to run as root; use `--sudo` (the default)
so only the server subprocess is elevated, and invoke `llama-bench` as your
regular user.

---

## Version Compatibility

The tool is tested against llama.cpp build **8133** (`EXPECTED_LLAMA_SERVER_VERSION`
in `llama_bench/__init__.py`).

Version detection runs `llama-server --version` **without** sudo and parses a
line of the form:

```
version: 8133 (2b6dfe824)
```

Only the **numeric build number** (e.g. `"8133"`) is extracted and compared against
`EXPECTED_LLAMA_SERVER_VERSION`.  If they differ, a warning is printed:

```
Version warning: llama-server version mismatch: expected '8133', got '9000'.
Results may differ from baseline.
```

The benchmark will still run regardless; the warning is informational only.

---

## Troubleshooting: Server Startup Failures

`llama-bench` classifies server startup failures into four categories and
surfaces them during the tuning loop.

### Failure types

| `failure_reason` | Meaning | Common cause |
|---|---|---|
| `model_not_found` | llama-server could not open the model file | Wrong `--model` path, typo, or the file was deleted |
| `out_of_vram` | GPU memory allocation failed | Too many layers offloaded (`--n-gpu-layers`), model too large for VRAM |
| `server_exited` | Server process exited before `/health` returned 200, with no recognised stderr pattern | Misconfigured flags, missing shared libraries |
| `server_startup_timeout` | Server did not respond to `/health` within 90 seconds | Very large model, slow disk I/O, system under heavy load |

### How failures are detected

When a server startup failure occurs `llama-bench`:

1. Polls `GET /health` once per second and simultaneously checks whether the
   server process is still alive.
2. If the process exits early, polling stops immediately (no repeated connection
   attempts against a dead process).
3. Reads the tail of the server stderr log and matches it against known error
   patterns to produce a structured `failure_reason`.
4. Stores the `stderr_path`, `stdout_path`, and a short `server_error_excerpt`
   in the JSONL result record for inspection.

### Adaptive OOM handling

When `failure_reason` is `out_of_vram`, the tuner automatically backs off
`--n-gpu-layers` by `--ngl-step` and retries the same context size, up to
`--max-retries` times.

When the server reports a fit-failure line (projected vs. free memory), the tuner
uses a **big-step ladder**: if free memory is less than 50% of projected usage, the
context is halved for the retry; otherwise a proportional reduction is applied.

### `model_not_found` is a critical failure

If any attempt fails with `model_not_found`, the tuner exits immediately with
an actionable error message and the path to the server stderr log.

### Practical tips

* **`model_not_found`** — pass an absolute path to `--model` and verify the
  file exists: `ls -lh /path/to/model.gguf`.
* **`out_of_vram`** — reduce `--n-gpu-layers` starting value or widen the
  `--ngl-step` / `--max-retries` budget so the tuner has more room to adapt.
  Check free VRAM with `vulkaninfo | grep -i memory`.
* **Engine mismatch warning** — the server binary does not have Vulkan support
  compiled in, or the Vulkan ICD is missing.  Try `vulkaninfo` to check Vulkan
  availability, or rebuild `llama-server` with Vulkan support.
* **`server_exited`** — inspect the full stderr log for clues.  Common culprits:
  wrong binary for the hardware, missing Vulkan ICD.
* **`server_startup_timeout`** — the model may be loading from a slow storage
  device.  The startup timeout is currently fixed at 90 seconds.
