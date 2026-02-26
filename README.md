# llama-bench

**llama-bench** is a command-line benchmarking and parameter-search tool for
[llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server`. It measures
end-to-end latency, time-to-first-token (TTFT), and decode throughput for realistic
workloads, then helps you find the best combination of server flags for your hardware
and use-case.

---

## Table of Contents

1. [Why llama-bench](#why-llama-bench)
2. [Installation](#installation)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Full CLI Reference](#full-cli-reference)
6. [4-Preset Characterization Bench](#4-preset-characterization-bench)
7. [Goal Presets & Scoring](#goal-presets--scoring)
8. [TUI Dashboard](#tui-dashboard)
9. [Graceful Shutdown](#graceful-shutdown)
10. [Hardware Monitoring](#hardware-monitoring)
11. [Vulkan GPU Discovery](#vulkan-gpu-discovery)
12. [Prompt Workload](#prompt-workload)
13. [Custom Prompt Packs](#custom-prompt-packs)
14. [Profile Files](#profile-files)
15. [OOM-Adaptive Retry Strategy](#oom-adaptive-retry-strategy)
16. [Output Formats](#output-formats)
17. [Verbose Logging](#verbose-logging)
18. [Security Notes](#security-notes)
19. [Version Compatibility](#version-compatibility)
20. [Troubleshooting](#troubleshooting)
21. [Architecture](#architecture)

---

## Why llama-bench

`llama-bench` answers the question: *what combination of `llama-server` flags gives the
lowest latency and highest throughput for my model, GPU, and workload?*

Running llama-server by hand with trial-and-error is tedious. Picking random flags leaves
performance on the table. llama-bench automates the sweep, handles OOM failures gracefully,
measures real workloads (not synthetic token counts), and surfaces multi-objective results
so you can make an informed choice — all without writing a single line of code.

**Key features:**

- **`bench`** — 4-preset characterization bench: runs four independent performance
  sweeps (max context, fastest response, throughput, long-context RAG), then applies
  user-selected goal weights (20 points across 4 presets) to compute a normalized
  weighted score and recommend the optimal `(ngl, batch)` configuration for your
  specific use-case.
- **`search`** — Staged two-phase parameter search: coarse sweep over the full config
  space, then refines the top 25% for higher-fidelity measurements. Supports early
  stopping.
- **`explore`** — Continuous multi-objective explorer: runs indefinitely, sweeping the
  full `(ctx × ngl × batch)` grid and maintaining a live **Hall of Fame** across five
  objectives.
- **`report`** — Offline report generator: converts JSONL result files into ranked
  Markdown tables.
- **Rich TUI** — 11-panel full-screen dashboard with live hardware monitors, activity
  log, settings display, results table, Hall of Fame, progress bar, server log tail,
  and verbose log panel.
- **Reverse-engineering workload** — Built-in ~1200-token system prompt + shared
  binary-analysis prefix exercising KV-cache prefix reuse heavily. Corpus files from
  the repo are included when available.
- **Vulkan GPU auto-discovery** — Detects GPUs via `vulkaninfo` and sets
  `GGML_VK_VISIBLE_DEVICES` automatically.
- **Graceful shutdown** — `q` or `Ctrl+C` stops all processes, then generates a final
  report from whatever partial results were collected.

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
| Python ≥ 3.10 | |
| `llama-server` binary | Build from [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| A `.gguf` model file | Any GGUF-format model |
| Vulkan drivers + `vulkaninfo` | Optional; enables GPU auto-discovery |
| `sudo` access | Required when `--sudo` is set (default) |

---

## Quick Start

### Run a preset characterization bench (default goal: `general`)

```bash
llama-bench bench --model /path/to/model.gguf
```

Target a specific use-case — e.g. coding assistant (fast TTFT matters most):

```bash
llama-bench bench \
  --server /opt/llama.cpp/llama-server \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --goal coding \
  --n-gpu-layers 48 \
  --flash-attn \
  --output results/codellama_coding.jsonl
```

Use a custom 20-point weight budget across all four presets:

```bash
llama-bench bench \
  --model /data/models/my-model.gguf \
  --goal custom \
  --w-max-context 8 \
  --w-fastest-response 4 \
  --w-throughput 4 \
  --w-long-context-rag 4
```

### Run a staged parameter search

```bash
llama-bench search \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --np-tests 1-2 \
  --ctx-tests 49152,98304 \
  --ngl-tests 40,45,47 \
  --max-configs 30
```

### Run the continuous multi-objective explorer

Sweeps `(ctx, ngl, batch)` indefinitely, updating a live Hall of Fame.
Press `q` or `Ctrl+C` to stop — results are saved automatically.

```bash
# Vary only context (single ngl/batch)
llama-bench explore \
  --model /data/models/my-model.gguf \
  --ctx 131072 --ctx-min 8192 --ctx-step 8192

# Full grid sweep with multiple ngl and batch values (multi-GPU)
llama-bench explore \
  --model /data/models/my-model.gguf \
  --ctx 131072 --ctx-min 32768 \
  --ngl-tests 37,41,45 \
  --batch-tests 1024,1536 \
  --vk-devices 0,1
```

### Generate a Markdown report from results

```bash
llama-bench report results/bench_*.jsonl --output results/report.md
```

---

## Full CLI Reference

### `llama-bench bench`

Runs the **4-preset characterization bench**. Each preset sweeps its own parameter
space independently, then the results are combined using a 20-point goal-weight budget
to recommend the optimal `(ngl, batch)` server configuration for your use-case.

**Preset-specific flags** (new):

| Flag | Default | Description |
|------|---------|-------------|
| `--goal` | `general` | Optimization goal: `reverse_engineering`, `coding`, `chatting`, `rag_research`, `general`, or `custom` |
| `--w-max-context` | `5` | Weight for max-context phase (`--goal custom` only; all 4 weights must sum to 20) |
| `--w-fastest-response` | `5` | Weight for fastest-response phase (`--goal custom` only) |
| `--w-throughput` | `5` | Weight for throughput phase (`--goal custom` only) |
| `--w-long-context-rag` | `5` | Weight for long-context RAG phase (`--goal custom` only) |
| `--ngl-tests` | base ngl only | Comma-separated ngl values to sweep in grid presets, e.g. `40,43,45` |
| `--batch-tests` | base batch only | Comma-separated batch-size values to sweep, e.g. `1024,1536` |

**Shared server flags** (inherited by all modes):

| Flag | Default | Description |
|------|---------|-------------|
| `--server`, `-s` | `./llama-server` | Path to llama-server binary (resolved to absolute path) |
| `--model`, `-m` | *(required)* | Path to model `.gguf` file |
| `--host` | `0.0.0.0` | Server bind address |
| `--port`, `-p` | `5001` | Server port |
| `--np` | `1` | Number of parallel request slots (`-np` in llama-server) |
| `--ctx`, `-c` | `49152` | Starting / maximum context size (tokens) |
| `--ctx-max` | `--ctx` value | Explicit maximum context override |
| `--ctx-min` | `8192` | Minimum context size to try |
| `--ctx-step` | `8192` | Context step between candidates |
| `--n-gpu-layers`, `-ngl` | `45` | Layers to offload to GPU |
| `--ngl-step` | `4` | ngl reduction step on OOM retry |
| `--batch-step` | `256` | `batch-size` reduction step on OOM retry |
| `--max-retries` | `5` | Max retry attempts per candidate config on OOM |
| `--flash-attn` / `--no-flash-attn` | `True` | Flash Attention |
| `--batch-size` | `1536` | Logical batch size |
| `--ubatch-size` | `512` | Micro (physical) batch size |
| `--cache-type-k` | `q8_0` | K cache quantisation type |
| `--cache-type-v` | `q8_0` | V cache quantisation type |
| `--kv-unified` / `--no-kv-unified` | `True` | Unified KV cache pool |
| `--cache-reuse` | `512` | Min prefix tokens for KV cache reuse |
| `--cont-batching` / `--no-cont-batching` | `True` | Continuous batching |
| `--threads` | `8` | CPU inference threads |
| `--threads-batch` | `8` | CPU batch-processing threads |
| `--split-mode` | `none` | GPU split mode: `none`, `layer`, or `row` |
| `--vk-devices` | auto | `GGML_VK_VISIBLE_DEVICES` value (auto-discovered if omitted) |
| `--sudo` / `--no-sudo` | `True` | Launch server with sudo |
| `--max-ttft` | `60.0` | Maximum acceptable TTFT (seconds) — configs exceeding this are skipped |
| `--min-tokens-per-sec` | `1.0` | Minimum acceptable decode throughput |
| `--ctx-pct-threshold` | `0.90` | Fraction of max usable ctx for secondary throughput ranking |
| `--n-followups` | `4` | Follow-up prompts per run |
| `--max-tokens` | `512` | Max tokens generated per request (controls decode length) |
| `--prompt-pack` | — | Path to a custom prompt pack file (JSON/YAML) |
| `--output`, `-o` | `results/bench_<ts>.jsonl` | JSONL output path |
| `--summary` | `results/summary.json` | Summary JSON output path |
| `--no-tui` | `False` | Disable TUI, use plain terminal output |
| `-v` / `-vv` | off | Verbose (INFO) / debug (DEBUG) logging to stderr + file |

### `llama-bench search`

Run a staged two-phase parameter search over a config space. Inherits all `bench` flags, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--np-tests` | `1` | Range spec for `-np` values, e.g. `1-2` or `1,2` |
| `--ctx-tests` | `49152` | Range spec for `-c` values |
| `--ngl-tests` | `45` | Range spec for `--n-gpu-layers` values |
| `--max-configs` | `50` | Maximum number of configs to evaluate in Phase 1 |

**Range spec formats:**

| Spec | Expands to |
|------|-----------|
| `"1-4"` | `[1, 2, 3, 4]` |
| `"1,2,4"` | `[1, 2, 4]` |
| `"49152"` | `[49152]` |

### `llama-bench explore`

Runs **indefinitely** until you press `q` or `Ctrl+C`. Each round sweeps the full
`ctx × ngl × batch` Cartesian product, updates the Hall of Fame after every run,
and appends results to the JSONL file immediately.

| Flag | Default | Description |
|------|---------|-------------|
| `--server`, `-s` | `./llama-server` | Path to llama-server binary |
| `--model`, `-m` | *(required)* | Path to model `.gguf` file |
| `--host` | `0.0.0.0` | Server bind address |
| `--port`, `-p` | `5001` | Server port |
| `--np` | `1` | Number of parallel request slots |
| `--ctx`, `-c` | `131072` | Maximum context size for the sweep |
| `--ctx-min` | `8192` | Minimum context size to try |
| `--ctx-step` | `8192` | Context step between candidates |
| `--n-gpu-layers`, `-ngl` | `45` | Starting GPU layers (used when `--ngl-tests` is omitted) |
| `--ngl-tests` | — | Comma-separated ngl values to sweep, e.g. `37,41,45` |
| `--flash-attn` / `--no-flash-attn` | `True` | Flash Attention |
| `--batch-size` | `1536` | Starting batch size (used when `--batch-tests` is omitted) |
| `--batch-tests` | — | Comma-separated batch values to sweep, e.g. `1024,1536` |
| `--ubatch-size` | `512` | Micro batch size |
| `--cache-type-k` | `q8_0` | K cache quantisation type |
| `--cache-type-v` | `q8_0` | V cache quantisation type |
| `--kv-unified` / `--no-kv-unified` | `True` | Unified KV cache pool |
| `--cache-reuse` | `512` | Min prefix tokens for KV cache reuse |
| `--cont-batching` / `--no-cont-batching` | `True` | Continuous batching |
| `--threads` | `8` | CPU inference threads |
| `--threads-batch` | `8` | CPU batch-processing threads |
| `--split-mode` | `none` | GPU split mode: `none`, `layer`, or `row` |
| `--vk-devices` | auto | `GGML_VK_VISIBLE_DEVICES` value |
| `--sudo` / `--no-sudo` | `True` | Launch server with sudo |
| `--ngl-step` | `4` | ngl reduction step on OOM retry |
| `--batch-step` | `256` | Batch size reduction step on OOM retry |
| `--ngl-min` | `0` | Minimum ngl value during OOM retry |
| `--max-retries` | `5` | Max OOM retry attempts per candidate |
| `--max-ttft` | `60.0` | Maximum acceptable TTFT (seconds) |
| `--min-tokens-per-sec` | `1.0` | Minimum acceptable throughput |
| `--n-followups` | `2` | Follow-up prompts per run (lower = faster sweeps) |
| `--max-tokens` | `512` | Max tokens generated per request |
| `--output`, `-o` | `results/explore_<ts>.jsonl` | JSONL output path |
| `--no-tui` | `False` | Disable TUI, use plain output |
| `-v` / `-vv` | off | Verbose / debug logging |

**Hall of Fame objectives:**

| Objective | What is optimised |
|-----------|-------------------|
| Max Context | Highest `ctx` that completed successfully |
| Fastest TTFT | Lowest cold time-to-first-token |
| Best Warm TTFT | Lowest warm TTFT (KV cache prefix hit on follow-up requests) |
| Best Throughput | Highest decode tokens/s |
| Best Overall | Balanced score: `√ctx × tok/s ÷ (1 + cold_ttft)` |

### `llama-bench report`

Generate a Markdown report from one or more JSONL result files.

```
llama-bench report <file1.jsonl> [file2.jsonl ...] [--output report.md] [--top-n 10]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `<input>_report.md` | Markdown output path |
| `--top-n` | `10` | Number of top results to include in the ranked table |

---

## 4-Preset Characterization Bench

`bench` mode runs four independent sweeps to characterize your hardware and model across
four performance dimensions, then computes a weighted score to surface the optimal
`(n-gpu-layers, batch-size)` pair for your target use-case.

### The 4 Presets

**1. `max_context`** — *How much context can my hardware actually hold?*

- Sweeps context sizes high → low, retrying on OOM (adaptive ngl/batch reduction)
- Short prompt, `max_tokens=32`, no follow-ups
- **Primary metric**: highest successful context size
- Results feed into `long_context_rag` to skip known-OOM context sizes

**2. `fastest_response`** — *What is the minimum time-to-first-token?*

- Fixed `ctx=4096`, sweeps the `ngl × batch` grid (via `--ngl-tests` / `--batch-tests`)
- Short prompt (~300 tokens), `max_tokens=128`, no follow-ups
- **Primary metric**: cold TTFT (lower is better)

**3. `throughput_king`** — *What is the peak decode throughput?*

- Fixed `ctx=8192`, sweeps the `ngl × batch` grid
- Short input, `max_tokens=2048`, no follow-ups
- **Primary metric**: tokens/second (higher is better)

**4. `long_context_rag`** — *How fast is the first-token at near-max context?*

- Context candidates: `[32768, 65536, 131072]` (skips sizes that failed `max_context`)
- Synthetic padding (lorem ipsum) fills the context; short question appended at end
- `max_tokens=256`, 1 follow-up
- **Primary metric**: cold TTFT at the largest successful context (lower is better)

### Execution Order

```
max_context → fastest_response → throughput_king → long_context_rag
```

Presets with weight=0 are skipped entirely (saves runtime).

---

## Goal Presets & Scoring

After all presets complete, results are combined using a **20-point weight budget** that
you allocate across the four presets. Each preset's raw metric is min-max normalized,
inverted where lower-is-better, then weighted:

```
final_score(ngl, batch) = Σ( weight_i × normalized_i ) / 20
```

The `(ngl, batch)` pair with the highest final score is the recommended server configuration.

### Builtin Goal Presets

| Goal | max_context | fastest_response | throughput | long_context_rag |
|------|-------------|-----------------|-----------|-----------------|
| `reverse_engineering` | 12 | 1 | 3 | 4 |
| `coding` | 3 | 10 | 5 | 2 |
| `chatting` | 3 | 8 | 5 | 4 |
| `rag_research` | 4 | 2 | 2 | 12 |
| `general` *(default)* | 5 | 5 | 5 | 5 |
| `custom` | set via flags | set via flags | set via flags | set via flags |

All builtin weights sum to exactly 20. The `custom` goal requires you to set all four
`--w-*` flags such that they sum to 20 — the CLI rejects runs where they do not.

### Selecting a Goal

Via the interactive menu:
```
llama-bench          # opens the TUI menu — pick bench mode, then select a goal
```

Via CLI:
```bash
llama-bench bench --model /path/to/model.gguf --goal coding

# Custom weights (must sum to 20)
llama-bench bench --model /path/to/model.gguf --goal custom \
  --w-max-context 10 --w-fastest-response 5 --w-throughput 3 --w-long-context-rag 2
```

### Bench Output

On completion, `bench` writes:
- **JSONL** — one record per attempt across all presets (`--output`)
- **Markdown report** — preset summaries + scoring table + recommendation
- **Summary JSON** — machine-readable optimal config (`--summary`)

---

## TUI Dashboard

All three running commands (`bench`, `search`, `explore`) display the same full-screen
[Textual](https://textual.textualize.io/) dashboard. Press `q` or `Ctrl+C` at any time
to stop — a final report is always generated before the process exits.

| Panel | Description |
|-------|-------------|
| **Current Run** | Active phase label, current config (ctx/ngl/batch), elapsed time, ETA |
| **Hardware** | Per-GPU VRAM bar + utilisation % + temperature; CPU %; RAM used/total |
| **Settings** | Configuration snapshot: model name, context sweep range, NGL/batch/µbatch, Flash Attention, KV-Unified, continuous batching flags, KV cache types and reuse threshold, split mode, Vulkan devices, thread counts |
| **Server Command** | Full `llama-server` invocation as actually executed — never truncated |
| **Activity** | Timestamped event log: server start, health-poll progress, prompt sends, results received, retry events |
| **Progress** | Sweep progress bar with candidate count and ETA |
| **Results** | Rolling table of completed runs: ctx, ngl, batch, cold/warm TTFT, tok/s sparkline trend, peak VRAM |
| **Best Configs (Hall of Fame)** | Live best-per-objective table — updated after every `explore` run; shows informational hint for `bench`/`search` |
| **Retries** | OOM/failure retry history: reason, parameter reduction applied |
| **Server Log** | Live tail of `llama-server` stderr |
| **Verbose Log** | Tail of the bench tool's own log file (visible only when `-v`/`-vv` is passed) |

---

## Graceful Shutdown

Pressing `q` in the TUI, or `Ctrl+C` in any mode, triggers a graceful shutdown sequence:

1. The `stop_event` is set, signalling all workers to finish after the current run.
2. If the server is mid-request, the current request completes before shutdown (no truncated measurements).
3. The `llama-server` subprocess is terminated; if it does not exit within the timeout, a SIGKILL is sent (and the process group is killed to catch sudo children).
4. Any partial results accumulated up to that point are collected via `accumulated_attempts` / `accumulated_results`.
5. A final report is generated and written to disk — JSONL, summary JSON, and a console summary table — exactly as if the run had completed normally.

A second `Ctrl+C` force-kills immediately (the original signal handler is restored after the first press).

---

## Hardware Monitoring

The `HWMonitor` polls hardware metrics in a background thread and pushes updates to the TUI.

| Source | Metrics |
|--------|---------|
| **NVIDIA GPUs** | VRAM used/total (MiB), GPU utilisation %, temperature °C — via `nvidia-smi` |
| **AMD / Vulkan GPUs** | VRAM used/total (MiB), utilisation % — via `rocm-smi` or Vulkan extensions |
| **CPU** | Per-core and aggregate utilisation % — via `psutil` |
| **RAM** | Used / total memory (GiB) — via `psutil` |

Monitoring is optional — if no GPU tool is available the hardware panel shows CPU/RAM only.

---

## Vulkan GPU Discovery

When `--vk-devices` is not explicitly set, llama-bench runs `vulkaninfo` at startup to
enumerate available GPUs:

```
Vulkan GPUs discovered: 2
  [0] AMD Radeon RX 7900 XTX
  [1] AMD Radeon RX 7900 XTX
```

The discovered device indices are joined (e.g. `"0,1"`) and passed as
`GGML_VK_VISIBLE_DEVICES` in the server environment.

When multiple Vulkan devices are detected and `--split-mode` is still `none`, the tool
automatically upgrades it to `layer` and prints:

```
Auto-set split-mode=layer for multi-GPU
```

To override, pass `--vk-devices 0` (single GPU) or `--split-mode row` explicitly.

---

## Prompt Workload

The built-in workload is designed to stress-test KV-cache prefix reuse and realistic
multi-turn inference, not just token counting.

**System prompt** (~1200 tokens): A detailed reverse-engineer persona covering x86-64/ARM64
assembly, binary analysis tooling (IDA Pro, Ghidra, Binary Ninja, lldb, Frida), heap
exploitation, ROP chains, and vulnerability classes.

**Shared prefix / corpus context**: The initial user message and every follow-up re-send
the same long shared context (corpus files from the repo, or the built-in ~600-token
firmware binary decompilation pseudocode as fallback). This allows `--cache-reuse` to
trigger on follow-up requests, making the benchmark sensitive to the cache reuse threshold.

**Corpus files** (loaded from repo root when available):
- `README.md`
- `agent.md`
- `artifacts.md`

`solution.md` is intentionally excluded.

**Follow-up questions** (cycled, up to `--n-followups`):

1. Root cause vulnerability and CWE mapping
2. Exploit proof-of-concept in Python/pwntools
3. Stack canary bypass via the netlink call path
4. TLV parsing integer-truncation risks
5. lldb conditional breakpoint for specific tlv values
6. Frida Interceptor hook for argument logging
7. Exploit mitigations analysis (RELRO, NX, PIE, CET)
8. Reconstruct C source for `process_fw_chunk`

**Cold vs warm TTFT**: The first request in a sequence is the *cold* TTFT (no KV cache
hit). Subsequent follow-ups are *warm* TTFT (the shared prefix is in cache). The
difference reflects how well the server's prefix cache is working.

---

## Custom Prompt Packs

You can replace the built-in workload with your own prompts via `--prompt-pack`:

```bash
llama-bench bench --model /path/to/model.gguf --prompt-pack my_prompts.yaml
```

The file must be a JSON or YAML list of message-sequence dicts matching the format
produced by `build_prompt_sequence`:

```yaml
- messages:
    - role: system
      content: "You are a helpful assistant."
    - role: user
      content: "Explain quantum entanglement in one paragraph."
  is_followup: false
  expected_prefix_len_tokens: 150

- messages:
    - role: system
      content: "You are a helpful assistant."
    - role: user
      content: "Now give me a concrete experiment that demonstrates it."
  is_followup: true
  expected_prefix_len_tokens: 150
```

Both `.json` and `.yaml` / `.yml` extensions are supported. The top-level value must be
a list. `is_followup` controls whether the measurement is recorded as warm or cold TTFT.

---

## Profile Files

Profile YAML files capture a named search space and benchmark configuration for
repeatable runs. See [`profiles/reverse_engineering.yaml`](profiles/reverse_engineering.yaml)
for an example.

```yaml
name: reverse_engineering
search_space:
  np: [1, 2]
  ctx: [49152, 98304]
  n_gpu_layers: [40, 45, 47]
  batch_size: [1024, 1536]
benchmark:
  n_followups: 4
  max_tokens: 512
objective: minimize_e2e_latency
```

*Profile loading via CLI is planned for a future release. Currently, pass values
directly with `--np-tests`, `--ctx-tests`, `--ngl-tests`, `--batch-tests`, etc.*

---

## OOM-Adaptive Retry Strategy

When a server startup fails with `out_of_vram`, llama-bench does not give up — it
automatically reduces parameters and retries the same context size (up to `--max-retries`
attempts):

1. **ngl reduction**: reduce `n-gpu-layers` by `--ngl-step` (default 4).
   If the server emitted a memory-fit log line (e.g. `llama_kv_cache_init: failed,
   need X MiB, have Y MiB`), the ngl reduction is computed from the ratio `Y/X`
   instead, for a smarter step size.
2. **batch reduction**: once ngl reaches `--ngl-min`, reduce `batch-size` by
   `--batch-step` (default 256).
3. **Give up on candidate**: if max retries are exhausted or batch-size drops below
   the minimum (256), the candidate is marked failed and the sweep moves on.
4. **Context step-down**: after exhausting retries at one context size, the tuner
   moves to the next (lower) context size and resets ngl/batch to their starting
   values.

This means that on a machine where `ctx=131072, ngl=45` OOMs, llama-bench will
automatically find the highest context/ngl combination that actually fits.

---

## Output Formats

### JSONL (bench / explore)

Results are appended as one JSON object per line. Each record contains:

```json
{
  "run_id": "tune_2024-01-01T12:00:00+00:00",
  "timestamp": "2024-01-01T12:00:00+00:00",
  "config": {
    "server": "/path/to/llama-server",
    "model": "/path/to/model.gguf",
    "ctx": 49152,
    "n_gpu_layers": 41,
    "batch_size": 1536,
    "ubatch_size": 512,
    "flash_attn": true,
    "kv_unified": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "cache_reuse": 512,
    "cont_batching": true,
    "split_mode": "layer",
    "vk_devices": "0,1",
    "np": 1
  },
  "success": true,
  "failure_reason": null,
  "ttft_s": 0.215,
  "warm_ttft_s": 0.031,
  "tokens_per_sec": 20.4,
  "ctx": 49152,
  "n_gpu_layers": 41,
  "batch_size": 1536,
  "stderr_path": "results/server_stderr_20240101T120000.log",
  "stdout_path": "results/server_stdout_20240101T120000.log",
  "server_error_excerpt": null,
  "projected_mib": null,
  "free_mib": null,
  "log_file": "results/llama_bench_20240101T120000.log"
}
```

### Summary JSON (bench only)

Written to `results/summary.json` (or `--summary`). Contains:

```json
{
  "max_ctx_result": { ... },
  "top_throughput": [ { ... }, ... ],
  "recommended": { ... }
}
```

- `max_ctx_result` — the successful attempt with the highest context size.
- `top_throughput` — up to 5 configs with the highest tok/s within
  `ctx_pct_threshold × max_ctx` (default 90%).
- `recommended` — the config with the best balance of context, TTFT, and throughput.

### JSONL (search)

Search results wrap each attempt inside a `metrics` array and include additional fields:

```json
{
  "config": { ... },
  "metrics": [ { ... }, ... ],
  "phase": 1,
  "best_score": 1.234
}
```

### JSONL (explore)

Explorer results use the same per-attempt schema as `bench`, appended one line at a time.
The Hall of Fame is displayed live in the TUI and printed to the console on exit, but is
not persisted to JSONL.

### Markdown report

Generated by `llama-bench report` (or automatically at session end). Contains:

- Run metadata header (model, timestamp, flags)
- Ranked results table sorted by end-to-end latency
- Multi-objective summary: max context, top throughput, recommended config

---

## Verbose Logging

Use `-v` (INFO) or `-vv` (DEBUG) to enable detailed logging. Logs are written to:

- **stderr** (or the TUI verbose log panel when TUI is active)
- A timestamped file: `results/llama_bench_<timestamp>.log`

The log file path is printed at startup and recorded in each JSONL result as `log_file`.

Verbose logs include:

- Resolved absolute server binary path
- Version check result and any mismatch warning
- Server start command, PID, and log file paths
- Health-check polling progress (one line per poll)
- Each benchmark request: TTFT, end-to-end latency, tok/s
- Server log parsing summary (lines matched for failure classification)

```bash
# INFO-level verbose, plain output
llama-bench bench -v --no-tui --model /path/to/model.gguf

# DEBUG-level verbose, plain output
llama-bench bench -vv --no-tui --model /path/to/model.gguf

# DEBUG-level verbose with TUI (logs go to file only; TUI is shown in terminal)
llama-bench bench -vv --model /path/to/model.gguf
```

---

## Security Notes

By default, `llama-bench` launches `llama-server` with `sudo` (`--sudo` is `True`).
This is required on some systems where GPU memory allocation needs elevated privileges.

- Use `--no-sudo` if your user already has the necessary GPU permissions.
- Never run untrusted binaries with sudo.
- The sudo invocation runs the binary directly — no shell expansion is used.

### Running `llama-bench` itself under sudo

`llama-bench` is installed as a console script inside the active virtual environment
(e.g. `~/.local/share/.../venv/bin/llama-bench`). When you invoke `sudo llama-bench`,
sudo's restricted `PATH` will often **not** include the venv's `bin/` directory, causing
a *"command not found"* error.

**Recommended workaround — preserve PATH:**

```bash
sudo env PATH="$PATH" llama-bench bench --server /absolute/path/to/llama-server ...
```

**Alternative — system-wide symlink:**

```bash
sudo ln -s "$(which llama-bench)" /usr/local/bin/llama-bench
```

After creating the symlink, plain `sudo llama-bench ...` works without the wrapper.

**Best practice:** only the `llama-server` binary needs elevated privileges. Run
`llama-bench` as your regular user with `--sudo` (the default) so only the server
subprocess is elevated.

---

## Version Compatibility

The tool is tested against llama.cpp build **8133** (`EXPECTED_LLAMA_SERVER_VERSION`
in `llama_bench/__init__.py`).

Version detection runs `llama-server --version` **without** sudo and parses a line of the form:

```
version: 8133 (2b6dfe824)
```

Only the numeric build number is extracted and compared. If they differ, a warning is
printed but the benchmark runs regardless:

```
Version warning: llama-server version mismatch: expected '8133', got '9000'.
Results may differ from baseline.
```

The `--server` path is always resolved to an absolute path before version detection, so
`sudo: ./llama-server: command not found` errors are never mis-parsed as version strings.

To suppress the warning, update `EXPECTED_LLAMA_SERVER_VERSION` in
`llama_bench/__init__.py` to match your build number.

---

## Troubleshooting

### Server startup failure types

`llama-bench` classifies every server startup failure and records it in the JSONL output:

| `failure_reason` | Meaning | Common cause |
|---|---|---|
| `model_not_found` | llama-server could not open the model file | Wrong `--model` path, typo, or file deleted |
| `out_of_vram` | GPU memory allocation failed | Too many layers offloaded, model too large for VRAM |
| `server_exited` | Server process exited before `/health` returned 200, with no recognised stderr pattern | Misconfigured flags, missing shared libraries, wrong binary for hardware |
| `server_startup_timeout` | Server did not respond to `/health` within 90 seconds | Very large model, slow disk I/O, system under heavy load |

### How failures are detected

When a startup failure occurs:

1. `GET /health` is polled once per second while simultaneously checking whether the process is still alive.
2. If the process exits early, polling stops immediately (no retries against a dead process).
3. The tail of the server stderr log is read and matched against known error patterns to produce a structured `failure_reason`.
4. `stderr_path`, `stdout_path`, and a short `server_error_excerpt` are stored in the JSONL record for later inspection.

### Behaviour per mode

**`bench` mode**: exits with a non-zero code and prints an actionable error, e.g.:

```
Error: Server startup failed (model_not_found).
  Server stderr log: results/server_stderr_20260101T120000.log
  Excerpt:
    gguf_init_from_file: failed to open GGUF file /bad/path.gguf (No such file or directory)
```

The JSONL result file is still written.

**`search` mode**: startup failures are treated as expected outcomes — the failing
configuration is marked `success=false` and the search continues. An OOM failure at
high `--n-gpu-layers` will not abort the whole search.

**`explore` mode**: same as `search` — failures are recorded and the explorer moves
on to the next candidate.

### Config validation warnings

`llama-bench` validates the configuration before starting and prints warnings for:

- `ubatch-size` > `batch-size` (micro batch larger than logical batch)
- `cache-reuse` ≥ `ctx` (reuse threshold at least as large as context)
- Unusual cache quantisation type combinations

These are warnings, not errors — the run proceeds.

### Server path resolution

`--server` is resolved in this order:

1. `~` expansion (e.g. `~/bin/llama-server`)
2. Relative path → absolute (`./llama-server` → `/cwd/llama-server`)
3. PATH lookup for bare names (`llama-server` → `$(which llama-server)`)

If the resolved path does not exist or is not executable, the tool prints a clear error
and exits before attempting to start anything.

### Practical tips

- **`model_not_found`** — pass an absolute path to `--model` and verify: `ls -lh /path/to/model.gguf`
- **`out_of_vram`** — reduce `--n-gpu-layers` (or let `search`/`explore` find the max automatically). Check free VRAM with `vulkaninfo | grep -i memory`
- **`server_exited`** — inspect the full stderr log. Common culprits: CUDA binary on a Vulkan-only machine, missing Vulkan ICD, wrong shared library paths
- **`server_startup_timeout`** — startup timeout is fixed at 90 seconds. If loading from slow storage, consider a smaller model or a faster drive
- **No Vulkan GPUs discovered** — `vulkaninfo` is not installed or not on `PATH`. Install the Vulkan SDK or pass `--vk-devices 0` manually

---

## Architecture

```
llama_bench/
├── __init__.py          # Package version, EXPECTED_LLAMA_SERVER_VERSION constant
├── cli.py               # Click CLI entry points: bench, search, explore, report
│                        # Graceful shutdown (_setup_graceful_shutdown, _finalize_bench)
│                        # Server path resolution (_resolve_server_path)
│                        # Vulkan GPU discovery (_discover_and_print_gpus)
├── config.py            # BenchConfig dataclass, SearchSpace, parse_range, validate_config
│                        # configs_from_args factory, default_search_space
├── runner.py            # BenchmarkRunner: start_server, wait_for_server_ready, stop_server
│                        # check_version_mismatch, run_benchmark (single config)
│                        # Streaming HTTP client measuring TTFT and tok/s
├── tuner.py             # AdaptiveTuner: context sweep + OOM-adaptive retry
│                        # TunerBounds, TunerThresholds dataclasses
│                        # select_best_configs (multi-objective), write_summary_json
│                        # accumulated_attempts property (partial results on interrupt)
├── search.py            # StagedSearcher: Phase 1 coarse + Phase 2 refinement
│                        # Early stopping, save_results
│                        # accumulated_results property (partial results on interrupt)
├── explorer.py          # ContinuousExplorer: indefinite ctx×ngl×batch sweep
│                        # HallOfFame: tracks best config per objective
│                        # accumulated_attempts property (partial results on interrupt)
├── metrics.py           # RunMetrics dataclass, failure_reason classification
│                        # parse_server_log (stderr pattern matching)
├── prompts.py           # REVERSE_ENGINEERING_SYSTEM_PROMPT (~1200 tokens)
│                        # SHARED_PREFIX_TEMPLATE (firmware binary pseudocode)
│                        # build_prompt_sequence, load_prompt_pack (JSON/YAML)
│                        # load_corpus_files (README.md, agent.md, artifacts.md)
├── report.py            # generate_markdown_report, load_results, print_summary_table
│                        # generate_bench_report (4-preset Markdown report)
├── gpu.py               # discover_vulkan_gpus (vulkaninfo), default_vk_devices
│                        # build_env (GGML_VK_VISIBLE_DEVICES injection)
├── hw_monitor.py        # HWMonitor: background polling of GPU/CPU/RAM metrics
│                        # NVIDIA (nvidia-smi), AMD (rocm-smi), CPU/RAM (psutil)
├── tui.py               # BenchTUI: 11-panel Textual dashboard
│                        # Graceful q/Ctrl+C handling, SIGKILL fallback for sudo children
│                        # handle_event, update_progress, set_best
├── logging_setup.py     # setup_logging: configures file + stderr handlers
│                        # Verbosity levels: 0=WARNING, 1=INFO, 2=DEBUG
│
│   # New files (4-preset bench system):
├── presets.py           # PresetConfig, PresetResult, GoalPreset dataclasses
│                        # BUILTIN_GOALS, PRESET_REGISTRY, ALL_PRESET_NAMES
│                        # PresetRunner: runs each preset sweep independently
│                        # get_goal_preset factory (builtin + custom)
├── scoring.py           # ConfigScore, ScoringResult dataclasses
│                        # score_presets(results, goal): min-max normalize + weighted sum
│                        # _generate_recommendation: picks optimal (ngl, batch)
└── orchestrator.py      # OrchestratorResult, PresetBenchOrchestrator
                         # Runs all 4 presets in order, emits TUI events, collects results
```

---

## License

See [LICENSE](LICENSE) for details.
