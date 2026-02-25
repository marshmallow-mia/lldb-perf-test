# llama-bench

**llama-bench** is a command-line benchmarking and parameter-search tool for
[llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server`.  It measures
end-to-end latency, time-to-first-token (TTFT), and decode throughput for realistic
workloads, then helps you find the best combination of server flags for your hardware
and use-case.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Full CLI Reference](#full-cli-reference)
6. [TUI Dashboard](#tui-dashboard)
7. [Profile Files](#profile-files)
8. [How It Works](#how-it-works)
9. [Output Format](#output-format)
10. [Security Note](#security-note)
11. [Version Compatibility](#version-compatibility)
12. [Troubleshooting: Server Startup Failures](#troubleshooting-server-startup-failures)
---

## Overview

`llama-bench` was built to answer the question: *what combination of
`llama-server` flags gives the lowest latency for my model, GPU, and workload?*

Key features:

- **Single-config benchmark** (`bench`): adaptive sweep from max context down to
  minimum, with OOM retry logic, collecting cold/warm TTFT and throughput.
- **Staged parameter search** (`search`): coarse sweep over a large config space,
  then refine the top 25% — all with early-stopping to skip obviously bad configs.
- **Continuous explorer** (`explore`): runs indefinitely, sweeping the full
  `(ctx, ngl, batch)` grid and maintaining a live **Hall of Fame** across five
  objectives: Max Context, Fastest TTFT, Best Warm TTFT, Best Throughput, Best Overall.
- **Rich TUI**: 8-panel live dashboard with hardware monitors, activity log, settings
  display, results table, Hall of Fame, progress bar, server log tail, and verbose log.
- **Report generation** (`report`): JSONL → Markdown with ranked tables.
- **Reverse-engineering workload**: built-in 1000-token system prompt + shared
  code-analysis prefix, exercising KV-cache prefix reuse heavily.
- **Vulkan GPU discovery**: auto-detects GPUs via `vulkaninfo` and sets
  `GGML_VK_VISIBLE_DEVICES`.

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
| Vulkan drivers + `vulkaninfo` | Optional; enables GPU discovery |
| `sudo` access | Required when `--sudo` is set (default) |

---

## Quick Start

### Run a single benchmark

```bash
llama-bench bench --model /path/to/model.gguf
```

Override the binary path and a few flags:

```bash
llama-bench bench \
  --server /opt/llama.cpp/llama-server \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --n-gpu-layers 48 \
  --ctx 65536 \
  --flash-attn \
  --output results/codellama_bench.jsonl
```

### Run a parameter search

```bash
llama-bench search \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --np-tests 1-2 \
  --ctx-tests 49152,98304 \
  --ngl-tests 40,45,47 \
  --max-configs 30
```

### Run the continuous multi-objective explorer

Sweeps `(ctx, ngl, batch)` indefinitely and tracks the best config for five objectives.
Press `q` or `Ctrl+C` to stop at any time.

```bash
# Vary only context (single ngl/batch)
llama-bench explore \
  --model /data/models/my-model.gguf \
  --ctx 131072 --ctx-min 8192 --ctx-step 8192

# Full grid sweep across multiple ngl and batch values (multi-GPU)
llama-bench explore \
  --model /data/models/my-model.gguf \
  --ctx 131072 --ctx-min 32768 \
  --ngl-tests 37,41,45 \
  --batch-tests 1024,1536 \
  --vk-devices 0,1
```

### Run with verbose logging (no TUI)

Use `-v` for INFO-level logs and `-vv` for DEBUG-level logs.  Both are written
to stderr **and** to a timestamped log file under `results/`:

```bash
# INFO-level verbose, plain output (no TUI)
llama-bench search \
  -v --no-tui \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --server /opt/llama.cpp/llama-server

# DEBUG-level verbose, plain output
llama-bench search \
  -vv --no-tui \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --server /opt/llama.cpp/llama-server

# DEBUG-level verbose with TUI (logs go to file only; TUI is shown in terminal)
llama-bench search \
  -vv \
  --model /data/models/codellama-34b.Q5_K_M.gguf \
  --server /opt/llama.cpp/llama-server
```

Verbose logs include:
- Resolved server binary path
- Version check result
- Server start command, PID, and log file paths
- Health-check polling progress
- Each benchmark request with TTFT, end-to-end latency, and tok/s
- Server log parsing summary

The log file path is also recorded in each JSONL result entry as `log_file`.

### Generate a report

```bash
llama-bench report results/bench_*.jsonl --output results/report.md
```

---

## Full CLI Reference

### `llama-bench bench`

Run a benchmark with a single configuration.

| Flag | Default | Description |
|------|---------|-------------|
| `--server`, `-s` | `./llama-server` | Path to llama-server binary |
| `--model`, `-m` | *(required)* | Path to model `.gguf` file |
| `--host` | `0.0.0.0` | Server bind address |
| `--port`, `-p` | `5001` | Server port |
| `--np` | `1` | Number of parallel request slots |
| `--ctx`, `-c` | `49152` | Total KV-cache context tokens |
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
| `--vk-devices` | auto | `GGML_VK_VISIBLE_DEVICES` value |
| `--sudo` / `--no-sudo` | `True` | Launch server with sudo |
| `--n-followups` | `4` | Follow-up prompts per run |
| `--max-tokens` | `512` | Max tokens generated per request (controls decode length) |
| `--output`, `-o` | auto | JSONL output path |
| `--prompt-pack` | — | Custom prompt pack file (JSON/YAML) |
| `--no-tui` | `False` | Disable TUI, use plain output |
| `-v` / `--verbose` | off | Verbose logs to stderr + file (`-vv` for debug) |

### `llama-bench search`

All flags from `bench`, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--np-tests` | `1` | Range spec for `-np`, e.g. `1-2` or `1,2` |
| `--ctx-tests` | `49152` | Range spec for `-c` |
| `--ngl-tests` | `45` | Range spec for `--n-gpu-layers` |
| `--max-configs` | `50` | Maximum configs to evaluate |
| `--no-tui` | `False` | Disable TUI |
| `-v` / `--verbose` | off | Verbose logs to stderr + file (`-vv` for debug) |

Range specs accept:
- `"1-4"` → `[1, 2, 3, 4]`
- `"1,2,4"` → `[1, 2, 4]`
- `"49152"` → `[49152]`

### `llama-bench explore`

Runs **indefinitely** until you press `q` or `Ctrl+C`.  Each round sweeps the full
Cartesian product of `ctx × ngl × batch` values and updates the Hall of Fame after
every completed run.  Results are appended to the JSONL output immediately — nothing
is lost if you stop mid-sweep.

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
| `--split-mode` | `none` | GPU split mode (`none`/`layer`/`row`) |
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
| `--output`, `-o` | auto | JSONL output path (default: `results/explore_<ts>.jsonl`) |
| `--no-tui` | `False` | Disable TUI, use plain output |
| `-v` / `--verbose` | off | Verbose logs to stderr + file (`-vv` for debug) |

**Hall of Fame objectives tracked:**

| Objective | Optimises |
|-----------|-----------|
| Max Context | Highest `ctx` that completed successfully |
| Fastest TTFT | Lowest cold time-to-first-token |
| Best Warm TTFT | Lowest warm TTFT (KV cache hit on follow-up requests) |
| Best Throughput | Highest decode tokens/s |
| Best Overall | Balanced score: `√ctx × tok/s ÷ (1 + cold_ttft)` |

### `llama-bench report`

```
llama-bench report <file1.jsonl> [file2.jsonl ...] [--output report.md] [--top-n 10]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | derived from input | Markdown output path |
| `--top-n` | `10` | Top results to include in table |

---

## TUI Dashboard

All three commands (`bench`, `explore`, `search`) display the same full-screen
Textual dashboard.  Press `q` or `Ctrl+C` at any time to stop.

| Panel | Description |
|-------|-------------|
| **Current Run** | Active phase, config (ctx/ngl/batch), elapsed time, ETA |
| **Hardware** | Per-GPU VRAM bar, utilisation %, temperature; CPU %; RAM |
| **Settings** | Base configuration snapshot: model name, context sweep range, NGL/batch/µbatch, Flash/KV-Unified/ContBatch flags, KV cache types and reuse threshold, split mode, Vulkan devices, thread counts |
| **Server Command** | Full `llama-server` invocation as actually executed (never truncated) |
| **Activity** | Timestamped event log: server start, health-poll progress, prompt sends, results, retries |
| **Progress** | Sweep progress bar with candidate count and ETA |
| **Results** | Rolling table of completed runs: ctx, ngl, batch, cold/warm TTFT, tok/s sparkline trend, peak VRAM |
| **Best Configs (Hall of Fame)** | Live best-per-objective table — updated after every `explore` run; shows a hint for `bench`/`search` |
| **Retries** | OOM/failure retry history with reason and parameter adjustment |
| **Server Log** | Live tail of `llama-server` stderr |
| **Verbose Log** | Tail of the bench tool’s own log file (visible when `-v`/`-vv` is passed) |

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

*Profile loading via CLI is on the roadmap; currently use `--np-tests`, `--ctx-tests`, etc.*

---

## How It Works

### Adaptive Tuner (`bench`)

1. Generates candidates from `ctx_max` down to `ctx_min` in steps of `ctx_step`.
2. Starts `llama-server` for each candidate and runs the prompt workload.
3. On OOM: reduces `ngl` by `--ngl-step` (or uses a heuristic from the server’s
   memory-fit log line), then reduces `--batch-size` when ngl reaches its minimum.
4. Records cold TTFT, warm TTFT, and tok/s for each attempt.
5. Emits a multi-objective summary: max usable context, top-throughput configs near
   max context, and a recommended stable config.

### Staged Search (`search`)

1. **Phase 1 — coarse sweep**: generate the Cartesian product of all search-space
   values (capped at `--max-configs`), run each with 2 follow-up prompts.
2. **Phase 2 — refinement**: take the top 25% of Phase-1 configs by score, re-run
   with 4 follow-up prompts for higher-fidelity measurements.
3. **Early stopping**: a config is abandoned if it fails, causes OOM, or exceeds
   `timeout_factor × best_observed_time`.
4. Results are sorted ascending by end-to-end latency.

### Continuous Explorer (`explore`)

1. Builds the full `ctx × ngl × batch` Cartesian product each round.
   Round 1 runs candidates **high-to-low** (most promising first).
   Subsequent rounds **shuffle** the order so different regions get covered.
2. Each candidate uses the same OOM-adaptive retry logic as `bench`.
3. After every completed run the **Hall of Fame** is updated across all five
   objectives and pushed live to the TUI.
4. Results are appended to the JSONL output file immediately — nothing is
   lost when you stop.
5. Runs until `stop_event` fires (press `q` / `Ctrl+C`).

### KV-Cache Prefix Reuse Probing

Each prompt sequence includes a long shared prefix (the binary analysis context,
~600 tokens).  Follow-up messages re-send the same prefix, allowing `--cache-reuse`
to kick in if the server has stored the prefix in its KV cache.  This makes the
benchmark sensitive to the `--cache-reuse` threshold.

### Metrics Collection

- **Client metrics**: TTFT, end-to-end latency, streaming tok/s — measured by the
  client using `requests` with `stream=True`.
- **Server metrics**: prompt eval time, decode time, decode tok/s — parsed from the
  `llama-server` stderr log after each run.

---

## Output Format

Results are saved as **JSONL** (one JSON object per line).  Each line contains:

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
    "split_mode": "layer",
    "vk_devices": "0,1"
  },
  "success": true,
  "failure_reason": null,
  "ttft_s": 0.215,
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

Search results additionally wrap each run inside a `metrics` array and include
`config` (the full BenchConfig dict), `phase`, and `best_score`.

Explorer results use the same per-attempt JSONL schema, appended one line at a time.
The output file is named `results/explore_<timestamp>.jsonl` by default.
Query `hall_of_fame` data is displayed in the TUI live but is not persisted to JSONL.

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

**Alternative — system-wide symlink:**

```bash
sudo ln -s "$(which llama-bench)" /usr/local/bin/llama-bench
```

After creating the symlink, plain `sudo llama-bench ...` works without the
`env PATH=...` wrapper.

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

The `--server` path is **always resolved to an absolute path** before version
detection runs, so `sudo: ./llama-server: command not found` errors can never
be mis-parsed as a version string.

To suppress the warning, update `EXPECTED_LLAMA_SERVER_VERSION` to match your
build number (e.g. `"9000"`).

The benchmark will still run regardless; the warning is informational only.

---

## Troubleshooting: Server Startup Failures

`llama-bench` classifies server startup failures into four categories and
surfaces them in `bench`, `search`, and `explore` modes.

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

### `bench` mode behaviour

If the server fails to start, `llama-bench bench` exits with a non-zero code
and prints an actionable error, e.g.:

```
Error: Server startup failed (model_not_found).
  Server stderr log: results/server_stderr_20260101T120000.log
  Excerpt:
    gguf_init_from_file: failed to open GGUF file /bad/path.gguf (No such file or directory)
```

The JSONL result file is still written so you can inspect it later.

### `search` mode behaviour

In search mode startup failures are treated as expected outcomes: the failing
configuration is marked `success=false` with the appropriate `failure_reason`,
and the search continues with the next configuration.  This means that an OOM
failure at high `--n-gpu-layers` values will not abort the whole search.

### Practical tips

* **`model_not_found`** — pass an absolute path to `--model` and verify the
  file exists: `ls -lh /path/to/model.gguf`.
* **`out_of_vram`** — reduce `--n-gpu-layers` (or let `search` discover the
  maximum that fits).  Check free VRAM with `vulkaninfo | grep -i memory`.
* **`server_exited`** — inspect the full stderr log printed in the error
  message for clues.  Common culprits: wrong binary for the hardware (e.g.
  CUDA binary on a Vulkan-only machine), missing Vulkan ICD.
* **`server_startup_timeout`** — the model may be loading from a slow storage
  device.  The startup timeout is currently fixed at 90 seconds; if your
  hardware is particularly slow, consider using a smaller model or faster
  storage.
