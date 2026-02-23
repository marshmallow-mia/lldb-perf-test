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
6. [Profile Files](#profile-files)
7. [How It Works](#how-it-works)
8. [Output Format](#output-format)
9. [Security Note](#security-note)
10. [Version Compatibility](#version-compatibility)

---

## Overview

`llama-bench` was built to answer the question: *what combination of
`llama-server` flags gives the lowest latency for my model, GPU, and workload?*

Key features:

- **Single-config benchmark** (`bench`): run one configuration, collect metrics,
  save JSONL.
- **Staged parameter search** (`search`): coarse sweep over a large config space,
  then refine the top 25% — all with early-stopping to skip obviously bad configs.
- **Rich TUI**: live progress bar, rolling results table, best-so-far display.
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
| `--output`, `-o` | auto | JSONL output path |
| `--prompt-pack` | — | Custom prompt pack file (JSON/YAML) |
| `--no-tui` | `False` | Disable TUI, use plain output |

### `llama-bench search`

All flags from `bench`, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--np-tests` | `1` | Range spec for `-np`, e.g. `1-2` or `1,2` |
| `--ctx-tests` | `49152` | Range spec for `-c` |
| `--ngl-tests` | `45` | Range spec for `--n-gpu-layers` |
| `--max-configs` | `50` | Maximum configs to evaluate |

Range specs accept:
- `"1-4"` → `[1, 2, 3, 4]`
- `"1,2,4"` → `[1, 2, 4]`
- `"49152"` → `[49152]`

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

*Profile loading via CLI is on the roadmap; currently use `--np-tests`, `--ctx-tests`, etc.*

---

## How It Works

### Staged Search

1. **Phase 1 — coarse sweep**: generate the Cartesian product of all search-space
   values (capped at `--max-configs`), run each with 2 follow-up prompts.
2. **Phase 2 — refinement**: take the top 25% of Phase-1 configs by score, re-run
   with 4 follow-up prompts for higher-fidelity measurements.
3. **Early stopping**: a config is abandoned if it fails, causes OOM, or exceeds
   `timeout_factor × best_observed_time`.
4. Results are sorted ascending by end-to-end latency.

### KV-Cache Prefix Reuse Probing

Each prompt sequence includes a long shared prefix (the binary analysis context,
~600 tokens).  Follow-up messages re-send the same prefix, allowing `--cache-reuse`
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
  "run_id": "run_20240101T120000_abcd1234",
  "timestamp": "2024-01-01T12:00:00+00:00",
  "config_hash": "abcd1234",
  "success": true,
  "failure_reason": null,
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

Search results additionally wrap each run inside a `metrics` array and include
`config` (the full BenchConfig dict), `phase`, and `best_score`.

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

The tool is tested against llama.cpp build **b4200** (`EXPECTED_LLAMA_SERVER_VERSION`
in `llama_bench/__init__.py`).

Version detection runs `llama-server --version` **without** sudo and parses a
line of the form:

```
version: 8133 (2b6dfe824)
```

The parsed version string (e.g. `"8133 (2b6dfe824)"`) is compared against
`EXPECTED_LLAMA_SERVER_VERSION`.  If they differ, a warning is printed:

```
Version warning: llama-server version mismatch: expected 'b4200', got '8133 (2b6dfe824)'.
Results may differ from baseline.
```

The `--server` path is **always resolved to an absolute path** before version
detection runs, so `sudo: ./llama-server: command not found` errors can never
be mis-parsed as a version string.

To suppress the warning, update `EXPECTED_LLAMA_SERVER_VERSION` to match your
build (e.g. `"8133 (2b6dfe824)"`).

The benchmark will still run regardless; the warning is informational only.
