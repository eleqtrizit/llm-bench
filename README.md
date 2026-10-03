# llm-bench

Benchmark token generation throughput (tok/s) on any OpenAI-compatible server, across prompt/context lengths.

## Quick Start

Install the CLI globally:

```bash
uv tool install git+https://github.com/eleqtrizit/llm-bench
```

Then run it against any OpenAI-compatible server:

```bash
llm-bench --model <model> --host <host> --port <port>
```

## Install as a global CLI

```bash
uv tool install git+https://github.com/eleqtrizit/llm-bench
```

Or from a local checkout:

```bash
uv tool install .
```

To upgrade an existing install to the latest version:

```bash
uv tool upgrade llm-bench
```

If the upgrade does not pick up changes, uninstall and reinstall:

```bash
uv tool uninstall llm-bench
uv tool install git+https://github.com/eleqtrizit/llm-bench
```

This puts `llm-bench` on your PATH. You can also run it without installing:

```bash
uvx git+https://github.com/eleqtrizit/llm-bench --model <model> --host <host> --port <port>
```

## Usage

Before measuring, the tool sends two short zero-context warmup requests so the
server has the model fully loaded.

Measured runs stream their response. When the server is SGLang with
`--enable-metrics` or vLLM with request stats logging left enabled, the tool
snapshots `/metrics` before and after each single-request run and reports
prefill and decode tok/s from the server's own counters (TTFT, end-to-end
latency, prompt and generation token counters). The engine is detected from
`owned_by` on `/v1/models`, with a `/metrics` sniff as a fallback, and the
matching `sglang:` or `vllm:` metric family is used.
When the server is TensorFold (detected by `owned_by: "tensorfold"` on
`/v1/models`), the rates come from the engine's own telemetry block attached
to each response (`prefill_s`, `decode_s`), and each row also shows the
speculative draft acceptance rate (`draft=NN%`).
Otherwise it autodetects the best client-side phase-rate source per run:
llama.cpp-style `timings` counters from the final chunk when
present, otherwise time-to-first-token (which includes reasoning tokens on
thinking models). Generation tok/s always counts first token to last. Rows
marked with an asterisk used chunk-counted tokens because the server sent no
usage object, so treat them as approximate.

Each run's filler context is shuffled fresh, so no two requests share token
prefixes long enough for server-side prompt or KV-cache reuse. That keeps
cache hits from inflating the prefill numbers.

```bash
llm-bench --model qwen2.5-7b-instruct --host 127.0.0.1 --port 8080
```

If you omit `--model`, the tool queries the server's `/v1/models` endpoint and
shows a numbered menu to pick from:

```bash
llm-bench --host 127.0.0.1 --port 8080
```

If you omit `--ctx`, the tool shows a multi-select menu. Move with the arrow
keys, toggle options with the spacebar, and press Enter to start. The preset
options are 2, 8, 32, 64, 128, and 200 (kilotokens). Pass `--ctx` to skip the
menu:

```bash
llm-bench --host 127.0.0.1 --port 8080 --ctx 2,8,16
```

The `--task` flag picks what the model generates. The built-in tasks are
`code` (Write a Python Snake game) and `prose` (Write me a poem about Agents
and LLMs). If you omit `--task`, a multi-select
menu opens with the same spacebar controls; you must select at least one:

```bash
llm-bench --host 127.0.0.1 --port 8080 --ctx 2,8 --task code,prose
```

Options:

| Option | Description |
| --- | --- |
| `--model` | Model name as served by the endpoint (optional; omit to pick interactively from `/v1/models`) |
| `-h`, `--host` | Server host (required) |
| `-p`, `--port` | Server port (required) |
| `--ctx` | Comma-separated context sizes in kilotokens, minimum 2, for example `2,8,16` (optional; omit to pick from the interactive menu) |
| `--task` | Comma-separated generation tasks (`code`, `prose`; optional; omit to pick from the interactive menu) |
| `--gen-tokens` | Max tokens generated per run (default `2048`) |
| `--timeout` | Request timeout in seconds (default `600`) |

Example output:

```text
llm-bench: http://127.0.0.1:8080  model=qwen2.5-7b-instruct
gen_tokens=2048, one run per context size

     ctx task      prompt_tok    gen_tok    prefill tok/s        tok/s
----------------------------------------------------------------------
      8k code            6160       2048          4520.31       178.40
      8k prose           6201        180          4108.83       165.12

```

## Development

```bash
make install   # uv sync (creates .venv)
make test      # pytest
make lint      # compileall + flake8 + mypy
make format    # autopep8
make run       # run the CLI in the venv
```

## Uninstall

```bash
uv tool uninstall llm-bench
```
