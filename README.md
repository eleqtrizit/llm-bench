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

Measured runs stream their response. Time-to-first-token bounds the prefill
phase, so prefill tok/s is estimated as `prompt_tokens / TTFT`. Generation
tok/s counts only the time after the first token. A dash in the prefill
column means the server reported no prompt token count.

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
options are 0, 8, 32, 64, 128, and 200 (kilotokens). Pass `--ctx` to skip the
menu:

```bash
llm-bench --host 127.0.0.1 --port 8080 --ctx 0,8,16
```

Options:

| Option | Description |
| --- | --- |
| `--model` | Model name as served by the endpoint (optional; omit to pick interactively from `/v1/models`) |
| `-h`, `--host` | Server host (required) |
| `-p`, `--port` | Server port (required) |
| `--ctx` | Comma-separated context sizes in kilotokens, for example `0,8,16` (optional; omit to pick from the interactive menu) |
| `--gen-tokens` | Max tokens generated per run (default `256`) |
| `--timeout` | Request timeout in seconds (default `600`) |

Example output:

```text
llm-bench: http://127.0.0.1:8080  model=qwen2.5-7b-instruct
gen_tokens=256, one run per context size

     ctx    prompt_tok    gen_tok    prefill tok/s        tok/s
---------------------------------------------------------------
       0            20         93                -       187.74
      8k          6160        256          4520.31       178.40

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
