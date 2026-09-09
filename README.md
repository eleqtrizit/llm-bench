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

```bash
llm-bench --model qwen2.5-7b-instruct --host 127.0.0.1 --port 8080
```

If you omit `--model`, the tool queries the server's `/v1/models` endpoint and
shows a numbered menu to pick from:

```bash
llm-bench --host 127.0.0.1 --port 8080
```

Options:

| Option | Description |
| --- | --- |
| `--model` | Model name as served by the endpoint (optional; omit to pick interactively from `/v1/models`) |
| `-h`, `--host` | Server host (required) |
| `-p`, `--port` | Server port (required) |
| `--lengths` | Context lengths to test (default `0 8 16 32 64 128`) |
| `--gen-tokens` | Max tokens generated per run (default `256`) |
| `--runs` | Runs per length (default `3`; first run is a discarded warmup) |
| `--keep-warmup` | Include the warmup run in the results |
| `--timeout` | Request timeout in seconds (default `600`) |

Example output:

```text
llm-bench: http://127.0.0.1:8080  model=qwen2.5-7b-instruct

  ctx prompt_tok gen_tok        tok/s       runs
--------------------------------------------------
Summary (mean tok/s):
  ctx=0      10.00 tok/s  |#####
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
