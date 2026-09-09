# llm-bench

Benchmark token generation throughput (tok/s) on any OpenAI-compatible server, across prompt/context lengths.

## Install as a global CLI

```bash
uv tool install git+https://github.com/your-user/llm-bench
```

Or from a local checkout:

```bash
uv tool install .
```

This puts `llm-spec-bench` on your PATH. You can also run it without installing:

```bash
uvx git+https://github.com/your-user/llm-bench --model <model> --port <port>
```

## Usage

```bash
llm-spec-bench --model qwen2.5-7b-instruct --ip 127.0.0.1 --port 8080
```

Options:

| Option | Description |
| --- | --- |
| `--model` | Model name as served by the endpoint (required) |
| `--ip` | Server IP (default `127.0.0.1`) |
| `--port` | Server port (required) |
| `--lengths` | Context lengths to test (default `0 8 16 32 64 128`) |
| `--gen-tokens` | Max tokens generated per run (default `256`) |
| `--runs` | Runs per length (default `3`; first run is a discarded warmup) |
| `--keep-warmup` | Include the warmup run in the results |
| `--timeout` | Request timeout in seconds (default `600`) |

Example output:

```text
llm-spec-bench: http://127.0.0.1:8080  model=qwen2.5-7b-instruct

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
