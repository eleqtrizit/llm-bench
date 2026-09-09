#!/usr/bin/env python3
"""llm-bench: benchmark tok/s on any OpenAI-compatible server.

Benchmarks generation throughput (tok/s) at prompt/context lengths of
0, 8, 16, 32, 64 and 128 tokens (or override with ``--lengths``).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Dict, List, Tuple

DEFAULT_LENGTHS = [0, 8, 16, 32, 64, 128]
DEFAULT_GEN_TOKENS = 256
DEFAULT_TIMEOUT = 600

# Filler words used to build context (the token count is only approximate,
# which is fine: we report the server's own usage.prompt_tokens).
FILLER = (
    "The quick brown fox jumps over the lazy dog while the calm river "
    "flows steadily through the valley under a bright morning sky. "
)


def build_prompt(ctx_tokens: int) -> str:
    """Build a prompt with roughly ``ctx_tokens`` filler tokens.

    Args:
        ctx_tokens: Approximate number of context tokens to pad the prompt with.

    Returns:
        The prompt text, padded with filler words when ``ctx_tokens > 0``.
    """
    if ctx_tokens <= 0:
        return "Count from 1 to 20."
    n_words = max(1, int(ctx_tokens * 0.75))  # ~0.75 words per token
    filler = (FILLER * (n_words // len(FILLER.split()) + 1))[: n_words * 7]
    return (
        f"Read the following text carefully:\n\n{filler}\n\n"
        "Now, ignoring the text above entirely, count from 1 to 20."
    )


def run_completion(
    base_url: str, model: str, prompt: str, gen_tokens: int, timeout: int
) -> Tuple[float, int, int, float]:
    """Run a single non-streaming chat completion.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        model: Model name as served by the endpoint.
        prompt: The user prompt text.
        gen_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        A ``(tok_s, prompt_tokens, gen_tokens, seconds)`` tuple.

    Raises:
        RuntimeError: If the server responds but generates no tokens.
        urllib.error.URLError: If the server is unreachable or times out.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": gen_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    dt = time.perf_counter() - t0

    usage = body.get("usage", {})
    n_gen = usage.get("completion_tokens", 0)
    n_prompt = usage.get("prompt_tokens", 0)
    if n_gen <= 0:
        # Fall back to counting content words.
        try:
            n_gen = len(body["choices"][0]["message"]["content"].split())
        except (KeyError, IndexError, TypeError):
            n_gen = 0
    if n_gen <= 0:
        raise RuntimeError(f"no tokens generated: {body}")
    return n_gen / dt, n_prompt, n_gen, dt


def query_models(base_url: str, timeout: int) -> List[str]:
    """Query the server's served model list.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        timeout: Request timeout in seconds.

    Returns:
        The model identifiers from ``GET /v1/models``, in served order.

    Raises:
        urllib.error.URLError: If the server is unreachable or times out.
        RuntimeError: If the response has no models.
    """
    req = urllib.request.Request(base_url + "/v1/models", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    models = [m["id"] for m in body.get("data", []) if m.get("id")]
    if not models:
        raise RuntimeError(f"no models in /v1/models response: {body}")
    return models


def choose_model(models: List[str]) -> int:
    """Render a numbered menu and read the user's model selection.

    Args:
        models: The model identifiers to choose from.

    Returns:
        The zero-based index of the chosen model, or -1 to quit.

    Raises:
        KeyboardInterrupt: If the user aborts with Ctrl-C.
    """
    print("\nAvailable models:")
    for i, model in enumerate(models):
        print(f"  [{i + 1}] {model}")
    while True:
        try:
            raw = input(f"Select a model [1-{len(models)}] (q to quit): ").strip()
        except EOFError:
            print("\nNo model selected.")
            return -1
        if raw.lower() in ("q", "quit", "exit"):
            return -1
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return int(raw) - 1
        print(f"Invalid selection: {raw!r}. Enter a number between 1 and {len(models)}.")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed argparse namespace.
    """
    # We reclaim -h for --host, so the automatic -h/--help pair is disabled and
    # help is re-added under the long form only.
    ap = argparse.ArgumentParser(
        prog="llm-bench",
        description="Benchmark tok/s on any OpenAI-compatible server.",
        add_help=False,
    )
    ap.add_argument(
        "--model",
        help="model name as served by the endpoint (omit to pick from /v1/models)",
    )
    ap.add_argument("-h", "--host", required=True, help="server host")
    ap.add_argument("-p", "--port", required=True, type=int, help="server port")
    ap.add_argument(
        "--help", action="help", default=argparse.SUPPRESS, help="show this help message and exit"
    )
    ap.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help="context lengths to test (default: 0 8 16 32 64 128)",
    )
    ap.add_argument(
        "--gen-tokens",
        type=int,
        default=DEFAULT_GEN_TOKENS,
        help=f"max tokens to generate per run (default {DEFAULT_GEN_TOKENS})",
    )
    ap.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, help="request timeout seconds"
    )
    return ap.parse_args(argv)


def format_header() -> str:
    """Build the aligned header row for the per-length results table.

    Returns:
        The fixed-width header line; its length matches the dash separator.
    """
    return f"{'ctx':>5} {'prompt_tok':>10} {'gen_tok':>7} {'tok/s':>9}"


def format_row(ctx: int, prompt_tok: int, gen_tok: int, tok_s: float) -> str:
    """Build one aligned results-table row.

    Args:
        ctx: The context length of this row.
        prompt_tok: Prompt tokens reported by the server for this run.
        gen_tok: Completion tokens generated in this run.
        tok_s: Generation throughput of this run in tokens per second.

    Returns:
        The formatted row, aligned with the header columns.
    """
    return f"{ctx:>5} {prompt_tok:>10} {gen_tok:>7} {tok_s:>9.2f}"


def bench_lengths(
    base_url: str,
    model: str,
    lengths: List[int],
    gen_tokens: int,
    timeout: int,
) -> Dict[int, float]:
    """Benchmark each context length once and print its result row.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        model: Model name as served by the endpoint.
        lengths: Context lengths to test.
        gen_tokens: Maximum tokens to generate per run.
        timeout: Request timeout in seconds.

    Returns:
        Mapping of context length to tok/s for lengths that completed
        successfully.
    """
    results: Dict[int, float] = {}
    for ctx in lengths:
        prompt = build_prompt(ctx)
        try:
            tok_s, prompt_tok, g_tok, _dt = run_completion(
                base_url, model, prompt, gen_tokens, timeout
            )
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{ctx:>5}  FAILED: {e}")
            continue
        results[ctx] = tok_s
        print(format_row(ctx, prompt_tok, g_tok, tok_s))
    return results


def print_summary(results: Dict[int, float]) -> None:
    """Print a token-per-second summary with bar chart.

    Args:
        results: Mapping of context length to tok/s.
    """
    if len(results) <= 1:
        return
    print("\nSummary (tok/s):")
    for ctx, tok_s in results.items():
        bar = "#" * int(tok_s / 2)
        print(f"  ctx={ctx:<4} {tok_s:8.2f} tok/s  |{bar}")


def main(argv: List[str] | None = None) -> None:
    """CLI entry point: run the benchmark against one OpenAI-compatible server.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Raises:
        SystemExit: If the server cannot be reached.
    """
    args = parse_args(argv)
    host = f"[{args.host}]" if ":" in args.host else args.host
    base_url = f"http://{host}:{args.port}"

    model = args.model
    if model is None:
        try:
            models = query_models(base_url, args.timeout)
        except Exception as e:
            raise SystemExit(f"error: cannot reach server at {base_url}: {e}") from e
        if len(models) == 1:
            model = models[0]
            print(f"Only one model served, using: {model}")
        else:
            chosen = choose_model(models)
            if chosen < 0:
                raise SystemExit("No model selected; exiting.")
            model = models[chosen]

    try:
        run_completion(base_url, model, build_prompt(8), 8, args.timeout)
    except Exception as e:
        raise SystemExit(f"error: cannot reach server at {base_url}: {e}") from e

    print(f"llm-bench: {base_url}  model={model}")
    print(f"gen_tokens={args.gen_tokens}, one run per context length")
    hdr = format_header()
    print("\n" + hdr)
    print("-" * len(hdr))

    results = bench_lengths(
        base_url,
        model,
        args.lengths,
        args.gen_tokens,
        args.timeout,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
