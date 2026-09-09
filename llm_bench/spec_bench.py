#!/usr/bin/env python3
"""llm-bench: benchmark tok/s on any OpenAI-compatible server.

Benchmarks generation throughput (tok/s) at prompt/context lengths of
0, 8, 16, 32, 64 and 128 tokens (or override with ``--lengths``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from typing import Dict, List, Tuple

DEFAULT_LENGTHS = [0, 8, 16, 32, 64, 128]
DEFAULT_GEN_TOKENS = 256
DEFAULT_RUNS = 3
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
    ap.add_argument("--model", required=True, help="model name as served by the endpoint")
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
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=(
            f"runs per length (default {DEFAULT_RUNS}, first discarded as warmup "
            "unless --keep-warmup)"
        ),
    )
    ap.add_argument(
        "--keep-warmup",
        action="store_true",
        help="include the first (warmup) run in results",
    )
    ap.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, help="request timeout seconds"
    )
    return ap.parse_args(argv)


def bench_lengths(
    base_url: str,
    model: str,
    lengths: List[int],
    gen_tokens: int,
    runs: int,
    keep_warmup: bool,
    timeout: int,
) -> Dict[int, List[float]]:
    """Benchmark each context length and print its result row.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        model: Model name as served by the endpoint.
        lengths: Context lengths to test.
        gen_tokens: Maximum tokens to generate per run.
        runs: Number of measured runs per length.
        keep_warmup: When True, the first run counts in the results.
        timeout: Request timeout in seconds.

    Returns:
        Mapping of context length to the list of tok/s readings for lengths
        that completed successfully.
    """
    results: Dict[int, List[float]] = {}
    for ctx in lengths:
        prompt = build_prompt(ctx)
        run_results: List[float] = []
        prompt_tok = 0
        n_gen = 0
        last_error = ""
        for i in range(runs + (0 if keep_warmup else 1)):
            try:
                tok_s, p_tok, g_tok, _dt = run_completion(
                    base_url, model, prompt, gen_tokens, timeout
                )
            except Exception as e:  # noqa: BLE001 - report and continue
                last_error = str(e)
                run_results = []
                break
            if i == 0 and not keep_warmup:
                continue  # discard warmup
            run_results.append(tok_s)
            prompt_tok, n_gen = p_tok, g_tok

        if run_results:
            mean = statistics.mean(run_results)
            results[ctx] = run_results
            runs_str = " ".join(f"{x:.1f}" for x in run_results)
            print(f"{ctx:>5} {prompt_tok:>10} {n_gen:>7} {mean:>9.2f} {runs_str:>10}")
        else:
            print(f"{ctx:>5}  FAILED: {last_error}")
    return results


def print_summary(results: Dict[int, List[float]]) -> None:
    """Print a token-per-second summary with bar chart.

    Args:
        results: Mapping of context length to the list of tok/s readings.
    """
    if len(results) <= 1:
        return
    print("\nSummary (mean tok/s):")
    for ctx, runs in results.items():
        mean = statistics.mean(runs)
        bar = "#" * int(mean / 2)
        print(f"  ctx={ctx:<4} {mean:8.2f} tok/s  |{bar}")


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

    try:
        run_completion(base_url, args.model, build_prompt(8), 8, args.timeout)
    except Exception as e:
        raise SystemExit(f"error: cannot reach server at {base_url}: {e}") from e

    print(f"llm-bench: {base_url}  model={args.model}")
    warmup_note = "(+warmup)" if not args.keep_warmup else ""
    print(f"gen_tokens={args.gen_tokens}  runs/length={args.runs} {warmup_note}")
    hdr = f"{'ctx':>5} {'prompt_tok':>10} {'gen_tok':>7} {'tok/s':>12} {'runs':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))

    results = bench_lengths(
        base_url,
        args.model,
        args.lengths,
        args.gen_tokens,
        args.runs,
        args.keep_warmup,
        args.timeout,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
