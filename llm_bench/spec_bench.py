#!/usr/bin/env python3
"""llm-bench: benchmark tok/s on any OpenAI-compatible server.

Benchmarks generation throughput (tok/s) at prompt/context lengths of
0, 8, 16, 32, 64 and 128 tokens (or override with ``--lengths``).
"""

from __future__ import annotations

import argparse
import json
import sys
import termios
import time
import tty
import urllib.request
from typing import Dict, List, Tuple

DEFAULT_CTX_OPTIONS = [0, 8, 32, 64, 128, 200]
DEFAULT_GEN_TOKENS = 256
DEFAULT_TIMEOUT = 600

# Filler words used to build context (the token count is only approximate,
# which is fine: we report the server's own usage.prompt_tokens).
FILLER = (
    "The quick brown fox jumps over the lazy dog while the calm river "
    "flows steadily through the valley under a bright morning sky. "
)


def build_prompt(ctx_k: int) -> str:
    """Build a prompt with roughly ``ctx_k`` kilotokens of filler context.

    Args:
        ctx_k: Context size in kilotokens; 0 produces a short prompt.

    Returns:
        The prompt text, padded with filler words when ``ctx_k > 0``.
    """
    if ctx_k <= 0:
        return "Count from 1 to 20."
    n_words = max(1, int(ctx_k * 1024 * 0.75))  # ~0.75 words per token
    filler = (FILLER * (n_words // len(FILLER.split()) + 1))[: n_words * 7]
    return (
        f"Read the following text carefully:\n\n{filler}\n\n"
        "Now, ignoring the text above entirely, count from 1 to 20."
    )


def run_completion(
    base_url: str, model: str, prompt: str, gen_tokens: int, timeout: int
) -> Tuple[float, float, int, int, float]:
    """Run a single streaming chat completion and time its phases.

    The request streams server-sent events. Time-to-first-token bounds the
    prefill phase, so prefill tok/s is estimated as ``prompt_tokens / TTFT``.
    Generation tok/s counts only the time after the first token.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        model: Model name as served by the endpoint.
        prompt: The user prompt text.
        gen_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        A ``(gen_tok_s, prefill_tok_s, prompt_tokens, gen_tokens, ttft_s)``
        tuple. ``prefill_tok_s`` is 0.0 when the server produced no prompt
        token count.

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
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()

    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    n_chunk_tokens = 0
    n_prompt = 0
    n_gen = 0
    ttft = 0.0
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            prefix = b"data:"
            data = line[len(prefix):].strip()
            if data in (b"", b"[DONE]"):
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage:
                n_prompt = usage.get("prompt_tokens", n_prompt)
                n_gen = usage.get("completion_tokens", n_gen)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    n_chunk_tokens += 1
                    if ttft == 0.0:
                        ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0

    if n_gen <= 0:
        n_gen = n_chunk_tokens
    if n_gen <= 0:
        raise RuntimeError("no tokens generated in streaming response")
    gen_seconds = max(total - ttft, 1e-9)
    gen_tok_s = n_gen / gen_seconds
    prefill_tok_s = n_prompt / ttft if (n_prompt and ttft > 0) else 0.0
    return gen_tok_s, prefill_tok_s, n_prompt, n_gen, ttft


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


def read_key() -> str:
    """Read one keypress in raw mode and map it to a menu action.

    Returns:
        One of ``up``, ``down``, ``space``, ``enter``, ``quit`` or the raw
        lowercase character.

    Raises:
        KeyboardInterrupt: On Ctrl-C.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            return {"[A": "up", "[B": "down"}.get(seq, "")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch in ("q", "\x03"):
            return "quit"
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def render_menu(options: List[int], selected: set[int], cursor: int, first: bool) -> str:
    """Render the context-length selection menu.

    Args:
        options: The selectable context sizes, in kilotokens.
        selected: Indices of currently toggled-on options.
        cursor: Index of the option the cursor is on.
        first: When True, render the header line too.

    Returns:
        The escape-sequence string to write to the terminal.
    """
    rows = []
    if first:
        rows.append(
            "Select context lengths (space: toggle, up/down: move,"
            " a: all/none, enter: start, q: quit)"
        )
    for i, opt in enumerate(options):
        mark = "x" if i in selected else " "
        pointer = ">" if i == cursor else " "
        label = f"{opt}k" if opt else "0"
        rows.append(f" {pointer} [{mark}] {label}")
    # Move the cursor above the rendered rows and redraw them, erasing each line.
    return "\x1b[s" + "".join(f"\r\x1b[2K{row}\n" for row in rows) + f"\x1b[{len(rows)}A"


def menu_select_ctx(options: List[int]) -> List[int] | None:
    """Show the interactive context-length multi-select menu.

    All options start selected. Space toggles the option under the cursor,
    ``a`` toggles all, enter confirms, and ``q`` aborts.

    Args:
        options: The selectable context sizes, in kilotokens.

    Returns:
        The chosen context sizes in kilotokens, or None when the user quits.

    Raises:
        ValueError: If not attached to a terminal.
        KeyboardInterrupt: On Ctrl-C.
    """
    if not sys.stdin.isatty():
        raise ValueError("no terminal attached; pass --ctx 0,8,... instead")
    selected: set[int] = set(range(len(options)))
    cursor = 0
    print(render_menu(options, selected, cursor, True), end="", flush=True)
    while True:
        key = read_key()
        if key == "quit":
            print(f"\x1b[{len(options) + 1}B")
            return None
        if key == "enter":
            print(f"\x1b[{len(options) + 1}B")
            return [options[i] for i in sorted(selected)]
        if key == "up":
            cursor = (cursor - 1) % len(options)
        elif key == "down":
            cursor = (cursor + 1) % len(options)
        elif key == "space":
            if cursor in selected:
                selected.remove(cursor)
            else:
                selected.add(cursor)
        elif key == "a":
            if len(selected) == len(options):
                selected = set()
            else:
                selected = set(range(len(options)))
        print(render_menu(options, selected, cursor, False), end="", flush=True)


def parse_ctx(raw: str) -> List[int]:
    """Parse a comma-separated ``--ctx`` value list.

    Args:
        raw: Comma-separated context sizes in kilotokens, for example
            ``0,8,16``.

    Returns:
        The parsed context sizes, in kilotokens.

    Raises:
        argparse.ArgumentTypeError: If any value is not a non-negative integer.
    """
    try:
        values = [int(v.strip()) for v in raw.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid --ctx value: {raw!r}") from None
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError(f"--ctx values must be >= 0: {raw!r}")
    return values


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
        "--ctx",
        type=parse_ctx,
        help=(
            "comma-separated context sizes in kilotokens, for example 0,8,16"
            " (omit to pick from an interactive menu)"
        ),
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


WARMUP_PROMPT = "Say hello and nothing else."
WARMUP_GEN_TOKENS = 16
WARMUP_PASSES = 2


def warm_up(base_url: str, model: str, timeout: int) -> None:
    """Run short warmup completions so the server has the model loaded.

    Two tiny zero-context requests prime any lazy model loading or cache
    setup before measured runs begin.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        model: Model name as served by the endpoint.
        timeout: Request timeout in seconds.

    Raises:
        RuntimeError: If any warmup completion generates no tokens.
        urllib.error.URLError: If the server is unreachable or times out.
    """
    for i in range(WARMUP_PASSES):
        run_completion(base_url, model, WARMUP_PROMPT, WARMUP_GEN_TOKENS, timeout)
        print(f"warmup pass {i + 1}/{WARMUP_PASSES} done")


def format_header() -> str:
    """Build the aligned header row for the per-length results table.

    Returns:
        The fixed-width header line; its length matches the dash separator.
    """
    return (
        f"{'ctx':>5} {'prompt_tok':>10} {'gen_tok':>7} "
        f"{'prefill tok/s':>13} {'tok/s':>9}"
    )


def format_row(
    ctx_k: int, prompt_tok: int, gen_tok: int, prefill_tok_s: float, tok_s: float
) -> str:
    """Build one aligned results-table row.

    Args:
        ctx_k: The context size of this row, in kilotokens.
        prompt_tok: Prompt tokens reported by the server for this run.
        gen_tok: Completion tokens generated in this run.
        prefill_tok_s: Estimated prefill throughput; 0.0 means unknown.
        tok_s: Generation throughput of this run in tokens per second.

    Returns:
        The formatted row, aligned with the header columns. Unknown prefill
        is rendered as ``-``.
    """
    ctx_str = f"{ctx_k}k" if ctx_k else "0"
    prefill_str = f"{prefill_tok_s:.2f}" if prefill_tok_s > 0 else "-"
    return (
        f"{ctx_str:>5} {prompt_tok:>10} {gen_tok:>7} {prefill_str:>13} {tok_s:>9.2f}"
    )


def bench_lengths(
    base_url: str,
    model: str,
    lengths: List[int],
    gen_tokens: int,
    timeout: int,
) -> Dict[int, float]:
    """Benchmark each context size once and print its result row.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        model: Model name as served by the endpoint.
        lengths: Context sizes to test, in kilotokens.
        gen_tokens: Maximum tokens to generate per run.
        timeout: Request timeout in seconds.

    Returns:
        Mapping of context size to tok/s for sizes that completed
        successfully.
    """
    results: Dict[int, float] = {}
    for ctx in lengths:
        prompt = build_prompt(ctx)
        try:
            tok_s, prefill_tok_s, prompt_tok, g_tok, _ttft = run_completion(
                base_url, model, prompt, gen_tokens, timeout
            )
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{ctx:>5}  FAILED: {e}")
            continue
        results[ctx] = tok_s
        print(format_row(ctx, prompt_tok, g_tok, prefill_tok_s, tok_s))
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

    lengths = args.ctx
    if lengths is None:
        try:
            lengths = menu_select_ctx(DEFAULT_CTX_OPTIONS)
        except ValueError as e:
            raise SystemExit(f"error: {e}") from e
        except KeyboardInterrupt:
            raise SystemExit("cancelled.") from None
        if lengths is None:
            raise SystemExit("cancelled.")

    try:
        warm_up(base_url, model, args.timeout)
    except Exception as e:
        raise SystemExit(f"error: cannot reach server at {base_url}: {e}") from e

    print(f"llm-bench: {base_url}  model={model}")
    print(f"gen_tokens={args.gen_tokens}, one run per context size")
    hdr = format_header()
    print("\n" + hdr)
    print("-" * len(hdr))

    results = bench_lengths(
        base_url,
        model,
        lengths,
        args.gen_tokens,
        args.timeout,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
