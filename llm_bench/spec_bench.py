#!/usr/bin/env python3
"""llm-bench: benchmark tok/s on any OpenAI-compatible server.

Benchmarks generation throughput (tok/s) at prompt/context lengths of
0, 8, 16, 32, 64 and 128 tokens (or override with ``--lengths``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
import sys
import termios
import time
import tty
import urllib.error
import urllib.request
from typing import Dict, List

DEFAULT_CTX_OPTIONS = [0, 8, 32, 64, 128, 200]

# SGLang Prometheus metrics used for server-reported throughput numbers. The
# bench snapshots /metrics before and after each single-request run and reads
# the deltas, so every number comes from the server's own accounting.
METRIC_PROMPT_TOKENS = "sglang:prompt_tokens_total"
METRIC_GEN_TOKENS = "sglang:generation_tokens_total"
METRIC_TTFT_SUM = "sglang:time_to_first_token_seconds_sum"
METRIC_TTFT_COUNT = "sglang:time_to_first_token_seconds_count"
METRIC_E2E_SUM = "sglang:e2e_request_latency_seconds_sum"
METRIC_E2E_COUNT = "sglang:e2e_request_latency_seconds_count"
REQUIRED_METRICS = (
    METRIC_PROMPT_TOKENS,
    METRIC_GEN_TOKENS,
    METRIC_TTFT_SUM,
    METRIC_TTFT_COUNT,
    METRIC_E2E_SUM,
    METRIC_E2E_COUNT,
)
METRICS_POLL_INTERVAL_S = 0.5
METRICS_POLL_DEADLINE_S = 10.0
METRICS_SNAPSHOT_RETRIES = 5

# TensorFold serves no /metrics endpoint; it attaches an engine telemetry
# block to every completion response instead. The block is named after the
# engine ("tensorfold") and carries its own prefill/decode accounting.
TENSORFOLD_ENGINE = "tensorfold"
TASKS: Dict[str, str] = {
    "code": "Write a Python Snake game.",
    "prose": "Write me a poem about Agents and LLMs.",
}
DEFAULT_GEN_TOKENS = 2048
DEFAULT_TIMEOUT = 600

# Filler words used to build context (the token count is only approximate,
# which is fine: we report the server's own usage.prompt_tokens).
FILLER = (
    "The quick brown fox jumps over the lazy dog while the calm river "
    "flows steadily through the valley under a bright morning sky. "
)


def build_prompt(
    ctx_k: int, task_prompt: str = "", rng: random.Random | None = None
) -> str:
    """Build a prompt with roughly ``ctx_k`` kilotokens of filler context.

    The filler is drawn from ``FILLER`` but shuffled fresh on every call, and
    the shuffled order is unique per call. This keeps every measured prompt
    from sharing token-block prefixes with any other request, which prevents
    server-side prompt and KV-cache reuse from inflating the prefill numbers.

    Args:
        ctx_k: Context size in kilotokens; 0 produces just the task prompt.
        task_prompt: The generation instruction appended after the filler.
        rng: Optional seeded random source for deterministic prompts in tests.

    Returns:
        The prompt text, padded with shuffled filler words when ``ctx_k > 0``.
    """
    task_prompt = task_prompt or "Write a Python Snake game."
    if ctx_k <= 0:
        return task_prompt
    rng = rng if rng is not None else random.Random()
    pool = FILLER.split()
    n_words = max(1, int(ctx_k * 1024 * 0.75))  # ~0.75 words per token
    words: List[str] = []
    while len(words) < n_words:
        batch = pool[:]
        rng.shuffle(batch)
        words.extend(batch)
    filler = " ".join(words[:n_words])
    return (
        f"Read the following text carefully:\n\n{filler}\n\n"
        f"Now, ignoring the text above entirely, {task_prompt}"
    )


@dataclass(frozen=True)
class RunResult:
    """Measured outcome of one streaming chat completion.

    Attributes:
        prefill_tok_s: Server-reported prefill throughput when available,
            otherwise the TTFT-based estimate; 0.0 when unknowable.
        gen_tok_s: Generation throughput, first token to last.
        prompt_tok: Prompt tokens reported by the server, else 0.
        gen_tok: Generated tokens, server count when available.
        ttft_s: Client-measured time to first payload token.
        source: Which mechanism produced the phase rates: ``timings`` for
            llama.cpp-style server counters, ``usage`` for the TTFT-based
            fallback, ``estimated`` when token counts were chunk-counted,
            ``sglang`` when the rates came from SGLang /metrics deltas, and
            ``tensorfold`` when the rates came from the TensorFold telemetry
            block attached to the completion response.
        draft_acceptance: Fraction of drafted speculative tokens the engine
            accepted, from the TensorFold telemetry block; 0.0 when absent.
    """

    prefill_tok_s: float
    gen_tok_s: float
    prompt_tok: int
    gen_tok: int
    ttft_s: float
    source: str
    draft_acceptance: float = 0.0


def run_completion(
    base_url: str, model: str, prompt: str, gen_tokens: int, timeout: int
) -> RunResult:
    """Run a single streaming chat completion and time its phases.

    The request streams server-sent events. TTFT is the time to the first
    payload token of any kind, including reasoning deltas from thinking
    models. When the final chunk carries llama.cpp-style ``timings``, those
    server-computed counters are used directly; otherwise prefill tok/s is
    ``prompt_tokens / TTFT``. Token counts prefer the server's ``usage`` and
    fall back to content-chunk counting, which is approximate because chunks
    may carry several tokens.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        model: Model name as served by the endpoint.
        prompt: The user prompt text.
        gen_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        A :class:`RunResult` with both phase rates and their provenance.

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
    usage: Dict[str, int] | None = None
    timings: Dict[str, float] | None = None
    engine_stats: Dict[str, float] | None = None
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
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            if isinstance(chunk.get("tensorfold"), dict):
                engine_stats = chunk["tensorfold"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                # Reasoning deltas stream before content on thinking models;
                # both signal that prefill finished and generation began.
                # TensorFold names the field reasoning_content, llama.cpp
                # squeezing-style servers use reasoning.
                if (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                ):
                    n_chunk_tokens += 1
                    if ttft == 0.0:
                        ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0

    n_prompt_server = (usage or {}).get("prompt_tokens", 0)
    n_gen_server = (usage or {}).get("completion_tokens", 0)

    n_gen = n_gen_server or n_chunk_tokens
    n_prompt = n_prompt_server
    source = "usage"
    draft_acceptance = 0.0
    prefill_tok_s = 0.0
    gen_tok_s = 0.0
    if engine_stats and engine_stats.get("prefill_s") and engine_stats.get("decode_s"):
        # TensorFold reports its own prefill and decode wall time per request,
        # so both rates come from the server's accounting, not client timing.
        prefill_s = float(engine_stats["prefill_s"])
        decode_s = float(engine_stats["decode_s"])
        drafted = float(engine_stats.get("drafted") or 0)
        accepted = float(engine_stats.get("accepted") or 0)
        prefill_tok_s = n_prompt / prefill_s if (n_prompt and prefill_s > 0) else 0.0
        gen_tok_s = n_gen / decode_s if decode_s > 0 else 0.0
        source = "tensorfold"
        draft_acceptance = accepted / drafted if drafted > 0 else 0.0
    elif timings and timings.get("prompt_n") and timings.get("prompt_ms"):
        prompt_ms = float(timings["prompt_ms"])
        prefill_tok_s = timings["prompt_n"] / (prompt_ms / 1000.0)
        n_prompt = int(timings["prompt_n"])
        if timings.get("predicted_n") and timings.get("predicted_ms"):
            n_gen = int(timings["predicted_n"])
            gen_tok_s = n_gen / (float(timings["predicted_ms"]) / 1000.0)
        else:
            gen_tok_s = n_gen / max(total - ttft, 1e-9)
        source = "timings"
    else:
        gen_seconds = max(total - ttft, 1e-9)
        gen_tok_s = n_gen / gen_seconds
        prefill_tok_s = n_prompt / ttft if (n_prompt and ttft > 0) else 0.0
        source = "usage" if n_gen_server else "estimated"

    if n_gen <= 0:
        raise RuntimeError("no tokens generated in streaming response")
    return RunResult(
        prefill_tok_s=prefill_tok_s,
        gen_tok_s=gen_tok_s,
        prompt_tok=n_prompt,
        gen_tok=n_gen,
        ttft_s=ttft,
        source=source,
        draft_acceptance=draft_acceptance,
    )


def parse_metrics_text(text: str) -> Dict[str, float]:
    """Parse a Prometheus exposition document into a metric-name sum map.

    Labeled series such as ``metric{a="b"} 1.0`` are aggregated by base metric
    name: counters, ``_sum`` and ``_count`` series add up, so streaming and
    non-streaming series of the same histogram collapse into one value. Bucket
    series are dropped because the deltas the bench needs never read them.

    Args:
        text: The raw ``/metrics`` response body.

    Returns:
        A mapping from base metric name (for example
        ``sglang:time_to_first_token_seconds_sum``) to the summed value.
    """
    values: Dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("sglang:") or line.startswith("#"):
            continue
        name, _, raw_value = line.rpartition(" ")
        key = name.split("{")[0]
        if key.endswith("_bucket"):
            continue
        try:
            values[key] = values.get(key, 0.0) + float(raw_value)
        except ValueError:
            continue
    return values


def snapshot_server_metrics(
    base_url: str, timeout: int, required: tuple[str, ...] = REQUIRED_METRICS
) -> Dict[str, float]:
    """Fetch and parse ``/metrics``, retrying until every required name appears.

    SGLang can serve a truncated or partially flushed scrape while a busy
    scheduler updates its histograms, so a missing required name triggers a
    short retry instead of trusting a partial snapshot.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        timeout: Per-request timeout in seconds.
        required: Metric names that must be present in the snapshot.

    Returns:
        The parsed metric map.

    Raises:
        RuntimeError: If the endpoint never yields all required metrics.
        urllib.error.URLError: If the server is unreachable or times out.
    """
    last_error: Exception | None = None
    for _ in range(METRICS_SNAPSHOT_RETRIES):
        try:
            with urllib.request.urlopen(base_url + "/metrics", timeout=timeout) as resp:
                values = parse_metrics_text(resp.read().decode())
            missing = [name for name in required if name not in values]
            if not missing:
                return values
            last_error = RuntimeError(f"/metrics is missing {', '.join(missing)}")
        except urllib.error.HTTPError as e:
            # Engines without a /metrics endpoint (for example TensorFold)
            # answer 404, which means "metrics unavailable", not unreachable.
            last_error = RuntimeError(f"/metrics returned HTTP {e.code}")
        except urllib.error.URLError:
            raise
        except Exception as e:  # noqa: BLE001 - retry transient scrape failures
            last_error = e
        time.sleep(METRICS_POLL_INTERVAL_S)
    raise RuntimeError(f"cannot read /metrics after retries: {last_error}")


def diff_metrics(
    before: Dict[str, float], after: Dict[str, float]
) -> Dict[str, float]:
    """Compute per-metric deltas between two snapshots.

    Args:
        before: The pre-run snapshot.
        after: The post-run snapshot.

    Returns:
        A mapping from metric name to ``after - before`` for every metric
        present in ``after``; names only in ``before`` are ignored.
    """
    return {k: v - before[k] for k, v in after.items() if k in before}


def metrics_to_rates(
    deltas: Dict[str, float], client_result: "RunResult"
) -> "RunResult | None":
    """Turn single-request metric deltas into server-measured rates.

    The design assumes exactly one request ran between the snapshots, so each
    histogram ``_count`` must have advanced by exactly one. Anything else
    (concurrent traffic, missing deltas, non-positive timings) returns None so
    the caller can fall back to client-side estimates.

    Args:
        deltas: Metric deltas from :func:`diff_metrics`.
        client_result: The client-measured result, reused for token counts
            whenever a server delta is unavailable.

    Returns:
        A :class:`RunResult` with ``source == "sglang"``, or None when the
        deltas do not describe exactly one clean request.
    """
    ttft_count = deltas.get(METRIC_TTFT_COUNT, 0.0)
    e2e_count = deltas.get(METRIC_E2E_COUNT, 0.0)
    ttft = deltas.get(METRIC_TTFT_SUM, 0.0)
    e2e = deltas.get(METRIC_E2E_SUM, 0.0)
    prompt_tok = deltas.get(METRIC_PROMPT_TOKENS, 0.0)
    gen_tok = deltas.get(METRIC_GEN_TOKENS, 0.0)
    if ttft_count != 1 or e2e_count != 1 or ttft <= 0 or e2e <= ttft:
        return None
    decode_seconds = e2e - ttft
    return RunResult(
        prefill_tok_s=prompt_tok / ttft if prompt_tok > 0 else 0.0,
        gen_tok_s=gen_tok / decode_seconds if gen_tok > 0 else 0.0,
        prompt_tok=int(prompt_tok or client_result.prompt_tok),
        gen_tok=int(gen_tok or client_result.gen_tok),
        ttft_s=client_result.ttft_s,
        source="sglang",
    )


def run_completion_measured(
    base_url: str,
    model: str,
    prompt: str,
    gen_tokens: int,
    timeout: int,
    engine: str = "",
) -> RunResult:
    """Run one completion and report throughput from SGLang server metrics.

    A ``/metrics`` snapshot is taken before and after the request. When the
    server exposes the SGLang counters and the deltas describe exactly one
    request, prefill and decode tok/s come from the server's own TTFT and
    end-to-end latency accounting. Otherwise the client-measured result from
    :func:`run_completion` is returned unchanged.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        model: Model name as served by the endpoint.
        prompt: The user prompt text.
        gen_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        engine: Engine name from :func:`detect_engine`; when it is
            ``tensorfold``, the SGLang /metrics snapshot is skipped because
            TensorFold attaches its telemetry to each response instead.

    Returns:
        A :class:`RunResult`; ``source`` is ``sglang`` when server metrics
        were used, otherwise the provenance recorded by the client fallback.

    Raises:
        RuntimeError: If the server generates no tokens.
        urllib.error.URLError: If the server is unreachable or times out.
    """
    if engine == TENSORFOLD_ENGINE:
        return run_completion(base_url, model, prompt, gen_tokens, timeout)

    metrics_unavailable: Exception | None = None
    before: Dict[str, float] | None = None
    try:
        before = snapshot_server_metrics(base_url, timeout)
    except (RuntimeError, urllib.error.URLError) as e:
        if isinstance(e, urllib.error.URLError):
            raise
        metrics_unavailable = e

    result = run_completion(base_url, model, prompt, gen_tokens, timeout)
    if result.source == TENSORFOLD_ENGINE:
        return result
    if before is None:
        if metrics_unavailable is not None:
            print(f"  (server metrics unavailable, using client timing: {metrics_unavailable})")
        return result

    deadline = time.monotonic() + METRICS_POLL_DEADLINE_S
    while True:
        after = snapshot_server_metrics(base_url, timeout)
        rates = metrics_to_rates(diff_metrics(before, after), result)
        if rates is not None:
            return rates
        if time.monotonic() >= deadline:
            break
        time.sleep(METRICS_POLL_INTERVAL_S)
    return result


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


def detect_engine(base_url: str, timeout: int) -> str:
    """Identify the serving engine behind an OpenAI-compatible endpoint.

    SGLang exposes ``sglang:`` Prometheus series on ``/metrics``; TensorFold
    signs every model entry in ``/v1/models`` with ``owned_by: "tensorfold"``.
    Detection is best effort: any failure to reach or parse the endpoint
    yields an empty string and the bench falls back to client timing.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        timeout: Request timeout in seconds.

    Returns:
        The engine name, currently ``"tensorfold"`` when detected, else ``""``.
    """
    try:
        req = urllib.request.Request(base_url + "/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        for entry in body.get("data", []):
            if entry.get("owned_by") == TENSORFOLD_ENGINE:
                return TENSORFOLD_ENGINE
    except Exception:  # noqa: BLE001 - detection is best effort
        pass
    return ""


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


def render_menu(labels: List[str], selected: set[int], cursor: int, first: bool, title: str) -> str:
    """Render a multi-select menu.

    Args:
        labels: The text shown for each option.
        selected: Indices of currently toggled-on options.
        cursor: Index of the option the cursor is on.
        first: When True, render the title line too.
        title: The menu's instruction line.

    Returns:
        The escape-sequence string to write to the terminal.
    """
    rows = []
    if first:
        rows.append("")  # blank line separating this menu from earlier output
    rows.append(title)
    for i, label in enumerate(labels):
        mark = "x" if i in selected else " "
        pointer = ">" if i == cursor else " "
        rows.append(f" {pointer} [{mark}] {label}")
    rows.append("")  # blank line separating the menu from what follows
    # Redraw the menu in place, erasing each line before rewriting it. The
    # title is rendered on every frame because the erase pass wipes it.
    return "\x1b[s" + "".join(f"\r\x1b[2K{row}\n" for row in rows) + f"\x1b[{len(rows)}A"


def menu_line_count(num_labels: int) -> int:
    """Count the terminal lines a redrawn menu occupies below its title.

    Args:
        num_labels: The number of option rows in the menu.

    Returns:
        The number of lines from the title line to the trailing separator
        line, which is also the distance to move the cursor down when the
        menu ends.
    """
    return num_labels + 2


def menu_multi_select(
    labels: List[str], title: str, allow_empty: bool, empty_hint: str = ""
) -> set[int] | None:
    """Show an interactive multi-select menu and return the chosen indices.

    All options start selected. Space toggles the option under the cursor,
    ``a`` toggles all, enter confirms, and ``q`` aborts.

    Args:
        labels: The text shown for each option.
        title: The menu's instruction line.
        allow_empty: When False, an empty selection re-prompts instead of
            confirming.
        empty_hint: Message shown when confirming an empty selection is not
            allowed.

    Returns:
        The chosen option indices, or None when the user quits.

    Raises:
        ValueError: If not attached to a terminal.
        KeyboardInterrupt: If the user aborts with Ctrl-C.
    """
    if not sys.stdin.isatty():
        raise ValueError("no terminal attached; use the --ctx/--task flags instead")
    rows_len = menu_line_count(len(labels))
    selected: set[int] = set(range(len(labels)))
    cursor = 0
    print(render_menu(labels, selected, cursor, True, title), end="", flush=True)
    while True:
        key = read_key()
        if key == "quit":
            print(f"\x1b[{rows_len}B")
            return None
        if key == "enter":
            if not selected and not allow_empty:
                print(f"\r\x1b[2K{empty_hint}", end="", flush=True)
                continue
            print(f"\x1b[{rows_len}B")
            return selected
        if key == "up":
            cursor = (cursor - 1) % len(labels)
        elif key == "down":
            cursor = (cursor + 1) % len(labels)
        elif key == "space":
            if cursor in selected:
                selected.remove(cursor)
            else:
                selected.add(cursor)
        elif key == "a":
            if len(selected) == len(labels):
                selected = set()
            else:
                selected = set(range(len(labels)))
        print(render_menu(labels, selected, cursor, False, title), end="", flush=True)


def menu_select_ctx(options: List[int]) -> List[int] | None:
    """Show the interactive context-length multi-select menu.

    Args:
        options: The selectable context sizes, in kilotokens.

    Returns:
        The chosen context sizes in kilotokens, or None when the user quits.
    """
    if not sys.stdin.isatty():
        raise ValueError("no terminal attached; pass --ctx 0,8,... instead")
    chosen = menu_multi_select(
        [f"{opt}k" if opt else "0" for opt in options],
        "Select context lengths (space: toggle, up/down: move,"
        " a: all/none, enter: start, q: quit)",
        allow_empty=False,
        empty_hint="Select at least one context size.",
    )
    if chosen is None:
        return None
    return [options[i] for i in sorted(chosen)]


def menu_select_tasks() -> List[str] | None:
    """Show the interactive generation-task multi-select menu.

    Returns:
        The chosen task names, or None when the user quits.
    """
    if not sys.stdin.isatty():
        raise ValueError("no terminal attached; pass --task code,prose")
    chosen = menu_multi_select(
        [f"{name} ({prompt.rstrip('.')})" for name, prompt in TASKS.items()],
        "Select generation tasks (space: toggle, up/down: move,"
        " a: all/none, enter: start, q: quit)",
        allow_empty=False,
        empty_hint="Select at least one task.",
    )
    if chosen is None:
        return None
    return [list(TASKS)[i] for i in sorted(chosen)]


def parse_task(raw: str) -> List[str]:
    """Parse a comma-separated ``--task`` value list.

    Args:
        raw: Comma-separated task names, for example ``code,prose``.

    Returns:
        The parsed task names.

    Raises:
        argparse.ArgumentTypeError: If any name is not a known task.
    """
    names = [n.strip() for n in raw.split(",") if n.strip()]
    unknown = [n for n in names if n not in TASKS]
    if not names:
        raise argparse.ArgumentTypeError("select at least one task")
    if unknown:
        known = ", ".join(TASKS)
        raise argparse.ArgumentTypeError(
            f"unknown task(s): {', '.join(unknown)} (known: {known})"
        )
    return names


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
        "--task",
        type=parse_task,
        help=(
            "comma-separated generation tasks (code, prose); required"
            " if non-interactive, otherwise the menu opens"
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
    print("Starting warm-up passes...")
    for _ in range(WARMUP_PASSES):
        run_completion(base_url, model, WARMUP_PROMPT, WARMUP_GEN_TOKENS, timeout)
    print("Finished warm-ups.")


def format_header() -> str:
    """Build the aligned header row for the per-length results table.

    Returns:
        The fixed-width header line; its length matches the dash separator.
    """
    return (
        f"{'ctx':>8} {'task':<6} {'prompt_tok':>13} {'gen_tok':>10} "
        f"{'prefill tok/s':>16} {'tok/s':>12}"
    )


def format_row(
    ctx_k: int,
    task: str,
    prompt_tok: int,
    gen_tok: int,
    prefill_tok_s: float,
    tok_s: float,
    source: str = "usage",
    draft_acceptance: float = 0.0,
) -> str:
    """Build one aligned results-table row.

    Args:
        ctx_k: The context size of this row, in kilotokens.
        task: The generation task of this row.
        prompt_tok: Prompt tokens reported by the server for this run.
        gen_tok: Completion tokens generated in this run.
        prefill_tok_s: Estimated prefill throughput; 0.0 means unknown.
        tok_s: Generation throughput of this run in tokens per second.
        source: Provenance of the rates; ``estimated`` rows get an asterisk
            on the throughput columns.
        draft_acceptance: TensorFold speculative-draft acceptance rate; when
            above 0 a ``draft=NN%`` note is appended to the row.

    Returns:
        The formatted row, aligned with the header columns. Unknown prefill
        is rendered as ``-``.
    """
    ctx_str = f"{ctx_k}k" if ctx_k else "0"
    prefill_str = f"{prefill_tok_s:.2f}" if prefill_tok_s > 0 else "-"
    if source == "estimated":
        prefill_str += "*"
        tok_str = f"{tok_s:.2f}*"
    else:
        tok_str = f"{tok_s:.2f}"
    row = (
        f"{ctx_str:>8} {task:<6} {prompt_tok:>13} {gen_tok:>10} "
        f"{prefill_str:>16} {tok_str:>12}"
    )
    if draft_acceptance > 0:
        row += f"  draft={draft_acceptance:.0%}"
    return row


def bench_lengths(
    base_url: str,
    model: str,
    lengths: List[int],
    tasks: List[str],
    gen_tokens: int,
    timeout: int,
    engine: str = "",
) -> None:
    """Benchmark every selected task at each context size and print the rows.

    Args:
        base_url: Server root, for example ``http://127.0.0.1:8080``.
        model: Model name as served by the endpoint.
        lengths: Context sizes to test, in kilotokens.
        tasks: Generation task names, in :data:`TASKS`.
        gen_tokens: Maximum tokens to generate per run.
        timeout: Request timeout in seconds.
        engine: Engine name from :func:`detect_engine`, for the measured path
            to select the server-metrics mechanism.

    Returns:
        None. Each completed run is printed as one table row.
    """
    for ctx in lengths:
        for task in tasks:
            prompt = build_prompt(ctx, TASKS[task])
            try:
                result = run_completion_measured(
                    base_url, model, prompt, gen_tokens, timeout, engine
                )
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"{ctx:>8} {task:<6}   FAILED: {e}")
                continue
            print(
                format_row(
                    ctx,
                    task,
                    result.prompt_tok,
                    result.gen_tok,
                    result.prefill_tok_s,
                    result.gen_tok_s,
                    result.source,
                    result.draft_acceptance,
                )
            )


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

    tasks = args.task
    if tasks is None:
        try:
            tasks = menu_select_tasks()
        except ValueError as e:
            raise SystemExit(f"error: {e}") from e
        except KeyboardInterrupt:
            raise SystemExit("cancelled.") from None
        if tasks is None:
            raise SystemExit("cancelled.")

    try:
        warm_up(base_url, model, args.timeout)
    except Exception as e:
        raise SystemExit(f"error: cannot reach server at {base_url}: {e}") from e

    engine = detect_engine(base_url, args.timeout)
    engine_note = f" engine={engine}" if engine else " engine=unknown"
    print(f"llm-bench: {base_url}  model={model}{engine_note}")
    print(
        "throughput numbers come from the SGLang server's own /metrics"
        " when available, or the TensorFold per-response telemetry block;"
        " client timing is the fallback"
    )
    print(f"gen_tokens={args.gen_tokens}, one run per context size")
    hdr = format_header()
    print("\n" + hdr)
    print("-" * len(hdr))

    bench_lengths(
        base_url,
        model,
        lengths,
        tasks,
        args.gen_tokens,
        args.timeout,
        engine,
    )


if __name__ == "__main__":
    main()
