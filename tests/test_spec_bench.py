"""Tests for llm_bench.spec_bench."""

import argparse
import io
import json
import re
import time
import urllib.error
from unittest.mock import patch

import pytest

from llm_bench.spec_bench import (
    BearerAuthHandler,
    METRIC_E2E_COUNT,
    METRIC_E2E_SUM,
    METRIC_GEN_TOKENS,
    METRIC_PROMPT_TOKENS,
    METRIC_TTFT_COUNT,
    METRIC_TTFT_SUM,
    WARMUP_GEN_TOKENS,
    WARMUP_PROMPT,
    TASKS,
    bench_lengths,
    build_prompt,
    choose_model,
    diff_metrics,
    detect_engine,
    format_header,
    format_row,
    menu_select_ctx,
    menu_select_tasks,
    metrics_to_rates,
    parse_args,
    parse_ctx,
    parse_metrics_text,
    parse_parallel,
    parse_task,
    probe_tensorfold,
    query_models,
    render_menu,
    resolve_base_url,
    run_completion,
    run_completion_measured,
    run_parallel_batch,
    snapshot_server_metrics,
    TENSORFOLD_ENGINE,
    warm_up,
)

from llm_bench.spec_bench import RunResult


class TestBuildPrompt:
    """build_prompt behavior."""

    def test_zero_ctx_returns_short_prompt(self) -> None:
        assert build_prompt(0) == "Write a Python Snake game."

    def test_negative_ctx_returns_short_prompt(self) -> None:
        assert build_prompt(-5) == "Write a Python Snake game."

    def test_larger_ctx_gives_longer_prompt(self) -> None:
        short = build_prompt(1)
        long = build_prompt(8)
        assert len(long) > len(short)

    def test_kilotokens_scale_the_prompt(self) -> None:
        one_k = len(build_prompt(1))
        two_k = len(build_prompt(2))
        assert abs(two_k / one_k - 2) < 0.1

    def test_prompt_mentions_task(self) -> None:
        assert "Write a Python Snake game." in build_prompt(1)

    def test_prompts_are_unique_per_call(self) -> None:
        assert build_prompt(8) != build_prompt(8)

    def test_seeded_prompts_are_deterministic(self) -> None:
        import random

        rng = random.Random(42)
        assert build_prompt(1, "Count.", rng) == build_prompt(1, "Count.", random.Random(42))


class TestParseArgs:
    """parse_args behavior."""

    def test_required_args(self) -> None:
        args = parse_args(["--model", "m", "http://10.0.0.5:8080"])
        assert args.model == "m"
        assert args.base_url == "http://10.0.0.5:8080"
        assert args.gen_tokens == 2048

    def test_missing_base_url_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--model", "m"])

    def test_custom_ctx(self) -> None:
        args = parse_args(["--model", "m", "http://h:1", "--ctx", "2,8,16"])
        assert args.ctx == [2, 8, 16]

    def test_invalid_ctx_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--model", "m", "http://h:1", "--ctx", "2,eight"])

    def test_negative_ctx_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--model", "m", "http://h:1", "--ctx", "-4"])


class TestResolveBaseUrl:
    """resolve_base_url behavior."""

    def test_strips_trailing_v1(self) -> None:
        with patch("llm_bench.spec_bench.query_models") as qm:
            assert resolve_base_url("http://h:8000/v1/", 10) == "http://h:8000"
        qm.assert_not_called()

    def test_bare_url_reachable(self) -> None:
        with patch("llm_bench.spec_bench.query_models") as qm:
            qm.return_value = ["m"]
            assert resolve_base_url("http://h:8000", 10) == "http://h:8000"
            qm.assert_called_once_with("http://h:8000", 10)

    def test_falls_back_to_v1_suffix(self) -> None:
        with patch("llm_bench.spec_bench.query_models") as qm:
            qm.side_effect = [urllib.error.URLError("no"), ["m"]]
            assert resolve_base_url("http://h:8000", 10) == "http://h:8000/v1"
            assert qm.call_args_list[1].args == ("http://h:8000/v1", 10)

    def test_unreachable_exits(self) -> None:
        with patch("llm_bench.spec_bench.query_models") as qm:
            qm.side_effect = urllib.error.URLError("no")
            with pytest.raises(SystemExit):
                resolve_base_url("http://h:8000", 10)


class TestMenuSelectCtx:
    """menu_select_ctx behavior."""

    def test_requires_tty(self) -> None:
        with patch("llm_bench.spec_bench.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            with pytest.raises(ValueError, match="no terminal"):
                menu_select_ctx([2, 8])


class TestParseCtx:
    """parse_ctx behavior."""

    def test_parses_comma_separated_values(self) -> None:
        assert parse_ctx("2,8,16") == [2, 8, 16]

    def test_whitespace_is_ignored(self) -> None:
        assert parse_ctx("2, 8 ,16") == [2, 8, 16]

    def test_rejects_non_numeric_values(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_ctx("2,eight")

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_ctx("2,-4")


class TestQueryModels:
    """query_models behavior."""

    def _body(self, body: dict) -> io.BytesIO:
        return io.BytesIO(json.dumps(body).encode())

    def test_returns_model_ids(self) -> None:
        resp = self._body({"data": [{"id": "a"}, {"id": "b"}]})
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            assert query_models("http://host:1", 10) == ["a", "b"]

    def test_empty_model_list_raises(self) -> None:
        resp = self._body({"data": []})
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            with pytest.raises(RuntimeError, match="no models"):
                query_models("http://host:1", 10)


class TestChooseModel:
    """choose_model behavior."""

    def test_selects_numbered_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda *_: "2")
        assert choose_model(["a", "b"]) == 1

    def test_retries_on_invalid_then_accepts(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        answers = iter(["nope", "1"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))
        assert choose_model(["a"]) == 0
        assert "Invalid selection" in capsys.readouterr().out

    def test_quit_flag_returns_minus_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda *_: "q")
        assert choose_model(["a", "b"]) == -1

    def test_eof_returns_minus_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_eof(*_: object) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert choose_model(["a"]) == -1


class TestTableFormatting:
    """format_header and format_row alignment behavior."""

    def test_row_has_five_columns(self) -> None:
        row = format_row(8, "prose", 155, 81, 900.0, 152.65)
        cols = row.split()
        assert cols == ["8k", "prose", "155", "81", "900.00", "152.65"]

    def test_row_shows_dash_for_unknown_prefill(self) -> None:
        row = format_row(0, "prose", 20, 93, 0.0, 191.65)
        assert row.split()[4] == "-"

    def test_tok_s_header_and_value_share_right_edge(self) -> None:
        header = format_header()
        row = format_row(0, "prose", 20, 93, 900.0, 191.65)
        assert header.split() == [
            "ctx", "task", "prompt_tok", "gen_tok", "prefill", "tok/s", "tok/s"
        ]
        header_end = header.rindex("tok/s") + len("tok/s")
        row_end = row.index("191.65") + len("191.65")
        assert header_end == row_end

    def test_row_appends_draft_acceptance_when_positive(self) -> None:
        row = format_row(0, "code", 20, 93, 900.0, 191.65, "tensorfold", 0.654)
        assert row.endswith("draft=65%")

    def test_row_has_no_draft_note_when_zero(self) -> None:
        row = format_row(0, "code", 20, 93, 900.0, 191.65)
        assert "draft" not in row

    def test_row_appends_parallel_note_when_above_one(self) -> None:
        row = format_row(0, "code", 20, 93, 900.0, 191.65, parallel=4)
        assert row.endswith("par=4")

    def test_row_has_no_parallel_note_when_single(self) -> None:
        row = format_row(0, "code", 20, 93, 900.0, 191.65)
        assert "par=" not in row


class TestRunCompletion:
    """run_completion streaming behavior."""

    def _sse_body(self, chunks: list) -> io.BytesIO:
        lines = [b'data: ' + json.dumps(c).encode() for c in chunks]
        lines.append(b"data: [DONE]")
        return io.BytesIO(b"\n".join(lines) + b"\n")

    def test_measures_prefill_and_generation_from_usage(self) -> None:
        chunks = [
            {"choices": [{"delta": {"content": "He"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
            {"usage": {"prompt_tokens": 600, "completion_tokens": 2}},
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            r = run_completion("http://host:1", "m", "hi", 8, 10)
        assert r.prompt_tok == 600
        assert r.gen_tok == 2
        assert r.ttft_s > 0
        assert r.prefill_tok_s > 0
        assert r.gen_tok_s > 0
        assert r.source == "usage"

    def test_reasoning_deltas_start_the_ttft_clock(self) -> None:
        chunks = [
            {"choices": [{"delta": {"reasoning": "think"}}]},
            {"choices": [{"delta": {"content": "ans"}}]},
            {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            r = run_completion("http://host:1", "m", "hi", 8, 10)
        assert r.ttft_s > 0
        assert r.gen_tok == 2
        assert r.source == "usage"

    def test_uses_llama_cpp_timings_when_present(self) -> None:
        chunks = [
            {"choices": [{"delta": {"content": "a"}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 900, "completion_tokens": 50},
                "timings": {
                    "prompt_n": 1000,
                    "prompt_ms": 100.0,
                    "predicted_n": 50,
                    "predicted_ms": 500.0,
                },
            },
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            r = run_completion("http://host:1", "m", "hi", 8, 10)
        assert r.source == "timings"
        assert r.prefill_tok_s == 1000 / 0.1
        assert r.gen_tok_s == 50 / 0.5
        assert r.prompt_tok == 1000
        assert r.gen_tok == 50

    def test_falls_back_to_chunk_count_and_flags_estimate(self) -> None:
        chunks = [
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}}]},
            {"choices": [{"delta": {"content": "c"}}]},
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            r = run_completion("http://host:1", "m", "hi", 8, 10)
        assert r.source == "estimated"
        assert r.gen_tok == 3
        assert r.prompt_tok == 0
        assert r.prefill_tok_s == 0.0

    def test_no_tokens_raises(self) -> None:
        resp = self._sse_body([])
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            with pytest.raises(RuntimeError, match="no tokens"):
                run_completion("http://host:1", "m", "hi", 8, 10)

    def test_uses_tensorfold_telemetry_block(self) -> None:
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "ans"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 80},
                "tensorfold": {
                    "prefill_s": 0.05,
                    "decode_s": 0.4,
                    "drafts": True,
                    "drafted": 100,
                    "accepted": 60,
                    "decode_tps": 200.0,
                },
            },
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            r = run_completion("http://host:1", "m", "hi", 8, 10)
        assert r.source == TENSORFOLD_ENGINE
        assert r.prefill_tok_s == pytest.approx(100 / 0.05)
        assert r.gen_tok_s == pytest.approx(80 / 0.4)
        assert r.prompt_tok == 100
        assert r.gen_tok == 80
        assert r.draft_acceptance == pytest.approx(0.6)

    def test_tensorfold_reasoning_content_starts_ttft(self) -> None:
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            r = run_completion("http://host:1", "m", "hi", 8, 10)
        assert r.ttft_s > 0
        assert r.gen_tok == 2
        assert r.draft_acceptance == 0.0

    def test_tensorfold_block_missing_prefill_is_ignored(self) -> None:
        chunks = [
            {"choices": [{"delta": {"content": "a"}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                "tensorfold": {"prefill_s": 0.0, "decode_s": 0.0},
            },
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            r = run_completion("http://host:1", "m", "hi", 8, 10)
        assert r.source == "usage"
        assert r.draft_acceptance == 0.0

    def test_reads_renamed_telemetry_keys_and_speculative_block(self) -> None:
        # Exact shape served through llama-swap on vyper: prefill_seconds/
        # seconds names, draft counters in a sibling "speculative" block.
        chunks = [
            {"choices": [{"delta": {"content": "ans"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 8},
                "tensorfold": {
                    "engine": "lanes",
                    "tokens_per_second": 1165.27,
                    "seconds": 0.027085224,
                    "prefill_seconds": 0.021047919,
                    "time_to_first_token": 0.021342127,
                    "sampling": "exact",
                },
                "speculative": {
                    "rounds": 1,
                    "drafted": 7,
                    "accepted": 7,
                    "acceptance_rate": 1.0,
                },
            },
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            r = run_completion("http://host:1", "m", "hi", 8, 10)
        assert r.source == TENSORFOLD_ENGINE
        assert r.prefill_tok_s == pytest.approx(17 / 0.021047919)
        assert r.gen_tok_s == pytest.approx(8 / 0.027085224)
        assert r.draft_acceptance == pytest.approx(1.0)

    def test_prefill_seconds_without_seconds_is_ignored(self) -> None:
        chunks = [
            {"choices": [{"delta": {"content": "a"}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                "tensorfold": {"prefill_seconds": 0.02},
            },
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            r = run_completion("http://host:1", "m", "hi", 8, 10)
        assert r.source == "usage"


class TestDetectEngine:
    """detect_engine behavior."""

    def _body(self, body: dict) -> io.BytesIO:
        return io.BytesIO(json.dumps(body).encode())

    def test_detects_tensorfold_owned_by(self) -> None:
        body = {"data": [{"id": "m", "owned_by": "tensorfold"}]}
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=self._body(body)):
            assert detect_engine("http://host:1", 10) == "tensorfold"

    def test_detects_sglang_owned_by(self) -> None:
        body = {"data": [{"id": "m", "owned_by": "sglang"}]}
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=self._body(body)):
            assert detect_engine("http://host:1", 10) == "sglang"

    def test_detects_vllm_owned_by(self) -> None:
        body = {"data": [{"id": "m", "owned_by": "vllm"}]}
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=self._body(body)):
            assert detect_engine("http://host:1", 10) == "vllm"

    def test_sniffs_vllm_from_metrics_when_owned_by_unknown(self) -> None:
        body = {"data": [{"id": "m", "owned_by": "organization"}]}
        metrics = io.BytesIO(b"vllm:prompt_tokens_total{model_name=\"m\"} 1.0\n")
        with patch(
            "llm_bench.spec_bench.urllib.request.urlopen",
            side_effect=[self._body(body), metrics],
        ):
            assert detect_engine("http://host:1", 10) == "vllm"

    def test_unknown_engine_returns_empty(self) -> None:
        body = {"data": [{"id": "m", "owned_by": "organization"}]}
        metrics = io.BytesIO(b"# HELP python_info Python platform.\n")
        with patch(
            "llm_bench.spec_bench.urllib.request.urlopen",
            side_effect=[self._body(body), metrics],
        ):
            assert detect_engine("http://host:1", 10) == ""

    def test_detects_tensorfold_behind_proxy_via_probe(self) -> None:
        # llama-swap re-signs every model and serves its own /metrics.
        body = {"data": [{"id": "nemo-lightning", "owned_by": "llama-swap"}]}
        sse = b'data: {"choices": [], "tensorfold": {"seconds": 0.1}}\n\ndata: [DONE]\n'
        with patch(
            "llm_bench.spec_bench.urllib.request.urlopen",
            side_effect=[
                self._body(body),
                io.BytesIO(b"# HELP llamaswap_cpu_util_percent CPU.\n"),
                io.BytesIO(sse),
            ],
        ):
            assert detect_engine("http://host:1", 10) == "tensorfold"

    def test_probe_failure_falls_back_to_empty(self) -> None:
        body = {"data": [{"id": "m", "owned_by": "llama-swap"}]}
        with patch(
            "llm_bench.spec_bench.urllib.request.urlopen",
            side_effect=[self._body(body), self._body(body), RuntimeError("boom")],
        ):
            assert detect_engine("http://host:1", 10) == ""

    def test_probe_tensorfold_scans_raw_stream(self) -> None:
        sse = b'data: {"choices": [], "tensorfold": {"seconds": 0.1}}\n\ndata: [DONE]\n'
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=io.BytesIO(sse)):
            assert probe_tensorfold("http://host:1", "m", 10) is True

    def test_unreachable_server_returns_empty(self) -> None:
        def raise_url_error(*_: object) -> None:
            raise RuntimeError("boom")

        with patch("llm_bench.spec_bench.urllib.request.urlopen", side_effect=raise_url_error):
            assert detect_engine("http://host:1", 10) == ""

    def test_http_404_metrics_is_unavailable_not_fatal(self) -> None:
        with patch(
            "llm_bench.spec_bench.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "http://host:1/metrics", 404, "not found", None, None  # type: ignore[arg-type]
            ),
        ):
            with pytest.raises(RuntimeError, match="HTTP 404"):
                snapshot_server_metrics("http://host:1", 10)


class TestRunCompletionMeasuredTensorfold:
    """TensorFold shortcut path in run_completion_measured."""

    def test_skips_metrics_snapshot_when_engine_detected(self) -> None:
        client = _client_result(source="tensorfold", draft_acceptance=0.65)
        with (
            patch("llm_bench.spec_bench.snapshot_server_metrics") as mock_snap,
            patch("llm_bench.spec_bench.run_completion", return_value=client) as mock_run,
        ):
            r = run_completion_measured(
                "http://host:1", "m", "hi", 8, 10, engine=TENSORFOLD_ENGINE
            )
        assert r is client
        mock_snap.assert_not_called()
        mock_run.assert_called_once()

    def test_engine_argument_is_passed_when_engine_unknown(self) -> None:
        before = {METRIC_TTFT_SUM: 1.0, METRIC_TTFT_COUNT: 1.0,
                  METRIC_E2E_SUM: 3.0, METRIC_E2E_COUNT: 1.0,
                  METRIC_PROMPT_TOKENS: 100.0, METRIC_GEN_TOKENS: 100.0}
        after = {k: v + d for k, v, d in zip(
            list(before), before.values(),
            [0.5, 1.0, 2.5, 1.0, 1000.0, 200.0])}
        client = _client_result()
        with (
            patch("llm_bench.spec_bench.snapshot_server_metrics", side_effect=[before, after]),
            patch("llm_bench.spec_bench.run_completion", return_value=client),
        ):
            r = run_completion_measured("http://host:1", "m", "hi", 8, 10)
        assert r.source == "sglang"


class TestRunCompletionMeasuredVllm:
    """vLLM /metrics path in run_completion_measured."""

    def _vllm_snapshots(self) -> tuple[dict, dict]:
        before = {
            "vllm:time_to_first_token_seconds_sum": 1.0,
            "vllm:time_to_first_token_seconds_count": 1.0,
            "vllm:e2e_request_latency_seconds_sum": 3.0,
            "vllm:e2e_request_latency_seconds_count": 1.0,
            "vllm:prompt_tokens_total": 100.0,
            "vllm:generation_tokens_total": 100.0,
        }
        after = {k: v + d for k, v, d in zip(
            list(before), before.values(),
            [0.5, 1.0, 2.5, 1.0, 1000.0, 200.0])}
        return before, after

    def test_reports_sglang_source_from_vllm_metrics(self) -> None:
        before, after = self._vllm_snapshots()
        client = _client_result()
        with (
            patch(
                "llm_bench.spec_bench.snapshot_server_metrics",
                side_effect=[before, after],
            ),
            patch("llm_bench.spec_bench.run_completion", return_value=client),
        ):
            r = run_completion_measured(
                "http://host:1", "m", "hi", 8, 10, engine="vllm"
            )
        assert r.source == "sglang"
        assert r.prompt_tok == 1000
        assert r.gen_tok == 200

    def test_snapshot_receives_vllm_prefix_and_names(self) -> None:
        before, after = self._vllm_snapshots()
        client = _client_result()
        with (
            patch(
                "llm_bench.spec_bench.snapshot_server_metrics",
                side_effect=[before, after],
            ) as mock_snap,
            patch("llm_bench.spec_bench.run_completion", return_value=client),
        ):
            run_completion_measured("http://host:1", "m", "hi", 8, 10, engine="vllm")
        for call in mock_snap.call_args_list:
            assert call.kwargs["prefix"] == "vllm:"
            assert all(name.startswith("vllm:") for name in call.kwargs["required"])


class TestWarmUp:
    """warm_up behavior."""

    def test_runs_two_short_passes(self) -> None:
        calls = []

        def fake_completion(
            base_url: str, model: str, prompt: str, gen_tokens: int, timeout: int
        ) -> tuple[float, int, int, float]:
            calls.append((prompt, gen_tokens))
            return 1.0, 2, 3, 4.0

        with patch("llm_bench.spec_bench.run_completion", side_effect=fake_completion):
            warm_up("http://host:1", "m", 10)
        assert len(calls) == 2
        assert all(p == WARMUP_PROMPT for p, _g in calls)
        assert all(g == WARMUP_GEN_TOKENS for _p, g in calls)


class TestRenderMenu:
    """render_menu spacing behavior."""

    def test_menu_is_surrounded_by_blank_lines(self) -> None:
        lines = render_menu(["a", "b"], {0}, 0, True, "Pick").split("\n")
        plain = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\r", "", line) for line in lines]
        assert plain[0] == "" and plain[1].strip() == "Pick"  # blank line before title
        assert plain[-2] == ""  # blank line after the last option

    def test_options_are_single_spaced(self) -> None:
        lines = render_menu(["a", "b"], {0}, 0, True, "Pick").split("\n")
        plain = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\r", "", line) for line in lines]
        option_rows = [line for line in plain if "[x]" in line or "[ ]" in line]
        assert len(option_rows) == 2
        a_idx, b_idx = plain.index(option_rows[0]), plain.index(option_rows[1])
        assert b_idx - a_idx == 1  # options adjacent, no blank lines between

    def test_redraw_keeps_title_and_does_not_duplicate_rows(self) -> None:
        labels = ["a", "b", "c"]
        redraw = render_menu(labels, {0}, 1, False, "Pick").split("\n")
        plain = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\r", "", line) for line in redraw]
        assert plain[0].strip() == "Pick"  # title re-rendered every frame
        option_rows = [line for line in plain if "[x]" in line or "[ ]" in line]
        assert len(option_rows) == len(labels)  # no duplicated trailing rows
        assert plain[-2] == ""  # exactly one trailing separator line


class TestParseTask:
    """parse_task behavior."""

    def test_parses_comma_separated_names(self) -> None:
        assert parse_task("code,prose") == ["code", "prose"]

    def test_rejects_unknown_task(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="unknown task"):
            parse_task("sing")

    def test_empty_selection_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="at least one"):
            parse_task("")


class TestMenuSelectTasks:
    """menu_select_tasks behavior."""

    def test_requires_tty(self) -> None:
        with patch("llm_bench.spec_bench.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            with pytest.raises(ValueError, match="--task"):
                menu_select_tasks()

    def test_returns_selected_task_names(self) -> None:
        with patch("llm_bench.spec_bench.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = True
            with patch("llm_bench.spec_bench.menu_multi_select", return_value={0, 1}):
                assert menu_select_tasks() == [list(TASKS)[0], list(TASKS)[1]]

    def test_requires_tty_context_menu(self) -> None:
        with patch("llm_bench.spec_bench.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            with pytest.raises(ValueError, match="--ctx"):
                menu_select_ctx([2, 8])


def _client_result(**overrides: object) -> RunResult:
    """Build a representative client-measured RunResult for tests."""
    fields = {
        "prefill_tok_s": 10.0,
        "gen_tok_s": 20.0,
        "prompt_tok": 100,
        "gen_tok": 50,
        "ttft_s": 0.4,
        "source": "usage",
        "draft_acceptance": 0.0,
    }
    fields.update(overrides)
    return RunResult(**fields)  # type: ignore[arg-type]


class TestParseMetricsText:
    """parse_metrics_text behavior."""

    def test_aggregates_labeled_series_by_base_name(self) -> None:
        text = (
            'sglang:prompt_tokens_total{is_streaming="false"} 10.0\n'
            'sglang:prompt_tokens_total{is_streaming="true"} 32.0\n'
            'sglang:time_to_first_token_seconds_sum{is_streaming="true"} 0.5\n'
        )
        values = parse_metrics_text(text)
        assert values["sglang:prompt_tokens_total"] == 42.0
        assert values["sglang:time_to_first_token_seconds_sum"] == 0.5

    def test_skips_help_type_and_bucket_lines(self) -> None:
        text = (
            "# HELP sglang:prompt_tokens_total Number of prefill tokens.\n"
            "# TYPE sglang:prompt_tokens_total counter\n"
            'sglang:latency_bucket{le="0.1"} 3.0\n'
            'sglang:latency_count{is_streaming="true"} 2.0\n'
        )
        values = parse_metrics_text(text)
        assert "sglang:prompt_tokens_total" not in values
        assert "sglang:latency_bucket" not in values
        assert values["sglang:latency_count"] == 2.0

    def test_skips_unparseable_lines(self) -> None:
        values = parse_metrics_text("sglang:weird_metric not_a_number\n")
        assert values == {}

    def test_parses_vllm_prefix_when_requested(self) -> None:
        text = (
            'vllm:prompt_tokens_total{engine="0",model_name="m"} 24.0\n'
            "# HELP vllm:generation_tokens_total Number of generated tokens.\n"
            'vllm:time_to_first_token_seconds_sum{engine="0"} 0.75\n'
            'sglang:prompt_tokens_total{is_streaming="false"} 10.0\n'
        )
        values = parse_metrics_text(text, "vllm:")
        assert values["vllm:prompt_tokens_total"] == 24.0
        assert values["vllm:time_to_first_token_seconds_sum"] == 0.75
        assert "sglang:prompt_tokens_total" not in values


class TestSnapshotServerMetrics:
    """snapshot_server_metrics behavior."""

    def test_returns_values_when_all_required_present(self) -> None:
        text = "".join(
            f"{name}{{m=\"GLM\"}} 1.0\n"
            for name in (
                METRIC_PROMPT_TOKENS,
                METRIC_GEN_TOKENS,
                METRIC_TTFT_SUM,
                METRIC_TTFT_COUNT,
                METRIC_E2E_SUM,
                METRIC_E2E_COUNT,
            )
        )
        resp = io.BytesIO(text.encode())
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            values = snapshot_server_metrics("http://host:1", 10)
        assert values[METRIC_PROMPT_TOKENS] == 1.0

    def test_retries_until_required_metrics_appear(self) -> None:
        partial = io.BytesIO(b"sglang:prompt_tokens_total 1.0\n")
        full = io.BytesIO(
            b"".join(
                f"{name} 1.0\n".encode()
                for name in (
                    METRIC_PROMPT_TOKENS,
                    METRIC_GEN_TOKENS,
                    METRIC_TTFT_SUM,
                    METRIC_TTFT_COUNT,
                    METRIC_E2E_SUM,
                    METRIC_E2E_COUNT,
                )
            )
        )
        with (
            patch(
                "llm_bench.spec_bench.urllib.request.urlopen",
                side_effect=[partial, full],
            ),
            patch("llm_bench.spec_bench.time.sleep") as mock_sleep,
        ):
            values = snapshot_server_metrics("http://host:1", 10)
        assert values[METRIC_GEN_TOKENS] == 1.0
        mock_sleep.assert_called()

    def test_raises_after_exhausted_retries(self) -> None:
        resp = io.BytesIO(b"sglang:prompt_tokens_total 1.0\n")
        with (
            patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp),
            patch("llm_bench.spec_bench.time.sleep"),
            pytest.raises(RuntimeError, match="cannot read /metrics"),
        ):
            snapshot_server_metrics("http://host:1", 10)


class TestDiffMetrics:
    """diff_metrics behavior."""

    def test_computes_deltas_for_shared_keys(self) -> None:
        before = {"a": 1.0, "b": 10.0}
        after = {"a": 4.0, "b": 7.0, "c": 2.0}
        assert diff_metrics(before, after) == {"a": 3.0, "b": -3.0}

    def test_handles_empty_snapshots(self) -> None:
        assert diff_metrics({}, {"a": 1.0}) == {}
        assert diff_metrics({"a": 1.0}, {}) == {}


class TestMetricsToRates:
    """metrics_to_rates behavior."""

    def _deltas(self, **overrides: float) -> dict:
        deltas = {
            METRIC_TTFT_SUM: 0.5,
            METRIC_TTFT_COUNT: 1.0,
            METRIC_E2E_SUM: 2.5,
            METRIC_E2E_COUNT: 1.0,
            METRIC_PROMPT_TOKENS: 1000.0,
            METRIC_GEN_TOKENS: 200.0,
        }
        deltas.update(overrides)
        return deltas

    def test_computes_server_measured_rates(self) -> None:
        r = metrics_to_rates(self._deltas(), _client_result())
        assert r is not None
        assert r.source == "sglang"
        assert r.prefill_tok_s == 1000 / 0.5
        assert r.gen_tok_s == 200 / (2.5 - 0.5)
        assert r.gen_tok == 200
        assert r.prompt_tok == 1000

    def test_rejects_overlapping_requests(self) -> None:
        deltas = self._deltas(**{METRIC_TTFT_COUNT: 2.0, METRIC_E2E_COUNT: 2.0})
        assert metrics_to_rates(deltas, _client_result()) is None

    def test_rejects_missing_or_bad_deltas(self) -> None:
        assert metrics_to_rates({}, _client_result()) is None
        assert (
            metrics_to_rates(self._deltas(**{METRIC_TTFT_SUM: 0.0}), _client_result())
            is None
        )
        assert metrics_to_rates(self._deltas(**{METRIC_E2E_SUM: 0.1}), _client_result()) is None

    def test_falls_back_to_client_token_counts_when_delta_missing(self) -> None:
        deltas = self._deltas(**{METRIC_PROMPT_TOKENS: 0.0, METRIC_GEN_TOKENS: 0.0})
        r = metrics_to_rates(deltas, _client_result(prompt_tok=77, gen_tok=33))
        assert r is not None
        assert r.prompt_tok == 77
        assert r.gen_tok == 33


class TestRunCompletionMeasured:
    """run_completion_measured behavior."""

    def test_uses_server_metrics_on_clean_deltas(self) -> None:
        before = {
            METRIC_TTFT_SUM: 1.0,
            METRIC_TTFT_COUNT: 5.0,
            METRIC_E2E_SUM: 10.0,
            METRIC_E2E_COUNT: 5.0,
            METRIC_PROMPT_TOKENS: 100.0,
            METRIC_GEN_TOKENS: 100.0,
        }
        after = {k: v + delta for k, v, delta in zip(
            list(before), before.values(),
            [0.5, 1.0, 2.5, 1.0, 1000.0, 200.0],
        )}
        client = _client_result()
        with (
            patch(
                "llm_bench.spec_bench.snapshot_server_metrics",
                side_effect=[before, after],
            ),
            patch("llm_bench.spec_bench.run_completion", return_value=client),
        ):
            r = run_completion_measured("http://host:1", "m", "hi", 8, 10)
        assert r.source == "sglang"
        assert r.prefill_tok_s == 1000 / 0.5
        assert r.gen_tok_s == 200 / 2.0

    def test_falls_back_when_metrics_endpoint_missing(self, capsys) -> None:
        client = _client_result()
        with (
            patch(
                "llm_bench.spec_bench.snapshot_server_metrics",
                side_effect=RuntimeError("no /metrics"),
            ),
            patch("llm_bench.spec_bench.run_completion", return_value=client),
        ):
            r = run_completion_measured("http://host:1", "m", "hi", 8, 10)
        assert r is client
        assert "client timing" in capsys.readouterr().out

    def test_falls_back_after_poll_deadline(self) -> None:
        client = _client_result()
        empty: dict = {}
        with (
            patch(
                "llm_bench.spec_bench.snapshot_server_metrics",
                side_effect=[{}, empty],
            ),
            patch("llm_bench.spec_bench.run_completion", return_value=client),
            patch("llm_bench.spec_bench.METRICS_POLL_DEADLINE_S", 0.0),
            patch("llm_bench.spec_bench.time.sleep"),
        ):
            r = run_completion_measured("http://host:1", "m", "hi", 8, 10)
        assert r is client
        assert r.source == "usage"


class TestApiKey:
    """Tests for --api-key parsing and header injection."""

    def test_parses_api_key(self) -> None:
        args = parse_args(["http://localhost:8000", "--api-key", "sekrit"])
        assert args.api_key == "sekrit"

    def test_api_key_defaults_to_none(self) -> None:
        args = parse_args(["http://localhost:8000"])
        assert args.api_key is None

    def test_handler_adds_bearer_header(self) -> None:
        req = urllib.request.Request("http://localhost:8000/v1/models")
        stamped = BearerAuthHandler("sekrit").http_request(req)
        assert stamped.get_header("Authorization") == "Bearer sekrit"


class TestParseParallel:
    """parse_parallel and --parallel argument behavior."""

    def test_parses_comma_separated_levels(self) -> None:
        assert parse_parallel("2,4,8,16") == [2, 4, 8, 16]

    def test_single_level(self) -> None:
        assert parse_parallel("8") == [8]

    def test_whitespace_is_ignored(self) -> None:
        assert parse_parallel(" 2, 16 ") == [2, 16]

    def test_rejects_unsupported_level(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_parallel("3")

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_parallel("two")

    def test_parallel_defaults_to_none(self) -> None:
        args = parse_args(["--model", "m", "http://h:1"])
        assert args.parallel is None

    def test_custom_parallel(self) -> None:
        args = parse_args(["--model", "m", "http://h:1", "--parallel", "2,4,8,16"])
        assert args.parallel == [2, 4, 8, 16]

    def test_invalid_parallel_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--model", "m", "http://h:1", "--parallel", "3,5"])


class TestRunParallelBatch:
    """run_parallel_batch concurrent execution behavior."""

    def test_all_requests_run_concurrently(self) -> None:
        active = 0
        peak = 0

        def fake_completion(
            base_url: str, model: str, prompt: str, gen_tokens: int, timeout: int
        ) -> RunResult:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(0.05)
            active -= 1
            return RunResult(0.0, 100.0, 10, 20, 0.5, "usage")

        with patch("llm_bench.spec_bench.run_completion", side_effect=fake_completion):
            results = run_parallel_batch(
                "http://h:1", "m", ["a", "b", "c", "d"], 16, 5
            )
        assert peak == 4
        assert len(results) == 4
        assert all(r.gen_tok == 20 for r in results)

    def test_results_keep_prompt_order(self) -> None:
        def fake_completion(
            base_url: str, model: str, prompt: str, gen_tokens: int, timeout: int
        ) -> RunResult:
            return RunResult(0.0, 1.0, 0, int(prompt), 0.1, "usage")

        with patch("llm_bench.spec_bench.run_completion", side_effect=fake_completion):
            results = run_parallel_batch("http://h:1", "m", ["1", "2", "3"], 16, 5)
        assert [r.gen_tok for r in results] == [1, 2, 3]

    def test_failure_propagates_after_all_complete(self) -> None:
        with pytest.raises(RuntimeError), patch(
            "llm_bench.spec_bench.run_completion", side_effect=RuntimeError("boom")
        ):
            run_parallel_batch("http://h:1", "m", ["a", "b"], 16, 5)


class TestBenchLengthsParallel:
    """bench_lengths concurrency-level reporting."""

    def _result(self) -> RunResult:
        return RunResult(40.0, 100.0, 1000, 500, 0.5, "usage")

    def test_parallel_run_prints_aggregate_row(self, capsys) -> None:
        def fake(*args) -> RunResult:
            return self._result()

        with patch("llm_bench.spec_bench.run_completion", side_effect=fake):
            bench_lengths("http://h:1", "m", [2], ["code"], 16, 5, parallel=[2])
        out = capsys.readouterr().out
        assert "par=2" in out
        assert f"{2000:>13}" in out  # total prompt tokens
        assert f"{1000:>10}" in out  # total generated tokens
        assert "4000.00" in out  # aggregate prefill: 2000 / 0.5 slowest TTFT

    def test_parallel_run_flags_mixed_token_sources(self, capsys) -> None:
        results = [
            RunResult(40.0, 100.0, 1000, 500, 0.5, "usage"),
            RunResult(40.0, 100.0, 1000, 500, 0.5, "estimated"),
        ]
        with patch(
            "llm_bench.spec_bench.run_completion", side_effect=results
        ), patch(
            "llm_bench.spec_bench.build_prompt", side_effect=lambda *a: "p"
        ):
            bench_lengths("http://h:1", "m", [2], ["code"], 16, 5, parallel=[2])
        out = capsys.readouterr().out
        assert "*" in out  # estimated rows are asterisked

    def test_parallel_failure_prints_failed_row(self, capsys) -> None:
        with patch(
            "llm_bench.spec_bench.run_completion", side_effect=RuntimeError("boom")
        ):
            bench_lengths("http://h:1", "m", [2], ["code"], 16, 5, parallel=[2])
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "par=" not in out

    def test_single_run_keeps_measured_path(self, capsys) -> None:
        with patch(
            "llm_bench.spec_bench.run_completion_measured", return_value=self._result()
        ) as measured, patch(
            "llm_bench.spec_bench.run_completion"
        ) as client:
            bench_lengths("http://h:1", "m", [2], ["code"], 16, 5)
        out = capsys.readouterr().out
        measured.assert_called_once()
        client.assert_not_called()
        assert "par=" not in out
