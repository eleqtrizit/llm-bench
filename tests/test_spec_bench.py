"""Tests for llm_bench.spec_bench."""

import argparse
import io
import json
from unittest.mock import patch

import pytest

from llm_bench.spec_bench import (
    WARMUP_GEN_TOKENS,
    WARMUP_PROMPT,
    build_prompt,
    choose_model,
    format_header,
    format_row,
    menu_select_ctx,
    parse_args,
    parse_ctx,
    query_models,
    run_completion,
    warm_up,
)


class TestBuildPrompt:
    """build_prompt behavior."""

    def test_zero_ctx_returns_short_prompt(self) -> None:
        assert build_prompt(0) == "Count from 1 to 20."

    def test_negative_ctx_returns_short_prompt(self) -> None:
        assert build_prompt(-5) == "Count from 1 to 20."

    def test_larger_ctx_gives_longer_prompt(self) -> None:
        short = build_prompt(1)
        long = build_prompt(8)
        assert len(long) > len(short)

    def test_kilotokens_scale_the_prompt(self) -> None:
        one_k = len(build_prompt(1))
        two_k = len(build_prompt(2))
        assert abs(two_k / one_k - 2) < 0.1

    def test_prompt_mentions_task(self) -> None:
        assert "count from 1 to 20" in build_prompt(1)


class TestParseArgs:
    """parse_args behavior."""

    def test_required_args(self) -> None:
        args = parse_args(["--model", "m", "--host", "10.0.0.5", "--port", "8080"])
        assert args.model == "m"
        assert args.host == "10.0.0.5"
        assert args.port == 8080
        assert args.gen_tokens == 256

    def test_missing_host_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--model", "m", "--port", "8080"])

    def test_missing_model_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--port", "8080"])

    def test_custom_ctx(self) -> None:
        args = parse_args(["--model", "m", "--host", "h", "--port", "1", "--ctx", "0,8,16"])
        assert args.ctx == [0, 8, 16]

    def test_invalid_ctx_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--model", "m", "--host", "h", "--port", "1", "--ctx", "0,eight"])

    def test_negative_ctx_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--model", "m", "--host", "h", "--port", "1", "--ctx", "-4"])


class TestMenuSelectCtx:
    """menu_select_ctx behavior."""

    def test_requires_tty(self) -> None:
        with patch("llm_bench.spec_bench.sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            with pytest.raises(ValueError, match="no terminal"):
                menu_select_ctx([0, 8])


class TestParseCtx:
    """parse_ctx behavior."""

    def test_parses_comma_separated_values(self) -> None:
        assert parse_ctx("0,8,16") == [0, 8, 16]

    def test_whitespace_is_ignored(self) -> None:
        assert parse_ctx("0, 8 ,16") == [0, 8, 16]

    def test_rejects_non_numeric_values(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_ctx("0,eight")

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_ctx("0,-4")


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
        row = format_row(8, 155, 81, 900.0, 152.65)
        cols = row.split()
        assert cols == ["8k", "155", "81", "900.00", "152.65"]

    def test_row_shows_dash_for_unknown_prefill(self) -> None:
        row = format_row(0, 20, 93, 0.0, 191.65)
        assert row.split()[3] == "-"

    def test_tok_s_header_and_value_share_right_edge(self) -> None:
        header = format_header()
        row = format_row(0, 20, 93, 900.0, 191.65)
        assert header.split() == ["ctx", "prompt_tok", "gen_tok", "prefill", "tok/s", "tok/s"]
        header_end = header.rindex("tok/s") + len("tok/s")
        row_end = row.index("191.65") + len("191.65")
        assert header_end == row_end


class TestRunCompletion:
    """run_completion streaming behavior."""

    def _sse_body(self, chunks: list) -> io.BytesIO:
        lines = [b'data: ' + json.dumps(c).encode() for c in chunks]
        lines.append(b"data: [DONE]")
        return io.BytesIO(b"\n".join(lines) + b"\n")

    def test_measures_prefill_and_generation(self) -> None:
        chunks = [
            {"choices": [{"delta": {"content": "He"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
            {"usage": {"prompt_tokens": 600, "completion_tokens": 2}},
        ]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            gen_tok_s, prefill_tok_s, prompt, gen, ttft = run_completion(
                "http://host:1", "m", "hi", 8, 10
            )
        assert prompt == 600
        assert gen == 2
        assert ttft > 0
        assert prefill_tok_s > 0
        assert gen_tok_s > 0

    def test_no_usage_means_unknown_prefill(self) -> None:
        chunks = [{"choices": [{"delta": {"content": "Hey"}}]}]
        resp = self._sse_body(chunks)
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            _g, prefill_tok_s, prompt, _n, _t = run_completion(
                "http://host:1", "m", "hi", 8, 10
            )
        assert prompt == 0
        assert prefill_tok_s == 0.0

    def test_no_tokens_raises(self) -> None:
        resp = self._sse_body([])
        with patch("llm_bench.spec_bench.urllib.request.urlopen", return_value=resp):
            with pytest.raises(RuntimeError, match="no tokens"):
                run_completion("http://host:1", "m", "hi", 8, 10)


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
