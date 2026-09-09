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
    print_summary,
    query_models,
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

    def test_row_has_four_columns(self) -> None:
        row = format_row(8, 155, 81, 152.65)
        cols = row.split()
        assert cols == ["8k", "155", "81", "152.65"]

    def test_tok_s_header_and_value_share_right_edge(self) -> None:
        header = format_header()
        row = format_row(0, 20, 93, 191.65)
        assert header.split() == ["ctx", "prompt_tok", "gen_tok", "tok/s"]
        header_end = header.index("tok/s") + len("tok/s")
        row_end = row.index("191.65") + len("191.65")
        assert header_end == row_end


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


class TestPrintSummary:
    """print_summary behavior."""

    def test_skips_when_single_result(self, capsys: pytest.CaptureFixture) -> None:
        print_summary({0: 10.0})
        assert capsys.readouterr().out == ""

    def test_prints_bars_for_multiple_results(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        results = {0: 10.0, 8: 20.0}
        print_summary(results)
        out = capsys.readouterr().out
        assert "Summary (tok/s):" in out
        assert "ctx=0" in out and "ctx=8" in out
        assert "#####" in out  # 20.0 / 2 = 10 hashes
