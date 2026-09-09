"""Tests for llm_bench.spec_bench."""

import io
import json
from unittest.mock import patch

import pytest

from llm_bench.spec_bench import build_prompt, choose_model, parse_args, print_summary, query_models


class TestBuildPrompt:
    """build_prompt behavior."""

    def test_zero_ctx_returns_short_prompt(self) -> None:
        assert build_prompt(0) == "Count from 1 to 20."

    def test_negative_ctx_returns_short_prompt(self) -> None:
        assert build_prompt(-5) == "Count from 1 to 20."

    def test_larger_ctx_gives_longer_prompt(self) -> None:
        short = build_prompt(8)
        long = build_prompt(128)
        assert len(long) > len(short)

    def test_prompt_mentions_task(self) -> None:
        assert "count from 1 to 20" in build_prompt(64)


class TestParseArgs:
    """parse_args behavior."""

    def test_required_args(self) -> None:
        args = parse_args(["--model", "m", "--host", "10.0.0.5", "--port", "8080"])
        assert args.model == "m"
        assert args.host == "10.0.0.5"
        assert args.port == 8080
        assert args.gen_tokens == 256
        assert args.runs == 3
        assert not args.keep_warmup

    def test_missing_host_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--model", "m", "--port", "8080"])

    def test_missing_model_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--port", "8080"])

    def test_custom_lengths(self) -> None:
        args = parse_args(
            ["--model", "m", "--host", "10.0.0.5", "--port", "1", "--lengths", "4", "12"]
        )
        assert args.lengths == [4, 12]


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


class TestPrintSummary:
    """print_summary behavior."""

    def test_skips_when_single_result(self, capsys: pytest.CaptureFixture) -> None:
        print_summary({0: [10.0]})
        assert capsys.readouterr().out == ""

    def test_prints_bars_for_multiple_results(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        results = {0: [10.0], 8: [20.0]}
        print_summary(results)
        out = capsys.readouterr().out
        assert "Summary (mean tok/s):" in out
        assert "ctx=0" in out and "ctx=8" in out
        assert "#####" in out  # 20.0 / 2 = 10 hashes
