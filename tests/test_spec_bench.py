"""Tests for llm_bench.spec_bench."""

import pytest

from llm_bench.spec_bench import build_prompt, parse_args, print_summary


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
        args = parse_args(["--model", "m", "--port", "8080"])
        assert args.model == "m"
        assert args.port == 8080
        assert args.ip == "127.0.0.1"
        assert args.gen_tokens == 256
        assert args.runs == 3
        assert not args.keep_warmup

    def test_missing_model_exits(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--port", "8080"])

    def test_custom_lengths(self) -> None:
        args = parse_args(["--model", "m", "--port", "1", "--lengths", "4", "12"])
        assert args.lengths == [4, 12]


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
