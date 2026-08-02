"""Tests for search.rewriter.QueryRewriter (Phase 3, DD-14, FD-32...FD-37)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from llm.base import GenerationResult
from search.rewriter import QueryRewriter, _is_vietnamese, rewrite


def _client_returning(text: str) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = GenerationResult(text=text)
    return client


class TestQueryRewriterBasics:
    def test_empty_query_returns_empty_list(self):
        assert QueryRewriter().rewrite("") == []

    def test_whitespace_query_returns_empty_list(self):
        assert QueryRewriter().rewrite("   ") == []

    def test_no_llm_client_returns_original_only(self, monkeypatch):
        monkeypatch.setattr(QueryRewriter, "_get_client", lambda self: None)
        result = QueryRewriter().rewrite("hello world")
        assert result == ["hello world"]

    def test_query_is_stripped(self, monkeypatch):
        monkeypatch.setattr(QueryRewriter, "_get_client", lambda self: None)
        result = QueryRewriter().rewrite("  hello world  ")
        assert result == ["hello world"]


class TestQueryRewriterWithClient:
    def test_rewrite_adds_llm_candidates(self):
        rw = QueryRewriter()
        rw._client = _client_returning(json.dumps({"rewrites": ["expanded query", "another variant"]}))
        result = rw.rewrite("original query", max_candidates=3)
        assert result[0] == "original query"
        assert "expanded query" in result
        assert "another variant" in result
        assert len(result) == 3

    def test_rewrite_dedupes_case_insensitive(self):
        rw = QueryRewriter()
        rw._client = _client_returning(json.dumps({"rewrites": ["Original Query", "new one"]}))
        result = rw.rewrite("original query")
        assert result.count("original query") == 1
        assert "new one" in result

    def test_rewrite_limits_to_max_candidates_from_llm(self):
        rw = QueryRewriter()
        rw._client = _client_returning(json.dumps({"rewrites": ["a", "b", "c", "d", "e"]}))
        result = rw.rewrite("q", max_candidates=2)
        assert result[0] == "q"
        # _rewrite_with_llm caps at max_candidates before dedup/append.
        assert len(result) <= 3

    def test_malformed_json_falls_back_to_original(self):
        rw = QueryRewriter()
        rw._client = _client_returning("not json at all")
        result = rw.rewrite("original query")
        assert result == ["original query"]

    def test_non_list_rewrites_field_falls_back(self):
        rw = QueryRewriter()
        rw._client = _client_returning(json.dumps({"rewrites": "not a list"}))
        result = rw.rewrite("original query")
        assert result == ["original query"]

    def test_non_string_items_are_dropped(self):
        # max_candidates caps the slice *before* type-filtering, so request
        # enough candidates that both valid strings survive the slice.
        rw = QueryRewriter()
        rw._client = _client_returning(json.dumps({"rewrites": ["valid", 42, None, "also valid"]}))
        result = rw.rewrite("q", max_candidates=4)
        assert "valid" in result
        assert "also valid" in result
        assert len(result) == 3  # original + 2 valid strings

    def test_llm_call_exception_falls_back_to_original(self):
        rw = QueryRewriter()
        client = MagicMock()
        client.generate.side_effect = RuntimeError("network error")
        rw._client = client
        result = rw.rewrite("original query")
        assert result == ["original query"]

    def test_empty_generation_result_falls_back(self):
        rw = QueryRewriter()
        rw._client = _client_returning("")
        result = rw.rewrite("original query")
        assert result == ["original query"]

    def test_uses_module_level_language_detector_not_missing_method(self):
        """Regression test: _rewrite_with_llm previously called the
        nonexistent self._is_vietnamese(...), which raised AttributeError on
        every real LLM call and was silently swallowed by rewrite()'s
        try/except, so LLM rewriting never actually ran. This asserts the
        LLM is actually invoked (i.e. no AttributeError short-circuits it).
        """
        rw = QueryRewriter()
        client = _client_returning(json.dumps({"rewrites": ["ok"]}))
        rw._client = client
        result = rw.rewrite("plain english query")
        client.generate.assert_called_once()
        assert "ok" in result


class TestLanguageDetection:
    def test_english_not_detected_as_vietnamese(self):
        assert _is_vietnamese("What is the meaning of life?") is False

    def test_empty_string_not_vietnamese(self):
        assert _is_vietnamese("") is False

    def test_vietnamese_diacritics_detected(self):
        assert _is_vietnamese("Xin chào, bạn có khỏe không?") is True


class TestModuleLevelRewrite:
    def test_rewrite_function_no_client_returns_original(self, monkeypatch):
        monkeypatch.setattr(QueryRewriter, "_get_client", lambda self: None)
        result = rewrite("plain query")
        assert result == ["plain query"]
