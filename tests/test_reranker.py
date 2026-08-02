"""Tests for search.reranker.Reranker (Phase 3, DD-17, FD-75...FD-77)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from llm.base import GenerationResult
from models import SearchResult
from search.reranker import Reranker, rerank


def _client_returning(text: str) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = GenerationResult(text=text)
    return client


def _results(n: int) -> list[SearchResult]:
    return [
        SearchResult(text=f"doc {i}", metadata={"source": f"f{i}.txt"}, distance=0.1 * i, collection="c")
        for i in range(n)
    ]


class TestRerankerBasics:
    def test_empty_query_returns_results_unchanged(self):
        results = _results(3)
        assert Reranker().rerank("", results) == list(results)

    def test_empty_results_returns_empty(self):
        assert Reranker().rerank("query", []) == []

    def test_fewer_results_than_top_k_skips_llm_entirely(self):
        rw = Reranker()
        client = MagicMock()
        rw._client = client
        results = _results(2)
        out = rw.rerank("query", results, top_k=10)
        assert len(out) == 2
        client.generate.assert_not_called()

    def test_no_llm_client_returns_original_order_truncated(self, monkeypatch):
        monkeypatch.setattr(Reranker, "_get_client", lambda self: None)
        results = _results(5)
        out = Reranker().rerank("query", results, top_k=2)
        assert len(out) == 2
        assert [r.text for r in out] == ["doc 0", "doc 1"]


class TestRerankerWithClient:
    def test_rerank_sorts_by_descending_score(self):
        rr = Reranker()
        # 3 results, scores favor the last one most.
        rr._client = _client_returning(json.dumps({"scores": [1, 5, 9]}))
        results = _results(3)
        out = rr.rerank("query", results, top_k=2)
        assert len(out) == 2
        assert out[0].text == "doc 2"  # score 9
        assert out[1].text == "doc 1"  # score 5

    def test_llm_scoring_failure_falls_back_to_original_order(self):
        rr = Reranker()
        rr._client = _client_returning("not valid json")
        results = _results(3)
        out = rr.rerank("query", results, top_k=2)
        assert [r.text for r in out] == ["doc 0", "doc 1"]

    def test_mismatched_score_count_falls_back(self):
        rr = Reranker()
        rr._client = _client_returning(json.dumps({"scores": [1, 2]}))  # only 2 scores for 3 results
        results = _results(3)
        out = rr.rerank("query", results, top_k=2)
        assert [r.text for r in out] == ["doc 0", "doc 1"]

    def test_non_numeric_score_falls_back(self):
        rr = Reranker()
        rr._client = _client_returning(json.dumps({"scores": [1, "bad", 3]}))
        results = _results(3)
        out = rr.rerank("query", results, top_k=2)
        assert [r.text for r in out] == ["doc 0", "doc 1"]

    def test_llm_exception_falls_back(self):
        rr = Reranker()
        client = MagicMock()
        client.generate.side_effect = RuntimeError("boom")
        rr._client = client
        results = _results(3)
        out = rr.rerank("query", results, top_k=2)
        assert [r.text for r in out] == ["doc 0", "doc 1"]

    def test_batches_large_result_sets(self):
        """batch_size is 10 — 15 results should trigger 2 calls to generate."""
        rr = Reranker()
        scores_batch_1 = list(range(10))
        scores_batch_2 = list(range(5))
        client = MagicMock()
        client.generate.side_effect = [
            GenerationResult(text=json.dumps({"scores": scores_batch_1})),
            GenerationResult(text=json.dumps({"scores": scores_batch_2})),
        ]
        rr._client = client
        results = _results(15)
        out = rr.rerank("query", results, top_k=3)
        assert client.generate.call_count == 2
        assert len(out) == 3


class TestResultConversion:
    def test_result_to_dict_and_back_roundtrips(self):
        r = SearchResult(text="hello", metadata={"source": "a.txt"}, distance=0.5, collection="c1")
        d = Reranker._result_to_dict(r)
        assert d["text"] == "hello"
        assert d["distance"] == 0.5
        back = Reranker._dict_to_result(d)
        assert isinstance(back, SearchResult)
        assert back.text == "hello"

    def test_plain_dict_passthrough(self):
        d = {"text": "raw", "foo": "bar"}
        assert Reranker._result_to_dict(d) is d


class TestModuleLevelRerank:
    def test_rerank_function_no_client_returns_truncated_original(self, monkeypatch):
        monkeypatch.setattr(Reranker, "_get_client", lambda self: None)
        results = _results(5)
        out = rerank(results, "query", top_k=3)
        assert len(out) == 3
