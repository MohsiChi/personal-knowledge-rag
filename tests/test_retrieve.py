"""检索编排的端到端测试（HashEmbedder + sample_wiki，全离线确定性）。

断言基于 2026-09-06 的真实运行输出，不是凭猜：
hash 档下 q005 会漏召回 sparse-vs-dense（词法不匹配），这是刻意保留的已知局限，
因此本文件不断言 hash 档的检索质量，只断言管线行为正确。
"""
import pytest

from pkr.models import RetrievalTrace
from pkr.retrieve import MODES, build_index, search


@pytest.fixture(scope="module")
def index(sample_settings):
    return build_index(sample_settings)


def test_sample_corpus_is_indexed_and_clean(index):
    assert len(index) == 22
    assert all(c.status != "stub" for c in index.chunks.values()), "stub 页不应进索引"
    assert all(c.knowledge_layer != "index" for c in index.chunks.values()), "index 层不应进索引"
    assert {c.knowledge_layer for c in index.chunks.values()} == {"concept", "connection", "question"}


def test_all_modes_run_and_return_scored_chunks(index):
    for mode in MODES:
        hits = search(index, "BM25 的词频饱和", top_k=3, mode=mode)
        assert hits, f"mode={mode} 没有返回结果"
        assert all(h.score == h.score for h in hits), "分数不应为 NaN"
        assert [h.chunk.id for h in hits] == list(dict.fromkeys(h.chunk.id for h in hits)), "结果不得重复"


def test_hybrid_ranks_bm25_page_first_for_bm25_query(index):
    hits = search(index, "BM25 的词频饱和是什么意思", top_k=3, mode="hybrid")
    assert hits[0].chunk.source_path.endswith("bm25.md")


def test_hybrid_ranks_rrf_page_first_for_rrf_query(index):
    hits = search(index, "为什么 RRF 用排名而不是分数", top_k=1, mode="hybrid")
    assert hits[0].chunk.id == "sample::wiki/concepts/reciprocal-rank-fusion.md::1"


def test_ranks_are_recorded_for_both_routes(index):
    hits = search(index, "BM25 的词频饱和", top_k=3, mode="hybrid")
    assert all(h.dense_rank is not None and h.sparse_rank is not None for h in hits), \
        "hybrid 模式下两路名次都应记录，否则无法解释融合结果"


def test_filter_by_layer(index):
    hits = search(index, "ANN 索引", top_k=10, layers=["question"])
    assert hits and all(h.chunk.knowledge_layer == "question" for h in hits)


def test_filter_by_wiki_excludes_everything_when_unknown(index):
    assert search(index, "BM25", top_k=5, wikis=["不存在的库"]) == []


def test_filter_is_pre_not_post(index):
    """过滤必须前置：后置过滤会让 top_k 被筛空、指标虚低。"""
    hits = search(index, "ANN 索引", top_k=3, layers=["question"])
    assert len(hits) == 3, "前置过滤应能填满 top_k（question 层有 4 个 chunk）"


def test_trace_is_populated(index):
    tr = RetrievalTrace()
    search(index, "BM25", top_k=3, mode="hybrid", trace=tr)
    assert tr.query == "BM25"
    assert tr.dense and tr.sparse and tr.fused
    assert tr.filters["eligible"] == 22


def test_unknown_mode_raises(index):
    with pytest.raises(ValueError):
        search(index, "x", mode="vector")


def test_empty_query_returns_empty(index):
    assert search(index, "", top_k=5) == []
    assert search(index, "   ", top_k=5) == []