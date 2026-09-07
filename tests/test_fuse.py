"""RRF 测试。融合逻辑必须能手算验证，否则评测数字无从解释。"""
import pytest

from pkr.fuse import rrf


def test_rrf_matches_hand_computation():
    """rankings = [[a,b],[b,c]], k=60
    a = 1/61 = 0.0163934
    b = 1/62 + 1/61 = 0.0161290 + 0.0163934 = 0.0325224
    c = 1/62 = 0.0161290
    """
    out = dict(rrf([["a", "b"], ["b", "c"]], k=60))
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["a"] == pytest.approx(1 / 61)
    assert out["c"] == pytest.approx(1 / 62)
    order = [d for d, _ in rrf([["a", "b"], ["b", "c"]], k=60)]
    assert order == ["b", "a", "c"], "被两路同时召回的文档必须排在最前"


def test_rrf_doc_in_both_rankings_beats_single_rank_top():
    out = rrf([["x", "y"], ["y", "z"]], k=60)
    assert out[0][0] == "y"


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError):
        rrf([["a"]], k=0)
    with pytest.raises(ValueError):
        rrf([["a"]], k=-1)


def test_rrf_tie_break_is_deterministic():
    """两路各自的第一名分数相同，必须按 id 字典序决胜，不能依赖 dict 迭代顺序。"""
    a = rrf([["b"], ["a"]], k=60)
    b = rrf([["b"], ["a"]], k=60)
    assert a == b
    assert [d for d, _ in a] == ["a", "b"], "同分时按 id 字典序"


def test_rrf_empty_and_single():
    assert rrf([], k=60) == []
    assert rrf([[]], k=60) == []
    assert [d for d, _ in rrf([["only"]], k=60)] == ["only"]


def test_larger_k_flattens_head_difference():
    """k 越大头部名次之间的分差越小（越"民主"）——这是 k 的实际语义。"""
    def gap(k):
        s = dict(rrf([["a", "b", "c"]], k=k))
        return s["a"] - s["c"]
    assert gap(1) > gap(60) > gap(1000)