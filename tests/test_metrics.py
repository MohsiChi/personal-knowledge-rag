"""指标测试：全部与手算值对照，防止评测框架自己算错。"""
import math

import pytest

from pkr.metrics import evaluate, mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_recall_and_precision():
    got = ["a", "b", "c"]
    rel = {"a", "c", "z"}
    assert recall_at_k(got, rel, 3) == pytest.approx(2 / 3)
    assert precision_at_k(got, rel, 3) == pytest.approx(2 / 3)
    assert precision_at_k(got, {"a"}, 5) == pytest.approx(1 / 5), "Precision@k 的分母是 k"


def test_recall_without_relevant_is_nan():
    assert math.isnan(recall_at_k(["a"], set(), 3))


def test_mrr():
    assert mrr(["x", "a", "b"], {"a"}) == pytest.approx(0.5)
    assert mrr(["a"], {"a"}) == pytest.approx(1.0)
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_matches_hand_computation():
    """retrieved=[a,b], gains={a:2,b:1,c:2}, k=2
    DCG  = (2^2-1)/log2(2) + (2^1-1)/log2(3) = 3 + 0.630930 = 3.630930
    IDCG = 3/log2(2) + 3/log2(3)             = 3 + 1.892789 = 4.892789
    NDCG = 0.7420981
    """
    val = ndcg_at_k(["a", "b"], {"a": 2, "b": 1, "c": 2}, 2)
    dcg = 3 / math.log2(2) + 1 / math.log2(3)
    idcg = 3 / math.log2(2) + 3 / math.log2(3)
    assert val == pytest.approx(dcg / idcg, abs=1e-9)
    assert val == pytest.approx(0.7420981, abs=1e-6)


def test_ndcg_perfect_order_is_one():
    assert ndcg_at_k(["c", "a"], {"a": 2, "c": 2}, 2) == pytest.approx(1.0)


def test_ndcg_without_gains_is_nan():
    assert math.isnan(ndcg_at_k(["a"], {}, 3))


def test_evaluate_skips_queries_without_relevant():
    ds = [
        {"id": "q1", "query": "有答案的", "relevant": ["a"]},
        {"id": "q2", "query": "没标注的", "relevant": []},
    ]
    res = evaluate(ds, lambda q, f: ["a", "b"], k=2)
    assert res["n_queries"] == 1
    assert res["skipped_no_relevant"] == 1
    assert res["recall@k"] == pytest.approx(1.0)


def test_evaluate_counts_partial_as_hit_but_lower_gain():
    """partial 计入 Recall/Precision/MRR 的命中，但在 NDCG 里增益低于 relevant。"""
    ds = [{"id": "q1", "query": "x", "relevant": ["a"], "partial": ["b"]}]
    res = evaluate(ds, lambda q, f: ["b", "a"], k=2)
    assert res["recall@k"] == pytest.approx(1.0), "partial 也算命中"
    assert res["mrr"] == pytest.approx(1.0), "第一位的 partial 就是首个命中"
    full = evaluate(ds, lambda q, f: ["a", "b"], k=2)
    assert full["ndcg@k"] > res["ndcg@k"], "相关文档排在部分相关之前，NDCG 应更高"