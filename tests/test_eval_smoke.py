"""评测管线的端到端 smoke 测试。

断言的是**管线正确性**（能跑、指标在合法区间、三种 mode 都可比、逐条明细可追溯），
不是检索质量——hash embedder 没有语义泛化能力，质量结论必须用 fastembed + 真实语料 + 人工标注集。
dataset.smoke.json 的 _known_issue 里记录了一个刻意保留的漏召回（q005）。
"""
import json
from pathlib import Path

import pytest

from pkr.metrics import evaluate
from pkr.retrieve import MODES, build_index, search

DATASET = Path(__file__).resolve().parent.parent / "evaluation" / "dataset.smoke.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(DATASET.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def index(sample_settings):
    return build_index(sample_settings)


def _run(index, mode, k):
    def run(query, filters):
        return [h.chunk.id for h in search(index, query, top_k=k, mode=mode,
                                           wikis=filters.get("wikis"),
                                           layers=filters.get("layers"))]
    return run


def test_dataset_references_real_chunks(payload, index):
    """标注里的 id 必须真实存在，否则评测会静默算出 0 分而无人察觉。"""
    valid = set(index.chunks)
    for item in payload["items"]:
        for cid in list(item.get("relevant", [])) + list(item.get("partial", [])):
            assert cid in valid, f"{item['id']} 标注了不存在的 chunk: {cid}"


def test_all_modes_produce_valid_metrics(payload, index):
    k = payload.get("k", 5)
    for mode in MODES:
        res = evaluate(payload["items"], _run(index, mode, k), k=k)
        assert res["n_queries"] == len(payload["items"])
        assert res["skipped_no_relevant"] == 0
        for key in ("recall@k", "precision@k", "mrr", "ndcg@k"):
            v = res[key]
            assert v == v, f"{mode}/{key} 为 NaN"
            assert 0.0 <= v <= 1.0, f"{mode}/{key}={v} 超出 [0,1]"


def test_hybrid_candidates_come_from_both_routes(payload, index):
    """结构性不变量：hybrid 的结果必须来自两路候选的并集（RRF 的定义所保证）。

    刻意**不**断言"hybrid 质量不低于单路"——那是质量结论，
    在合成语料 + hash embedder 上没有意义（实测 hybrid 的 Recall@5 就低于 sparse）。
    质量结论必须由真实语料 + 真实语义模型 + 人工标注集给出，README 写明了这条边界。
    """
    k = payload.get("k", 5)
    for item in payload["items"]:
        q = item["query"]
        dense = {h.chunk.id for h in search(index, q, top_k=k, recall_k=50, mode="dense")}
        sparse = {h.chunk.id for h in search(index, q, top_k=k, recall_k=50, mode="sparse")}
        hybrid = {h.chunk.id for h in search(index, q, top_k=k, recall_k=50, mode="hybrid")}
        assert hybrid <= (dense | sparse), f"{item['id']}: hybrid 出现了两路都没有的候选"


def test_per_query_detail_is_traceable(payload, index):
    k = payload.get("k", 5)
    res = evaluate(payload["items"], _run(index, "hybrid", k), k=k)
    ids = [pq["id"] for pq in res["per_query"]]
    assert ids == [i["id"] for i in payload["items"]], "逐条明细必须与评测集一一对应且有序"


def test_known_limitation_is_disclosed_in_dataset(payload):
    """刻意保留的漏召回必须在数据文件里写明，不能只存在于口头。"""
    assert payload.get("_known_issue"), "dataset.smoke.json 缺少 _known_issue 说明"
    assert payload.get("_caveat"), "dataset.smoke.json 缺少 _caveat 说明"