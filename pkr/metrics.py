"""检索评测指标：Recall@k / Precision@k / MRR / NDCG@k。

口径（必须与标注工具一致，否则数字无法解释）：
- 分级相关性 relevant=2、partial=1、未标注=0，与标注的 y/p/n 一一对应；
- Recall / Precision / MRR 的"命中"= gain>0，即 partial 也算命中；
- NDCG@k 用指数增益 (2^gain - 1)/log2(rank + 1)，能区分"相关"与"部分相关"；
- 没有任何相关文档的 query 不计入平均——否则会用 0 拉低均值，掩盖真实表现。

已知偏差（README 如实披露）：评测集用候选池法（pooling）标注，
所有方法都没召回的页面默认视为不相关，这会系统性高估各方法的绝对分数，
但**不影响方法之间的相对比较**，因为所有方法共用同一个池。
"""
from __future__ import annotations

import math
from typing import Callable, Iterable


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def _dcg(gains: Iterable[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: list[str], gains: dict[str, int], k: int) -> float:
    got = [int(gains.get(d, 0)) for d in retrieved[:k]]
    ideal = sorted((int(g) for g in gains.values() if g > 0), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(got) / idcg if idcg > 0 else float("nan")


def evaluate(dataset: list[dict], run: Callable[[str, dict], list[str]], k: int = 5) -> dict:
    """dataset 每条：{id, query, relevant:[...], partial:[...], filters:{...}}
    run(query, filters) -> 检索到的 chunk_id 列表（按分数降序）。
    """
    acc = {"recall": [], "precision": [], "mrr": [], "ndcg": []}
    per_query, skipped = [], 0
    for item in dataset:
        relevant = set(item.get("relevant") or [])
        partial = set(item.get("partial") or []) - relevant
        hit_set = relevant | partial
        gains = {d: 2 for d in relevant}
        gains.update({d: 1 for d in partial})
        if not hit_set:
            skipped += 1
            continue
        got = list(run(item["query"], item.get("filters") or {}))
        vals = {"recall": recall_at_k(got, hit_set, k),
                "precision": precision_at_k(got, hit_set, k),
                "mrr": mrr(got, hit_set),
                "ndcg": ndcg_at_k(got, gains, k)}
        for key, v in vals.items():
            if v == v:  # 过滤 NaN
                acc[key].append(v)
        per_query.append({"id": item.get("id"), "query": item["query"],
                          **{key: (None if v != v else round(v, 4)) for key, v in vals.items()}})

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {"k": k, "n_queries": len(per_query), "skipped_no_relevant": skipped,
            "recall@k": mean(acc["recall"]), "precision@k": mean(acc["precision"]),
            "mrr": mean(acc["mrr"]), "ndcg@k": mean(acc["ndcg"]),
            "per_query": per_query}