"""Reciprocal Rank Fusion —— 自己实现，不依赖任何检索框架。

    score(d) = Σ_over_rankers  1 / (k + rank_r(d))

只用**排名**、不用原始分数，因此天然免疫两路分数量纲不一致的问题
（余弦相似度在 [-1,1]，BM25 分数无上界且随语料统计变化）——这正是 RRF 的价值。
k 是平滑常数，默认 60（原论文取值）：k 越大头部结果之间的权重差被压得越平（越"民主"），
k 越小越偏向单路的头部结果。

并列时按 doc_id 字典序决胜，保证确定性（同样输入必得同样输出，评测才可复现）。
"""
from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: 每个元素是一路的 doc_id 排序列表（最好在前）。返回 [(doc_id, score)]。"""
    if k <= 0:
        raise ValueError(f"RRF 的 k 必须为正数，收到 {k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))