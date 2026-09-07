"""索引层：向量索引（暴力余弦）与 BM25 稀疏索引。

VectorIndex 是抽象接口，v0 只提供 NumpyVectorIndex。
在个人知识库规模（几百到几千 chunk）上暴力检索是毫秒级，ANN 索引（FAISS / HNSW）
属于过度设计；等语料到十万级再换实现，接口不动。"知道什么时候不需要 ANN"
和"会用 ANN"同样是工程判断，这条取舍写进 README。

v0 刻意不做索引持久化：语料小、重建快（hash 档瞬时、fastembed 档数秒），
而持久化会引入**静默陈旧**问题——语料改了但索引没更新，检索结果与语料不一致且无从察觉。
宁可每次重建，也不要静默陈旧。持久化与增量更新列在 roadmap。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from rank_bm25 import BM25Okapi

from pkr.models import Chunk
from pkr.text import tokenize


def _top_k(scores: np.ndarray, ids: list[str], k: int) -> list[tuple[str, float]]:
    """取 top-k，分数降序、并列按 id 字典序（确定性）。"""
    if len(ids) == 0 or k <= 0:
        return []
    k = min(k, len(ids))
    idx = np.argpartition(-scores, k - 1)[:k]
    ranked = sorted(((ids[i], float(scores[i])) for i in idx), key=lambda kv: (-kv[1], kv[0]))
    return ranked


class VectorIndex(ABC):
    """向量索引接口。换 ANN 实现时只改这一层。"""

    @abstractmethod
    def build(self, ids: list[str], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int,
               allowed: set[str] | None = None) -> list[tuple[str, float]]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class NumpyVectorIndex(VectorIndex):
    """暴力余弦检索。要求向量已 L2 归一化（Embedder 契约保证），故点积即余弦。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._mat = np.zeros((0, 0), dtype=np.float32)

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        if len(ids) != vecs.shape[0]:
            raise ValueError(f"ids({len(ids)}) 与 vectors({vecs.shape[0]}) 数量不一致")
        if ids and vecs.ndim != 2:
            raise ValueError(f"vectors 必须是二维矩阵，收到 ndim={vecs.ndim}")
        self._ids = list(ids)
        self._mat = vecs

    def search(self, vector, k, allowed=None):
        if not self._ids or k <= 0:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._mat.shape[1]:
            raise ValueError(f"查询维度 {q.shape[0]} 与索引维度 {self._mat.shape[1]} 不一致")
        scores = self._mat @ q
        if allowed is not None:
            mask = np.fromiter((i in allowed for i in self._ids), dtype=bool, count=len(self._ids))
            scores = np.where(mask, scores, -np.inf)
            if not mask.any():
                return []
        return [(i, s) for i, s in _top_k(scores, self._ids, k) if np.isfinite(s)]

    def __len__(self) -> int:
        return len(self._ids)


class BM25Index:
    """BM25Okapi 稀疏索引，与向量索引共用同一批 chunk id。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._toksets: list[set[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._ids = [c.id for c in chunks]
        self._toksets = [set(tokenize(c.text)) for c in chunks]
        # rank-bm25 对空 token 列表会崩，用占位 token 兜底
        self._bm25 = BM25Okapi([tokenize(c.text) or [""] for c in chunks]) if chunks else None

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """稀疏召回。

        入选判据是**词重叠**而不是分数 > 0：小语料下 BM25 的 IDF 会退化为 0
        （某词出现在半数文档时 log((N-df+0.5)/(df+0.5)) = log(1) = 0），
        此时"有真实词重叠但分数为 0"是合法情况；若按分数过滤会让整路召回为空。
        排序用 (score 降, overlap 降, id 升)，保证确定性。
        """
        if self._bm25 is None or not self._ids or k <= 0:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        scores = np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float64)
        cand: list[tuple[str, float, int]] = []
        for i, cid in enumerate(self._ids):
            if allowed is not None and cid not in allowed:
                continue
            overlap = len(q_set & self._toksets[i])
            score = float(scores[i])
            if score > 0 or overlap > 0:
                cand.append((cid, score, overlap))
        cand.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return [(cid, s) for cid, s, _ in cand[:k]]

    def __len__(self) -> int:
        return len(self._ids)