"""检索编排：dense + sparse -> RRF 融合 -> metadata 过滤 -> ScoredChunk + trace。

mode 支持 dense / sparse / hybrid，这是评测做 ablation 的入口：
同一份语料、同一个评测集，只切 mode 就能对比三种策略——
"我用数据决定要不要加某个组件"必须建立在这个能力之上。

过滤是**前置**的（在排名之前把不合格候选剔除），而不是取完 top-k 再筛。
后置过滤会导致 top_k 被筛空，指标虚低且无法解释。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pkr.config import Settings
from pkr.embed import Embedder, build_embedder
from pkr.fuse import rrf
from pkr.index import BM25Index, NumpyVectorIndex, VectorIndex
from pkr.ingest import build_chunks
from pkr.models import Chunk, RetrievalTrace, ScoredChunk

MODES = ("dense", "sparse", "hybrid")


@dataclass
class KnowledgeIndex:
    chunks: dict[str, Chunk]
    order: list[str]
    vector: VectorIndex
    sparse: BM25Index
    embedder: Embedder
    embedder_name: str = ""

    def __len__(self) -> int:
        return len(self.order)


def build_index(settings: Settings, embedder: Embedder | None = None) -> KnowledgeIndex:
    """从配置扫描语料并建双路索引。v0 每次重建（见 index.py 关于不做持久化的理由）。"""
    chunks = build_chunks(settings)
    emb = embedder or build_embedder(settings.embedder, settings.embed_model)
    order = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    matrix = emb.embed(texts) if texts else np.zeros((0, emb.dim), dtype=np.float32)

    vector = NumpyVectorIndex()
    vector.build(order, matrix)
    sparse = BM25Index()
    sparse.build(chunks)
    return KnowledgeIndex(chunks={c.id: c for c in chunks}, order=order, vector=vector,
                          sparse=sparse, embedder=emb, embedder_name=settings.embedder)


def _eligible(index: KnowledgeIndex, wikis, layers) -> set[str] | None:
    """None 表示不过滤。返回空集合表示过滤后无候选（调用方需区分这两种情况）。"""
    if not wikis and not layers:
        return None
    ws = set(wikis) if wikis else None
    ls = set(layers) if layers else None
    return {cid for cid, c in index.chunks.items()
            if (ws is None or c.source_wiki in ws) and (ls is None or c.knowledge_layer in ls)}


def _rank(hits: list[tuple[str, float]], cid: str) -> int | None:
    for i, (doc_id, _) in enumerate(hits, start=1):
        if doc_id == cid:
            return i
    return None


def search(index: KnowledgeIndex, query: str, *, top_k: int = 5, recall_k: int = 50,
           rrf_k: int = 60, mode: str = "hybrid", wikis=None, layers=None,
           trace: RetrievalTrace | None = None) -> list[ScoredChunk]:
    if mode not in MODES:
        raise ValueError(f"未知 mode: {mode!r}（可选 {' | '.join(MODES)}）")
    if not query or not query.strip():
        return []

    allowed = _eligible(index, wikis, layers)
    if allowed is not None and not allowed:
        if trace is not None:
            trace.query, trace.filters = query, {"wikis": list(wikis or []),
                                                 "layers": list(layers or []), "eligible": 0}
        return []

    dense_hits: list[tuple[str, float]] = []
    sparse_hits: list[tuple[str, float]] = []
    if mode in ("dense", "hybrid"):
        dense_hits = index.vector.search(index.embedder.embed([query])[0], recall_k, allowed=allowed)
    if mode in ("sparse", "hybrid"):
        sparse_hits = index.sparse.search(query, recall_k, allowed=allowed)

    if mode == "dense":
        fused = list(dense_hits)
    elif mode == "sparse":
        fused = list(sparse_hits)
    else:
        fused = rrf([[c for c, _ in dense_hits], [c for c, _ in sparse_hits]], k=rrf_k)

    if trace is not None:
        trace.query = query
        trace.filters = {"wikis": list(wikis or []), "layers": list(layers or []),
                         "eligible": len(allowed) if allowed is not None else len(index)}
        trace.dense = [(c, round(s, 6)) for c, s in dense_hits]
        trace.sparse = [(c, round(s, 6)) for c, s in sparse_hits]
        trace.fused = [(c, round(s, 6)) for c, s in fused[:top_k]]

    out: list[ScoredChunk] = []
    for cid, score in fused[:top_k]:
        chunk = index.chunks.get(cid)
        if chunk is None:      # 索引与 chunks 不一致属内部 bug，静默跳过会掩盖它
            raise KeyError(f"检索结果里的 {cid!r} 不在 chunks 表中（索引不一致）")
        out.append(ScoredChunk(chunk=chunk, score=float(score),
                               dense_rank=_rank(dense_hits, cid),
                               sparse_rank=_rank(sparse_hits, cid)))
    return out