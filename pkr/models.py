"""数据契约。

Chunk.id 稳定可复现（"{wiki}::{相对路径}::{块序号}"），这样重建索引后 id 不变，
评测集里人工标注的 relevant 列表不会失效——这一点对可重复评测很关键。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """一个可检索单元。"""

    id: str
    text: str
    source_wiki: str                 # 哪个知识库（支持多库联邦检索）
    source_path: str                 # 相对 wiki 根的路径（正斜杠）
    knowledge_layer: str             # concept | connection | question | paper | index
    source_type: str                 # v0 恒为 "wiki"；raw 层在 roadmap
    heading_path: tuple[str, ...]    # 标题层级，用于溯源展示
    chunk_index: int
    status: str = ""                 # frontmatter 的 status（如 stub），缺失为 ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None    # 保留两路各自的排名，调 RRF 权重时要用
    sparse_rank: int | None = None


@dataclass
class RetrievalTrace:
    """检索过程留痕。没有它你只能看到结果、看不到为什么——排查检索质量的前提。"""

    query: str = ""
    filters: dict = field(default_factory=dict)
    dense: list = field(default_factory=list)    # [(chunk_id, score)]
    sparse: list = field(default_factory=list)
    fused: list = field(default_factory=list)