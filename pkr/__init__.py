"""pkr — Personal Knowledge RAG.

给结构化知识库（兼容 knowledge-wiki 的 wiki/ 目录约定）做的混合检索层：
dense + BM25 双路召回 -> RRF 融合 -> 带出处的结果与 retrieval_trace。

设计取舍见 README。v0 刻意不含：raw PDF 层、reranker、LLM 生成、query 分类、
图扩展、Web UI —— 先把检索质量做到可评测，再谈加组件。
"""

__version__ = "0.1.0"