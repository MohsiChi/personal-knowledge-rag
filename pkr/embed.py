"""Embedding 抽象与两个实现。

- HashEmbedder：确定性哈希投影。零依赖、离线、可复现，用于测试与无网 demo。
  它是真实的词袋哈希投影（能区分文本），但**没有语义泛化能力**。
  用它跑出的评测数字只能验证管线正确性，不能当作检索质量结论——README 明确写了这条。
- FastEmbedder：fastembed（ONNX 本地推理）。**不调外部 API、不需要 key、不产生费用**；
  首次运行下载模型（multilingual-MiniLM 约 220MB），之后完全离线并缓存在本地。
  语料是个人学习笔记，走本地推理意味着它们不会被上传给任何第三方。
  模型选择见 config.py 的 DEFAULT_MODEL 注释（含实测依据与其局限）。
  两个候选模型的 fastembed 元数据都注明 "Prefixes for queries/documents: not necessary"，
  因此本实现不给 query 加指令前缀——这是查库得到的结论，不是推测。

两者实现同一接口，切换只改配置（embedder: hash | fastembed），上层代码零改动。
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from pkr.text import tokenize


class Embedder(ABC):
    """把文本编码成 L2 归一化向量。归一化后余弦相似度等于点积。"""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """返回 shape=(len(texts), dim) 的 float32 矩阵。"""


class HashEmbedder(Embedder):
    """确定性哈希投影（测试/离线 demo 用）。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(t) for t in texts]).astype(np.float32)

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v


class FastEmbedder(Embedder):
    """fastembed 本地 ONNX 推理。延迟导入：没装 fastembed 也能用 HashEmbedder。"""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover - 取决于是否安装 requirements-embed.txt
            raise RuntimeError(
                "未安装 fastembed。真实 embedding 需先执行："
                "pip install -r requirements-embed.txt"
            ) from e
        self._model = TextEmbedding(model_name=model_name)
        self._name = model_name
        # 维度靠一次探针编码取回，不写死：换模型不需要改代码。
        # 注意 fastembed 对该模型的 pooling 行为随版本变过（0.8.0 起为 mean pooling），
        # 所以评测数字必须与 fastembed 版本一起记录，否则日后无法复现。
        probe = next(iter(self._model.embed(["probe"])))
        self.dim = int(len(probe))

    @property
    def describe(self) -> str:
        """可复现性所需的完整描述：库版本 + 模型名 + 维度。"""
        try:
            import fastembed
            ver = getattr(fastembed, "__version__", "unknown")
        except Exception:      # pragma: no cover
            ver = "unknown"
        return f"fastembed {ver} / {self._name} / dim={self.dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = [np.asarray(v, dtype=np.float32) for v in self._model.embed(texts)]
        mat = np.stack(vecs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)


def build_embedder(name: str, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> Embedder:
    """按配置构造 embedder。未知名字直接报错，不静默回退——静默降级会让评测结果无法解释。"""
    key = (name or "hash").strip().lower()
    if key == "hash":
        return HashEmbedder()
    if key in ("fastembed", "fastembed-onnx"):
        return FastEmbedder(model_name=model)
    raise ValueError(f"未知 embedder: {name!r}（可选 hash | fastembed）")