"""索引层测试：向量索引与 BM25 索引，含 metadata 过滤与维度校验。"""
import numpy as np
import pytest

from pkr.index import BM25Index, NumpyVectorIndex
from pkr.models import Chunk


def _chunk(cid, text, layer="concept", wiki="w"):
    return Chunk(id=cid, text=text, source_wiki=wiki, source_path=f"wiki/{cid}.md",
                 knowledge_layer=layer, source_type="wiki", heading_path=(cid,),
                 chunk_index=0)


def test_vector_index_returns_nearest_first():
    vi = NumpyVectorIndex()
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    vi.build(["a", "b", "c"], vecs)
    got = vi.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert [i for i, _ in got] == ["a", "c"], "与查询同向的 a 第一，接近的 c 第二"
    assert got[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vector_index_allowed_filter():
    vi = NumpyVectorIndex()
    vi.build(["a", "b"], np.eye(2, dtype=np.float32))
    got = vi.search(np.array([1.0, 0.0], dtype=np.float32), k=2, allowed={"b"})
    assert [i for i, _ in got] == ["b"], "被过滤掉的候选不得出现在结果里"


def test_vector_index_empty_allowed_returns_empty():
    vi = NumpyVectorIndex()
    vi.build(["a"], np.array([[1.0, 0.0]], dtype=np.float32))
    assert vi.search(np.array([1.0, 0.0], dtype=np.float32), k=1, allowed=set()) == []


def test_vector_index_dim_mismatch_raises():
    vi = NumpyVectorIndex()
    vi.build(["a"], np.array([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        vi.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)


def test_vector_index_build_size_mismatch_raises():
    vi = NumpyVectorIndex()
    with pytest.raises(ValueError):
        vi.build(["a", "b"], np.array([[1.0, 0.0]], dtype=np.float32))


def test_bm25_ranks_exact_term_first():
    idx = BM25Index()
    idx.build([_chunk("a", "讲的是 BM25 打分函数"), _chunk("b", "讲的是向量检索"),
               _chunk("c", "完全无关的内容")])
    got = idx.search("BM25", k=3)
    assert got and got[0][0] == "a"


def test_bm25_excludes_docs_without_term_overlap():
    """与查询无任何词重叠的文档不得进入稀疏召回（否则会白占 RRF 的排名位）。"""
    idx = BM25Index()
    idx.build([_chunk("a", "苹果香蕉"), _chunk("b", "橙子葡萄")])
    got = idx.search("苹果", k=5)
    assert [i for i, _ in got] == ["a"]


def test_bm25_survives_idf_degeneracy_on_tiny_corpus():
    """回归测试：N=2、df=1 时 IDF=log(1)=0，分数全为 0。

    曾经用"分数 > 0"当入选判据，导致这种小语料下整路召回为空——
    真实词重叠的文档被误杀。判据必须是词重叠，不是分数。
    """
    idx = BM25Index()
    idx.build([_chunk("a", "苹果香蕉"), _chunk("b", "橙子葡萄")])
    got = idx.search("苹果", k=5)
    assert got, "IDF 退化为 0 时仍必须召回有词重叠的文档"
    assert got[0][1] == 0.0, "此语料下 BM25 分数确实为 0（IDF 退化），这是预期而非 bug"
    assert idx.search("完全无关词", k=5) == [], "无词重叠时必须为空"


def test_bm25_chinese_query_works():
    """中文没有空格，默认空白分词会整句成一个 token；这里必须能命中。

    语料给到 6 篇：N=2 时 IDF 会退化（见下一条测试），测的就不是中文分词而是退化行为了。
    """
    idx = BM25Index()
    idx.build([
        _chunk("a", "稀疏检索依赖词频统计"),
        _chunk("b", "稠密检索依赖向量相似度"),
        _chunk("c", "神经网络的反向传播算法"),
        _chunk("d", "数据库索引的 B+ 树结构"),
        _chunk("e", "操作系统的进程调度策略"),
        _chunk("f", "编译器的语法分析阶段"),
    ])
    got = idx.search("稀疏检索", k=3)
    assert got and got[0][0] == "a", got
    # 注意是 >= 0 而不是 > 0：查询里的「索」在 6 篇中出现 3 次，
    # IDF = log((6-3+0.5)/(3+0.5)) = log(1) = 0，即该词无区分力，
    # 含它的文档得 0 分但仍因词重叠入选。这是 BM25 的正常行为。
    assert all(s >= 0 for _, s in got), f"语料够大时 IDF 不应为负，实际 {got}"
    assert got[0][1] > 0, "命中多个高 IDF 词的首位文档分数应为正"


def test_bm25_idf_degeneracy_on_tiny_corpus_is_documented():
    """记录性测试：N=2 且查询词同时出现在两篇时，rank-bm25 的 epsilon 修正会得到负 IDF。

    后果是分数为负、且长度归一化方向反转（短文档反而更低）。这是库的已知行为，
    不是本项目的 bug。本项目**刻意不修改 BM25 打分**，以保持与标准实现可比。
    真正的结论是：评测语料不能太小，否则稀疏路的数字没有解释力——已写进 README。
    """
    idx = BM25Index()
    idx.build([_chunk("a", "稀疏检索依赖词频统计"), _chunk("b", "稠密检索依赖向量相似度")])
    got = idx.search("稀疏检索", k=2)
    assert len(got) == 2, "有词重叠的文档仍应被召回"
    assert all(s < 0 for _, s in got), f"预期负分（IDF 退化），实际 {got}"


def test_bm25_allowed_filter_and_empty_corpus():
    idx = BM25Index()
    idx.build([_chunk("a", "苹果"), _chunk("b", "苹果香蕉")])
    assert [i for i, _ in idx.search("苹果", k=2, allowed={"b"})] == ["b"]
    empty = BM25Index()
    empty.build([])
    assert empty.search("任何", k=3) == []
    assert len(empty) == 0