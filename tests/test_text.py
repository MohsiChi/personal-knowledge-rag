"""分词器测试。中文 BM25 的正确性完全取决于这里。"""
from pkr.text import tokenize


def test_latin_lowercased_and_kept_as_term():
    assert "bm25" in tokenize("BM25 是稀疏检索")
    assert "multilingual-e5-small" in tokenize("模型 multilingual-e5-small 的维度")


def test_cjk_unigram_and_bigram():
    toks = tokenize("是稀疏检索")
    for u in "是稀疏检索":
        assert u in toks, f"缺 unigram {u}"
    for b in ("稀疏", "疏检", "检索"):
        assert b in toks, f"缺 bigram {b}"


def test_bigram_does_not_cross_non_cjk_gap():
    """「中文abc中文」不应产生假词「文中」——bigram 只在同一连续中文段内生成。"""
    toks = tokenize("中文abc中文")
    assert "文中" not in toks
    assert "中文" in toks and "abc" in toks


def test_deterministic():
    assert tokenize("混合检索 RRF 融合") == tokenize("混合检索 RRF 融合")


def test_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("，。！？") == []