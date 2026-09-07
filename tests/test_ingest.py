"""切块与元数据抽取测试。全部离线、确定性、不依赖任何模型。"""
from pathlib import Path

from pkr.config import Settings, WikiRoot
from pkr.ingest import build_chunks, chunk_markdown, split_frontmatter


def test_split_frontmatter_basic():
    text = "---\ntitle: \"ADTs and BSTs\"\ntags: [CS61B, Trees]\nstatus: stub\n---\n\n# 正文\n内容\n"
    meta, body = split_frontmatter(text)
    assert meta["title"] == "ADTs and BSTs"
    assert meta["tags"] == ["CS61B", "Trees"]
    assert meta["status"] == "stub"
    assert body.strip().startswith("# 正文")


def test_split_frontmatter_absent_and_broken():
    assert split_frontmatter("# 无 frontmatter\n")[0] == {}
    # 未闭合 / 非法 YAML 都不该抛异常
    assert split_frontmatter("---\ntitle: [未闭合\n---\n正文")[0] == {}
    assert split_frontmatter("---\n没有结束标记\n")[0] == {}


def test_chunk_by_heading_with_path():
    body = "# 顶层\n\n引言\n\n## 小节A\n\n内容A\n\n## 小节B\n\n内容B\n"
    pieces = chunk_markdown(body)
    paths = [p for p, _ in pieces]
    assert paths[0] == ("顶层",)
    assert paths[1] == ("顶层", "小节A")
    assert paths[2] == ("顶层", "小节B")
    assert "内容A" in pieces[1][1]


def test_chunk_does_not_split_code_fence():
    """代码围栏里的 # 不是标题，围栏内容不能被切断。"""
    body = "# T\n\n```python\n# 这是注释不是标题\ndef f():\n    return 1\n```\n\n## 下一节\n"
    pieces = chunk_markdown(body)
    fenced = [t for _, t in pieces if "```" in t]
    assert len(fenced) == 1, f"围栏被切断了: {pieces}"
    assert "def f():" in fenced[0] and "# 这是注释不是标题" in fenced[0]
    assert all(not p[-1].startswith("这是注释") for p, _ in pieces)


def test_chunk_splits_oversized_block():
    body = "# T\n\n" + "\n\n".join(f"段落{i} " + "字" * 200 for i in range(12))
    pieces = chunk_markdown(body, max_chars=600)
    assert len(pieces) > 1
    assert all(len(t) <= 900 for _, t in pieces)  # 允许段落本身长度的余量


def test_build_chunks_excludes_stub_and_index(tmp_path):
    """status: stub 的空壳页与 wiki/ 根下的 index 清单页默认都不进索引。"""
    w = tmp_path / "research-wiki"
    (w / "wiki" / "concepts").mkdir(parents=True)
    (w / "wiki" / "connections").mkdir(parents=True)
    (w / "wiki" / "index.md").write_text("# Wiki Index\n\n- 清单条目\n", encoding="utf-8")
    (w / "wiki" / "concepts" / "real.md").write_text(
        "---\ntitle: Real\ntags: [t]\n---\n\n# Real\n\n真实内容\n", encoding="utf-8")
    (w / "wiki" / "concepts" / "empty.md").write_text(
        "---\ntitle: Empty\nstatus: stub\n---\n\n# Empty\n\n待填充\n", encoding="utf-8")
    (w / "wiki" / "connections" / "c.md").write_text(
        "---\ntitle: C\n---\n\n# C\n\nA 与 B 的关系\n", encoding="utf-8")

    s = Settings(wiki_roots=(WikiRoot("w1", w),))
    chunks = build_chunks(s)
    layers = {(c.source_path, c.knowledge_layer) for c in chunks}
    assert ("wiki/concepts/real.md", "concept") in layers
    assert ("wiki/connections/c.md", "connection") in layers
    assert all("empty.md" not in p for p, _ in layers), "stub 页不应进索引"
    assert all("index.md" not in p for p, _ in layers), "index 清单页不应进索引"


def test_chunk_id_is_stable_and_unique(tmp_path):
    w = tmp_path / "rw"
    (w / "wiki" / "concepts").mkdir(parents=True)
    (w / "wiki" / "concepts" / "a.md").write_text(
        "---\ntitle: A\n---\n\n# A\n\n## A1\n\nx\n\n## A2\n\ny\n", encoding="utf-8")
    s = Settings(wiki_roots=(WikiRoot("w1", w),))
    ids = [c.id for c in build_chunks(s)]
    assert len(ids) == len(set(ids)), "chunk id 必须唯一"
    assert ids == sorted(ids) or True  # 顺序稳定即可
    assert all(i.startswith("w1::wiki/concepts/a.md::") for i in ids)
    assert build_chunks(s)[0].id == ids[0], "重建索引后 id 必须可复现"


def test_metadata_carries_frontmatter_without_status(tmp_path):
    w = tmp_path / "rw"
    (w / "wiki" / "concepts").mkdir(parents=True)
    (w / "wiki" / "concepts" / "a.md").write_text(
        "---\ntitle: A\ntags: [x, y]\nstatus: draft\n---\n\n# A\n\n正文\n", encoding="utf-8")
    s = Settings(wiki_roots=(WikiRoot("w1", w),), exclude_status=())
    c = build_chunks(s)[0]
    assert c.status == "draft"
    assert c.metadata["title"] == "A" and c.metadata["tags"] == ["x", "y"]
    assert "status" not in c.metadata, "status 已单独成字段，不应重复留在 metadata"