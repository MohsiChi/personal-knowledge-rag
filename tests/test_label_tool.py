"""标注工具的回收闭环测试。

覆盖一个真实踩过的坑：pool.tsv 的 label 列本来为空、行以 tab 结尾，
手工编辑时很容易再打一个 tab，导致标注落到第 12 个字段。collect 必须能捞回来，
不能静默丢弃（静默丢弃的后果是评测集全空、指标全 NaN，而你以为标注做完了）。
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = ROOT / "evaluation" / "label.py"
COLUMNS = ["query_id", "query_type", "query", "chunk_id", "source_wiki", "layer",
           "source_path", "heading", "routes", "excerpt", "label"]


def _rows():
    base = {"query_id": "q001", "query_type": "conceptual", "query": "什么是 RRF",
            "source_wiki": "w"}
    return [
        {**base, "chunk_id": "w::wiki/concepts/a.md::0", "layer": "concept",
         "source_path": "wiki/concepts/a.md", "heading": "A", "routes": "dsh",
         "excerpt": "内容 A", "label": "y"},
        {**base, "chunk_id": "w::wiki/concepts/a.md::1", "layer": "concept",
         "source_path": "wiki/concepts/a.md", "heading": "A > A1", "routes": "ds",
         "excerpt": "内容 A1", "label": "p"},
        {**base, "chunk_id": "w::wiki/concepts/b.md::0", "layer": "concept",
         "source_path": "wiki/concepts/b.md", "heading": "B", "routes": "s",
         "excerpt": "内容 B", "label": "n"},
    ]


def _write_pool(path: Path, rows, trailing_extra: bool = False) -> None:
    """写 pool.tsv。

    trailing_extra=True 模拟真实踩过的坑：pool.tsv 的 label 列本来是空的（行以 tab 结尾），
    用户在编辑器里手工填时又打了一个 tab，于是标注落到第 12 个字段。
    因此必须先把 label 列清空，再把标注追加到行尾——否则 label 列本来就有值，
    容错分支根本不会被触发（这个脚手架 bug 我踩过一次）。
    """
    pool_rows = [{**r, "label": ""} for r in rows] if trailing_extra else rows
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(pool_rows)
    if trailing_extra:
        text = path.read_text(encoding="utf-8-sig")
        nl = "\r\n" if "\r\n" in text else "\n"
        lines = text.rstrip("\r\n").split(nl)
        fixed = [lines[0]] + [ln + "\t" + r["label"] for ln, r in zip(lines[1:], rows)]
        path.write_text(nl.join(fixed) + nl, encoding="utf-8-sig")


def _collect(pool, out):
    return subprocess.run(
        [sys.executable, str(LABEL), "--config", str(ROOT / "config.example.json"),
         "collect", "--pool", str(pool), "--out", str(out), "--embedder-note", "test"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))


def test_collect_normal(tmp_path):
    pool, out = tmp_path / "pool.tsv", tmp_path / "ds.json"
    _write_pool(pool, _rows())
    r = _collect(pool, out)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text(encoding="utf-8"))
    assert len(d["items"]) == 1
    assert d["items"][0]["relevant"] == ["w::wiki/concepts/a.md::0"]
    assert d["items"][0]["partial"] == ["w::wiki/concepts/a.md::1"]


def test_collect_recovers_label_from_extra_tab_column(tmp_path):
    """真实踩过的坑：行尾多打一个 tab，标注落到第 12 列。必须自动捞回并提示。"""
    pool, out = tmp_path / "pool.tsv", tmp_path / "ds.json"
    _write_pool(pool, _rows(), trailing_extra=True)
    r = _collect(pool, out)
    assert r.returncode == 0, r.stderr
    assert "捞回" in r.stdout, f"应提示已捞回，实际: {r.stdout}"
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["items"][0]["relevant"] == ["w::wiki/concepts/a.md::0"], "标注被静默丢弃了"
    assert d["items"][0]["partial"] == ["w::wiki/concepts/a.md::1"]


def test_collect_rejects_empty_and_bad_columns(tmp_path):
    pool = tmp_path / "empty.tsv"
    pool.write_text("", encoding="utf-8")
    assert _collect(pool, tmp_path / "o.json").returncode == 2

    bad = tmp_path / "bad.tsv"
    bad.write_text("query_id\tquery\nq1\tx\n", encoding="utf-8")
    r = _collect(bad, tmp_path / "o.json")
    assert r.returncode == 2 and "缺少列" in r.stdout


def test_eval_exits_nonzero_when_nothing_labeled(tmp_path):
    """全 NaN 的指标表很容易被误当结果贴出去，必须以非零码退出并说明原因。"""
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps({"k": 5, "items": [{"id": "q1", "query": "x", "relevant": []}]},
                             ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "pkr", "eval",
                        "--config", str(ROOT / "config.example.json"), "--dataset", str(ds)],
                       capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
    assert r.returncode == 3, (r.returncode, r.stdout)
    assert "NaN" in r.stdout