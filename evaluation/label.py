"""人工标注工具（候选池法 / pooling）。

为什么用候选池法：让你逐页判断"这个 query 和全部 N 个 chunk 相关吗"是 O(N*M) 的苦力。
改为先让检索器给出 top-P 候选，你只在候选里勾 y/p/n —— 成本从"读完整个语料"降到"读 P 条摘要"。

**必须知道的偏差**（README 也写了）：所有检索器都没召回的页面永远不会进入候选池，
因而默认被视为不相关。这会系统性**高估**各方法的绝对分数。
但它不影响方法之间的相对比较，因为所有方法共用同一个池 —— 这正是 TREC 采用 pooling 的理由。

用法：
    # 1. 写 query 清单，一行一条。两种格式都支持：
    #      纯查询文本                      -> 自动编号 q001...，type 记为 unknown
    #      id<TAB>type<TAB>查询文本        -> 用你自己的 id 与类型
    #    type 建议取 factual / conceptual / relational / research
    # 2. 生成候选池工作表（TSV，可用 Excel 或 VSCode 打开）
    python evaluation/label.py pool --queries my_queries.txt --pool-size 20 --out evaluation/pool.tsv
    # 3. 在 pool.tsv 最后一列 label 填 y（相关）/ p（部分相关）/ n（不相关，留空等同 n）
    # 4. 回收成评测集
    python evaluation/label.py collect --pool evaluation/pool.tsv --out evaluation/dataset.json
    # 5. 跑评测
    python -m pkr eval --dataset evaluation/dataset.json --per-query
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pkr.config import load_settings          # noqa: E402
from pkr.models import RetrievalTrace         # noqa: E402
from pkr.retrieve import build_index, search  # noqa: E402

COLUMNS = ["query_id", "query_type", "query", "chunk_id", "source_wiki", "layer",
           "source_path", "heading", "routes", "excerpt", "label"]
_EXCERPT_CHARS = 140


def _clean(text: str) -> str:
    """TSV 里不能有制表符与换行，否则列会错位。"""
    return " ".join(str(text).split())


def parse_queries(path: Path) -> list[dict]:
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            qid, qtype, query = parts[0].strip(), parts[1].strip(), "\t".join(parts[2:]).strip()
        else:
            qid, qtype, query = f"q{len(items) + 1:03d}", "unknown", line
        if not query:
            continue
        items.append({"id": qid, "type": qtype, "query": query})
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        dup = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"query id 重复: {sorted(dup)}")
    return items


def cmd_pool(args) -> int:
    queries = parse_queries(Path(args.queries))
    if not queries:
        print(f"{args.queries} 里没有有效 query")
        return 2
    settings = load_settings(args.config)
    index = build_index(settings)
    print(f"语料: {len(index)} chunks | embedder: {settings.embedder} | query: {len(queries)} 条")
    if settings.embedder == "hash":
        print("⚠ 当前是 hash embedder（无语义能力）。用它生成候选池会漏掉语义相关但字面不匹配的页面，")
        print("  这些页面永远不会进入标注范围 -> 评测会系统性高估。建议先装 requirements-embed.txt")
        print("  并用 --embedder fastembed 生成候选池。若只是想先跑通流程，继续也可以。")

    rows = []
    for q in queries:
        pools: dict[str, set[str]] = {}
        for mode in ("dense", "sparse", "hybrid"):
            tr = RetrievalTrace()
            search(index, q["query"], top_k=args.pool_size, recall_k=max(50, args.pool_size),
                   mode=mode, trace=tr)
            pools[mode] = {cid for cid, _ in tr.fused} or {
                h.chunk.id for h in search(index, q["query"], top_k=args.pool_size, mode=mode)}
        # 候选池 = 三种 mode 的并集（pooling 的标准做法：池越全，偏差越小）
        union: list[str] = []
        for mode in ("hybrid", "sparse", "dense"):
            for cid in pools[mode]:
                if cid not in union:
                    union.append(cid)
        for cid in union[:args.pool_size]:
            c = index.chunks[cid]
            routes = "".join(m[0] for m in ("dense", "sparse", "hybrid") if cid in pools[m])
            rows.append({
                "query_id": q["id"], "query_type": q["type"], "query": q["query"],
                "chunk_id": cid, "source_wiki": c.source_wiki, "layer": c.knowledge_layer,
                "source_path": c.source_path, "heading": _clean(" > ".join(c.heading_path)),
                "routes": routes, "excerpt": _clean(c.text)[:_EXCERPT_CHARS], "label": "",
            })
        print(f"  {q['id']}: 候选 {min(len(union), args.pool_size)} 条")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:  # BOM 让 Excel 正确识别 UTF-8
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写入 {out}（{len(rows)} 行）")
    print("下一步：在 label 列填 y / p / n（留空视为 n），然后运行 collect 子命令。")
    return 0


def cmd_collect(args) -> int:
    path = Path(args.pool)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", restkey="_extra"))
    if not rows:
        print(f"{path} 是空的")
        return 2
    missing = [c for c in COLUMNS if c not in rows[0]]
    if missing:
        print(f"{path} 缺少列: {missing}")
        return 2

    grouped: dict[str, dict] = {}
    bad_labels = set()
    recovered = 0
    for r in rows:
        qid = r["query_id"]
        g = grouped.setdefault(qid, {"id": qid, "query": r["query"], "query_type": r["query_type"],
                                     "relevant": [], "partial": []})
        label = (r.get("label") or "").strip()
        if not label:
            # 容错：pool.tsv 的 label 列本来就空，行以 tab 结尾。手工编辑时很容易又加一个 tab，
            # 于是标注落到第 12 个字段（DictReader 收进 restkey）。这里把它捞回来。
            extras = [str(v).strip() for v in (r.get("_extra") or []) if str(v).strip()]
            if extras:
                label = extras[-1]
                recovered += 1
        label = label.lower()
        if label in ("y", "yes", "2"):
            g["relevant"].append(r["chunk_id"])
        elif label in ("p", "partial", "1"):
            g["partial"].append(r["chunk_id"])
        elif label in ("", "n", "no", "0"):
            pass
        else:
            bad_labels.add(label)
    if recovered:
        print(f"ℹ 有 {recovered} 行的标注落在多余列里（手工编辑常见的多打一个 tab），已自动捞回")
    if bad_labels:
        print(f"⚠ 无法识别的 label 值（已当作 n 处理）: {sorted(bad_labels)}")

    items = [grouped[k] for k in sorted(grouped)]
    unlabeled = [i["id"] for i in items if not i["relevant"] and not i["partial"]]
    payload = {
        "_purpose": "人工标注评测集（候选池法）。relevant=y, partial=p。",
        "_pooling_bias": "候选池之外的页面默认视为不相关，会系统性高估绝对分数；不影响方法间相对比较。",
        "_embedder_at_label_time": args.embedder_note,
        "k": args.k,
        "items": items,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}")
    print(f"  query 数: {len(items)}")
    print(f"  有标注的: {len(items) - len(unlabeled)}")
    if unlabeled:
        print(f"  ⚠ 全为 n / 未标注（评测时会被跳过）: {unlabeled}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="label", description="检索评测集标注工具（候选池法）")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("pool", help="生成候选池工作表")
    a.add_argument("--queries", required=True, help="query 清单文件（一行一条）")
    a.add_argument("--pool-size", type=int, default=20)
    a.add_argument("--out", default="evaluation/pool.tsv")
    a.add_argument("--embedder", default=None, choices=["hash", "fastembed"])
    a.set_defaults(func=cmd_pool)

    b = sub.add_parser("collect", help="把标注好的工作表回收成 dataset.json")
    b.add_argument("--pool", default="evaluation/pool.tsv")
    b.add_argument("--out", default="evaluation/dataset.json")
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--embedder-note", default="unknown",
                   help="标注时候选池用的 embedder，写进 dataset 便于日后解释数字")
    b.set_defaults(func=cmd_collect)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "embedder", None):
        import os
        os.environ["PKR_EMBEDDER"] = args.embedder
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())