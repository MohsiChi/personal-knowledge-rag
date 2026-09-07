"""命令行入口：python -m pkr <stats|search|eval>

刻意只做 CLI，不做 Web UI：v0 的目标是把检索质量做到可评测，
界面不解决任何检索质量问题，却会让评测结果难以复现。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pkr.config import load_settings
from pkr.metrics import evaluate
from pkr.models import RetrievalTrace
from pkr.retrieve import MODES, build_index, search


def _print_hits(hits, show_text: int) -> None:
    for rank, h in enumerate(hits, start=1):
        c = h.chunk
        head = " > ".join(c.heading_path) or "(无标题)"
        print(f"{rank}. [{c.knowledge_layer}] {c.source_wiki}/{c.source_path}")
        print(f"   {head}")
        print(f"   score={h.score:.6f}  dense_rank={h.dense_rank}  sparse_rank={h.sparse_rank}")
        if show_text:
            snippet = " ".join(c.text.split())[:show_text]
            print(f"   {snippet}{'…' if len(c.text) > show_text else ''}")
        print()


def cmd_stats(args) -> int:
    settings = load_settings(args.config)
    index = build_index(settings)
    print(f"embedder = {settings.embedder}"
          + (f" ({settings.embed_model})" if settings.embedder != "hash" else " (离线确定性，仅用于验证管线)"))
    print(f"wiki 根 = {[r.name for r in settings.wiki_roots]}")
    print(f"chunk 总数 = {len(index)}")
    by_wiki: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    for c in index.chunks.values():
        by_wiki[c.source_wiki] = by_wiki.get(c.source_wiki, 0) + 1
        by_layer[c.knowledge_layer] = by_layer.get(c.knowledge_layer, 0) + 1
    print("按库:", json.dumps(by_wiki, ensure_ascii=False))
    print("按层:", json.dumps(by_layer, ensure_ascii=False))
    print(f"已排除: status in {list(settings.exclude_status)}, layer in {list(settings.exclude_layers)}")
    return 0


def cmd_search(args) -> int:
    settings = load_settings(args.config)
    index = build_index(settings)
    trace = RetrievalTrace() if args.trace else None
    hits = search(index, args.query, top_k=args.top_k, recall_k=settings.recall_k,
                  rrf_k=settings.rrf_k, mode=args.mode,
                  wikis=args.wiki or None, layers=args.layer or None, trace=trace)
    if not hits:
        print("（无结果）")
        return 0
    _print_hits(hits, args.snippet)
    if trace is not None:
        print("--- retrieval_trace ---")
        print("eligible:", trace.filters.get("eligible"))
        for name, route in (("dense", trace.dense), ("sparse", trace.sparse)):
            print(f"{name} ({len(route)}):")
            for cid, s in route[:10]:
                print(f"   {s:>10.4f}  {cid}")
        print("fused:")
        for cid, s in trace.fused:
            print(f"   {s:>10.6f}  {cid}")
    return 0


def cmd_eval(args) -> int:
    path = Path(args.dataset)
    if not path.exists():
        print(f"评测集不存在: {path}")
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"] if isinstance(payload, dict) else payload
    k = int(payload.get("k", args.k)) if isinstance(payload, dict) else args.k

    settings = load_settings(args.config)
    index = build_index(settings)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            print(f"未知 mode: {m}（可选 {' | '.join(MODES)}）")
            return 2

    rows = []
    for mode in modes:
        def run(query, filters, _mode=mode):
            hits = search(index, query, top_k=k, recall_k=settings.recall_k,
                          rrf_k=settings.rrf_k, mode=_mode,
                          wikis=filters.get("wikis"), layers=filters.get("layers"))
            return [h.chunk.id for h in hits]

        res = evaluate(items, run, k=k)
        rows.append((mode, res))

    emb_desc = getattr(index.embedder, "describe", None) or f"{settings.embedder} / dim={index.embedder.dim}"
    print(f"\n评测集: {path.name}   k={k}   语料={len(index)} chunks")
    print(f"embedder: {emb_desc}   （评测数字必须与此版本绑定，否则不可复现）")
    if settings.embedder == "hash":
        print("⚠ hash embedder 无语义泛化能力：以下数字只验证管线正确性，"
              "**不能**当作检索质量结论。真实评测需 --embedder fastembed。")
    print()
    print("| Method | Recall@{k} | Precision@{k} | MRR | NDCG@{k} |".format(k=k))
    print("|---|---:|---:|---:|---:|")
    for mode, res in rows:
        print(f"| {mode} | {res['recall@k']:.4f} | {res['precision@k']:.4f} "
              f"| {res['mrr']:.4f} | {res['ndcg@k']:.4f} |")
    nq = rows[0][1]["n_queries"]
    print(f"\n有效 query 数: {nq}（无相关文档而跳过: {rows[0][1]['skipped_no_relevant']}）")
    if nq == 0:
        # 全 NaN 的表格很容易被误当成结果贴出去，这里直接以非零码退出
        print("✗ 没有任何带标注的 query，指标全为 NaN。请检查 dataset 的 relevant/partial 是否为空，")
        print("  或标注是否落到了多余列（label.py collect 会自动捞回并提示）。")
        return 3

    if args.per_query:
        print("\n逐条明细:")
        for mode, res in rows:
            print(f"\n[{mode}]")
            for pq in res["per_query"]:
                print(f"  {pq['id']}  R@{k}={pq['recall']}  MRR={pq['mrr']}  NDCG={pq['ndcg']}  {pq['query']}")
    if args.out:
        out = {"_embedder": emb_desc, "_corpus_chunks": len(index),
               **{mode: {kk: vv for kk, vv in res.items() if kk != "per_query"} for mode, res in rows}}
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.out}")
    return 0


def _common() -> argparse.ArgumentParser:
    """全局选项。用 parents 挂到主 parser 与每个子命令上，
    这样 `pkr --embedder fastembed eval` 和 `pkr eval --embedder fastembed` 都能用——
    argparse 的父级选项默认只认子命令之前的位置，这个坑不值得让用户踩。
    """
    # default 必须是 SUPPRESS 而不是 None：父解析器与子解析器共享同一个 dest，
    # 若子解析器的默认值是 None，它会在"用户把选项写在子命令之前"时把父解析器
    # 已解析到的值覆盖成 None —— 于是 `pkr --config X stats` 会静默退回 config.local.json，
    # 用户可能对着错误的语料跑完整套评测而毫不知情。这个 bug 真实发生过（2026-09-06）。
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--config", default=argparse.SUPPRESS,
                   help="配置文件路径（默认 config.local.json > config.example.json）")
    c.add_argument("--embedder", default=argparse.SUPPRESS, choices=["hash", "fastembed"],
                   help="覆盖配置里的 embedder")
    return c


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    p = argparse.ArgumentParser(prog="pkr", description="Personal Knowledge RAG (v0)",
                                parents=[common])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="语料统计", parents=[common]).set_defaults(func=cmd_stats)

    s = sub.add_parser("search", help="检索", parents=[common])
    s.add_argument("query")
    s.add_argument("--mode", default="hybrid", choices=list(MODES))
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--wiki", action="append", help="只检索指定库（可重复）")
    s.add_argument("--layer", action="append", help="只检索指定层（concept/connection/question/paper）")
    s.add_argument("--snippet", type=int, default=120, help="打印正文摘要长度，0 表示不打印")
    s.add_argument("--trace", action="store_true", help="打印两路各自的召回与融合过程")
    s.set_defaults(func=cmd_search)

    e = sub.add_parser("eval", help="在评测集上跑 ablation", parents=[common])
    e.add_argument("--dataset", default="evaluation/dataset.smoke.json")
    e.add_argument("--modes", default="dense,sparse,hybrid")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--per-query", action="store_true", help="打印逐条明细")
    e.add_argument("--out", default=None, help="把汇总指标写成 JSON")
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # 用了 SUPPRESS 之后，未指定的选项根本不会成为属性，这里统一补默认值
    if not hasattr(args, "config"):
        args.config = None
    if not hasattr(args, "embedder"):
        args.embedder = None
    if args.embedder:
        import os
        os.environ["PKR_EMBEDDER"] = args.embedder
    return args.func(args)