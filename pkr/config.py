"""配置加载。

优先级：显式路径 > config.local.json > config.example.json > 内置默认（sample_wiki）。

语料路径永远来自本地配置，仓库里只有合成示例——课程笔记属个人/课程材料，
不该进公开仓库。config.local.json 已在 .gitignore 中。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG = PROJECT_ROOT / "config.local.json"
EXAMPLE_CONFIG = PROJECT_ROOT / "config.example.json"
# 默认用多语言模型而不是中文专用模型：课程笔记是中英混排（中文叙述 + 英文术语），
# 2026-09-06 的三文档探针实测显示 bge-small-zh-v1.5 在这种语料上几乎无区分力
# （"什么是 BM25" 三篇分数 0.4395/0.4502/0.4347，字面含 BM25 的那篇只排第二），
# 而 multilingual-MiniLM 给出 0.4914 vs 0.0488/0.0929。
# 注意：这是 3 文档探针的方向性信号，不是质量结论——最终选型必须由评测集决定。
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class WikiRoot:
    name: str
    path: Path


@dataclass(frozen=True)
class Settings:
    wiki_roots: tuple[WikiRoot, ...]
    embedder: str = "hash"            # hash（离线确定性，测试/demo）| fastembed（真实）
    embed_model: str = DEFAULT_MODEL
    top_k: int = 5
    recall_k: int = 50
    rrf_k: int = 60
    exclude_status: tuple[str, ...] = ("stub",)
    exclude_layers: tuple[str, ...] = ("index",)

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)


def _resolve(raw_path: str, base: Path) -> Path:
    p = Path(raw_path).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_settings(config_path: Path | str | None = None) -> Settings:
    candidates = [c for c in (config_path, LOCAL_CONFIG, EXAMPLE_CONFIG) if c]
    for cand in candidates:
        cand = Path(cand)
        if not cand.exists():
            continue
        raw = json.loads(cand.read_text(encoding="utf-8"))
        base = cand.resolve().parent
        roots = tuple(
            WikiRoot(name=r["name"], path=_resolve(r["path"], base))
            for r in raw.get("wiki_roots", [])
            if r.get("name") and r.get("path")
        )
        existing = tuple(r for r in roots if r.path.exists())
        if not existing:
            existing = (WikiRoot("sample", (PROJECT_ROOT / "sample_wiki").resolve()),)
        return Settings(
            wiki_roots=existing,
            embedder=os.getenv("PKR_EMBEDDER", raw.get("embedder", "hash")),
            embed_model=os.getenv("PKR_EMBED_MODEL", raw.get("embed_model", DEFAULT_MODEL)),
            top_k=int(raw.get("top_k", 5)),
            recall_k=int(raw.get("recall_k", 50)),
            rrf_k=int(raw.get("rrf_k", 60)),
            exclude_status=tuple(raw.get("exclude_status", ["stub"])),
            exclude_layers=tuple(raw.get("exclude_layers", ["index"])),
        )
    return Settings(wiki_roots=(WikiRoot("sample", PROJECT_ROOT / "sample_wiki"),))