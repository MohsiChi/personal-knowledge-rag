"""把 wiki 目录变成 Chunk 列表。

三件职责放在一个文件里（它们总是一起变）：
1. 扫描 wiki 根，按目录名判定 knowledge_layer
2. 解析 YAML frontmatter 成 metadata
3. 按 markdown 标题层级切块，且不切断代码围栏

已按真实语料核对过的三个细节：
- frontmatter 是 YAML（title/date/tags/status），部分页面带 status: stub（空壳占位），
  默认排除，否则检索会被无内容的页面污染；
- wiki/ 根下的散文件（如 index.md 清单页）判为 index 层，默认排除，否则它会匹配一切；
- 正文含表格与代码围栏，切块时必须跟踪围栏状态，否则会把代码块从中间切断。
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from pkr.config import Settings
from pkr.models import Chunk

LAYER_BY_DIR = {
    "concepts": "concept",
    "connections": "connection",
    "questions": "question",
    "papers": "paper",
}

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def split_frontmatter(text: str) -> tuple[dict, str]:
    """拆出 YAML frontmatter。没有或解析失败时返回 ({}, 原文)，不抛异常。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            try:
                meta = yaml.safe_load("\n".join(lines[1:i])) or {}
            except yaml.YAMLError:
                meta = {}
            return (meta if isinstance(meta, dict) else {}), "\n".join(lines[i + 1:])
    return {}, text


def chunk_markdown(body: str, max_chars: int = 1200) -> list[tuple[tuple[str, ...], str]]:
    """按标题层级切块 -> [(heading_path, text), ...]。

    规则：
    - 遇到标题就收束当前块，标题层级栈给出完整 heading_path（便于溯源）
    - 代码围栏（``` / ~~~）内部的 # 不当作标题，围栏内容不被切断
    - 单块超过 max_chars 时按空行二次切分，避免大块稀释向量语义
    """
    out: list[tuple[tuple[str, ...], str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    cur_path: tuple[str, ...] = ()
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        buf = []
        if text:
            out.append((cur_path, text))

    for line in body.splitlines():
        fm = _FENCE.match(line)
        if fm:
            marker = fm.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            buf.append(line)
            continue

        hm = None if in_fence else _HEADING.match(line)
        if hm:
            flush()
            level = len(hm.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, hm.group(2)))
            cur_path = tuple(t for _, t in stack)
            buf.append(line)
        else:
            buf.append(line)
    flush()

    result: list[tuple[tuple[str, ...], str]] = []
    for path, text in out:
        if len(text) <= max_chars:
            result.append((path, text))
            continue
        parts: list[str] = []
        cur = ""
        for para in text.split("\n\n"):
            if cur and len(cur) + len(para) + 2 > max_chars:
                parts.append(cur)
                cur = para
            else:
                cur = f"{cur}\n\n{para}" if cur else para
        if cur:
            parts.append(cur)
        result.extend((path, p) for p in parts)
    return result


def iter_wiki_files(wiki_root: Path):
    """产出 (绝对路径, 相对路径 posix, layer)。

    wiki_root 按 knowledge-wiki 约定是含 wiki/ 的目录（research-wiki/）；
    若直接指向 wiki 目录本身也能容错。
    """
    base = wiki_root / "wiki"
    if not base.exists():
        base = wiki_root
    for p in sorted(base.rglob("*.md")):
        rel = p.relative_to(wiki_root)
        layer = "index"
        for seg in rel.parts:
            if seg in LAYER_BY_DIR:
                layer = LAYER_BY_DIR[seg]
                break
        yield p, rel.as_posix(), layer


def build_chunks(settings: Settings) -> list[Chunk]:
    """扫描全部 wiki 根，产出 Chunk 列表（已按配置排除 stub / index 层）。"""
    chunks: list[Chunk] = []
    for root in settings.wiki_roots:
        for path, rel, layer in iter_wiki_files(root.path):
            text = path.read_text(encoding="utf-8", errors="replace")
            meta, body = split_frontmatter(text)
            status = str(meta.get("status") or "")
            if status and status in settings.exclude_status:
                continue
            if layer in settings.exclude_layers:
                continue
            pieces = chunk_markdown(body)
            for idx, (hpath, ctext) in enumerate(pieces):
                chunks.append(Chunk(
                    id=f"{root.name}::{rel}::{idx}",
                    text=ctext,
                    source_wiki=root.name,
                    source_path=rel,
                    knowledge_layer=layer,
                    source_type="wiki",
                    heading_path=hpath,
                    chunk_index=idx,
                    status=status,
                    metadata={k: v for k, v in meta.items() if k != "status"},
                ))
    return chunks