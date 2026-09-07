"""分词器：BM25 与 HashEmbedder 共用。

中文没有空格，rank-bm25 默认按空白切分会把整句当成一个 token，对中文语料等于失效。
这里的零依赖策略：
- 拉丁字母/数字连续段 -> 一个小写词（保留 CS61B、BM25、multilingual-e5-small 这类术语）
- 每个连续 CJK 段 -> 单字（unigram）+ 段内相邻二字（bigram）

bigram 只在同一连续中文段内生成，不跨非中文间隙拼接，避免造出"文中"这类假词。
效果不如 jieba 等分词器，但零依赖、确定性、可复现。jieba 列在 roadmap：
它会引入词典依赖与分词版本不一致的风险，而评测集一旦标注完成，分词器变更会让标注失效。
"""
from __future__ import annotations

import re

_LATIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.]*")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def tokenize(text: str) -> list[str]:
    """确定性分词。同一输入永远得到同一输出（评测可复现的前提）。"""
    out: list[str] = [m.group(0).lower() for m in _LATIN.finditer(text)]
    for run in _CJK_RUN.findall(text):
        out.extend(run)
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out