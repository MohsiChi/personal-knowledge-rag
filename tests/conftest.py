"""测试夹具。

**测试必须与本地环境配置无关**：绝不调用 load_settings()，因为它会读 config.local.json——
那是用户指向真实课程语料的本地配置（可能几百个 chunk + 需要下载模型）。
一旦依赖它，测试既变慢又会因为 chunk id 对不上而失败。

这个坑真实踩过：2026-09-06 创建了 config.local.json 之后，
test_retrieve / test_eval_smoke 立刻从 58 绿变成 2 红，且单次运行从 10 秒涨到 90 秒以上。
"""
from pathlib import Path

import pytest

from pkr.config import PROJECT_ROOT, Settings, WikiRoot

SAMPLE_WIKI = PROJECT_ROOT / "sample_wiki"


@pytest.fixture(scope="session")
def sample_settings() -> Settings:
    """固定指向仓库自带的合成示例语料 + hash embedder：离线、确定性、快。"""
    assert SAMPLE_WIKI.exists(), f"示例语料缺失: {SAMPLE_WIKI}"
    return Settings(
        wiki_roots=(WikiRoot("sample", SAMPLE_WIKI),),
        embedder="hash",
        top_k=5,
        recall_k=50,
        rrf_k=60,
    )


@pytest.fixture(scope="session")
def example_config() -> str:
    """给需要走子进程/CLI 的测试用：显式传 --config，绕开 config.local.json。"""
    p = PROJECT_ROOT / "config.example.json"
    assert p.exists(), f"缺少 {p}"
    return str(p)