"""CLI 全局选项的解析回归测试。

针对一个真实发生过的静默 bug：argparse 的 parents 共享 dest 时，
子解析器的 default=None 会覆盖父解析器已解析的值，导致
`pkr --config X stats` 静默忽略 X、退回 config.local.json。
后果是用户可能对着错误的语料跑完整套评测而毫不知情。

两个位置都必须生效，所以两种写法各测一次。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_wiki"


@pytest.fixture()
def marker_config(tmp_path):
    """一个带独特库名的配置，用来确认到底加载了哪个文件。"""
    cfg = tmp_path / "marker.json"
    cfg.write_text(json.dumps({
        "wiki_roots": [{"name": "UNIQUE_MARKER_WIKI", "path": str(SAMPLE)}],
        "embedder": "hash",
    }, ensure_ascii=False), encoding="utf-8")
    return cfg


def _run(args):
    return subprocess.run([sys.executable, "-m", "pkr", *args],
                          capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))


def test_config_before_subcommand_is_honored(marker_config):
    """`pkr --config X stats`：全局选项写在子命令之前。"""
    r = _run(["--config", str(marker_config), "stats"])
    assert r.returncode == 0, r.stderr
    assert "UNIQUE_MARKER_WIKI" in r.stdout, f"--config 被静默忽略了:\n{r.stdout}"


def test_config_after_subcommand_is_honored(marker_config):
    """`pkr stats --config X`：全局选项写在子命令之后。"""
    r = _run(["stats", "--config", str(marker_config)])
    assert r.returncode == 0, r.stderr
    assert "UNIQUE_MARKER_WIKI" in r.stdout, f"--config 被静默忽略了:\n{r.stdout}"


def test_embedder_override_both_positions(marker_config):
    """--embedder 覆盖配置，两个位置都要生效。"""
    for args in (["--embedder", "hash", "stats", "--config", str(marker_config)],
                 ["stats", "--config", str(marker_config), "--embedder", "hash"]):
        r = _run(args)
        assert r.returncode == 0, r.stderr
        assert "embedder = hash" in r.stdout, f"{args} -> {r.stdout}"


def test_never_reads_local_config_when_config_given(marker_config):
    """显式 --config 时绝不能读到 config.local.json（本机可能存在，指向真实语料）。"""
    r = _run(["--config", str(marker_config), "stats"])
    assert r.returncode == 0, r.stderr
    for leaked in ("CS61B", "DS-final", "database"):
        assert leaked not in r.stdout, f"泄漏了本地语料配置: {leaked}"