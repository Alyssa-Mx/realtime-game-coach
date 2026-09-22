# -*- coding: utf-8 -*-
"""解析 knowledge/decision_tree.md 的缩进树（模块 / 子模块 / 决策项）。

md 是标签体系的唯一出处：playbook 里写的分支路径和决策项在加载时逐条对照这棵树，
对不上直接抛错，不允许代码里出现树上没有的名字。

决策框架原件（4 个模块、20 个子模块、58 条决策项）是同事整理的，不随仓库公开；
仓库里的 decision_tree.md 只列规则实际挂上的 38 条，名称是转述。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FRAMEWORK_MD = Path(__file__).resolve().parent / "knowledge" / "decision_tree.md"


@dataclass(frozen=True)
class Branch:
    module: str
    name: str
    items: tuple

    @property
    def path(self) -> str:
        return f"{self.module}/{self.name}"


def load_framework(md_path: Path = FRAMEWORK_MD) -> list[Branch]:
    out, mod, name, items = [], None, None, []

    def flush():
        nonlocal name, items
        if name is not None:
            out.append(Branch(mod, name, tuple(items)))
        name, items = None, []

    for raw in md_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()
        if not text.startswith("- "):
            continue
        text = text[2:].strip()
        if indent == 0:
            flush(); mod = text
        elif indent == 2:
            flush(); name = text
        else:
            items.append(text)
    flush()
    if not out:
        raise RuntimeError(f"框架文件解析为空: {md_path}")
    return out


BRANCHES = load_framework()
BY_PATH = {b.path: b for b in BRANCHES}
MODULES = list(dict.fromkeys(b.module for b in BRANCHES))


def check_route(path: str, item: str | None) -> None:
    """playbook 加载时用：分支路径必须在树上，决策项必须逐字在该分支下。"""
    b = BY_PATH.get(path)
    if b is None:
        raise KeyError(f"框架里没有分支 {path!r}")
    if item is not None and item not in b.items:
        raise KeyError(f"分支 {path!r} 下没有决策项 {item!r}；可选: {list(b.items)}")


if __name__ == "__main__":
    n_items = sum(len(b.items) for b in BRANCHES)
    print(f"{len(MODULES)} 模块 / {len(BRANCHES)} 分支 / {n_items} 决策项")
    for b in BRANCHES:
        print(f"  {b.path}: {len(b.items)} 项")
