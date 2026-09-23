"""按 lock 文件拉取并固定公开 baseline 仓库。

示例：``python3 scripts/fetch_public_baselines.py --root .``。
脚本只处理 git 仓库，不下载模型权重，也不会覆盖已有的脏工作区。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def _run(arguments: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(arguments, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _git(repository: Path, *arguments: str) -> str:
    """执行 `_git` 所需的处理。"""

    return _run(["git", "-c", f"safe.directory={repository.as_posix()}", *arguments], cwd=repository)


def _read_head_without_git(repository: Path) -> str | None:
    """执行 `_read_head_without_git` 所需的处理。"""

    git_dir = repository / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="ascii").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = git_dir / Path(ref)
    if loose.is_file():
        return loose.read_text(encoding="ascii").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        suffix = " " + ref
        for line in packed.read_text(encoding="ascii").splitlines():
            if line.endswith(suffix):
                return line.split(" ", 1)[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pinned public-source anomaly baselines")
    parser.add_argument("--root", default=".", help="project root")
    parser.add_argument("--include-optional", action="store_true", help="also fetch non-main MMR")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    lock_path = root / "configs" / "baselines" / "public_baselines.lock.json"
    with lock_path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)

    for key, spec in lock["baselines"].items():
        if not spec["main_table"] and not args.include_optional:
            continue
        target = root / spec["directory"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "--filter=blob:none", spec["url"], str(target)])
        if not (target / ".git").exists():
            raise RuntimeError(f"{target} exists but is not a git repository")
        # 已经位于固定提交时完全不触碰仓库。这既更快，也兼容只读挂载盘。
        current = _read_head_without_git(target)
        if current == spec["commit"]:
            print(f"{key}: {target} @ {current} (already pinned)")
            continue
        dirty = _git(target, "status", "--porcelain")
        if dirty:
            raise RuntimeError(f"refusing to change dirty third-party repository: {target}")
        _git(target, "fetch", "origin", spec["commit"])
        _git(target, "checkout", "--detach", spec["commit"])
        actual = _git(target, "rev-parse", "HEAD")
        if actual != spec["commit"]:
            raise RuntimeError(f"commit mismatch for {key}: {actual}")
        print(f"{key}: {target} @ {actual}")


if __name__ == "__main__":
    main()
