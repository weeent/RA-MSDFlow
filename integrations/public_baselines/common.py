"""作者仓库 runner 共用的少量文件 I/O。

这些 runner 独立于 ``src/msdflow`` 安装路径运行，便于在不同机器上复现。
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("JSON file must contain one object")
    return value


def read_index(prepare_summary: Mapping[str, object]) -> list[dict]:
    rows: list[dict] = []
    with Path(str(prepare_summary["protocol_index"])).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prediction_rows(index_rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """只返回 reference 与 evaluation；source train 绝不进入推理导出。"""

    return [dict(row) for row in index_rows if row["role"] in {"reference", "evaluation"}]


def rows_by_staged_path(index_rows: Sequence[Mapping[str, object]]) -> dict[str, dict]:
    return {
        str(Path(str(row["staged_image_path"])).resolve()): dict(row)
        for row in prediction_rows(index_rows)
    }


def write_native_predictions(rows: Sequence[Mapping[str, object]], output: str | Path) -> Path:
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def save_map(array: np.ndarray, output_directory: Path, base_id: str) -> str:
    digest = __import__("hashlib").sha256(base_id.encode("utf-8")).hexdigest()[:16]
    target = output_directory / "maps" / f"{digest}.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(array, dtype=np.float32))
    return str(target.resolve())


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def author_repository(path: str | Path) -> Iterator[Path]:
    """暂时把作者仓库置于 import 首位，并在结束后恢复 cwd/sys.path。"""

    repository = Path(path).resolve()
    if not repository.is_dir():
        raise FileNotFoundError(repository)
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        __import__("os").chdir(repository)
        sys.path.insert(0, str(repository))
        yield repository
    finally:
        __import__("os").chdir(old_cwd)
        sys.path[:] = old_path
