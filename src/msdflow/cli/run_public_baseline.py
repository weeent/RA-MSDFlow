"""端到端执行一个公开源码 baseline 作业。

调用顺序固定为 prepare → 作者 runner → finalize。runner 以独立 Python 进程运行，
避免五个作者仓库中同名 ``datasets``/``models`` 模块互相污染。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from msdflow.baselines.public_protocol import (
    BaselineProtocolConfig,
    finalize_public_baseline,
    prepare_public_baseline,
)
from msdflow.baselines.public_registry import PUBLIC_BASELINES


RUNNERS = {
    "wtflow": "run_wtflow.py",
    "refp": "run_refp.py",
    "msflow": "run_msflow.py",
    "rdpp": "run_rdpp.py",
    "gnl": "run_gnl.py",
}


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("config must contain one JSON object")
    return value


def _runner_arguments(value: Mapping[str, Any]) -> list[str]:
    """把少量显式覆盖转成 argparse 参数，不允许任意 shell 文本。"""

    result: list[str] = []
    for key, raw in value.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(raw, bool):
            if raw:
                result.append(flag)
        elif raw is not None:
            result.extend([flag, str(raw)])
    return result


def run_from_config(config: Mapping[str, Any]) -> Path:
    protocol = BaselineProtocolConfig.from_mapping(config)
    if protocol.method not in RUNNERS:
        raise ValueError(f"no executable runner for {protocol.method}")
    project_root = Path(str(config.get("project_root", "."))).resolve()
    output = Path(protocol.output_directory).resolve()
    prepare_summary = prepare_public_baseline(protocol)
    spec = PUBLIC_BASELINES[protocol.method]
    repository = project_root / spec.local_directory
    runner = project_root / "integrations" / "public_baselines" / RUNNERS[protocol.method]
    if not repository.is_dir():
        raise FileNotFoundError(
            f"missing {repository}; run python3 scripts/fetch_public_baselines.py --root {project_root}"
        )
    command = [
        sys.executable,
        str(runner),
        "--prepare-summary",
        str(prepare_summary),
        "--repository",
        str(repository),
        "--output",
        str(output / "official"),
        "--device",
        str(config.get("device", "cuda")),
        *_runner_arguments(dict(config.get("runner_args", {}))),
    ]
    # 列表参数 + shell=False，路径含空格时也不会被重新解释。
    subprocess.run(command, cwd=project_root, check=True)
    native = output / "official" / "native_predictions.jsonl"
    return finalize_public_baseline(
        prepare_summary_path=prepare_summary,
        native_predictions_path=native,
        output_directory=output / "final",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one public-source baseline end to end")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result = run_from_config(_load(args.config))
    print(json.dumps({"baseline_summary": str(result)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
