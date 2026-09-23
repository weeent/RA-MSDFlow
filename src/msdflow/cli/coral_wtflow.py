"""端到端执行 CORAL alignment + 冻结 WT-Flow scorer 消融。"""

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


_ALLOWED_RUNNER_ARGS = {
    "checkpoint",
    "batch_size",
    "workers",
    "image_size",
    "steps",
    "calibration_folds",
    "coral_shrinkage",
    "coral_max_patches",
    "top_fraction",
}


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("config must contain one JSON object")
    return value


def run_from_config(config: Mapping[str, Any]) -> Path:
    """prepare → OOF CORAL+WT-Flow → 统一 finalize；不训练 WT-Flow。"""

    source = dict(config)
    requested_method = str(source.get("method", "coral_wtflow"))
    if requested_method not in {"coral_wtflow", "wtflow"}:
        raise ValueError("coral_wtflow config has an incompatible method")
    # 数据暂存仍使用公开 WT-Flow 的协议与来源记录；最终结果单独标记消融名。
    source["method"] = "wtflow"
    protocol = BaselineProtocolConfig.from_mapping(source)
    project_root = Path(str(config.get("project_root", "."))).resolve()
    output = Path(protocol.output_directory).resolve()
    prepare_summary = prepare_public_baseline(protocol)
    repository = project_root / PUBLIC_BASELINES["wtflow"].local_directory
    runner = project_root / "integrations" / "public_baselines" / "run_coral_wtflow.py"
    if not repository.is_dir() or not runner.is_file():
        raise FileNotFoundError("WT-Flow repository or coral_wtflow runner is missing")

    runner_args = dict(config.get("runner_args", {}))
    unknown = set(runner_args).difference(_ALLOWED_RUNNER_ARGS)
    if unknown:
        raise ValueError(f"unknown coral_wtflow runner args: {sorted(unknown)}")
    checkpoint = runner_args.pop("checkpoint", None)
    if checkpoint in {None, ""}:
        raise ValueError("runner_args.checkpoint must point to an existing source-trained WT-Flow final.pt")
    command = [
        sys.executable,
        str(runner),
        "--prepare-summary", str(prepare_summary),
        "--repository", str(repository),
        "--checkpoint", str(Path(str(checkpoint)).resolve()),
        "--output", str(output / "official"),
        "--device", str(config.get("device", "cuda")),
    ]
    for key, value in runner_args.items():
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    subprocess.run(command, cwd=project_root, check=True)
    native = output / "official" / "native_predictions.jsonl"
    return finalize_public_baseline(
        prepare_summary_path=prepare_summary,
        native_predictions_path=native,
        output_directory=output / "final",
        reported_method="coral_wtflow",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CORAL-aligned frozen WT-Flow ablation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result = run_from_config(_load(args.config))
    print(json.dumps({"ablation_summary": str(result)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
