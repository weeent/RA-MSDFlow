"""公开源码 baseline 的准备与收尾入口。

模型训练/推理由作者仓库负责。本入口只处理公共协议，因此不同 baseline 不会
因为各自的数据 loader 或分数范围不同而得到不公平的 reference/test 划分。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from msdflow.baselines.public_protocol import (
    BaselineProtocolConfig,
    finalize_public_baseline,
    prepare_public_baseline,
)


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("config must contain one JSON object")
    return value


def run_prepare(config: Mapping[str, Any]) -> Path:
    """输入一份作业配置，输出 MVTec 目录和 ``protocol_index.jsonl``。"""

    return prepare_public_baseline(BaselineProtocolConfig.from_mapping(config))


def run_finalize(config: Mapping[str, Any]) -> Path:
    """输入准备摘要与作者原始预测，输出校准预测、指标和审计摘要。"""

    for key in ("prepare_summary", "native_predictions"):
        if key not in config or config[key] in {None, ""}:
            raise ValueError(f"finalize config is missing {key!r}")
    return finalize_public_baseline(
        prepare_summary_path=config["prepare_summary"],
        native_predictions_path=config["native_predictions"],
        output_directory=config.get("output_directory"),
        reported_method=config.get("reported_method"),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare/finalize one public-source baseline job")
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = _load(args.config)
    result = run_prepare(config) if args.action == "prepare" else run_finalize(config)
    print(json.dumps({"action": args.action, "summary": str(result)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
