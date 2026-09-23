"""在多张 GPU 上调度 OT-CFM 主表作业并合并结果。

矩阵采用显式 ``jobs``，因为 RobustAD 各类别的目标环境不同，MVTec AD 2 各类别
又需要更少 reference shots。每个子进程只看见一张 GPU，配置内始终写 ``cuda:0``。
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from envfm.training.checkpoint import atomic_write_json


def _format_tree(value: Any, fields: Mapping[str, object]) -> Any:
    if isinstance(value, str):
        return value.format(**fields)
    if isinstance(value, list):
        return [_format_tree(item, fields) for item in value]
    if isinstance(value, dict):
        return {key: _format_tree(item, fields) for key, item in value.items()}
    return value


def expand_matrix(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把共享 base 与显式类别 jobs 合并为逐进程配置。"""

    base = matrix.get("base")
    jobs = matrix.get("jobs")
    if not isinstance(base, dict) or not isinstance(jobs, list) or not jobs:
        raise ValueError("OT-CFM matrix requires a base object and non-empty jobs list")
    expanded: list[dict[str, Any]] = []
    for raw in jobs:
        if not isinstance(raw, dict):
            raise TypeError("each matrix job must be an object")
        required = {"dataset", "category", "manifest", "inference_bundle", "target_domains", "output_directory"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"matrix job misses fields: {sorted(missing)}")
        seeds = raw.get("seeds", [raw.get("train_seed", base.get("train_seed", 9827))])
        for seed in seeds:
            fields = {
                "dataset": raw["dataset"],
                "category": raw["category"],
                # 主表的重复实验同时改变 reference draw 与优化器初始化。
                "train_seed": int(seed),
                "reference_seed": int(raw.get("reference_seed", seed)),
            }
            config = deepcopy(base)
            config.update({key: value for key, value in raw.items() if key != "seeds"})
            config.update(fields)
            expanded.append(_format_tree(config, fields))
    outputs = [str(config["output_directory"]) for config in expanded]
    if len(outputs) != len(set(outputs)):
        raise ValueError("matrix contains duplicate output directories")
    return expanded


def materialize(matrix_path: str | Path, output_directory: str | Path) -> tuple[list[Path], dict[str, object]]:
    source = Path(matrix_path).resolve()
    matrix = json.loads(source.read_text(encoding="utf-8"))
    jobs = expand_matrix(matrix)
    output = Path(output_directory).resolve()
    config_directory = output / "configs"
    log_directory = output / "logs"
    config_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    records: list[dict[str, object]] = []
    for index, job in enumerate(jobs):
        job_id = f"{index:03d}_{job['dataset']}_{job['category']}_seed{job['train_seed']}"
        path = config_directory / f"{job_id}.json"
        atomic_write_json(job, path)
        paths.append(path)
        records.append(
            {
                "job_id": job_id,
                "config": str(path),
                "output_directory": str(Path(job["output_directory"]).resolve()),
            }
        )
    plan: dict[str, object] = {
        "phase": "reference_otcfm_main_table_plan",
        "matrix": str(source),
        "gpu_ids": list(matrix.get("gpu_ids", [0])),
        "jobs": records,
        "job_count": len(records),
    }
    atomic_write_json(plan, output / "plan.json")
    return paths, plan


def _write_metric_rows(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """写入一组长表记录；没有记录时不制造空 CSV。"""

    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _merge_metrics(plan: Mapping[str, object], output: Path) -> Path:
    """合并逐作业指标，并把论文主方法与内部组件对照分开。

    ``main_table_long.csv`` 只保留本论文的完整 ``reference_otcfm``。第三方
    baseline 由各自官方实现产生结果后再填入论文主表。``frozen_source``、
    ``target_threshold_only`` 与 ``coral`` 只回答组件是否有效，因此单独写入
    ``component_controls_long.csv``，避免误将它们当成主表竞争方法。
    """

    rows: list[dict[str, object]] = []
    for job in plan["jobs"]:  # type: ignore[index]
        record = dict(job)
        metrics_path = Path(str(record["output_directory"])) / "metrics.csv"
        if not metrics_path.is_file():
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for metric in csv.DictReader(handle):
                rows.append({"job_id": record["job_id"], **metric})
    result = output / "main_table_long.csv"
    main_rows = [row for row in rows if row.get("method") == "reference_otcfm"]
    control_rows = [row for row in rows if row.get("method") != "reference_otcfm"]
    _write_metric_rows(main_rows, result)
    _write_metric_rows(control_rows, output / "component_controls_long.csv")
    return result


def execute(paths: Sequence[Path], plan: Mapping[str, object], output: Path) -> None:
    """每张卡保持一个进程；已存在完整 summary 的作业直接跳过。"""

    gpu_ids = [str(value) for value in plan["gpu_ids"]]  # type: ignore[index]
    pending = list(enumerate(paths))
    records = list(plan["jobs"])  # type: ignore[arg-type]
    running: dict[str, tuple[int, subprocess.Popen[bytes], object, Path]] = {}
    completed: list[dict[str, object]] = []
    project = Path(__file__).resolve().parents[3]
    while pending or running:
        for gpu in gpu_ids:
            if gpu in running or not pending:
                continue
            index, config_path = pending.pop(0)
            summary = Path(str(records[index]["output_directory"])) / "otcfm_summary.json"
            if summary.is_file():
                completed.append({"job_index": index, "gpu_id": None, "return_code": 0, "skipped_complete": True})
                continue
            log_path = output / "logs" / f"job_{index:03d}.log"
            handle = log_path.open("wb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONPATH"] = str(project / "src")
            process = subprocess.Popen(
                [sys.executable, "-m", "msdflow.cli.main", "reference-otcfm", "--config", str(config_path)],
                cwd=project,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (index, process, handle, log_path)
        for gpu, (index, process, handle, log_path) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()  # type: ignore[union-attr]
            completed.append(
                {"job_index": index, "gpu_id": gpu, "return_code": return_code, "log": str(log_path)}
            )
            del running[gpu]
        atomic_write_json(
            {"completed": completed, "pending": len(pending), "running_gpu_ids": sorted(running)},
            output / "status.json",
        )
        if pending or running:
            time.sleep(1.0)
    failures = [record for record in completed if int(record["return_code"]) != 0]
    _merge_metrics(plan, output)
    if failures:
        raise RuntimeError(f"{len(failures)} OT-CFM jobs failed; inspect status.json and logs")


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plan or run reference-alignment jobs")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_directory).resolve()
    paths, plan = materialize(args.matrix, output)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.execute:
        execute(paths, plan, output)


if __name__ == "__main__":
    main()
