"""RA-MSDFlow 的 checkpoint 模块。"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import torch


CHECKPOINT_FORMAT_VERSION = 1
SELECTION_METRIC = "min_val_fm_loss"
OPTIMIZER_STATE_STORAGE_DTYPES = ("float32", "float16", "bfloat16")


_ADAM_MOMENT_NAMES = frozenset({"exp_avg", "exp_avg_sq", "max_exp_avg_sq"})


def _copy_to_cpu(value: Any) -> Any:
    """执行 `_copy_to_cpu` 所需的处理。"""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    return value


def pack_optimizer_state_dict(state: Mapping[str, Any], storage_dtype: str) -> dict[str, Any]:
    """执行 `pack_optimizer_state_dict` 所需的处理。"""

    if storage_dtype not in OPTIMIZER_STATE_STORAGE_DTYPES:
        raise ValueError(
            f"optimizer state storage dtype must be one of {OPTIMIZER_STATE_STORAGE_DTYPES}"
        )
    target_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[storage_dtype]
    packed = _copy_to_cpu(dict(state))
    # `pack_optimizer_state_dict` 的实现说明。
    # `pack_optimizer_state_dict` 的实现说明。
    # `pack_optimizer_state_dict` 的实现说明。
    for slots in packed.get("state", {}).values():
        if not isinstance(slots, dict):
            continue
        for name in _ADAM_MOMENT_NAMES:
            value = slots.get(name)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                slots[name] = value.to(dtype=target_dtype)
    return packed


def restore_optimizer_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """执行 `restore_optimizer_state_dict` 所需的处理。"""

    restored = _copy_to_cpu(dict(state))
    for slots in restored.get("state", {}).values():
        if not isinstance(slots, dict):
            continue
        for name in _ADAM_MOMENT_NAMES:
            value = slots.get(name)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                slots[name] = value.to(dtype=torch.float32)
    return restored


def _clone_model_state_to_cpu(state: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """执行 `_clone_model_state_to_cpu` 所需的处理。"""

    output: OrderedDict[str, Any] = OrderedDict()
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.detach().to(device="cpu", copy=True)
        else:
            output[key] = value
    return output


@dataclass(frozen=True, slots=True)
class CheckpointSaveOutcome:
    """`CheckpointSaveOutcome` 组件。"""

    is_best: bool
    wrote_last: bool
    wrote_best: bool
    optimizer_state_storage_dtype: str


@dataclass(frozen=True, slots=True)
class MetricsReconciliationOutcome:
    """`MetricsReconciliationOutcome` 组件。"""

    durable_epoch: int
    discarded_tail_rows: int
    restored_durable_row: bool


@contextmanager
def checkpoint_write_lock(path: str | Path | None):
    """执行 `checkpoint_write_lock` 所需的处理。"""

    if path is None:
        yield
        return
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+b") as handle:
        if target.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def source_tree_digest(source_root: str | Path) -> str:
    """执行 `source_tree_digest` 所需的处理。"""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {root}")
    digest = sha256()
    files = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def detect_git_revision(project_root: str | Path) -> str:
    """执行 `detect_git_revision` 所需的处理。"""

    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(project_root).resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"
    revision = completed.stdout.strip()
    return revision if revision else "unavailable"


def file_digest(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """执行 `file_digest` 所需的处理。"""

    digest = sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_parent_directory(path: Path) -> None:
    """执行 `_fsync_parent_directory` 所需的处理。"""

    if os.name == "nt":
        # 处理 Windows 平台兼容性。
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch_save(state: Mapping[str, Any], output_path: str | Path) -> Path:
    """执行 `atomic_torch_save` 所需的处理。"""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    # 步骤 1：写入检查点。
    if temporary.exists():
        temporary.unlink()
    # 步骤 1：按当前协议处理。
    # `atomic_torch_save` 的实现说明。
    # 保证检查点写入的一致性。
    with temporary.open("wb") as handle:
        torch.save(dict(state), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_parent_directory(target.parent)
    return target


def atomic_write_json(value: Mapping[str, Any], output_path: str | Path) -> Path:
    """执行 `atomic_write_json` 所需的处理。"""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    # 步骤 2：写入检查点。
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_parent_directory(target.parent)
    return target


def load_training_checkpoint(input_path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """执行 `load_training_checkpoint` 所需的处理。"""

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"training checkpoint does not exist: {source}")
    # 步骤 3：加载当前输入。
    try:
        state = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:  # `load_training_checkpoint` 的实现说明。
        state = torch.load(source, map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a dictionary, got {type(state)!r}")
    required = {
        "format_version",
        "checkpoint_role",
        "epoch",
        "global_step",
        "optimizer_update_step",
        "amp_overflow_skipped_steps",
        "model_state_dict",
        "model_state_scope",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "amp_scaler_state_dict",
        "config",
        "provenance",
        "metrics",
        "rng_state",
        "loader_generator_states",
        "best_epoch",
        "best_val_fm_loss",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")
    if int(state["format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format version: {state['format_version']}")
    if state["checkpoint_role"] != "last_resumable":
        raise ValueError("resume requires a last_resumable checkpoint, not a best-weights checkpoint")
    storage_dtype = str(state.get("optimizer_state_storage_dtype", "float32"))
    if storage_dtype not in OPTIMIZER_STATE_STORAGE_DTYPES:
        raise ValueError(f"unsupported optimizer state storage dtype: {storage_dtype!r}")
    # 保证检查点写入的一致性。
    # 保证检查点写入的一致性。
    state["optimizer_state_storage_dtype"] = storage_dtype
    state["optimizer_state_dict"] = restore_optimizer_state_dict(state["optimizer_state_dict"])
    return state


class CheckpointManager:
    """`CheckpointManager` 组件。"""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        selection_metric: str = SELECTION_METRIC,
        min_delta: float = 0.0,
        resume: bool = False,
        write_lock_path: str | Path | None = None,
    ) -> None:
        if selection_metric != SELECTION_METRIC:
            raise ValueError(f"checkpoint selection is fixed to {SELECTION_METRIC!r}")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.run_directory = Path(run_directory).resolve()
        self.last_path = self.run_directory / "checkpoints" / "last.pt"
        self.best_path = self.run_directory / "checkpoints" / "best.pt"
        self.metrics_path = self.run_directory / "logs" / "epoch_metrics.jsonl"
        self.summary_path = self.run_directory / "logs" / "training_summary.json"
        existing = [path for path in (self.last_path, self.best_path, self.metrics_path, self.summary_path) if path.exists()]
        if existing and not resume:
            raise FileExistsError(
                "run directory already contains training artifacts; choose a new run id or resume explicitly: "
                + ", ".join(str(path) for path in existing)
            )
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.selection_metric = selection_metric
        self.min_delta = float(min_delta)
        self.best_val_fm_loss = float("inf")
        self.best_epoch = -1
        self.write_lock_path = None if write_lock_path is None else Path(write_lock_path).resolve()
        # 保证检查点写入的一致性。
        # 保证检查点写入的一致性。
        # `__init__` 的实现说明。
        self._pending_best_state: dict[str, Any] | None = None

    def restore_selection_state(self, checkpoint: Mapping[str, Any]) -> None:
        """执行 `restore_selection_state` 所需的处理。"""

        self.best_val_fm_loss = float(checkpoint["best_val_fm_loss"])
        self.best_epoch = int(checkpoint["best_epoch"])

    def save_epoch(
        self,
        full_state: Mapping[str, Any],
        *,
        validation_loss: float,
        persist_checkpoint: bool,
        optimizer_state_storage_dtype: str,
    ) -> CheckpointSaveOutcome:
        """执行 `save_epoch` 所需的处理。"""

        epoch = int(full_state["epoch"])
        # 步骤 4：按当前协议处理。
        is_best = validation_loss < self.best_val_fm_loss - self.min_delta
        if is_best:
            self.best_val_fm_loss = float(validation_loss)
            self.best_epoch = epoch
        enriched = dict(full_state)
        enriched["best_val_fm_loss"] = self.best_val_fm_loss
        enriched["best_epoch"] = self.best_epoch
        enriched["checkpoint_role"] = "last_resumable"
        if is_best:
            # 步骤 5：按当前协议处理。
            # 处理 CUDA 设备兼容性。
            best_state = {
                key: enriched[key]
                for key in (
                    "format_version",
                    "epoch",
                    "global_step",
                    "optimizer_update_step",
                    "amp_overflow_skipped_steps",
                    "model_state_dict",
                    "model_state_scope",
                    "config",
                    "provenance",
                    "metrics",
                    "best_epoch",
                    "best_val_fm_loss",
                )
            }
            best_state["model_state_dict"] = _clone_model_state_to_cpu(
                enriched["model_state_dict"]
            )
            best_state["checkpoint_role"] = "best_for_evaluation"
            self._pending_best_state = best_state

        wrote_best = False
        wrote_last = False
        if persist_checkpoint:
            # 步骤 6：按当前协议处理。
            # 保证检查点写入的一致性。
            # 保证检查点写入的一致性。
            with checkpoint_write_lock(self.write_lock_path):
                # `save_epoch` 的实现说明。
                # `save_epoch` 的实现说明。
                # 保证检查点写入的一致性。
                if self._pending_best_state is not None:
                    atomic_torch_save(self._pending_best_state, self.best_path)
                    self._pending_best_state = None
                    wrote_best = True
                durable = dict(enriched)
                durable["optimizer_state_storage_dtype"] = optimizer_state_storage_dtype
                durable["optimizer_state_dict"] = pack_optimizer_state_dict(
                    enriched["optimizer_state_dict"], optimizer_state_storage_dtype
                )
                atomic_torch_save(durable, self.last_path)
                wrote_last = True
        return CheckpointSaveOutcome(
            is_best=is_best,
            wrote_last=wrote_last,
            wrote_best=wrote_best,
            optimizer_state_storage_dtype=optimizer_state_storage_dtype,
        )

    def truncate_metrics_after(self, epoch: int) -> int:
        """执行 `truncate_metrics_after` 所需的处理。"""

        if not self.metrics_path.is_file():
            return 0
        kept: list[str] = []
        removed = 0
        for line_number, line in enumerate(
            self.metrics_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid epoch metrics JSONL line {line_number}") from error
            if int(record["epoch"]) <= epoch:
                kept.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
            else:
                removed += 1
        if removed:
            self._durable_replace_metric_lines(kept)
        return removed

    def _durable_replace_metric_lines(self, lines: list[str]) -> None:
        """执行 `_durable_replace_metric_lines` 所需的处理。"""

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.metrics_path.with_suffix(self.metrics_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.metrics_path)
        _fsync_parent_directory(self.metrics_path.parent)

    def reconcile_metrics_to_checkpoint(
        self,
        *,
        durable_epoch: int,
        durable_metrics: Mapping[str, Any],
    ) -> MetricsReconciliationOutcome:
        """执行 `reconcile_metrics_to_checkpoint` 所需的处理。"""

        checkpoint_record = dict(durable_metrics)
        if int(checkpoint_record.get("epoch", -1)) != int(durable_epoch):
            raise ValueError("checkpoint metrics do not match the durable checkpoint epoch")

        records: list[dict[str, Any]] = []
        if self.metrics_path.is_file():
            for line_number, line in enumerate(
                self.metrics_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid epoch metrics JSONL line {line_number}") from error
                if not isinstance(record, dict) or "epoch" not in record:
                    raise ValueError(f"epoch metrics JSONL line {line_number} has no epoch")
                records.append(record)

        # 步骤 7：按当前协议处理。
        # 保证检查点写入的一致性。
        for index, record in enumerate(records):
            if int(record["epoch"]) != index:
                raise ValueError("epoch_metrics.jsonl must contain contiguous epochs starting at zero")

        kept = [record for record in records if int(record["epoch"]) <= durable_epoch]
        discarded = len(records) - len(kept)
        last_logged_epoch = int(kept[-1]["epoch"]) if kept else -1
        restored = False
        if last_logged_epoch < durable_epoch:
            if last_logged_epoch != durable_epoch - 1:
                raise ValueError(
                    "epoch_metrics.jsonl is missing more than the durable checkpoint row; "
                    "the intervening metrics cannot be reconstructed"
                )
            recovery = dict(checkpoint_record)
            recovery["resume_recovery"] = {
                "source": "last.pt:metrics",
                "reason": "epoch_metrics_lagged_durable_checkpoint",
            }
            kept.append(recovery)
            restored = True

        if discarded or restored:
            lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in kept]
            self._durable_replace_metric_lines(lines)
        return MetricsReconciliationOutcome(
            durable_epoch=int(durable_epoch),
            discarded_tail_rows=discarded,
            restored_durable_row=restored,
        )

    def append_epoch_metrics(self, metrics: Mapping[str, Any]) -> None:
        """执行 `append_epoch_metrics` 所需的处理。"""

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(metrics), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            # 步骤 8：按当前协议处理。
            # `append_epoch_metrics` 的实现说明。
            handle.flush()
            os.fsync(handle.fileno())

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        return atomic_write_json(summary, self.summary_path)

    def artifact_summary(self) -> dict[str, object]:
        """执行 `artifact_summary` 所需的处理。"""

        output: dict[str, object] = {}
        for name, path in (
            ("last", self.last_path),
            ("best", self.best_path),
            ("epoch_metrics", self.metrics_path),
            ("training_summary", self.summary_path),
        ):
            output[name] = (
                {"path": str(path), "bytes": path.stat().st_size, "sha256": file_digest(path)}
                if path.is_file()
                else {"path": str(path), "exists": False}
            )
        return output
