"""统一子命令入口：``python3 -m msdflow.cli.main <command> --config ...``。"""

from __future__ import annotations

import argparse
from typing import Sequence


COMMANDS = (
    "train-descriptor",
    "cache",
    "fit-conditions",
    "train-transport",
    "train-normality",
    "infer",
    "diagnose-routing",
    "reference-pilot",
    "reference-otcfm",
    "coral-wtflow",
    "summarize",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MSD-Flow experiment CLI")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    forwarded = ["--config", args.config]
    if args.command == "train-descriptor":
        from .train import main_descriptor
        main_descriptor(forwarded)
    elif args.command == "cache":
        from .prepare import main_cache
        main_cache(forwarded)
    elif args.command == "fit-conditions":
        from .prepare import main_conditions
        main_conditions(forwarded)
    elif args.command == "train-transport":
        from .train import main_transport
        main_transport(forwarded)
    elif args.command == "train-normality":
        from .train import main_normality
        main_normality(forwarded)
    elif args.command == "infer":
        from .infer import main as infer_main
        infer_main(forwarded)
    elif args.command == "diagnose-routing":
        from .diagnose import main as diagnose_main
        diagnose_main(forwarded)
    elif args.command == "reference-pilot":
        from .reference_pilot import main as reference_main
        reference_main(forwarded)
    elif args.command == "reference-otcfm":
        from .reference_otcfm import main as reference_otcfm_main
        reference_otcfm_main(forwarded)
    elif args.command == "coral-wtflow":
        from .coral_wtflow import main as coral_wtflow_main
        coral_wtflow_main(forwarded)
    else:
        from .summarize import main as summarize_main
        summarize_main(forwarded)


if __name__ == "__main__":
    main()
