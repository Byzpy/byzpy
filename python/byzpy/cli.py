from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import platform
from typing import Iterable, List, Sequence, Type

from . import __version__


def _load_subclasses(package: str, base: Type) -> List[str]:
    """Return fully-qualified class names defined under ``package``."""
    module = importlib.import_module(package)
    if not hasattr(module, "__path__"):
        return []
    results = set()
    for mod_info in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
        # Skip test modules
        if "tests" in mod_info.name.split("."):
            continue

        try:
            mod = importlib.import_module(mod_info.name)
        except ImportError:
            continue

        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, base) and obj is not base:
                # Only include the class if it is defined in this module
                if obj.__module__ == mod.__name__:
                    results.add(f"{mod.__name__}.{name}")
    return sorted(results)


def _doctor() -> dict:
    """Collect a lightweight status report about optional runtime features."""
    report = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": {"available": False, "cuda": False},
        "cupy": {"available": False},
        "ucx": {"available": False},
    }

    try:
        import torch

        report["torch"]["available"] = True
        report["torch"]["version"] = torch.__version__
        report["torch"]["cuda"] = bool(torch.cuda.is_available())
        if report["torch"]["cuda"]:
            report["torch"]["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - informational only
        report["torch"]["error"] = str(exc)

    try:
        import cupy  # type: ignore

        report["cupy"]["available"] = True
        report["cupy"]["version"] = cupy.__version__  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        report["cupy"]["error"] = str(exc)

    try:
        import byzpy.engine.actor.transports.ucx as ucx_mod

        report["ucx"]["available"] = ucx_mod.have_ucx()
    except Exception as exc:  # pragma: no cover
        report["ucx"]["error"] = str(exc)

    return report


def _cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    data = _doctor()
    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_val in value.items():
                    print(f"  - {sub_key}: {sub_val}")
            else:
                print(f"{key}: {value}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    target = args.component
    if target == "aggregators":
        from byzpy.aggregators.base import Aggregator as _Base

        values = _load_subclasses("byzpy.aggregators", _Base)
    elif target == "attacks":
        from byzpy.attacks.base import Attack as _Base

        values = _load_subclasses("byzpy.attacks", _Base)
    else:
        from byzpy.pre_aggregators.base import PreAggregator as _Base

        values = _load_subclasses("byzpy.pre_aggregators", _Base)

    if args.format == "json":
        print(json.dumps({"component": target, "items": values}, indent=2))
    else:
        if not values:
            print(f"No {target} found.")
        for name in values:
            print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="byzpy",
        description="Utilities for inspecting ByzPy installations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print the installed ByzPy version.")
    version_parser.set_defaults(func=_cmd_version)

    doctor_parser = subparsers.add_parser("doctor", help="Diagnose local dependencies.")
    doctor_parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="Choose output format (default: human-readable).",
    )
    doctor_parser.set_defaults(func=_cmd_doctor)

    list_parser = subparsers.add_parser(
        "list",
        help="List built-in components (aggregators, attacks, pre-aggregators).",
    )
    list_parser.add_argument(
        "component",
        choices=("aggregators", "attacks", "pre-aggregators"),
        help="Component family to inspect.",
    )
    list_parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="Choose output format (default: human-readable).",
    )
    list_parser.set_defaults(func=_cmd_list)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
