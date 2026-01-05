from __future__ import annotations

import argparse
from typing import Sequence

DEFAULT_WORKER_COUNTS = (2, 4, 6)


def parse_worker_counts(raw: str) -> list[int]:
    """Parse a comma/space separated worker list from the CLI."""
    tokens = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("Expected at least one worker count (example: 2,4,6).")
    try:
        return [int(part) for part in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid worker count in {raw!r}.") from exc


def coerce_worker_counts(value: object) -> list[int]:
    """Coerce a CLI/namespace worker specification into a list of ints."""
    if isinstance(value, str):
        return parse_worker_counts(value)
    if isinstance(value, int):
        return [value]
    if isinstance(value, Sequence):
        counts: list[int] = []
        for entry in value:
            if isinstance(entry, int):
                counts.append(entry)
            elif isinstance(entry, str):
                counts.extend(parse_worker_counts(entry))
            else:
                raise TypeError(f"Unsupported worker count entry: {entry!r}")
        if not counts:
            raise ValueError("Worker count list cannot be empty.")
        return counts
    raise TypeError(f"Unable to parse worker counts from {value!r}")
