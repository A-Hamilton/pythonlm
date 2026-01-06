"""Summarize evaluation results across checkpoints.

Scans a directory of JSON evaluation outputs, selects the checkpoint
with the highest pass@1 score, and prints a concise report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PASS_KEYS = ("pass@1", "pass_at_1", "pass1", "pass_at1")
EPOCH_KEYS = ("epoch", "checkpoint_epoch", "ckpt_epoch", "data_epoch")
DECODE_KEYS = ("decode", "decode_settings", "generation", "sampling", "decoding")

_MISSING = object()


@dataclass(frozen=True)
class EvalResult:
    path: Path
    epoch: int | None
    pass_at_1: float
    decode_settings: Any


def _find_value(data: Any, keys: Iterable[str]) -> Any:
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        for value in data.values():
            found = _find_value(value, keys)
            if found is not _MISSING:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_value(item, keys)
            if found is not _MISSING:
                return found
    return _MISSING


def _parse_epoch(path: Path, data: Any) -> int | None:
    raw_epoch = _find_value(data, EPOCH_KEYS)
    if raw_epoch is not _MISSING:
        try:
            return int(raw_epoch)
        except (TypeError, ValueError):
            return None
    for part in path.stem.split("_"):
        if part.isdigit():
            return int(part)
    return None


def _normalize_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            return float(stripped.strip("%")) / 100.0
        return float(stripped)
    return None


def _load_eval_file(path: Path) -> EvalResult | None:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    raw_pass = _find_value(data, PASS_KEYS)
    if raw_pass is _MISSING:
        return None
    pass_at_1 = _normalize_score(raw_pass)
    if pass_at_1 is None:
        return None
    epoch = _parse_epoch(path, data)
    decode_settings = _find_value(data, DECODE_KEYS)
    if decode_settings is _MISSING:
        decode_settings = None
    return EvalResult(path=path, epoch=epoch, pass_at_1=pass_at_1, decode_settings=decode_settings)


def _iter_results(results_dir: Path, pattern: str) -> list[EvalResult]:
    results: list[EvalResult] = []
    for path in results_dir.rglob(pattern):
        if not path.is_file():
            continue
        try:
            result = _load_eval_file(path)
        except json.JSONDecodeError:
            continue
        if result is not None:
            results.append(result)
    return results


def _render_decode_settings(decode_settings: Any) -> str:
    if decode_settings is None:
        return "(not found)"
    return json.dumps(decode_settings, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the best checkpoint by pass@1 and summarize evaluation settings.",
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="eval_results",
        help="Directory containing JSON evaluation outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for JSON files (default: *.json).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary report as JSON.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 2

    results = _iter_results(results_dir, args.pattern)
    if not results:
        print("No evaluation results with pass@1 found.", file=sys.stderr)
        return 1

    best = max(results, key=lambda item: item.pass_at_1)

    report = {
        "best_checkpoint": str(best.path),
        "best_epoch": best.epoch,
        "pass@1": best.pass_at_1,
        "decode_settings": best.decode_settings,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("Best checkpoint summary")
    print("=" * 24)
    print(f"Path: {best.path}")
    print(f"Epoch: {best.epoch if best.epoch is not None else 'unknown'}")
    print(f"Pass@1: {best.pass_at_1:.4f}")
    print("Decode settings:")
    print(_render_decode_settings(best.decode_settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
