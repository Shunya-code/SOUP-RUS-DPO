#!/usr/bin/env python3
"""
Verify that the LoRA adapter actually changed between saved checkpoints.

This is stronger than checking that:
  - loss changed,
  - checkpoint files exist, or
  - final adapter values are non-zero.

It compares the actual adapter tensors in two checkpoints.

Usage:
    python verify_parameter_updates.py \
        /path/to/checkpoint-200 \
        /path/to/checkpoint-339

Optional:
    --min-mean-delta 1e-12
    --min-changed-fraction 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: install safetensors before running this check.", file=sys.stderr)
    raise SystemExit(2)


def load_tensors(path: Path) -> dict[str, object]:
    file_path = path / "adapter_model.safetensors"
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    tensors = {}
    with safe_open(str(file_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).float()
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--min-mean-delta", type=float, default=1e-12)
    parser.add_argument("--min-changed-fraction", type=float, default=0.01)
    args = parser.parse_args()

    before = load_tensors(args.before)
    after = load_tensors(args.after)

    before_keys = set(before)
    after_keys = set(after)

    missing_before = sorted(after_keys - before_keys)
    missing_after = sorted(before_keys - after_keys)

    if missing_before or missing_after:
        print("FAIL: adapter tensor sets differ.")
        if missing_before:
            print("Only in after:", missing_before)
        if missing_after:
            print("Only in before:", missing_after)
        return 1

    total_elements = 0
    changed_elements = 0
    sum_abs_delta = 0.0
    max_abs_delta = 0.0
    changed_tensors = 0

    for key in sorted(before_keys):
        a = before[key]
        b = after[key]

        if a.shape != b.shape:
            print(f"FAIL: shape changed for {key}: {a.shape} -> {b.shape}")
            return 1

        delta = (b - a).abs()
        n = delta.numel()
        changed = int((delta > 0).sum().item())

        total_elements += n
        changed_elements += changed
        sum_abs_delta += float(delta.sum().item())
        max_abs_delta = max(max_abs_delta, float(delta.max().item()))

        if changed:
            changed_tensors += 1

    mean_abs_delta = sum_abs_delta / total_elements if total_elements else 0.0
    changed_fraction = changed_elements / total_elements if total_elements else 0.0

    print("=== LoRA PARAMETER UPDATE VERIFICATION ===")
    print(f"Before checkpoint : {args.before}")
    print(f"After checkpoint  : {args.after}")
    print(f"Tensor count      : {len(before_keys)}")
    print(f"Total parameters  : {total_elements:,}")
    print(f"Changed tensors   : {changed_tensors:,}")
    print(f"Changed elements  : {changed_elements:,}")
    print(f"Changed fraction  : {changed_fraction:.8f}")
    print(f"Mean abs delta    : {mean_abs_delta:.8e}")
    print(f"Max abs delta     : {max_abs_delta:.8e}")

    passed = (
        math.isfinite(mean_abs_delta)
        and math.isfinite(max_abs_delta)
        and mean_abs_delta > args.min_mean_delta
        and changed_fraction >= args.min_changed_fraction
    )

    if passed:
        print("RESULT: PASS — adapter parameters changed materially between checkpoints.")
        return 0

    print("RESULT: FAIL — insufficient evidence of a material adapter update.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
