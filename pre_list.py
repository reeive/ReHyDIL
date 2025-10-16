#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split .npy mask names into train/val/test with an 8:1:1 ratio.
- Reads all .npy files under MASK_DIR (optionally recursive)
- Strips the ".npy" suffix
- Writes three .list files: train.list, val.list, test.list

Configuration is hard-coded below.
"""

from pathlib import Path
from typing import List
import random
import os

# =========================
# CONFIG (edit as needed)
# =========================
MASK_DIR: str   = " "   # directory containing .npy masks
RECURSIVE: bool = True               # search subdirectories if True
STRIP_SUFFIX: bool = True            # remove ".npy" from names
OUT_DIR: str    = "lists"            # where to save the .list files
PREFIX: str     = ""                 # optional filename prefix, e.g., "t1_"
SEED: int       = 1111               # random seed for deterministic split
TRAIN_NAME: str = "train.list"
VAL_NAME: str   = "val.list"
TEST_NAME: str  = "test.list"
# =========================


def collect_from_dir(mask_dir: Path, recursive: bool = True, strip_suffix: bool = True) -> List[str]:
    """Collect unique basenames from a directory containing .npy files."""
    pattern = "**/*.npy" if recursive else "*.npy"
    items: List[str] = []
    seen = set()
    for p in sorted(mask_dir.glob(pattern)):
        if not p.is_file():
            continue
        name = p.stem if strip_suffix else p.name
        if name not in seen:
            seen.add(name)
            items.append(name)
    return items


def split_8_1_1(names: List[str], seed: int = 1111) -> tuple[List[str], List[str], List[str]]:
    """Split into 80% train, 10% val, 10% test (deterministic)."""
    rng = random.Random(seed)
    base = sorted(names)   # stable order before shuffling
    rng.shuffle(base)

    n = len(base)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val  # remainder goes to test

    train = base[:n_train]
    val = base[n_train:n_train + n_val]
    test = base[n_train + n_val:]
    return train, val, test


def write_list(path: Path, items: List[str]) -> None:
    """Write items to a .list file (one per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items), encoding="utf-8")


def main():
    mask_dir = Path(MASK_DIR).expanduser().resolve()
    assert mask_dir.exists(), f"Directory does not exist: {mask_dir}"

    names = collect_from_dir(mask_dir, recursive=RECURSIVE, strip_suffix=STRIP_SUFFIX)
    assert len(names) > 0, "No .npy files found."

    train, val, test = split_8_1_1(names, seed=SEED)

    out_dir = Path(OUT_DIR).expanduser().resolve()
    write_list(out_dir / f"{PREFIX}{TRAIN_NAME}", train)
    write_list(out_dir / f"{PREFIX}{VAL_NAME}",   val)
    write_list(out_dir / f"{PREFIX}{TEST_NAME}",  test)

    print(f"[OK] total={len(names)}  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Saved to: {out_dir}")
    print(f"Files:")
    print(f"  {out_dir / (PREFIX + TRAIN_NAME)}")
    print(f"  {out_dir / (PREFIX + VAL_NAME)}")
    print(f"  {out_dir / (PREFIX + TEST_NAME)}")


if __name__ == "__main__":
    main()
