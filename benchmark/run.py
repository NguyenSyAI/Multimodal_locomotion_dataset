#!/usr/bin/env python3
from __future__ import annotations

import argparse

from config import BASELINES, DEFAULT_EPOCHS, DEFAULT_NUM_WORKERS, get_spec
from data import prepare_cache
from train import train_loso


def parse_args():
    parser = argparse.ArgumentParser(
        description="LOSO multi-task terrain baselines: terrain + transition presence + transition timing"
    )
    parser.add_argument(
        "--baseline", required=True,
        choices=list(BASELINES) + ["shank", "chest", "two_imu"],
    )
    parser.add_argument("--mode", required=True, choices=["prepare", "train", "all"])
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--test-subject", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = get_spec(args.baseline)
    batch_size = args.batch_size or spec.batch_size

    if args.mode in {"prepare", "all"}:
        prepare_cache(spec, overwrite=args.overwrite_cache)
    if args.mode in {"train", "all"}:
        train_loso(
            spec,
            only_test_subject=args.test_subject,
            epochs=args.epochs,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
