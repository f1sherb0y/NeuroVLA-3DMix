#!/usr/bin/env python3
"""Validate unified AlphaBrain/LIBERO imports and optionally create one simulator."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import AlphaBrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulator",
        action="store_true",
        help="Create, reset, render, and step one LIBERO environment without loading a policy",
    )
    parser.add_argument("--suite", default="libero_goal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites = benchmark.get_benchmark_dict()
    if args.suite not in suites:
        raise SystemExit(f"Unknown suite {args.suite!r}; available: {sorted(suites)}")

    print(f"AlphaBrain import: {AlphaBrain.__file__}")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"LIBERO bddl_files: {get_libero_path('bddl_files')}")

    if not args.simulator:
        print("LIBERO import smoke test passed")
        return

    task_suite = suites[args.suite]()
    task = task_suite.get_task(0)
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=64,
        camera_widths=64,
    )
    try:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(task_suite.get_task_init_states(0)[0])
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        image = np.asarray(obs["agentview_image"])
        if image.shape[:2] != (64, 64):
            raise RuntimeError(f"Unexpected render shape: {image.shape}")
        print(f"LIBERO simulator smoke test passed: image_shape={image.shape}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
