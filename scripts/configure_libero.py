#!/usr/bin/env python3
"""Create a non-interactive LIBERO path config for AlphaBrain evaluation."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import yaml


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} directory does not exist: {path}")


def build_path_config(libero_home: Path, datasets_dir: Path) -> dict[str, str]:
    benchmark_root = libero_home / "libero" / "libero"
    required = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "assets": benchmark_root / "assets",
    }
    for label, path in required.items():
        _require_directory(path, f"LIBERO {label}")

    datasets_dir.mkdir(parents=True, exist_ok=True)
    return {**{key: str(path) for key, path in required.items()}, "datasets": str(datasets_dir)}


def write_env_file(
    path: Path,
    *,
    libero_home: Path,
    config_dir: Path,
    env_name: str,
    mujoco_gl: str,
) -> None:
    values = {
        "CONDA_ENV": env_name,
        "LIBERO_HOME": str(libero_home),
        "LIBERO_CONFIG_PATH": str(config_dir),
        "LIBERO_PYTHON": sys.executable,
        "MUJOCO_GL": mujoco_gl,
        "NEUROVLA_PYTHON": sys.executable,
    }
    contents = "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-home", required=True, help="LIBERO repository root")
    parser.add_argument(
        "--config-dir",
        default=str(project_root / ".libero"),
        help="Directory that will contain LIBERO config.yaml",
    )
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Optional native LIBERO dataset directory (not LeRobot training data)",
    )
    parser.add_argument(
        "--env-file",
        default=str(project_root / ".env.libero"),
        help="Shell environment file consumed by run_eval_libero.sh",
    )
    parser.add_argument("--env-name", default="neurovla", help="Unified Conda environment name")
    parser.add_argument("--mujoco-gl", default="egl", choices=("egl", "glfw", "osmesa"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    libero_home = _absolute(args.libero_home)
    config_dir = _absolute(args.config_dir)
    datasets_dir = _absolute(args.datasets_dir or libero_home / "libero" / "datasets")
    env_file = _absolute(args.env_file)

    _require_directory(libero_home, "LIBERO home")
    path_config = build_path_config(libero_home, datasets_dir)

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(path_config, sort_keys=True), encoding="utf-8")
    write_env_file(
        env_file,
        libero_home=libero_home,
        config_dir=config_dir,
        env_name=args.env_name,
        mujoco_gl=args.mujoco_gl,
    )

    print(f"LIBERO config: {config_path}")
    print(f"Evaluation environment: {env_file}")


if __name__ == "__main__":
    main()
