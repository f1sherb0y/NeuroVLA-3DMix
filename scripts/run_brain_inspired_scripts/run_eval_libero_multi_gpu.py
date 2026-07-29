#!/usr/bin/env python3
"""Run NeuroVLA LIBERO evaluation concurrently across multiple GPUs."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path


SUPPORTED_SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10")
TASKS_PER_SUITE = 10


@dataclasses.dataclass(frozen=True)
class Shard:
    index: int
    gpu: str
    suite: str
    task_ids: tuple[int, ...]

    @property
    def task_ids_csv(self) -> str:
        return ",".join(str(task_id) for task_id in self.task_ids)

    @property
    def name(self) -> str:
        first, last = self.task_ids[0], self.task_ids[-1]
        return f"{self.suite}_tasks_{first}-{last}_gpu_{self.gpu}"

    @property
    def label(self) -> str:
        return f"gpu={self.gpu} {self.suite} tasks={self.task_ids_csv}"


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU ID")
    if any(not gpu.isdigit() for gpu in gpus):
        raise ValueError(f"GPU IDs must be non-negative integers: {gpus}")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"Duplicate GPU IDs are not allowed: {gpus}")
    return gpus


def _split_tasks(worker_count: int) -> list[tuple[int, ...]]:
    worker_count = min(worker_count, TASKS_PER_SUITE)
    base, remainder = divmod(TASKS_PER_SUITE, worker_count)
    groups = []
    start = 0
    for worker_index in range(worker_count):
        count = base + (1 if worker_index < remainder else 0)
        groups.append(tuple(range(start, start + count)))
        start += count
    return groups


def build_shard_plan(gpus: list[str], suites: list[str]) -> list[Shard]:
    if len(gpus) < len(suites):
        raise ValueError(
            f"Evaluating {len(suites)} suites concurrently requires at least "
            f"{len(suites)} GPUs; received {len(gpus)}"
        )
    if len(gpus) > TASKS_PER_SUITE * len(suites):
        raise ValueError(
            f"At most {TASKS_PER_SUITE * len(suites)} GPUs can be used for "
            f"{len(suites)} suites without duplicating tasks"
        )

    base, remainder = divmod(len(gpus), len(suites))
    workers_per_suite = [base + (1 if index < remainder else 0) for index in range(len(suites))]

    shards = []
    gpu_index = 0
    for suite, worker_count in zip(suites, workers_per_suite):
        for task_ids in _split_tasks(worker_count):
            shards.append(
                Shard(
                    index=len(shards),
                    gpu=gpus[gpu_index],
                    suite=suite,
                    task_ids=task_ids,
                )
            )
            gpu_index += 1
    return shards


def _result_path(output_root: Path, shard: Shard) -> Path:
    return output_root / "shards" / shard.name / shard.suite / "eval_results.json"


def select_pending_shards(output_root: Path, shards: list[Shard], resume: bool) -> list[Shard]:
    if not resume:
        return shards
    return [shard for shard in shards if not _result_path(output_root, shard).is_file()]


def aggregate_results(
    output_root: Path,
    shards: list[Shard],
    *,
    checkpoint: str,
    trials_per_task: int,
) -> dict:
    task_results: dict[tuple[str, int], dict] = {}

    for shard in shards:
        result_path = _result_path(output_root, shard)
        if not result_path.is_file():
            raise RuntimeError(f"Missing shard result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("task_suite_name") != shard.suite:
            raise RuntimeError(f"Suite mismatch in {result_path}")
        if result.get("task_ids") != list(shard.task_ids):
            raise RuntimeError(f"Task-ID mismatch in {result_path}")

        for task_result in result.get("task_results", []):
            key = (shard.suite, int(task_result["task_id"]))
            if key in task_results:
                raise RuntimeError(f"Duplicate result for {key[0]} task {key[1]}")
            if int(task_result["total_episodes"]) != trials_per_task:
                raise RuntimeError(
                    f"Expected {trials_per_task} episodes for {key[0]} task {key[1]}, "
                    f"found {task_result['total_episodes']}"
                )
            task_results[key] = task_result

    suite_names = list(dict.fromkeys(shard.suite for shard in shards))
    suites_summary = {}
    overall_episodes = 0
    overall_successes = 0

    for suite in suite_names:
        expected_ids = sorted(task_id for shard in shards if shard.suite == suite for task_id in shard.task_ids)
        tasks = []
        for task_id in expected_ids:
            key = (suite, task_id)
            if key not in task_results:
                raise RuntimeError(f"Missing result for {suite} task {task_id}")
            tasks.append(task_results[key])

        episodes = sum(int(task["total_episodes"]) for task in tasks)
        successes = sum(int(task["total_successes"]) for task in tasks)
        suite_summary = {
            "total_episodes": episodes,
            "total_successes": successes,
            "success_rate": successes / episodes,
            "task_results": tasks,
        }
        suites_summary[suite] = suite_summary
        overall_episodes += episodes
        overall_successes += successes

        suite_dir = output_root / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        (suite_dir / "eval_results.json").write_text(
            json.dumps(suite_summary, indent=2), encoding="utf-8"
        )

    summary = {
        "checkpoint": checkpoint,
        "trials_per_task": trials_per_task,
        "gpus": list(dict.fromkeys(shard.gpu for shard in shards)),
        "total_episodes": overall_episodes,
        "total_successes": overall_successes,
        "success_rate": overall_successes / overall_episodes,
        "suites": suites_summary,
    }
    (output_root / "eval_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _stream_output(process: subprocess.Popen[str], log_path: Path, label: str) -> None:
    prefix = f"[{label}] "
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(prefix + line)
            sys.stdout.flush()


def _build_command(
    single_gpu_script: Path,
    shard: Shard,
    args: argparse.Namespace,
    output_root: Path,
) -> list[str]:
    shard_root = output_root / "shards" / shard.name
    command = [
        "bash",
        str(single_gpu_script),
        "--pretrained",
        str(Path(args.pretrained).expanduser().resolve()),
        "--suite",
        shard.suite,
        "--task-ids",
        shard.task_ids_csv,
        "--trials",
        str(args.trials),
        "--seed",
        str(args.seed),
        "--gpu",
        shard.gpu,
        "--video-out",
        str(shard_root),
    ]
    if args.online_stdp:
        command.append("--online-stdp")
    if not args.save_videos:
        command.append("--no-video")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True, help="Self-contained NeuroVLA checkpoint directory")
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU IDs, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--suite",
        default="all",
        choices=("all", *SUPPORTED_SUITES),
        help="LIBERO suite to evaluate (default: all)",
    )
    parser.add_argument("--trials", type=int, default=50, help="Rollouts per task (default: 50)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None, help="Output root directory")
    parser.add_argument("--online-stdp", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed shards in --output")
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save every rollout video; disabled by default for throughput",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise SystemExit("ERROR: --trials must be positive")
    if args.resume and not args.output:
        raise SystemExit("ERROR: --resume requires --output")

    try:
        gpus = parse_gpus(args.gpus)
        suites = list(SUPPORTED_SUITES) if args.suite == "all" else [args.suite]
        shards = build_shard_plan(gpus, suites)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    checkpoint = Path(args.pretrained).expanduser().resolve()
    if not checkpoint.exists():
        raise SystemExit(f"ERROR: checkpoint does not exist: {checkpoint}")

    project_root = Path(__file__).resolve().parents[2]
    single_gpu_script = project_root / "scripts" / "run_brain_inspired_scripts" / "run_eval_libero.sh"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output).expanduser().resolve() if args.output else (
        project_root / "results" / "evaluation" / f"neurovla_multi_gpu_{timestamp}"
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Suites:     {', '.join(suites)}")
    print(f"GPUs:       {', '.join(gpus)}")
    print(f"Shards:     {len(shards)}")
    print(f"Output:     {output_root}")
    print(f"Videos:     {args.save_videos}")
    for shard in shards:
        print(f"  - {shard.label}")

    pending_shards = select_pending_shards(output_root, shards, args.resume)
    commands = [_build_command(single_gpu_script, shard, args, output_root) for shard in pending_shards]
    if args.dry_run:
        for command in commands:
            print(f"Dry-run command: {shlex.join(command)}")
        return

    output_root.mkdir(parents=True, exist_ok=args.resume)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(exist_ok=args.resume)

    if args.resume:
        for shard in shards:
            if shard not in pending_shards:
                print(f"Reusing completed shard: {shard.label}")

    processes: list[tuple[Shard, subprocess.Popen[str], threading.Thread]] = []
    try:
        for shard, command in zip(pending_shards, commands):
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            thread = threading.Thread(
                target=_stream_output,
                args=(process, logs_dir / f"{shard.name}.log", shard.label),
                daemon=True,
            )
            thread.start()
            processes.append((shard, process, thread))

        failures = []
        for shard, process, thread in processes:
            return_code = process.wait()
            thread.join()
            if return_code != 0:
                failures.append((shard, return_code))
    except KeyboardInterrupt:
        print("\nInterrupted; terminating evaluation shards...", file=sys.stderr)
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, thread in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            thread.join(timeout=2)
        raise SystemExit(130)

    if failures:
        for shard, return_code in failures:
            print(f"FAILED: {shard.label} exited with code {return_code}", file=sys.stderr)
        raise SystemExit("One or more evaluation shards failed; aggregate results were not written")

    summary = aggregate_results(
        output_root,
        shards,
        checkpoint=str(checkpoint),
        trials_per_task=args.trials,
    )
    print("\nEvaluation summary")
    for suite, result in summary["suites"].items():
        print(f"  {suite:15s} {result['success_rate'] * 100:6.2f}%")
    print(f"  {'overall':15s} {summary['success_rate'] * 100:6.2f}%")
    print(f"Results: {output_root / 'eval_results.json'}")


if __name__ == "__main__":
    main()
