import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_brain_inspired_scripts.run_eval_libero_multi_gpu import (
    SUPPORTED_SUITES,
    _result_path,
    aggregate_results,
    build_shard_plan,
    select_pending_shards,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "run_brain_inspired_scripts" / "run_eval_libero_multi_gpu.py"


class LiberoMultiGpuTests(unittest.TestCase):
    def test_eight_gpu_all_suite_plan(self):
        shards = build_shard_plan([str(index) for index in range(8)], list(SUPPORTED_SUITES))

        self.assertEqual(len(shards), 8)
        for suite_index, suite in enumerate(SUPPORTED_SUITES):
            suite_shards = [shard for shard in shards if shard.suite == suite]
            self.assertEqual([shard.gpu for shard in suite_shards], [str(suite_index * 2), str(suite_index * 2 + 1)])
            self.assertEqual(suite_shards[0].task_ids, tuple(range(5)))
            self.assertEqual(suite_shards[1].task_ids, tuple(range(5, 10)))

    def test_aggregate_results_combines_all_tasks(self):
        shards = build_shard_plan(["0", "1", "2", "3"], list(SUPPORTED_SUITES))
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            for shard in shards:
                result_path = _result_path(output_root, shard)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                task_results = []
                for task_id in shard.task_ids:
                    successes = task_id % 3
                    task_results.append(
                        {
                            "task_id": task_id,
                            "task_description": f"task {task_id}",
                            "total_episodes": 2,
                            "total_successes": successes,
                            "success_rate": successes / 2,
                            "success_history": [1.0] * successes + [0.0] * (2 - successes),
                        }
                    )
                result_path.write_text(
                    json.dumps(
                        {
                            "task_suite_name": shard.suite,
                            "task_ids": list(shard.task_ids),
                            "task_results": task_results,
                        }
                    ),
                    encoding="utf-8",
                )

            summary = aggregate_results(
                output_root,
                shards,
                checkpoint="/models/neurovla",
                trials_per_task=2,
            )

            self.assertEqual(summary["total_episodes"], 80)
            self.assertEqual(set(summary["suites"]), set(SUPPORTED_SUITES))
            self.assertTrue((output_root / "eval_results.json").is_file())
            for suite in SUPPORTED_SUITES:
                self.assertEqual(len(summary["suites"][suite]["task_results"]), 10)

    def test_dry_run_prints_eight_commands_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            output = root / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "--pretrained",
                    str(checkpoint),
                    "--gpus",
                    "0,1,2,3,4,5,6,7",
                    "--suite",
                    "all",
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.stdout.count("Dry-run command:"), 8)
            self.assertIn("--task-ids 0,1,2,3,4", result.stdout)
            self.assertIn("--task-ids 5,6,7,8,9", result.stdout)
            self.assertIn("--no-video", result.stdout)
            self.assertFalse(output.exists())

    def test_resume_skips_completed_shards(self):
        shards = build_shard_plan([str(index) for index in range(8)], list(SUPPORTED_SUITES))
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            completed_result = _result_path(output_root, shards[0])
            completed_result.parent.mkdir(parents=True)
            completed_result.write_text("{}", encoding="utf-8")

            pending = select_pending_shards(output_root, shards, resume=True)

        self.assertEqual(pending, shards[1:])


if __name__ == "__main__":
    unittest.main()
