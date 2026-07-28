import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LiberoSetupTests(unittest.TestCase):
    def test_pytorch_26_patch_changes_libero_init_state_loader(self):
        patch = (PROJECT_ROOT / "patches" / "libero-pytorch-2.6.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch.load(init_states_path, weights_only=False)", patch)

    def test_configure_libero_writes_noninteractive_config_and_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            libero_home = root / "LIBERO"
            benchmark_root = libero_home / "libero" / "libero"
            for name in ("bddl_files", "init_files", "assets"):
                (benchmark_root / name).mkdir(parents=True, exist_ok=True)

            config_dir = root / "config"
            env_file = root / ".env.libero"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "configure_libero.py"),
                    "--libero-home",
                    str(libero_home),
                    "--config-dir",
                    str(config_dir),
                    "--env-file",
                    str(env_file),
                ],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            config = yaml.safe_load((config_dir / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["benchmark_root"], str(benchmark_root))
            self.assertEqual(config["bddl_files"], str(benchmark_root / "bddl_files"))
            self.assertEqual(config["init_states"], str(benchmark_root / "init_files"))
            self.assertEqual(config["assets"], str(benchmark_root / "assets"))
            self.assertTrue(Path(config["datasets"]).is_dir())

            env_text = env_file.read_text(encoding="utf-8")
            self.assertIn("CONDA_ENV=neurovla", env_text)
            self.assertIn(f"LIBERO_HOME={libero_home}", env_text)
            self.assertIn(f"LIBERO_CONFIG_PATH={config_dir}", env_text)
            self.assertIn("MUJOCO_GL=egl", env_text)
            self.assertIn(f"NEUROVLA_PYTHON={sys.executable}", env_text)

    def test_shell_scripts_parse(self):
        for relative_path in (
            "scripts/download_neurovla_checkpoint.sh",
            "scripts/download_vggt_checkpoint.sh",
            "scripts/run_brain_inspired_scripts/run_neurovla_3d_mix_pretrain.sh",
            "scripts/setup_neurovla_env.sh",
            "scripts/run_brain_inspired_scripts/run_eval_libero.sh",
        ):
            subprocess.run(
                ["bash", "-n", str(PROJECT_ROOT / relative_path)],
                check=True,
                env=os.environ.copy(),
            )

    def test_checkpoint_download_dry_run_uses_home_models_without_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env["HOME"] = temp_dir
            result = subprocess.run(
                [
                    "bash",
                    str(PROJECT_ROOT / "scripts" / "download_neurovla_checkpoint.sh"),
                    "--dry-run",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

            expected = Path(temp_dir) / "models" / "neurovla-libero-all4suite"
            self.assertIn("AlphaBrainGroup/neurovla-libero-all4suite", result.stdout)
            self.assertIn(str(expected), result.stdout)
            self.assertFalse(expected.exists())

    def test_checkpoint_download_streams_conda_output(self):
        script = (
            PROJECT_ROOT / "scripts" / "download_neurovla_checkpoint.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("conda run --no-capture-output", script)
        self.assertIn("unset HF_HUB_DISABLE_PROGRESS_BARS", script)

    def test_vggt_download_is_pinned_and_streams_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    "bash",
                    str(PROJECT_ROOT / "scripts" / "download_vggt_checkpoint.sh"),
                    "--models-dir",
                    temp_dir,
                    "--dry-run",
                ],
                check=True,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
            )

            target = Path(temp_dir) / "VGGT-1B"
            self.assertIn("facebook/VGGT-1B", result.stdout)
            self.assertIn("860abec7937da0a4c03c41d3c269c366e82abdf9", result.stdout)
            self.assertIn(str(target), result.stdout)
            self.assertFalse(target.exists())

        script = (PROJECT_ROOT / "scripts" / "download_vggt_checkpoint.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("conda run --no-capture-output", script)
        self.assertIn("unset HF_HUB_DISABLE_PROGRESS_BARS", script)


if __name__ == "__main__":
    unittest.main()
