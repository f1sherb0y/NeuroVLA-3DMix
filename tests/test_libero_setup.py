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
            "scripts/setup_neurovla_env.sh",
            "scripts/run_brain_inspired_scripts/run_eval_libero.sh",
        ):
            subprocess.run(
                ["bash", "-n", str(PROJECT_ROOT / relative_path)],
                check=True,
                env=os.environ.copy(),
            )


if __name__ == "__main__":
    unittest.main()
