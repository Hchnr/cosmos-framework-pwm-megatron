"""Restart discovery must distinguish directories even when W&B reuses its ID."""

import tempfile
import unittest
from pathlib import Path

from tools.swanlab_sidecar import discover, legacy_state_dir, verified_worker


class DiscoveryTest(unittest.TestCase):
    def test_manual_restart_reusing_wandb_id_selects_new_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wandb = root / "cosmos3_action/action128/job/wandb"
            old = wandb / "offline-run-20261031_080946-66a84b09/run-66a84b09.wandb"
            new = wandb / "offline-run-20261031_125000-66a84b09/run-66a84b09.wandb"
            for path in [old, new]:
                path.parent.mkdir(parents=True)
                path.write_bytes(b"x" * 256)
            self.assertEqual(discover(root), new.resolve())

    def test_validation_runs_under_logs_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / "cosmos3_action/action128/job/wandb/offline-run-20261031_080946-abc/run-abc.wandb"
            validation = root / "logs/test/wandb/offline-run-20261101_120000-xyz/run-xyz.wandb"
            for path in [real, validation]:
                path.parent.mkdir(parents=True)
                path.write_bytes(b"x" * 256)
            self.assertEqual(discover(root), real.resolve())

    def test_legacy_adoption_requires_matching_source(self):
        import json

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "logs/launch_old/monitor/swanlab_bridge"
            state.mkdir(parents=True)
            source = (root / "run.wandb").resolve()
            (state / "online_state.json").write_text("{}")
            (state / "extraction.json").write_text(json.dumps({"source": str(source)}))
            self.assertEqual(legacy_state_dir(root, source), state.resolve())
            self.assertIsNone(legacy_state_dir(root, root / "different.wandb"))
            self.assertFalse(verified_worker(None, source, state))


if __name__ == "__main__":
    unittest.main()
