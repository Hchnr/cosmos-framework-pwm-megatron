"""Optional media failures must not interrupt scalar upload or lose retries."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "swanlab_bridge_test_subject", Path(__file__).with_name("swanlab_wandb_bridge.py")
)
assert _SPEC is not None and _SPEC.loader is not None
bridge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bridge)


class SwanlabMediaRetryTest(unittest.TestCase):
    def test_missing_image_dependency_preserves_scalars_and_retries_media(self):
        row = {
            "step": 5000,
            "scalars": {"iteration": 5000, "train/loss": 0.2},
            "media": {"preview": {"_type": "image-file", "path": "preview.png"}},
        }
        sent = {}
        errors = []
        with patch.object(bridge, "media_value", side_effect=ImportError("private dependency details")):
            values = bridge.pending_values(row, sent, Path("unused"), errors)
        self.assertEqual(values, row["scalars"])
        self.assertEqual(errors, [{"step": 5000, "key": "preview", "error_type": "ImportError"}])
        sent.update({key: 5000 for key in values})
        marker = object()
        with patch.object(bridge, "media_value", return_value=marker):
            values = bridge.pending_values(row, sent, Path("unused"), [])
        self.assertEqual(values, {"preview": marker})
        sent.update({key: 5000 for key in values})
        with patch.object(bridge, "media_value") as convert:
            self.assertEqual(bridge.pending_values(row, sent, Path("unused"), []), {})
            convert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
