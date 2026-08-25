import json
import tempfile
import unittest
from pathlib import Path

from hdc.data import load_dataset


class DatasetTests(unittest.TestCase):
    def write_dataset(self, payload):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "dataset.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_labeled_categorical_records(self):
        path = self.write_dataset(
            {
                "training": [
                    {"label": "left", "features": {"color": "red"}},
                    {"label": "right", "features": {"color": "blue"}},
                ],
                "test": [
                    {"label": "left", "features": {"color": "orange"}},
                ],
            }
        )

        training, test = load_dataset(path)

        self.assertEqual(training[0], ("left", {"color": "red"}))
        self.assertEqual(test, [("left", {"color": "orange"})])

    def test_rejects_dataset_without_training_records(self):
        path = self.write_dataset({"training": [], "test": []})

        with self.assertRaisesRegex(ValueError, "training"):
            load_dataset(path)

    def test_rejects_record_without_features(self):
        path = self.write_dataset(
            {
                "training": [{"label": "left"}],
                "test": [{"label": "left", "features": {"color": "red"}}],
            }
        )

        with self.assertRaisesRegex(ValueError, "features"):
            load_dataset(path)


if __name__ == "__main__":
    unittest.main()
