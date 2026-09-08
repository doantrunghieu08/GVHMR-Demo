import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_old_yolo_config = os.environ.get("YOLO_CONFIG_DIR")
_yolo_config = tempfile.TemporaryDirectory()
os.environ["YOLO_CONFIG_DIR"] = _yolo_config.name

from tools.demo import demo

if _old_yolo_config is None:
    os.environ.pop("YOLO_CONFIG_DIR")
else:
    os.environ["YOLO_CONFIG_DIR"] = _old_yolo_config


class VitPosePrecomputeTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        _yolo_config.cleanup()

    def test_reuses_one_extractor_for_all_people(self):
        extractors = []
        calls = []

        class FakeExtractor:
            def __init__(self):
                extractors.append(self)

        def fake_run(cfg, bbx_xyxy, person_id, extractor):
            calls.append((person_id, bbx_xyxy, extractor))

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            demo,
            "get_paths_for_person",
            side_effect=lambda cfg, person_id: {"vitpose": str(Path(temp_dir) / f"vitpose_p{person_id}.pt")},
        ), patch.object(demo, "VitPoseExtractor", FakeExtractor), patch.object(
            demo, "run_vitpose", side_effect=fake_run
        ), patch.object(
            demo.torch.cuda, "empty_cache"
        ) as empty_cache:
            tracks = [object(), object()]
            demo.precompute_vitpose(SimpleNamespace(), tracks)

        self.assertEqual(len(extractors), 1)
        self.assertEqual([call[:2] for call in calls], list(enumerate(tracks)))
        self.assertTrue(all(call[2] is extractors[0] for call in calls))
        empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
