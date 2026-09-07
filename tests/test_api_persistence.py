import os
import tempfile
import unittest
from pathlib import Path


_temp_dir = tempfile.TemporaryDirectory()
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(_temp_dir.name) / 'jobs.db').as_posix()}"

from pydantic import ValidationError

from app.database import Base, engine
from app.models.schemas import CalculateMetricsRequest
from app.services.job_service import create_job_entry, get_job, update_job


class ApiPersistenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls):
        engine.dispose()
        _temp_dir.cleanup()

    def test_job_survives_new_database_connection(self):
        create_job_entry("job-1", "video-1")
        update_job("job-1", {"status": "COMPLETED", "result": {"ok": True}})
        engine.dispose()

        job = get_job("job-1")
        self.assertEqual(job["status"], "COMPLETED")
        self.assertEqual(job["result"], {"ok": True})

    def test_metrics_request_rejects_invalid_options(self):
        with self.assertRaises(ValidationError):
            CalculateMetricsRequest(unit="cm", pelvis_idxs=[])


if __name__ == "__main__":
    unittest.main()
