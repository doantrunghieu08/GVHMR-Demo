import os
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

import app.config as config
from app.database import Base, engine
from app.models.schemas import CalculateMetricsRequest
from app.services.job_service import create_job_entry, get_job, update_job


class ApiPersistenceTest(unittest.TestCase):
    def test_job_in_memory(self):
        config.USE_SQL = False
        create_job_entry("job-mem-1", "video-1")
        update_job("job-mem-1", {"status": "COMPLETED", "result": {"ok": True}})

        job = get_job("job-mem-1")
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "COMPLETED")
        self.assertEqual(job["result"], {"ok": True})

    @unittest.skipIf(Base is None, "SQLAlchemy is not installed")
    def test_job_sql_backend(self):
        config.USE_SQL = True
        Base.metadata.create_all(bind=engine)
        create_job_entry("job-sql-1", "video-2")
        update_job("job-sql-1", {"status": "COMPLETED", "result": {"ok": True}})
        engine.dispose()

        job = get_job("job-sql-1")
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "COMPLETED")
        self.assertEqual(job["result"], {"ok": True})

    def test_metrics_request_rejects_invalid_options(self):
        with self.assertRaises(ValidationError):
            CalculateMetricsRequest(unit="cm", pelvis_idxs=[])


if __name__ == "__main__":
    unittest.main()
