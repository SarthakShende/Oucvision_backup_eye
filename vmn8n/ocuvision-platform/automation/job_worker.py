import os
import time
import logging
from pathlib import Path

from . import job_database

logging.basicConfig(level=os.environ.get("OCUVISION_LOG_LEVEL", "INFO"))
logger = logging.getLogger("job_worker")

class JobWorker:
    def __init__(self, db: job_database.JobDB):
        self.db = db
        self.max_concurrent = int(os.environ.get("OCUVISION_MAX_CONCURRENT_JOBS", "2"))
        self.stop_flag = False

    def run(self):
        logger.info("Worker started")
        while not self.stop_flag:
            try:
                running = self.db.count_running()
                if running >= self.max_concurrent:
                    time.sleep(2)
                    continue
                queued = self.db.get_queued()
                if not queued:
                    time.sleep(2)
                    continue
                job = queued[0]
                job_id = job["job_id"]
                self.db.update_job(job_id, status="running", started_at=time.time(), progress=5)
                logger.info(f"Executing job {job_id} type={job['job_type']}")
                # Placeholder: real implementations later
                time.sleep(1)
                self.db.update_job(job_id, status="completed", completed_at=time.time(), progress=100, result={"message": "stub execution (not implemented yet)"})
            except Exception as e:
                logger.exception(f"Worker error: {e}")
                time.sleep(5)
