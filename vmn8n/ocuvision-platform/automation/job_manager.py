from . import job_database, job_worker

class JobManager:
    def __init__(self):
        self.db = job_database.JobDB()
        self.worker = job_worker.JobWorker(self.db)

    def start(self):
        self.worker.run()
