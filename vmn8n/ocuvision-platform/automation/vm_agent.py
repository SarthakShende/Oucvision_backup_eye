import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import schemas, authentication, job_database

logging.basicConfig(level=os.environ.get("OCUVISION_LOG_LEVEL", "INFO"))
logger = logging.getLogger("vm_agent")

app = FastAPI(
    title="OcuVision VM Agent",
    description="Secure VM-side API for dataset ops and training",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["127.0.0.1"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

db = job_database.JobDB()

@app.on_event("startup")
def startup():
    db.recover_interrupted()
    logger.info("VM Agent started; interrupted jobs recovered")

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/system/status")
def system_status(_: bool = Depends(authentication.verify_token)):
    import psutil
    gpu_available = False
    gpu_util = None
    gpu_mem_used = None
    gpu_mem_total = None
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_util = 0.0
            mem = torch.cuda.memory_stats(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            gpu_mem_used = torch.cuda.memory_allocated(0) // (1024**2)
    except Exception:
        pass
    disk = psutil.disk_usage("/")
    mem = psutil.virtual_memory()
    active = db.count_running()
    return schemas.SystemStatus(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=mem.percent,
        memory_available_gb=round(mem.available / (1024**3), 2),
        disk_percent=disk.percent,
        disk_free_gb=round(disk.free / (1024**3), 2),
        gpu_available=gpu_available,
        gpu_util=gpu_util,
        gpu_memory_used_mb=gpu_mem_used,
        gpu_memory_total_mb=gpu_mem_total,
        active_jobs=active,
        max_concurrent_jobs=int(os.environ.get("OCUVISION_MAX_CONCURRENT_JOBS", "2")),
        disk_reserve_gb=int(os.environ.get("OCUVISION_DISK_RESERVE_GB", "25")),
    )

@app.get("/jobs")
def list_jobs(_: bool = Depends(authentication.verify_token)):
    return db.list_jobs()

@app.get("/jobs/{job_id}")
def get_job(job_id: str, _: bool = Depends(authentication.verify_token)):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/jobs/discover-manifest")
def discover_manifest(req: schemas.JobCreate, _: bool = Depends(authentication.verify_token)):
    job_id = db.create_job(schemas.JobType.DISCOVER_MANIFEST.value, req.configuration)
    return {"job_id": job_id, "status": "queued"}

@app.post("/jobs/download")
def download(req: schemas.JobCreate, _: bool = Depends(authentication.verify_token)):
    job_id = db.create_job(schemas.JobType.DOWNLOAD.value, req.configuration)
    return {"job_id": job_id, "status": "queued"}

@app.post("/jobs/prepare-dataset")
def prepare_dataset(req: schemas.JobCreate, _: bool = Depends(authentication.verify_token)):
    job_id = db.create_job(schemas.JobType.PREPARE_DATASET.value, req.configuration)
    return {"job_id": job_id, "status": "queued"}

@app.post("/jobs/validate-dataset")
def validate_dataset(req: schemas.JobCreate, _: bool = Depends(authentication.verify_token)):
    job_id = db.create_job(schemas.JobType.VALIDATE_DATASET.value, req.configuration)
    return {"job_id": job_id, "status": "queued"}

@app.post("/jobs/train")
def train(req: schemas.JobCreate, _: bool = Depends(authentication.verify_token)):
    job_id = db.create_job(schemas.JobType.TRAIN.value, req.configuration)
    return {"job_id": job_id, "status": "queued", "requires_approval": True}

@app.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, _: bool = Depends(authentication.verify_token)):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail="Job not cancellable")
    db.update_job(job_id, status="cancelled", completed_at=time.time(), error="Cancelled by API")
    return {"job_id": job_id, "status": "cancelled"}
