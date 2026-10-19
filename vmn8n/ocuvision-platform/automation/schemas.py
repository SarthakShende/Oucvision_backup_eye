from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class JobType(str, Enum):
    DISCOVER_MANIFEST = "discover-manifest"
    DOWNLOAD = "download"
    PREPARE_DATASET = "prepare-dataset"
    VALIDATE_DATASET = "validate-dataset"
    TRAIN = "train"

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobBase(BaseModel):
    job_type: JobType
    configuration: Dict[str, Any] = Field(default_factory=dict)

class JobCreate(JobBase):
    pass

class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    configuration: Dict[str, Any] = Field(default_factory=dict)

class SystemStatus(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    gpu_available: bool
    gpu_util: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    active_jobs: int
    max_concurrent_jobs: int
    disk_reserve_gb: int
