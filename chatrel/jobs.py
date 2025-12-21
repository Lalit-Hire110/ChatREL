"""
Job management system for ChatREL v4
Handles job state persistence and progress tracking
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from . import config

JOBS_DIR = config.PROJECT_ROOT / "jobs"
JOBS_DIR.mkdir(exist_ok=True)


class Job:
    """Represents an analysis job with state persistence."""
    
    def __init__(self, job_id: str = None):
        """
        Initialize or load a job.
        
        Args:
            job_id: Existing job ID to load, or None to create new
        """
        if job_id is None:
            self.job_id = str(uuid.uuid4())
            self.created_at = datetime.now().isoformat()
            self.status = "created"
            self.progress = 0
            self.total_messages = 0
            self.error = None
            self.result = None
            self.pseudonymized = False
            self.sender_map = {}
        else:
            self.job_id = job_id
            self._load()
    
    @property
    def job_dir(self) -> Path:
        """Get job directory path."""
        path = JOBS_DIR / self.job_id
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def state_file(self) -> Path:
        """Get job state file path."""
        return self.job_dir / "job.json"
    
    def _load(self):
        """Load job state from disk."""
        if not self.state_file.exists():
            raise ValueError(f"Job {self.job_id} not found")
        
        with open(self.state_file, "r") as f:
            data = json.load(f)
        
        self.created_at = data.get("created_at")
        self.status = data.get("status", "unknown")
        self.progress = data.get("progress", 0)
        self.total_messages = data.get("total_messages", 0)
        self.error = data.get("error")
        self.result = data.get("result")
        self.pseudonymized = data.get("pseudonymized", False)
        self.sender_map = data.get("sender_map", {})
    
    def save(self):
        """Save job state to disk."""
        data = {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "status": self.status,
            "progress": self.progress,
            "total_messages": self.total_messages,
            "error": self.error,
            "result": self.result,
            "pseudonymized": self.pseudonymized,
            "sender_map": self.sender_map,
        }
        
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def update_status(self, status: str, progress: Optional[int] = None):
        """Update job status and optionally progress."""
        self.status = status
        if progress is not None:
            self.progress = progress
        self.save()
    
    def set_error(self, error_msg: str):
        """Mark job as failed with error message."""
        self.status = "failed"
        self.error = error_msg
        self.save()
    
    def set_result(self, result: Dict[str, Any]):
        """Set job result and mark as completed."""
        self.status = "completed"
        self.progress = 100
        self.result = result
        self.save()
    
    def get_report_path(self) -> Path:
        """Get path to JSON report file."""
        return self.job_dir / "report.json"
    
    def get_csv_path(self) -> Path:
        """Get path to annotated messages CSV."""
        return self.job_dir / "annotated.csv"
    
    def get_json_path(self) -> Path:
        """Get path to annotated messages JSON."""
        return self.job_dir / "annotated.json"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "status": self.status,
            "progress": self.progress,
            "total_messages": self.total_messages,
            "error": self.error,
            "result": self.result,
            "pseudonymized": self.pseudonymized,
        }


def list_jobs(limit: int = 50) -> list:
    """
    List recent jobs.
    
    Args:
        limit: Maximum number of jobs to return
    
    Returns:
        List of job dictionaries sorted by creation time (newest first)
    """
    jobs = []
    
    for job_dir in JOBS_DIR.iterdir():
        if job_dir.is_dir():
            state_file = job_dir / "job.json"
            if state_file.exists():
                try:
                    job = Job(job_dir.name)
                    jobs.append(job.to_dict())
                except Exception:
                    continue
    
    # Sort by created_at descending
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return jobs[:limit]


def cleanup_old_jobs(days: int = 7):
    """
    Clean up jobs older than specified days.
    
    Args:
        days: Age threshold in days
    
    Returns:
        Number of jobs deleted
    """
    import shutil
    from datetime import timedelta
    
    cutoff = datetime.now() - timedelta(days=days)
    deleted = 0
    
    for job_dir in JOBS_DIR.iterdir():
        if job_dir.is_dir():
            state_file = job_dir / "job.json"
            if state_file.exists():
                try:
                    job = Job(job_dir.name)
                    created = datetime.fromisoformat(job.created_at)
                    
                    if created < cutoff:
                        shutil.rmtree(job_dir)
                        deleted += 1
                except Exception:
                    continue
    
    return deleted


if __name__ == "__main__":
    # Test job system
    print("Testing Job System:")
    
    # Create new job
    job = Job()
    print(f"Created job: {job.job_id}")
    job.save()
    
    # Update progress
    job.update_status("processing", progress=50)
    print(f"Updated to: {job.status} ({job.progress}%)")
    
    # Load job
    loaded = Job(job.job_id)
    print(f"Loaded job: status={loaded.status}, progress={loaded.progress}%")
    
    # List jobs
    print(f"\nAll jobs: {list_jobs(limit=5)}")
