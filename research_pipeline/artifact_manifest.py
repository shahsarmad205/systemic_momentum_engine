"""Pipeline stage: artifact manifest.

Responsibility: Track all artifacts produced by the pipeline.
Writes a JSON manifest at the end of each run.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArtifactEntry:
    name: str
    path: str
    stage: str
    row_count: int = 0
    file_size_bytes: int = 0
    status: str = "written"
    notes: str = ""


@dataclass
class ArtifactManifest:
    """Complete record of all artifacts produced by a pipeline run."""
    run_id: str
    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0
    status: str = "running"
    contract_hash: str = ""
    artifacts: list[ArtifactEntry] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add_artifact(
        self,
        name: str,
        path: str,
        stage: str,
        row_count: int = 0,
        notes: str = "",
    ) -> None:
        """Register an artifact."""
        file_size = 0
        try:
            file_size = Path(path).stat().st_size
        except OSError:
            pass

        entry = ArtifactEntry(
            name=name,
            path=path,
            stage=stage,
            row_count=row_count,
            file_size_bytes=file_size,
            notes=notes,
        )
        self.artifacts.append(entry)

    def complete(self, status: str = "success") -> None:
        """Mark manifest as complete."""
        self.end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        start = time.strptime(self.start_time, "%Y-%m-%dT%H:%M:%S")
        end = time.strptime(self.end_time, "%Y-%m-%dT%H:%M:%S")
        self.duration_seconds = time.mktime(end) - time.mktime(start)
        self.status = status
        self.summary = {
            "n_artifacts": len(self.artifacts),
            "n_stages": len(set(a.stage for a in self.artifacts)),
            "total_bytes": sum(a.file_size_bytes for a in self.artifacts),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "contract_hash": self.contract_hash,
            "artifacts": [
                {
                    "name": a.name,
                    "path": a.path,
                    "stage": a.stage,
                    "row_count": a.row_count,
                    "file_size_bytes": a.file_size_bytes,
                    "status": a.status,
                    "notes": a.notes,
                }
                for a in self.artifacts
            ],
            "summary": self.summary,
        }

    def write(self, path: str | Path) -> Path:
        """Write manifest to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Artifact manifest written: %s", path)
        return path


def create_manifest(run_id: str = "") -> ArtifactManifest:
    """Create a new artifact manifest."""
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    return ArtifactManifest(
        run_id=run_id,
        start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
