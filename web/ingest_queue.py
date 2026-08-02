"""SRAG Ingest Queue — sequential per-folder ingestion with priority preemption.

A single background worker thread processes one folder at a time (SD.md §7
threading model). ``prioritize()`` lets the UI/API preempt whatever is
currently running: the in-flight job is cooperatively cancelled at its next
pipeline checkpoint, requeued (status ``paused``) for automatic retry, and the
requested folder starts immediately. Re-running an ingest is safe because
chunk IDs are deterministic (upsert, not duplicate).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ingest.folder_registry import FolderRegistry, collection_name_for_path
from ingest.pipeline import IngestCancelled, run as run_ingest

logger = logging.getLogger(__name__)


class TaskStatus:
    """String status constants for an :class:`IngestTask`."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestTask:
    task_id: str
    folder_id: int
    folder_path: str
    kind: str = "ingest"  # "ingest" | "sync" — both run a full (idempotent) re-ingest today
    status: str = TaskStatus.PENDING
    files_done: int = 0
    files_total: int = 0
    chunks_done: int = 0
    message: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        progress = (self.files_done / self.files_total) if self.files_total else (
            1.0 if self.status == TaskStatus.COMPLETED else 0.0
        )
        return {
            "task_id": self.task_id,
            "folder_id": self.folder_id,
            "folder_path": self.folder_path,
            "kind": self.kind,
            "status": self.status,
            "progress": round(progress, 4),
            "files_done": self.files_done,
            "files_total": self.files_total,
            "chunks_done": self.chunks_done,
            "message": self.message,
            "error": self.error,
        }


class IngestQueue:
    """FIFO queue of per-folder ingest jobs with priority preemption."""

    def __init__(self) -> None:
        self._tasks: dict[str, IngestTask] = {}
        self._queue: list[str] = []
        self._folder_tasks: dict[int, list[str]] = {}  # folder_id -> task_ids, newest last
        self._current_task_id: str | None = None
        self._current_cancel_event: threading.Event | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="ingest-queue", daemon=True)
        self._thread.start()
        logger.info("IngestQueue worker started.")

    def stop(self) -> None:
        self._stop_event.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info("IngestQueue worker stopped.")

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def push(self, folder_id: int, folder_path: str, kind: str = "ingest") -> str:
        """Queue a folder for ingestion. Returns the new task's ID."""
        task = IngestTask(task_id=uuid.uuid4().hex[:12], folder_id=folder_id, folder_path=folder_path, kind=kind)
        with self._lock:
            self._tasks[task.task_id] = task
            self._queue.append(task.task_id)
            self._folder_tasks.setdefault(folder_id, []).append(task.task_id)
        self._wake.set()
        return task.task_id

    def prioritize(self, folder_id: int, folder_path: str, kind: str = "ingest") -> str:
        """Preempt the queue so *folder_id* runs next.

        If a task for this folder is already pending, it's moved to the
        front. If a *different* folder's ingest is currently running, that
        job is cooperatively cancelled (it re-queues itself as ``paused`` and
        will resume automatically later — idempotent by chunk ID).
        """
        with self._lock:
            if self._current_task_id is not None and self._tasks[self._current_task_id].folder_id == folder_id:
                return self._current_task_id  # already running, nothing to preempt

            pending_for_folder = [tid for tid in self._queue if self._tasks[tid].folder_id == folder_id]
            if pending_for_folder:
                task_id = pending_for_folder[-1]
                self._queue.remove(task_id)
                self._queue.insert(0, task_id)
            else:
                task = IngestTask(task_id=uuid.uuid4().hex[:12], folder_id=folder_id, folder_path=folder_path, kind=kind)
                self._tasks[task.task_id] = task
                self._folder_tasks.setdefault(folder_id, []).append(task.task_id)
                self._queue.insert(0, task.task_id)
                task_id = task.task_id

            if self._current_task_id is not None and self._current_cancel_event is not None:
                logger.info("Preempting running task %s for folder %d", self._current_task_id, folder_id)
                self._current_cancel_event.set()

        self._wake.set()
        return task_id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None

    def get_status_for_folder(self, folder_id: int) -> dict[str, Any] | None:
        """Return the most recent task for *folder_id*, if any."""
        with self._lock:
            task_ids = self._folder_tasks.get(folder_id)
            if not task_ids:
                return None
            return self._tasks[task_ids[-1]].to_dict()

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                task_id = self._queue.pop(0) if self._queue else None
                if task_id is not None:
                    self._current_task_id = task_id
                    self._current_cancel_event = threading.Event()
                    self._tasks[task_id].status = TaskStatus.RUNNING
                    self._tasks[task_id].updated_at = time.time()

            if task_id is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue

            self._run_task(task_id)

            with self._lock:
                self._current_task_id = None
                self._current_cancel_event = None

    def _run_task(self, task_id: str) -> None:
        from web.dependencies import get_config, get_converter, get_embedder, get_store

        task = self._tasks[task_id]
        cancel_event = self._current_cancel_event
        registry = FolderRegistry(get_config().folders_db_path)
        collection_name = collection_name_for_path(task.folder_path)

        registry.update_folder_metadata(task.folder_id, ingest_status="indexing", last_error=None)

        def on_progress(files_done: int, files_total: int, current_file: str) -> None:
            with self._lock:
                task.files_done = files_done
                task.files_total = files_total
                task.message = current_file
                task.updated_at = time.time()

        try:
            result = run_ingest(
                folder_path=task.folder_path,
                converter=get_converter(),
                embedder=get_embedder(),
                store=get_store(),
                collection_name=collection_name,
                cancel_flag=(lambda: cancel_event.is_set()) if cancel_event else None,
                batch_size=get_config().pipeline_embed_batch_size,
                on_progress=on_progress,
            )
            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.chunks_done = result.chunks_created
                task.files_total = task.files_total or result.files_processed + result.files_failed
                task.message = f"{result.files_processed} file(s), {result.chunks_created} chunk(s)"
                task.error = "; ".join(result.errors) if result.errors and result.files_failed else None
                task.updated_at = time.time()
            registry.update_folder_metadata(
                task.folder_id,
                ingest_status="ready",
                last_indexed=time.time(),
                last_errors=result.errors,
            )
        except IngestCancelled:
            logger.info("Task %s paused (preempted); re-queueing.", task_id)
            with self._lock:
                task.status = TaskStatus.PAUSED
                task.message = "Paused (preempted by a higher-priority folder); will resume automatically."
                task.updated_at = time.time()
                self._queue.append(task_id)
            self._wake.set()
        except Exception as exc:
            logger.exception("Ingest failed for folder '%s'", task.folder_path)
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                task.updated_at = time.time()
            registry.update_folder_metadata(task.folder_id, ingest_status="error", last_error=str(exc))
        finally:
            registry.close()


# ─── Module-level singleton ────────────────────────────────────────────

_queue_instance: IngestQueue | None = None
_queue_lock = threading.Lock()


def get_ingest_queue() -> IngestQueue:
    """Return the process-wide :class:`IngestQueue` singleton."""
    global _queue_instance
    if _queue_instance is None:
        with _queue_lock:
            if _queue_instance is None:
                _queue_instance = IngestQueue()
    return _queue_instance


__all__ = ["IngestQueue", "IngestTask", "TaskStatus", "get_ingest_queue"]
