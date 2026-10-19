"""Persistent ordered job queue and runner."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from simplecomfy.comfy_client import ComfyClient, ComfyError
from simplecomfy.config import AppConfig
from simplecomfy.effects import (
    MAX_EFFECTS,
    EffectRegistry,
    effect_lora_stack,
    merge_prompts,
    merge_video_prompts,
)
from simplecomfy.history import GalleryStore, media_kind
from simplecomfy.imagesize import fit_video_size, probe_image_size
from simplecomfy.presets import PresetRegistry
from simplecomfy.puller import ModelPuller, PullError
from simplecomfy.supervisor import ComfySupervisor
from simplecomfy.video_ops import (
    VideoOpsError,
    concat_chain_xfade,
    concat_overlap_crossfade,
    extract_first_frame,
    extract_last_frame,
    extract_tail_clip,
    first_video_in_job,
    smooth_video,
    trim_end_frames,
    trim_start_frames,
)
from simplecomfy.video_pack import FAST_PROFILES, VideoPackRegistry
from simplecomfy.workflows import (
    build_txt2img,
    build_wan_flf2vid,
    build_wan_img2vid,
    build_wan_txt2vid,
)

log = logging.getLogger("simplecomfy.jobs")
QUEUE_STATE_VERSION = 1


class JobCancelled(Exception):
    """Raised when a user cancels an in-flight job."""

    def __init__(self, job_id: str, message: str = "cancelled") -> None:
        super().__init__(message)
        self.job_id = job_id

MODALITIES = ("txt2img", "txt2vid", "img2vid", "smooth", "continue")
IMPLEMENTED_MODALITIES = frozenset({"txt2img", "txt2vid", "img2vid", "smooth", "continue"})
# Default continue length matches pack (~5s @ 16fps with 4n+1 frames)
DEFAULT_CONTINUE_LENGTH = 81
# Default seam crossfade (seconds) after motion-prefix continue
DEFAULT_CONTINUE_OVERLAP_S = 0.75
# Cap: blend may use at most this fraction of parent or chunk duration
DEFAULT_CONTINUE_OVERLAP_MAX_FRACTION = 0.4
# Multi-frame motion prefix fed into WanImageToVideo (last N frames of parent)
DEFAULT_CONTINUE_CONTEXT_FRAMES = 16
# Extra free frames to discard after the locked prefix (first free frames are often janky)
DEFAULT_CONTINUE_WARMUP_FRAMES = 8
# FLF bridge length between real parent tail and new chunk head (~1.5s @ 16fps)
DEFAULT_CONTINUE_BRIDGE_FRAMES = 25
# Drop this many frames from each end of the bridge when splicing (avoid locked endpoints)
DEFAULT_CONTINUE_BRIDGE_TRIM = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    queued = "queued"
    pulling = "pulling"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


@dataclass
class GenerateRequest:
    preset: str
    prompt: str
    negative_prompt: str | None = None
    seed: int = -1
    width: int | None = None
    height: int | None = None
    auto_pull: bool | None = None
    # None = preset.default_effects; [] = none; list = exact set
    effects: list[str] | None = None
    effect_weights: dict[str, float] | None = None
    modality: str = "txt2img"
    # img2vid source: gallery job OR uploaded input basename (comfy input/)
    source_job_id: str | None = None
    source_filename: str | None = None
    source_input: str | None = None  # uploaded file name under comfy input/
    # Multi-frame motion prefix mp4 basename in Comfy input/ (img2vid / continue)
    source_video_input: str | None = None
    motion_prefix_frames: int | None = None
    # video knobs (optional; pack defaults otherwise)
    length: int | None = None
    fps: float | None = None
    video_pack: str | None = None
    # draft | final | fast | quality
    video_profile: str | None = None
    # smooth / continue parents
    parent_job_id: str | None = None
    smooth_factor: int | None = None  # 2 or 4
    continue_seconds: float | None = None  # optional; maps to length via fps
    # How much tail/head to blend at the continue seam (pixel crossfade)
    continue_overlap_seconds: float | None = None
    continue_overlap_frames: int | None = None
    # Max fraction of either clip the blend may consume (long-parent safety)
    continue_overlap_max_fraction: float | None = None
    # Last N frames of parent used as WanImageToVideo multi-frame start (motion)
    continue_context_frames: int | None = None
    # Drop this many frames after the locked prefix before joining (mask boundary jank)
    continue_warmup_frames: int | None = None
    # Continue join strategy: flf_bridge (default) | prefix
    continue_method: str | None = None
    # FLF bridge length in frames (4n+1)
    continue_bridge_frames: int | None = None
    # Long single-shot: WanContextWindowsManual (None = auto when length > window)
    context_windows: bool | None = None
    context_length: int | None = None  # window size frames (4n+1), default 81
    context_overlap: int | None = None  # shared frames between windows (continuity)


@dataclass
class QueueEntry:
    """One slot in the ordered queue: a job or a breakpoint."""

    kind: str  # "job" | "breakpoint"
    id: str  # job_id or breakpoint id
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "id": self.id}
        if self.kind == "breakpoint":
            d["label"] = self.label or "Break"
        return d

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> QueueEntry:
        kind = str(raw.get("kind") or "job")
        eid = str(raw.get("id") or raw.get("job_id") or "")
        return QueueEntry(kind=kind, id=eid, label=str(raw.get("label") or ""))


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: str
    request: GenerateRequest
    updated_at: str = ""
    error: str | None = None
    error_detail: dict[str, Any] | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    phase: str | None = "queued"
    progress: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "status": self.status.value,
            "phase": self.phase,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "modality": self.request.modality,
            "preset": self.request.preset,
            "prompt": self.request.prompt,
            "negative_prompt": self.request.negative_prompt,
            "effects": self.request.effects,
            "source_job_id": self.request.source_job_id,
            "source_filename": self.request.source_filename,
            "source_input": self.request.source_input,
            "parent_job_id": self.request.parent_job_id or self.meta.get("parent_job_id"),
            "seed": self.meta.get("seed", self.request.seed),
            "error": self.error,
            "error_detail": self.error_detail,
            "outputs": self.outputs,
            "meta": self.meta,
            "video_profile": self.request.video_profile or self.meta.get("profile"),
        }
        if include_events:
            d["events"] = self.events[-30:]
        return d

    def push_event(self, event: dict[str, Any]) -> None:
        e = {"ts": _utc_now(), **event}
        self.events.append(e)
        if len(self.events) > 200:
            self.events = self.events[-100:]
        self.updated_at = e["ts"]
        # Mirror phase/progress from event when present
        if "phase" in event and event["phase"]:
            self.phase = str(event["phase"])
        if any(k in event for k in ("current", "total", "unit", "message")):
            prog: dict[str, Any] = dict(self.progress or {})
            for k in ("current", "total", "unit", "message"):
                if k in event:
                    prog[k] = event[k]
            self.progress = prog

    def set_phase(
        self,
        phase: str,
        *,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        unit: str | None = None,
        event_type: str | None = None,
        **extra: Any,
    ) -> None:
        self.phase = phase
        if current is not None or total is not None or message is not None or unit is not None:
            prog = dict(self.progress or {})
            if current is not None:
                prog["current"] = current
            if total is not None:
                prog["total"] = total
            if unit is not None:
                prog["unit"] = unit
            if message is not None:
                prog["message"] = message
            self.progress = prog
        elif message is not None:
            self.progress = {**(self.progress or {}), "message": message}
        ev: dict[str, Any] = {"type": event_type or "phase", "phase": phase, **extra}
        if message is not None:
            ev["message"] = message
        if current is not None:
            ev["current"] = current
        if total is not None:
            ev["total"] = total
        if unit is not None:
            ev["unit"] = unit
        self.push_event(ev)

    def snapshot_dict(self) -> dict[str, Any]:
        """Full snapshot for disk persistence of non-terminal jobs."""
        return {
            "id": self.id,
            "status": self.status.value,
            "phase": self.phase,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "error_detail": self.error_detail,
            "outputs": self.outputs,
            "meta": self.meta,
            "events": self.events[-50:],
            "request": asdict(self.request),
        }

    @staticmethod
    def from_snapshot(raw: dict[str, Any]) -> Job:
        req_raw = raw.get("request") or {}
        # Only known GenerateRequest fields
        fields = set(GenerateRequest.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        req_kwargs = {k: v for k, v in req_raw.items() if k in fields}
        req = GenerateRequest(**req_kwargs)
        status_s = str(raw.get("status") or "queued")
        try:
            status = JobStatus(status_s)
        except ValueError:
            status = JobStatus.queued
        job = Job(
            id=str(raw.get("id") or uuid.uuid4()),
            status=status,
            created_at=str(raw.get("created_at") or _utc_now()),
            request=req,
            updated_at=str(raw.get("updated_at") or ""),
            error=raw.get("error"),
            error_detail=raw.get("error_detail"),
            outputs=list(raw.get("outputs") or []),
            meta=dict(raw.get("meta") or {}),
            events=list(raw.get("events") or []),
            phase=raw.get("phase") or "queued",
            progress=raw.get("progress"),
        )
        return job


class JobManager:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        registry: PresetRegistry,
        effects: EffectRegistry,
        puller: ModelPuller,
        client: ComfyClient,
        supervisor: ComfySupervisor,
        video_packs: VideoPackRegistry | None = None,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.effects = effects
        self.puller = puller
        self.client = client
        self.supervisor = supervisor
        self.video_packs = video_packs or VideoPackRegistry(cfg)
        self.gallery = GalleryStore(cfg)
        self._jobs: dict[str, Job] = {}
        self._entries: list[QueueEntry] = []
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._active_job_id: str | None = None
        self._paused_at_breakpoint = False
        self._pause_breakpoint_id: str | None = None
        self._cancel_flags: set[str] = set()

    @property
    def _queue_path(self) -> Path:
        return self.cfg.state_dir / "queue.json"

    @property
    def _jobs_state_dir(self) -> Path:
        return self.cfg.state_dir / "jobs"

    async def start(self) -> None:
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self._jobs_state_dir.mkdir(parents=True, exist_ok=True)
        await self._load_state()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="simplecomfy-worker")
        self._wake.set()

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        async with self._lock:
            self._save_queue_unlocked()
            for job in self._jobs.values():
                if job.status in (JobStatus.queued, JobStatus.pulling, JobStatus.running):
                    self._save_job_state(job)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self, *, limit: int = 50) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def is_busy(self) -> bool:
        if self._active_job_id:
            j = self._jobs.get(self._active_job_id)
            if j and j.status in (JobStatus.pulling, JobStatus.running):
                return True
        return any(
            j.status in (JobStatus.pulling, JobStatus.running) for j in self._jobs.values()
        )

    def queued_job_count(self) -> int:
        return sum(1 for e in self._entries if e.kind == "job")

    def _persist(self, job: Job) -> None:
        try:
            self.gallery.persist_from_job_dict(job.to_dict(include_events=False))
        except Exception:
            log.exception("failed to persist job sidecar %s", job.id)
        if job.status in (
            JobStatus.completed,
            JobStatus.failed,
            JobStatus.cancelled,
        ):
            self._delete_job_state(job.id)
        else:
            self._save_job_state(job)

    def _save_job_state(self, job: Job) -> None:
        try:
            self._jobs_state_dir.mkdir(parents=True, exist_ok=True)
            path = self._jobs_state_dir / f"{job.id}.json"
            path.write_text(
                json.dumps(job.snapshot_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            log.exception("failed to save job state %s", job.id)

    def _delete_job_state(self, job_id: str) -> None:
        path = self._jobs_state_dir / f"{job_id}.json"
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass

    def _save_queue_unlocked(self) -> None:
        try:
            self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": QUEUE_STATE_VERSION,
                "updated_at": _utc_now(),
                "active_job_id": self._active_job_id,
                "entries": [e.to_dict() for e in self._entries],
            }
            tmp = self._queue_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._queue_path)
        except Exception:
            log.exception("failed to save queue state")

    async def _load_state(self) -> None:
        # Load job snapshots
        if self._jobs_state_dir.is_dir():
            for path in sorted(self._jobs_state_dir.glob("*.json")):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    job = Job.from_snapshot(raw)
                    # Interrupted mid-run → re-queue
                    if job.status in (JobStatus.pulling, JobStatus.running):
                        job.status = JobStatus.queued
                        job.phase = "queued"
                        job.progress = None
                        job.error = None
                        job.push_event(
                            {
                                "type": "requeued",
                                "phase": "queued",
                                "message": "re-queued after restart",
                            }
                        )
                    self._jobs[job.id] = job
                except Exception:
                    log.exception("failed to load job state %s", path)

        entries: list[QueueEntry] = []
        if self._queue_path.is_file():
            try:
                raw = json.loads(self._queue_path.read_text(encoding="utf-8"))
                for item in raw.get("entries") or []:
                    if not isinstance(item, dict):
                        continue
                    ent = QueueEntry.from_dict(item)
                    if not ent.id:
                        continue
                    if ent.kind == "job" and ent.id not in self._jobs:
                        continue
                    if ent.kind == "job":
                        j = self._jobs[ent.id]
                        if j.status != JobStatus.queued:
                            continue
                    entries.append(ent)
                # Any queued jobs missing from entries → append
                listed = {e.id for e in entries if e.kind == "job"}
                for jid, job in self._jobs.items():
                    if job.status == JobStatus.queued and jid not in listed:
                        entries.append(QueueEntry(kind="job", id=jid))
            except Exception:
                log.exception("failed to load queue state")
        else:
            for jid, job in self._jobs.items():
                if job.status == JobStatus.queued:
                    entries.append(QueueEntry(kind="job", id=jid))

        self._entries = entries
        self._active_job_id = None
        self._save_queue_unlocked()

    def queue_snapshot(self) -> dict[str, Any]:
        """Public queue view for API/UI."""
        entries_out: list[dict[str, Any]] = []
        pos = 0
        for ent in self._entries:
            item = ent.to_dict()
            if ent.kind == "job":
                job = self._jobs.get(ent.id)
                if job:
                    item["job"] = job.to_dict()
                    item["queue_position"] = pos
                    pos += 1
                else:
                    continue
            elif ent.kind == "breakpoint":
                item["label"] = ent.label or "Break"
                if self._paused_at_breakpoint and self._pause_breakpoint_id == ent.id:
                    item["active_pause"] = True
            entries_out.append(item)

        active = None
        if self._active_job_id:
            aj = self._jobs.get(self._active_job_id)
            if aj:
                active = aj.to_dict()

        return {
            "entries": entries_out,
            "active_job": active,
            "active_job_id": self._active_job_id,
            "paused": self._paused_at_breakpoint,
            "pause_breakpoint_id": self._pause_breakpoint_id,
            "queued_jobs": self.queued_job_count(),
            "max_depth": self.cfg.queue.max_depth,
            "busy": self.is_busy(),
        }

    async def enqueue(self, req: GenerateRequest) -> Job:
        jobs = await self.enqueue_batch(req, batch_size=1)
        return jobs[0]

    async def enqueue_batch(self, req: GenerateRequest, *, batch_size: int = 1) -> list[Job]:
        modality = (req.modality or "txt2img").strip().lower()
        if modality not in MODALITIES:
            raise RuntimeError(f"unknown modality '{req.modality}'; use {', '.join(MODALITIES)}")
        req.modality = modality

        if modality == "txt2img":
            self.registry.require(req.preset)
        if req.effects is not None and len(req.effects) > MAX_EFFECTS:
            raise RuntimeError(f"too many effects (max {MAX_EFFECTS})")

        n = max(1, int(batch_size))
        max_batch = max(1, int(self.cfg.queue.max_batch or 8))
        if n > max_batch:
            raise RuntimeError(f"batch_size {n} exceeds max_batch={max_batch}")

        async with self._lock:
            depth = self.queued_job_count()
            # Count active running as occupying a slot toward depth? Plan: pending only.
            if depth + n > self.cfg.queue.max_depth:
                raise RuntimeError(
                    f"queue full (max_depth={self.cfg.queue.max_depth}, "
                    f"queued={depth}, batch={n})"
                )

            created: list[Job] = []
            base_seed = int(req.seed) if req.seed is not None else -1
            for i in range(n):
                # Deep-ish copy request per job
                r = GenerateRequest(**asdict(req))
                r.modality = modality
                if base_seed >= 0:
                    r.seed = base_seed + i
                else:
                    r.seed = -1
                job = Job(
                    id=str(uuid.uuid4()),
                    status=JobStatus.queued,
                    created_at=_utc_now(),
                    request=r,
                    phase="queued",
                )
                job.meta["modality"] = modality
                if n > 1:
                    job.meta["batch_index"] = i
                    job.meta["batch_size"] = n
                job.push_event({"type": "queued", "phase": "queued", "modality": modality})
                self._jobs[job.id] = job
                self._entries.append(QueueEntry(kind="job", id=job.id))
                self._save_job_state(job)
                created.append(job)
            self._save_queue_unlocked()
            self._wake.set()
            return created

    async def add_breakpoint(
        self,
        *,
        after_id: str | None = None,
        position: int | None = None,
        label: str = "Break",
    ) -> dict[str, Any]:
        async with self._lock:
            ent = QueueEntry(
                kind="breakpoint",
                id=str(uuid.uuid4()),
                label=(label or "Break").strip() or "Break",
            )
            if after_id:
                idx = next((i for i, e in enumerate(self._entries) if e.id == after_id), None)
                if idx is None:
                    raise RuntimeError(f"queue entry not found: {after_id}")
                self._entries.insert(idx + 1, ent)
            elif position is not None:
                pos = max(0, min(int(position), len(self._entries)))
                self._entries.insert(pos, ent)
            else:
                self._entries.append(ent)
            self._save_queue_unlocked()
            self._wake.set()
            return ent.to_dict()

    async def reorder(
        self,
        entry_id: str,
        *,
        action: str | None = None,
        ordered_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            if ordered_ids is not None:
                id_set = {e.id for e in self._entries}
                if set(ordered_ids) != id_set or len(ordered_ids) != len(self._entries):
                    raise RuntimeError("ordered_ids must list every queue entry id exactly once")
                by_id = {e.id: e for e in self._entries}
                self._entries = [by_id[i] for i in ordered_ids]
            else:
                act = (action or "").strip().lower()
                if act not in ("up", "down", "top", "bottom"):
                    raise RuntimeError("action must be up|down|top|bottom")
                idx = next((i for i, e in enumerate(self._entries) if e.id == entry_id), None)
                if idx is None:
                    raise RuntimeError(f"queue entry not found: {entry_id}")
                ent = self._entries.pop(idx)
                if act == "up":
                    self._entries.insert(max(0, idx - 1), ent)
                elif act == "down":
                    self._entries.insert(min(len(self._entries), idx + 1), ent)
                elif act == "top":
                    self._entries.insert(0, ent)
                else:
                    self._entries.append(ent)
            self._save_queue_unlocked()
            self._wake.set()
            return self.queue_snapshot()

    def is_cancel_requested(self, job_id: str) -> bool:
        return job_id in self._cancel_flags

    def _raise_if_cancelled(self, job: Job) -> None:
        if self.is_cancel_requested(job.id) or job.meta.get("cancel_requested"):
            raise JobCancelled(job.id)

    def _mark_cancelled(self, job: Job, *, reason: str = "user_cancel") -> None:
        job.status = JobStatus.cancelled
        job.phase = "cancelled"
        job.progress = None
        job.error = "cancelled"
        job.error_detail = {"reason": reason}
        job.meta["cancel_requested"] = True
        job.push_event({"type": "cancelled", "phase": "cancelled", "reason": reason})
        self._cancel_flags.discard(job.id)
        self._persist(job)

    async def remove_entry(self, entry_id: str) -> dict[str, Any]:
        """Remove a breakpoint or cancel a queued job."""
        async with self._lock:
            idx = next((i for i, e in enumerate(self._entries) if e.id == entry_id), None)
            if idx is None:
                raise RuntimeError(f"queue entry not found: {entry_id}")
            ent = self._entries.pop(idx)
            if ent.kind == "job":
                job = self._jobs.get(ent.id)
                if job and job.status == JobStatus.queued:
                    self._mark_cancelled(job)
            if self._pause_breakpoint_id == entry_id:
                self._paused_at_breakpoint = False
                self._pause_breakpoint_id = None
            self._save_queue_unlocked()
            self._wake.set()
            return self.queue_snapshot()

    async def cancel_job(self, job_id: str) -> Job:
        """Cancel a queued job, or request interrupt of the active running job."""
        interrupt = False
        prompt_id: str | None = None
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise RuntimeError(f"job not found: {job_id}")
            if job.status == JobStatus.queued:
                self._entries = [
                    e for e in self._entries if not (e.kind == "job" and e.id == job_id)
                ]
                self._mark_cancelled(job)
                self._save_queue_unlocked()
                self._wake.set()
                return job
            if job.status in (JobStatus.pulling, JobStatus.running):
                self._cancel_flags.add(job_id)
                job.meta["cancel_requested"] = True
                job.set_phase(
                    "cancelling",
                    message="Stopping…",
                    event_type="cancel_requested",
                )
                raw_pid = job.meta.get("comfy_prompt_id")
                prompt_id = str(raw_pid) if raw_pid else None
                interrupt = True
                self._save_job_state(job)
                self._wake.set()
            else:
                raise RuntimeError(f"job already {job.status.value}")

        if interrupt:
            # Best-effort stop Comfy; worker will finalize cancelled status
            try:
                await self.client.interrupt()
            except Exception as e:
                log.warning("Comfy interrupt failed for %s: %s", job_id, e)
            if prompt_id:
                try:
                    await self.client.delete_queue_items([prompt_id])
                except Exception as e:
                    log.warning("Comfy queue delete failed for %s: %s", job_id, e)
        return job

    async def _worker_loop(self) -> None:
        while True:
            job: Job | None = None
            try:
                async with self._lock:
                    self._paused_at_breakpoint = False
                    self._pause_breakpoint_id = None
                    if not self._entries:
                        self._active_job_id = None
                        self._save_queue_unlocked()
                    else:
                        head = self._entries[0]
                        if head.kind == "breakpoint":
                            self._paused_at_breakpoint = True
                            self._pause_breakpoint_id = head.id
                            self._active_job_id = None
                            self._save_queue_unlocked()
                        elif head.kind == "job":
                            self._entries.pop(0)
                            job = self._jobs.get(head.id)
                            if job and job.status == JobStatus.queued:
                                self._active_job_id = job.id
                            else:
                                job = None
                            self._save_queue_unlocked()

                if job is None:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        pass
                    continue

                try:
                    self._raise_if_cancelled(job)
                    await self._run_job(job)
                    # If cancel landed after run finished completing, leave result
                    if (
                        self.is_cancel_requested(job.id)
                        and job.status
                        not in (
                            JobStatus.completed,
                            JobStatus.failed,
                            JobStatus.cancelled,
                        )
                    ):
                        self._mark_cancelled(job)
                except JobCancelled:
                    if job.status not in (
                        JobStatus.completed,
                        JobStatus.failed,
                        JobStatus.cancelled,
                    ):
                        self._mark_cancelled(job)
                except Exception as e:
                    if self.is_cancel_requested(job.id) or str(e) == "cancelled":
                        self._mark_cancelled(job)
                    else:
                        log.exception("job %s crashed", job.id)
                        job.status = JobStatus.failed
                        job.phase = "failed"
                        job.error = str(e)
                        job.push_event(
                            {"type": "failed", "phase": "failed", "error": str(e)}
                        )
                        self._persist(job)
                finally:
                    self._cancel_flags.discard(job.id)
                    async with self._lock:
                        if self._active_job_id == job.id:
                            self._active_job_id = None
                        self._save_queue_unlocked()
                    self._wake.set()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("worker loop error")
                await asyncio.sleep(0.5)

    async def _run_job(self, job: Job) -> None:
        req = job.request
        job.meta["modality"] = req.modality

        if req.modality not in IMPLEMENTED_MODALITIES:
            job.status = JobStatus.failed
            job.error = "modality_not_implemented"
            job.error_detail = {
                "modality": req.modality,
                "implemented": sorted(IMPLEMENTED_MODALITIES),
            }
            job.push_event({"type": "failed", "error": job.error, "detail": job.error_detail})
            self._persist(job)
            return

        if req.modality == "txt2img":
            await self._run_txt2img(job)
        elif req.modality in ("txt2vid", "img2vid"):
            await self._run_video(job, finalize=True)
        elif req.modality == "smooth":
            await self._run_smooth(job)
        elif req.modality == "continue":
            await self._run_continue(job)
        else:
            job.status = JobStatus.failed
            job.error = f"unsupported modality {req.modality}"
            self._persist(job)

    async def _run_txt2img(self, job: Job) -> None:
        req = job.request
        try:
            preset = self.registry.require(req.preset)
        except KeyError as e:
            job.status = JobStatus.failed
            job.error = str(e)
            self._persist(job)
            return

        try:
            effects = self.effects.resolve_for_generate(
                preset,
                req.effects,
                weights=req.effect_weights,
            )
        except (KeyError, ValueError) as e:
            job.status = JobStatus.failed
            job.error = str(e)
            job.push_event({"type": "failed", "error": str(e)})
            self._persist(job)
            return

        # Pull is a preliminary phase of generate (default on)
        auto_pull = True if req.auto_pull is None else bool(req.auto_pull)

        job.status = JobStatus.pulling
        job.set_phase(
            "ensuring_models",
            message="Checking models…",
            event_type="pulling",
            effects=[e.id for e in effects],
        )

        async def on_progress(ev: dict[str, Any]) -> None:
            self._raise_if_cancelled(job)
            self._apply_pull_progress(job, ev)

        try:
            self._raise_if_cancelled(job)
            ckpt_name, preset_loras = await self.puller.ensure_preset(
                preset, auto_pull=auto_pull, on_progress=on_progress
            )
            self._raise_if_cancelled(job)
            effect_specs = [s for e in effects if (s := e.as_lora_spec()) is not None]
            effect_loras = await self.puller.ensure_lora_specs(
                effect_specs, auto_pull=auto_pull, on_progress=on_progress
            )
        except JobCancelled:
            raise
        except PullError as e:
            job.status = JobStatus.failed
            job.phase = "failed"
            job.error = str(e)
            job.error_detail = e.detail
            job.push_event({"type": "failed", "phase": "failed", "error": str(e), "detail": e.detail})
            self._persist(job)
            return

        lora_files = list(preset_loras) + list(effect_loras)

        self._raise_if_cancelled(job)
        job.set_phase("waiting_comfy", message="Starting ComfyUI…")
        if not await self._ensure_comfy(job):
            return

        try:
            available = await self.client.list_checkpoints()
        except Exception as e:
            available = []
            log.warning("list_checkpoints failed: %s", e)

        if available and ckpt_name not in available:
            hint = (
                f"Checkpoint '{ckpt_name}' is on disk for simpleComfy but ComfyUI "
                f"does not list it. Restart ComfyUI after model path changes "
                f"(./scripts/start-comfyui.sh). Currently visible: "
                f"{', '.join(available[:12])}"
                + ("…" if len(available) > 12 else "")
            )
            job.status = JobStatus.failed
            job.error = "checkpoint_not_visible_to_comfy"
            job.error_detail = {
                "checkpoint": ckpt_name,
                "available": available,
                "hint": hint,
                "models_dir": str(self.cfg.models_dir),
            }
            job.push_event(
                {
                    "type": "failed",
                    "error": "checkpoint_not_visible_to_comfy",
                    "detail": job.error_detail,
                }
            )
            self._persist(job)
            return

        if available and lora_files:
            try:
                lora_avail = set(await self.client.list_loras())
            except Exception:
                lora_avail = set()
            if lora_avail:
                missing_l = [n for n, _ in lora_files if n not in lora_avail]
                if missing_l:
                    job.status = JobStatus.failed
                    job.error = "lora_not_visible_to_comfy"
                    job.error_detail = {
                        "missing": missing_l,
                        "hint": "Restart ComfyUI so it rescans loras/.",
                    }
                    job.push_event(
                        {
                            "type": "failed",
                            "error": "lora_not_visible_to_comfy",
                            "detail": job.error_detail,
                        }
                    )
                    self._persist(job)
                    return

        width = req.width or preset.size.width or self.cfg.generate.default_width
        height = req.height or preset.size.height or self.cfg.generate.default_height
        width = max(64, min(int(width), self.cfg.generate.max_width))
        height = max(64, min(int(height), self.cfg.generate.max_height))
        width = width - (width % 8)
        height = height - (height % 8)

        positive, negative = merge_prompts(
            preset, effects, req.prompt, req.negative_prompt
        )

        workflow, meta = build_txt2img(
            preset=preset,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            checkpoint_name=ckpt_name,
            lora_files=lora_files,
            width=width,
            height=height,
            seed=req.seed,
            filename_prefix=f"simplecomfy/{job.id[:8]}",
            positive=positive,
            negative=negative,
        )
        meta["modality"] = "txt2img"
        meta["effects"] = [
            {"id": e.id, "weight": e.weight, "needs_lora": e.needs_lora} for e in effects
        ]
        job.meta = meta
        await self._execute_workflow(job, workflow, timeout_s=self.cfg.comfy.timeout_s)

    def _resolve_video_pack(self, req: GenerateRequest):
        pid = (req.video_pack or self.cfg.models.video_pack or "sample").strip()
        pack = self.video_packs.get(pid)
        if pack is None:
            pack = self.video_packs.default()
        return pack

    def _resolve_source_path(self, req: GenerateRequest) -> tuple[Path, str]:
        """
        Resolve img2vid source file.
        Returns (absolute_path, comfy_load_name).
        """
        import shutil

        input_dir = self.cfg.comfy_input_dir
        if not input_dir:
            raise RuntimeError(
                "comfy.input_dir not set and ComfyUI input/ not found — "
                "set comfy.root or comfy.input_dir in config.toml"
            )
        input_dir.mkdir(parents=True, exist_ok=True)

        # Uploaded input already in Comfy input/ (or data/uploads staged there)
        if req.source_input:
            name = Path(req.source_input).name
            if "/" in req.source_input or "\\" in req.source_input or name.startswith("."):
                raise RuntimeError("invalid source_input name")
            direct = input_dir / name
            if direct.is_file():
                return direct, name
            upload = self.cfg.data_dir / "uploads" / name
            if upload.is_file():
                dest = input_dir / name
                if not dest.is_file() or dest.stat().st_size != upload.stat().st_size:
                    shutil.copy2(upload, dest)
                return dest, name
            raise RuntimeError(f"uploaded source not found: {name}")

        if not req.source_job_id:
            raise RuntimeError("source_job_id or source_input required for img2vid")

        item = self.gallery.get(req.source_job_id)
        if not item:
            raise RuntimeError(f"source job not found: {req.source_job_id}")
        outputs = item.get("outputs") or []
        chosen = None
        if req.source_filename:
            for o in outputs:
                if o.get("filename") == req.source_filename:
                    chosen = o
                    break
        if not chosen:
            chosen = next((o for o in outputs if o.get("kind") == "image"), None)
        if not chosen:
            chosen = outputs[0] if outputs else None
        if not chosen or not chosen.get("filename"):
            raise RuntimeError("source job has no image output")

        src = Path(chosen.get("path") or "")
        if not src.is_file():
            src = self.cfg.outputs_dir / req.source_job_id / chosen["filename"]
        if not src.is_file():
            raise RuntimeError(f"source image missing on disk: {src}")

        dest_name = f"sc_{req.source_job_id[:8]}_{chosen['filename']}"
        dest = input_dir / dest_name
        if not dest.is_file() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        return dest, dest_name

    def _stage_source_image(self, req: GenerateRequest) -> tuple[str, Path]:
        """Stage source into Comfy input/; return (load_name, source_path)."""
        path, name = self._resolve_source_path(req)
        return name, path

    async def _run_video(self, job: Job, *, finalize: bool = True) -> None:
        req = job.request
        try:
            pack = self._resolve_video_pack(req)
        except KeyError as e:
            job.status = JobStatus.failed
            job.error = str(e)
            self._persist(job)
            return

        profile_id = (
            req.video_profile or pack.default_profile or "draft"
        ).strip().lower()
        try:
            prof = pack.profile(profile_id)
        except KeyError as e:
            job.status = JobStatus.failed
            job.error = str(e)
            self._persist(job)
            return
        profile_id = prof.id

        arts = pack.artifacts_for(req.modality, profile_id)
        auto_pull = True if req.auto_pull is None else bool(req.auto_pull)

        # Optional video effects (same `effects` field as txt2img)
        try:
            video_effects = self.effects.resolve_for_video(
                req.effects,
                modality=req.modality,
                weights=req.effect_weights,
            )
        except (KeyError, ValueError) as e:
            job.status = JobStatus.failed
            job.phase = "failed"
            job.error = str(e)
            job.push_event({"type": "failed", "phase": "failed", "error": str(e)})
            self._persist(job)
            return

        job.status = JobStatus.pulling
        job.set_phase(
            "ensuring_models",
            message=f"Checking video models ({pack.id}/{profile_id})…",
            event_type="pulling",
            pack=pack.id,
            profile=profile_id,
            modality=req.modality,
            artifacts=len(arts),
            effects=[e.id for e in video_effects],
        )

        async def on_progress(ev: dict[str, Any]) -> None:
            self._raise_if_cancelled(job)
            self._apply_pull_progress(job, ev)

        try:
            self._raise_if_cancelled(job)
            resolved = await self.puller.ensure_video_artifacts(
                arts, auto_pull=auto_pull, on_progress=on_progress
            )
            # Effect LoRAs (may be multi-file high/low)
            for eff in video_effects:
                specs = eff.as_lora_specs()
                if specs:
                    await self.puller.ensure_lora_specs(
                        specs, auto_pull=auto_pull, on_progress=on_progress
                    )
        except JobCancelled:
            raise
        except PullError as e:
            job.status = JobStatus.failed
            job.phase = "failed"
            job.error = str(e)
            job.error_detail = e.detail
            job.push_event({"type": "failed", "phase": "failed", "error": str(e), "detail": e.detail})
            self._persist(job)
            return

        def name_for(filename: str) -> str:
            return resolved.get(filename) or filename

        def resolve_lora_name(filename: str, source: str) -> str:
            remembered = self.puller.resolved_name(
                role="lora", filename=filename or "", source=source or ""
            )
            if remembered:
                return remembered
            if filename:
                found = self.puller.find("lora", filename)
                if found:
                    return found.name
            return filename

        try:
            te = next(a for a in pack.shared if a.role == "text_encoder")
            vae = next(a for a in pack.shared if a.role == "vae")
            text_encoder = name_for(te.filename)
            vae_name = name_for(vae.filename)
            unets = pack.unets_for(req.modality)
            if len(unets) < 2:
                raise RuntimeError("video pack missing high/low unet pair")
            high_unet = name_for(unets[0].filename)
            low_unet = name_for(unets[1].filename)
            # Prefer noise tags if present
            by_noise = { (a.noise or "").lower(): name_for(a.filename) for a in unets }
            if by_noise.get("high") and by_noise.get("low"):
                high_unet, low_unet = by_noise["high"], by_noise["low"]
            hi_stack, lo_stack = pack.lora_stack_for(req.modality, profile_id)
            high_loras = [(name_for(n), w) for n, w in hi_stack]
            low_loras = [(name_for(n), w) for n, w in lo_stack]
            # Append optional effect LoRAs after profile stack
            eff_hi, eff_lo = effect_lora_stack(video_effects)
            for fn, w, src, _eid in eff_hi:
                high_loras.append((resolve_lora_name(fn, src), w))
            for fn, w, src, _eid in eff_lo:
                low_loras.append((resolve_lora_name(fn, src), w))
        except Exception as e:
            job.status = JobStatus.failed
            job.error = f"video_pack_invalid: {e}"
            self._persist(job)
            return

        start_image = None
        start_video = None
        source_path: Path | None = None
        if req.modality == "img2vid":
            try:
                if req.source_video_input:
                    # Multi-frame motion prefix already staged in Comfy input/
                    name = Path(req.source_video_input).name
                    if "/" in req.source_video_input or name.startswith("."):
                        raise RuntimeError("invalid source_video_input name")
                    input_dir = self.cfg.comfy_input_dir
                    if not input_dir:
                        raise RuntimeError("comfy.input_dir not set")
                    vp = input_dir / name
                    if not vp.is_file():
                        raise RuntimeError(f"source video not found: {name}")
                    start_video = name
                    source_path = vp
                else:
                    start_image, source_path = self._stage_source_image(req)
            except Exception as e:
                job.status = JobStatus.failed
                job.error = str(e)
                job.push_event({"type": "failed", "error": str(e)})
                self._persist(job)
                return

        self._raise_if_cancelled(job)
        job.set_phase("waiting_comfy", message="Starting ComfyUI…")
        if not await self._ensure_comfy(job):
            return

        d = pack.resolved_defaults(profile_id)
        pack_neg = d.negative or ""
        user_neg = req.negative_prompt
        # Effect prompt glue + pack negative; user negative appends
        pos_prompt, neg = merge_video_prompts(
            video_effects,
            req.prompt or "",
            user_neg,
            pack_negative=pack_neg,
        )
        prefix = f"simplecomfy/video/{job.id[:8]}"

        # Resolution: explicit request > source image (img2vid) > pack default
        width = req.width
        height = req.height
        if req.modality == "img2vid" and (width is None or height is None) and source_path:
            if start_video:
                # Prefer request/meta size; video probe is heavy — use pack if unset
                pass
            else:
                dims = probe_image_size(source_path)
                if dims:
                    sw, sh = fit_video_size(dims[0], dims[1], max_side=832)
                    if width is None:
                        width = sw
                    if height is None:
                        height = sh
                    job.meta["source_width"] = dims[0]
                    job.meta["source_height"] = dims[1]

        effect_meta = [
            {
                "id": e.id,
                "weight": e.weight,
                "exclusive_group": e.exclusive_group or None,
            }
            for e in video_effects
        ]

        if req.modality == "txt2vid":
            if not (req.prompt or "").strip():
                job.status = JobStatus.failed
                job.error = "prompt is required for txt2vid"
                self._persist(job)
                return
            workflow, meta = build_wan_txt2vid(
                prompt=pos_prompt,
                negative_prompt=neg or "",
                high_unet=high_unet,
                low_unet=low_unet,
                high_loras=high_loras,
                low_loras=low_loras,
                text_encoder=text_encoder,
                vae_name=vae_name,
                defaults=d,
                width=width,
                height=height,
                length=req.length,
                seed=req.seed,
                fps=req.fps,
                filename_prefix=prefix,
                profile=profile_id,
                context_windows=req.context_windows,
                context_length=req.context_length,
                context_overlap=req.context_overlap,
            )
        else:
            workflow, meta = build_wan_img2vid(
                prompt=pos_prompt,
                negative_prompt=neg or "",
                start_image_name=start_image or "",
                start_video_name=start_video,
                high_unet=high_unet,
                low_unet=low_unet,
                high_loras=high_loras,
                low_loras=low_loras,
                text_encoder=text_encoder,
                vae_name=vae_name,
                defaults=d,
                width=width,
                height=height,
                length=req.length,
                seed=req.seed,
                fps=req.fps,
                filename_prefix=prefix,
                profile=profile_id,
                context_windows=req.context_windows,
                context_length=req.context_length,
                context_overlap=req.context_overlap,
                motion_prefix_frames=req.motion_prefix_frames,
            )
            meta["source_job_id"] = req.source_job_id
            meta["source_filename"] = req.source_filename
            meta["source_input"] = req.source_input
            meta["source_video_input"] = req.source_video_input
            if job.meta.get("source_width"):
                meta["source_width"] = job.meta["source_width"]
                meta["source_height"] = job.meta["source_height"]

        meta["pack"] = pack.id
        meta["profile"] = profile_id
        meta["effects"] = effect_meta
        meta["positive"] = pos_prompt
        job.meta = meta
        # Full-step profiles need longer timeouts; draft drafts stay short
        timeout = self.cfg.comfy.video_timeout_s or self.cfg.comfy.timeout_s
        if profile_id not in FAST_PROFILES:
            timeout = max(timeout, 2400.0)
        # Long context-window jobs scale with frame count
        ctx = meta.get("context_windows") or {}
        frames = int(meta.get("length") or 81)
        if ctx.get("enabled") or frames > 81:
            # rough: more windows → more time
            timeout = max(timeout, min(7200.0, 900.0 + frames * 12.0))
        await self._execute_workflow(
            job, workflow, timeout_s=timeout, finalize=finalize
        )

    def _gallery_or_mem(self, job_id: str) -> dict[str, Any] | None:
        item = self.gallery.get(job_id)
        if item:
            return item
        mem = self._jobs.get(job_id)
        return mem.to_dict() if mem else None

    def _parent_video_path(self, parent_id: str) -> tuple[dict[str, Any], Path]:
        parent = self._gallery_or_mem(parent_id)
        if not parent:
            raise RuntimeError(f"parent job not found: {parent_id}")
        if parent.get("status") not in (None, "completed", JobStatus.completed.value):
            # allow missing status on legacy sidecars with media
            if not (parent.get("outputs") or parent.get("has_video")):
                raise RuntimeError(f"parent job not completed: {parent_id}")
        job_dir = self.cfg.outputs_dir / parent_id
        path = first_video_in_job(parent.get("outputs") or [], job_dir)
        if not path or not path.is_file():
            raise RuntimeError(f"parent job has no video output: {parent_id}")
        return parent, path

    async def _run_smooth(self, job: Job) -> None:
        """Temporal upsample parent mp4 (ffmpeg minterpolate) — no Comfy."""
        req = job.request
        parent_id = (req.parent_job_id or "").strip()
        if not parent_id:
            job.status = JobStatus.failed
            job.error = "parent_job_id required for smooth"
            self._persist(job)
            return
        factor = int(req.smooth_factor or 2)
        if factor not in (2, 4):
            job.status = JobStatus.failed
            job.error = "smooth_factor must be 2 or 4"
            self._persist(job)
            return

        job.status = JobStatus.running
        job.set_phase(
            "postprocess",
            message=f"Smoothing {factor}×…",
            event_type="running",
            modality="smooth",
            factor=factor,
        )
        try:
            parent, src = self._parent_video_path(parent_id)
        except RuntimeError as e:
            job.status = JobStatus.failed
            job.phase = "failed"
            job.error = str(e)
            job.push_event({"type": "failed", "phase": "failed", "error": str(e)})
            self._persist(job)
            return

        pmeta = parent.get("meta") or {}
        src_fps = req.fps or pmeta.get("fps")
        try:
            src_fps_f = float(src_fps) if src_fps is not None else None
        except (TypeError, ValueError):
            src_fps_f = None

        out_dir = self.cfg.outputs_dir / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "00.mp4"

        try:
            info = await asyncio.to_thread(
                smooth_video, src, dest, factor=factor, src_fps=src_fps_f
            )
        except VideoOpsError as e:
            job.status = JobStatus.failed
            job.phase = "failed"
            job.error = str(e)
            job.error_detail = e.detail
            job.push_event({"type": "failed", "phase": "failed", "error": str(e), "detail": e.detail})
            self._persist(job)
            return
        except Exception as e:
            job.status = JobStatus.failed
            job.phase = "failed"
            job.error = str(e)
            job.push_event({"type": "failed", "phase": "failed", "error": str(e)})
            self._persist(job)
            return

        job.outputs = [
            {
                "filename": "00.mp4",
                "kind": "video",
                "path": str(dest),
                "url": f"/v1/outputs/{job.id}/00.mp4",
                "bytes": dest.stat().st_size,
            }
        ]
        job.meta = {
            "modality": "smooth",
            "parent_job_id": parent_id,
            "smooth_factor": factor,
            "source_fps": info.get("source_fps"),
            "fps": info.get("output_fps"),
            "width": pmeta.get("width") or parent.get("width"),
            "height": pmeta.get("height") or parent.get("height"),
            "prompt": parent.get("prompt") or pmeta.get("positive") or "",
            "positive": parent.get("prompt") or pmeta.get("positive") or "",
            "profile": pmeta.get("profile"),
            "pack": pmeta.get("pack"),
            "seed": parent.get("seed") if parent.get("seed") is not None else pmeta.get("seed"),
        }
        job.status = JobStatus.completed
        job.phase = "done"
        job.progress = None
        job.push_event({"type": "completed", "phase": "done", "outputs": 1, "smooth": info})
        self._persist(job)

    async def _run_continue(self, job: Job) -> None:
        """
        Extend parent video with motion-aware I2V + FLF bridge (default).

        1. Multi-frame I2V prefix from parent tail (velocity).
        2. Strip locked prefix + warmup free frames.
        3. FLF bridge: real parent last frame → real chunk first frame
           (short WanFirstLastFrameToVideo morph — the “next lever”).
        4. Chain xfade: parent ⊕ bridge ⊕ chunk.
        """
        req = job.request
        parent_id = (req.parent_job_id or "").strip()
        if not parent_id:
            job.status = JobStatus.failed
            job.error = "parent_job_id required for continue"
            self._persist(job)
            return

        try:
            parent, parent_video = self._parent_video_path(parent_id)
        except RuntimeError as e:
            job.status = JobStatus.failed
            job.error = str(e)
            job.push_event({"type": "failed", "error": str(e)})
            self._persist(job)
            return

        pmeta = parent.get("meta") or {}
        # Inherit prompt / size / profile unless overridden
        prompt = (req.prompt or "").strip() or (
            parent.get("prompt") or pmeta.get("positive") or ""
        ).strip()
        neg = req.negative_prompt
        if neg is None:
            neg = parent.get("negative_prompt") or pmeta.get("negative")
        profile_id = (
            req.video_profile
            or pmeta.get("profile")
            or "draft"
        )
        if isinstance(profile_id, str):
            profile_id = profile_id.strip().lower()

        width = req.width or pmeta.get("width") or parent.get("width")
        height = req.height or pmeta.get("height") or parent.get("height")
        fps = req.fps if req.fps is not None else pmeta.get("fps")
        try:
            fps_f = float(fps) if fps is not None else 16.0
        except (TypeError, ValueError):
            fps_f = 16.0

        # New motion length (user-facing "extend by")
        new_length = req.length
        if new_length is None and req.continue_seconds is not None:
            n = max(1, int(round(float(req.continue_seconds) * fps_f)))
            new_length = max(5, ((n - 1) // 4) * 4 + 1)
        if new_length is None:
            new_length = int(pmeta.get("length") or DEFAULT_CONTINUE_LENGTH)
        new_length = max(5, int(new_length))

        # Motion prefix: last N frames of parent
        ctx_frames = req.continue_context_frames
        if ctx_frames is None:
            ctx_frames = DEFAULT_CONTINUE_CONTEXT_FRAMES
        ctx_frames = max(1, int(ctx_frames))

        warmup = req.continue_warmup_frames
        if warmup is None:
            warmup = DEFAULT_CONTINUE_WARMUP_FRAMES if ctx_frames > 1 else 0
        warmup = max(0, int(warmup))

        # Total I2V length = prefix + warmup discard + user-visible new frames
        # (warmup is generated free motion we throw away at the mask boundary)
        gen_body = new_length + warmup
        # Keep prefix from eating the whole gen; allow up to ~half of total
        max_prefix = max(1, gen_body)
        ctx_frames = min(ctx_frames, max_prefix)

        total_length = max(5, ((ctx_frames + gen_body - 1) // 4) * 4 + 1)
        if total_length < ctx_frames + warmup + 5:
            total_length = max(5, ((ctx_frames + warmup + 5 - 1) // 4) * 4 + 1)

        # Seam blend. None → default; 0 → hard cut.
        overlap_s = req.continue_overlap_seconds
        if overlap_s is None and req.continue_overlap_frames is None:
            overlap_s = DEFAULT_CONTINUE_OVERLAP_S
        max_frac = (
            req.continue_overlap_max_fraction
            if req.continue_overlap_max_fraction is not None
            else DEFAULT_CONTINUE_OVERLAP_MAX_FRACTION
        )
        try:
            max_frac = float(max_frac)
        except (TypeError, ValueError):
            max_frac = DEFAULT_CONTINUE_OVERLAP_MAX_FRACTION
        max_frac = max(0.05, min(0.49, max_frac))

        input_dir = self.cfg.comfy_input_dir
        if not input_dir:
            job.status = JobStatus.failed
            job.error = "comfy.input_dir not set — cannot stage continue media"
            self._persist(job)
            return
        input_dir.mkdir(parents=True, exist_ok=True)

        job.status = JobStatus.running
        job.set_phase(
            "extract_motion_prefix",
            message="Extracting motion prefix…",
            event_type="running",
            modality="continue",
            parent_job_id=parent_id,
            context_frames=ctx_frames,
            overlap_seconds=overlap_s,
        )

        use_video_prefix = ctx_frames > 1
        frame_name: str | None = None
        video_name: str | None = None
        try:
            if use_video_prefix:
                video_name = f"sc_cont_{job.id[:8]}_tail.mp4"
                tail_path = input_dir / video_name
                await asyncio.to_thread(
                    extract_tail_clip,
                    parent_video,
                    tail_path,
                    frames=ctx_frames,
                    fps=fps_f,
                )
            else:
                frame_name = f"sc_cont_{job.id[:8]}_last.png"
                await asyncio.to_thread(
                    extract_last_frame, parent_video, input_dir / frame_name
                )
        except VideoOpsError as e:
            job.status = JobStatus.failed
            job.error = str(e)
            job.error_detail = e.detail
            job.push_event({"type": "failed", "error": str(e), "detail": e.detail})
            self._persist(job)
            return

        chunk_req = GenerateRequest(
            preset="",
            prompt=prompt,
            negative_prompt=neg,
            seed=req.seed,
            width=int(width) if width else None,
            height=int(height) if height else None,
            auto_pull=req.auto_pull,
            effects=list(req.effects) if req.effects is not None else None,
            effect_weights=dict(req.effect_weights) if req.effect_weights else None,
            modality="img2vid",
            source_input=frame_name,
            source_video_input=video_name,
            motion_prefix_frames=ctx_frames if use_video_prefix else 1,
            length=int(total_length),
            fps=fps_f,
            video_pack=req.video_pack or pmeta.get("pack") or "sample",
            video_profile=str(profile_id),
            parent_job_id=parent_id,
            context_windows=req.context_windows,
            context_length=req.context_length,
            context_overlap=req.context_overlap,
        )
        job.request = chunk_req
        job.meta["parent_job_id"] = parent_id
        job.meta["continue_from"] = parent_id
        job.meta["continue_phase"] = "generate_chunk"
        job.meta["continue_context_frames"] = ctx_frames
        job.meta["continue_warmup_frames"] = warmup
        job.meta["continue_new_length"] = new_length
        job.meta["continue_total_length"] = total_length
        job.meta["continue_overlap_seconds"] = overlap_s
        job.meta["continue_overlap_max_fraction"] = max_frac
        job.meta["motion_prefix"] = use_video_prefix
        job.set_phase("generate_chunk", message="Generating continue chunk…")
        job.push_event(
            {
                "type": "continue_chunk",
                "phase": "generate_chunk",
                "profile": profile_id,
                "length": total_length,
                "context_frames": ctx_frames,
                "warmup_frames": warmup,
                "new_length": new_length,
                "motion_prefix": use_video_prefix,
                "overlap_seconds": overlap_s,
            }
        )

        # Do not finalize — FLF bridge / join still to run
        await self._run_video(job, finalize=False)
        if job.status == JobStatus.failed:
            job.meta["modality"] = "continue"
            job.request.modality = "continue"
            self._persist(job)
            return
        if not job.outputs:
            out_scan = self.cfg.outputs_dir / job.id
            found = first_video_in_job([], out_scan)
            if found:
                job.outputs = [
                    {
                        "filename": found.name,
                        "kind": "video",
                        "path": str(found),
                        "url": f"/v1/outputs/{job.id}/{found.name}",
                        "bytes": found.stat().st_size,
                    }
                ]

        job.status = JobStatus.running
        job.meta["continue_phase"] = "post"
        method = (req.continue_method or "flf_bridge").strip().lower()
        if method not in ("flf_bridge", "prefix"):
            method = "flf_bridge"

        bridge_frames = req.continue_bridge_frames
        if bridge_frames is None:
            bridge_frames = DEFAULT_CONTINUE_BRIDGE_FRAMES
        bridge_frames = max(9, ((int(bridge_frames) - 1) // 4) * 4 + 1)

        # Strip locked prefix + warmup free frames (boundary is often the worst)
        strip = (ctx_frames + warmup) if use_video_prefix else warmup
        job.status = JobStatus.running
        job.push_event(
            {
                "type": "continue_post",
                "parent_job_id": parent_id,
                "method": method,
                "strip_prefix_frames": strip,
                "context_frames": ctx_frames,
                "warmup_frames": warmup,
                "bridge_frames": bridge_frames if method == "flf_bridge" else 0,
                "overlap_seconds": overlap_s,
            }
        )

        out_dir = self.cfg.outputs_dir / job.id
        chunk_path = first_video_in_job(job.outputs, out_dir)
        if not chunk_path:
            job.status = JobStatus.failed
            job.error = "continue chunk produced no video"
            job.meta["modality"] = "continue"
            job.request.modality = "continue"
            self._persist(job)
            return

        chunk_raw = out_dir / "chunk_raw.mp4"
        chunk_keep = out_dir / "chunk.mp4"
        bridge_path = out_dir / "bridge.mp4"
        final_path = out_dir / "00.mp4"
        # Preserve continue knobs across _run_video meta overwrite
        cont_meta = {
            "continue_context_frames": ctx_frames,
            "continue_warmup_frames": warmup,
            "continue_new_length": new_length,
            "continue_total_length": total_length,
            "continue_overlap_seconds": overlap_s,
            "continue_overlap_max_fraction": max_frac,
            "continue_method": method,
            "continue_bridge_frames": bridge_frames if method == "flf_bridge" else 0,
            "motion_prefix": use_video_prefix,
            "parent_job_id": parent_id,
            "continue_from": parent_id,
        }
        work_chunk = out_dir / "_chunk_work.mp4"
        try:
            shutil.copy2(chunk_path, chunk_raw)
            if chunk_path.name == "00.mp4" and chunk_path.is_file():
                try:
                    chunk_path.unlink()
                except OSError:
                    pass

            if strip > 0:
                await asyncio.to_thread(
                    trim_start_frames,
                    chunk_raw,
                    chunk_keep,
                    skip_frames=strip,
                    fps=fps_f,
                )
            else:
                shutil.copy2(chunk_raw, chunk_keep)

            info: dict[str, Any]
            if method == "flf_bridge":
                # --- FLF bridge: real parent tail frame → real chunk head frame ---
                job.meta["continue_phase"] = "flf_bridge"
                job.set_phase(
                    "flf_bridge",
                    message=f"FLF bridge ({bridge_frames}f)…",
                    event_type="continue_bridge",
                    frames=bridge_frames,
                )
                input_dir = self.cfg.comfy_input_dir
                assert input_dir is not None
                start_png = input_dir / f"sc_br_{job.id[:8]}_start.png"
                end_png = input_dir / f"sc_br_{job.id[:8]}_end.png"
                await asyncio.to_thread(extract_last_frame, parent_video, start_png)
                await asyncio.to_thread(extract_first_frame, chunk_keep, end_png)

                # Reuse weights already resolved in job.meta from chunk gen
                m = job.meta or {}
                pack = self._resolve_video_pack(
                    GenerateRequest(
                        preset="",
                        prompt=prompt,
                        modality="img2vid",
                        video_pack=req.video_pack or pmeta.get("pack") or "sample",
                    )
                )
                d = pack.resolved_defaults(profile_id)
                hi_stack, lo_stack = pack.lora_stack_for("img2vid", profile_id)
                # Prefer names Comfy already saw on the chunk run
                high_unet = str(m.get("high_unet") or "")
                low_unet = str(m.get("low_unet") or "")
                text_encoder = str(m.get("text_encoder") or "")
                vae_name = str(m.get("vae") or "")
                if not (high_unet and low_unet and text_encoder and vae_name):
                    arts = pack.artifacts_for("img2vid", profile_id)
                    resolved = await self.puller.ensure_video_artifacts(
                        arts, auto_pull=True
                    )

                    def _n(fn: str) -> str:
                        return resolved.get(fn) or fn

                    te = next(a for a in pack.shared if a.role == "text_encoder")
                    vae = next(a for a in pack.shared if a.role == "vae")
                    text_encoder, vae_name = _n(te.filename), _n(vae.filename)
                    unets = pack.unets_for("img2vid")
                    by_n = {(a.noise or "").lower(): _n(a.filename) for a in unets}
                    high_unet = by_n.get("high") or _n(unets[0].filename)
                    low_unet = by_n.get("low") or _n(unets[1].filename)
                    hi_loras = [(_n(n), w) for n, w in hi_stack]
                    lo_loras = [(_n(n), w) for n, w in lo_stack]
                else:
                    # lora basenames from meta
                    raw_l = m.get("loras") or []
                    hi_loras = [
                        (str(x.get("name")), float(x.get("weight") or 1.0))
                        for x in raw_l
                        if isinstance(x, dict) and x.get("noise") == "high"
                    ]
                    lo_loras = [
                        (str(x.get("name")), float(x.get("weight") or 1.0))
                        for x in raw_l
                        if isinstance(x, dict) and x.get("noise") == "low"
                    ]

                bridge_prompt = (
                    f"{prompt}, smooth continuous motion, seamless transition, "
                    f"no cut, no freeze frame, natural continuation"
                ).strip(", ")
                bw = int(width) if width else int(m.get("width") or d.width)
                bh = int(height) if height else int(m.get("height") or d.height)
                bridge_wf, bridge_meta = build_wan_flf2vid(
                    prompt=bridge_prompt,
                    negative_prompt=neg or d.negative or "",
                    start_image_name=start_png.name,
                    end_image_name=end_png.name,
                    high_unet=high_unet,
                    low_unet=low_unet,
                    high_loras=hi_loras,
                    low_loras=lo_loras,
                    text_encoder=text_encoder,
                    vae_name=vae_name,
                    defaults=d,
                    width=bw,
                    height=bh,
                    length=bridge_frames,
                    seed=req.seed,
                    fps=fps_f,
                    filename_prefix=f"simplecomfy/video/{job.id[:8]}_bridge",
                    profile=profile_id,
                )
                timeout = self.cfg.comfy.video_timeout_s or self.cfg.comfy.timeout_s
                if profile_id not in FAST_PROFILES:
                    timeout = max(timeout, 2400.0)
                bridge_outs = await self._queue_and_download(
                    job,
                    bridge_wf,
                    timeout_s=timeout,
                    name_prefix="bridge_",
                    finalize=False,
                )
                if not bridge_outs:
                    raise RuntimeError("FLF bridge produced no video")
                # Move primary bridge output to bridge.mp4
                bsrc = Path(bridge_outs[0]["path"])
                if bsrc.is_file():
                    shutil.copy2(bsrc, bridge_path)

                # Trim locked endpoints of the bridge so xfades land on free frames
                bridge_trim = min(
                    DEFAULT_CONTINUE_BRIDGE_TRIM,
                    max(0, (bridge_frames // 8)),
                )
                bridge_mid = out_dir / "bridge_mid.mp4"
                if bridge_trim > 0 and bridge_path.is_file():
                    tmp_b = out_dir / "_bridge_trim.mp4"
                    await asyncio.to_thread(
                        trim_start_frames,
                        bridge_path,
                        tmp_b,
                        skip_frames=bridge_trim,
                        fps=fps_f,
                    )
                    # also drop last bridge_trim via reverse-trim-reverse
                    await asyncio.to_thread(
                        trim_end_frames,
                        tmp_b,
                        bridge_mid,
                        drop=bridge_trim,
                        fps=fps_f,
                    )
                    if tmp_b.is_file():
                        try:
                            tmp_b.unlink()
                        except OSError:
                            pass
                else:
                    shutil.copy2(bridge_path, bridge_mid)

                # Drop first frame of chunk (matched by bridge end) for cleaner splice
                chunk_tail = out_dir / "chunk_tail.mp4"
                await asyncio.to_thread(
                    trim_start_frames,
                    chunk_keep,
                    chunk_tail,
                    skip_frames=1,
                    fps=fps_f,
                )

                job.set_phase("joining", message="Joining clips…")
                info = await asyncio.to_thread(
                    concat_chain_xfade,
                    [parent_video, bridge_mid, chunk_tail],
                    final_path,
                    overlap_seconds=float(overlap_s) if overlap_s is not None else 0.75,
                    fps=fps_f,
                    max_fraction=max_frac,
                )
                info["motion_prefix_frames"] = ctx_frames
                info["warmup_frames"] = warmup
                info["stripped_frames"] = strip
                info["bridge_frames"] = bridge_frames
                info["bridge_trim"] = bridge_trim
                info["method"] = "flf_bridge"
                cont_meta["bridge_meta"] = {
                    k: bridge_meta.get(k)
                    for k in ("seed", "steps", "length", "profile", "start_image", "end_image")
                }
            else:
                # Legacy: direct parent ⊕ chunk xfade
                shutil.copy2(chunk_keep, work_chunk)
                job.set_phase("joining", message="Crossfading seam…")
                info = await asyncio.to_thread(
                    concat_overlap_crossfade,
                    parent_video,
                    work_chunk,
                    final_path,
                    overlap_seconds=float(overlap_s) if overlap_s is not None else 0.0,
                    overlap_frames=req.continue_overlap_frames,
                    fps=fps_f,
                    max_fraction=max_frac,
                )
                info["motion_prefix_frames"] = ctx_frames
                info["warmup_frames"] = warmup
                info["stripped_frames"] = strip
                info["method"] = "prefix"
        except VideoOpsError as e:
            job.status = JobStatus.failed
            job.error = f"continue concat failed: {e}"
            job.error_detail = e.detail
            job.meta["modality"] = "continue"
            job.request.modality = "continue"
            job.push_event({"type": "failed", "error": str(e), "detail": e.detail})
            self._persist(job)
            return
        except Exception as e:
            job.status = JobStatus.failed
            job.error = f"continue post failed: {e}"
            job.meta["modality"] = "continue"
            job.request.modality = "continue"
            job.push_event({"type": "failed", "error": str(e)})
            self._persist(job)
            return
        finally:
            for tmp in (work_chunk, out_dir / "_bridge_trim.mp4"):
                if tmp.is_file():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

        outputs = [
            {
                "filename": "00.mp4",
                "kind": "video",
                "path": str(final_path),
                "url": f"/v1/outputs/{job.id}/00.mp4",
                "bytes": final_path.stat().st_size,
            }
        ]
        for extra in ("chunk.mp4", "chunk_raw.mp4", "bridge.mp4", "bridge_mid.mp4"):
            ep = out_dir / extra
            if ep.is_file():
                outputs.append(
                    {
                        "filename": extra,
                        "kind": "video",
                        "path": str(ep),
                        "url": f"/v1/outputs/{job.id}/{extra}",
                        "bytes": ep.stat().st_size,
                    }
                )
        job.outputs = outputs
        base = dict(job.meta or {})
        base.update(cont_meta)
        base["modality"] = "continue"
        base["concat"] = info
        base["continue_overlap_seconds"] = info.get("overlap_seconds", overlap_s)
        if info.get("step_overlaps"):
            base["continue_overlap_seconds"] = overlap_s
        base["prompt"] = prompt
        base["positive"] = prompt
        base["fps"] = fps_f
        base["profile"] = profile_id
        base["length"] = total_length
        base["continue_phase"] = "done"
        job.meta = base
        job.request.modality = "continue"
        job.request.prompt = prompt
        job.status = JobStatus.completed
        job.phase = "done"
        job.progress = None
        job.push_event(
            {"type": "completed", "phase": "done", "outputs": len(outputs), "concat": info}
        )
        self._persist(job)

    async def enqueue_smooth(
        self, parent_job_id: str, *, factor: int = 2
    ) -> Job:
        return await self.enqueue(
            GenerateRequest(
                preset="",
                prompt="",
                modality="smooth",
                parent_job_id=parent_job_id,
                smooth_factor=factor,
            )
        )

    async def enqueue_continue(
        self,
        parent_job_id: str,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        video_profile: str | None = None,
        length: int | None = None,
        continue_seconds: float | None = None,
        continue_overlap_seconds: float | None = None,
        continue_overlap_frames: int | None = None,
        continue_overlap_max_fraction: float | None = None,
        continue_context_frames: int | None = None,
        continue_method: str | None = None,
        continue_bridge_frames: int | None = None,
        continue_warmup_frames: int | None = None,
        seed: int = -1,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        video_pack: str | None = None,
        auto_pull: bool | None = None,
        effects: list[str] | None = None,
        effect_weights: dict[str, float] | None = None,
        context_windows: bool | None = None,
        context_length: int | None = None,
        context_overlap: int | None = None,
    ) -> Job:
        return await self.enqueue(
            GenerateRequest(
                preset="",
                prompt=prompt or "",
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                auto_pull=auto_pull,
                effects=effects,
                effect_weights=effect_weights,
                modality="continue",
                parent_job_id=parent_job_id,
                length=length,
                fps=fps,
                video_pack=video_pack,
                video_profile=video_profile,
                continue_seconds=continue_seconds,
                continue_overlap_seconds=continue_overlap_seconds,
                continue_overlap_frames=continue_overlap_frames,
                continue_overlap_max_fraction=continue_overlap_max_fraction,
                continue_context_frames=continue_context_frames,
                continue_method=continue_method,
                continue_bridge_frames=continue_bridge_frames,
                continue_warmup_frames=continue_warmup_frames,
                context_windows=context_windows,
                context_length=context_length,
                context_overlap=context_overlap,
            )
        )

    def _apply_pull_progress(self, job: Job, ev: dict[str, Any]) -> None:
        """Map puller progress events onto job.phase / progress."""
        phase = str(ev.get("phase") or "")
        name = ev.get("name") or ev.get("role") or ""
        if phase == "download":
            cur = ev.get("bytes_done") or ev.get("downloaded") or ev.get("current")
            tot = ev.get("bytes_total") or ev.get("total")
            try:
                cur_i = int(cur) if cur is not None else None
                tot_i = int(tot) if tot is not None else None
            except (TypeError, ValueError):
                cur_i, tot_i = None, None
            msg = f"Downloading {name}" if name else "Downloading…"
            if cur_i is not None and tot_i:
                pct = int(100 * cur_i / tot_i) if tot_i else 0
                msg = f"{msg} ({pct}%)"
            job.set_phase(
                "downloading",
                message=msg,
                current=cur_i,
                total=tot_i,
                unit="byte",
                event_type="pulling",
                name=name,
                role=ev.get("role"),
                source=ev.get("source"),
            )
        elif phase == "start":
            job.set_phase(
                "ensuring_models",
                message=f"Fetching {name}…" if name else "Checking models…",
                event_type="pulling",
                name=name,
                role=ev.get("role"),
                source=ev.get("source"),
            )
        elif phase == "done":
            job.set_phase(
                "ensuring_models",
                message=f"Ready: {name}" if name else "Model ready",
                event_type="pulling",
                name=name,
                role=ev.get("role"),
            )
        else:
            job.push_event(ev)

    async def _ensure_comfy(self, job: Job) -> bool:
        sup = await self.supervisor.ensure_running()
        if not sup.get("reachable"):
            job.status = JobStatus.failed
            job.phase = "failed"
            job.error = "comfy_unreachable"
            job.error_detail = sup
            job.push_event(
                {
                    "type": "failed",
                    "phase": "failed",
                    "error": "comfy_unreachable",
                    "detail": sup,
                }
            )
            self._persist(job)
            return False
        return True

    async def _queue_and_download(
        self,
        job: Job,
        workflow: dict[str, Any],
        *,
        timeout_s: float,
        name_prefix: str = "",
        finalize: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Queue a Comfy workflow and download outputs into the job directory.
        When finalize=False, leaves job in running state (multi-phase jobs).
        """
        job.status = JobStatus.running
        job.set_phase("waiting_comfy", message="Queued in ComfyUI…")

        async def on_comfy_progress(ev: dict[str, Any]) -> None:
            self._raise_if_cancelled(job)
            phase = str(ev.get("phase") or "sampling")
            job.set_phase(
                phase,
                message=str(ev.get("message") or "Sampling…"),
                current=ev.get("current"),
                total=ev.get("total"),
                unit=ev.get("unit") or "step",
                event_type=str(ev.get("type") or "comfy_progress"),
                node=ev.get("node"),
            )

        self._raise_if_cancelled(job)
        prompt_id, client_id = await self.client.queue_prompt(workflow)
        job.meta["comfy_prompt_id"] = prompt_id
        job.meta["comfy_client_id"] = client_id
        if name_prefix:
            job.meta[f"comfy_prompt_id_{name_prefix.rstrip('_')}"] = prompt_id
        job.set_phase("sampling", message="Generating…")
        try:
            history = await self.client.wait_history(
                prompt_id,
                timeout_s=timeout_s,
                client_id=client_id,
                on_progress=on_comfy_progress,
                should_cancel=lambda: self.is_cancel_requested(job.id)
                or bool(job.meta.get("cancel_requested")),
            )
        except ComfyError as e:
            if str(e) == "cancelled" or self.is_cancel_requested(job.id):
                raise JobCancelled(job.id) from e
            raise
        media = self.client.extract_media(history)
        outputs: list[dict[str, Any]] = []
        out_dir = self.cfg.outputs_dir / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        job.set_phase(
            "saving",
            message="Saving outputs…",
            current=0,
            total=max(1, len(media)),
            unit="item",
        )
        for i, item in enumerate(media):
            data = await self.client.download_image(
                filename=item["filename"],
                subfolder=item.get("subfolder") or "",
                folder_type=item.get("type") or "output",
            )
            ext = Path(item["filename"]).suffix or (
                ".mp4" if item.get("kind") == "video" else ".png"
            )
            local_name = f"{name_prefix}{i:02d}{ext}"
            kind = (
                media_kind(local_name)
                or media_kind(item["filename"])
                or item.get("kind")
                or "image"
            )
            path = out_dir / local_name
            path.write_bytes(data)
            outputs.append(
                {
                    "filename": local_name,
                    "kind": kind,
                    "path": str(path),
                    "url": f"/v1/outputs/{job.id}/{local_name}",
                    "bytes": len(data),
                }
            )
            job.set_phase(
                "saving",
                message=f"Saving {i + 1}/{len(media)}…",
                current=i + 1,
                total=len(media),
                unit="item",
            )
        if not outputs:
            if finalize:
                job.status = JobStatus.failed
                job.phase = "failed"
                job.error = "no_outputs"
                job.error_detail = {"hint": "Comfy finished but produced no media files"}
                job.push_event({"type": "failed", "phase": "failed", "error": "no_outputs"})
                self._persist(job)
            return []
        if finalize:
            job.outputs = outputs
            job.status = JobStatus.completed
            job.phase = "done"
            job.progress = None
            job.push_event(
                {"type": "completed", "phase": "done", "outputs": len(outputs)}
            )
            self._persist(job)
        else:
            job.push_event(
                {
                    "type": "phase_outputs",
                    "prefix": name_prefix,
                    "outputs": len(outputs),
                }
            )
        return outputs

    async def _execute_workflow(
        self,
        job: Job,
        workflow: dict[str, Any],
        *,
        timeout_s: float,
        finalize: bool = True,
    ) -> None:
        job.status = JobStatus.running
        job.set_phase(
            "waiting_comfy",
            message="Submitting workflow…",
            event_type="running",
            seed=job.meta.get("seed"),
            modality=job.meta.get("modality"),
        )
        try:
            outs = await self._queue_and_download(
                job, workflow, timeout_s=timeout_s, finalize=finalize
            )
            if not finalize and outs:
                job.outputs = outs
        except JobCancelled:
            raise
        except ComfyError as e:
            if str(e) == "cancelled" or self.is_cancel_requested(job.id):
                raise JobCancelled(job.id) from e
            job.status = JobStatus.failed
            job.phase = "failed"
            job.error = str(e)
            job.error_detail = e.detail
            job.push_event(
                {
                    "type": "failed",
                    "phase": "failed",
                    "error": str(e),
                    "detail": e.detail,
                }
            )
            self._persist(job)
        except Exception as e:
            if self.is_cancel_requested(job.id):
                raise JobCancelled(job.id) from e
            job.status = JobStatus.failed
            job.phase = "failed"
            job.error = str(e)
            job.push_event({"type": "failed", "phase": "failed", "error": str(e)})
            self._persist(job)
            raise
