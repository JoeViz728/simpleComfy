"""Persistent output gallery — scan data/outputs for media + job.json sidecars."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simplecomfy.config import AppConfig

JOB_META_NAME = "job.json"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv"}


def _utc_from_mtime(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def media_kind(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    # Video first — never treat .mp4 as image
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    return None


def normalize_output(job_id: str, o: dict[str, Any]) -> dict[str, Any]:
    """Ensure url/kind are consistent (kind always follows file extension)."""
    out = dict(o)
    fn = str(out.get("filename") or "")
    inferred = media_kind(fn) if fn else None
    if inferred:
        out["kind"] = inferred
    elif not out.get("kind"):
        out["kind"] = "image"
    if fn and not out.get("url"):
        out["url"] = f"/v1/outputs/{job_id}/{fn}"
    return out


def list_media_files(job_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not job_dir.is_dir():
        return items
    for p in sorted(job_dir.iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.name == JOB_META_NAME:
            continue
        kind = media_kind(p.name)
        if not kind:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        items.append(
            {
                "filename": p.name,
                "kind": kind,
                "bytes": size,
                "url": f"/v1/outputs/{job_dir.name}/{p.name}",
                "path": str(p),
            }
        )
    return items


def write_job_sidecar(job_dir: Path, payload: dict[str, Any]) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / JOB_META_NAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_job_sidecar(job_dir: Path) -> dict[str, Any] | None:
    path = job_dir / JOB_META_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def gallery_item_from_dir(job_dir: Path) -> dict[str, Any] | None:
    """Build a gallery card from a job output directory."""
    if not job_dir.is_dir() or not UUID_RE.match(job_dir.name):
        return None
    media = list_media_files(job_dir)
    meta = load_job_sidecar(job_dir)

    if not media and not meta:
        return None

    # Prefer sidecar; fill gaps from filesystem
    created = None
    if meta:
        created = meta.get("created_at") or meta.get("updated_at")
    if not created and media:
        try:
            created = _utc_from_mtime(job_dir / media[0]["filename"])
        except OSError:
            created = _utc_from_mtime(job_dir)
    elif not created:
        created = _utc_from_mtime(job_dir)

    if meta:
        raw_outputs = meta.get("outputs") or media
        # Prefer on-disk media list when sidecar kinds/paths are stale
        by_name = {m["filename"]: m for m in media}
        outputs = []
        seen_names: set[str] = set()
        for o in raw_outputs:
            fn = str(o.get("filename") or "")
            if fn and fn in by_name:
                merged = {**o, **by_name[fn]}
                outputs.append(normalize_output(job_dir.name, merged))
                seen_names.add(fn)
            else:
                outputs.append(normalize_output(job_dir.name, o))
                if fn:
                    seen_names.add(fn)
        for m in media:
            if m["filename"] not in seen_names:
                outputs.append(normalize_output(job_dir.name, m))
        item = {
            "id": job_dir.name,
            "status": meta.get("status") or "completed",
            "created_at": created,
            "updated_at": meta.get("updated_at") or created,
            "modality": meta.get("modality") or "txt2img",
            "preset": meta.get("preset"),
            "prompt": meta.get("prompt") or "",
            "negative_prompt": meta.get("negative_prompt"),
            "seed": meta.get("seed"),
            "parent_job_id": meta.get("parent_job_id"),
            "effects": meta.get("effects"),
            "meta": meta.get("meta") or {},
            "outputs": outputs,
            "error": meta.get("error"),
            "has_sidecar": True,
        }
    else:
        # Legacy folders: media only, no recipe
        item = {
            "id": job_dir.name,
            "status": "completed",
            "created_at": created,
            "updated_at": created,
            "modality": "txt2img",
            "preset": None,
            "prompt": "",
            "negative_prompt": None,
            "seed": None,
            "effects": None,
            "meta": {},
            "outputs": [normalize_output(job_dir.name, m) for m in media],
            "error": None,
            "has_sidecar": False,
        }

    # Thumb = first image, else first video (with kind for the UI)
    thumb = None
    thumb_kind = None
    for o in item["outputs"]:
        if o.get("kind") == "image":
            thumb, thumb_kind = o.get("url"), "image"
            break
    if not thumb:
        for o in item["outputs"]:
            if o.get("kind") == "video":
                thumb, thumb_kind = o.get("url"), "video"
                break
    if not thumb and item["outputs"]:
        o0 = item["outputs"][0]
        thumb, thumb_kind = o0.get("url"), o0.get("kind") or "image"
    item["thumb_url"] = thumb
    item["thumb_kind"] = thumb_kind
    item["has_video"] = any(o.get("kind") == "video" for o in item["outputs"])
    item["has_image"] = any(o.get("kind") == "image" for o in item["outputs"])

    # Flatten common meta for gallery cards / “use in”
    m = item.get("meta") or {}
    item["checkpoint"] = m.get("checkpoint")
    item["loras"] = m.get("loras") or []
    item["width"] = m.get("width")
    item["height"] = m.get("height")
    item["positive"] = m.get("positive") or item.get("prompt")
    item["negative"] = m.get("negative") or item.get("negative_prompt")
    item["steps"] = m.get("steps")
    item["cfg"] = m.get("cfg")
    item["sampler_name"] = m.get("sampler_name")
    item["scheduler"] = m.get("scheduler")
    item["profile"] = m.get("profile")
    item["fps"] = m.get("fps")
    item["length"] = m.get("length")
    item["parent_job_id"] = (
        meta.get("parent_job_id") if meta else None
    ) or m.get("parent_job_id")
    item["smooth_factor"] = m.get("smooth_factor")
    # Resolved effect ids from meta if request effects missing
    if item.get("effects") is None and m.get("effects"):
        item["effects"] = [
            e.get("id") for e in m["effects"] if isinstance(e, dict) and e.get("id")
        ]
    return item


class GalleryStore:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    @property
    def root(self) -> Path:
        return self.cfg.outputs_dir

    def list_items(
        self,
        *,
        limit: int = 48,
        offset: int = 0,
        modality: str | None = None,
        has_image: bool | None = None,
        has_video: bool | None = None,
        has_media: bool | None = None,
    ) -> dict[str, Any]:
        dirs = [
            p
            for p in self.root.iterdir()
            if p.is_dir() and UUID_RE.match(p.name)
        ] if self.root.is_dir() else []

        items: list[dict[str, Any]] = []
        for d in dirs:
            item = gallery_item_from_dir(d)
            if not item:
                continue
            if modality and item.get("modality") != modality:
                continue
            outs = item.get("outputs") or []
            if has_image is True and not any(o.get("kind") == "image" for o in outs):
                continue
            if has_video is True and not any(o.get("kind") == "video" for o in outs):
                continue
            if has_media is True and not outs:
                continue
            items.append(item)

        items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        total = len(items)
        page = items[offset : offset + limit]
        return {"items": page, "total": total, "limit": limit, "offset": offset}

    def get(self, job_id: str) -> dict[str, Any] | None:
        if not UUID_RE.match(job_id):
            return None
        return gallery_item_from_dir(self.root / job_id)

    def persist_from_job_dict(self, job: dict[str, Any]) -> Path | None:
        """Write sidecar for a completed/failed job dict (Job.to_dict shape)."""
        jid = job.get("id")
        if not jid:
            return None
        job_dir = self.root / str(jid)
        meta = job.get("meta") or {}
        payload = {
            "id": jid,
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "modality": meta.get("modality") or job.get("modality") or "txt2img",
            "preset": job.get("preset"),
            "prompt": job.get("prompt") or meta.get("prompt") or meta.get("positive"),
            "negative_prompt": job.get("negative_prompt"),
            "seed": job.get("seed"),
            "parent_job_id": job.get("parent_job_id") or meta.get("parent_job_id"),
            "effects": job.get("effects")
            if job.get("effects") is not None
            else [
                e.get("id")
                for e in (meta.get("effects") or [])
                if isinstance(e, dict)
            ],
            "meta": meta,
            "outputs": job.get("outputs") or [],
            "error": job.get("error"),
            "error_detail": job.get("error_detail"),
        }
        return write_job_sidecar(job_dir, payload)
