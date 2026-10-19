"""FastAPI application — small surface for humans and LLMs."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simplecomfy import __version__
from simplecomfy.comfy_client import ComfyClient, ComfyError, write_extra_model_paths
from simplecomfy.config import AppConfig, load_config
from simplecomfy.effects import EffectRegistry, summarize_effect
from simplecomfy.history import GalleryStore
from simplecomfy.jobs import (
    IMPLEMENTED_MODALITIES,
    MODALITIES,
    GenerateRequest,
    JobManager,
)
from simplecomfy.presets import PresetRegistry
from simplecomfy.puller import ModelPuller, PullError, summarize_preset_status
from simplecomfy.supervisor import ComfySupervisor
from simplecomfy.video_pack import VideoPackRegistry


class GenerateBody(BaseModel):
    modality: str = Field(
        default="txt2img",
        description="txt2img | txt2vid | img2vid | smooth | continue",
    )
    preset: str | None = Field(
        default=None,
        description="Preset id (required for txt2img)",
    )
    prompt: str = Field(default="", description="User prompt / motion prompt")
    negative_prompt: str | None = None
    seed: int = -1
    width: int | None = None
    height: int | None = None
    effects: list[str] | None = Field(
        default=None,
        description=(
            "Optional effect ids from GET /v1/effects. "
            "txt2img: omit = preset.default_effects; [] = none. "
            "Video (txt2vid/img2vid/continue): omit or [] = none; "
            "ids are optional motion/act LoRAs (not profile baseline)."
        ),
    )
    effect_weights: dict[str, float] | None = Field(
        default=None,
        description='Optional per-effect LoRA weight overrides, e.g. {"soft_light": 0.55}',
    )
    source_job_id: str | None = Field(
        default=None,
        description="Gallery job id for img2vid source frame",
    )
    source_filename: str | None = Field(
        default=None,
        description="Media filename within source job (default: first image)",
    )
    source_input: str | None = Field(
        default=None,
        description="Uploaded input basename from POST /v1/inputs (img2vid)",
    )
    length: int | None = Field(default=None, description="Video frames (WAN: 4n+1, default 81)")
    fps: float | None = Field(default=None, description="Output FPS (default 16)")
    video_pack: str | None = Field(default=None, description="Video pack id (default sample)")
    video_profile: str | None = Field(
        default=None,
        description=(
            "Video profile: draft | final | fast | quality "
            "(default pack.default_profile = draft)"
        ),
    )
    context_windows: bool | None = Field(
        default=None,
        description=(
            "Long-clip context windows (WanContextWindowsManual). "
            "null=auto (on when length > context_length), true=force on, false=off"
        ),
    )
    context_length: int | None = Field(
        default=None,
        description="Context window size in frames (4n+1, default 81)",
    )
    context_overlap: int | None = Field(
        default=None,
        description=(
            "Frames shared between adjacent windows (continuity). "
            "Default 30; higher = smoother long-clip joins, more compute"
        ),
    )
    parent_job_id: str | None = Field(
        default=None,
        description="Parent gallery/job id for smooth or continue",
    )
    smooth_factor: int | None = Field(
        default=None,
        description="For modality=smooth: 2 or 4 (frame interpolation factor)",
    )
    continue_seconds: float | None = Field(
        default=None,
        description="For modality=continue: approximate extra seconds (mapped to WAN length)",
    )
    continue_overlap_seconds: float | None = Field(
        default=None,
        description=(
            "Continue seam blend window in seconds (default 0.75). "
            "0 = hard cut. Clamped to ≤40% of either clip so long parents "
            "do not force a huge blend."
        ),
    )
    continue_overlap_frames: int | None = Field(
        default=None,
        description="Alternate to continue_overlap_seconds (takes priority if set)",
    )
    continue_overlap_max_fraction: float | None = Field(
        default=None,
        ge=0.05,
        le=0.49,
        description="Max fraction of parent or chunk duration the blend may use (default 0.4)",
    )
    continue_context_frames: int | None = Field(
        default=None,
        description=(
            "Last N frames of parent fed as multi-frame I2V start (motion prefix). "
            "Default 16; 1 = still-only continue. Higher = more velocity continuity."
        ),
    )
    continue_method: str | None = Field(
        default=None,
        description="Continue join: flf_bridge (default) | prefix",
    )
    continue_bridge_frames: int | None = Field(
        default=None,
        description="FLF bridge length in frames (4n+1, default 25)",
    )
    continue_warmup_frames: int | None = Field(
        default=None,
        description="Free frames dropped after locked prefix (default 8)",
    )
    auto_pull: bool | None = Field(
        default=None,
        description=(
            "Override auto-pull for this job. Default null = always pull missing "
            "models as a preliminary generate phase."
        ),
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description=(
            "Enqueue N independent jobs with the same settings. "
            "If seed >= 0, jobs use seed, seed+1, …; if seed is -1 each stays random. "
            "Clamped by queue.max_batch."
        ),
    )
    wait_seconds: float = Field(
        default=0,
        ge=0,
        le=1800,
        description="If >0, block until done or timeout (still returns job)",
    )


class BreakpointBody(BaseModel):
    after_id: str | None = Field(
        default=None,
        description="Insert breakpoint after this queue entry id",
    )
    position: int | None = Field(
        default=None,
        description="Insert at index (0 = head). Ignored if after_id set.",
    )
    label: str = Field(default="Break", description="Optional label")


class ReorderBody(BaseModel):
    id: str | None = Field(default=None, description="Entry id to move")
    action: str | None = Field(
        default=None,
        description="up | down | top | bottom",
    )
    ordered_ids: list[str] | None = Field(
        default=None,
        description="Full reordering of all queue entry ids",
    )


class SmoothBody(BaseModel):
    factor: int = Field(default=2, description="Interpolation factor: 2 or 4")
    wait_seconds: float = Field(default=0, ge=0, le=1800)


class ContinueBody(BaseModel):
    prompt: str | None = Field(
        default=None, description="Motion prompt (default: inherit from parent)"
    )
    negative_prompt: str | None = None
    video_profile: str | None = Field(
        default=None,
        description="Profile for the new chunk (default: parent's profile or draft)",
    )
    length: int | None = Field(default=None, description="Chunk frames (WAN 4n+1)")
    continue_seconds: float | None = Field(
        default=None, description="Approx extra seconds if length omitted"
    )
    continue_overlap_seconds: float | None = Field(
        default=None,
        description="Seam blend window seconds (default 0.75; 0 = hard cut)",
    )
    continue_overlap_frames: int | None = Field(
        default=None, description="Seam blend window in frames (overrides seconds)"
    )
    continue_overlap_max_fraction: float | None = Field(
        default=None,
        ge=0.05,
        le=0.49,
        description="Max fraction of either clip used for blend (default 0.4)",
    )
    continue_context_frames: int | None = Field(
        default=None,
        description="Motion prefix: last N parent frames (default 16; 1=still only)",
    )
    continue_method: str | None = Field(
        default=None, description="flf_bridge (default) | prefix"
    )
    continue_bridge_frames: int | None = Field(
        default=None, description="FLF bridge frames (default 25)"
    )
    continue_warmup_frames: int | None = None
    seed: int = -1
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_pack: str | None = None
    auto_pull: bool | None = None
    effects: list[str] | None = Field(
        default=None,
        description="Optional video effect ids (img2vid-compatible)",
    )
    effect_weights: dict[str, float] | None = None
    context_windows: bool | None = None
    context_length: int | None = None
    context_overlap: int | None = None
    wait_seconds: float = Field(default=0, ge=0, le=1800)


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    cfg = cfg or load_config()
    registry = PresetRegistry(cfg)
    effect_registry = EffectRegistry(cfg)
    video_packs = VideoPackRegistry(cfg)
    puller = ModelPuller(cfg)
    client = ComfyClient(cfg)
    supervisor = ComfySupervisor(cfg, client)
    jobs = JobManager(
        cfg,
        registry=registry,
        effects=effect_registry,
        puller=puller,
        client=client,
        supervisor=supervisor,
        video_packs=video_packs,
    )
    gallery = GalleryStore(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await client.start()
        await jobs.start()
        app.state.cfg = cfg
        app.state.registry = registry
        app.state.effects = effect_registry
        app.state.puller = puller
        app.state.client = client
        app.state.supervisor = supervisor
        app.state.jobs = jobs
        app.state.gallery = gallery
        yield
        await jobs.stop()
        await supervisor.stop()
        await client.stop()

    app = FastAPI(
        title="simpleComfy",
        version=__version__,
        description=(
            "Preset-first local image generation. Optional LoRA/prompt effects. "
            "No content guardrails."
        ),
        lifespan=lifespan,
    )

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = cfg.server.api_key
        if not expected:
            return
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    auth = Depends(require_key)

    @app.get("/v1/health")
    async def health(_: None = auth) -> dict[str, Any]:
        reachable = await client.reachable()
        return {
            "ok": True,
            "version": __version__,
            "comfy_reachable": reachable,
            "comfy_url": cfg.comfy.base_url,
            "auto_pull": cfg.models.auto_pull,
            "presets": len(registry.list()),
            "effects": len(effect_registry.list()),
            "modalities": {
                "all": list(MODALITIES),
                "implemented": sorted(IMPLEMENTED_MODALITIES),
            },
            "data_dir": str(cfg.data_dir),
            # Stamp so UI/ops can confirm the running process has current continue code
            "features": {
                "continue_method_default": "flf_bridge",
                "continue_motion_prefix": True,
                "continue_cfr_xfade": True,
                "continue_context_frames_default": 16,
                "context_windows": True,
            },
        }

    @app.get("/v1/capabilities")
    async def capabilities(_: None = auth) -> dict[str, Any]:
        packs = []
        for p in video_packs.list():
            profs = []
            for pid, pr in p.profiles.items():
                miss = puller.missing_video_artifacts(p.artifacts_for("img2vid", pid))
                profs.append(
                    {
                        "id": pid,
                        "label": pr.label,
                        "description": pr.description,
                        "steps": pr.steps,
                        "cfg": pr.cfg,
                        "use_light_lora": pr.use_light_lora,
                        "use_final_lora": pr.use_final_lora,
                        "ready": len(miss) == 0,
                    }
                )
            missing_t2v = puller.missing_video_artifacts(
                p.artifacts_for("txt2vid", p.default_profile)
            )
            missing_i2v = puller.missing_video_artifacts(
                p.artifacts_for("img2vid", p.default_profile)
            )
            packs.append(
                {
                    "id": p.id,
                    "label": p.label,
                    "description": p.description,
                    "size_gb_approx": p.size_gb_approx,
                    "default_profile": p.default_profile,
                    "profiles": profs,
                    "txt2vid_ready": len(missing_t2v) == 0,
                    "img2vid_ready": len(missing_i2v) == 0,
                    "defaults": {
                        "width": p.defaults.width,
                        "height": p.defaults.height,
                        "length": p.defaults.length,
                        "fps": p.defaults.fps,
                    },
                }
            )
        return {
            "modalities": [
                {
                    "id": "txt2img",
                    "label": "Text → Image",
                    "implemented": True,
                    "needs_preset": True,
                    "needs_source_image": False,
                },
                {
                    "id": "txt2vid",
                    "label": "Text → Video",
                    "implemented": True,
                    "needs_preset": False,
                    "needs_source_image": False,
                    "engine": "sample",
                    "note": "the video pack — draft with draft, finalize with final; long clips use context windows",
                    "profiles": ["draft", "final", "fast", "quality"],
                    "context_windows": {
                        "auto_when_length_gt": 81,
                        "default_context_length": 81,
                        "default_context_overlap": 30,
                    },
                },
                {
                    "id": "img2vid",
                    "label": "Image → Video",
                    "implemented": True,
                    "needs_preset": False,
                    "needs_source_image": True,
                    "engine": "sample",
                    "note": "the video pack I2V — default profile draft (draft); long clips use context windows",
                    "profiles": ["draft", "final", "fast", "quality"],
                    "context_windows": {
                        "auto_when_length_gt": 81,
                        "default_context_length": 81,
                        "default_context_overlap": 30,
                        "retain_first_frame": True,
                    },
                },
                {
                    "id": "vid2vid",
                    "label": "Video → Video",
                    "implemented": True,
                    "needs_preset": False,
                    "needs_source_image": False,
                    "needs_source_video": True,
                    "ops": [
                        {
                            "id": "smooth",
                            "label": "Smooth",
                            "backend": "smooth",
                            "note": "Frame-interpolate 2×/4× (ffmpeg)",
                        },
                        {
                            "id": "continue",
                            "label": "Continue",
                            "backend": "continue",
                            "note": "Last frame → new chunk → concat",
                        },
                        {
                            "id": "finalize",
                            "label": "Final",
                            "backend": "generate",
                            "note": "Re-run recipe at final (full steps)",
                        },
                    ],
                    "note": (
                        "UI mode over smooth / continue / finalize. "
                        "API still accepts modality=smooth|continue or dedicated job routes."
                    ),
                },
                {
                    "id": "smooth",
                    "label": "Smooth (API)",
                    "implemented": True,
                    "needs_preset": False,
                    "needs_source_image": False,
                    "api_only": True,
                    "note": "POST /v1/generate modality=smooth or POST /v1/jobs/{id}/smooth",
                },
                {
                    "id": "continue",
                    "label": "Continue (API)",
                    "implemented": True,
                    "needs_preset": False,
                    "needs_source_image": False,
                    "api_only": True,
                    "note": "POST /v1/generate modality=continue or POST /v1/jobs/{id}/continue",
                },
            ],
            "video_packs": packs,
            "video_profiles": ["draft", "final", "fast", "quality"],
            "effects": True,
            "gallery": True,
            "post": {
                "smooth": "POST /v1/jobs/{id}/smooth {factor: 2|4}",
                "continue": "POST /v1/jobs/{id}/continue {video_profile?, continue_seconds?}",
            },
        }

    @app.get("/v1/help")
    async def help_root(_: None = auth) -> dict[str, Any]:
        return {
            "name": "simpleComfy",
            "version": __version__,
            "summary": (
                "Local uncensored generation via managed ComfyUI. "
                "Presets + effects for images; gallery history for reuse into "
                "txt2img / txt2vid / img2vid."
            ),
            "flow": [
                "GET /v1/presets — pick a look",
                "GET /v1/effects — optional LoRAs / filters (filter by base_family)",
                "GET /v1/gallery — output history with metadata",
                "POST /v1/generate — {modality, preset?, prompt, effects?, batch_size?}",
                "GET /v1/queue — live ordered queue (phase/progress)",
                "GET /v1/jobs/{id} — poll until completed; use outputs[].url",
            ],
            "modalities": {
                "txt2img": "implemented",
                "txt2vid": "the video pack (models auto-pulled on generate)",
                "img2vid": "the video pack I2V (gallery source + pack)",
            },
            "generate_effects": {
                "omit": "use preset.default_effects",
                "empty_list": "no optional effects",
                "list": "exact effect ids to apply",
                "effect_weights": "optional {effect_id: weight} overrides",
            },
            "notes": [
                "No content filtering.",
                "Missing models are pulled automatically as the first job phase.",
                "Effects declare base_families; incompatible combos are rejected.",
                "Completed jobs write data/outputs/<id>/job.json for the gallery.",
                "Queue persists across restarts; breakpoints pause the worker.",
            ],
            "endpoints": {
                "health": "GET /v1/health",
                "capabilities": "GET /v1/capabilities",
                "presets": "GET /v1/presets",
                "preset": "GET /v1/presets/{id}",
                "pull_preset": "POST /v1/presets/{id}/pull",
                "effects": "GET /v1/effects",
                "effect": "GET /v1/effects/{id}",
                "pull_effect": "POST /v1/effects/{id}/pull",
                "gallery": "GET /v1/gallery",
                "gallery_item": "GET /v1/gallery/{id}",
                "video_packs": "GET /v1/video/packs",
                "pull_video": "POST /v1/video/packs/{id}/pull?modality=txt2vid|img2vid|all",
                "generate": "POST /v1/generate",
                "queue": "GET /v1/queue",
                "queue_breakpoint": "POST /v1/queue/breakpoint",
                "queue_reorder": "POST /v1/queue/reorder",
                "queue_remove": "DELETE /v1/queue/entries/{id}",
                "job": "GET /v1/jobs/{id}",
                "jobs": "GET /v1/jobs",
                "cancel_job": "POST /v1/jobs/{id}/cancel",
                "smooth": "POST /v1/jobs/{id}/smooth",
                "continue": "POST /v1/jobs/{id}/continue",
                "unload_models": "POST /v1/admin/unload-models",
                "outputs": "GET /v1/outputs/{job_id}/{filename}",
            },
        }

    @app.get("/v1/video/packs")
    async def list_video_packs(_: None = auth) -> dict[str, Any]:
        video_packs.reload()
        items = []
        for p in video_packs.list():
            profiles = {}
            for pid in p.profiles:
                profiles[pid] = {
                    "txt2vid_missing": puller.missing_video_artifacts(
                        p.artifacts_for("txt2vid", pid)
                    ),
                    "img2vid_missing": puller.missing_video_artifacts(
                        p.artifacts_for("img2vid", pid)
                    ),
                    "ready": len(puller.missing_video_artifacts(p.artifacts_for("img2vid", pid)))
                    == 0
                    and len(puller.missing_video_artifacts(p.artifacts_for("txt2vid", pid)))
                    == 0,
                    "label": p.profiles[pid].label,
                    "description": p.profiles[pid].description,
                    "steps": p.profiles[pid].steps,
                }
            items.append(
                {
                    "id": p.id,
                    "label": p.label,
                    "description": p.description,
                    "size_gb_approx": p.size_gb_approx,
                    "default_profile": p.default_profile,
                    "profiles": profiles,
                    "defaults": {
                        "width": p.defaults.width,
                        "height": p.defaults.height,
                        "length": p.defaults.length,
                        "fps": p.defaults.fps,
                    },
                }
            )
        return {"packs": items}

    @app.post("/v1/video/packs/{pack_id}/pull")
    async def pull_video_pack(
        pack_id: str,
        modality: str = Query(
            default="all",
            description="txt2vid | img2vid | all (shared+both)",
        ),
        profile: str = Query(
            default="all",
            description="draft | final | fast | quality | all",
        ),
        _: None = auth,
    ) -> dict[str, Any]:
        video_packs.reload()
        pack = video_packs.get(pack_id)
        if not pack:
            raise HTTPException(status_code=404, detail=f"unknown video pack '{pack_id}'")
        mod = (modality or "all").strip().lower()
        prof = (profile or "all").strip().lower()
        mods = ["txt2vid", "img2vid"] if mod == "all" else [mod]
        if any(m not in ("txt2vid", "img2vid") for m in mods):
            raise HTTPException(status_code=400, detail="modality must be txt2vid|img2vid|all")
        profs = list(pack.profiles.keys()) if prof == "all" else [prof]
        if any(p not in pack.profiles for p in profs):
            raise HTTPException(
                status_code=400,
                detail=f"profile must be one of {list(pack.profiles)} or all",
            )
        arts = []
        seen: set[str] = set()
        for m in mods:
            for pr in profs:
                for a in pack.artifacts_for(m, pr):
                    if a.filename in seen:
                        continue
                    seen.add(a.filename)
                    arts.append(a)
        try:
            resolved = await puller.ensure_video_artifacts(arts, auto_pull=True)
        except PullError as e:
            raise HTTPException(status_code=400, detail={"error": str(e), **e.detail}) from e
        return {
            "pack": pack.id,
            "modality": mod,
            "profile": prof,
            "files": resolved,
            "ready": True,
        }

    @app.get("/v1/presets")
    async def list_presets(_: None = auth) -> dict[str, Any]:
        registry.reload()
        items = [summarize_preset_status(p, puller) for p in registry.list()]
        return {"presets": items}

    @app.get("/v1/presets/{preset_id}")
    async def get_preset(preset_id: str, _: None = auth) -> dict[str, Any]:
        registry.reload()
        try:
            p = registry.require(preset_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        detail = summarize_preset_status(p, puller)
        detail["prompt"] = {
            "prefix": p.prompt.prefix,
            "suffix": p.prompt.suffix,
            "negative": p.prompt.negative,
        }
        detail["loras"] = [
            {"source": l.source, "weight": l.weight, "filename": l.filename}
            for l in p.loras
        ]
        detail["checkpoint_filename"] = p.checkpoint_filename
        return detail

    @app.post("/v1/presets/{preset_id}/pull")
    async def pull_preset(preset_id: str, _: None = auth) -> dict[str, Any]:
        registry.reload()
        try:
            p = registry.require(preset_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        try:
            result = await puller.pull_preset(p)
        except PullError as e:
            raise HTTPException(status_code=400, detail={"error": str(e), **e.detail}) from e
        return result

    @app.get("/v1/effects")
    async def list_effects(
        family: str | None = Query(
            default=None, description="Filter to effects compatible with this base_family"
        ),
        modality: str | None = Query(
            default=None,
            description="Filter by modality: txt2img | txt2vid | img2vid (continue uses img2vid)",
        ),
        _: None = auth,
    ) -> dict[str, Any]:
        effect_registry.reload()
        items = []
        mod = (modality or "").strip().lower() or None
        if mod == "continue":
            mod = "img2vid"
        for e in effect_registry.list():
            if family and not e.compatible_with(family):
                continue
            if mod and not e.applies_to_modality(mod):
                continue
            items.append(summarize_effect(e, puller))
        return {"effects": items}

    @app.get("/v1/effects/{effect_id}")
    async def get_effect(effect_id: str, _: None = auth) -> dict[str, Any]:
        effect_registry.reload()
        try:
            e = effect_registry.require(effect_id)
        except KeyError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err
        return summarize_effect(e, puller)

    @app.post("/v1/effects/{effect_id}/pull")
    async def pull_effect(effect_id: str, _: None = auth) -> dict[str, Any]:
        effect_registry.reload()
        try:
            e = effect_registry.require(effect_id)
        except KeyError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err
        if not e.needs_lora:
            return {"effect": e.id, "ready": True, "lora": None, "note": "prompt-only effect"}
        try:
            result = await puller.pull_effect_lora(e)
        except PullError as err:
            raise HTTPException(
                status_code=400, detail={"error": str(err), **err.detail}
            ) from err
        return result

    @app.get("/v1/gallery")
    async def list_gallery(
        limit: int = Query(default=48, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        modality: str | None = Query(default=None),
        has_image: bool | None = Query(
            default=None, description="If true, only items with at least one image"
        ),
        has_video: bool | None = Query(
            default=None, description="If true, only items with at least one video"
        ),
        has_media: bool | None = Query(
            default=True,
            description="If true (default), only items with image or video outputs",
        ),
        _: None = auth,
    ) -> dict[str, Any]:
        return gallery.list_items(
            limit=limit,
            offset=offset,
            modality=modality,
            has_image=has_image,
            has_video=has_video,
            has_media=has_media,
        )

    @app.get("/v1/gallery/{job_id}")
    async def get_gallery_item(job_id: str, _: None = auth) -> dict[str, Any]:
        item = gallery.get(job_id)
        if not item:
            # Fall back to in-memory job if still hot
            mem = jobs.get(job_id)
            if mem:
                return mem.to_dict()
            raise HTTPException(status_code=404, detail="gallery item not found")
        return item

    @app.post("/v1/generate")
    async def generate(body: GenerateBody, _: None = auth) -> dict[str, Any]:
        registry.reload()
        effect_registry.reload()
        modality = (body.modality or "txt2img").strip().lower()
        if modality not in MODALITIES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown modality; use one of {list(MODALITIES)}",
            )

        preset_id = (body.preset or "").strip()
        if modality == "txt2img":
            if not preset_id:
                raise HTTPException(status_code=400, detail="preset is required for txt2img")
            if not (body.prompt or "").strip():
                raise HTTPException(status_code=400, detail="prompt is required")
            try:
                registry.require(preset_id)
            except KeyError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
        elif modality == "txt2vid":
            if not (body.prompt or "").strip():
                raise HTTPException(status_code=400, detail="prompt is required for txt2vid")
        elif modality == "img2vid":
            if not body.source_job_id and not body.source_input:
                raise HTTPException(
                    status_code=400,
                    detail="img2vid needs source_job_id (gallery) or source_input (upload)",
                )
        elif modality == "smooth":
            if not body.parent_job_id:
                raise HTTPException(
                    status_code=400, detail="smooth needs parent_job_id"
                )
            fac = body.smooth_factor if body.smooth_factor is not None else 2
            if fac not in (2, 4):
                raise HTTPException(status_code=400, detail="smooth_factor must be 2 or 4")
        elif modality == "continue":
            if not body.parent_job_id:
                raise HTTPException(
                    status_code=400, detail="continue needs parent_job_id"
                )

        if body.effects is not None:
            for eid in body.effects:
                if not effect_registry.get(eid):
                    raise HTTPException(status_code=400, detail=f"unknown effect '{eid}'")

        req = GenerateRequest(
            preset=preset_id,
            prompt=body.prompt or "",
            negative_prompt=body.negative_prompt,
            seed=body.seed,
            width=body.width,
            height=body.height,
            auto_pull=body.auto_pull,
            effects=body.effects,
            effect_weights=body.effect_weights,
            modality=modality,
            source_job_id=body.source_job_id,
            source_filename=body.source_filename,
            source_input=body.source_input,
            length=body.length,
            fps=body.fps,
            video_pack=body.video_pack,
            video_profile=body.video_profile,
            parent_job_id=body.parent_job_id,
            smooth_factor=body.smooth_factor,
            continue_seconds=body.continue_seconds,
            continue_overlap_seconds=body.continue_overlap_seconds,
            continue_overlap_frames=body.continue_overlap_frames,
            continue_overlap_max_fraction=body.continue_overlap_max_fraction,
            continue_context_frames=body.continue_context_frames,
            continue_method=body.continue_method,
            continue_bridge_frames=body.continue_bridge_frames,
            continue_warmup_frames=body.continue_warmup_frames,
            context_windows=body.context_windows,
            context_length=body.context_length,
            context_overlap=body.context_overlap,
        )
        batch_size = int(body.batch_size or 1)
        try:
            created = await jobs.enqueue_batch(req, batch_size=batch_size)
        except RuntimeError as e:
            # queue full vs validation
            msg = str(e)
            code = 429 if "queue full" in msg else 400
            raise HTTPException(status_code=code, detail=msg) from e

        if batch_size <= 1 and body.wait_seconds and body.wait_seconds > 0:
            waited = await _maybe_wait(jobs, created[0], body.wait_seconds)
            return {
                **waited,
                "jobs": [waited],
                "batch_size": 1,
            }

        dicts = [j.to_dict() for j in created]
        # Backward compatible: top-level fields mirror first job
        out = dict(dicts[0])
        out["jobs"] = dicts
        out["batch_size"] = len(dicts)
        return out

    async def _maybe_wait(job_mgr: JobManager, job: Any, wait_seconds: float) -> dict[str, Any]:
        if not wait_seconds or wait_seconds <= 0:
            return job.to_dict()
        import asyncio

        deadline = asyncio.get_running_loop().time() + wait_seconds
        while asyncio.get_running_loop().time() < deadline:
            j = job_mgr.get(job.id)
            if j and j.status.value in ("completed", "failed", "cancelled"):
                return j.to_dict()
            await asyncio.sleep(0.4)
        j = job_mgr.get(job.id)
        return j.to_dict() if j else job.to_dict()

    @app.get("/v1/queue")
    async def get_queue(_: None = auth) -> dict[str, Any]:
        return jobs.queue_snapshot()

    @app.post("/v1/queue/breakpoint")
    async def queue_breakpoint(body: BreakpointBody | None = None, _: None = auth) -> dict[str, Any]:
        body = body or BreakpointBody()
        try:
            ent = await jobs.add_breakpoint(
                after_id=body.after_id,
                position=body.position,
                label=body.label,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        snap = jobs.queue_snapshot()
        snap["created"] = ent
        return snap

    @app.post("/v1/queue/reorder")
    async def queue_reorder(body: ReorderBody, _: None = auth) -> dict[str, Any]:
        try:
            if body.ordered_ids is not None:
                return await jobs.reorder("", ordered_ids=body.ordered_ids)
            if not body.id or not body.action:
                raise HTTPException(
                    status_code=400,
                    detail="Provide id+action or ordered_ids",
                )
            return await jobs.reorder(body.id, action=body.action)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.delete("/v1/queue/entries/{entry_id}")
    async def queue_remove(entry_id: str, _: None = auth) -> dict[str, Any]:
        try:
            return await jobs.remove_entry(entry_id)
        except RuntimeError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/v1/jobs")
    async def list_jobs(
        limit: int = Query(default=50, ge=1, le=200), _: None = auth
    ) -> dict[str, Any]:
        return {"jobs": [j.to_dict() for j in jobs.list_jobs(limit=limit)]}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, _: None = auth) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job:
            return job.to_dict()
        # Persist-backed history after restart
        item = gallery.get(job_id)
        if item:
            return item
        raise HTTPException(status_code=404, detail="job not found")

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, _: None = auth) -> dict[str, Any]:
        try:
            job = await jobs.cancel_job(job_id)
        except RuntimeError as e:
            msg = str(e)
            code = 404 if "not found" in msg else 409
            raise HTTPException(status_code=code, detail=msg) from e
        return job.to_dict()

    @app.post("/v1/jobs/{job_id}/smooth")
    async def smooth_job(
        job_id: str, body: SmoothBody | None = None, _: None = auth
    ) -> dict[str, Any]:
        """Frame-interpolate a completed video job (2× or 4×)."""
        body = body or SmoothBody()
        factor = int(body.factor or 2)
        if factor not in (2, 4):
            raise HTTPException(status_code=400, detail="factor must be 2 or 4")
        parent = gallery.get(job_id) or (
            jobs.get(job_id).to_dict() if jobs.get(job_id) else None
        )
        if not parent:
            raise HTTPException(status_code=404, detail="parent job not found")
        if not parent.get("has_video") and not any(
            (o.get("kind") == "video") for o in (parent.get("outputs") or [])
        ):
            raise HTTPException(status_code=400, detail="parent job has no video")
        try:
            job = await jobs.enqueue_smooth(job_id, factor=factor)
        except RuntimeError as e:
            msg = str(e)
            code = 429 if "queue full" in msg else 400
            raise HTTPException(status_code=code, detail=msg) from e
        return await _maybe_wait(jobs, job, body.wait_seconds)

    @app.post("/v1/jobs/{job_id}/continue")
    async def continue_job(
        job_id: str, body: ContinueBody | None = None, _: None = auth
    ) -> dict[str, Any]:
        """Extend a completed video from its last frame and concat."""
        body = body or ContinueBody()
        parent = gallery.get(job_id) or (
            jobs.get(job_id).to_dict() if jobs.get(job_id) else None
        )
        if not parent:
            raise HTTPException(status_code=404, detail="parent job not found")
        if not parent.get("has_video") and not any(
            (o.get("kind") == "video") for o in (parent.get("outputs") or [])
        ):
            raise HTTPException(status_code=400, detail="parent job has no video")
        try:
            job = await jobs.enqueue_continue(
                job_id,
                prompt=body.prompt,
                negative_prompt=body.negative_prompt,
                video_profile=body.video_profile,
                length=body.length,
                continue_seconds=body.continue_seconds,
                continue_overlap_seconds=body.continue_overlap_seconds,
                continue_overlap_frames=body.continue_overlap_frames,
                continue_overlap_max_fraction=body.continue_overlap_max_fraction,
                continue_context_frames=body.continue_context_frames,
                continue_method=body.continue_method,
                continue_bridge_frames=body.continue_bridge_frames,
                continue_warmup_frames=body.continue_warmup_frames,
                seed=body.seed,
                width=body.width,
                height=body.height,
                fps=body.fps,
                video_pack=body.video_pack,
                auto_pull=body.auto_pull,
                effects=body.effects,
                effect_weights=body.effect_weights,
                context_windows=body.context_windows,
                context_length=body.context_length,
                context_overlap=body.context_overlap,
            )
        except RuntimeError as e:
            msg = str(e)
            code = 429 if "queue full" in msg else 400
            raise HTTPException(status_code=code, detail=msg) from e
        return await _maybe_wait(jobs, job, body.wait_seconds)

    @app.post("/v1/inputs")
    async def upload_input(
        file: UploadFile = File(...),
        _: None = auth,
    ) -> dict[str, Any]:
        """Upload an image for img2vid (also staged into ComfyUI input/)."""
        import re
        import uuid as _uuid
        from pathlib import Path as _Path

        from simplecomfy.imagesize import fit_video_size, probe_image_size

        raw_name = file.filename or "upload.png"
        base = _Path(raw_name).name
        ext = _Path(base).suffix.lower() or ".png"
        if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise HTTPException(
                status_code=400,
                detail="unsupported image type (use png/jpg/webp/gif)",
            )
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", _Path(base).stem)[:48] or "upload"
        name = f"up_{_uuid.uuid4().hex[:10]}_{safe}{ext}"

        upload_dir = cfg.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / name
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        if len(data) > 80 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="file too large (max 80MB)")
        dest.write_bytes(data)

        # Stage into Comfy input for LoadImage
        input_dir = cfg.comfy_input_dir
        comfy_name = name
        if input_dir:
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / name).write_bytes(data)

        dims = probe_image_size(dest)
        video_fit = fit_video_size(*dims, max_side=832) if dims else None
        return {
            "filename": name,
            "comfy_name": comfy_name,
            "url": f"/v1/inputs/{name}",
            "bytes": len(data),
            "width": dims[0] if dims else None,
            "height": dims[1] if dims else None,
            "video_width": video_fit[0] if video_fit else None,
            "video_height": video_fit[1] if video_fit else None,
        }

    @app.get("/v1/inputs/{filename}")
    async def get_input(filename: str, _: None = auth) -> FileResponse:
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="invalid filename")
        path = (cfg.data_dir / "uploads" / filename).resolve()
        root = (cfg.data_dir / "uploads").resolve()
        if not str(path).startswith(str(root)) or not path.is_file():
            raise HTTPException(status_code=404, detail="input not found")
        return FileResponse(path)

    @app.get("/v1/outputs/{job_id}/{filename}")
    async def get_output(job_id: str, filename: str, _: None = auth) -> FileResponse:
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="invalid filename")
        path = (cfg.outputs_dir / job_id / filename).resolve()
        out_root = cfg.outputs_dir.resolve()
        if not str(path).startswith(str(out_root)) or not path.is_file():
            raise HTTPException(status_code=404, detail="output not found")
        ext = path.suffix.lower()
        media_types = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mt = media_types.get(ext)
        # inline so <video>/<img> play in-page (attachment forces download)
        return FileResponse(
            path,
            media_type=mt,
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f'inline; filename="{filename}"',
            },
        )

    @app.post("/v1/admin/write-comfy-paths")
    async def admin_write_paths(
        comfy_root: str | None = None, _: None = auth
    ) -> dict[str, Any]:
        root = Path(comfy_root or cfg.comfy.root or "").expanduser()
        if not root or not root.is_dir():
            raise HTTPException(
                status_code=400,
                detail="Provide comfy_root query/body or set comfy.root in config",
            )
        try:
            dest = write_extra_model_paths(root, cfg.models_dir)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"wrote": str(dest), "models_dir": str(cfg.models_dir)}

    @app.post("/v1/admin/unload-models")
    async def admin_unload_models(_: None = auth) -> dict[str, Any]:
        """Unload ComfyUI models from VRAM when no job is actively running."""
        if jobs.is_busy():
            raise HTTPException(
                status_code=409,
                detail="cannot unload while a job is pulling or running",
            )
        try:
            result = await client.free_memory(unload_models=True, free_memory=True)
        except ComfyError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        return {"ok": True, "comfy": result, "queue": jobs.queue_snapshot()}

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():

        @app.get("/")
        async def ui_index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        app.mount(
            "/assets",
            StaticFiles(directory=str(static_dir)),
            name="assets",
        )

        @app.get("/styles.css")
        async def ui_css() -> FileResponse:
            return FileResponse(static_dir / "styles.css", media_type="text/css")

        @app.get("/app.js")
        async def ui_js() -> FileResponse:
            return FileResponse(
                static_dir / "app.js", media_type="application/javascript"
            )

    return app


def run_server(cfg: AppConfig | None = None) -> None:
    import uvicorn

    cfg = cfg or load_config()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")
