"""ComfyUI API-format workflow builders (fixed templates)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from simplecomfy.presets import Preset
from simplecomfy.video_pack import VideoDefaults

SAMPLER_MAP = {
    "dpmpp_2m": "dpmpp_2m",
    "dpmpp_2m_sde": "dpmpp_2m_sde",
    "dpmpp_sde": "dpmpp_sde",
    "euler": "euler",
    "euler_a": "euler_ancestral",
    "euler_ancestral": "euler_ancestral",
    "ddim": "ddim",
    "uni_pc": "uni_pc",
}

SCHEDULER_MAP = {
    "normal": "normal",
    "karras": "karras",
    "exponential": "exponential",
    "sgm_uniform": "sgm_uniform",
    "simple": "simple",
    "ddim_uniform": "ddim_uniform",
}


def resolve_seed(seed: int | None) -> int:
    if seed is None or seed < 0:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def build_txt2img(
    *,
    preset: Preset,
    prompt: str,
    negative_prompt: str,
    checkpoint_name: str,
    lora_files: list[tuple[str, float]],
    width: int,
    height: int,
    seed: int | None,
    filename_prefix: str = "simplecomfy",
    positive: str | None = None,
    negative: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build a standard CheckpointLoader + optional LoRAs + KSampler graph.
    Returns (workflow_for_comfy, meta).

    If positive/negative are provided, they are used as-is (already merged with effects).
    Otherwise scaffold from the preset alone.
    """
    sampler_name = SAMPLER_MAP.get(preset.sampler.name, preset.sampler.name)
    scheduler = SCHEDULER_MAP.get(preset.sampler.scheduler, preset.sampler.scheduler)
    steps = int(preset.sampler.steps)
    cfg = float(preset.sampler.cfg)
    seed_v = resolve_seed(seed)
    if positive is None:
        positive = preset.build_positive(prompt)
    if negative is None:
        negative = preset.build_negative(negative_prompt)

    workflow: dict[str, Any] = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint_name},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["4", 1]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed_v,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }

    model_ref: list[Any] = ["4", 0]
    clip_ref: list[Any] = ["4", 1]
    nid = 20
    for lora_name, weight in lora_files:
        sid = str(nid)
        workflow[sid] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "strength_model": float(weight),
                "strength_clip": float(weight),
                "model": model_ref,
                "clip": clip_ref,
            },
        }
        model_ref = [sid, 0]
        clip_ref = [sid, 1]
        nid += 1

    workflow["3"]["inputs"]["model"] = model_ref
    workflow["6"]["inputs"]["clip"] = clip_ref
    workflow["7"]["inputs"]["clip"] = clip_ref

    meta = {
        "seed": seed_v,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "width": width,
        "height": height,
        "checkpoint": checkpoint_name,
        "loras": [{"name": n, "weight": w} for n, w in lora_files],
        "positive": positive,
        "negative": negative,
        "preset": preset.id,
    }
    return workflow, meta


def _snap16(n: int) -> int:
    n = max(16, int(n))
    return n - (n % 16)


def _snap_length(n: int) -> int:
    """WAN length prefers 4k+1 (e.g. 81)."""
    n = max(1, int(n))
    # round down to 4k+1
    return max(1, ((n - 1) // 4) * 4 + 1)


# Context-window long gen (WanContextWindowsManual)
DEFAULT_CONTEXT_LENGTH = 81
DEFAULT_CONTEXT_OVERLAP = 30


@dataclass
class ContextWindowsConfig:
    """Sliding temporal windows for long WAN clips."""

    enabled: bool = False
    context_length: int = DEFAULT_CONTEXT_LENGTH
    context_overlap: int = DEFAULT_CONTEXT_OVERLAP
    context_schedule: str = "standard_uniform"
    fuse_method: str = "pyramid"
    freenoise: bool = True
    retain_first_frame: bool = False
    mode: str = "off"  # off | auto | on

    def as_meta(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "context_length": self.context_length,
            "context_overlap": self.context_overlap,
            "context_schedule": self.context_schedule,
            "fuse_method": self.fuse_method,
            "freenoise": self.freenoise,
            "retain_first_frame": self.retain_first_frame,
        }


def resolve_context_windows(
    *,
    total_frames: int,
    enabled: bool | None = None,
    context_length: int | None = None,
    context_overlap: int | None = None,
    retain_first_frame: bool = False,
) -> ContextWindowsConfig:
    """
    Decide whether to patch models with WanContextWindowsManual.

    enabled:
      None  → auto (on when total_frames > context_length)
      True  → force on
      False → force off
    context_overlap = how many frames consecutive windows share (continuity knob).
    """
    clen = int(context_length) if context_length is not None else DEFAULT_CONTEXT_LENGTH
    clen = _snap_length(max(5, clen))
    # Overlap must be < window; prefer node default 30, else ~1/3 of smaller windows
    if context_overlap is not None:
        cover = int(context_overlap)
    else:
        cover = DEFAULT_CONTEXT_OVERLAP if clen > DEFAULT_CONTEXT_OVERLAP else max(4, clen // 3)
    cover = max(0, min(cover, clen - 1))

    frames = _snap_length(total_frames)
    if enabled is True:
        mode = "on"
        use = True
    elif enabled is False:
        mode = "off"
        use = False
    else:
        mode = "auto"
        # Single window already covers the clip — no patch needed
        use = frames > clen

    return ContextWindowsConfig(
        enabled=use,
        context_length=clen,
        context_overlap=cover,
        retain_first_frame=bool(retain_first_frame),
        mode=mode,
    )


def _chain_model_loras(
    workflow: dict[str, Any],
    *,
    model_ref: list[Any],
    loras: list[tuple[str, float]],
    start_id: int,
) -> tuple[list[Any], int]:
    """Stack LoraLoaderModelOnly nodes; return (model_ref, next_id)."""
    ref = model_ref
    nid = start_id
    for name, weight in loras:
        if not name:
            continue
        sid = str(nid)
        workflow[sid] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ref,
                "lora_name": name,
                "strength_model": float(weight),
            },
        }
        ref = [sid, 0]
        nid += 1
    return ref, nid


def _patch_context_window(
    workflow: dict[str, Any],
    *,
    model_ref: list[Any],
    cfg: ContextWindowsConfig,
    node_id: str,
) -> list[Any]:
    """Wrap one MODEL with WanContextWindowsManual; return new model ref."""
    if not cfg.enabled:
        return model_ref
    workflow[node_id] = {
        "class_type": "WanContextWindowsManual",
        "inputs": {
            "model": model_ref,
            "context_length": int(cfg.context_length),
            "context_overlap": int(cfg.context_overlap),
            "context_schedule": cfg.context_schedule,
            "context_stride": 1,
            "closed_loop": False,
            "fuse_method": cfg.fuse_method,
            "freenoise": bool(cfg.freenoise),
            "retain_first_frame": bool(cfg.retain_first_frame),
            "split_conds_to_windows": False,
        },
    }
    return [node_id, 0]


def build_wan_txt2vid(
    *,
    prompt: str,
    negative_prompt: str,
    high_unet: str,
    low_unet: str,
    high_loras: list[tuple[str, float]] | None = None,
    low_loras: list[tuple[str, float]] | None = None,
    text_encoder: str,
    vae_name: str,
    defaults: VideoDefaults,
    width: int | None = None,
    height: int | None = None,
    length: int | None = None,
    seed: int | None = None,
    fps: float | None = None,
    filename_prefix: str = "simplecomfy/video",
    profile: str = "quality",
    context_windows: bool | None = None,
    context_length: int | None = None,
    context_overlap: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Text-to-video dual-UNET graph; optional high and low LoRA stacks."""
    d = defaults
    w = _snap16(width or d.width)
    h = _snap16(height or d.height)
    frames = _snap_length(length or d.length)
    seed_v = resolve_seed(seed)
    steps = max(1, int(d.steps))
    cfg = float(d.cfg)
    split = int(d.high_end_step)
    split = max(1, min(split, steps - 1)) if steps > 1 else steps
    fps_v = float(fps if fps is not None else d.fps)
    positive = (prompt or "").strip()
    negative = (negative_prompt or d.negative or "").strip()
    hi_loras = list(high_loras or [])
    lo_loras = list(low_loras or [])
    ctx = resolve_context_windows(
        total_frames=frames,
        enabled=context_windows,
        context_length=context_length,
        context_overlap=context_overlap,
        retain_first_frame=False,
    )

    workflow: dict[str, Any] = {
        "71": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder,
                "type": "wan",
                "device": "default",
            },
        },
        "73": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "75": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": high_unet, "weight_dtype": "default"},
        },
        "76": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": low_unet, "weight_dtype": "default"},
        },
    }
    high_ref, nid = _chain_model_loras(
        workflow, model_ref=["75", 0], loras=hi_loras, start_id=300
    )
    low_ref, nid = _chain_model_loras(
        workflow, model_ref=["76", 0], loras=lo_loras, start_id=nid
    )
    workflow["82"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": high_ref, "shift": float(d.shift)},
    }
    workflow["86"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": low_ref, "shift": float(d.shift)},
    }
    high_samp = _patch_context_window(
        workflow, model_ref=["82", 0], cfg=ctx, node_id="400"
    )
    low_samp = _patch_context_window(
        workflow, model_ref=["86", 0], cfg=ctx, node_id="401"
    )
    workflow["89"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": positive, "clip": ["71", 0]},
    }
    workflow["72"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": ["71", 0]},
    }
    workflow["74"] = {
        "class_type": "EmptyHunyuanLatentVideo",
        "inputs": {
            "width": w,
            "height": h,
            "length": frames,
            "batch_size": 1,
        },
    }
    workflow["81"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": high_samp,
            "add_noise": "enable",
            "noise_seed": seed_v,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": d.sampler,
            "scheduler": d.scheduler,
            "positive": ["89", 0],
            "negative": ["72", 0],
            "latent_image": ["74", 0],
            "start_at_step": 0,
            "end_at_step": split,
            "return_with_leftover_noise": "enable",
        },
    }
    workflow["78"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": low_samp,
            "add_noise": "disable",
            "noise_seed": seed_v,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": d.sampler,
            "scheduler": d.scheduler,
            "positive": ["89", 0],
            "negative": ["72", 0],
            "latent_image": ["81", 0],
            "start_at_step": split,
            "end_at_step": steps,
            "return_with_leftover_noise": "disable",
        },
    }
    workflow["87"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["78", 0], "vae": ["73", 0]},
    }
    workflow["114"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["87", 0], "fps": fps_v},
    }
    workflow["200"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["114", 0],
            "filename_prefix": filename_prefix,
            "format": "mp4",
            "codec": "auto",
        },
    }

    meta = {
        "modality": "txt2vid",
        "pack": "sample",
        "profile": profile,
        "seed": seed_v,
        "steps": steps,
        "cfg": cfg,
        "shift": d.shift,
        "sampler_name": d.sampler,
        "scheduler": d.scheduler,
        "width": w,
        "height": h,
        "length": frames,
        "fps": fps_v,
        "positive": positive,
        "negative": negative,
        "high_unet": high_unet,
        "low_unet": low_unet,
        "loras": (
            [{"name": n, "weight": w, "noise": "high"} for n, w in hi_loras]
            + [{"name": n, "weight": w, "noise": "low"} for n, w in lo_loras]
        ),
        "text_encoder": text_encoder,
        "vae": vae_name,
        "context_windows": ctx.as_meta(),
    }
    return workflow, meta


def build_wan_img2vid(
    *,
    prompt: str,
    negative_prompt: str,
    start_image_name: str = "",
    start_video_name: str | None = None,
    high_unet: str,
    low_unet: str,
    high_loras: list[tuple[str, float]] | None = None,
    low_loras: list[tuple[str, float]] | None = None,
    text_encoder: str,
    vae_name: str,
    defaults: VideoDefaults,
    width: int | None = None,
    height: int | None = None,
    length: int | None = None,
    seed: int | None = None,
    fps: float | None = None,
    filename_prefix: str = "simplecomfy/video",
    profile: str = "quality",
    context_windows: bool | None = None,
    context_length: int | None = None,
    context_overlap: int | None = None,
    # When start_video provides a multi-frame prefix, how many frames it holds
    # (recorded in meta; WanImageToVideo masks the whole batch automatically)
    motion_prefix_frames: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    the video pack I2V dual-UNET with optional LoRA stacks + start still or motion prefix.

    start_video_name: short mp4 in Comfy input/; frames become the multi-frame
    start_image batch (WanImageToVideo locks those frames — motion continuity).
    """
    d = defaults
    w = _snap16(width or d.width)
    h = _snap16(height or d.height)
    frames = _snap_length(length or d.length)
    seed_v = resolve_seed(seed)
    steps = max(1, int(d.steps))
    cfg = float(d.cfg)
    split = int(d.high_end_step)
    split = max(1, min(split, steps - 1)) if steps > 1 else steps
    fps_v = float(fps if fps is not None else d.fps)
    positive = (prompt or "").strip()
    negative = (negative_prompt or d.negative or "").strip()
    hi_loras = list(high_loras or [])
    lo_loras = list(low_loras or [])
    ctx = resolve_context_windows(
        total_frames=frames,
        enabled=context_windows,
        context_length=context_length,
        context_overlap=context_overlap,
        # Keep I2V start frame identity across sliding windows
        retain_first_frame=True,
    )

    workflow: dict[str, Any] = {
        "84": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder,
                "type": "wan",
                "device": "default",
            },
        },
        "90": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "95": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": high_unet, "weight_dtype": "default"},
        },
        "96": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": low_unet, "weight_dtype": "default"},
        },
    }
    high_ref, nid = _chain_model_loras(
        workflow, model_ref=["95", 0], loras=hi_loras, start_id=300
    )
    low_ref, nid = _chain_model_loras(
        workflow, model_ref=["96", 0], loras=lo_loras, start_id=nid
    )
    workflow["104"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": high_ref, "shift": float(d.shift)},
    }
    workflow["103"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": low_ref, "shift": float(d.shift)},
    }
    high_samp = _patch_context_window(
        workflow, model_ref=["104", 0], cfg=ctx, node_id="400"
    )
    low_samp = _patch_context_window(
        workflow, model_ref=["103", 0], cfg=ctx, node_id="401"
    )
    workflow["93"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": positive, "clip": ["84", 0]},
    }
    workflow["89"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": ["84", 0]},
    }

    # Start condition: multi-frame video prefix (motion) or single still
    use_video_prefix = bool(start_video_name and str(start_video_name).strip())
    if use_video_prefix:
        workflow["10"] = {
            "class_type": "LoadVideo",
            "inputs": {"file": str(start_video_name).strip()},
        }
        workflow["11"] = {
            "class_type": "GetVideoComponents",
            "inputs": {"video": ["10", 0]},
        }
        start_ref: list[Any] = ["11", 0]  # IMAGE batch
    else:
        if not (start_image_name or "").strip():
            raise ValueError("start_image_name or start_video_name required for img2vid")
        workflow["10"] = {
            "class_type": "LoadImage",
            "inputs": {"image": start_image_name},
        }
        start_ref = ["10", 0]

    workflow["98"] = {
        "class_type": "WanImageToVideo",
        "inputs": {
            "positive": ["93", 0],
            "negative": ["89", 0],
            "vae": ["90", 0],
            "width": w,
            "height": h,
            "length": frames,
            "batch_size": 1,
            "start_image": start_ref,
        },
    }
    workflow["86"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": high_samp,
            "add_noise": "enable",
            "noise_seed": seed_v,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": d.sampler,
            "scheduler": d.scheduler,
            "positive": ["98", 0],
            "negative": ["98", 1],
            "latent_image": ["98", 2],
            "start_at_step": 0,
            "end_at_step": split,
            "return_with_leftover_noise": "enable",
        },
    }
    workflow["85"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": low_samp,
            "add_noise": "disable",
            "noise_seed": seed_v,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": d.sampler,
            "scheduler": d.scheduler,
            "positive": ["98", 0],
            "negative": ["98", 1],
            "latent_image": ["86", 0],
            "start_at_step": split,
            "end_at_step": steps,
            "return_with_leftover_noise": "disable",
        },
    }
    workflow["87"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["85", 0], "vae": ["90", 0]},
    }
    workflow["117"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["87", 0], "fps": fps_v},
    }
    workflow["200"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["117", 0],
            "filename_prefix": filename_prefix,
            "format": "mp4",
            "codec": "auto",
        },
    }

    meta = {
        "modality": "img2vid",
        "pack": "sample",
        "profile": profile,
        "seed": seed_v,
        "steps": steps,
        "cfg": cfg,
        "shift": d.shift,
        "sampler_name": d.sampler,
        "scheduler": d.scheduler,
        "width": w,
        "height": h,
        "length": frames,
        "fps": fps_v,
        "positive": positive,
        "negative": negative,
        "start_image": start_image_name if not use_video_prefix else None,
        "start_video": start_video_name if use_video_prefix else None,
        "motion_prefix_frames": (
            int(motion_prefix_frames)
            if motion_prefix_frames is not None
            else (None if not use_video_prefix else None)
        ),
        "motion_prefix": use_video_prefix,
        "high_unet": high_unet,
        "low_unet": low_unet,
        "loras": (
            [{"name": n, "weight": w, "noise": "high"} for n, w in hi_loras]
            + [{"name": n, "weight": w, "noise": "low"} for n, w in lo_loras]
        ),
        "text_encoder": text_encoder,
        "vae": vae_name,
        "context_windows": ctx.as_meta(),
    }
    if use_video_prefix and motion_prefix_frames is not None:
        meta["motion_prefix_frames"] = int(motion_prefix_frames)
    return workflow, meta


def build_wan_flf2vid(
    *,
    prompt: str,
    negative_prompt: str,
    start_image_name: str,
    end_image_name: str,
    high_unet: str,
    low_unet: str,
    high_loras: list[tuple[str, float]] | None = None,
    low_loras: list[tuple[str, float]] | None = None,
    text_encoder: str,
    vae_name: str,
    defaults: VideoDefaults,
    width: int | None = None,
    height: int | None = None,
    length: int | None = None,
    seed: int | None = None,
    fps: float | None = None,
    filename_prefix: str = "simplecomfy/video",
    profile: str = "quality",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    the video pack First–Last Frame → Video (same I2V dual-UNET weights).

    Used as a short bridge between a real parent tail frame and the head of a
    newly generated continue chunk.
    """
    d = defaults
    w = _snap16(width or d.width)
    h = _snap16(height or d.height)
    # Short bridges default ~1.5s
    frames = _snap_length(length if length is not None else 25)
    seed_v = resolve_seed(seed)
    steps = max(1, int(d.steps))
    cfg = float(d.cfg)
    split = int(d.high_end_step)
    split = max(1, min(split, steps - 1)) if steps > 1 else steps
    fps_v = float(fps if fps is not None else d.fps)
    positive = (prompt or "").strip()
    negative = (negative_prompt or d.negative or "").strip()
    hi_loras = list(high_loras or [])
    lo_loras = list(low_loras or [])

    workflow: dict[str, Any] = {
        "84": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder,
                "type": "wan",
                "device": "default",
            },
        },
        "90": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "95": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": high_unet, "weight_dtype": "default"},
        },
        "96": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": low_unet, "weight_dtype": "default"},
        },
    }
    high_ref, nid = _chain_model_loras(
        workflow, model_ref=["95", 0], loras=hi_loras, start_id=300
    )
    low_ref, nid = _chain_model_loras(
        workflow, model_ref=["96", 0], loras=lo_loras, start_id=nid
    )
    workflow["104"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": high_ref, "shift": float(d.shift)},
    }
    workflow["103"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": low_ref, "shift": float(d.shift)},
    }
    workflow["93"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": positive, "clip": ["84", 0]},
    }
    workflow["89"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": ["84", 0]},
    }
    workflow["10"] = {
        "class_type": "LoadImage",
        "inputs": {"image": start_image_name},
    }
    workflow["12"] = {
        "class_type": "LoadImage",
        "inputs": {"image": end_image_name},
    }
    workflow["98"] = {
        "class_type": "WanFirstLastFrameToVideo",
        "inputs": {
            "positive": ["93", 0],
            "negative": ["89", 0],
            "vae": ["90", 0],
            "width": w,
            "height": h,
            "length": frames,
            "batch_size": 1,
            "start_image": ["10", 0],
            "end_image": ["12", 0],
        },
    }
    workflow["86"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["104", 0],
            "add_noise": "enable",
            "noise_seed": seed_v,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": d.sampler,
            "scheduler": d.scheduler,
            "positive": ["98", 0],
            "negative": ["98", 1],
            "latent_image": ["98", 2],
            "start_at_step": 0,
            "end_at_step": split,
            "return_with_leftover_noise": "enable",
        },
    }
    workflow["85"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["103", 0],
            "add_noise": "disable",
            "noise_seed": seed_v,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": d.sampler,
            "scheduler": d.scheduler,
            "positive": ["98", 0],
            "negative": ["98", 1],
            "latent_image": ["86", 0],
            "start_at_step": split,
            "end_at_step": steps,
            "return_with_leftover_noise": "disable",
        },
    }
    workflow["87"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["85", 0], "vae": ["90", 0]},
    }
    workflow["117"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["87", 0], "fps": fps_v},
    }
    workflow["200"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["117", 0],
            "filename_prefix": filename_prefix,
            "format": "mp4",
            "codec": "auto",
        },
    }

    meta = {
        "modality": "flf2vid",
        "pack": "sample",
        "profile": profile,
        "seed": seed_v,
        "steps": steps,
        "cfg": cfg,
        "shift": d.shift,
        "sampler_name": d.sampler,
        "scheduler": d.scheduler,
        "width": w,
        "height": h,
        "length": frames,
        "fps": fps_v,
        "positive": positive,
        "negative": negative,
        "start_image": start_image_name,
        "end_image": end_image_name,
        "high_unet": high_unet,
        "low_unet": low_unet,
        "loras": (
            [{"name": n, "weight": w, "noise": "high"} for n, w in hi_loras]
            + [{"name": n, "weight": w, "noise": "low"} for n, w in lo_loras]
        ),
        "text_encoder": text_encoder,
        "vae": vae_name,
    }
    return workflow, meta
