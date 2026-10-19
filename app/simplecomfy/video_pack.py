"""the video pack video pack + profiles (draft / final / fast / quality)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from simplecomfy.config import AppConfig, repo_root

VIDEO_PROFILES = ("draft", "final", "fast", "quality")
# Profiles that finish in ~4 draft steps (short Comfy timeout)
FAST_PROFILES = frozenset({"fast", "draft"})


@dataclass
class VideoArtifact:
    role: str  # unet | lora | vae | text_encoder
    filename: str
    source: str
    noise: str = ""  # high | low | ""
    weight: float = 1.0


@dataclass
class VideoDefaults:
    """Resolved sampling defaults for a profile (or pack fallback)."""

    width: int = 640
    height: int = 640
    length: int = 81
    fps: float = 16.0
    steps: int = 4
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    shift: float = 5.0
    high_end_step: int = 2
    negative: str = ""


@dataclass
class VideoProfile:
    id: str
    label: str
    description: str = ""
    use_light_lora: bool = False
    use_final_lora: bool = False
    steps: int = 4
    cfg: float = 1.0
    shift: float = 5.0
    high_end_step: int = 2
    sampler: str = "euler"
    scheduler: str = "simple"
    negative: str = ""

    def to_defaults(self, base: VideoDefaults) -> VideoDefaults:
        return VideoDefaults(
            width=base.width,
            height=base.height,
            length=base.length,
            fps=base.fps,
            steps=self.steps,
            cfg=self.cfg,
            sampler=self.sampler,
            scheduler=self.scheduler,
            shift=self.shift,
            high_end_step=self.high_end_step,
            negative=self.negative or base.negative,
        )


@dataclass
class VideoPack:
    id: str
    label: str
    description: str = ""
    size_gb_approx: float | None = None
    default_profile: str = "draft"
    defaults: VideoDefaults = field(default_factory=VideoDefaults)
    shared: list[VideoArtifact] = field(default_factory=list)
    txt2vid_unets: list[VideoArtifact] = field(default_factory=list)
    img2vid_unets: list[VideoArtifact] = field(default_factory=list)
    txt2vid_light_loras: list[VideoArtifact] = field(default_factory=list)
    img2vid_light_loras: list[VideoArtifact] = field(default_factory=list)
    final_loras: list[VideoArtifact] = field(default_factory=list)
    profiles: dict[str, VideoProfile] = field(default_factory=dict)

    def profile(self, profile_id: str | None) -> VideoProfile:
        pid = (profile_id or self.default_profile or "quality").strip().lower()
        if pid not in self.profiles:
            # fallback chain
            for cand in (self.default_profile, "draft", "final", "quality", "fast"):
                if cand in self.profiles:
                    return self.profiles[cand]
            raise KeyError(f"no video profiles on pack {self.id}")
        return self.profiles[pid]

    def unets_for(self, modality: str) -> list[VideoArtifact]:
        if modality == "txt2vid":
            return list(self.txt2vid_unets)
        if modality == "img2vid":
            return list(self.img2vid_unets)
        return []

    def light_loras_for(self, modality: str) -> list[VideoArtifact]:
        if modality == "txt2vid":
            return list(self.txt2vid_light_loras)
        if modality == "img2vid":
            return list(self.img2vid_light_loras)
        return []

    def artifacts_for(
        self, modality: str, profile_id: str | None = None
    ) -> list[VideoArtifact]:
        """All files needed to run modality under profile."""
        prof = self.profile(profile_id)
        arts: list[VideoArtifact] = list(self.shared) + self.unets_for(modality)
        if prof.use_light_lora:
            arts.extend(self.light_loras_for(modality))
        if prof.use_final_lora:
            arts.extend(self.final_loras)
        # de-dupe by filename preserve order
        seen: set[str] = set()
        out: list[VideoArtifact] = []
        for a in arts:
            if a.filename in seen:
                continue
            seen.add(a.filename)
            out.append(a)
        return out

    def lora_stack_for(
        self, modality: str, profile_id: str | None
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """
        Return (high_noise_loras, low_noise_loras) as (filename, weight) lists
        in apply order.
        """
        prof = self.profile(profile_id)
        high: list[tuple[str, float]] = []
        low: list[tuple[str, float]] = []

        def add(arts: list[VideoArtifact]) -> None:
            for a in arts:
                pair = (a.filename, float(a.weight))
                n = (a.noise or "").lower()
                if n == "high":
                    high.append(pair)
                elif n == "low":
                    low.append(pair)
                else:
                    # apply to both if unspecified
                    high.append(pair)
                    low.append(pair)

        if prof.use_light_lora:
            add(self.light_loras_for(modality))
        if prof.use_final_lora:
            add(self.final_loras)
        return high, low

    def resolved_defaults(self, profile_id: str | None) -> VideoDefaults:
        return self.profile(profile_id).to_defaults(self.defaults)


def _arts(raw: list[Any] | None) -> list[VideoArtifact]:
    out: list[VideoArtifact] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        out.append(
            VideoArtifact(
                role=str(item.get("role") or "unet"),
                filename=str(item.get("filename") or ""),
                source=str(item.get("source") or ""),
                noise=str(item.get("noise") or ""),
                weight=float(item.get("weight") if item.get("weight") is not None else 1.0),
            )
        )
    return out


def _profiles(raw: dict[str, Any] | None) -> dict[str, VideoProfile]:
    out: dict[str, VideoProfile] = {}
    for pid, data in (raw or {}).items():
        if not isinstance(data, dict):
            continue
        out[str(pid)] = VideoProfile(
            id=str(pid),
            label=str(data.get("label") or pid),
            description=str(data.get("description") or ""),
            use_light_lora=bool(data.get("use_light_lora", False)),
            use_final_lora=bool(data.get("use_final_lora", False)),
            steps=int(data.get("steps") if data.get("steps") is not None else 4),
            cfg=float(data.get("cfg") if data.get("cfg") is not None else 1.0),
            shift=float(data.get("shift") if data.get("shift") is not None else 5.0),
            high_end_step=int(
                data.get("high_end_step") if data.get("high_end_step") is not None else 2
            ),
            sampler=str(data.get("sampler") or "euler"),
            scheduler=str(data.get("scheduler") or "simple"),
            negative=str(data.get("negative") or ""),
        )
    return out


def load_video_pack(path: Path) -> VideoPack:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    d = data.get("defaults") or {}

    # Back-compat: old flat txt2vid/img2vid lists
    txt_unets = _arts(data.get("txt2vid_unets"))
    img_unets = _arts(data.get("img2vid_unets"))
    txt_light = _arts(data.get("txt2vid_light_loras"))
    img_light = _arts(data.get("img2vid_light_loras"))
    if not txt_unets and data.get("txt2vid"):
        old = _arts(data.get("txt2vid"))
        txt_unets = [a for a in old if a.role == "unet"]
        txt_light = [a for a in old if a.role == "lora"]
    if not img_unets and data.get("img2vid"):
        old = _arts(data.get("img2vid"))
        img_unets = [a for a in old if a.role == "unet"]
        img_light = [a for a in old if a.role == "lora"]

    profiles = _profiles(data.get("profiles"))
    if not profiles:
        # synthesize legacy single profile
        profiles = {
            "fast": VideoProfile(
                id="fast",
                label="Fast",
                use_light_lora=True,
                steps=int(d.get("steps") or 4),
                cfg=float(d.get("cfg") if d.get("cfg") is not None else 1.0),
                shift=float(d.get("shift") if d.get("shift") is not None else 5.0),
                high_end_step=int(d.get("high_end_step") or 2),
                sampler=str(d.get("sampler") or "euler"),
                scheduler=str(d.get("scheduler") or "simple"),
                negative=str(d.get("negative") or ""),
            )
        }

    return VideoPack(
        id=str(data.get("id") or path.stem),
        label=str(data.get("label") or path.stem),
        description=str(data.get("description") or ""),
        size_gb_approx=(
            float(data["size_gb_approx"]) if data.get("size_gb_approx") is not None else None
        ),
        default_profile=str(data.get("default_profile") or "draft"),
        defaults=VideoDefaults(
            width=int(d.get("width") or 640),
            height=int(d.get("height") or 640),
            length=int(d.get("length") or 81),
            fps=float(d.get("fps") or 16),
            steps=int(d.get("steps") or 4),
            cfg=float(d.get("cfg") if d.get("cfg") is not None else 1.0),
            sampler=str(d.get("sampler") or "euler"),
            scheduler=str(d.get("scheduler") or "simple"),
            shift=float(d.get("shift") if d.get("shift") is not None else 5.0),
            high_end_step=int(d.get("high_end_step") or 2),
            negative=str(d.get("negative") or ""),
        ),
        shared=_arts(data.get("shared")),
        txt2vid_unets=txt_unets,
        img2vid_unets=img_unets,
        txt2vid_light_loras=txt_light,
        img2vid_light_loras=img_light,
        final_loras=_arts(data.get("final_loras")),
        profiles=profiles,
    )


class VideoPackRegistry:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._by_id: dict[str, VideoPack] = {}
        self.reload()

    @property
    def root(self) -> Path:
        return repo_root() / "video"

    def reload(self) -> None:
        self._by_id.clear()
        if not self.root.is_dir():
            return
        for path in sorted(self.root.glob("*.yaml")):
            pack = load_video_pack(path)
            self._by_id[pack.id] = pack

    def list(self) -> list[VideoPack]:
        return list(self._by_id.values())

    def get(self, pack_id: str) -> VideoPack | None:
        return self._by_id.get(pack_id)

    def default(self) -> VideoPack:
        if "sample" in self._by_id:
            return self._by_id["sample"]
        if self._by_id:
            return next(iter(self._by_id.values()))
        raise KeyError("no video packs configured under video/")
