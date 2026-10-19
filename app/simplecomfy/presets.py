"""Preset registry — YAML files under presets/."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from simplecomfy.config import AppConfig


@dataclass
class LoraSpec:
    source: str
    weight: float = 0.7
    # Optional stable local filename; resolved from pull if empty
    filename: str = ""


@dataclass
class SamplerSpec:
    name: str = "euler_ancestral"
    scheduler: str = "normal"
    steps: int = 28
    cfg: float = 5.5


@dataclass
class SizeSpec:
    width: int = 1024
    height: int = 1024


@dataclass
class PromptSpec:
    prefix: str = ""
    suffix: str = ""
    negative: str = ""


@dataclass
class Preset:
    id: str
    label: str
    description: str = ""
    base_family: str = "sdxl"  # sdxl | illustrious | pony | sd15
    checkpoint_source: str = ""
    checkpoint_filename: str = ""
    loras: list[LoraSpec] = field(default_factory=list)
    # Effect ids pre-selected when client omits `effects` on generate
    default_effects: list[str] = field(default_factory=list)
    # Optional effect ids that work well with this preset (UI hints)
    suggested_effects: list[str] = field(default_factory=list)
    prompt: PromptSpec = field(default_factory=PromptSpec)
    sampler: SamplerSpec = field(default_factory=SamplerSpec)
    size: SizeSpec = field(default_factory=SizeSpec)
    # approx total download GB for UI/LLM honesty
    size_gb_approx: float | None = None
    path: Path | None = None

    def build_positive(self, user_prompt: str) -> str:
        parts = [self.prompt.prefix.strip(), user_prompt.strip(), self.prompt.suffix.strip()]
        return ", ".join(p for p in parts if p)

    def build_negative(self, user_negative: str | None) -> str:
        parts = [self.prompt.negative.strip()]
        if user_negative:
            parts.append(user_negative.strip())
        return ", ".join(p for p in parts if p)


def _lora_from(raw: dict[str, Any]) -> LoraSpec:
    return LoraSpec(
        source=str(raw.get("source") or ""),
        weight=float(raw.get("weight") if raw.get("weight") is not None else 0.7),
        filename=str(raw.get("filename") or ""),
    )


def preset_from_dict(data: dict[str, Any], *, path: Path | None = None) -> Preset:
    samp = data.get("sampler") or {}
    size = data.get("size") or {}
    prompt = data.get("prompt") or {}
    loras_raw = data.get("loras") or []
    default_fx = data.get("default_effects") or []
    suggested_fx = data.get("suggested_effects") or []
    if isinstance(default_fx, str):
        default_fx = [default_fx]
    if isinstance(suggested_fx, str):
        suggested_fx = [suggested_fx]
    return Preset(
        id=str(data["id"]),
        label=str(data.get("label") or data["id"]),
        description=str(data.get("description") or ""),
        base_family=str(data.get("base_family") or "sdxl"),
        checkpoint_source=str(data.get("checkpoint_source") or ""),
        checkpoint_filename=str(data.get("checkpoint_filename") or ""),
        loras=[_lora_from(x) for x in loras_raw if isinstance(x, dict)],
        default_effects=[str(x) for x in default_fx],
        suggested_effects=[str(x) for x in suggested_fx],
        prompt=PromptSpec(
            prefix=str(prompt.get("prefix") or ""),
            suffix=str(prompt.get("suffix") or ""),
            negative=str(prompt.get("negative") or ""),
        ),
        sampler=SamplerSpec(
            name=str(samp.get("name") or "euler_ancestral"),
            scheduler=str(samp.get("scheduler") or "normal"),
            steps=int(samp.get("steps") if samp.get("steps") is not None else 28),
            cfg=float(samp.get("cfg") if samp.get("cfg") is not None else 5.5),
        ),
        size=SizeSpec(
            width=int(size.get("width") if size.get("width") is not None else 1024),
            height=int(size.get("height") if size.get("height") is not None else 1024),
        ),
        size_gb_approx=(
            float(data["size_gb_approx"]) if data.get("size_gb_approx") is not None else None
        ),
        path=path,
    )


class PresetRegistry:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._by_id: dict[str, Preset] = {}
        self.reload()

    def reload(self) -> None:
        self._by_id.clear()
        root = self.cfg.presets_dir
        if not root.is_dir():
            return
        for path in sorted(root.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict) or "id" not in data:
                continue
            p = preset_from_dict(data, path=path)
            self._by_id[p.id] = p
        for path in sorted(root.glob("*.yml")):
            if path.with_suffix(".yaml").exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict) or "id" not in data:
                continue
            p = preset_from_dict(data, path=path)
            self._by_id[p.id] = p

    def get(self, preset_id: str) -> Preset | None:
        return self._by_id.get(preset_id)

    def list(self) -> list[Preset]:
        return list(self._by_id.values())

    def require(self, preset_id: str) -> Preset:
        p = self.get(preset_id)
        if not p:
            known = ", ".join(sorted(self._by_id)) or "(none)"
            raise KeyError(f"unknown preset '{preset_id}'; known: {known}")
        return p
