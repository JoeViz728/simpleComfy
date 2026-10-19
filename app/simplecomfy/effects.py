"""Optional effects — LoRAs and prompt-only filters for image and video jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from simplecomfy.config import AppConfig
from simplecomfy.presets import LoraSpec, Preset, PromptSpec

# Soft cap when stacking optional effects (image or video)
MAX_EFFECTS = 6


@dataclass
class EffectLora:
    """One LoRA file belonging to an effect (WAN may need high+low pairs)."""

    source: str = ""
    filename: str = ""
    noise: str = ""  # high | low | "" (apply to both stages)
    weight: float | None = None  # None → use effect.weight


@dataclass
class Effect:
    id: str
    label: str
    description: str = ""
    # Image: empty = any family; non-empty = only those base_families
    base_families: list[str] = field(default_factory=list)
    # Modalities this effect applies to. Empty = legacy image (txt2img only).
    modalities: list[str] = field(default_factory=list)
    # Only one effect per group may be active (e.g. bounce | twerk)
    exclusive_group: str = ""
    # Single-file LoRA (legacy image style) — optional if `loras` is set
    source: str = ""
    filename: str = ""
    weight: float = 0.7
    # Multi-file / dual-noise LoRAs (preferred for WAN video)
    loras: list[EffectLora] = field(default_factory=list)
    prompt: PromptSpec = field(default_factory=PromptSpec)
    size_mb_approx: float | None = None
    path: Path | None = None

    @property
    def needs_lora(self) -> bool:
        if self.loras:
            return any(L.source or L.filename for L in self.loras)
        return bool(self.source or self.filename)

    def applies_to_modality(self, modality: str) -> bool:
        m = (modality or "").strip().lower()
        if self.modalities:
            return m in {x.lower() for x in self.modalities}
        # Legacy image effects (no modalities key) → txt2img only
        return m == "txt2img"

    def compatible_with(self, base_family: str) -> bool:
        if not self.base_families:
            return True
        fam = (base_family or "").lower()
        return fam in {f.lower() for f in self.base_families}

    def lora_files(self) -> list[EffectLora]:
        """Normalized list of LoRA files for pull / stack."""
        if self.loras:
            out: list[EffectLora] = []
            for L in self.loras:
                if not (L.source or L.filename):
                    continue
                out.append(
                    EffectLora(
                        source=L.source,
                        filename=L.filename,
                        noise=(L.noise or "").lower(),
                        weight=float(L.weight) if L.weight is not None else None,
                    )
                )
            return out
        if self.source or self.filename:
            return [
                EffectLora(
                    source=self.source,
                    filename=self.filename,
                    noise="",
                    weight=None,
                )
            ]
        return []

    def as_lora_spec(self, weight: float | None = None) -> LoraSpec | None:
        """First LoRA only — for legacy image paths."""
        files = self.lora_files()
        if not files:
            return None
        L = files[0]
        w = weight if weight is not None else (
            float(L.weight) if L.weight is not None else self.weight
        )
        return LoraSpec(source=L.source, weight=float(w), filename=L.filename)

    def as_lora_specs(self, weight: float | None = None) -> list[LoraSpec]:
        specs: list[LoraSpec] = []
        for L in self.lora_files():
            w = weight if weight is not None else (
                float(L.weight) if L.weight is not None else self.weight
            )
            specs.append(
                LoraSpec(source=L.source, weight=float(w), filename=L.filename)
            )
        return specs

    def with_weight(self, weight: float) -> Effect:
        return Effect(
            id=self.id,
            label=self.label,
            description=self.description,
            base_families=list(self.base_families),
            modalities=list(self.modalities),
            exclusive_group=self.exclusive_group,
            source=self.source,
            filename=self.filename,
            weight=float(weight),
            loras=[
                EffectLora(
                    source=L.source,
                    filename=L.filename,
                    noise=L.noise,
                    weight=L.weight,
                )
                for L in self.loras
            ],
            prompt=self.prompt,
            size_mb_approx=self.size_mb_approx,
            path=self.path,
        )


def _loras_from_dict(data: dict[str, Any]) -> list[EffectLora]:
    raw = data.get("loras")
    out: list[EffectLora] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            out.append(
                EffectLora(
                    source=str(item.get("source") or ""),
                    filename=str(item.get("filename") or ""),
                    noise=str(item.get("noise") or ""),
                    weight=(
                        float(item["weight"])
                        if item.get("weight") is not None
                        else None
                    ),
                )
            )
    # Convenience: high: / low: blocks
    for noise_key in ("high", "low"):
        block = data.get(noise_key)
        if isinstance(block, dict) and (block.get("source") or block.get("filename")):
            out.append(
                EffectLora(
                    source=str(block.get("source") or ""),
                    filename=str(block.get("filename") or ""),
                    noise=noise_key,
                    weight=(
                        float(block["weight"])
                        if block.get("weight") is not None
                        else None
                    ),
                )
            )
    return out


def _effect_from_dict(data: dict[str, Any], *, path: Path | None = None) -> Effect:
    prompt = data.get("prompt") or {}
    fams = data.get("base_families") or data.get("families") or []
    if isinstance(fams, str):
        fams = [fams]
    mods = data.get("modalities") or data.get("modality") or []
    if isinstance(mods, str):
        mods = [mods]
    return Effect(
        id=str(data["id"]),
        label=str(data.get("label") or data["id"]),
        description=str(data.get("description") or ""),
        base_families=[str(f) for f in fams],
        modalities=[str(m).lower() for m in mods],
        exclusive_group=str(data.get("exclusive_group") or data.get("group") or ""),
        source=str(data.get("source") or ""),
        filename=str(data.get("filename") or ""),
        weight=float(data.get("weight") if data.get("weight") is not None else 0.7),
        loras=_loras_from_dict(data),
        prompt=PromptSpec(
            prefix=str(prompt.get("prefix") or ""),
            suffix=str(prompt.get("suffix") or ""),
            negative=str(prompt.get("negative") or ""),
        ),
        size_mb_approx=(
            float(data["size_mb_approx"])
            if data.get("size_mb_approx") is not None
            else None
        ),
        path=path,
    )


def _apply_exclusive_groups(ordered_ids: list[str], by_id: dict[str, Effect]) -> list[str]:
    """Keep last-selected effect in each exclusive_group (selection order)."""
    blocked: set[str] = set()
    kept_rev: list[str] = []
    for eid in reversed(ordered_ids):
        eff = by_id.get(eid)
        g = (eff.exclusive_group or "").strip() if eff else ""
        if g:
            if g in blocked:
                continue
            blocked.add(g)
        kept_rev.append(eid)
    return list(reversed(kept_rev))


class EffectRegistry:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._by_id: dict[str, Effect] = {}
        self.reload()

    def reload(self) -> None:
        self._by_id.clear()
        root = self.cfg.effects_dir
        if not root.is_dir():
            return
        for path in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
            if path.suffix == ".yml" and path.with_suffix(".yaml").exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict) or "id" not in data:
                continue
            e = _effect_from_dict(data, path=path)
            self._by_id[e.id] = e

    def get(self, effect_id: str) -> Effect | None:
        return self._by_id.get(effect_id)

    def list(self) -> list[Effect]:
        return list(self._by_id.values())

    def list_for_modality(self, modality: str) -> list[Effect]:
        return [e for e in self.list() if e.applies_to_modality(modality)]

    def require(self, effect_id: str) -> Effect:
        e = self.get(effect_id)
        if not e:
            known = ", ".join(sorted(self._by_id)) or "(none)"
            raise KeyError(f"unknown effect '{effect_id}'; known: {known}")
        return e

    def resolve_for_generate(
        self,
        preset: Preset,
        effect_ids: list[str] | None,
        *,
        weights: dict[str, float] | None = None,
    ) -> list[Effect]:
        """
        Image path. effect_ids is None → use preset.default_effects.
        effect_ids is [] → no optional effects.
        """
        weights = weights or {}
        if effect_ids is None:
            ids = list(preset.default_effects)
        else:
            ids = list(effect_ids)

        seen: set[str] = set()
        ordered: list[str] = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
        ordered = _apply_exclusive_groups(ordered, self._by_id)

        out: list[Effect] = []
        for eid in ordered:
            eff = self.require(eid)
            if not eff.applies_to_modality("txt2img"):
                raise ValueError(
                    f"effect '{eid}' is not an image (txt2img) effect "
                    f"(modalities: {eff.modalities or 'txt2img'})"
                )
            if not eff.compatible_with(preset.base_family):
                raise ValueError(
                    f"effect '{eid}' is not compatible with base_family "
                    f"'{preset.base_family}' (allows: {eff.base_families or 'any'})"
                )
            if eid in weights:
                eff = eff.with_weight(float(weights[eid]))
            out.append(eff)
        if len(out) > MAX_EFFECTS:
            raise ValueError(f"too many effects (max {MAX_EFFECTS})")
        return out

    def resolve_for_video(
        self,
        effect_ids: list[str] | None,
        *,
        modality: str,
        weights: dict[str, float] | None = None,
    ) -> list[Effect]:
        """
        Video path (txt2vid / img2vid / continue chunk).
        None or [] → no optional video effects (profiles still supply baseline LoRAs).
        """
        weights = weights or {}
        if not effect_ids:
            return []
        seen: set[str] = set()
        ordered: list[str] = []
        for i in effect_ids:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
        ordered = _apply_exclusive_groups(ordered, self._by_id)

        out: list[Effect] = []
        mod = (modality or "").strip().lower()
        # continue uses img2vid LoRAs under the hood
        check_mod = "img2vid" if mod == "continue" else mod
        for eid in ordered:
            eff = self.require(eid)
            if not eff.applies_to_modality(check_mod):
                raise ValueError(
                    f"effect '{eid}' does not apply to modality '{check_mod}' "
                    f"(allows: {eff.modalities or 'txt2img'})"
                )
            if eid in weights:
                eff = eff.with_weight(float(weights[eid]))
            out.append(eff)
        if len(out) > MAX_EFFECTS:
            raise ValueError(f"too many effects (max {MAX_EFFECTS})")
        return out


def merge_prompts(
    preset: Preset,
    effects: list[Effect],
    user_prompt: str,
    user_negative: str | None,
) -> tuple[str, str]:
    """Build final positive/negative with preset + effect glue + user text."""
    pre_bits = [preset.prompt.prefix.strip()]
    pre_bits += [e.prompt.prefix.strip() for e in effects if e.prompt.prefix.strip()]
    suf_bits = [e.prompt.suffix.strip() for e in effects if e.prompt.suffix.strip()]
    suf_bits.append(preset.prompt.suffix.strip())

    positive_parts = [p for p in pre_bits if p]
    positive_parts.append(user_prompt.strip())
    positive_parts += [p for p in suf_bits if p]
    positive = ", ".join(positive_parts)

    neg_parts = [preset.prompt.negative.strip()]
    neg_parts += [e.prompt.negative.strip() for e in effects if e.prompt.negative.strip()]
    if user_negative:
        neg_parts.append(user_negative.strip())
    negative = ", ".join(p for p in neg_parts if p)
    return positive, negative


def merge_video_prompts(
    effects: list[Effect],
    user_prompt: str,
    user_negative: str | None,
    *,
    pack_negative: str = "",
) -> tuple[str, str]:
    """Prompt glue for video jobs (no image preset)."""
    pre = [e.prompt.prefix.strip() for e in effects if e.prompt.prefix.strip()]
    suf = [e.prompt.suffix.strip() for e in effects if e.prompt.suffix.strip()]
    parts = [p for p in pre if p]
    if user_prompt.strip():
        parts.append(user_prompt.strip())
    parts += [p for p in suf if p]
    positive = ", ".join(parts)

    neg_parts = [pack_negative.strip()] if pack_negative else []
    neg_parts += [e.prompt.negative.strip() for e in effects if e.prompt.negative.strip()]
    if user_negative:
        neg_parts.append(user_negative.strip())
    negative = ", ".join(p for p in neg_parts if p)
    return positive, negative


def effect_lora_stack(
    effects: list[Effect],
) -> tuple[list[tuple[str, float, str, str]], list[tuple[str, float, str, str]]]:
    """
    Build (high, low) stacks of (filename, weight, source, effect_id)
    for ensure + apply. Filename may be empty until resolved.
    """
    high: list[tuple[str, float, str, str]] = []
    low: list[tuple[str, float, str, str]] = []
    for eff in effects:
        for L in eff.lora_files():
            w = float(L.weight) if L.weight is not None else float(eff.weight)
            entry = (L.filename or "", w, L.source or "", eff.id)
            n = (L.noise or "").lower()
            if n == "high":
                high.append(entry)
            elif n == "low":
                low.append(entry)
            else:
                high.append(entry)
                low.append(entry)
    return high, low


def summarize_effect(effect: Effect, puller: Any) -> dict[str, Any]:
    missing = False
    local_names: list[str] = []
    files_out: list[dict[str, Any]] = []
    if effect.needs_lora:
        for L in effect.lora_files():
            local_name = puller.resolved_name(
                role="lora", filename=L.filename, source=L.source
            )
            if local_name is None:
                missing = True
            else:
                local_names.append(local_name)
            files_out.append(
                {
                    "filename": L.filename or local_name,
                    "source": L.source or None,
                    "noise": L.noise or None,
                    "ready": local_name is not None,
                }
            )
    return {
        "id": effect.id,
        "label": effect.label,
        "description": effect.description,
        "base_families": effect.base_families,
        "modalities": effect.modalities or ["txt2img"],
        "exclusive_group": effect.exclusive_group or None,
        "needs_lora": effect.needs_lora,
        "ready": not missing,
        "weight": effect.weight,
        "source": effect.source or None,
        "filename": effect.filename or (local_names[0] if local_names else None),
        "loras": files_out,
        "size_mb_approx": effect.size_mb_approx,
        "prompt": {
            "prefix": effect.prompt.prefix,
            "suffix": effect.prompt.suffix,
            "negative": effect.prompt.negative,
        },
    }
