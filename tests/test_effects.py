from __future__ import annotations

import pytest

from simplecomfy.config import load_config
from simplecomfy.effects import (
    EffectRegistry,
    effect_lora_stack,
    merge_prompts,
    merge_video_prompts,
)
from simplecomfy.presets import PresetRegistry


def test_effects_load():
    cfg = load_config(None)
    reg = EffectRegistry(cfg)
    ids = {e.id for e in reg.list()}
    assert {"soft_light", "film_still", "detail", "smooth_motion", "motion_a"} <= ids
    assert reg.require("detail").needs_lora
    assert not reg.require("film_still").needs_lora
    assert reg.require("motion_a").applies_to_modality("img2vid")
    assert not reg.require("motion_a").applies_to_modality("txt2img")
    assert reg.require("detail").applies_to_modality("txt2img")
    assert not reg.require("detail").applies_to_modality("img2vid")


def test_family_filter():
    cfg = load_config(None)
    reg = EffectRegistry(cfg)
    detail = reg.require("detail")
    assert detail.compatible_with("sdxl")
    assert not detail.compatible_with("illustrious")
    film = reg.require("film_still")
    assert film.compatible_with("illustrious")


def test_resolve_defaults_and_empty():
    cfg = load_config(None)
    presets = PresetRegistry(cfg)
    effects = EffectRegistry(cfg)
    sample = presets.require("sample")
    assert "soft_light" in sample.default_effects

    resolved = effects.resolve_for_generate(sample, None)
    assert [e.id for e in resolved] == ["soft_light"]

    none = effects.resolve_for_generate(sample, [])
    assert none == []

    with pytest.raises(ValueError):
        effects.resolve_for_generate(sample, ["motion_a"])


def test_resolve_video_and_exclusive_group():
    cfg = load_config(None)
    effects = EffectRegistry(cfg)
    resolved = effects.resolve_for_video(
        ["smooth_motion", "motion_a", "motion_b"],
        modality="img2vid",
    )
    ids = [e.id for e in resolved]
    assert "smooth_motion" in ids
    assert "motion_b" in ids
    assert "motion_a" not in ids

    with pytest.raises(ValueError):
        effects.resolve_for_video(["soft_light"], modality="img2vid")

    hi, lo = effect_lora_stack(resolved)
    assert hi and lo
    assert all(len(t) == 4 for t in hi)


def test_merge_prompts():
    cfg = load_config(None)
    presets = PresetRegistry(cfg)
    effects = EffectRegistry(cfg)
    preset = presets.require("sample")
    fx = effects.resolve_for_generate(preset, ["soft_light", "film_still"])
    pos, neg = merge_prompts(preset, fx, "a red chair", "blurry")
    assert "photograph" in pos
    assert "a red chair" in pos
    assert "film" in pos.lower()
    assert "blurry" in neg


def test_merge_video_prompts():
    cfg = load_config(None)
    effects = EffectRegistry(cfg)
    fx = effects.resolve_for_video(["smooth_motion", "motion_b"], modality="img2vid")
    pos, neg = merge_video_prompts(fx, "the camera pans", "static", pack_negative="blurry")
    assert "the camera pans" in pos
    assert "smooth" in pos.lower()
    assert "handheld" in pos.lower()
    assert "blurry" in neg
    assert "static" in neg
