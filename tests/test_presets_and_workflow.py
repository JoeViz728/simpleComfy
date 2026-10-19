from __future__ import annotations

import pytest

from simplecomfy.config import load_config, repo_root
from simplecomfy.presets import PresetRegistry
from simplecomfy.workflows import build_txt2img


def test_repo_has_starter_preset():
    ids = {p.stem for p in (repo_root() / "presets").glob("*.yaml")}
    assert "sample" in ids


def test_registry_loads():
    cfg = load_config(None)
    reg = PresetRegistry(cfg)
    preset = reg.require("sample")
    assert preset.base_family == "sdxl"
    assert preset.checkpoint_filename == "sample.safetensors"


def test_build_txt2img_graph():
    cfg = load_config(None)
    reg = PresetRegistry(cfg)
    preset = reg.require("sample")
    wf, meta = build_txt2img(
        preset=preset,
        prompt="a red chair",
        negative_prompt="",
        checkpoint_name="dummy.safetensors",
        lora_files=[("style.safetensors", 0.6)],
        width=1024,
        height=1024,
        seed=42,
    )
    assert "4" in wf and wf["4"]["class_type"] == "CheckpointLoaderSimple"
    assert wf["4"]["inputs"]["ckpt_name"] == "dummy.safetensors"
    assert any(n.get("class_type") == "LoraLoader" for n in wf.values() if isinstance(n, dict))
    assert meta["seed"] == 42
    assert "a red chair" in meta["positive"]
    assert all(not k.startswith("_") for k in wf)


def test_unknown_preset():
    cfg = load_config(None)
    reg = PresetRegistry(cfg)
    with pytest.raises(KeyError):
        reg.require("nope")
