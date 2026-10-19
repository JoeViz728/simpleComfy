from __future__ import annotations

from simplecomfy.config import load_config, repo_root
from simplecomfy.video_pack import VideoPackRegistry, load_video_pack
from simplecomfy.workflows import (
    build_wan_flf2vid,
    build_wan_img2vid,
    build_wan_txt2vid,
    resolve_context_windows,
)


def test_sample_pack_profiles():
    path = repo_root() / "video" / "sample.yaml"
    pack = load_video_pack(path)
    assert pack.id == "sample"
    assert set(pack.profiles) >= {"draft", "final"}
    assert pack.default_profile == "draft"
    draft = pack.artifacts_for("img2vid", "draft")
    final = pack.artifacts_for("img2vid", "final")
    assert any("draft" in a.filename for a in draft)
    assert not any("draft" in a.filename for a in final)
    hi, lo = pack.lora_stack_for("img2vid", "draft")
    assert len(hi) >= 1 and len(lo) >= 1
    assert pack.resolved_defaults("draft").steps == 4


def test_build_wan_t2v_final_no_lora():
    cfg = load_config(None)
    pack = VideoPackRegistry(cfg).default()
    d = pack.resolved_defaults("final")
    wf, meta = build_wan_txt2vid(
        prompt="a cat walking in rain",
        negative_prompt="blurry",
        high_unet="hi.safetensors",
        low_unet="lo.safetensors",
        high_loras=[],
        low_loras=[],
        text_encoder="umt5.safetensors",
        vae_name="wan_vae.safetensors",
        defaults=d,
        seed=1,
        profile="final",
    )
    assert wf["81"]["inputs"]["steps"] == d.steps
    assert meta["profile"] == "final"
    assert meta["steps"] >= 20
    # no LoraLoader when stacks empty
    assert not any(
        isinstance(n, dict) and n.get("class_type") == "LoraLoaderModelOnly"
        for n in wf.values()
    )


def test_build_wan_i2v_draft_loras():
    cfg = load_config(None)
    pack = VideoPackRegistry(cfg).default()
    d = pack.resolved_defaults("draft")
    hi, lo = pack.lora_stack_for("img2vid", "draft")
    wf, meta = build_wan_img2vid(
        prompt="slow zoom",
        negative_prompt="",
        start_image_name="frame.png",
        high_unet="hi.safetensors",
        low_unet="lo.safetensors",
        high_loras=hi,
        low_loras=lo,
        text_encoder="umt5.safetensors",
        vae_name="wan_vae.safetensors",
        defaults=d,
        seed=2,
        profile="draft",
    )
    assert wf["98"]["class_type"] == "WanImageToVideo"
    lora_nodes = [
        n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "LoraLoaderModelOnly"
    ]
    assert len(lora_nodes) >= 2
    assert meta["profile"] == "draft"
    assert meta["steps"] == 4


def test_resolve_context_windows_auto():
    off = resolve_context_windows(total_frames=81, enabled=None)
    assert off.enabled is False
    assert off.mode == "auto"
    on = resolve_context_windows(total_frames=161, enabled=None, context_length=81)
    assert on.enabled is True
    assert on.context_length == 81
    assert on.context_overlap == 30
    forced_off = resolve_context_windows(total_frames=200, enabled=False)
    assert forced_off.enabled is False
    forced_on = resolve_context_windows(total_frames=40, enabled=True, context_length=81)
    assert forced_on.enabled is True
    # overlap clamped below window
    hi = resolve_context_windows(
        total_frames=200, enabled=True, context_length=41, context_overlap=999
    )
    assert hi.context_overlap == 40


def test_build_wan_long_clip_injects_context_nodes():
    cfg = load_config(None)
    pack = VideoPackRegistry(cfg).default()
    d = pack.resolved_defaults("draft")
    wf, meta = build_wan_txt2vid(
        prompt="long walk",
        negative_prompt="",
        high_unet="hi.safetensors",
        low_unet="lo.safetensors",
        text_encoder="umt5.safetensors",
        vae_name="wan_vae.safetensors",
        defaults=d,
        length=161,
        seed=9,
        profile="draft",
        context_windows=None,  # auto
        context_length=81,
        context_overlap=30,
    )
    ctx_nodes = [
        n
        for n in wf.values()
        if isinstance(n, dict) and n.get("class_type") == "WanContextWindowsManual"
    ]
    assert len(ctx_nodes) == 2  # high + low noise
    assert meta["context_windows"]["enabled"] is True
    assert meta["length"] == 161
    # samplers should point at context-patched models
    assert wf["81"]["inputs"]["model"] == ["400", 0]
    assert wf["78"]["inputs"]["model"] == ["401", 0]


def test_build_wan_flf2vid_node():
    cfg = load_config(None)
    pack = VideoPackRegistry(cfg).default()
    d = pack.resolved_defaults("draft")
    wf, meta = build_wan_flf2vid(
        prompt="bridge",
        negative_prompt="",
        start_image_name="a.png",
        end_image_name="b.png",
        high_unet="hi.safetensors",
        low_unet="lo.safetensors",
        text_encoder="umt5.safetensors",
        vae_name="wan_vae.safetensors",
        defaults=d,
        length=25,
        seed=7,
        profile="draft",
    )
    assert wf["98"]["class_type"] == "WanFirstLastFrameToVideo"
    assert wf["98"]["inputs"]["start_image"] == ["10", 0]
    assert wf["98"]["inputs"]["end_image"] == ["12", 0]
    assert meta["modality"] == "flf2vid"
    assert meta["length"] == 25


def test_build_wan_i2v_motion_prefix_video():
    cfg = load_config(None)
    pack = VideoPackRegistry(cfg).default()
    d = pack.resolved_defaults("draft")
    wf, meta = build_wan_img2vid(
        prompt="keep moving",
        negative_prompt="",
        start_video_name="tail.mp4",
        high_unet="hi.safetensors",
        low_unet="lo.safetensors",
        text_encoder="umt5.safetensors",
        vae_name="wan_vae.safetensors",
        defaults=d,
        length=93,
        seed=4,
        profile="draft",
        motion_prefix_frames=12,
    )
    assert wf["10"]["class_type"] == "LoadVideo"
    assert wf["11"]["class_type"] == "GetVideoComponents"
    assert wf["98"]["inputs"]["start_image"] == ["11", 0]
    assert meta["motion_prefix"] is True
    assert meta["motion_prefix_frames"] == 12
    assert meta["start_video"] == "tail.mp4"


def test_build_wan_short_clip_no_context_nodes():
    cfg = load_config(None)
    pack = VideoPackRegistry(cfg).default()
    d = pack.resolved_defaults("draft")
    wf, meta = build_wan_img2vid(
        prompt="short",
        negative_prompt="",
        start_image_name="x.png",
        high_unet="hi.safetensors",
        low_unet="lo.safetensors",
        text_encoder="umt5.safetensors",
        vae_name="wan_vae.safetensors",
        defaults=d,
        length=81,
        seed=1,
        profile="draft",
    )
    assert not any(
        isinstance(n, dict) and n.get("class_type") == "WanContextWindowsManual"
        for n in wf.values()
    )
    assert meta["context_windows"]["enabled"] is False
    # retain_first_frame flagged for i2v even when disabled (for when enabled)
    assert meta["context_windows"]["retain_first_frame"] is True
