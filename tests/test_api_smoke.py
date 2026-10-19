from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from simplecomfy.api import create_app
from simplecomfy.config import load_config


@pytest.fixture()
def client(tmp_path, monkeypatch):
    cfg = load_config(None)
    cfg.data_dir = tmp_path / "data"
    cfg.ensure_dirs()
    # Don't spawn comfy in tests
    cfg.comfy.root = ""
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def test_help_and_health(client: TestClient):
    r = client.get("/v1/help")
    assert r.status_code == 200
    assert "generate" in r.json()["endpoints"]

    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "comfy_reachable" in body


def test_list_presets(client: TestClient):
    r = client.get("/v1/presets")
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()["presets"]}
    assert "sample" in ids


def test_list_effects(client: TestClient):
    r = client.get("/v1/effects")
    assert r.status_code == 200
    ids = {e["id"] for e in r.json()["effects"]}
    assert "detail" in ids
    assert "film_still" in ids

    r = client.get("/v1/effects", params={"family": "sdxl"})
    assert r.status_code == 200
    sdxl_ids = {e["id"] for e in r.json()["effects"]}
    assert "detail" in sdxl_ids

    r = client.get("/v1/effects", params={"family": "illustrious"})
    ill_ids = {e["id"] for e in r.json()["effects"]}
    assert "detail" not in ill_ids
    assert "film_still" in ill_ids


def test_generate_unknown_preset(client: TestClient):
    r = client.post(
        "/v1/generate",
        json={"modality": "txt2img", "preset": "does-not-exist", "prompt": "hi"},
    )
    assert r.status_code == 404


def test_generate_unknown_effect(client: TestClient):
    r = client.post(
        "/v1/generate",
        json={
            "modality": "txt2img",
            "preset": "sample",
            "prompt": "hi",
            "effects": ["nope"],
        },
    )
    assert r.status_code == 400


def test_img2vid_requires_source(client: TestClient):
    r = client.post(
        "/v1/generate",
        json={"modality": "img2vid", "prompt": "pan left"},
    )
    assert r.status_code == 400


def test_upload_input(client: TestClient, tmp_path):
    # minimal 1x1 PNG
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 64, 96, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
    r = client.post(
        "/v1/inputs",
        files={"file": ("still.png", png, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"]
    assert body["width"] == 64
    assert body["height"] == 96
    assert body["video_width"] % 16 == 0
    assert body["video_height"] % 16 == 0


def test_video_packs_listed(client: TestClient):
    r = client.get("/v1/video/packs")
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()["packs"]}
    assert "sample" in ids

    r = client.get("/v1/capabilities")
    mods = {m["id"]: m for m in r.json()["modalities"]}
    assert mods["txt2vid"]["implemented"] is True
    assert mods["img2vid"]["implemented"] is True

