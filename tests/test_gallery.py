from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from simplecomfy.api import create_app
from simplecomfy.config import load_config
from simplecomfy.history import GalleryStore, gallery_item_from_dir, write_job_sidecar


def test_gallery_from_sidecar(tmp_path: Path):
    cfg = load_config(None)
    cfg.data_dir = tmp_path / "data"
    cfg.ensure_dirs()
    jid = "11111111-1111-1111-1111-111111111111"
    job_dir = cfg.outputs_dir / jid
    job_dir.mkdir(parents=True)
    (job_dir / "00.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    write_job_sidecar(
        job_dir,
        {
            "id": jid,
            "status": "completed",
            "created_at": "2026-08-13T12:00:00+00:00",
            "modality": "txt2img",
            "preset": "sample",
            "prompt": "test cat",
            "seed": 42,
            "meta": {
                "checkpoint": "sample.safetensors",
                "loras": [{"name": "style.safetensors", "weight": 0.6}],
                "positive": "masterpiece, test cat",
                "width": 832,
                "height": 1216,
            },
            "outputs": [
                {
                    "filename": "00.png",
                    "kind": "image",
                    "url": f"/v1/outputs/{jid}/00.png",
                }
            ],
        },
    )
    item = gallery_item_from_dir(job_dir)
    assert item is not None
    assert item["preset"] == "sample"
    assert item["seed"] == 42
    assert item["checkpoint"] == "sample.safetensors"
    assert item["thumb_url"].endswith("00.png")

    store = GalleryStore(cfg)
    listed = store.list_items(has_image=True)
    assert listed["total"] == 1
    assert listed["items"][0]["prompt"] == "test cat"


def test_gallery_api(tmp_path: Path):
    cfg = load_config(None)
    cfg.data_dir = tmp_path / "data"
    cfg.ensure_dirs()
    cfg.comfy.root = ""
    jid = "22222222-2222-2222-2222-222222222222"
    job_dir = cfg.outputs_dir / jid
    job_dir.mkdir(parents=True)
    (job_dir / "00.png").write_bytes(b"fakeimg")
    write_job_sidecar(
        job_dir,
        {
            "id": jid,
            "status": "completed",
            "created_at": "2026-08-13T13:00:00+00:00",
            "modality": "txt2img",
            "preset": "photo",
            "prompt": "balcony",
            "seed": 7,
            "meta": {"checkpoint": "photo.safetensors"},
            "outputs": [{"filename": "00.png", "kind": "image"}],
        },
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/gallery")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1
        ids = {i["id"] for i in body["items"]}
        assert jid in ids

        r = client.get(f"/v1/gallery/{jid}")
        assert r.status_code == 200
        assert r.json()["prompt"] == "balcony"

        r = client.get("/v1/capabilities")
        assert r.status_code == 200
        mods = {m["id"]: m for m in r.json()["modalities"]}
        assert mods["txt2img"]["implemented"] is True
        assert mods["img2vid"]["implemented"] is True
        assert mods["txt2vid"]["implemented"] is True


def test_legacy_folder_without_sidecar(tmp_path: Path):
    cfg = load_config(None)
    cfg.data_dir = tmp_path / "data"
    cfg.ensure_dirs()
    jid = "33333333-3333-3333-3333-333333333333"
    job_dir = cfg.outputs_dir / jid
    job_dir.mkdir(parents=True)
    (job_dir / "00.png").write_bytes(b"x")
    item = gallery_item_from_dir(job_dir)
    assert item is not None
    assert item["has_sidecar"] is False
    assert item["outputs"][0]["filename"] == "00.png"
