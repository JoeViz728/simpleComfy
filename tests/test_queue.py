"""Queue, batch, breakpoint, cancel, and job status payload tests."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from simplecomfy.api import create_app
from simplecomfy.config import load_config


@pytest.fixture()
def client(tmp_path):
    cfg = load_config(None)
    cfg.data_dir = tmp_path / "data"
    cfg.ensure_dirs()
    cfg.comfy.root = ""
    cfg.queue.max_depth = 8
    cfg.queue.max_batch = 4
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def test_queue_empty(client: TestClient):
    r = client.get("/v1/queue")
    assert r.status_code == 200
    body = r.json()
    assert body["entries"] == []
    assert body["active_job"] is None
    assert body["paused"] is False
    assert body["busy"] is False


def test_generate_batch_seeds(client: TestClient):
    r = client.post(
        "/v1/generate",
        json={
            "modality": "txt2img",
            "preset": "sample",
            "prompt": "batch test",
            "seed": 100,
            "batch_size": 3,
            # Fail fast without Comfy once dequeued — still tests enqueue
            "auto_pull": False,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["batch_size"] == 3
    jobs = body["jobs"]
    assert len(jobs) == 3
    seeds = [j["seed"] for j in jobs]
    # seed may still be request seed until runner fills meta
    req_seeds = []
    for j in jobs:
        full = client.get(f"/v1/jobs/{j['id']}").json()
        # request seed is exposed via seed field before run
        req_seeds.append(full.get("seed"))
        assert full["phase"] in ("queued", "ensuring_models", "failed", "waiting_comfy", "pulling")
        assert "events" in full
    # Enqueued with seed, seed+1, seed+2
    assert req_seeds == [100, 101, 102]
    # Jobs may already have failed (no Comfy) — list endpoint still has them
    listed = client.get("/v1/jobs").json()["jobs"]
    ids = {j["id"] for j in jobs}
    assert ids.issubset({j["id"] for j in listed})


def test_batch_exceeds_max(client: TestClient):
    r = client.post(
        "/v1/generate",
        json={
            "modality": "txt2img",
            "preset": "sample",
            "prompt": "too many",
            "batch_size": 99,
        },
    )
    # Pydantic le=32 → 422; larger than max_batch after validation → 400
    assert r.status_code in (400, 422)

    r2 = client.post(
        "/v1/generate",
        json={
            "modality": "txt2img",
            "preset": "sample",
            "prompt": "over max_batch",
            "batch_size": 8,  # within pydantic, over cfg max_batch=4
        },
    )
    assert r2.status_code == 400
    assert "max_batch" in r2.json()["detail"]


def test_breakpoint_and_reorder(client: TestClient):
    # Enqueue two jobs then a breakpoint at head via position 0
    r1 = client.post(
        "/v1/generate",
        json={
            "modality": "txt2img",
            "preset": "sample",
            "prompt": "a",
            "auto_pull": False,
        },
    )
    r2 = client.post(
        "/v1/generate",
        json={
            "modality": "txt2img",
            "preset": "sample",
            "prompt": "b",
            "auto_pull": False,
        },
    )
    assert r1.status_code == 200 and r2.status_code == 200
    id_a = r1.json()["id"]
    id_b = r2.json()["id"]

    br = client.post("/v1/queue/breakpoint", json={"position": 0, "label": "Stop"})
    assert br.status_code == 200
    snap = br.json()
    assert snap["created"]["kind"] == "breakpoint"
    break_id = snap["created"]["id"]

    # Give worker a tick — should pause at breakpoint without eating jobs behind it
    time.sleep(0.3)
    q = client.get("/v1/queue").json()
    assert q["paused"] is True
    kinds = [e["kind"] for e in q["entries"]]
    assert kinds[0] == "breakpoint"

    # Move breakpoint to bottom
    r = client.post(
        "/v1/queue/reorder",
        json={"id": break_id, "action": "bottom"},
    )
    assert r.status_code == 200
    time.sleep(0.2)
    q2 = client.get("/v1/queue").json()
    # Either running a job or jobs still queued; breakpoint not necessarily head
    entry_ids = [e["id"] for e in q2["entries"]]
    if break_id in entry_ids:
        assert entry_ids[-1] == break_id or q2.get("active_job")

    # Elevate remaining queued job if present
    queued = [e for e in q2["entries"] if e["kind"] == "job"]
    if len(queued) >= 1:
        target = queued[-1]["id"]
        up = client.post(
            "/v1/queue/reorder",
            json={"id": target, "action": "top"},
        )
        assert up.status_code == 200

    # Remove breakpoint
    rm = client.delete(f"/v1/queue/entries/{break_id}")
    assert rm.status_code == 200

    # Cancel a still-queued job if any
    q3 = client.get("/v1/queue").json()
    for e in q3["entries"]:
        if e["kind"] == "job":
            c = client.post(f"/v1/jobs/{e['id']}/cancel")
            assert c.status_code == 200
            assert c.json()["status"] == "cancelled"
            break

    # Sanity: known ids were created
    assert id_a and id_b


def test_cancel_unknown(client: TestClient):
    r = client.post("/v1/jobs/does-not-exist/cancel")
    assert r.status_code == 404


def test_cancel_queued_and_running_request(client: TestClient):
    # Queued cancel via API
    r = client.post(
        "/v1/generate",
        json={
            "modality": "txt2img",
            "preset": "sample",
            "prompt": "cancel me",
            "auto_pull": False,
            "batch_size": 2,
        },
    )
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    # Put breakpoint at head so second stays queued
    client.post("/v1/queue/breakpoint", json={"position": 0})
    time.sleep(0.25)
    q = client.get("/v1/queue").json()
    queued = [e for e in q["entries"] if e.get("kind") == "job"]
    if queued:
        jid = queued[0]["id"]
        c = client.post(f"/v1/jobs/{jid}/cancel")
        assert c.status_code == 200
        assert c.json()["status"] == "cancelled"

    # Running / pulling: request cancel (may already be failed without Comfy)
    active = client.get("/v1/queue").json().get("active_job")
    if not active:
        # pick any non-terminal from list
        for j in client.get("/v1/jobs").json()["jobs"]:
            if j["status"] in ("pulling", "running"):
                active = j
                break
    if active:
        c2 = client.post(f"/v1/jobs/{active['id']}/cancel")
        assert c2.status_code == 200
        body = c2.json()
        assert body["status"] in ("cancelled", "pulling", "running")
        # cancel_requested should be set while stopping
        if body["status"] != "cancelled":
            assert body.get("meta", {}).get("cancel_requested") or body.get("phase") == "cancelling"
    else:
        # At least cancel of finished job is rejected cleanly
        done = jobs[0]["id"]
        # wait a tick for fail
        time.sleep(0.4)
        c3 = client.post(f"/v1/jobs/{done}/cancel")
        assert c3.status_code in (200, 409)


def test_unload_when_idle(client: TestClient):
    # No Comfy → 502 is acceptable; must not be 409 when idle
    r = client.post("/v1/admin/unload-models")
    assert r.status_code in (200, 502)
    if r.status_code == 409:
        pytest.fail("unload should be allowed when idle")


def test_help_mentions_queue(client: TestClient):
    h = client.get("/v1/help").json()
    assert "queue" in h["endpoints"]
    assert "unload_models" in h["endpoints"]
