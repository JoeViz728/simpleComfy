"""HTTP client for a running ComfyUI instance."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from simplecomfy.config import AppConfig

log = logging.getLogger("simplecomfy.comfy")

ProgressCB = Callable[[dict[str, Any]], Awaitable[None] | None]


class ComfyError(RuntimeError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


class ComfyClient:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.base_url = cfg.comfy.base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.cfg.comfy.timeout_s, connect=10.0),
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def http(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("ComfyClient not started")
        return self._client

    def _ws_url(self, client_id: str) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc or "127.0.0.1:8188"
        return f"{scheme}://{netloc}/ws?clientId={client_id}"

    async def reachable(self) -> bool:
        try:
            r = await self.http.get("/system_stats")
            return r.status_code == 200
        except Exception:
            try:
                r = await self.http.get("/")
                return r.status_code < 500
            except Exception:
                return False

    async def free_memory(
        self, *, unload_models: bool = True, free_memory: bool = True
    ) -> dict[str, Any]:
        """Ask ComfyUI to unload models / free VRAM (POST /free)."""
        payload = {"unload_models": unload_models, "free_memory": free_memory}
        try:
            r = await self.http.post("/free", json=payload)
        except Exception as e:
            raise ComfyError("ComfyUI free failed", detail={"error": str(e)}) from e
        if r.status_code >= 400:
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text, "status": r.status_code}
            raise ComfyError("ComfyUI free rejected", detail=body)
        try:
            return r.json() if r.content else {"ok": True}
        except Exception:
            return {"ok": True, "status": r.status_code}

    async def interrupt(self) -> dict[str, Any]:
        """Interrupt the currently executing ComfyUI prompt (POST /interrupt)."""
        try:
            r = await self.http.post("/interrupt")
        except Exception as e:
            raise ComfyError("ComfyUI interrupt failed", detail={"error": str(e)}) from e
        if r.status_code >= 400:
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text, "status": r.status_code}
            raise ComfyError("ComfyUI interrupt rejected", detail=body)
        try:
            return r.json() if r.content else {"ok": True}
        except Exception:
            return {"ok": True, "status": r.status_code}

    async def delete_queue_items(self, prompt_ids: list[str]) -> dict[str, Any]:
        """Remove pending items from the ComfyUI queue by prompt id."""
        if not prompt_ids:
            return {"ok": True, "deleted": 0}
        # Comfy accepts {"delete": [prompt_id, ...]}
        payload = {"delete": list(prompt_ids)}
        try:
            r = await self.http.post("/queue", json=payload)
        except Exception as e:
            raise ComfyError("ComfyUI queue delete failed", detail={"error": str(e)}) from e
        if r.status_code >= 400:
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text, "status": r.status_code}
            raise ComfyError("ComfyUI queue delete rejected", detail=body)
        try:
            return r.json() if r.content else {"ok": True}
        except Exception:
            return {"ok": True}

    async def queue_prompt(
        self, workflow: dict[str, Any], *, client_id: str | None = None
    ) -> tuple[str, str]:
        """Queue a workflow. Returns (prompt_id, client_id)."""
        cid = client_id or str(uuid.uuid4())
        # Comfy rejects unknown top-level keys in some versions — strip _meta
        clean = {k: v for k, v in workflow.items() if not k.startswith("_")}
        payload = {"prompt": clean, "client_id": cid}
        r = await self.http.post("/prompt", json=payload)
        if r.status_code >= 400:
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text}
            raise ComfyError("ComfyUI rejected prompt", detail=body)
        data = r.json()
        if "error" in data:
            raise ComfyError("ComfyUI prompt error", detail=data)
        node_errors = data.get("node_errors") or {}
        if node_errors:
            raise ComfyError("ComfyUI node errors", detail=data)
        return data["prompt_id"], cid

    async def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        r = await self.http.get(f"/history/{prompt_id}")
        r.raise_for_status()
        data = r.json()
        return data.get(prompt_id)

    async def wait_history(
        self,
        prompt_id: str,
        *,
        poll_s: float = 0.5,
        timeout_s: float | None = None,
        client_id: str | None = None,
        on_progress: ProgressCB | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """
        Wait until Comfy finishes a prompt.
        Prefers WebSocket progress when client_id is set; falls back to history poll.
        If should_cancel() returns True, raises ComfyError("cancelled").
        """
        timeout = timeout_s if timeout_s is not None else self.cfg.comfy.timeout_s
        if client_id:
            try:
                return await self._wait_history_ws(
                    prompt_id,
                    client_id=client_id,
                    timeout_s=timeout,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                )
            except ComfyError:
                raise
            except Exception as e:
                log.warning("Comfy WS wait failed (%s); falling back to poll", e)
        return await self._wait_history_poll(
            prompt_id,
            poll_s=poll_s,
            timeout_s=timeout,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

    async def _emit(
        self, on_progress: ProgressCB | None, event: dict[str, Any]
    ) -> None:
        if not on_progress:
            return
        result = on_progress(event)
        if asyncio.iscoroutine(result):
            await result

    async def _wait_history_poll(
        self,
        prompt_id: str,
        *,
        poll_s: float = 0.5,
        timeout_s: float,
        on_progress: ProgressCB | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            if should_cancel and should_cancel():
                raise ComfyError(
                    "cancelled",
                    detail={"prompt_id": prompt_id, "reason": "user_cancel"},
                )
            hist = await self.get_history(prompt_id)
            if hist:
                status = hist.get("status") or {}
                if status.get("status_str") == "error" or status.get("completed") is False:
                    msgs = status.get("messages") or []
                    # Interrupt often surfaces as error — treat cancel flag first
                    if should_cancel and should_cancel():
                        raise ComfyError(
                            "cancelled",
                            detail={"prompt_id": prompt_id, "reason": "user_cancel"},
                        )
                    raise ComfyError(
                        "ComfyUI job failed",
                        detail={"prompt_id": prompt_id, "messages": msgs, "status": status},
                    )
                if hist.get("outputs"):
                    return hist
                if status.get("completed"):
                    return hist
            if loop.time() > deadline:
                raise ComfyError(
                    "Timed out waiting for ComfyUI job",
                    detail={"prompt_id": prompt_id},
                )
            await asyncio.sleep(poll_s)

    async def _wait_history_ws(
        self,
        prompt_id: str,
        *,
        client_id: str,
        timeout_s: float,
        on_progress: ProgressCB | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        import websockets
        from websockets.exceptions import ConnectionClosed

        ws_url = self._ws_url(client_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last_node: str | None = None

        async with websockets.connect(
            ws_url, open_timeout=10, close_timeout=5, max_size=8 * 1024 * 1024
        ) as ws:
            while True:
                if should_cancel and should_cancel():
                    raise ComfyError(
                        "cancelled",
                        detail={"prompt_id": prompt_id, "reason": "user_cancel"},
                    )
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise ComfyError(
                        "Timed out waiting for ComfyUI job",
                        detail={"prompt_id": prompt_id},
                    )
                # Periodic history check in case we missed the done event
                hist = await self.get_history(prompt_id)
                if hist:
                    status = hist.get("status") or {}
                    if status.get("status_str") == "error":
                        if should_cancel and should_cancel():
                            raise ComfyError(
                                "cancelled",
                                detail={"prompt_id": prompt_id, "reason": "user_cancel"},
                            )
                        raise ComfyError(
                            "ComfyUI job failed",
                            detail={
                                "prompt_id": prompt_id,
                                "messages": status.get("messages") or [],
                                "status": status,
                            },
                        )
                    if hist.get("outputs") or status.get("completed"):
                        return hist

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(1.0, remaining))
                except asyncio.TimeoutError:
                    continue
                except ConnectionClosed:
                    # Fall through to final history check
                    break

                if isinstance(raw, bytes):
                    continue  # preview binary frames
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                mtype = msg.get("type")
                data = msg.get("data") or {}
                # Only react to our prompt when sid is present
                sid = data.get("prompt_id") or data.get("sid")
                if sid and sid != prompt_id and mtype in (
                    "progress",
                    "executing",
                    "executed",
                    "execution_error",
                    "execution_success",
                    "execution_interrupted",
                ):
                    # Some builds omit prompt_id on progress — still accept if none
                    if data.get("prompt_id") and data.get("prompt_id") != prompt_id:
                        continue

                if mtype == "progress":
                    value = data.get("value")
                    maximum = data.get("max")
                    try:
                        cur = int(value) if value is not None else 0
                        tot = int(maximum) if maximum is not None else None
                    except (TypeError, ValueError):
                        cur, tot = 0, None
                    await self._emit(
                        on_progress,
                        {
                            "type": "comfy_progress",
                            "phase": "sampling",
                            "current": cur,
                            "total": tot,
                            "unit": "step",
                            "node": last_node,
                            "message": (
                                f"step {cur}/{tot}" if tot else f"step {cur}"
                            ),
                        },
                    )
                elif mtype == "executing":
                    node = data.get("node")
                    if node is None and (
                        not data.get("prompt_id") or data.get("prompt_id") == prompt_id
                    ):
                        # node=None means execution finished for this client
                        hist = await self.get_history(prompt_id)
                        if hist:
                            return hist
                    else:
                        last_node = str(node) if node is not None else last_node
                        await self._emit(
                            on_progress,
                            {
                                "type": "comfy_executing",
                                "phase": "sampling",
                                "node": last_node,
                                "message": f"node {last_node}" if last_node else "executing",
                            },
                        )
                elif mtype == "execution_error":
                    if data.get("prompt_id") and data.get("prompt_id") != prompt_id:
                        continue
                    raise ComfyError(
                        "ComfyUI execution error",
                        detail={"prompt_id": prompt_id, "error": data},
                    )
                elif mtype in ("execution_success", "executed"):
                    if data.get("prompt_id") and data.get("prompt_id") != prompt_id:
                        continue
                    if mtype == "execution_success" or data.get("node") is None:
                        hist = await self.get_history(prompt_id)
                        if hist:
                            return hist

        hist = await self.get_history(prompt_id)
        if hist and (hist.get("outputs") or (hist.get("status") or {}).get("completed")):
            return hist
        raise ComfyError(
            "ComfyUI WebSocket closed before job finished",
            detail={"prompt_id": prompt_id},
        )

    async def download_image(
        self,
        *,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = await self.http.get("/view", params=params)
        r.raise_for_status()
        return r.content

    async def list_checkpoints(self) -> list[str]:
        """Checkpoint basenames ComfyUI currently knows about."""
        try:
            r = await self.http.get("/object_info/CheckpointLoaderSimple")
            r.raise_for_status()
            data = r.json()
        except Exception:
            r = await self.http.get("/object_info")
            r.raise_for_status()
            data = r.json()
        node = data.get("CheckpointLoaderSimple") or {}
        required = (node.get("input") or {}).get("required") or {}
        ckpt = required.get("ckpt_name")
        if isinstance(ckpt, list) and ckpt:
            names = ckpt[0]
            if isinstance(names, list):
                return [str(n) for n in names]
        return []

    async def list_loras(self) -> list[str]:
        try:
            r = await self.http.get("/object_info/LoraLoader")
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []
        node = data.get("LoraLoader") or {}
        required = (node.get("input") or {}).get("required") or {}
        lora = required.get("lora_name")
        if isinstance(lora, list) and lora and isinstance(lora[0], list):
            return [str(n) for n in lora[0]]
        return []

    def extract_media(self, history: dict[str, Any]) -> list[dict[str, str]]:
        """Collect image/video outputs from a Comfy history entry."""
        out: list[dict[str, str]] = []
        outputs = history.get("outputs") or {}
        for _node, node_out in outputs.items():
            if not isinstance(node_out, dict):
                continue
            for key, kind in (
                ("images", "image"),
                ("gifs", "video"),
                ("videos", "video"),
            ):
                for item in node_out.get(key) or []:
                    if not isinstance(item, dict):
                        continue
                    fn = item.get("filename") or ""
                    if not fn:
                        continue
                    out.append(
                        {
                            "filename": fn,
                            "subfolder": item.get("subfolder", "") or "",
                            "type": item.get("type", "output") or "output",
                            "kind": kind,
                        }
                    )
            # Some builds nest under ui
            ui = node_out.get("ui") or {}
            if isinstance(ui, dict):
                for key, kind in (
                    ("images", "image"),
                    ("gifs", "video"),
                    ("videos", "video"),
                ):
                    for item in ui.get(key) or []:
                        if not isinstance(item, dict):
                            continue
                        fn = item.get("filename") or ""
                        if not fn:
                            continue
                        out.append(
                            {
                                "filename": fn,
                                "subfolder": item.get("subfolder", "") or "",
                                "type": item.get("type", "output") or "output",
                                "kind": kind,
                            }
                        )
        # de-dupe
        seen: set[tuple[str, str, str]] = set()
        uniq: list[dict[str, str]] = []
        for m in out:
            k = (m["filename"], m.get("subfolder") or "", m.get("type") or "")
            if k in seen:
                continue
            seen.add(k)
            uniq.append(m)
        return uniq

    def extract_images(self, history: dict[str, Any]) -> list[dict[str, str]]:
        return [m for m in self.extract_media(history) if m.get("kind") == "image"]


def write_extra_model_paths(
    comfy_root: Path,
    models_dir: Path,
    *,
    extra_model_dirs: list[Path] | None = None,
) -> Path:
    """Write extra_model_paths.yaml so ComfyUI sees our model tree (+ optional extras)."""
    comfy_root = comfy_root.expanduser().resolve()
    models_dir = models_dir.expanduser().resolve()
    if not (comfy_root / "main.py").is_file():
        raise FileNotFoundError(f"Not a ComfyUI root (no main.py): {comfy_root}")

    def block(name: str, base: Path) -> str:
        return f"""{name}:
    base_path: {base}
    checkpoints: checkpoints
    loras: loras
    vae: vae
    controlnet: controlnet
    clip: clip
    text_encoders: text_encoders
    diffusion_models: diffusion_models
    embeddings: embeddings
    upscale_models: upscale_models
"""

    parts = [
        "# Generated by simpleComfy — restart ComfyUI after changes",
        block("simplecomfy", models_dir),
    ]
    for i, extra in enumerate(extra_model_dirs or []):
        extra = extra.expanduser().resolve()
        if extra == models_dir or not extra.is_dir():
            continue
        parts.append(block(f"extra_{i}", extra))

    dest = comfy_root / "extra_model_paths.yaml"
    dest.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return dest
