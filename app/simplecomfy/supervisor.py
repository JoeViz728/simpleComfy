"""Optional ComfyUI process supervisor (spawn/stop when root is configured)."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from simplecomfy.comfy_client import ComfyClient, write_extra_model_paths
from simplecomfy.config import AppConfig

log = logging.getLogger("simplecomfy.supervisor")


class ComfySupervisor:
    def __init__(self, cfg: AppConfig, client: ComfyClient) -> None:
        self.cfg = cfg
        self.client = client
        self._proc: asyncio.subprocess.Process | None = None
        self._owned = False

    @property
    def root(self) -> Path | None:
        r = (self.cfg.comfy.root or "").strip()
        if not r:
            return None
        return Path(r).expanduser().resolve()

    def _listen_port(self) -> int:
        u = urlparse(self.cfg.comfy.base_url)
        if u.port:
            return u.port
        return 8188

    async def ensure_running(self) -> dict[str, Any]:
        """If Comfy is reachable, ok. Else try spawn when root is set."""
        if await self.client.reachable():
            return {"reachable": True, "spawned": False, "owned": self._owned}

        root = self.root
        if root is None:
            return {
                "reachable": False,
                "spawned": False,
                "hint": "Start ComfyUI manually or set comfy.root in config.toml",
            }

        if not (root / "main.py").is_file():
            return {
                "reachable": False,
                "spawned": False,
                "hint": f"comfy.root has no main.py: {root}",
            }

        try:
            write_extra_model_paths(root, self.cfg.models_dir)
        except Exception as e:
            log.warning("extra_model_paths: %s", e)

        await self._spawn(root)
        # wait for ready
        for _ in range(60):
            if await self.client.reachable():
                return {"reachable": True, "spawned": True, "owned": True}
            await asyncio.sleep(1.0)

        return {
            "reachable": False,
            "spawned": True,
            "owned": True,
            "hint": "ComfyUI process started but not responding yet",
        }

    async def _spawn(self, root: Path) -> None:
        if self._proc and self._proc.returncode is None:
            return

        venv_python = root / ".venv" / "bin" / "python"
        if sys.platform == "win32":
            venv_python = root / ".venv" / "Scripts" / "python.exe"
        python = str(venv_python) if venv_python.is_file() else sys.executable

        port = self._listen_port()
        cmd = [
            python,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        log.info("Starting ComfyUI: %s (cwd=%s)", " ".join(cmd), root)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(root),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=os.environ.copy(),
            start_new_session=True,
        )
        self._owned = True

    async def stop(self) -> None:
        if not self._owned or not self._proc:
            return
        proc = self._proc
        if proc.returncode is not None:
            return
        try:
            if sys.platform != "win32":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=15)
        except asyncio.TimeoutError:
            try:
                if sys.platform != "win32":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
        self._proc = None
        self._owned = False
