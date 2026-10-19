"""Download pinned model sources (civitai:VERSION, hf:repo:file, https://)."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from simplecomfy.config import AppConfig
from simplecomfy.presets import Preset

ProgressCB = Callable[[dict[str, Any]], Awaitable[None]]

_HF_RE = re.compile(r"^hf:(?P<repo>[^:]+):(?P<file>.+)$")
_CIV_RE = re.compile(r"^civitai:(?P<version>\d+)(?::(?P<file>.+))?$")


class PullError(RuntimeError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


class ModelPuller:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.models.max_concurrent_downloads)
        self._manifest_path = cfg.state_dir / "pulled.json"
        self._manifest: dict[str, str] = self._load_manifest()

    def _load_manifest(self) -> dict[str, str]:
        if not self._manifest_path.is_file():
            return {}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_manifest(self) -> None:
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _remember(self, source: str, path: Path) -> None:
        if source:
            self._manifest[source] = path.name
            self._save_manifest()

    def resolved_name(self, *, role: str, filename: str, source: str) -> str | None:
        if filename and self.find(role, filename):
            return Path(filename).name
        if source and source in self._manifest:
            name = self._manifest[source]
            if self.find(role, name):
                return name
        return None

    def _role_dir(self, role: str) -> Path:
        sub = {
            "checkpoint": "checkpoints",
            "lora": "loras",
            "vae": "vae",
            "embedding": "embeddings",
            "unet": "diffusion_models",
            "diffusion_model": "diffusion_models",
            "text_encoder": "text_encoders",
            "clip": "text_encoders",
        }.get(role, role)
        d = self.cfg.models_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _usable(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def find(self, role: str, name: str) -> Path | None:
        if not name:
            return None
        direct = self._role_dir(role) / Path(name).name
        if self._usable(direct):
            return direct
        base = Path(name).name
        for p in self.cfg.models_dir.rglob(base):
            if self._usable(p):
                return p
        return None

    def missing_for_preset(self, preset: Preset) -> list[dict[str, str]]:
        missing: list[dict[str, str]] = []
        ckpt = self.resolved_name(
            role="checkpoint",
            filename=preset.checkpoint_filename,
            source=preset.checkpoint_source,
        )
        if not ckpt:
            missing.append(
                {
                    "role": "checkpoint",
                    "name": preset.checkpoint_filename or "(unresolved)",
                    "source": preset.checkpoint_source,
                }
            )
        for lora in preset.loras:
            if not lora.source and not lora.filename:
                continue
            name = self.resolved_name(
                role="lora", filename=lora.filename, source=lora.source
            )
            if not name:
                missing.append(
                    {
                        "role": "lora",
                        "name": lora.filename or "(unresolved)",
                        "source": lora.source,
                    }
                )
        return missing

    def preset_ready(self, preset: Preset) -> bool:
        return len(self.missing_for_preset(preset)) == 0

    async def ensure_preset(
        self,
        preset: Preset,
        *,
        auto_pull: bool,
        on_progress: ProgressCB | None = None,
    ) -> tuple[str, list[tuple[str, float]]]:
        """
        Ensure checkpoint + loras on disk. Returns (ckpt_basename, [(lora_basename, weight)]).
        """
        if not preset.checkpoint_source and not preset.checkpoint_filename:
            raise PullError(
                "preset_has_no_checkpoint",
                detail={"preset": preset.id, "hint": "Set checkpoint_source in the preset YAML."},
            )

        # Prefer stable preset filename so Comfy graph names stay predictable.
        ckpt_name = preset.checkpoint_filename.strip() or ""
        ckpt_path = await self.ensure_artifact(
            role="checkpoint",
            name=ckpt_name,
            source=preset.checkpoint_source,
            auto_pull=auto_pull,
            on_progress=on_progress,
        )
        if ckpt_name and ckpt_path.name != Path(ckpt_name).name:
            # Ensure stable alias exists next to the downloaded file
            alias = self._dest("checkpoint", ckpt_name)
            if not self._usable(alias):
                self._materialize(ckpt_path, alias)
            ckpt_path = alias if self._usable(alias) else ckpt_path
            self._remember(preset.checkpoint_source, ckpt_path)

        lora_files: list[tuple[str, float]] = []
        for lora in preset.loras:
            if not lora.source and not lora.filename:
                continue
            want = lora.filename.strip() or ""
            lp = await self.ensure_artifact(
                role="lora",
                name=want,
                source=lora.source,
                auto_pull=auto_pull,
                on_progress=on_progress,
            )
            if want and lp.name != Path(want).name:
                alias = self._dest("lora", want)
                if not self._usable(alias):
                    self._materialize(lp, alias)
                lp = alias if self._usable(alias) else lp
                self._remember(lora.source, lp)
            lora_files.append((lp.name, lora.weight))
        return ckpt_path.name, lora_files

    async def pull_preset(
        self, preset: Preset, *, on_progress: ProgressCB | None = None
    ) -> dict[str, Any]:
        """Deliberate pull of a preset recipe. Always downloads missing pieces."""
        ckpt, loras = await self.ensure_preset(preset, auto_pull=True, on_progress=on_progress)
        return {
            "preset": preset.id,
            "checkpoint": ckpt,
            "loras": [{"name": n, "weight": w} for n, w in loras],
            "ready": True,
        }

    async def ensure_lora_specs(
        self,
        specs: list,
        *,
        auto_pull: bool,
        on_progress: ProgressCB | None = None,
    ) -> list[tuple[str, float]]:
        """Materialize a list of LoraSpec-like objects → [(basename, weight)]."""
        out: list[tuple[str, float]] = []
        for lora in specs:
            source = getattr(lora, "source", "") or ""
            filename = getattr(lora, "filename", "") or ""
            weight = float(getattr(lora, "weight", 0.7))
            if not source and not filename:
                continue
            want = filename.strip() or ""
            lp = await self.ensure_artifact(
                role="lora",
                name=want,
                source=source or None,
                auto_pull=auto_pull,
                on_progress=on_progress,
            )
            if want and lp.name != Path(want).name:
                alias = self._dest("lora", want)
                if not self._usable(alias):
                    self._materialize(lp, alias)
                lp = alias if self._usable(alias) else lp
                if source:
                    self._remember(source, lp)
            out.append((lp.name, weight))
        return out

    def missing_for_lora_spec(self, *, filename: str, source: str) -> bool:
        return (
            self.resolved_name(role="lora", filename=filename or "", source=source or "")
            is None
        )

    async def pull_effect_lora(
        self, effect: Any, *, on_progress: ProgressCB | None = None
    ) -> dict[str, Any]:
        """Pull all LoRA files for an effect (image single-file or video high/low)."""
        specs = []
        if hasattr(effect, "as_lora_specs"):
            specs = list(effect.as_lora_specs() or [])
        elif hasattr(effect, "as_lora_spec"):
            one = effect.as_lora_spec()
            if one:
                specs = [one]
        if not specs:
            return {"effect": effect.id, "ready": True, "lora": None, "loras": []}
        files = await self.ensure_lora_specs(
            specs, auto_pull=True, on_progress=on_progress
        )
        names = [n for n, _ in files]
        return {
            "effect": effect.id,
            "ready": True,
            "lora": names[0] if names else None,
            "loras": names,
        }

    async def ensure_artifact(
        self,
        *,
        role: str,
        name: str,
        source: str | None,
        auto_pull: bool,
        on_progress: ProgressCB | None = None,
    ) -> Path:
        remembered = self.resolved_name(role=role, filename=name or "", source=source or "")
        if remembered:
            found = self.find(role, remembered)
            if found:
                return found

        if name:
            existing = self.find(role, name)
            if existing:
                if source:
                    self._remember(source, existing)
                return existing

        if not source:
            raise PullError(
                "model_missing",
                detail={
                    "role": role,
                    "name": name or None,
                    "hint": "Set source on the preset or place the file under data/models.",
                },
            )

        if not auto_pull:
            raise PullError(
                "model_missing",
                detail={
                    "role": role,
                    "name": name or None,
                    "source": source,
                    "hint": "POST /v1/presets/{id}/pull or enable models.auto_pull",
                },
            )

        async with self._sem:
            if name:
                existing = self.find(role, name)
                if existing:
                    self._remember(source, existing)
                    return existing
            remembered = self.resolved_name(role=role, filename=name, source=source)
            if remembered:
                found = self.find(role, remembered)
                if found:
                    return found
            path = await self._pull(
                role=role, name=name, source=source, on_progress=on_progress
            )
            self._remember(source, path)
            return path

    async def _pull(
        self,
        *,
        role: str,
        name: str,
        source: str,
        on_progress: ProgressCB | None,
    ) -> Path:
        if on_progress:
            await on_progress(
                {"type": "pulling", "role": role, "name": name, "source": source, "phase": "start"}
            )

        hf = _HF_RE.match(source)
        civ = _CIV_RE.match(source)
        if hf:
            path = await self._pull_hf(
                role=role,
                name=name,
                repo=hf.group("repo"),
                filename=hf.group("file"),
                on_progress=on_progress,
            )
        elif civ:
            path = await self._pull_civitai(
                role=role,
                name=name,
                version_id=civ.group("version"),
                preferred_name=civ.group("file") or name,
                on_progress=on_progress,
            )
        elif source.startswith("http://") or source.startswith("https://"):
            path = await self._pull_url(
                role=role,
                name=name or Path(source).name,
                url=source,
                on_progress=on_progress,
            )
        else:
            raise PullError(
                "unsupported_source",
                detail={
                    "source": source,
                    "hint": "Use civitai:VERSION, hf:repo:file, or https:// URL",
                },
            )

        if on_progress:
            await on_progress(
                {
                    "type": "pulling",
                    "role": role,
                    "name": path.name,
                    "source": source,
                    "phase": "done",
                    "path": str(path),
                }
            )
        return path

    def _dest(self, role: str, filename: str) -> Path:
        dest = self._role_dir(role) / Path(filename).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    async def ensure_video_artifacts(
        self,
        artifacts: list[Any],
        *,
        auto_pull: bool,
        on_progress: ProgressCB | None = None,
    ) -> dict[str, str]:
        """
        Ensure a list of VideoArtifact-like objects.
        Returns map filename -> resolved basename (usually same).
        """
        resolved: dict[str, str] = {}
        for art in artifacts:
            role = getattr(art, "role", "unet")
            filename = getattr(art, "filename", "") or ""
            source = getattr(art, "source", "") or ""
            path = await self.ensure_artifact(
                role=role,
                name=filename,
                source=source or None,
                auto_pull=auto_pull,
                on_progress=on_progress,
            )
            # stable alias if needed
            if filename and path.name != Path(filename).name:
                alias = self._dest(role, filename)
                if not self._usable(alias):
                    self._materialize(path, alias)
                path = alias if self._usable(alias) else path
                if source:
                    self._remember(source, path)
            resolved[filename or path.name] = path.name
        return resolved

    def missing_video_artifacts(self, artifacts: list[Any]) -> list[dict[str, str]]:
        missing: list[dict[str, str]] = []
        for art in artifacts:
            role = getattr(art, "role", "unet")
            filename = getattr(art, "filename", "") or ""
            source = getattr(art, "source", "") or ""
            name = self.resolved_name(role=role, filename=filename, source=source)
            if not name:
                missing.append(
                    {
                        "role": role,
                        "name": filename or "(unresolved)",
                        "source": source,
                    }
                )
        return missing

    @staticmethod
    def _materialize(src: Path, dest: Path) -> None:
        src = src.resolve()
        if not src.is_file():
            raise FileNotFoundError(f"source missing: {src}")
        if dest.is_symlink() or dest.exists():
            if dest.is_symlink() or not dest.is_file() or dest.stat().st_size == 0:
                dest.unlink(missing_ok=True)
            elif dest.resolve() == src:
                return
            elif dest.is_file() and dest.stat().st_size > 0:
                return
            else:
                dest.unlink(missing_ok=True)
        try:
            dest.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dest)

    def _rewrite_civitai_url(self, url: str, base: str) -> str:
        # Keep downloads on configured host when possible
        if "civitai.com" in url and "civitai.red" in base:
            return url.replace("civitai.com", "civitai.red")
        if "civitai.red" in url and "civitai.com" in base:
            return url.replace("civitai.red", "civitai.com")
        return url

    async def _pull_hf(
        self,
        *,
        role: str,
        name: str,
        repo: str,
        filename: str,
        on_progress: ProgressCB | None,
    ) -> Path:
        out_name = Path(name).name if name else Path(filename).name
        dest = self._dest(role, out_name)

        def _download() -> Path:
            from huggingface_hub import hf_hub_download

            token = self.cfg.huggingface.token or None
            cache = str(self.cfg.data_dir / "hf-cache")
            Path(cache).mkdir(parents=True, exist_ok=True)
            local = hf_hub_download(
                repo_id=repo,
                filename=filename,
                token=token,
                cache_dir=cache,
            )
            src = Path(local).resolve()
            self._materialize(src, dest)
            return dest

        if on_progress:
            await on_progress(
                {
                    "type": "pulling",
                    "role": role,
                    "source": f"hf:{repo}:{filename}",
                    "phase": "download",
                }
            )
        return await asyncio.to_thread(_download)

    async def _pull_civitai(
        self,
        *,
        role: str,
        name: str,
        version_id: str,
        preferred_name: str,
        on_progress: ProgressCB | None,
    ) -> Path:
        base = self.cfg.civitai.base_url.rstrip("/")
        token = self.cfg.civitai.api_token
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            meta = await client.get(
                f"{base}/api/v1/model-versions/{version_id}", headers=headers
            )
            if meta.status_code >= 400:
                # fallback host
                alt = (
                    "https://civitai.com"
                    if "civitai.red" in base
                    else "https://civitai.red"
                )
                meta = await client.get(
                    f"{alt}/api/v1/model-versions/{version_id}", headers=headers
                )
                if meta.status_code < 400:
                    base = alt
            if meta.status_code >= 400:
                raise PullError(
                    "pull_failed",
                    detail={
                        "source": f"civitai:{version_id}",
                        "status": meta.status_code,
                        "body": meta.text[:400],
                    },
                )
            data = meta.json()
            files = data.get("files") or []
            chosen = None
            pref = Path(preferred_name).name if preferred_name else ""
            for f in files:
                if pref and f.get("name") == pref:
                    chosen = f
                    break
            if not chosen:
                safes = [
                    f
                    for f in files
                    if str(f.get("name", "")).endswith(".safetensors")
                ]
                chosen = safes[0] if safes else (files[0] if files else None)
            if not chosen:
                raise PullError(
                    "pull_failed",
                    detail={"detail": "No files on model version", "version": version_id},
                )

            remote_name = str(chosen.get("name") or f"model_{version_id}.safetensors")
            # Always prefer caller-requested stable filename when provided
            if pref and (
                pref.endswith(".safetensors")
                or pref.endswith(".ckpt")
                or pref.endswith(".gguf")
            ):
                out_name = Path(pref).name
            elif name and (
                str(name).endswith(".safetensors")
                or str(name).endswith(".ckpt")
                or str(name).endswith(".gguf")
            ):
                out_name = Path(name).name
            else:
                out_name = remote_name

            size = int(chosen.get("sizeKB", 0) * 1024) if chosen.get("sizeKB") else 0
            if self.cfg.models.max_pull_gb and size:
                limit = self.cfg.models.max_pull_gb * (1024**3)
                if size > limit:
                    raise PullError(
                        "pull_too_large",
                        detail={
                            "bytes": size,
                            "max_gb": self.cfg.models.max_pull_gb,
                            "name": out_name,
                        },
                    )

            url = chosen.get("downloadUrl") or f"{base}/api/download/models/{version_id}"
            url = self._rewrite_civitai_url(str(url), base)

            dest = self._dest(role, out_name)
            if self._usable(dest):
                return dest

            if on_progress:
                await on_progress(
                    {
                        "type": "pulling",
                        "role": role,
                        "name": out_name,
                        "source": f"civitai:{version_id}",
                        "phase": "download",
                        "bytes_total": size or None,
                    }
                )

            # stream download
            tmp = dest.with_suffix(dest.suffix + ".part")
            params = {}
            if token and "token=" not in url:
                params["token"] = token
            async with client.stream(
                "GET", url, headers=headers, params=params or None
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread())[:400]
                    raise PullError(
                        "pull_failed",
                        detail={
                            "status": resp.status_code,
                            "body": body.decode("utf-8", errors="replace"),
                            "url": url.split("?")[0],
                        },
                    )
                downloaded = 0
                with tmp.open("wb") as f:
                    async for chunk in resp.aiter_bytes(1024 * 256):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if on_progress and size and downloaded % (8 * 1024 * 1024) < 256 * 1024:
                            await on_progress(
                                {
                                    "type": "pulling",
                                    "role": role,
                                    "name": out_name,
                                    "phase": "download",
                                    "bytes_done": downloaded,
                                    "bytes_total": size,
                                }
                            )
            tmp.replace(dest)
            return dest

    async def _pull_url(
        self,
        *,
        role: str,
        name: str,
        url: str,
        on_progress: ProgressCB | None,
    ) -> Path:
        dest = self._dest(role, name or Path(url).name)
        if self._usable(dest):
            return dest
        if on_progress:
            await on_progress(
                {"type": "pulling", "role": role, "name": dest.name, "phase": "download", "source": url}
            )
        tmp = dest.with_suffix(dest.suffix + ".part")
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as f:
                    async for chunk in resp.aiter_bytes(1024 * 256):
                        f.write(chunk)
        tmp.replace(dest)
        return dest


def summarize_preset_status(preset: Preset, puller: ModelPuller) -> dict[str, Any]:
    missing = puller.missing_for_preset(preset)
    return {
        "id": preset.id,
        "label": preset.label,
        "description": preset.description,
        "base_family": preset.base_family,
        "ready": len(missing) == 0,
        "missing": missing,
        "size_gb_approx": preset.size_gb_approx,
        "checkpoint_source": preset.checkpoint_source,
        "default_effects": list(preset.default_effects),
        "suggested_effects": list(preset.suggested_effects),
        "default_size": {"width": preset.size.width, "height": preset.size.height},
        "sampler": {
            "name": preset.sampler.name,
            "scheduler": preset.sampler.scheduler,
            "steps": preset.sampler.steps,
            "cfg": preset.sampler.cfg,
        },
    }
