"""Load config.toml + environment overrides."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def repo_root() -> Path:
    """simpleComfy repo root (parent of app/)."""
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return repo_root() / "data"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8790
    api_key: str = ""


@dataclass
class ComfyConfig:
    base_url: str = "http://127.0.0.1:8188"
    root: str = ""
    # Image jobs; video often needs longer (first WAN run especially)
    timeout_s: float = 600.0
    video_timeout_s: float = 1800.0
    # ComfyUI input/ for LoadImage (img2vid). Empty = <comfy.root>/input
    input_dir: str = ""


@dataclass
class QueueConfig:
    max_in_flight: int = 1
    max_depth: int = 16
    # Max jobs created by one generate request (batch_size)
    max_batch: int = 8


@dataclass
class ModelsConfig:
    auto_pull: bool = True
    max_concurrent_downloads: int = 2
    # WAN 14B fp8 weights are ~14–16GB each
    max_pull_gb: float = 40.0
    # Default video pack id under video/
    video_pack: str = "sample"


@dataclass
class CivitaiConfig:
    base_url: str = "https://civitai.red"
    api_token: str = ""


@dataclass
class HuggingFaceConfig:
    token: str = ""


@dataclass
class GenerateConfig:
    default_width: int = 1024
    default_height: int = 1024
    max_width: int = 2048
    max_height: int = 2048


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    comfy: ComfyConfig = field(default_factory=ComfyConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    civitai: CivitaiConfig = field(default_factory=CivitaiConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    data_dir: Path = field(default_factory=default_data_dir)
    config_path: Path | None = None

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    @property
    def presets_dir(self) -> Path:
        return repo_root() / "presets"

    @property
    def effects_dir(self) -> Path:
        return repo_root() / "effects"

    def ensure_dirs(self) -> None:
        for sub in (
            "checkpoints",
            "loras",
            "vae",
            "embeddings",
            "controlnet",
            "upscale_models",
            "diffusion_models",
            "text_encoders",
        ):
            (self.models_dir / sub).mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def comfy_input_dir(self) -> Path | None:
        raw = (self.comfy.input_dir or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        root = (self.comfy.root or "").strip()
        if root:
            return (Path(root).expanduser() / "input").resolve()
        # common default
        home = Path.home() / "ComfyUI" / "input"
        if home.is_dir():
            return home.resolve()
        return None


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    v = raw.get(name) or {}
    return v if isinstance(v, dict) else {}


def load_config(path: Path | None = None) -> AppConfig:
    """Load config from path, COMFY env, or defaults."""
    cfg_path = path
    if cfg_path is None:
        env = os.environ.get("SIMPLECOMFY_CONFIG", "").strip()
        if env:
            cfg_path = Path(env)
        else:
            candidate = repo_root() / "config.toml"
            cfg_path = candidate if candidate.is_file() else None

    raw: dict[str, Any] = {}
    if cfg_path and cfg_path.is_file():
        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)

    server = _section(raw, "server")
    comfy = _section(raw, "comfy")
    paths = _section(raw, "paths")
    queue = _section(raw, "queue")
    models = _section(raw, "models")
    civitai = _section(raw, "civitai")
    hf = _section(raw, "huggingface")
    gen = _section(raw, "generate")

    data_dir = paths.get("data_dir") or ""
    data_path = Path(data_dir).expanduser() if data_dir else default_data_dir()
    if not data_path.is_absolute():
        data_path = (repo_root() / data_path).resolve()

    api_key = (
        os.environ.get("SIMPLECOMFY_API_KEY", "").strip()
        or str(server.get("api_key") or "")
    )
    civ_token = (
        os.environ.get("CIVITAI_API_TOKEN", "").strip()
        or str(civitai.get("api_token") or "")
    )
    hf_token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
        or str(hf.get("token") or "")
    )
    comfy_url = os.environ.get("COMFY_URL", "").strip() or str(
        comfy.get("base_url") or "http://127.0.0.1:8188"
    )
    comfy_root = os.environ.get("COMFY_ROOT", "").strip() or str(comfy.get("root") or "")
    civ_base = os.environ.get("CIVITAI_BASE_URL", "").strip() or str(
        civitai.get("base_url") or "https://civitai.red"
    )

    host = os.environ.get("SIMPLECOMFY_HOST", "").strip() or str(
        server.get("host") or "127.0.0.1"
    )
    port_s = os.environ.get("SIMPLECOMFY_PORT", "").strip()
    port = int(port_s) if port_s else int(server.get("port") or 8790)

    auto_pull_env = os.environ.get("SIMPLECOMFY_AUTO_PULL", "").strip().lower()
    if auto_pull_env in ("0", "false", "no"):
        auto_pull = False
    elif auto_pull_env in ("1", "true", "yes"):
        auto_pull = True
    else:
        auto_pull = bool(models.get("auto_pull", True))

    cfg = AppConfig(
        server=ServerConfig(host=host, port=port, api_key=api_key),
        comfy=ComfyConfig(
            base_url=comfy_url,
            root=comfy_root,
            timeout_s=float(comfy.get("timeout_s") or 600.0),
            video_timeout_s=float(comfy.get("video_timeout_s") or 1800.0),
            input_dir=str(comfy.get("input_dir") or ""),
        ),
        queue=QueueConfig(
            max_in_flight=int(queue.get("max_in_flight") or 1),
            max_depth=int(queue.get("max_depth") or 16),
            max_batch=int(queue.get("max_batch") or 8),
        ),
        models=ModelsConfig(
            auto_pull=auto_pull,
            max_concurrent_downloads=int(models.get("max_concurrent_downloads") or 2),
            max_pull_gb=float(models.get("max_pull_gb") or 40.0),
            video_pack=str(models.get("video_pack") or "sample"),
        ),
        civitai=CivitaiConfig(base_url=civ_base, api_token=civ_token),
        huggingface=HuggingFaceConfig(token=hf_token),
        generate=GenerateConfig(
            default_width=int(gen.get("default_width") or 1024),
            default_height=int(gen.get("default_height") or 1024),
            max_width=int(gen.get("max_width") or 2048),
            max_height=int(gen.get("max_height") or 2048),
        ),
        data_dir=data_path,
        config_path=cfg_path,
    )
    cfg.ensure_dirs()
    return cfg
