"""CLI entrypoint: simplecomfy serve | pull | presets | effects | write-comfy-paths."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from simplecomfy.comfy_client import write_extra_model_paths
from simplecomfy.config import load_config
from simplecomfy.effects import EffectRegistry, summarize_effect
from simplecomfy.presets import PresetRegistry
from simplecomfy.puller import ModelPuller, summarize_preset_status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="simplecomfy", description="simpleComfy CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (or set SIMPLECOMFY_CONFIG)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Run the API + UI server")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")

    sub.add_parser("presets", help="List presets and readiness")

    p_fx = sub.add_parser("effects", help="List effects and readiness")
    p_fx.add_argument("--family", default=None, help="Filter by base_family")

    p_pull = sub.add_parser(
        "pull",
        help="Pull preset, effect:id, or video:pack[:modality]",
    )
    p_pull.add_argument(
        "target",
        help="e.g. sample | effect:soft_light | video:sample | video:sample:img2vid",
    )

    p_paths = sub.add_parser(
        "write-comfy-paths",
        help="Write extra_model_paths.yaml into a ComfyUI install",
    )
    p_paths.add_argument(
        "--comfy-root",
        type=Path,
        default=None,
        help="ComfyUI root (folder with main.py). Default: config comfy.root",
    )
    p_paths.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print YAML to stdout only",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "serve":
        if args.host:
            cfg.server.host = args.host
        if args.port:
            cfg.server.port = args.port
        if args.reload:
            import uvicorn

            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            uvicorn.run(
                "simplecomfy.api:create_app",
                factory=True,
                host=cfg.server.host,
                port=cfg.server.port,
                reload=True,
            )
        else:
            from simplecomfy.api import run_server

            run_server(cfg)
        return

    if args.cmd == "presets":
        reg = PresetRegistry(cfg)
        puller = ModelPuller(cfg)
        items = [summarize_preset_status(p, puller) for p in reg.list()]
        print(json.dumps({"presets": items}, indent=2))
        return

    if args.cmd == "effects":
        reg = EffectRegistry(cfg)
        puller = ModelPuller(cfg)
        items = []
        for e in reg.list():
            if args.family and not e.compatible_with(args.family):
                continue
            items.append(summarize_effect(e, puller))
        print(json.dumps({"effects": items}, indent=2))
        return

    if args.cmd == "pull":
        puller = ModelPuller(cfg)
        target = args.target
        if target.startswith("effect:"):
            eid = target.split(":", 1)[1]
            reg = EffectRegistry(cfg)
            try:
                effect = reg.require(eid)
            except KeyError as e:
                print(str(e), file=sys.stderr)
                sys.exit(1)

            async def _run_fx() -> dict:
                return await puller.pull_effect_lora(effect)

            try:
                result = asyncio.run(_run_fx())
            except Exception as e:
                print(f"pull failed: {e}", file=sys.stderr)
                detail = getattr(e, "detail", None)
                if detail:
                    print(json.dumps(detail, indent=2), file=sys.stderr)
                sys.exit(1)
            print(json.dumps(result, indent=2))
            return

        if target.startswith("video:"):
            from simplecomfy.video_pack import VideoPackRegistry

            parts = target.split(":")
            # video:sample or video:sample:img2vid
            pack_id = parts[1] if len(parts) > 1 else "sample"
            mod = parts[2] if len(parts) > 2 else "all"
            vreg = VideoPackRegistry(cfg)
            pack = vreg.get(pack_id)
            if not pack:
                print(f"unknown video pack '{pack_id}'", file=sys.stderr)
                sys.exit(1)
            if mod == "all":
                arts = pack.artifacts_for("txt2vid") + [
                    a
                    for a in pack.artifacts_for("img2vid")
                    if a.filename
                    not in {x.filename for x in pack.artifacts_for("txt2vid")}
                ]
            elif mod in ("txt2vid", "img2vid"):
                arts = pack.artifacts_for(mod)
            else:
                print("modality must be txt2vid|img2vid|all", file=sys.stderr)
                sys.exit(1)

            async def _run_vid() -> dict:
                return await puller.ensure_video_artifacts(arts, auto_pull=True)

            try:
                result = asyncio.run(_run_vid())
            except Exception as e:
                print(f"pull failed: {e}", file=sys.stderr)
                detail = getattr(e, "detail", None)
                if detail:
                    print(json.dumps(detail, indent=2), file=sys.stderr)
                sys.exit(1)
            print(json.dumps({"pack": pack_id, "modality": mod, "files": result}, indent=2))
            return

        reg = PresetRegistry(cfg)
        try:
            preset = reg.require(target)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        async def _run() -> dict:
            return await puller.pull_preset(preset)

        try:
            result = asyncio.run(_run())
        except Exception as e:
            print(f"pull failed: {e}", file=sys.stderr)
            detail = getattr(e, "detail", None)
            if detail:
                print(json.dumps(detail, indent=2), file=sys.stderr)
            sys.exit(1)
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "write-comfy-paths":
        root = args.comfy_root or (Path(cfg.comfy.root) if cfg.comfy.root else None)
        models = cfg.models_dir.resolve()
        yaml_text = f"""# Generated by simpleComfy — shared model tree
simplecomfy:
    base_path: {models}
    checkpoints: checkpoints
    loras: loras
    vae: vae
    controlnet: controlnet
    clip: clip
    embeddings: embeddings
    upscale_models: upscale_models
"""
        if args.print_only or root is None:
            print(yaml_text)
            if root is None and not args.print_only:
                print(
                    "# Pass --comfy-root /path/to/ComfyUI to write this file.",
                    file=sys.stderr,
                )
            return
        dest = write_extra_model_paths(Path(root), models)
        print(f"Wrote {dest}")
        return

    parser.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
