"""Local video post-process helpers (smooth / last-frame / concat) via bundled ffmpeg."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("simplecomfy.video_ops")


class VideoOpsError(RuntimeError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


def ffmpeg_exe() -> str:
    """Resolve ffmpeg: PATH first, else imageio-ffmpeg static binary."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise VideoOpsError(
            "ffmpeg not available — install system ffmpeg or pip install imageio-ffmpeg",
            detail={"error": str(e)},
        ) from e


def _run(cmd: list[str], *, timeout_s: float = 600.0) -> None:
    log.info("video_ops: %s", " ".join(cmd[:8]) + ("…" if len(cmd) > 8 else ""))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise VideoOpsError("ffmpeg timed out", detail={"cmd": cmd[:6]}) from e
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise VideoOpsError(
            "ffmpeg failed",
            detail={
                "returncode": proc.returncode,
                "stderr_tail": err[-2000:] if err else "",
            },
        )


def _ffprobe_bin() -> str | None:
    probe = shutil.which("ffprobe")
    if probe:
        return probe
    try:
        ff = ffmpeg_exe()
    except VideoOpsError:
        return None
    sibling = Path(ff).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    return None


def _ffprobe_show(path: Path, entries: str, *, select_streams: str | None = "v:0") -> str:
    probe = _ffprobe_bin()
    if not probe:
        return ""
    cmd = [probe, "-v", "error"]
    if select_streams:
        cmd.extend(["-select_streams", select_streams])
    cmd.extend(
        [
            "-show_entries",
            entries,
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except Exception:
        return ""
    return (proc.stdout or "").strip()


def probe_fps(path: Path) -> float | None:
    """Best-effort FPS from ffprobe or ffmpeg -i."""
    raw = _ffprobe_show(path, "stream=r_frame_rate")
    if raw:
        try:
            line = raw.splitlines()[0].strip()
            if "/" in line:
                a, b = line.split("/", 1)
                num, den = float(a), float(b)
                if den:
                    return num / den
            return float(line)
        except Exception:
            log.debug("ffprobe fps parse failed: %r", raw, exc_info=True)

    try:
        ff = ffmpeg_exe()
        proc = subprocess.run(
            [ff, "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        blob = (proc.stderr or "") + "\n" + (proc.stdout or "")
        m = re.search(r"(\d+(?:\.\d+)?)\s*fps", blob)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*tbr", blob)
        if m:
            return float(m.group(1))
    except Exception:
        log.debug("ffmpeg -i fps fallback failed", exc_info=True)
    return None


def probe_duration(path: Path) -> float | None:
    """Duration in seconds (format or stream; ffmpeg -i fallback)."""
    raw = _ffprobe_show(path, "format=duration", select_streams=None)
    if not raw:
        raw = _ffprobe_show(path, "stream=duration")
    if raw:
        try:
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.upper() == "N/A":
                    continue
                d = float(line)
                if d > 0:
                    return d
        except Exception:
            log.debug("ffprobe duration parse failed: %r", raw, exc_info=True)

    # imageio-ffmpeg ships ffmpeg only — parse `Duration: HH:MM:SS.xx` from -i
    try:
        ff = ffmpeg_exe()
        proc = subprocess.run(
            [ff, "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        blob = (proc.stderr or "") + "\n" + (proc.stdout or "")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", blob)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            d = h * 3600 + mi * 60 + s
            if d > 0:
                return d
    except Exception:
        log.debug("ffmpeg -i duration fallback failed", exc_info=True)
    return None


def probe_nb_frames(path: Path) -> int | None:
    """Frame count when container metadata has it."""
    raw = _ffprobe_show(path, "stream=nb_frames")
    if not raw:
        return None
    try:
        for line in raw.splitlines():
            line = line.strip()
            if line.isdigit():
                n = int(line)
                if n > 0:
                    return n
    except Exception:
        pass
    # fallback: duration * fps
    dur = probe_duration(path)
    fps = probe_fps(path)
    if dur and fps:
        return max(1, int(round(dur * fps)))
    return None


def resolve_overlap_seconds(
    *,
    overlap_seconds: float | None,
    overlap_frames: int | None,
    fps: float,
    parent_duration_s: float,
    chunk_duration_s: float,
    max_fraction: float = 0.4,
) -> float:
    """
    Clamp user overlap so xfade stays valid.
    max_fraction caps how much of either clip the blend may consume
    (important as parent videos grow long).
    """
    fps = fps if fps and fps > 0 else 16.0
    if overlap_frames is not None and overlap_frames > 0:
        want = float(overlap_frames) / fps
    elif overlap_seconds is not None:
        want = float(overlap_seconds)
    else:
        want = 0.75  # default ~12f @ 16fps

    if want <= 0:
        return 0.0

    # Never eat more than max_fraction of either side; leave ≥2 frames solid
    min_solid = 2.0 / fps
    cap_a = max(0.0, parent_duration_s * max_fraction)
    cap_b = max(0.0, chunk_duration_s * max_fraction)
    hard_cap_a = max(0.0, parent_duration_s - min_solid)
    hard_cap_b = max(0.0, chunk_duration_s - min_solid)
    overlap = min(want, cap_a, cap_b, hard_cap_a, hard_cap_b)
    # Need at least ~1 frame of blend to matter
    if overlap < (0.5 / fps):
        return 0.0
    return float(overlap)


def extract_last_frame(video_path: Path, dest_png: Path) -> Path:
    """Write the last decoded frame of a video as PNG."""
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    # sseof seeks from end; -update 1 writes a single image
    _run(
        [
            ff,
            "-y",
            "-sseof",
            "-0.15",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(dest_png),
        ],
        timeout_s=120.0,
    )
    if not dest_png.is_file() or dest_png.stat().st_size < 32:
        raise VideoOpsError("failed to extract last frame", detail={"dest": str(dest_png)})
    return dest_png


def extract_first_frame(video_path: Path, dest_png: Path) -> Path:
    """Write the first decoded frame of a video as PNG."""
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    _run(
        [
            ff,
            "-y",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(dest_png),
        ],
        timeout_s=120.0,
    )
    if not dest_png.is_file() or dest_png.stat().st_size < 32:
        raise VideoOpsError("failed to extract first frame", detail={"dest": str(dest_png)})
    return dest_png


def concat_chain_xfade(
    parts: list[Path],
    dest: Path,
    *,
    overlap_seconds: float = 0.5,
    fps: float | None = None,
    max_fraction: float = 0.35,
) -> dict[str, Any]:
    """
    Sequentially xfade-join 2+ clips (A⊕B⊕C…).
    Falls back to hard concat between any pair that cannot blend.
    """
    if len(parts) < 2:
        raise VideoOpsError("concat_chain_xfade needs ≥2 parts")
    for p in parts:
        if not p.is_file():
            raise VideoOpsError(f"missing part: {p}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sc_chain_") as td:
        td_path = Path(td)
        current = parts[0]
        modes: list[str] = []
        overlaps: list[float] = []
        for i, nxt in enumerate(parts[1:]):
            out_i = td_path / f"step_{i}.mp4"
            info = concat_overlap_crossfade(
                current,
                nxt,
                out_i,
                overlap_seconds=overlap_seconds,
                fps=fps,
                max_fraction=max_fraction,
            )
            modes.append(str(info.get("mode") or "?"))
            overlaps.append(float(info.get("overlap_seconds") or 0.0))
            current = out_i
        shutil.copy2(current, dest)

    return {
        "parts": len(parts),
        "bytes": dest.stat().st_size,
        "mode": "chain_xfade",
        "step_modes": modes,
        "step_overlaps": overlaps,
        "overlap_seconds": overlap_seconds,
        "output_duration_s": probe_duration(dest),
    }


def extract_tail_clip(
    video_path: Path,
    dest_mp4: Path,
    *,
    frames: int,
    fps: float | None = None,
) -> dict[str, Any]:
    """
    Export the last `frames` of a video as a short mp4 (for multi-frame I2V prefix).
    """
    frames = max(1, int(frames))
    if not video_path.is_file():
        raise VideoOpsError(f"source video missing: {video_path}")
    dest_mp4.parent.mkdir(parents=True, exist_ok=True)
    fps_v = float(fps or probe_fps(video_path) or 16.0)
    if fps_v < 1:
        fps_v = 16.0
    # Pull a little more than needed from the end, then keep last N frames
    pad_s = (frames + 2) / fps_v + 0.25
    ff = ffmpeg_exe()
    # select last N via reverse/select/reverse is robust without nb_frames
    vf = (
        f"fps={fps_v:.4f},"
        f"reverse,select='lt(n\\,{frames})',reverse,"
        f"setpts=PTS-STARTPTS"
    )
    try:
        _run(
            [
                ff,
                "-y",
                "-sseof",
                f"-{pad_s:.3f}",
                "-i",
                str(video_path),
                "-vf",
                vf,
                "-frames:v",
                str(frames),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "14",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest_mp4),
            ],
            timeout_s=180.0,
        )
    except VideoOpsError:
        # simpler fallback: take last pad_s seconds
        _run(
            [
                ff,
                "-y",
                "-sseof",
                f"-{pad_s:.3f}",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps_v:.4f},setpts=PTS-STARTPTS",
                "-frames:v",
                str(frames),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "14",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest_mp4),
            ],
            timeout_s=180.0,
        )
    if not dest_mp4.is_file() or dest_mp4.stat().st_size < 64:
        raise VideoOpsError("failed to extract tail clip", detail={"dest": str(dest_mp4)})
    got = probe_nb_frames(dest_mp4)
    return {
        "path": str(dest_mp4),
        "frames_requested": frames,
        "frames": got,
        "fps": fps_v,
        "bytes": dest_mp4.stat().st_size,
    }


def trim_end_frames(
    src: Path,
    dest: Path,
    *,
    drop: int,
    fps: float | None = None,
) -> dict[str, Any]:
    """Drop the last `drop` frames (reverse → trim start → reverse)."""
    drop = max(0, int(drop))
    if drop <= 0:
        if src.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        return {"dropped": 0, "mode": "copy", "bytes": dest.stat().st_size}

    dest.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    fps_v = float(fps or probe_fps(src) or 16.0)
    with tempfile.TemporaryDirectory(prefix="sc_tend_") as td:
        td_path = Path(td)
        rev = td_path / "rev.mp4"
        mid = td_path / "mid.mp4"
        _run(
            [
                ff,
                "-y",
                "-i",
                str(src),
                "-vf",
                "reverse",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(rev),
            ],
            timeout_s=300.0,
        )
        trim_start_frames(rev, mid, skip_frames=drop, fps=fps_v)
        _run(
            [
                ff,
                "-y",
                "-i",
                str(mid),
                "-vf",
                "reverse",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(dest),
            ],
            timeout_s=300.0,
        )
    if not dest.is_file() or dest.stat().st_size < 64:
        raise VideoOpsError("trim_end_frames produced empty output")
    return {"dropped": drop, "mode": "trim_end", "bytes": dest.stat().st_size}


def trim_start_frames(
    src: Path,
    dest: Path,
    *,
    skip_frames: int,
    fps: float | None = None,
) -> dict[str, Any]:
    """Drop the first skip_frames from a video (remove multi-frame I2V prefix)."""
    skip_frames = max(0, int(skip_frames))
    if skip_frames <= 0:
        if src.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        return {"skipped": 0, "bytes": dest.stat().st_size, "mode": "copy"}

    if not src.is_file():
        raise VideoOpsError(f"trim source missing: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fps_v = float(fps or probe_fps(src) or 16.0)
    ff = ffmpeg_exe()
    # select from n >= skip
    vf = f"select='gte(n\\,{skip_frames})',setpts=PTS-STARTPTS"
    _run(
        [
            ff,
            "-y",
            "-i",
            str(src),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dest),
        ],
        timeout_s=300.0,
    )
    if not dest.is_file() or dest.stat().st_size < 64:
        raise VideoOpsError("trim_start_frames produced empty output")
    return {
        "skipped": skip_frames,
        "fps": fps_v,
        "bytes": dest.stat().st_size,
        "mode": "trim",
        "output_duration_s": probe_duration(dest),
    }


def smooth_video(
    src: Path,
    dest: Path,
    *,
    factor: int = 2,
    src_fps: float | None = None,
) -> dict[str, Any]:
    """
    Temporal upsample with ffmpeg minterpolate (motion-compensated in-betweens).
    factor=2 → ~2× frames / fps; factor=4 → ~4×.
    """
    if factor not in (2, 4):
        raise VideoOpsError("smooth factor must be 2 or 4", detail={"factor": factor})
    if not src.is_file():
        raise VideoOpsError(f"source video missing: {src}")

    base_fps = src_fps or probe_fps(src) or 16.0
    if base_fps < 1 or base_fps > 120:
        base_fps = 16.0
    out_fps = base_fps * factor
    dest.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()

    # mi_mode=mci = motion compensated interpolation
    vf = (
        f"minterpolate=fps={out_fps:.4f}:mi_mode=mci:mc_mode=aobmc:"
        f"me_mode=bidir:vsbmc=1"
    )
    try:
        _run(
            [
                ff,
                "-y",
                "-i",
                str(src),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest),
            ],
            timeout_s=900.0,
        )
    except VideoOpsError:
        # Fallback: lighter blend interp if MCI fails on some builds
        vf_blend = f"minterpolate=fps={out_fps:.4f}:mi_mode=blend"
        _run(
            [
                ff,
                "-y",
                "-i",
                str(src),
                "-vf",
                vf_blend,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest),
            ],
            timeout_s=900.0,
        )

    if not dest.is_file() or dest.stat().st_size < 64:
        raise VideoOpsError("smooth produced empty output")
    return {
        "source_fps": base_fps,
        "output_fps": out_fps,
        "factor": factor,
        "bytes": dest.stat().st_size,
    }


def concat_videos(parts: list[Path], dest: Path) -> dict[str, Any]:
    """Concatenate mp4 parts with the concat demuxer (re-encode for safety)."""
    if len(parts) < 2:
        raise VideoOpsError("concat needs at least two clips")
    for p in parts:
        if not p.is_file():
            raise VideoOpsError(f"missing concat part: {p}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="sc_concat_") as td:
        list_path = Path(td) / "list.txt"
        # Re-encode path is more reliable across mismatched encodes than -c copy
        lines = []
        for p in parts:
            # escape single quotes for concat file format
            sp = str(p.resolve()).replace("'", "'\\''")
            lines.append(f"file '{sp}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _run(
            [
                ff,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest),
            ],
            timeout_s=600.0,
        )
    if not dest.is_file() or dest.stat().st_size < 64:
        raise VideoOpsError("concat produced empty output")
    return {"parts": len(parts), "bytes": dest.stat().st_size, "mode": "hard"}


def probe_video_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) from ffmpeg -i."""
    try:
        ff = ffmpeg_exe()
        proc = subprocess.run(
            [ff, "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        blob = (proc.stderr or "") + "\n" + (proc.stdout or "")
        m = re.search(r"Video:.*?\s(\d{2,5})x(\d{2,5})\b", blob)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        log.debug("probe_video_size failed", exc_info=True)
    return None


def _normalize_cfr_clip(
    src: Path,
    dest: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> None:
    """Re-encode to constant frame rate + exact size (required for xfade)."""
    ff = ffmpeg_exe()
    w = max(16, int(width) - (int(width) % 2))
    h = max(16, int(height) - (int(height) % 2))
    fps_v = float(fps) if fps and fps > 0 else 16.0
    # scale then fps — order matters for imageio ffmpeg xfade
    vf = f"scale={w}:{h}:flags=bicubic,fps={fps_v:.4f},format=yuv420p"
    _run(
        [
            ff,
            "-y",
            "-i",
            str(src),
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{fps_v:.4f}",
            str(dest),
        ],
        timeout_s=300.0,
    )
    if not dest.is_file() or dest.stat().st_size < 64:
        raise VideoOpsError("normalize cfr clip failed", detail={"src": str(src)})


def concat_overlap_crossfade(
    parent: Path,
    chunk: Path,
    dest: Path,
    *,
    overlap_seconds: float = 0.75,
    overlap_frames: int | None = None,
    fps: float | None = None,
    max_fraction: float = 0.4,
) -> dict[str, Any]:
    """
    Join parent + chunk with an xfade over the shared tail/head.

    Both sides are normalized to the same CFR + resolution first — raw WAN
    mp4s often carry invalid timebases that make xfade fail (hard cut).
    """
    if not parent.is_file() or not chunk.is_file():
        raise VideoOpsError("concat_overlap_crossfade missing input")

    fps_v = float(fps or probe_fps(parent) or probe_fps(chunk) or 16.0)
    size = probe_video_size(parent) or probe_video_size(chunk) or (640, 640)
    width, height = size

    dest.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()

    with tempfile.TemporaryDirectory(prefix="sc_xfade_") as td:
        td_path = Path(td)
        a_n = td_path / "a.mp4"
        b_n = td_path / "b.mp4"
        try:
            _normalize_cfr_clip(parent, a_n, width=width, height=height, fps=fps_v)
            _normalize_cfr_clip(chunk, b_n, width=width, height=height, fps=fps_v)
        except VideoOpsError as e:
            info = concat_videos([parent, chunk], dest)
            info["mode"] = "hard_fallback_normalize"
            info["overlap_seconds"] = 0.0
            info["error"] = str(e)
            return info

        dur_a = probe_duration(a_n) or 0.0
        dur_b = probe_duration(b_n) or 0.0
        if dur_a <= 0 or dur_b <= 0:
            info = concat_videos([a_n, b_n], dest)
            info["mode"] = "hard_fallback_no_duration"
            info["overlap_seconds"] = 0.0
            return info

        overlap = resolve_overlap_seconds(
            overlap_seconds=overlap_seconds,
            overlap_frames=overlap_frames,
            fps=fps_v,
            parent_duration_s=dur_a,
            chunk_duration_s=dur_b,
            max_fraction=max_fraction,
        )
        if overlap <= 0:
            info = concat_videos([a_n, b_n], dest)
            info["overlap_seconds"] = 0.0
            return info

        offset = max(0.0, dur_a - overlap)
        # Working graph (validated): scale,fps,format — no setpts before xfade
        fc = (
            f"[0:v]scale={width}:{height}:flags=bicubic,fps={fps_v:.4f},format=yuv420p[v0];"
            f"[1:v]scale={width}:{height}:flags=bicubic,fps={fps_v:.4f},format=yuv420p[v1];"
            f"[v0][v1]xfade=transition=fade:duration={overlap:.4f}:offset={offset:.4f},"
            f"format=yuv420p[vout]"
        )
        try:
            _run(
                [
                    ff,
                    "-y",
                    "-i",
                    str(a_n),
                    "-i",
                    str(b_n),
                    "-filter_complex",
                    fc,
                    "-map",
                    "[vout]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    "-r",
                    f"{fps_v:.4f}",
                    str(dest),
                ],
                timeout_s=600.0,
            )
        except VideoOpsError as e:
            # Last resort hard concat of normalized clips
            info = concat_videos([a_n, b_n], dest)
            info["mode"] = "hard_fallback_xfade"
            info["overlap_seconds_requested"] = overlap
            info["error"] = str(e)
            info["error_detail"] = getattr(e, "detail", None)
            return info

    if not dest.is_file() or dest.stat().st_size < 64:
        raise VideoOpsError("overlap crossfade produced empty output")

    out_dur = probe_duration(dest)
    return {
        "parts": 2,
        "bytes": dest.stat().st_size,
        "mode": "xfade",
        "overlap_seconds": overlap,
        "overlap_frames": int(round(overlap * fps_v)),
        "fps": fps_v,
        "width": width,
        "height": height,
        "parent_duration_s": dur_a,
        "chunk_duration_s": dur_b,
        "output_duration_s": out_dur,
        "xfade_offset_s": offset,
        "max_fraction": max_fraction,
    }


def first_video_in_job(outputs: list[dict[str, Any]], job_dir: Path) -> Path | None:
    """Resolve path to first video output for a job."""
    for o in outputs or []:
        fn = str(o.get("filename") or "")
        if not fn:
            continue
        ext = Path(fn).suffix.lower()
        if ext not in {".mp4", ".webm", ".mov", ".mkv"}:
            if (o.get("kind") or "") != "video":
                continue
        p = Path(o.get("path") or "")
        if p.is_file():
            return p
        cand = job_dir / fn
        if cand.is_file():
            return cand
    # filesystem fallback
    if job_dir.is_dir():
        for p in sorted(job_dir.iterdir()):
            if p.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"} and p.is_file():
                return p
    return None
