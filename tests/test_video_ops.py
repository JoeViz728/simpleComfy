from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from simplecomfy.video_ops import (
    VideoOpsError,
    concat_overlap_crossfade,
    concat_videos,
    extract_last_frame,
    extract_tail_clip,
    ffmpeg_exe,
    probe_duration,
    resolve_overlap_seconds,
    smooth_video,
    trim_start_frames,
)


def _have_ffmpeg() -> bool:
    try:
        ffmpeg_exe()
        return True
    except VideoOpsError:
        return False


pytestmark = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")


def _make_test_mp4(path: Path, *, frames: int = 8, fps: int = 8) -> Path:
    """Tiny solid-color mp4 via ffmpeg lavfi."""
    ff = ffmpeg_exe()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 320x240 color bars, short
    cmd = [
        ff,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=red:s=320x240:d={frames / fps}",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-t",
        str(frames / fps),
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert path.is_file() and path.stat().st_size > 0
    return path


def test_smooth_2x(tmp_path: Path):
    src = _make_test_mp4(tmp_path / "src.mp4")
    dest = tmp_path / "smooth.mp4"
    info = smooth_video(src, dest, factor=2, src_fps=8.0)
    assert dest.is_file()
    assert info["factor"] == 2
    assert info["output_fps"] == pytest.approx(16.0)


def test_extract_last_frame(tmp_path: Path):
    src = _make_test_mp4(tmp_path / "src.mp4")
    dest = tmp_path / "last.png"
    extract_last_frame(src, dest)
    assert dest.is_file() and dest.stat().st_size > 32


def test_concat(tmp_path: Path):
    a = _make_test_mp4(tmp_path / "a.mp4", frames=6)
    b = _make_test_mp4(tmp_path / "b.mp4", frames=6)
    dest = tmp_path / "cat.mp4"
    info = concat_videos([a, b], dest)
    assert dest.is_file()
    assert info["parts"] == 2


def test_resolve_overlap_clamps_long_parent():
    # 20s parent, 5s chunk, user asks 8s → capped by 40% of chunk (2s)
    ov = resolve_overlap_seconds(
        overlap_seconds=8.0,
        overlap_frames=None,
        fps=16.0,
        parent_duration_s=20.0,
        chunk_duration_s=5.0,
        max_fraction=0.4,
    )
    assert ov == pytest.approx(2.0)


def test_resolve_overlap_zero():
    assert (
        resolve_overlap_seconds(
            overlap_seconds=0,
            overlap_frames=None,
            fps=16,
            parent_duration_s=5,
            chunk_duration_s=5,
        )
        == 0.0
    )


def test_extract_tail_and_trim(tmp_path: Path):
    src = _make_test_mp4(tmp_path / "long.mp4", frames=24, fps=8)
    tail = tmp_path / "tail.mp4"
    info = extract_tail_clip(src, tail, frames=6, fps=8)
    assert tail.is_file()
    assert info["frames_requested"] == 6
    trimmed = tmp_path / "trim.mp4"
    tinfo = trim_start_frames(src, trimmed, skip_frames=8, fps=8)
    assert trimmed.is_file()
    assert tinfo["skipped"] == 8


def test_concat_overlap_crossfade(tmp_path: Path):
    # Distinct colors so xfade has something to blend
    a = _make_test_mp4(tmp_path / "a.mp4", frames=24, fps=8)
    # second clip blue
    ff = ffmpeg_exe()
    b = tmp_path / "b.mp4"
    subprocess.run(
        [
            ff,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=3",
            "-r",
            "8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-t",
            "3",
            str(b),
        ],
        capture_output=True,
        check=True,
    )
    dest = tmp_path / "xfade.mp4"
    info = concat_overlap_crossfade(
        a, b, dest, overlap_seconds=0.5, fps=8.0, max_fraction=0.4
    )
    assert dest.is_file() and dest.stat().st_size > 0
    assert info["mode"] in (
        "xfade",
        "hard_fallback_xfade",
        "hard",
        "hard_fallback_no_duration",
    )
    if info["mode"] == "xfade":
        assert info["overlap_seconds"] == pytest.approx(0.5)
        # blended duration ≈ da + db - overlap
        da = probe_duration(a) or 0
        db = probe_duration(b) or 0
        out = probe_duration(dest) or 0
        assert out == pytest.approx(da + db - 0.5, abs=0.35)
