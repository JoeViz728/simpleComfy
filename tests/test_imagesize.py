from __future__ import annotations

import struct
from pathlib import Path

from simplecomfy.imagesize import fit_video_size, probe_image_size, snap_dim


def test_snap_and_fit():
    assert snap_dim(641) == 640
    assert snap_dim(15) == 16
    w, h = fit_video_size(1920, 1080, max_side=832)
    assert w <= 832 and h <= 832
    assert w % 16 == 0 and h % 16 == 0
    # aspect roughly preserved
    assert abs(w / h - 1920 / 1080) < 0.05


def test_probe_png(tmp_path: Path):
    # minimal 32x48 PNG IHDR
    path = tmp_path / "t.png"
    ihdr = struct.pack(">IIBBBBB", 32, 48, 8, 2, 0, 0, 0)
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
    path.write_bytes(raw)
    assert probe_image_size(path) == (32, 48)
