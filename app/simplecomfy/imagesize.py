"""Lightweight image dimension probe (no Pillow required)."""

from __future__ import annotations

import struct
from pathlib import Path


def probe_image_size(path: Path | str) -> tuple[int, int] | None:
    """Return (width, height) for common formats, or None."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            head = f.read(32)
            if len(head) < 24:
                return None
            # PNG
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            # JPEG
            if head.startswith(b"\xff\xd8"):
                f.seek(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        return None
                    mtype = marker[1]
                    while mtype == 0xFF:
                        b = f.read(1)
                        if not b:
                            return None
                        mtype = b[0]
                    # SOF0–SOF3, SOF5–SOF7, SOF9–SOF11, SOF13–SOF15
                    if mtype in (
                        0xC0,
                        0xC1,
                        0xC2,
                        0xC3,
                        0xC5,
                        0xC6,
                        0xC7,
                        0xC9,
                        0xCA,
                        0xCB,
                        0xCD,
                        0xCE,
                        0xCF,
                    ):
                        seg = f.read(7)
                        if len(seg) < 7:
                            return None
                        h, w = struct.unpack(">xHH", seg)
                        return int(w), int(h)
                    if mtype in (0xD8, 0xD9):
                        return None
                    length = f.read(2)
                    if len(length) < 2:
                        return None
                    (ln,) = struct.unpack(">H", length)
                    f.seek(ln - 2, 1)
            # WebP (VP8X / VP8 / VP8L)
            if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
                f.seek(12)
                while True:
                    chunk = f.read(8)
                    if len(chunk) < 8:
                        return None
                    tag, size = struct.unpack("<4sI", chunk)
                    data = f.read(min(size, 16))
                    if tag == b"VP8X" and len(data) >= 10:
                        w = 1 + int.from_bytes(data[4:7], "little")
                        h = 1 + int.from_bytes(data[7:10], "little")
                        return w, h
                    if tag == b"VP8 " and len(data) >= 10 and data[3:6] == b"\x9d\x01\x2a":
                        w, h = struct.unpack("<HH", data[6:10])
                        return int(w) & 0x3FFF, int(h) & 0x3FFF
                    if tag == b"VP8L" and len(data) >= 5 and data[0] == 0x2F:
                        b = int.from_bytes(data[1:5], "little")
                        w = (b & 0x3FFF) + 1
                        h = ((b >> 14) & 0x3FFF) + 1
                        return w, h
                    # skip rest of chunk (odd sizes padded)
                    skip = size - len(data) + (size & 1)
                    if skip > 0:
                        f.seek(skip, 1)
            # GIF
            if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
                w, h = struct.unpack("<HH", head[6:10])
                return int(w), int(h)
    except OSError:
        return None
    return None


def snap_dim(n: int, *, multiple: int = 16, minimum: int = 16) -> int:
    n = max(minimum, int(n))
    return n - (n % multiple)


def fit_video_size(
    width: int,
    height: int,
    *,
    max_side: int = 832,
    multiple: int = 16,
) -> tuple[int, int]:
    """
    Preserve aspect ratio, snap to multiple, cap longest side for VRAM safety.
    """
    w, h = float(width), float(height)
    if w <= 0 or h <= 0:
        return 640, 640
    scale = 1.0
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
    w2 = snap_dim(round(w * scale), multiple=multiple)
    h2 = snap_dim(round(h * scale), multiple=multiple)
    return max(multiple, w2), max(multiple, h2)
