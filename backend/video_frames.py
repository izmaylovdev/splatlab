"""Video → still frames.

Extracts evenly-spaced frames from a video into an image folder so the rest of
the pipeline (COLMAP → training) can treat a clip exactly like a photo set.

Uses the ffmpeg binary bundled with the ``imageio-ffmpeg`` wheel, so there is
nothing to install system-wide (works the same on Windows/Linux/macOS).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Callable

ProgressCb = Callable[[str, str], None]  # (stage, detail)

# For a good splat we want plenty of well-overlapping views, but COLMAP slows
# down and photo-matching adds little past a few hundred frames.
DEFAULT_TARGET_FRAMES = 150
MAX_TARGET_FRAMES = 400
MIN_FPS = 0.5
MAX_FPS = 6.0

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg  # lazy: only needed when a video is actually uploaded

    return imageio_ffmpeg.get_ffmpeg_exe()


def _probe_duration(exe: str, video: str) -> float | None:
    """Seconds of video, parsed from ffmpeg's stderr, or None if unknown."""
    # `ffmpeg -i <file>` with no output prints stream info to stderr and exits
    # non-zero ("At least one output file must be specified") — that's expected.
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", video],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    m = _DURATION_RE.search(proc.stderr or "")
    if not m:
        return None
    h, mnt, sec = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(sec)


def extract_frames(
    video: str,
    out_dir: str,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    prefix: str = "frame",
    max_image_side: int | None = None,
    progress: ProgressCb = lambda *_: None,
) -> int:
    """Decode ``video`` into ~``target_frames`` JPEGs under ``out_dir``.

    Returns the number of frames written. Existing files in ``out_dir`` are left
    untouched; new frames are named ``<prefix>_000001.jpg`` … skipping over any
    names already present so repeated uploads don't clobber each other.
    """
    target_frames = max(1, min(int(target_frames), MAX_TARGET_FRAMES))
    os.makedirs(out_dir, exist_ok=True)
    exe = _ffmpeg_exe()

    duration = _probe_duration(exe, video)
    if duration and duration > 0:
        fps = target_frames / duration
        fps = max(MIN_FPS, min(fps, MAX_FPS))
    else:
        fps = 2.0  # unknown length → a sane default sampling rate
    progress("extract",
             f"sampling {fps:.2f} fps"
             + (f" from {duration:.0f}s" if duration else ""))

    vf = [f"fps={fps:.4f}"]
    if max_image_side:
        # cap the long edge; keep aspect ratio and even dimensions for the encoder
        vf.append(
            f"scale='if(gt(iw,ih),min(iw,{max_image_side}),-2)':"
            f"'if(gt(iw,ih),-2,min(ih,{max_image_side}))'"
        )

    with tempfile.TemporaryDirectory() as tmp:
        pattern = os.path.join(tmp, "f_%06d.jpg")
        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", video, "-vf", ",".join(vf), "-qscale:v", "2", pattern,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed to decode the video:\n"
                + (proc.stderr or "").strip()[-500:]
            )

        frames = sorted(f for f in os.listdir(tmp) if f.endswith(".jpg"))
        if not frames:
            raise RuntimeError("No frames could be extracted from the video.")

        # If sampling overshot the target (long clip hitting the fps floor),
        # thin evenly down to target_frames so COLMAP stays fast.
        if len(frames) > target_frames:
            step = len(frames) / target_frames
            frames = [frames[int(i * step)] for i in range(target_frames)]

        n = _next_index(out_dir, prefix)
        for i, name in enumerate(frames):
            dest = os.path.join(out_dir, f"{prefix}_{n + i:06d}.jpg")
            os.replace(os.path.join(tmp, name), dest)

    progress("extract", f"{len(frames)} frames extracted")
    return len(frames)


def _next_index(out_dir: str, prefix: str) -> int:
    """First unused <prefix>_NNNNNN index in out_dir."""
    hi = 0
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.jpg$", re.IGNORECASE)
    for f in os.listdir(out_dir):
        m = pat.match(f)
        if m:
            hi = max(hi, int(m.group(1)))
    return hi + 1
