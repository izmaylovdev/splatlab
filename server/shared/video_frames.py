"""Video → still frames.

Extracts evenly-spaced frames from a video into an image folder so the rest of
the pipeline (COLMAP → training) can treat a clip exactly like a photo set.

Uses the ffmpeg binary bundled with the ``imageio-ffmpeg`` wheel, so there is
nothing to install system-wide (works the same on Windows/Linux/macOS).
"""
from __future__ import annotations

import os
import re
import statistics
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
    oversample: float = 1.0,
    blur_radius: float = 0.5,
    max_blur: float | None = None,
    blur_margin: float | None = None,
    progress: ProgressCb = lambda *_: None,
) -> int:
    """Decode ``video`` into ~``target_frames`` JPEGs under ``out_dir``.

    Returns the number of frames written. Existing files in ``out_dir`` are left
    untouched; new frames are named ``<prefix>_000001.jpg`` … skipping over any
    names already present so repeated uploads don't clobber each other.

    ``oversample`` > 1 samples that many times more candidate frames than needed,
    then keeps only the *sharpest* frame in each evenly-spaced time window — so
    motion-blurred frames are dropped while temporal coverage and the target
    frame count are preserved. ``oversample=1`` (default) thins evenly with no
    sharpness scoring, matching the original behaviour.

    Blur rejection (needs ``oversample`` > 1) picks the sharpest candidate near
    each output slot within ``blur_radius`` slot-spacings and drops picks still
    blurrier than a ``max_blur``/``blur_margin`` cutoff, so the output may be
    fewer than ``target_frames``. A larger ``blur_radius`` is stricter (swaps
    motion-blur for sharp neighbours, fewer frames); see :func:`_select_frames`.
    """
    target_frames = max(1, min(int(target_frames), MAX_TARGET_FRAMES))
    oversample = max(1.0, float(oversample))
    os.makedirs(out_dir, exist_ok=True)
    exe = _ffmpeg_exe()

    duration = _probe_duration(exe, video)
    if duration and duration > 0:
        fps = target_frames * oversample / duration
        fps = max(MIN_FPS, min(fps, MAX_FPS))
    else:
        fps = min(2.0 * oversample, MAX_FPS)  # unknown length → sane default
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

        # If sampling overshot the target, cut down to target_frames: pick the
        # sharpest frame per time window when oversampling for blur rejection,
        # otherwise thin evenly (long clip that hit the fps floor).
        if oversample > 1.0 and len(frames) > 1:
            frames = _select_frames(tmp, frames, target_frames, blur_radius,
                                    max_blur, blur_margin, progress)
        elif len(frames) > target_frames:
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


# Long edge (px) the blur metric downscales to. Blur ranking is stable at low
# resolution and scoring a full 1600px frame is needlessly slow.
_SHARP_METRIC_SIDE = 512


def _load_gray(path: str) -> "object":
    """Grayscale float32 array of the frame, downscaled for metric speed."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L")
        if max(im.size) > _SHARP_METRIC_SIDE:
            im.thumbnail((_SHARP_METRIC_SIDE, _SHARP_METRIC_SIDE))
        return np.asarray(im, dtype=np.float32)


def _box_blur_axis(a: "object", k: int, axis: int) -> "object":
    """Length-``k`` moving-average of ``a`` along ``axis`` (edge-padded)."""
    import numpy as np

    if axis == 0:
        return _box_blur_axis(a.T, k, 1).T
    w = a.shape[1]
    k = max(1, min(k, w))
    pad_l, pad_r = k // 2, k - 1 - k // 2
    ap = np.pad(a, ((0, 0), (pad_l, pad_r)), mode="edge")
    cs = np.cumsum(ap, axis=1)
    cs = np.pad(cs, ((0, 0), (1, 0)), mode="constant")
    return (cs[:, k:] - cs[:, :-k]) / float(k)


def _blur(path: str) -> float:
    """Perceptual blur in [0, 1] — Crete–Roffet. 0 = crisp, 1 = fully blurred.

    Unlike variance-of-Laplacian, this is *normalised for scene content*: it
    re-blurs the frame and measures how much neighbouring-pixel variation is
    lost. A sharp frame loses a lot (there was real high-frequency detail to
    destroy); an already-blurry one barely changes. The ratio cancels out how
    textured the scene is, so a sharp blank wall and a sharp patterned rug both
    score low — which a raw edge-energy measure gets badly wrong.
    """
    import numpy as np

    a = _load_gray(path)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 1.0
    bh = _box_blur_axis(a, 9, axis=1)   # horizontal low-pass
    bv = _box_blur_axis(a, 9, axis=0)   # vertical low-pass

    # neighbour differences (variation) of original vs re-blurred, per direction
    df_h = np.abs(np.diff(a, axis=1))
    df_v = np.abs(np.diff(a, axis=0))
    db_h = np.abs(np.diff(bh, axis=1))
    db_v = np.abs(np.diff(bv, axis=0))

    s_f_h, s_f_v = df_h.sum(), df_v.sum()
    # variation that survived re-blurring; the drop is what blur already removed
    v_h = np.maximum(0.0, df_h - db_h).sum()
    v_v = np.maximum(0.0, df_v - db_v).sum()

    blur_h = (s_f_h - v_h) / s_f_h if s_f_h > 1e-6 else 1.0
    blur_v = (s_f_v - v_v) / s_f_v if s_f_v > 1e-6 else 1.0
    return float(max(blur_h, blur_v))


def _select_frames(tmp: str, frames: list[str], target: int, radius: float,
                   max_blur: float | None, blur_margin: float | None,
                   progress: ProgressCb) -> list[str]:
    """Pick the sharpest frame near each of ``target`` even time slots.

    ``frames`` is the full, time-ordered candidate list. For each of ``target``
    evenly-spaced slots we take the sharpest (least-blurred) candidate within a
    search window of half-width ``radius`` slot-spacings, then drop duplicate
    consecutive picks.

    ``radius`` is the strictness dial. The Crete blur score is only reliable
    *within one scene* (a sharp wall and a blurry rug can score the same in
    absolute terms), so an absolute threshold can't tell them apart — but a
    slot that lands on a blurry pan-moment has a sharp neighbour a fraction of a
    second away, and there Crete *does* rank correctly. A wider ``radius`` lets
    each slot reach that neighbour, so raising it trades frame count for
    sharpness: ~0.5 = adjacent, non-overlapping windows (loosest); 2-3 reliably
    swaps out motion-blur at the cost of ~half the frames.

    As a backstop, a pick still blurrier than the cutoff is dropped (a gap beats
    a smeared frame). The cutoff is ``max_blur`` (absolute 0..1) if given, else
    ``blur_margin`` above the kept frames' median blur, else no dropping. At a
    high ``radius`` the picks are already tight so the backstop rarely fires; it
    mainly catches stretches where the whole neighbourhood is blurred.
    """
    n = len(frames)
    blur = [0.0] * n
    for i, name in enumerate(frames):
        blur[i] = _blur(os.path.join(tmp, name))
        if i % 25 == 0 or i == n - 1:
            progress("sharpness", f"scoring frames {i + 1}/{n}")

    # sharpest candidate within radius of each slot; dedup consecutive picks
    # (a wide radius makes neighbouring slots converge on the same sharp frame)
    slots = min(target, n)
    half = max(1.0, radius) * n / slots
    picks: list[int] = []
    for k in range(slots):
        c = (k + 0.5) * n / slots
        lo = max(0, int(c - half))
        hi = min(n, int(c + half) + 1)
        best = min(range(lo, hi), key=lambda j: blur[j])
        if not picks or picks[-1] != best:
            picks.append(best)

    if max_blur is not None:
        cutoff: float | None = max_blur
    elif blur_margin is not None and picks:
        med = statistics.median(blur[j] for j in picks)
        cutoff = med + blur_margin
        progress("sharpness",
                 f"blur cutoff {cutoff:.2f} (clip median {med:.2f} + {blur_margin:.2f})")
    else:
        cutoff = None

    chosen = [frames[j] for j in picks if cutoff is None or blur[j] <= cutoff]
    dropped = len(picks) - len(chosen)
    if cutoff is not None:
        progress("sharpness",
                 f"kept {len(chosen)} sharp frames; dropped {dropped} too-blurry "
                 f"(blur > {cutoff:.2f})")
    else:
        progress("sharpness", f"kept {len(chosen)} sharpest of {n}")
    return chosen
