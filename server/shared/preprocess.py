"""Extract (and downscale) video frames LOCALLY, before uploading.

Uploading a raw video through an SSH tunnel to the GPU box is slow: the whole
clip goes over the wire and only then does the remote extract frames. But
training downsamples every image to ``TrainConfig.max_image_side`` (1600px)
anyway — so extracting at that size locally loses nothing and turns a
hundreds-of-MB video into a few MB of JPEGs.

Run this on your laptop, then upload the output folder (drag-drop it into the
UI's photo picker, or scp it straight to ``data/<project-id>/photos`` on the
box). Reuses the exact same frame extractor the server uses, so the frames are
identical to what an in-browser video upload would have produced.

    python -m server.shared.preprocess clip.mp4 -o frames/
    python -m server.shared.preprocess a.mp4 b.mov -o frames/ --frames 200 --max-side 1600
    python -m server.shared.preprocess ./clips -o frames/          # all videos in a folder
"""
from __future__ import annotations

import argparse
import os
import sys

from .constants import VIDEO_EXTS
from .train_common import TrainConfig
from .video_frames import DEFAULT_TARGET_FRAMES, extract_frames


def _iter_videos(inputs: list[str]) -> list[str]:
    """Expand the given paths (files or directories) into a list of videos."""
    vids: list[str] = []
    for path in inputs:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.lower().endswith(VIDEO_EXTS):
                    vids.append(os.path.join(path, name))
        elif os.path.isfile(path):
            if not path.lower().endswith(VIDEO_EXTS):
                print(f"!! skipping non-video: {path}", file=sys.stderr)
                continue
            vids.append(path)
        else:
            print(f"!! not found: {path}", file=sys.stderr)
    return vids


def _dir_size(path: str) -> int:
    return sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m server.shared.preprocess",
        description="Extract + downscale video frames locally, ready to upload.",
    )
    ap.add_argument("inputs", nargs="+",
                    help="video file(s), or folder(s) of videos")
    ap.add_argument("-o", "--out", required=True,
                    help="output folder for the extracted JPEGs")
    ap.add_argument("--frames", type=int, default=DEFAULT_TARGET_FRAMES,
                    help=f"target frames per video (default {DEFAULT_TARGET_FRAMES})")
    ap.add_argument("--max-side", type=int, default=TrainConfig.max_image_side,
                    help="cap the long edge in px; 0 = keep full resolution "
                         f"(default {TrainConfig.max_image_side}, matches training)")
    ap.add_argument("--oversample", type=float, default=4.0,
                    help="sample Nx more candidates than needed, to choose sharp "
                         "frames from (default 4.0)")
    ap.add_argument("--strictness", type=float, default=2.5,
                    help="how hard to fight blur (search radius in output-slot "
                         "spacings, default 2.5). Higher = each frame is the "
                         "sharpest over a wider span, so motion-blur is swapped "
                         "for a sharp neighbour — at the cost of fewer frames. "
                         "~0.5 = loose; 2-4 = strict.")
    ap.add_argument("--blur-margin", type=float, default=0.06,
                    help="backstop: drop frames blurrier than the clip's median "
                         "by more than this (default 0.06). Ignored if --max-blur "
                         "is given.")
    ap.add_argument("--max-blur", type=float, default=None,
                    help="backstop: absolute blur cutoff (0=crisp..1=blurred) "
                         "instead of the adaptive margin.")
    ap.add_argument("--no-sharp", action="store_true",
                    help="disable blur filtering (even sampling, keep all frames)")
    args = ap.parse_args(argv)

    oversample = 1.0 if args.no_sharp else max(1.0, args.oversample)
    blur_radius = max(0.5, args.strictness)
    if args.no_sharp:
        max_blur = blur_margin = None
    elif args.max_blur is not None:
        max_blur, blur_margin = args.max_blur, None      # absolute override
    else:
        max_blur, blur_margin = None, args.blur_margin   # adaptive (default)

    videos = _iter_videos(args.inputs)
    if not videos:
        print("No videos found in the given inputs.", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    max_side = args.max_side or None

    total = 0
    for video in videos:
        stem = os.path.splitext(os.path.basename(video))[0] or "video"
        prefix = "".join(c if c.isalnum() else "_" for c in stem)[:40] or "video"
        in_size = os.path.getsize(video)
        print(f"==> {os.path.basename(video)}  ({_human(in_size)})")
        try:
            saved = extract_frames(
                video, args.out, target_frames=args.frames, prefix=prefix,
                max_image_side=max_side, oversample=oversample,
                blur_radius=blur_radius, max_blur=max_blur, blur_margin=blur_margin,
                progress=lambda stage, detail: print(f"    {stage}: {detail}"),
            )
        except RuntimeError as e:
            print(f"    !! failed: {e}", file=sys.stderr)
            continue
        total += saved

    if not total:
        print("No frames extracted.", file=sys.stderr)
        return 1

    out_size = _dir_size(args.out)
    print(f"\n{total} frames -> {args.out}  ({_human(out_size)} total)")
    print("Upload this folder: drag it into the UI photo picker, or")
    print(f"  scp -P <SSH_PORT> {args.out}/* root@<HOST>:/workspace/splatlab/data/<project-id>/photos/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
