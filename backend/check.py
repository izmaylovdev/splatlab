"""Environment check: python -m backend.check"""
from __future__ import annotations

import shutil


def main() -> None:
    print("SplatLab environment check")
    print("-" * 40)
    try:
        import torch
        print(f"torch          {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA           OK — {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA           NOT available (real training will not work)")
    except ImportError:
        print("torch          not installed (only --mock mode will work)")
    try:
        import gsplat
        print(f"gsplat         {gsplat.__version__}")
    except ImportError:
        print("gsplat         not installed (only --mock mode will work)")
    try:
        import pycolmap
        print(f"pycolmap       {getattr(pycolmap, '__version__', 'ok')}")
    except ImportError:
        print("pycolmap       not installed")
    cli = shutil.which("colmap")
    print(f"colmap CLI     {cli or 'not found (pycolmap will be used)'}")
    try:
        import fastapi, uvicorn  # noqa: F401
        print("fastapi        OK")
    except ImportError:
        print("fastapi        MISSING — pip install -r requirements.txt")


if __name__ == "__main__":
    main()
