"""Download the optional neural model (YOLOX-nano, 3.6 MB, Apache-2.0) used for photographs."""
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
DEST = Path(__file__).resolve().parent.parent / "models" / "yolox_nano.onnx"

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 3_000_000:
        print("already present:", DEST); sys.exit(0)
    print("downloading", URL)
    urllib.request.urlretrieve(URL, DEST)
    print("saved", DEST, f"({DEST.stat().st_size / 1e6:.1f} MB)")
