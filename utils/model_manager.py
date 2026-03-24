"""
CPU Model Manager
=================
Handles model selection, download, and ONNX export for fast CPU inference.

Model performance on CPU (approximate, i5/Ryzen 5 class):
  yolov8n.pt    ~15-25 FPS   (3.2 MB)   ← default, good balance
  yolov8n.onnx  ~25-40 FPS   (6.2 MB)   ← ONNX runtime, fastest CPU
  yolov8s.pt    ~8-15 FPS    (11 MB)    ← more accurate, slower
  yolov8s.onnx  ~12-20 FPS   (22 MB)    ← ONNX small

Recommended for CPU-only systems: yolov8n.onnx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger("ModelManager")

# Available model presets
MODEL_PRESETS = {
    "nano": {
        "file": "yolov8n.pt",
        "size_mb": 3.2,
        "speed": "fastest",
        "accuracy": "good",
    },
    "nano-onnx": {
        "file": "yolov8n.onnx",
        "size_mb": 6.2,
        "speed": "fastest+",
        "accuracy": "good",
    },
    "small": {
        "file": "yolov8s.pt",
        "size_mb": 11.4,
        "speed": "medium",
        "accuracy": "better",
    },
    "small-onnx": {
        "file": "yolov8s.onnx",
        "size_mb": 22.4,
        "speed": "medium+",
        "accuracy": "better",
    },
    "medium": {
        "file": "yolov8m.pt",
        "size_mb": 25.9,
        "speed": "slow",
        "accuracy": "best",
    },
}


def list_models():
    """Print a table of available models."""
    print("\n  Available Models (CPU-optimized recommendations)")
    print("  " + "─" * 62)
    print(f"  {'Preset':<14} {'File':<18} {'Size':>7}  {'Speed':<10} {'Accuracy'}")
    print("  " + "─" * 62)
    for name, info in MODEL_PRESETS.items():
        marker = " ◄ recommended" if name == "nano-onnx" else ""
        print(
            f"  {name:<14} {info['file']:<18} {info['size_mb']:>5.1f}MB  "
            f"{info['speed']:<10} {info['accuracy']}{marker}"
        )
    print("  " + "─" * 62)
    print()


def export_to_onnx(pt_path: str = "yolov8n.pt", output_path: str = None) -> str:
    """
    Export a .pt model to ONNX format for faster CPU inference.

    Returns the path to the exported .onnx file.

    Usage:
        python -m utils.model_manager export --model yolov8n.pt
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return ""

    pt_path = Path(pt_path)
    if not pt_path.exists():
        logger.info(f"Downloading base model: {pt_path.name}")

    logger.info(f"Loading: {pt_path}")
    model = YOLO(str(pt_path))

    logger.info("Exporting to ONNX (this takes ~30 seconds)...")
    export_path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,  # broad compatibility
        simplify=True,  # optimize graph
        dynamic=False,  # fixed input size — faster on CPU
    )

    onnx_path = Path(export_path)
    if output_path:
        import shutil

        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx_path, dest)
        onnx_path = dest

    logger.info(
        f"ONNX model saved: {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)"
    )
    logger.info("Use with:  python main.py --model " + str(onnx_path))
    return str(onnx_path)


def benchmark_model(model_path: str, n_frames: int = 50) -> dict:
    """
    Quick benchmark: run N dummy frames and report avg inference time.

    Returns: {"avg_ms": float, "fps": float, "model": str}
    """
    import time
    import numpy as np

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        return {}

    logger.info(f"Benchmarking: {model_path}  ({n_frames} frames)")
    model = YOLO(model_path)
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warm up
    for _ in range(3):
        model(dummy, verbose=False)

    times = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        model(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    fps = 1000.0 / avg_ms

    logger.info(f"  Avg inference : {avg_ms:.1f} ms")
    logger.info(f"  Theoretical FPS: {fps:.1f}")
    return {"avg_ms": avg_ms, "fps": fps, "model": model_path}


def auto_select_model(prefer_speed: bool = True) -> str:
    """
    Auto-select the best available model for the current system.
    Checks for existing ONNX files first, then falls back to .pt.
    """
    search_dirs = [Path("."), Path("models")]

    # Prefer ONNX for CPU speed
    candidates = (
        ["yolov8n.onnx", "yolov8s.onnx", "yolov8n.pt", "yolov8s.pt"]
        if prefer_speed
        else ["yolov8s.onnx", "yolov8m.pt", "yolov8s.pt", "yolov8n.onnx", "yolov8n.pt"]
    )

    for candidate in candidates:
        for d in search_dirs:
            p = d / candidate
            if p.exists():
                logger.info(f"Auto-selected model: {p}")
                return str(p)

    # Nothing found locally — default will trigger auto-download
    logger.info("No local model found. Will download yolov8n.pt automatically.")
    return "yolov8n.pt"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model management utility")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List available model presets")

    p_export = sub.add_parser(
        "export", help="Export .pt → .onnx for faster CPU inference"
    )
    p_export.add_argument("--model", default="yolov8n.pt", help="Source .pt model")
    p_export.add_argument("--output", default=None, help="Output .onnx path")

    p_bench = sub.add_parser("benchmark", help="Benchmark model inference speed")
    p_bench.add_argument("--model", required=True)
    p_bench.add_argument("--frames", type=int, default=50)

    p_auto = sub.add_parser("auto", help="Auto-select best available model")

    args = parser.parse_args()

    if args.cmd == "list":
        list_models()
    elif args.cmd == "export":
        export_to_onnx(args.model, args.output)
    elif args.cmd == "benchmark":
        benchmark_model(args.model, args.frames)
    elif args.cmd == "auto":
        print(auto_select_model())
    else:
        parser.print_help()
        list_models()
