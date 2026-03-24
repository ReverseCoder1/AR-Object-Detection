"""
AR Object Detection System - Main Entry Point
============================================
Computer Vision + Augmented Reality pipeline using YOLOv8.
Supports multiple application domains and video sources.

Usage:
    python main.py --domain traffic --source 0          # webcam
    python main.py --domain retail --source video.mp4   # video file
    python main.py --domain safety --source 0 --save    # save output
    python main.py --demo                               # demo mode (no camera)
"""

import argparse
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import AppConfig, Domain
from core.detector import ObjectDetector
from utils.model_manager import (
    list_models,
    export_to_onnx,
    benchmark_model,
    auto_select_model,
)
from core.pipeline import ARPipeline
from utils.logger import setup_logger
from utils.video_source import VideoSource


def parse_args():
    parser = argparse.ArgumentParser(
        description="AR Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="traffic",
        choices=["traffic", "retail", "safety", "sports"],
        help="Application domain (default: traffic)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: 0 for webcam, or path to video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Detection confidence threshold (default: 0.45)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save output video to ./output/"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Run without GUI display"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run demo mode using a synthetic test frame"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Show available CPU model presets and exit",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export --model (.pt) to ONNX for faster CPU inference, then exit",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark --model inference speed, then exit",
    )
    return parser.parse_args()


def run_demo_mode(pipeline, logger):
    import numpy as np

    logger.info("Running in DEMO mode (no camera required)")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    cv2.rectangle(frame, (100, 200), (400, 500), (80, 120, 80), -1)
    cv2.rectangle(frame, (500, 150), (900, 550), (80, 80, 120), -1)
    cv2.rectangle(frame, (950, 300), (1180, 600), (120, 80, 80), -1)
    cv2.putText(
        frame,
        "DEMO MODE - No Camera Required",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (200, 200, 200),
        2,
    )
    cv2.putText(
        frame,
        "Model will detect real objects from live/video input",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        1,
    )
    processed, stats = pipeline.process_frame(frame)
    cv2.imshow("AR Detection System - DEMO", processed)
    logger.info(f"Demo stats: {stats}")
    logger.info("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    logger = setup_logger("ARDetection")
    logger.info("=" * 60)
    logger.info("  AR Object Detection System")
    logger.info("=" * 60)
    logger.info(f"  Domain  : {args.domain.upper()}")
    logger.info(f"  Source  : {args.source}")
    logger.info(f"  Model   : {args.model}")
    logger.info(f"  Conf    : {args.conf}")
    logger.info("=" * 60)

    # ── Model management shortcuts ─────────────────────────────────
    if args.list_models:
        list_models()
        return
    if args.export_onnx:
        export_to_onnx(args.model)
        return
    if args.benchmark:
        benchmark_model(args.model)
        return

    # Auto-select best available model if user left default
    if args.model == "yolov8n.pt":
        args.model = auto_select_model(prefer_speed=True)

    config = AppConfig(
        domain=Domain[args.domain.upper()],
        model_path=args.model,
        confidence_threshold=args.conf,
        frame_width=args.width,
        frame_height=args.height,
        save_output=args.save,
        display=not args.no_display,
    )

    logger.info("Loading YOLO model...")
    detector = ObjectDetector(config)

    logger.info("Initializing AR pipeline...")
    pipeline = ARPipeline(config, detector)

    if args.demo:
        run_demo_mode(pipeline, logger)
        return

    source = int(args.source) if args.source.isdigit() else args.source
    logger.info(f"Opening video source: {source}")
    video = VideoSource(source, config)

    if not video.is_open():
        logger.error("Failed to open video source. Try --demo for demo mode.")
        sys.exit(1)

    logger.info("Starting detection loop. Press 'q' to quit, 's' to screenshot.")
    frame_count = 0
    total_fps = 0.0

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                logger.info("End of video stream.")
                break

            processed_frame, stats = pipeline.process_frame(frame)

            if config.display:
                cv2.imshow(f"AR Detection - {args.domain.upper()}", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User requested quit.")
                    break
                elif key == ord("s"):
                    Path("output").mkdir(exist_ok=True)
                    screenshot_path = f"output/screenshot_{frame_count:04d}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")

            frame_count += 1
            if stats.get("fps"):
                total_fps += stats["fps"]

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        video.release()
        pipeline.finalize()
        cv2.destroyAllWindows()
        if frame_count > 0:
            logger.info(
                f"Processed {frame_count} frames | "
                f"Avg FPS: {total_fps / frame_count:.1f}"
            )
        logger.info("Done.")


if __name__ == "__main__":
    main()
