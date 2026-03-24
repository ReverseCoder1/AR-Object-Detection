"""
Model Evaluation Script
========================
Evaluate detection performance on a video or image folder.
Reports per-class statistics, FPS, and saves annotated outputs.

Usage:
    python evaluate.py --source test_video.mp4 --domain traffic
    python evaluate.py --source images/ --domain safety --save
    python evaluate.py --source 0 --domain sports --duration 30
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import AppConfig, Domain
from core.detector import ObjectDetector
from core.pipeline import ARPipeline
from utils.logger import setup_logger
from utils.video_source import VideoSource


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AR detection system")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument(
        "--domain",
        type=str,
        default="traffic",
        choices=["traffic", "retail", "safety", "sports"],
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Evaluation duration in seconds (0 = full video)",
    )
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("Evaluate")

    config = AppConfig(
        domain=Domain[args.domain.upper()],
        model_path=args.model,
        confidence_threshold=args.conf,
        save_output=args.save,
    )

    detector = ObjectDetector(config)
    pipeline = ARPipeline(config, detector)

    source = int(args.source) if args.source.isdigit() else args.source
    video = VideoSource(source, config)

    if not video.is_open():
        logger.error("Could not open source.")
        sys.exit(1)

    logger.info(f"Evaluating: {args.domain.upper()} | source: {source}")

    frame_count = 0
    total_dets = 0
    total_inf_ms = 0.0
    class_counts: dict = {}
    fps_list = []
    start = time.time()

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            t0 = time.perf_counter()
            processed, stats = pipeline.process_frame(frame)
            elapsed_frame = time.perf_counter() - t0

            frame_count += 1
            total_dets += stats.get("detections", 0)
            total_inf_ms += stats.get("inference_ms", 0)
            fps_list.append(stats.get("fps", 0))

            for cls, cnt in stats.get("counts", {}).items():
                class_counts[cls] = class_counts.get(cls, 0) + cnt

            if config.display:
                cv2.imshow("Evaluation", processed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.duration and (time.time() - start) >= args.duration:
                break

    except KeyboardInterrupt:
        pass
    finally:
        video.release()
        pipeline.finalize()
        cv2.destroyAllWindows()

    # Report
    logger.info("\n" + "=" * 50)
    logger.info("  EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Frames processed : {frame_count}")
    logger.info(f"  Total detections : {total_dets}")
    logger.info(f"  Avg detections/frame: {total_dets / max(1, frame_count):.2f}")
    logger.info(f"  Avg inference    : {total_inf_ms / max(1, frame_count):.1f} ms")
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    logger.info(f"  Avg FPS          : {avg_fps:.1f}")
    logger.info(f"  Elapsed time     : {time.time() - start:.1f}s")
    logger.info("\n  Class detection counts:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {cls:<20} {cnt}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
