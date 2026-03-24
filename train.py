"""
Fine-tune YOLOv8 on a Custom Dataset
=====================================
Use this script to fine-tune (transfer learn) YOLOv8
on your own labeled dataset for a specific domain.

Dataset format:
    dataset/
      images/
        train/   *.jpg
        val/     *.jpg
      labels/
        train/   *.txt   (YOLO format: class cx cy w h — normalized)
        val/     *.txt
      dataset.yaml

Usage:
    python train.py --data dataset/dataset.yaml --domain safety --epochs 50
    python train.py --data dataset/dataset.yaml --model yolov8s.pt --epochs 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import setup_logger

logger = setup_logger("Training")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu, 0 (GPU), mps (Apple M-series)",
    )
    parser.add_argument(
        "--project", type=str, default="models/runs", help="Output directory for runs"
    )
    parser.add_argument("--name", type=str, default="custom_train", help="Run name")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info("=" * 55)
    logger.info("  YOLOv8 Fine-Tuning")
    logger.info("=" * 55)
    logger.info(f"  Data   : {args.data}")
    logger.info(f"  Model  : {args.model}")
    logger.info(f"  Epochs : {args.epochs}")
    logger.info(f"  Batch  : {args.batch}")
    logger.info(f"  ImgSz  : {args.imgsz}")
    logger.info(f"  Device : {args.device}")
    logger.info("=" * 55)

    # Load model
    if args.resume:
        ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if ckpt.exists():
            logger.info(f"Resuming from: {ckpt}")
            model = YOLO(str(ckpt))
        else:
            logger.warning(f"Checkpoint not found at {ckpt}, starting fresh.")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        augment=True,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        verbose=True,
    )

    logger.info("Training complete!")
    logger.info(f"Best model saved at: {results.save_dir}/weights/best.pt")

    # Validation
    logger.info("Running validation on best model...")
    best_model = YOLO(str(Path(results.save_dir) / "weights" / "best.pt"))
    val_results = best_model.val(data=args.data, verbose=True)
    logger.info(f"mAP50    : {val_results.box.map50:.4f}")
    logger.info(f"mAP50-95 : {val_results.box.map:.4f}")

    logger.info("\nTo use your fine-tuned model:")
    logger.info(
        f"  python main.py --model {results.save_dir}/weights/best.pt --domain <domain>"
    )


if __name__ == "__main__":
    main()
