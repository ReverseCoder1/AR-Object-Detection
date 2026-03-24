"""
Dataset Preparation Utility
============================
Helps organize and validate datasets in YOLO format.
Can:
  1. Validate an existing YOLO dataset
  2. Generate a dataset.yaml from a folder structure
  3. Split a flat image+label folder into train/val/test sets
  4. Show class distribution stats

Usage:
    python dataset_prep.py validate --data dataset/
    python dataset_prep.py split --images /path/to/images --labels /path/to/labels
    python dataset_prep.py stats --data dataset/
    python dataset_prep.py yaml --data dataset/ --classes car person bicycle
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import setup_logger

logger = setup_logger("DataPrep")


def validate_dataset(data_root: str):
    root = Path(data_root)
    issues = []

    for split in ["train", "val"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists():
            issues.append(f"Missing: {img_dir}")
            continue
        if not lbl_dir.exists():
            issues.append(f"Missing: {lbl_dir}")
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))

        img_stems = {f.stem for f in images}
        lbl_stems = {f.stem for f in labels}

        unmatched = img_stems - lbl_stems
        if unmatched:
            issues.append(f"{split}: {len(unmatched)} images have no label file")

        logger.info(f"[{split}] images={len(images)}  labels={len(labels)}")

    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        issues.append(f"Missing dataset.yaml at {yaml_path}")

    if issues:
        logger.warning("Validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Dataset validation passed!")
    return len(issues) == 0


def split_dataset(
    images_dir: str, labels_dir: str, output_dir: str, train_ratio=0.80, val_ratio=0.15
):
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    out = Path(output_dir)

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split, imgs in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in imgs:
            lbl = lbl_dir / (img.stem + ".txt")
            shutil.copy(img, out / "images" / split / img.name)
            if lbl.exists():
                shutil.copy(lbl, out / "labels" / split / lbl.name)
        logger.info(f"[{split}] {len(imgs)} samples")

    logger.info(f"Dataset split complete → {out}")


def generate_yaml(data_root: str, class_names: list):
    root = Path(data_root)
    yaml_content = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    out = root / "dataset.yaml"
    with open(out, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    logger.info(f"dataset.yaml written to: {out}")
    logger.info(f"  Classes ({len(class_names)}): {class_names}")


def print_stats(data_root: str):
    root = Path(data_root)
    yaml_path = root / "dataset.yaml"

    class_names = []
    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        class_names = cfg.get("names", [])

    counts = {}
    for lbl_file in root.rglob("labels/**/*.txt"):
        for line in lbl_file.read_text().strip().splitlines():
            if not line:
                continue
            cls_id = int(line.split()[0])
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            counts[name] = counts.get(name, 0) + 1

    logger.info("Class distribution:")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, count // max(1, max(counts.values()) // 40))
        logger.info(f"  {name:<20} {bar} {count}")


def main():
    parser = argparse.ArgumentParser(description="Dataset preparation utility")
    sub = parser.add_subparsers(dest="command")

    p_val = sub.add_parser("validate")
    p_val.add_argument("--data", required=True)

    p_split = sub.add_parser("split")
    p_split.add_argument("--images", required=True)
    p_split.add_argument("--labels", required=True)
    p_split.add_argument("--output", default="dataset_split")
    p_split.add_argument("--train", type=float, default=0.80)
    p_split.add_argument("--val", type=float, default=0.15)

    p_yaml = sub.add_parser("yaml")
    p_yaml.add_argument("--data", required=True)
    p_yaml.add_argument("--classes", nargs="+", required=True)

    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--data", required=True)

    args = parser.parse_args()

    if args.command == "validate":
        validate_dataset(args.data)
    elif args.command == "split":
        split_dataset(args.images, args.labels, args.output, args.train, args.val)
    elif args.command == "yaml":
        generate_yaml(args.data, args.classes)
    elif args.command == "stats":
        print_stats(args.data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
