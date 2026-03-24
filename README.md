# AR Object Detection System

> Real-time object detection with Augmented Reality overlays using YOLOv8 | Python 3.11 | CPU-Only

---

## What It Does

Detects objects from your webcam or video file and overlays AR information on top — price tags, risk levels, speed estimates, motion trails, and more. Supports 4 domains: **Traffic, Retail, Safety, Sports**.

No GPU needed. No dataset needed. Just install and run.

---

## How It Works

```
Webcam / Video → YOLOv8 Model → Detect Objects → AR Overlay → Display
```

1. **YOLOv8** reads each frame and detects objects with bounding boxes
2. **Domain filter** keeps only relevant classes (e.g. retail keeps bottle, cup, phone)
3. **ByteTrack** assigns unique IDs and records motion trails
4. **AR Renderer** draws corner brackets, labels, info cards, and trails
5. **HUD** shows FPS, object counts, and domain info on screen

### Why No Dataset Is Needed

The model (`yolov8n.pt`) comes **pre-trained** by Ultralytics on the COCO dataset — 118,000 images with 1.5 million labeled objects. You are just using that trained model, not training one yourself. It already knows 80 common objects like cars, people, bottles, phones, scissors, and more.

> Only train a custom model if you need objects **not** in those 80 classes (e.g. hard hats, company logos).

---

## Project Structure

```
ar_detection/
├── main.py               ← Run this
├── train.py              ← Custom training (optional)
├── evaluate.py           ← Benchmark performance
├── dataset_prep.py       ← Prepare custom datasets
├── requirements.txt
├── config/settings.py    ← Domain configs, colors, class maps
├── core/
│   ├── detector.py       ← YOLOv8 inference
│   ├── pipeline.py       ← Frame orchestration
│   └── tracker.py        ← Motion trail tracker
├── ar/
│   ├── renderer.py       ← AR overlays (boxes, labels, trails)
│   └── hud.py            ← Stats HUD (FPS, counts, timer)
└── utils/
    ├── model_manager.py  ← ONNX export & benchmarking
    ├── video_source.py   ← Webcam / video file input
    └── logger.py         ← Terminal logging
```

---

## Installation & Running

### 1. Install

```bash
conda create -n ar_detection python=3.11.14
conda activate ar_detection
pip install -r requirements.txt
```

### 2. Test (no camera needed)

```bash
python main.py --demo
```

### 3. Run with webcam

```bash
python main.py --domain traffic --source 0
```

The model (~6MB) **downloads automatically** on first run.

### 4. Try all domains

```bash
python main.py --domain traffic --source 0   # cars, people, traffic lights
python main.py --domain retail  --source 0   # products with price tags
python main.py --domain safety  --source 0   # risk levels + hazard alerts
python main.py --domain sports  --source 0   # player/ball tracking
```

### 5. Speed boost (optional, one-time)

```bash
python main.py --export-onnx --model yolov8n.pt          # export once (~30s)
python main.py --domain traffic --source 0 --model yolov8n.onnx  # 2x faster
```

### Keyboard shortcuts

| Key | Action                       |
| --- | ---------------------------- |
| `Q` | Quit                         |
| `S` | Save screenshot to `output/` |

---

## What to Put in Front of Camera

| Domain  | Objects to Test                                                   |
| ------- | ----------------------------------------------------------------- |
| Traffic | Toy car, yourself walking, phone showing car/bus image            |
| Retail  | Water bottle, cup, apple, phone, laptop, book, backpack           |
| Safety  | Yourself (MEDIUM risk), scissors or knife (HIGH risk + red alert) |
| Sports  | Yourself moving (PLAYER), any round ball (BALL)                   |

---

## All Arguments

| Argument        | Default      | Description                                |
| --------------- | ------------ | ------------------------------------------ |
| `--domain`      | `traffic`    | `traffic` / `retail` / `safety` / `sports` |
| `--source`      | `0`          | `0` for webcam, or path to video file      |
| `--model`       | `yolov8n.pt` | Model file (`.pt` or `.onnx`)              |
| `--conf`        | `0.45`       | Confidence threshold (0.0–1.0)             |
| `--save`        | off          | Save output video to `output/`             |
| `--demo`        | off          | Run without camera                         |
| `--export-onnx` | off          | Export model to ONNX and exit              |
| `--benchmark`   | off          | Benchmark model speed and exit             |
| `--list-models` | off          | Show all model options and exit            |

---

## Model Options (CPU Speed)

| Model          | Size    | FPS on CPU              |
| -------------- | ------- | ----------------------- |
| `yolov8n.pt`   | 3.2 MB  | 15–25 FPS               |
| `yolov8n.onnx` | 6.2 MB  | 25–40 FPS ← recommended |
| `yolov8s.pt`   | 11.4 MB | 8–15 FPS                |
| `yolov8s.onnx` | 22.4 MB | 12–20 FPS               |

---

## Troubleshooting

| Problem                              | Fix                                                               |
| ------------------------------------ | ----------------------------------------------------------------- |
| `TypeError: bad operand for unary -` | Open `ar/hud.py`, delete `-` at start of line 1                   |
| No camera found                      | Try `--source 1` or use `--demo`                                  |
| Low FPS                              | Export to ONNX: `python main.py --export-onnx --model yolov8n.pt` |
| 0 detections                         | Try `--conf 0.25` or use retail domain with household objects     |
| `lap` warning on first run           | Normal — auto-installs once, restart script                       |
| Model download fails                 | Download `yolov8n.pt` manually from github.com/ultralytics/assets |

---

## Show All 80 Detectable Objects

```bash
python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); print(list(m.names.values()))"
```

Includes: person, car, truck, bus, bottle, cup, phone, laptop, scissors, apple, banana, book, backpack, chair, cat, dog, and 65 more.
