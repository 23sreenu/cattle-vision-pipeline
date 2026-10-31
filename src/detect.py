"""
Cattle Detection Module — YOLOv11
Author: Sreenivasulu Reddy Settipalli
Affiliation: Texas A&M University
"""

import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


CLASS_NAMES = {0: "cattle", 1: "calf", 2: "bull"}
CLASS_COLORS = {0: (0, 200, 100), 1: (255, 165, 0), 2: (180, 60, 255)}


def load_model(weights: str = "weights/cattle_yolo11.pt", device: str = "auto") -> YOLO:
    """Load the YOLOv11 cattle detection model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)
    model.to(device)
    print(f"[Detect] Model loaded on {device}")
    return model


def detect_image(
    model: YOLO,
    image_path: str,
    conf: float = 0.50,
    iou: float = 0.45,
    visualize: bool = True,
    save_path: str = None,
) -> dict:
    """
    Run detection on a single image.

    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf: Confidence threshold (default 0.50)
        iou: IoU threshold for NMS (default 0.45)
        visualize: Draw bounding boxes on the image
        save_path: Optional output path for annotated image

    Returns:
        dict with keys: boxes, classes, confidences, count, image
    """
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    output = {
        "boxes": boxes,
        "classes": classes,
        "confidences": confs,
        "count": len(boxes),
        "class_counts": {name: 0 for name in CLASS_NAMES.values()},
    }

    for cls in classes:
        output["class_counts"][CLASS_NAMES[cls]] += 1

    if visualize:
        img = cv2.imread(image_path)
        for box, conf_val, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            color = CLASS_COLORS[cls]
            label = f"{CLASS_NAMES[cls]} {conf_val:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        count_text = f"Total cattle: {output['count']}"
        cv2.putText(img, count_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 200), 3)
        output["image"] = img

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"[Detect] Saved to {save_path}")

    print(f"[Detect] Found {output['count']} animals → {output['class_counts']}")
    return output


def detect_video(
    model: YOLO,
    video_path: str,
    output_path: str = "results/detection_output.mp4",
    conf: float = 0.50,
) -> None:
    """Run detection on every frame of a video and save annotated output."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)[0]
        annotated = results.plot()
        out.write(annotated)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"[Detect] Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"[Detect] Video saved → {output_path} ({frame_idx} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cattle Detection with YOLOv11")
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument("--weights", default="weights/cattle_yolo11.pt")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", default="results/output.jpg")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = load_model(args.weights, args.device)
    source = Path(args.source)

    if source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        detect_video(model, str(source), args.output, args.conf)
    else:
        detect_image(model, str(source), args.conf, args.iou, visualize=True, save_path=args.output)
