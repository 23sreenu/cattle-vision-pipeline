"""
End-to-End Cattle Vision Pipeline: YOLO Detection → SAM Segmentation → Tracking
Author: Sreenivasulu Reddy Settipalli
Affiliation: Texas A&M University
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from .detect import load_model, detect_image
from .segment import load_sam, segment_from_boxes


class CattlePipeline:
    """
    Full pipeline: YOLOv11 detection → SAM 2.1 segmentation → analytics.

    Usage:
        pipeline = CattlePipeline(
            yolo_weights="weights/cattle_yolo11.pt",
            sam_checkpoint="weights/sam2.1_hiera_large.pt",
        )
        results = pipeline.run("farm.jpg", visualize=True)
        print(f"Cattle count: {results['count']}")
    """

    def __init__(
        self,
        yolo_weights: str = "weights/cattle_yolo11.pt",
        sam_checkpoint: str = "weights/sam2.1_hiera_large.pt",
        device: str = "auto",
        yolo_conf: float = 0.50,
        yolo_iou: float = 0.45,
        use_sam: bool = True,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.use_sam = use_sam

        print(f"[Pipeline] Initializing on {device}")
        self.yolo = load_model(yolo_weights, device)

        if use_sam:
            self.sam = load_sam(sam_checkpoint, device)
        else:
            self.sam = None
            print("[Pipeline] SAM disabled — detection-only mode")

    def run(
        self,
        image_path: str,
        visualize: bool = True,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Process a single image through the full pipeline.

        Returns:
            {
              count: int,
              class_counts: dict,
              boxes: ndarray,
              masks: list,
              conf: float,           # mean confidence
              image: ndarray,        # annotated image (if visualize=True)
            }
        """
        # Stage 1: YOLO Detection
        det = detect_image(
            self.yolo, image_path,
            conf=self.yolo_conf, iou=self.yolo_iou,
            visualize=False,
        )

        result = {
            "count": det["count"],
            "class_counts": det["class_counts"],
            "boxes": det["boxes"],
            "confidences": det["confidences"],
            "conf": float(det["confidences"].mean()) if len(det["confidences"]) else 0.0,
            "masks": [],
        }

        # Stage 2: SAM Segmentation (box-prompted)
        image = cv2.imread(image_path)
        if self.use_sam and self.sam is not None and det["count"] > 0:
            seg = segment_from_boxes(self.sam, image, det["boxes"], visualize=visualize)
            result["masks"] = seg["masks"]
            if visualize:
                result["image"] = seg.get("image", image)
        elif visualize:
            # Detection-only visualization
            from .detect import detect_image as _det
            det_vis = _det(self.yolo, image_path, self.yolo_conf, self.yolo_iou, visualize=True)
            result["image"] = det_vis.get("image", image)

        if save_path and "image" in result:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, result["image"])
            print(f"[Pipeline] Result saved → {save_path}")

        print(f"[Pipeline] Done | Count: {result['count']} | Mean conf: {result['conf']:.3f}")
        return result

    def run_video(
        self,
        video_path: str,
        output: str = "results/pipeline_output.mp4",
        sample_every: int = 1,
    ) -> None:
        """Process a video file frame by frame."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        import tempfile, os
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    cv2.imwrite(tmp.name, frame)
                    r = self.run(tmp.name, visualize=True)
                    out_frame = r.get("image", frame)
                    os.unlink(tmp.name)
            else:
                out_frame = frame

            writer.write(out_frame)
            frame_idx += 1
            if frame_idx % 60 == 0:
                print(f"[Pipeline] Frame {frame_idx}...")

        cap.release()
        writer.release()
        print(f"[Pipeline] Video saved → {output}")
