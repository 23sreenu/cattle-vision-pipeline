"""
SAM 2.1 Segmentation Module — Cattle Instance Segmentation
Author: Sreenivasulu Reddy Settipalli
Affiliation: Texas A&M University
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Optional


def load_sam(checkpoint: str = "weights/sam2.1_hiera_large.pt", device: str = "auto"):
    """
    Load SAM 2.1 model.

    Requires: pip install git+https://github.com/facebookresearch/segment-anything-2.git
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "SAM 2 not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything-2.git"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"[Segment] SAM 2.1 loaded on {device}")
    return predictor


def segment_from_boxes(
    predictor,
    image: np.ndarray,
    boxes: np.ndarray,
    visualize: bool = True,
) -> dict:
    """
    Segment cattle using bounding boxes as prompts (box-prompted mode).

    Args:
        predictor: SAM2ImagePredictor instance
        image: BGR image as numpy array
        boxes: Nx4 array of bounding boxes [x1, y1, x2, y2]
        visualize: Whether to produce annotated image

    Returns:
        dict with masks, scores, annotated image
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    all_masks, all_scores, all_logits = [], [], []

    for box in boxes:
        box_np = np.array(box)[None, :]  # (1, 4)
        masks, scores, logits = predictor.predict(
            box=box_np,
            multimask_output=False,
        )
        all_masks.append(masks[0])
        all_scores.append(scores[0])
        all_logits.append(logits[0])

    output = {
        "masks": all_masks,
        "scores": all_scores,
        "count": len(all_masks),
    }

    if visualize and all_masks:
        annotated = image.copy()
        overlay = image.copy()
        palette = _generate_palette(len(all_masks))

        for i, (mask, score) in enumerate(zip(all_masks, all_scores)):
            color = palette[i]
            overlay[mask.astype(bool)] = color
            # Draw mask boundary
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

        # Blend overlay
        annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)
        cv2.putText(annotated, f"Segmented: {len(all_masks)} cattle", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 180), 3)
        output["image"] = annotated

    print(f"[Segment] Segmented {len(all_masks)} cattle instances")
    return output


def segment_auto(predictor, image: np.ndarray, points_per_side: int = 32) -> dict:
    """
    Automatic everything mode — no prompts needed. Useful for exploratory analysis.

    Args:
        predictor: SAM2ImagePredictor instance
        image: BGR image
        points_per_side: Grid density for auto segmentation

    Returns:
        dict with masks and annotated image
    """
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        raise ImportError("SAM 2 not installed. See load_sam() for install instructions.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_generator = SAM2AutomaticMaskGenerator(
        predictor.model,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
    )
    masks = mask_generator.generate(image_rgb)

    annotated = image.copy()
    palette = _generate_palette(len(masks))
    for i, m in enumerate(masks):
        annotated[m["segmentation"]] = np.array(palette[i]) * 0.5 + annotated[m["segmentation"]] * 0.5

    return {"masks": masks, "count": len(masks), "image": annotated}


def _generate_palette(n: int) -> List[tuple]:
    """Generate N distinct BGR colors for visualization."""
    import colorsys
    palette = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.85, 0.95)]
        palette.append((b, g, r))  # BGR
    return palette


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cattle Segmentation with SAM 2.1")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--checkpoint", default="weights/sam2.1_hiera_large.pt")
    parser.add_argument("--mode", choices=["auto", "box"], default="auto")
    parser.add_argument("--output", default="results/segmentation_output.jpg")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    predictor = load_sam(args.checkpoint, args.device)
    image = cv2.imread(args.image)
    assert image is not None, f"Could not read image: {args.image}"

    if args.mode == "auto":
        result = segment_auto(predictor, image)
    else:
        # Demo: segment full image area as one box prompt
        h, w = image.shape[:2]
        boxes = np.array([[0, 0, w, h]])
        result = segment_from_boxes(predictor, image, boxes)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if "image" in result:
        cv2.imwrite(args.output, result["image"])
        print(f"[Segment] Output saved → {args.output}")
