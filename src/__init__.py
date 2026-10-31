"""Cattle Vision Pipeline — YOLO + SAM for livestock management."""
from .detect import load_model, detect_image, detect_video
from .segment import load_sam, segment_from_boxes, segment_auto
from .pipeline import CattlePipeline

__version__ = "1.0.0"
__author__ = "Sreenivasulu Reddy Settipalli"
__email__ = "sreenivasulureddy@tamu.edu"
