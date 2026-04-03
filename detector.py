"""
detector.py - Lightweight YOLO-based object detection module
Uses YOLOv8 nano for CPU-efficient inference
"""

import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs relevant to aerial surveillance
# car=2, truck=7, airplane=4, boat=8, motorcycle=3, bus=5
SURVEILLANCE_CLASSES = {
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    7: "truck",
    8: "boat",
}

# Color map for each class (BGR format for OpenCV)
CLASS_COLORS = {
    "car":        (0, 255, 0),       # Green
    "motorcycle": (0, 200, 100),     # Teal
    "airplane":   (0, 100, 255),     # Orange
    "bus":        (0, 165, 255),     # Deep orange
    "truck":      (0, 255, 255),     # Yellow
    "boat":       (255, 100, 0),     # Blue
    "unknown":    (200, 200, 200),   # Gray
}


class SurveillanceDetector:
    """
    Wraps YOLOv8n for aerial surveillance detection.
    Filters only relevant classes and returns normalized detections.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35, input_size: int = 416):
        """
        Args:
            model_path: Path to YOLOv8 weights (auto-downloads if not present)
            confidence: Detection confidence threshold
            input_size: Frame resize dimension for inference (smaller = faster)
        """
        self.confidence = confidence
        self.input_size = input_size

        logger.info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.overrides["verbose"] = False  # Suppress per-frame logs

        logger.info("Detector initialized (CPU mode)")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detection dicts with keys:
              bbox   → [x1, y1, x2, y2] in pixel coords
              conf   → float confidence
              class_id → int COCO class id
              label  → str class name
              color  → BGR tuple for drawing
        """
        results = self.model(
            frame,
            imgsz=self.input_size,
            conf=self.confidence,
            device="cpu",
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in SURVEILLANCE_CLASSES:
                    continue  # Skip irrelevant classes

                label = SURVEILLANCE_CLASSES[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "bbox":     [int(x1), int(y1), int(x2), int(y2)],
                    "conf":     round(conf, 3),
                    "class_id": cls_id,
                    "label":    label,
                    "color":    CLASS_COLORS.get(label, CLASS_COLORS["unknown"]),
                })

        return detections

    def get_class_color(self, label: str) -> tuple:
        return CLASS_COLORS.get(label, CLASS_COLORS["unknown"])