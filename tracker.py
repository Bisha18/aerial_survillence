"""
tracker.py - Lightweight SORT (Simple Online and Realtime Tracking) implementation
Pure NumPy/SciPy — no heavy dependencies. Assigns unique IDs across frames.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Kalman Filter for single-object state tracking
# ──────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Tracks a single object using a Kalman filter.
    State vector: [x1, y1, x2, y2, vx, vy, vw, vh]
    """
    _count = 0  # Global ID counter

    def __init__(self, bbox: list):
        """
        Args:
            bbox: [x1, y1, x2, y2]
        """
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.label = "unknown"
        self.conf = 0.0

        # State: position + velocity
        self.state = np.array([
            bbox[0], bbox[1], bbox[2], bbox[3],  # x1, y1, x2, y2
            0.0, 0.0, 0.0, 0.0                   # velocities
        ], dtype=float)

        # Velocity covariance (uncertainty)
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 1000.0  # High uncertainty in velocity initially

        # Process noise
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4) * 1.0

        # Transition matrix (constant velocity model)
        self.F = np.eye(8)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.F[3, 7] = 1.0

        # Measurement matrix (only observe position)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

    def predict(self):
        """Kalman predict step — advance state by one time step."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.state[:4].astype(int).tolist()

    def update(self, bbox: list, label: str = "unknown", conf: float = 0.0):
        """Kalman update step — incorporate new measurement."""
        z = np.array(bbox, dtype=float)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.label = label
        self.conf = conf

    def get_state(self) -> list:
        return self.state[:4].astype(int).tolist()


# ──────────────────────────────────────────────
# IoU utility
# ──────────────────────────────────────────────

def iou(box1: list, box2: list) -> float:
    """Compute Intersection over Union of two boxes [x1,y1,x2,y2]."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def associate_detections_to_trackers(detections: list, trackers: list, iou_threshold: float = 0.3):
    """
    Hungarian algorithm to match detections to existing trackers.

    Returns:
        matches       → list of (det_idx, trk_idx) pairs
        unmatched_det → indices of detections with no tracker
        unmatched_trk → indices of trackers with no detection
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    # Build IoU cost matrix
    cost_matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            cost_matrix[d, t] = iou(det["bbox"], trk)

    # Hungarian assignment (maximize IoU → minimize negative IoU)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    matches, unmatched_det, unmatched_trk = [], [], []

    matched_det_set = set()
    matched_trk_set = set()

    for d, t in zip(row_ind, col_ind):
        if cost_matrix[d, t] < iou_threshold:
            unmatched_det.append(d)
            unmatched_trk.append(t)
        else:
            matches.append((d, t))
            matched_det_set.add(d)
            matched_trk_set.add(t)

    for d in range(len(detections)):
        if d not in matched_det_set:
            unmatched_det.append(d)

    for t in range(len(trackers)):
        if t not in matched_trk_set:
            unmatched_trk.append(t)

    return matches, unmatched_det, unmatched_trk


# ──────────────────────────────────────────────
# SORT Tracker
# ──────────────────────────────────────────────

class SORTTracker:
    """
    SORT: Simple Online and Realtime Tracking
    Maintains per-object Kalman trackers across frames.
    """

    def __init__(self, max_age: int = 5, min_hits: int = 2, iou_threshold: float = 0.3):
        """
        Args:
            max_age:       Frames to keep a tracker without updates before deletion
            min_hits:      Minimum hits before a tracker is reported
            iou_threshold: Minimum IoU for detection-tracker association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

        # Track full trajectories: id → list of center points
        self.trajectories = defaultdict(list)

        logger.info("SORT Tracker initialized")

    def reset_ids(self):
        """Reset global ID counter (call between different video files)."""
        KalmanBoxTracker._count = 0
        self.trackers = []
        self.trajectories.clear()
        self.frame_count = 0

    def update(self, detections: list) -> list[dict]:
        """
        Update tracker with new detections for the current frame.

        Args:
            detections: List of detection dicts from detector.py

        Returns:
            List of active tracked objects with bbox, id, label, conf
        """
        self.frame_count += 1

        # 1. Predict new positions for all existing trackers
        predicted_boxes = []
        to_delete = []
        for i, trk in enumerate(self.trackers):
            predicted = trk.predict()
            # Sanity-check prediction is on-screen (rough check)
            if any(np.isnan(predicted)):
                to_delete.append(i)
                continue
            predicted_boxes.append(predicted)

        for i in reversed(to_delete):
            self.trackers.pop(i)
            predicted_boxes.pop(i) if i < len(predicted_boxes) else None

        # 2. Match detections to trackers
        matches, unmatched_det, unmatched_trk = associate_detections_to_trackers(
            detections, predicted_boxes, self.iou_threshold
        )

        # 3. Update matched trackers
        for d_idx, t_idx in matches:
            det = detections[d_idx]
            self.trackers[t_idx].update(det["bbox"], det["label"], det["conf"])

        # 4. Create new trackers for unmatched detections
        for d_idx in unmatched_det:
            det = detections[d_idx]
            new_trk = KalmanBoxTracker(det["bbox"])
            new_trk.update(det["bbox"], det["label"], det["conf"])
            self.trackers.append(new_trk)

        # 5. Build output list and prune dead trackers
        active_tracks = []
        survivors = []

        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                continue  # Prune stale tracker

            survivors.append(trk)

            # Only report trackers that have been seen enough times
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = trk.get_state()
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                self.trajectories[trk.id].append((cx, cy))

                # Keep trajectory length bounded
                if len(self.trajectories[trk.id]) > 50:
                    self.trajectories[trk.id].pop(0)

                active_tracks.append({
                    "id":         trk.id,
                    "bbox":       bbox,
                    "label":      trk.label,
                    "conf":       trk.conf,
                    "trajectory": list(self.trajectories[trk.id]),
                })

        self.trackers = survivors
        return active_tracks