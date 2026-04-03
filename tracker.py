"""
tracker.py - Lightweight SORT (Simple Online and Realtime Tracking) implementation
Pure NumPy/SciPy — no heavy dependencies. Assigns unique IDs across frames.

BUGS FIXED:
  1. associate_detections_to_trackers: The original first loop added low-IoU pairs to
     unmatched_det/unmatched_trk, but since those indices were never placed in
     matched_*_set, the sweep loops below added them AGAIN — causing each unmatched
     detection to spawn two KalmanBoxTrackers per frame. Fixed by only collecting
     true matches in the first loop and letting the sweep loops handle unmatched.

  2. SORTTracker.__init__ now resets KalmanBoxTracker._count to 0 so track IDs
     restart from 1 on each new session (critical for Streamlit multi-run scenarios
     where the same process handles many sequential video uploads).

  3. Cleaned up predicted_boxes / trackers sync: instead of a post-hoc deletion loop
     with an ambiguous ternary pop, we now build valid_trackers and predicted_boxes
     together in a single pass, keeping them guaranteed to be in sync.
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
    _count = 0  # Global ID counter — reset via SORTTracker.__init__ or reset_ids()

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


def associate_detections_to_trackers(
    detections: list, trackers: list, iou_threshold: float = 0.3
):
    """
    Hungarian algorithm to match detections to existing trackers.

    BUG FIX: The original implementation added (d, t) pairs with IoU < threshold to
    unmatched_det and unmatched_trk inside the first loop.  Because those indices were
    never added to matched_det_set / matched_trk_set, the sweep loops at the bottom
    added them a second time — so every rejected pair produced a duplicate entry.
    On the next tracker.update() call those duplicate indices caused two brand-new
    KalmanBoxTrackers to be created from a single detection.

    Fix: only collect genuine matches in the first loop; let the list-comprehension
    sweeps below build the unmatched sets exclusively.

    Returns:
        matches       → list of (det_idx, trk_idx) pairs above iou_threshold
        unmatched_det → detection indices that were not matched
        unmatched_trk → tracker indices that were not matched
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

    matches: list = []
    matched_det_set: set = set()
    matched_trk_set: set = set()

    # FIXED: only register pairs that meet the IoU threshold as true matches.
    # Low-IoU pairs are simply ignored here; the sweeps below handle them correctly.
    for d, t in zip(row_ind, col_ind):
        if cost_matrix[d, t] >= iou_threshold:
            matches.append((d, t))
            matched_det_set.add(d)
            matched_trk_set.add(t)

    unmatched_det = [d for d in range(len(detections)) if d not in matched_det_set]
    unmatched_trk = [t for t in range(len(trackers))   if t not in matched_trk_set]

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
        self.trackers: list = []
        self.frame_count = 0

        # Track full trajectories: id → list of center points
        self.trajectories: dict = defaultdict(list)

        # FIX: Reset global ID counter so each new SORTTracker session starts from 1.
        # Without this, IDs grow unboundedly across Streamlit reruns (or any scenario
        # where multiple SORTTracker instances are created in the same process), making
        # HUD labels confusingly large and wasting trajectory dict memory.
        KalmanBoxTracker._count = 0

        logger.info("SORT Tracker initialized")

    def reset_ids(self):
        """Reset global ID counter and all internal state (call between different video files)."""
        KalmanBoxTracker._count = 0
        self.trackers = []
        self.trajectories.clear()
        self.frame_count = 0

    def update(self, detections: list) -> list:
        """
        Update tracker with new detections for the current frame.

        Args:
            detections: List of detection dicts from detector.py

        Returns:
            List of active tracked objects with bbox, id, label, conf
        """
        self.frame_count += 1

        # 1. Predict new positions for all existing trackers.
        #
        # FIX: Build valid_trackers and predicted_boxes in a single pass so the two
        # lists are always index-aligned.  The original code built predicted_boxes in
        # a loop and then tried to pop from both lists inside a separate deletion loop,
        # which required a fragile bounds-check ternary to avoid IndexError.
        predicted_boxes: list = []
        valid_trackers: list = []
        for trk in self.trackers:
            predicted = trk.predict()
            if not any(np.isnan(v) for v in predicted):
                predicted_boxes.append(predicted)
                valid_trackers.append(trk)
            # NaN trackers are simply dropped here — no separate deletion list needed.

        self.trackers = valid_trackers

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
        active_tracks: list = []
        survivors: list = []

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