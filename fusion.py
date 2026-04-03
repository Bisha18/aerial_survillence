"""
fusion.py - Simulated Multi-Modal Sensor Fusion
Generates thermal and radar-like views from optical frames, then merges detections.

BUG FIXED:
  merge_detections: The original code called `d.setdefault("sources", set())` directly
  on the input detection dicts, mutating them in place.  After the call returned, the
  caller's optical_dets / thermal_dets / radar_dets lists contained the extra "sources"
  key — which then leaked into the tracker and intelligence modules, polluting their
  data with unexpected fields and causing subtle isinstance / key-access bugs in
  downstream code that iterated over detection dicts.

  Fix: build shallow copies of each detection dict before tagging with "sources" so
  the originals are never modified.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiModalFusion:
    """
    Simulates multi-sensor fusion for a defense surveillance context:
      - Optical   → raw camera frame
      - Thermal   → false-color heat map via OpenCV colormap
      - Radar     → blurred + noisy + edge-enhanced frame

    Detections from all modalities are merged with a simple
    non-maximum-suppression style deduplication.
    """

    def __init__(self, enabled: bool = True, nms_iou_threshold: float = 0.4):
        self.enabled = enabled
        self.nms_iou_threshold = nms_iou_threshold
        logger.info(f"MultiModalFusion initialized (enabled={enabled})")

    # ──────────────────────────────────────────────
    # Frame transformations
    # ──────────────────────────────────────────────

    def to_thermal(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to a thermal-like representation.
        Uses Gaussian blur to smooth, converts to grayscale, then
        applies the COLORMAP_INFERNO (hot-to-cold palette).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Soften the image (thermal sensors are lower resolution)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        # Apply thermal color map
        thermal = cv2.applyColorMap(blurred, cv2.COLORMAP_INFERNO)
        return thermal

    def to_radar(self, frame: np.ndarray) -> np.ndarray:
        """
        Simulate a radar return: edge-enhanced + noisy + tinted green.
        Mimics SAR (Synthetic Aperture Radar) / FLIR radar-video overlay style.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add Gaussian noise
        noise = np.random.normal(0, 20, gray.shape).astype(np.int16)
        noisy = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Edge detection to simulate radar returns from object boundaries
        edges = cv2.Canny(noisy, threshold1=40, threshold2=120)

        # Blend edges back into noisy image
        combined = cv2.addWeighted(noisy, 0.6, edges, 0.4, 0)

        # Tint green for radar-screen aesthetic
        radar_bgr = np.zeros((*combined.shape, 3), dtype=np.uint8)
        radar_bgr[:, :, 1] = combined  # Green channel only
        radar_bgr[:, :, 0] = (combined * 0.2).astype(np.uint8)  # Slight blue

        return radar_bgr

    # ──────────────────────────────────────────────
    # Detection fusion
    # ──────────────────────────────────────────────

    def _iou(self, b1: list, b2: list) -> float:
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter == 0:
            return 0.0
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def merge_detections(
        self,
        optical_dets: list,
        thermal_dets: list,
        radar_dets:   list,
    ) -> list:
        """
        Merge detections from all three modalities.

        Strategy:
          1. Combine all detections into one pool.
          2. Boost confidence if multiple modalities agree on same region.
          3. Deduplicate with greedy NMS (highest conf first).

        Args:
            optical_dets: Detections from the raw frame
            thermal_dets: Detections from the thermal view
            radar_dets:   Detections from the radar view

        Returns:
            Merged, deduplicated list of detections (input lists are NOT mutated)
        """
        if not self.enabled:
            return optical_dets

        # FIX: Create shallow copies of each detection dict before adding the
        # "sources" key.  The originals must not be modified because the caller may
        # still reference them (e.g., optical_dets is also passed to the tracker
        # when fusion is disabled, and leaked keys like "sources" or "multi_sensor"
        # would confuse downstream dict-iteration in intelligence.py).
        tagged_optical  = [{**d, "sources": {"optical"}}  for d in optical_dets]
        tagged_thermal  = [{**d, "sources": {"thermal"}}  for d in thermal_dets]
        tagged_radar    = [{**d, "sources": {"radar"}}    for d in radar_dets]

        all_dets = tagged_optical + tagged_thermal + tagged_radar

        if not all_dets:
            return []

        # Cross-check: if two detections from different modalities overlap,
        # merge them (take the higher-confidence one and boost slightly)
        merged: list = []
        used = [False] * len(all_dets)

        # Sort by confidence descending
        sorted_idx = sorted(
            range(len(all_dets)),
            key=lambda i: all_dets[i]["conf"],
            reverse=True,
        )

        for i in sorted_idx:
            if used[i]:
                continue
            det = dict(all_dets[i])
            sources = set(det.get("sources", {"optical"}))
            boost = 0.0

            for j in sorted_idx:
                if i == j or used[j]:
                    continue
                other = all_dets[j]
                if self._iou(det["bbox"], other["bbox"]) > self.nms_iou_threshold:
                    # Overlapping detection from another modality
                    sources |= other.get("sources", set())
                    boost += 0.05  # Small confidence boost per corroborating sensor
                    used[j] = True

            # Apply multi-sensor confidence boost (cap at 0.99)
            det["conf"] = min(det["conf"] + boost, 0.99)
            det["sources"] = sources
            det["multi_sensor"] = len(sources) > 1  # Flag multi-modal confirmation
            used[i] = True
            merged.append(det)

        return merged

    # ──────────────────────────────────────────────
    # Sidebar overlay utilities
    # ──────────────────────────────────────────────

    def create_sensor_strip(
        self,
        thermal_frame: np.ndarray,
        radar_frame: np.ndarray,
        strip_height: int = 120,
    ) -> np.ndarray:
        """
        Creates a compact side-by-side thumbnail strip of thermal + radar views.
        Used for the HUD overlay display.
        """
        h = strip_height
        w = int(h * thermal_frame.shape[1] / thermal_frame.shape[0])

        thermal_thumb = cv2.resize(thermal_frame, (w, h))
        radar_thumb   = cv2.resize(radar_frame,   (w, h))

        # Add labels
        cv2.putText(thermal_thumb, "THERMAL", (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(radar_thumb,   "RADAR",   (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        strip = np.hstack([thermal_thumb, radar_thumb])
        return strip