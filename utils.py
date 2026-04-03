"""
utils.py - Drawing helpers, HUD overlay, logging setup, performance metrics
"""

import cv2
import numpy as np
import logging
import time
from collections import deque
from detector import CLASS_COLORS
from intelligence import AlertLevel, ALERT_COLORS


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def setup_logging(level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """Configure root logger with console (and optional file) handler."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("aerial_surveillance")


# ──────────────────────────────────────────────
# FPS Tracker
# ──────────────────────────────────────────────

class FPSCounter:
    """Rolling average FPS counter."""

    def __init__(self, window: int = 30):
        self._times = deque(maxlen=window)
        self._last  = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ──────────────────────────────────────────────
# Drawing: Bounding Boxes + Track IDs
# ──────────────────────────────────────────────

def draw_tracks(frame: np.ndarray, tracks: list, draw_trajectory: bool = True) -> np.ndarray:
    """
    Draw tracking boxes, IDs, and trajectories onto the frame.

    Args:
        frame:           BGR image
        tracks:          List of track dicts from SORTTracker.update()
        draw_trajectory: Whether to draw motion trails
    """
    for track in tracks:
        bbox  = track["bbox"]
        label = track.get("label", "?")
        tid   = track.get("id", -1)
        conf  = track.get("conf", 0.0)
        multi = track.get("multi_sensor", False)

        color = CLASS_COLORS.get(label, (200, 200, 200))

        x1, y1, x2, y2 = bbox

        # Draw box (double border for multi-sensor confirmed)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if multi:
            # Extra thin outer rectangle to indicate multi-sensor confirmation
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

        # Label badge
        tag = f"#{tid} {label} {conf:.2f}"
        if multi:
            tag += " ★"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_y1 = max(y1 - th - 6, 0)
        badge_y2 = max(y1, th + 6)

        cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 4, badge_y2), color, -1)
        cv2.putText(
            frame, tag,
            (x1 + 2, badge_y1 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 0), 1, cv2.LINE_AA
        )

        # Trajectory trail
        if draw_trajectory:
            traj = track.get("trajectory", [])
            for i in range(1, len(traj)):
                alpha = i / len(traj)  # Fade older points
                t_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, traj[i - 1], traj[i], t_color, 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# HUD Overlay
# ──────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    report,
    fps: float,
    frame_num: int,
    sensor_strip: np.ndarray = None,
    show_strip: bool = True,
) -> np.ndarray:
    """
    Render the tactical HUD overlay:
      - Top-left status panel
      - Top-right alert badge
      - Bottom-left object counts
      - Bottom-right sensor strip (optional)
    """
    h, w = frame.shape[:2]

    # ── Alert badge (top-right) ──────────────────
    alert_color = report.alert_color
    alert_text  = f"  {report.alert_level}  "
    (aw, ah), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    ax = w - aw - 20
    ay = 10

    # Pulsing effect simulation: brighter at even frames
    pulse = 255 if (frame_num % 30) < 15 else 200
    pcolor = tuple(min(int(c * pulse / 255), 255) for c in alert_color)

    cv2.rectangle(frame, (ax - 6, ay), (ax + aw + 4, ay + ah + 12), pcolor, -1)
    cv2.rectangle(frame, (ax - 8, ay - 2), (ax + aw + 6, ay + ah + 14), (255, 255, 255), 1)
    cv2.putText(frame, alert_text, (ax, ay + ah + 4),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # ── Status panel (top-left) ──────────────────
    panel_lines = [
        f"AERIAL SURVEILLANCE SYS",
        f"FPS: {fps:.1f}  |  FRAME: {frame_num}",
        f"OBJECTS: {report.total_objects}",
        f"MULTI-SENSOR: {report.multi_sensor_confirmed}",
        f"TIME: {report.timestamp}",
    ]
    panel_x, panel_y = 10, 10
    line_h = 18
    bg_h = len(panel_lines) * line_h + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 4, panel_y - 4),
                  (panel_x + 230, panel_y + bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(panel_lines):
        color = (0, 255, 200) if i == 0 else (200, 200, 200)
        scale = 0.45 if i > 0 else 0.5
        cv2.putText(frame, line, (panel_x, panel_y + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    # ── Object counts (bottom-left) ──────────────
    count_lines = [
        f"  CLASS BREAKDOWN  ",
    ]
    for cls, cnt in sorted(report.object_counts.items()):
        count_lines.append(f"  {cls:<12} {cnt:>2}")

    if report.aircraft_count:
        count_lines.append(f"  >> AIRCRAFT ALERT!")

    for i, line in enumerate(count_lines):
        cy = h - (len(count_lines) - i) * 18 - 10
        color = (0, 255, 255) if i == 0 else (180, 220, 180)
        if "AIRCRAFT" in line:
            color = (0, 50, 255)
        cv2.putText(frame, line, (10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # ── Sensor strip (bottom-right) ──────────────
    if show_strip and sensor_strip is not None:
        sh, sw = sensor_strip.shape[:2]
        sx = w - sw - 10
        sy = h - sh - 10
        if sy > 0 and sx > 0:
            roi = frame[sy:sy + sh, sx:sx + sw]
            blended = cv2.addWeighted(sensor_strip, 0.85, roi, 0.15, 0)
            frame[sy:sy + sh, sx:sx + sw] = blended
            cv2.rectangle(frame, (sx - 1, sy - 1), (sx + sw, sy + sh), (100, 100, 100), 1)

    # ── Tactical grid overlay (subtle) ───────────
    _draw_grid(frame, step=80)

    return frame


def _draw_grid(frame: np.ndarray, step: int = 80, alpha: float = 0.08):
    """Draw a faint tactical grid for the defense-HUD aesthetic."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), (0, 200, 100), 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), (0, 200, 100), 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ──────────────────────────────────────────────
# Frame resize utility
# ──────────────────────────────────────────────

def resize_for_inference(frame: np.ndarray, max_dim: int = 640) -> tuple:
    """
    Resize frame so the longest side = max_dim, preserving aspect ratio.

    Returns:
        (resized_frame, scale_factor)
    """
    h, w = frame.shape[:2]
    scale = min(max_dim / h, max_dim / w, 1.0)  # Never upscale
    if scale == 1.0:
        return frame, 1.0
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), scale


def scale_detections(detections: list, scale: float) -> list:
    """Re-scale detection bboxes back to original frame coordinates."""
    if scale == 1.0:
        return detections
    for d in detections:
        d["bbox"] = [int(v / scale) for v in d["bbox"]]
    return detections