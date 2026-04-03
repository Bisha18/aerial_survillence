"""
intelligence.py - Rule-Based Intelligence & Priority Engine
Evaluates tracked objects and emits threat/alert level with rationale.
"""

import logging
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Alert level definitions
# ──────────────────────────────────────────────

class AlertLevel:
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    CRITICAL = "CRITICAL"


# Alert → display color (BGR for OpenCV)
ALERT_COLORS = {
    AlertLevel.LOW:      (0, 200, 50),     # Green
    AlertLevel.MEDIUM:   (0, 165, 255),    # Orange
    AlertLevel.HIGH:     (0, 50, 255),     # Red
    AlertLevel.CRITICAL: (180, 0, 255),    # Purple-Red
}

# Alert → emoji/symbol for Streamlit
ALERT_SYMBOLS = {
    AlertLevel.LOW:      "🟢",
    AlertLevel.MEDIUM:   "🟡",
    AlertLevel.HIGH:     "🔴",
    AlertLevel.CRITICAL: "🚨",
}


@dataclass
class IntelligenceReport:
    """Structured output from the intelligence engine."""
    alert_level:    str
    alert_color:    tuple
    alert_symbol:   str
    rationale:      str
    object_counts:  dict = field(default_factory=dict)
    total_objects:  int = 0
    aircraft_count: int = 0
    vehicle_count:  int = 0
    watercraft_count: int = 0
    multi_sensor_confirmed: int = 0
    timestamp:      str = ""

    def summary(self) -> str:
        return (
            f"[{self.alert_symbol} {self.alert_level}] {self.rationale} | "
            f"Total: {self.total_objects} | "
            f"Air: {self.aircraft_count} Land: {self.vehicle_count} Sea: {self.watercraft_count}"
        )


# ──────────────────────────────────────────────
# Intelligence Engine
# ──────────────────────────────────────────────

class IntelligenceEngine:
    """
    Rule-based threat assessment engine.

    Evaluates:
      - Object type distribution (aircraft are weighted higher)
      - Total object count per frame
      - Rapid accumulation / surges across recent frames
      - Multi-sensor confirmed detections (higher certainty)
    """

    # Class groupings
    AIRCRAFT_LABELS   = {"airplane"}
    VEHICLE_LABELS    = {"car", "truck", "bus", "motorcycle"}
    WATERCRAFT_LABELS = {"boat"}

    # Threat weight per class (aircraft = highest concern from aerial perspective)
    THREAT_WEIGHTS = {
        "airplane":   5,
        "boat":       2,
        "truck":      2,
        "bus":        2,
        "car":        1,
        "motorcycle": 1,
    }

    def __init__(self, history_frames: int = 30):
        """
        Args:
            history_frames: Number of recent frames to track for trend analysis
        """
        self.history = deque(maxlen=history_frames)
        self._event_log: list[str] = []
        logger.info("Intelligence Engine initialized")

    def assess(self, tracked_objects: list) -> IntelligenceReport:
        """
        Assess the current frame's tracked objects and return an intelligence report.

        Args:
            tracked_objects: List of track dicts from SORTTracker.update()

        Returns:
            IntelligenceReport with alert level, rationale, and counts
        """
        # Count by category
        counts = {}
        aircraft_count   = 0
        vehicle_count    = 0
        watercraft_count = 0
        multi_confirmed  = 0
        threat_score     = 0

        for obj in tracked_objects:
            label = obj.get("label", "unknown")
            counts[label] = counts.get(label, 0) + 1

            # Category tallies
            if label in self.AIRCRAFT_LABELS:
                aircraft_count += 1
            elif label in self.VEHICLE_LABELS:
                vehicle_count += 1
            elif label in self.WATERCRAFT_LABELS:
                watercraft_count += 1

            # Multi-sensor confirmation
            if obj.get("multi_sensor", False):
                multi_confirmed += 1

            # Weighted threat score
            threat_score += self.THREAT_WEIGHTS.get(label, 1)

        total = len(tracked_objects)
        self.history.append(threat_score)

        # ── Trend: is threat score rising rapidly? ──
        trend_surge = False
        if len(self.history) >= 5:
            recent_avg  = sum(list(self.history)[-5:]) / 5
            older_avg   = sum(list(self.history)[:max(1, len(self.history) - 5)]) / max(1, len(self.history) - 5)
            trend_surge = recent_avg > older_avg * 1.8  # 80% increase

        # ── Rule-based alert classification ──
        alert_level, rationale = self._classify(
            total, aircraft_count, vehicle_count, watercraft_count,
            threat_score, multi_confirmed, trend_surge
        )

        report = IntelligenceReport(
            alert_level     = alert_level,
            alert_color     = ALERT_COLORS[alert_level],
            alert_symbol    = ALERT_SYMBOLS[alert_level],
            rationale       = rationale,
            object_counts   = counts,
            total_objects   = total,
            aircraft_count  = aircraft_count,
            vehicle_count   = vehicle_count,
            watercraft_count= watercraft_count,
            multi_sensor_confirmed = multi_confirmed,
            timestamp       = datetime.now().strftime("%H:%M:%S.%f")[:-3],
        )

        # Log significant events
        if alert_level in (AlertLevel.HIGH, AlertLevel.CRITICAL):
            msg = f"[{report.timestamp}] {report.summary()}"
            self._event_log.append(msg)
            if len(self._event_log) > 200:
                self._event_log.pop(0)

        return report

    def _classify(
        self,
        total: int,
        aircraft: int,
        vehicles: int,
        watercraft: int,
        threat_score: int,
        multi_confirmed: int,
        trend_surge: bool,
    ) -> tuple[str, str]:
        """Core classification rules. Returns (alert_level, rationale)."""

        # CRITICAL: Aircraft present + other activity
        if aircraft >= 1 and (vehicles >= 3 or threat_score >= 12):
            return AlertLevel.CRITICAL, f"Aircraft + {vehicles} vehicles detected — possible coordinated activity"

        # CRITICAL: Extreme vehicle concentration
        if threat_score >= 15 or total >= 10:
            return AlertLevel.CRITICAL, f"High-density object cluster ({total} objects, score={threat_score})"

        # HIGH: Any aircraft
        if aircraft >= 1:
            return AlertLevel.HIGH, f"{aircraft} aircraft detected in zone"

        # HIGH: Significant vehicle activity
        if threat_score >= 8 or vehicles >= 5:
            return AlertLevel.HIGH, f"Significant vehicle concentration ({vehicles} vehicles, {watercraft} watercraft)"

        # HIGH: Surge in activity
        if trend_surge and total >= 3:
            return AlertLevel.HIGH, f"Rapid activity surge detected ({total} objects)"

        # HIGH: Multi-sensor confirmation of multiple objects
        if multi_confirmed >= 3:
            return AlertLevel.HIGH, f"{multi_confirmed} objects confirmed by multiple sensors"

        # MEDIUM: Moderate activity
        if threat_score >= 4 or total >= 3:
            reason_parts = []
            if vehicles:    reason_parts.append(f"{vehicles} vehicle(s)")
            if watercraft:  reason_parts.append(f"{watercraft} watercraft")
            if multi_confirmed: reason_parts.append(f"{multi_confirmed} multi-sensor confirmed")
            return AlertLevel.MEDIUM, "Moderate activity: " + ", ".join(reason_parts) if reason_parts else "Moderate object density"

        # LOW: Minimal or no objects
        if total == 0:
            return AlertLevel.LOW, "No objects detected — zone clear"

        return AlertLevel.LOW, f"Minimal activity ({total} object{'s' if total > 1 else ''})"

    @property
    def event_log(self) -> list[str]:
        return list(self._event_log)