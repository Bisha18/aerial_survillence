"""
main.py - Aerial Surveillance System: Main Pipeline
Orchestrates detection → fusion → tracking → intelligence → HUD rendering
Usage:
    python main.py --source video.mp4
    python main.py --source 0              # webcam
    python main.py --source video.mp4 --save output.mp4
"""

import argparse
import cv2
import time
import logging
import os

from detector     import SurveillanceDetector
from tracker      import SORTTracker
from fusion       import MultiModalFusion
from intelligence import IntelligenceEngine
from utils        import (
    setup_logging, FPSCounter,
    draw_tracks, draw_hud,
    resize_for_inference, scale_detections,
)


def build_args():
    parser = argparse.ArgumentParser(description="Aerial Surveillance System")
    parser.add_argument("--source",     type=str,   default="0",
                        help="Video file path or webcam index (default: 0)")
    parser.add_argument("--model",      type=str,   default="yolov8n.pt",
                        help="YOLOv8 weights path")
    parser.add_argument("--conf",       type=float, default=0.35,
                        help="Detection confidence threshold")
    parser.add_argument("--imgsz",      type=int,   default=416,
                        help="Inference image size (smaller = faster)")
    parser.add_argument("--max-dim",    type=int,   default=640,
                        help="Max display/output frame dimension")
    parser.add_argument("--fusion",     action="store_true", default=False,
                        help="Enable multi-modal fusion (thermal+radar)")
    parser.add_argument("--save",       type=str,   default=None,
                        help="Save output to this .mp4 path")
    parser.add_argument("--skip-frames",type=int,   default=1,
                        help="Process every Nth frame (1=all, 2=every other, etc.)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live window (headless mode)")
    parser.add_argument("--log-file",   type=str,   default=None,
                        help="Write logs to file")
    return parser.parse_args()


def run_pipeline(args):
    logger = setup_logging(log_file=args.log_file)
    logger.info("=" * 60)
    logger.info("  AERIAL SURVEILLANCE SYSTEM — INITIALIZING")
    logger.info("=" * 60)

    # ── Initialize components ────────────────────────────────
    detector  = SurveillanceDetector(
        model_path  = args.model,
        confidence  = args.conf,
        input_size  = args.imgsz,
    )
    tracker   = SORTTracker(max_age=5, min_hits=2, iou_threshold=0.3)
    fusion    = MultiModalFusion(enabled=args.fusion)
    intel     = IntelligenceEngine(history_frames=30)
    fps_counter = FPSCounter(window=20)

    # ── Open video source ────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Could not open video source: {args.source}")
        return

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Source: {args.source} | {orig_w}x{orig_h} @ {src_fps:.1f} FPS | {total_frames} frames")

    # Scale display down if needed
    display_scale = min(args.max_dim / orig_w, args.max_dim / orig_h, 1.0)
    disp_w = int(orig_w * display_scale)
    disp_h = int(orig_h * display_scale)

    # ── Video writer ─────────────────────────────────────────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, min(src_fps, 15), (disp_w, disp_h))
        logger.info(f"Saving output to: {args.save}")

    # ── Main loop ────────────────────────────────────────────
    frame_num  = 0
    proc_count = 0
    last_tracks = []
    last_report = None

    logger.info("Pipeline running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream.")
            break

        frame_num += 1

        # ── Frame skip for performance ───────────────────────
        if frame_num % args.skip_frames != 0:
            continue

        proc_count += 1
        t_start = time.perf_counter()

        # ── Resize for display ──────────────────────────────
        display_frame = cv2.resize(frame, (disp_w, disp_h)) if display_scale < 1.0 else frame.copy()

        # ── Inference frame (further downscaled) ────────────
        infer_frame, infer_scale = resize_for_inference(display_frame, max_dim=args.imgsz)

        # ── Detection ───────────────────────────────────────
        optical_dets = detector.detect(infer_frame)
        optical_dets = scale_detections(optical_dets, infer_scale)

        # ── Multi-Modal Fusion ───────────────────────────────
        if args.fusion:
            thermal_frame = fusion.to_thermal(infer_frame)
            radar_frame   = fusion.to_radar(infer_frame)

            thermal_dets = scale_detections(detector.detect(thermal_frame), infer_scale)
            radar_dets   = scale_detections(detector.detect(radar_frame),   infer_scale)

            merged_dets = fusion.merge_detections(optical_dets, thermal_dets, radar_dets)
        else:
            merged_dets   = optical_dets
            thermal_frame = fusion.to_thermal(infer_frame)
            radar_frame   = fusion.to_radar(infer_frame)

        # ── Tracking ─────────────────────────────────────────
        tracks  = tracker.update(merged_dets)
        last_tracks = tracks

        # ── Intelligence assessment ──────────────────────────
        report       = intel.assess(tracks)
        last_report  = report

        # ── Draw ─────────────────────────────────────────────
        display_frame = draw_tracks(display_frame, tracks, draw_trajectory=True)

        # Sensor strip for HUD
        sensor_strip = fusion.create_sensor_strip(
            cv2.resize(thermal_frame, (disp_w, disp_h)) if infer_scale < 1.0 else thermal_frame,
            cv2.resize(radar_frame,   (disp_w, disp_h)) if infer_scale < 1.0 else radar_frame,
            strip_height=100,
        )

        fps_counter.tick()
        display_frame = draw_hud(
            display_frame, report,
            fps       = fps_counter.fps,
            frame_num = frame_num,
            sensor_strip = sensor_strip,
            show_strip   = True,
        )

        # ── Performance log ───────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if proc_count % 50 == 0:
            logger.info(
                f"Frame {frame_num:>5} | "
                f"{fps_counter.fps:.1f} FPS | "
                f"{elapsed_ms:.0f}ms/frame | "
                f"{report.total_objects} objects | "
                f"Alert: {report.alert_level}"
            )

        # ── Save / Display ────────────────────────────────────
        if writer:
            writer.write(display_frame)

        if not args.no_display:
            cv2.imshow("Aerial Surveillance System", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("User quit.")
                break
            elif key == ord("f"):  # Toggle fusion on-the-fly
                fusion.enabled = not fusion.enabled
                logger.info(f"Fusion toggled: {fusion.enabled}")

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        logger.info(f"Output saved: {args.save}")
    cv2.destroyAllWindows()

    # ── Final summary ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  PIPELINE COMPLETE — Processed {proc_count} frames")
    logger.info("  EVENT LOG (HIGH/CRITICAL alerts):")
    for event in intel.event_log[-10:]:
        logger.info(f"    {event}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline(build_args())