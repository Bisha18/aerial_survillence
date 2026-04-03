"""
app.py - Streamlit Dashboard for Aerial Surveillance System
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import logging
from PIL import Image

# Import our pipeline modules
from detector     import SurveillanceDetector
from tracker      import SORTTracker
from fusion       import MultiModalFusion
from intelligence import IntelligenceEngine, AlertLevel
from utils        import (
    FPSCounter, draw_tracks, draw_hud,
    resize_for_inference, scale_detections,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aerial Surveillance System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0a0f1a 0%, #0d2137 50%, #0a0f1a 100%);
        padding: 20px 30px;
        border-radius: 8px;
        border: 1px solid #1a4a6e;
        margin-bottom: 20px;
    }
    .main-header h1 {
        font-family: 'Share Tech Mono', monospace;
        color: #00e5ff;
        font-size: 1.8rem;
        margin: 0;
        text-shadow: 0 0 20px rgba(0,229,255,0.5);
        letter-spacing: 3px;
    }
    .main-header p {
        color: #7fb3c8;
        margin: 4px 0 0 0;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }
    .metric-card {
        background: #0d1b2a;
        border: 1px solid #1a3a5c;
        border-radius: 6px;
        padding: 12px 16px;
        text-align: center;
    }
    .alert-high    { border-color: #ff3333 !important; background: #1a0505 !important; }
    .alert-critical { border-color: #ff00ff !important; background: #1a0520 !important; }
    .alert-medium  { border-color: #ff9900 !important; background: #1a1005 !important; }
    .alert-low     { border-color: #00cc44 !important; background: #051a0a !important; }
    .stButton>button {
        background: linear-gradient(135deg, #0d2137, #1a4a6e);
        color: #00e5ff;
        border: 1px solid #1a4a6e;
        border-radius: 4px;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1a4a6e, #0d2137);
        border-color: #00e5ff;
    }
    div.stAlert { border-radius: 4px; }
    .scanline {
        font-family: 'Share Tech Mono', monospace;
        color: #00e5ff;
        font-size: 0.75rem;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛰️ AERIAL SURVEILLANCE INTELLIGENCE SYSTEM</h1>
    <p>CPU-Optimized  ·  YOLOv8n Detection  ·  SORT Tracking  ·  Multi-Modal Fusion  ·  Rule-Based Intelligence</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar Controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="scanline">▶ SYSTEM CONFIGURATION</p>', unsafe_allow_html=True)
    st.divider()

    conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.35, 0.05,
                                help="Lower = more detections, higher = fewer false positives")

    inference_size = st.select_slider("Inference Size (px)",
                                       options=[256, 320, 416, 512, 640],
                                       value=416,
                                       help="Smaller = faster (CPU-optimized)")

    skip_frames = st.slider("Frame Skip", 1, 5, 2,
                             help="Process every Nth frame. Higher = faster but fewer detections")

    fusion_enabled = st.toggle("Multi-Modal Fusion", value=False,
                                help="Enable thermal + radar simulation (slower but richer)")

    show_trajectories = st.toggle("Show Trajectories", value=True)

    st.divider()
    st.markdown('<p class="scanline">▶ OUTPUT OPTIONS</p>', unsafe_allow_html=True)
    save_output = st.toggle("Save Processed Video", value=False)

    st.divider()
    st.markdown("""
    <div style='font-family: Share Tech Mono; font-size: 0.7rem; color: #446688; line-height: 1.6;'>
    SYSTEM v1.0.0<br>
    MODEL: YOLOv8n (COCO)<br>
    TRACKER: SORT (Kalman)<br>
    PLATFORM: CPU OPTIMIZED<br>
    TARGET: 5-10 FPS
    </div>
    """, unsafe_allow_html=True)


# ── Input area ─────────────────────────────────────────────────────────────────
col_input, col_info = st.columns([2, 1])

with col_input:
    st.markdown("#### 📂 Input Source")
    input_type = st.radio("", ["Upload Video", "Upload Image", "Use Sample (Webcam)"],
                          horizontal=True, label_visibility="collapsed")

    uploaded_file = None
    if input_type == "Upload Video":
        uploaded_file = st.file_uploader("Drop video here", type=["mp4", "avi", "mov", "mkv"],
                                          label_visibility="collapsed")
    elif input_type == "Upload Image":
        uploaded_file = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"],
                                          label_visibility="collapsed")

with col_info:
    st.markdown("#### ⚠️ Alert Level Guide")
    st.markdown("""
    🟢 **LOW** — Zone clear / minimal objects  
    🟡 **MEDIUM** — Moderate activity detected  
    🔴 **HIGH** — Significant concentration / aircraft  
    🚨 **CRITICAL** — Aircraft + vehicles / extreme density  
    ★ = Multi-sensor confirmed object
    """)


st.divider()


# ── Helper: init components ────────────────────────────────────────────────────
@st.cache_resource
def load_detector(conf, imgsz):
    return SurveillanceDetector(confidence=conf, input_size=imgsz)


def process_single_image(frame: np.ndarray, conf: float, imgsz: int,
                          fusion_on: bool, draw_traj: bool) -> tuple:
    """Process one image/frame and return (annotated_frame, report)."""
    detector = load_detector(conf, imgsz)
    tracker  = SORTTracker()
    fusion   = MultiModalFusion(enabled=fusion_on)
    intel    = IntelligenceEngine()

    infer, scale = resize_for_inference(frame, max_dim=imgsz)
    dets = detector.detect(infer)
    dets = scale_detections(dets, scale)

    if fusion_on:
        th_frame = fusion.to_thermal(infer)
        rd_frame = fusion.to_radar(infer)
        th_dets  = scale_detections(detector.detect(th_frame), scale)
        rd_dets  = scale_detections(detector.detect(rd_frame), scale)
        dets     = fusion.merge_detections(dets, th_dets, rd_dets)
    else:
        th_frame = fusion.to_thermal(infer)
        rd_frame = fusion.to_radar(infer)

    tracks = tracker.update(dets)
    report = intel.assess(tracks)

    out = frame.copy()
    out = draw_tracks(out, tracks, draw_trajectory=draw_traj)

    strip = fusion.create_sensor_strip(
        cv2.resize(th_frame, (frame.shape[1], frame.shape[0])),
        cv2.resize(rd_frame, (frame.shape[1], frame.shape[0])),
        strip_height=90,
    )
    out = draw_hud(out, report, fps=0, frame_num=1, sensor_strip=strip, show_strip=True)

    return out, report, th_frame, rd_frame


# ── Image processing ──────────────────────────────────────────────────────────
if input_type == "Upload Image" and uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("🔍 Running detection pipeline..."):
        result_frame, report, thermal, radar = process_single_image(
            img_bgr, conf_threshold, inference_size, fusion_enabled, show_trajectories
        )

    # ── Result display ──────────────────────────────────────────────────────────
    st.markdown("### 🎯 Detection Results")

    # Alert banner
    alert_class_map = {
        AlertLevel.LOW: "alert-low",
        AlertLevel.MEDIUM: "alert-medium",
        AlertLevel.HIGH: "alert-high",
        AlertLevel.CRITICAL: "alert-critical",
    }
    st.markdown(
        f"""<div class="metric-card {alert_class_map.get(report.alert_level, '')}">
        <h2 style="margin:0; color: white; font-family: 'Share Tech Mono', monospace;">
        {report.alert_symbol} ALERT LEVEL: {report.alert_level}</h2>
        <p style="color:#aabbcc; margin:4px 0 0 0;">{report.rationale}</p>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("")

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Objects",   report.total_objects)
    m2.metric("Aircraft",        report.aircraft_count,  delta="⚠️ HIGH" if report.aircraft_count else None)
    m3.metric("Vehicles",        report.vehicle_count)
    m4.metric("Watercraft",      report.watercraft_count)
    m5.metric("Multi-Sensor ★",  report.multi_sensor_confirmed)

    # Image columns
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📡 Annotated Output**")
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, use_container_width=True)

    with c2:
        st.markdown("**🌡️ Thermal View**")
        thermal_resized = cv2.resize(thermal, (result_frame.shape[1], result_frame.shape[0]))
        st.image(cv2.cvtColor(thermal_resized, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Object class breakdown
    if report.object_counts:
        st.markdown("### 📊 Object Breakdown")
        breakdown_cols = st.columns(len(report.object_counts))
        for i, (cls, cnt) in enumerate(sorted(report.object_counts.items())):
            breakdown_cols[i].metric(cls.upper(), cnt)


# ── Video processing ──────────────────────────────────────────────────────────
elif input_type == "Upload Video" and uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Output path
    out_path = tmp_path.replace(".mp4", "_output.mp4")

    st.markdown("### 🎥 Video Processing")

    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        max_frames = st.number_input("Max Frames to Process (0 = all)", 0, 5000, 300, 50)
    with ctrl2:
        display_every = st.number_input("Preview every N frames", 1, 30, 5)
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶ START PROCESSING", use_container_width=True)

    if run_btn:
        # Live metrics placeholders
        prog_bar  = st.progress(0, text="Initializing...")
        frame_ph  = st.empty()

        m_col = st.columns(6)
        met_fps    = m_col[0].empty()
        met_frame  = m_col[1].empty()
        met_total  = m_col[2].empty()
        met_air    = m_col[3].empty()
        met_veh    = m_col[4].empty()
        met_alert  = m_col[5].empty()

        log_ph = st.empty()

        # Init components
        detector  = load_detector(conf_threshold, inference_size)
        tracker   = SORTTracker()
        fusion    = MultiModalFusion(enabled=fusion_enabled)
        intel     = IntelligenceEngine()
        fps_ctr   = FPSCounter(window=20)

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

        # Scale for display
        max_disp = 720
        disp_scale = min(max_disp / orig_w, max_disp / orig_h, 1.0)
        disp_w = int(orig_w * disp_scale)
        disp_h = int(orig_h * disp_scale)

        # Video writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, min(src_fps / skip_frames, 15), (disp_w, disp_h))

        frame_num  = 0
        proc_count = 0
        limit = max_frames if max_frames > 0 else float("inf")
        event_lines = []

        while proc_count < limit:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num % skip_frames != 0:
                continue

            proc_count += 1
            t0 = time.perf_counter()

            disp = cv2.resize(frame, (disp_w, disp_h)) if disp_scale < 1.0 else frame.copy()
            infer, iscale = resize_for_inference(disp, max_dim=inference_size)

            dets = detector.detect(infer)
            dets = scale_detections(dets, iscale)

            if fusion_enabled:
                th = fusion.to_thermal(infer)
                rd = fusion.to_radar(infer)
                dets = fusion.merge_detections(
                    dets,
                    scale_detections(detector.detect(th), iscale),
                    scale_detections(detector.detect(rd), iscale),
                )
            else:
                th = fusion.to_thermal(infer)
                rd = fusion.to_radar(infer)

            tracks = tracker.update(dets)
            report = intel.assess(tracks)
            fps_ctr.tick()

            disp = draw_tracks(disp, tracks, draw_trajectory=show_trajectories)
            strip = fusion.create_sensor_strip(
                cv2.resize(th, (disp_w, disp_h)),
                cv2.resize(rd, (disp_w, disp_h)),
                strip_height=90
            )
            disp = draw_hud(disp, report, fps=fps_ctr.fps,
                            frame_num=frame_num, sensor_strip=strip)

            if writer:
                writer.write(disp)

            # Update UI every N frames
            if proc_count % display_every == 0:
                prog = min(frame_num / max(total_frames, 1), 1.0)
                prog_bar.progress(prog, text=f"Processing frame {frame_num}/{total_frames}...")

                frame_ph.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB),
                               caption=f"Frame {frame_num} | Alert: {report.alert_level}",
                               use_container_width=True)

                met_fps.metric("FPS",     f"{fps_ctr.fps:.1f}")
                met_frame.metric("Frame",  frame_num)
                met_total.metric("Objects", report.total_objects)
                met_air.metric("Aircraft", report.aircraft_count)
                met_veh.metric("Vehicles", report.vehicle_count)
                met_alert.metric("Alert",  f"{report.alert_symbol} {report.alert_level}")

                # Event log
                if report.alert_level in (AlertLevel.HIGH, AlertLevel.CRITICAL):
                    event_lines.append(f"`[{report.timestamp}]` **{report.alert_level}** — {report.rationale}")
                    if len(event_lines) > 15:
                        event_lines.pop(0)
                    log_ph.markdown("**🗒️ Event Log (HIGH/CRITICAL):**\n" + "\n".join(event_lines[-8:]))

        cap.release()
        if writer:
            writer.release()

        prog_bar.progress(1.0, text="✅ Processing complete!")
        st.success(f"Processed {proc_count} frames.")

        if save_output and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Output Video",
                    data=f.read(),
                    file_name="surveillance_output.mp4",
                    mime="video/mp4",
                )

    # Cleanup temp files on rerun (best effort)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass


# ── Webcam / Demo mode ────────────────────────────────────────────────────────
elif input_type == "Use Sample (Webcam)":
    st.info("""
    **Webcam / Live Feed Mode**

    For live webcam processing, run the terminal pipeline directly:
    ```bash
    python main.py --source 0 --fusion
    ```
    The Streamlit dashboard is optimized for file-based processing.
    Upload a video file using the **Upload Video** option above for the full dashboard experience.
    """)

    st.markdown("#### 📌 Quick Start Examples")
    st.code("""
# Basic video file processing
python main.py --source path/to/video.mp4

# With multi-modal fusion enabled
python main.py --source video.mp4 --fusion

# Save output video
python main.py --source video.mp4 --save output.mp4

# Headless (no display window), save only
python main.py --source video.mp4 --save out.mp4 --no-display

# Webcam with every 2nd frame processed
python main.py --source 0 --skip-frames 2 --fusion
    """, language="bash")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div class="scanline" style="text-align:center; opacity:0.5;">'
    'AERIAL SURVEILLANCE INTELLIGENCE SYSTEM  ·  YOLOv8n + SORT  ·  CPU-OPTIMIZED  ·  v1.0.0'
    '</div>',
    unsafe_allow_html=True
)