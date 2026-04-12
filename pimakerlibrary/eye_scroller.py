"""
Hands-Free Eye Tracker for Web Scrolling
=========================================
Control page scrolling using only your eye gaze — no hands needed.
Look down to scroll down, look up to scroll up, center to stop.
Double-blink to left-click.

Usage:
    import pimakerlibrary as pimaker
    pimaker.start_eye_scroller()
    pimaker.start_eye_scroller(scroll_speed=8, enable_click=True, show_camera=True)

Controls:
    q        → Quit
    r        → Re-calibrate
"""

import os
import cv2
import time
import math
import urllib.request
import numpy as np
from collections import deque

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise ImportError(
        "mediapipe is required for start_eye_scroller(). "
        "Install with: pip install mediapipe"
    ) from exc

try:
    import pyautogui
except ImportError as exc:
    raise ImportError(
        "pyautogui is required for start_eye_scroller(). "
        "Install with: pip install pyautogui"
    ) from exc


# ──────────────────────────────────────────────────────────────
#  Landmark Index Constants (MediaPipe 478-point face model)
# ──────────────────────────────────────────────────────────────

# Iris centers (only available with output_face_blendshapes + refined landmarks)
IRIS_LEFT_CENTER  = 468   # Left iris center
IRIS_RIGHT_CENTER = 473   # Right iris center

# Left eye vertical span (top eyelid → bottom eyelid)
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
# Left eye horizontal span (outer → inner corner)
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133

# Right eye vertical span
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
# Right eye horizontal span
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263

# EAR (Eye Aspect Ratio) landmark sets for blink detection
# Each tuple: (top1, bottom1, top2, bottom2, left, right)
LEFT_EAR_IDS  = (159, 145, 158, 153, 33,  133)
RIGHT_EAR_IDS = (386, 374, 385, 380, 362, 263)


# ──────────────────────────────────────────────────────────────
#  Model File Helper
# ──────────────────────────────────────────────────────────────

def _ensure_face_model():
    """Download face_landmarker.task to project root if not present."""
    # First check the project root (where camera.py puts it)
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        print("Downloading MediaPipe Face Landmarker model (~3.7 MB)...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    return model_path


# ──────────────────────────────────────────────────────────────
#  Gaze / EAR Computation
# ──────────────────────────────────────────────────────────────

def _lm_px(landmark, w, h):
    """Convert a normalized MediaPipe landmark to pixel coordinates."""
    return int(landmark.x * w), int(landmark.y * h)


def _compute_vertical_ratio(landmarks, top_id, bottom_id, iris_id, w, h):
    """
    Compute how far down the iris sits between the top and bottom eyelid.
    Returns a value in [0, 1]:
        0.0 = iris at the very top (looking up)
        0.5 = iris centered (looking straight)
        1.0 = iris at the very bottom (looking down)
    Returns None if landmark index is out of range.
    """
    try:
        top    = landmarks[top_id]
        bottom = landmarks[bottom_id]
        iris   = landmarks[iris_id]
    except IndexError:
        return None

    top_y    = top.y * h
    bottom_y = bottom.y * h
    iris_y   = iris.y * h

    span = bottom_y - top_y
    if span < 1:
        return None

    ratio = (iris_y - top_y) / span
    return float(np.clip(ratio, 0.0, 1.0))


def _compute_ear(landmarks, ear_ids, w, h):
    """
    Compute the Eye Aspect Ratio (EAR) for blink detection.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    ear_ids: (top1, bottom1, top2, bottom2, left_corner, right_corner)
    """
    top1_id, bot1_id, top2_id, bot2_id, left_id, right_id = ear_ids
    try:
        top1  = np.array([landmarks[top1_id].x  * w, landmarks[top1_id].y  * h])
        bot1  = np.array([landmarks[bot1_id].x  * w, landmarks[bot1_id].y  * h])
        top2  = np.array([landmarks[top2_id].x  * w, landmarks[top2_id].y  * h])
        bot2  = np.array([landmarks[bot2_id].x  * w, landmarks[bot2_id].y  * h])
        left  = np.array([landmarks[left_id].x  * w, landmarks[left_id].y  * h])
        right = np.array([landmarks[right_id].x * w, landmarks[right_id].y * h])
    except IndexError:
        return None

    vert1 = np.linalg.norm(top1 - bot1)
    vert2 = np.linalg.norm(top2 - bot2)
    horiz = np.linalg.norm(left  - right)

    if horiz < 1:
        return None
    return (vert1 + vert2) / (2.0 * horiz)


# ──────────────────────────────────────────────────────────────
#  Calibration
# ──────────────────────────────────────────────────────────────

def _run_calibration(landmarker, cap, duration_s=2.5):
    """
    Show a calibration screen asking user to look straight ahead.
    Collects eye ratio samples and returns the personal neutral ratio.
    Returns (neutral_ratio, success: bool).
    """
    print("\n[Calibration] Look straight ahead at the camera for the countdown...")
    ratios = []
    start_t = time.time()
    last_timestamp_ms = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        elapsed = time.time() - start_t
        remaining = max(0.0, duration_s - elapsed)

        # ── Run detection ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        if ts <= last_timestamp_ms:
            ts = last_timestamp_ms + 1
        last_timestamp_ms = ts
        result = landmarker.detect_for_video(mp_image, ts)

        # Collect ratio samples
        if result.face_landmarks:
            lms = result.face_landmarks[0]
            if len(lms) > IRIS_LEFT_CENTER:
                left_ratio  = _compute_vertical_ratio(lms, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,  IRIS_LEFT_CENTER,  w, h)
                right_ratio = _compute_vertical_ratio(lms, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, IRIS_RIGHT_CENTER, w, h)
                if left_ratio is not None and right_ratio is not None:
                    ratios.append((left_ratio + right_ratio) / 2.0)

        # ── Draw calibration overlay ──
        overlay = frame.copy()

        # Dark semi-transparent background
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Crosshair target at center
        cx, cy = w // 2, h // 2
        color_cross = (80, 220, 255)
        cv2.line(frame, (cx - 40, cy), (cx + 40, cy), color_cross, 2)
        cv2.line(frame, (cx, cy - 40), (cx, cy + 40), color_cross, 2)
        cv2.circle(frame, (cx, cy), 14, color_cross, 2)
        cv2.circle(frame, (cx, cy), 4,  color_cross, -1)

        # Progress arc (draw a ring that fills up)
        progress = elapsed / duration_s
        axes = (60, 60)
        start_angle = -90
        end_angle   = int(start_angle + 360 * progress)
        cv2.ellipse(frame, (cx, cy), axes, 0, start_angle, end_angle, (100, 255, 150), 4)

        # Text
        cv2.putText(frame, "CALIBRATING", (cx - 90, cy - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Look straight at the camera", (cx - 145, cy + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, f"{remaining:.1f}s", (cx - 22, cy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_cross, 2)

        cv2.imshow("PiMaker Eye Scroller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 0.5, False  # fallback neutral

        if elapsed >= duration_s:
            break

    if len(ratios) < 10:
        print("[Calibration] Not enough data — using default neutral 0.5")
        return 0.5, False

    neutral = float(np.mean(ratios))
    print(f"[Calibration] Done! Your neutral eye ratio: {neutral:.3f}")
    return neutral, True


# ──────────────────────────────────────────────────────────────
#  HUD Renderer
# ──────────────────────────────────────────────────────────────

class EyeHUD:
    """Draws a clean, minimal HUD showing gaze state and actions."""

    # Colors (BGR)
    C_BG      = (18, 18, 18)
    C_ACCENT  = (80, 220, 255)   # Cyan
    C_UP      = (100, 255, 150)  # Green
    C_DOWN    = (80, 140, 255)   # Orange-red
    C_NEUTRAL = (160, 160, 160)  # Gray
    C_BLINK   = (255, 200, 50)   # Yellow
    C_TEXT    = (240, 240, 240)
    C_SUBTLE  = (100, 100, 100)

    def __init__(self):
        self._feedback_text  = ""
        self._feedback_time  = 0.0
        self._feedback_color = self.C_ACCENT
        self._fade_duration  = 1.2

    def flash(self, text, color=None):
        self._feedback_text  = text
        self._feedback_time  = time.time()
        self._feedback_color = color or self.C_ACCENT

    def draw(self, frame, gaze_dir, eye_ratio, neutral, scroll_speed,
             enable_click, blink_count, calibrated):
        h, w, _ = frame.shape
        overlay = frame.copy()

        # ── Top bar ──────────────────────────────────────────
        cv2.rectangle(overlay, (0, 0), (w, 48), self.C_BG, -1)
        title = "EYE SCROLLER  |  Look Down: Scroll  |  Look Up: Scroll  |  Double-Blink: Click"
        cv2.putText(overlay, title, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_ACCENT, 1)

        # Calibrated indicator (top-right)
        cal_color = self.C_UP if calibrated else (80, 80, 255)
        cal_label = "CALIBRATED" if calibrated else "UNCALIBRATED"
        cv2.putText(overlay, cal_label, (w - 165, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 1)

        # ── Bottom bar ────────────────────────────────────────
        bot_y = h - 60
        cv2.rectangle(overlay, (0, bot_y), (w, h), self.C_BG, -1)

        # Gaze direction label
        if gaze_dir == "DOWN":
            dir_color, dir_sym = self.C_DOWN,    "v  SCROLL DOWN"
        elif gaze_dir == "UP":
            dir_color, dir_sym = self.C_UP,      "^  SCROLL UP"
        else:
            dir_color, dir_sym = self.C_NEUTRAL, "o  NEUTRAL"

        cv2.putText(overlay, dir_sym, (16, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, dir_color, 2)

        # Eye ratio bar (shows iris position visually)
        bar_x, bar_y, bar_w, bar_h_ = w - 230, bot_y + 12, 180, 16
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_),
                      (40, 40, 40), -1)
        if eye_ratio is not None:
            fill = int(bar_w * eye_ratio)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h_),
                          dir_color, -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_),
                      self.C_SUBTLE, 1)

        # Neutral tick marker on the bar
        neutral_x = bar_x + int(bar_w * (neutral if neutral else 0.5))
        cv2.line(overlay, (neutral_x, bar_y - 3), (neutral_x, bar_y + bar_h_ + 3),
                 (255, 255, 255), 2)

        cv2.putText(overlay, "Iris position", (bar_x, bot_y + bar_h_ + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.C_SUBTLE, 1)

        # Blink count (if click enabled)
        if enable_click:
            cv2.putText(overlay, f"Blinks: {blink_count}  (double-blink=click)",
                        (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.C_SUBTLE, 1)

        # Quit / recalibrate hint
        cv2.putText(overlay, "q=quit  r=recalibrate", (w - 195, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.C_SUBTLE, 1)

        # ── Blend bars ───────────────────────────────────────
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # ── Side gaze arrow ──────────────────────────────────
        if gaze_dir in ("UP", "DOWN"):
            arrow_x = w - 22
            if gaze_dir == "DOWN":
                cv2.arrowedLine(frame, (arrow_x, h // 2 - 20),
                                (arrow_x, h // 2 + 50),
                                self.C_DOWN, 4, tipLength=0.35)
            else:
                cv2.arrowedLine(frame, (arrow_x, h // 2 + 20),
                                (arrow_x, h // 2 - 50),
                                self.C_UP, 4, tipLength=0.35)

        # ── Centred flash feedback ────────────────────────────
        elapsed = time.time() - self._feedback_time
        if self._feedback_text and elapsed < self._fade_duration:
            alpha = max(0.0, 1.0 - elapsed / self._fade_duration)

            font_scale, thickness = 1.4, 3
            ts = cv2.getTextSize(self._feedback_text, cv2.FONT_HERSHEY_SIMPLEX,
                                 font_scale, thickness)[0]
            cx = w // 2
            cy = h // 2

            pad_x, pad_y = 28, 16
            x1 = cx - ts[0] // 2 - pad_x
            y1 = cy - ts[1] // 2 - pad_y
            x2 = cx + ts[0] // 2 + pad_x
            y2 = cy + ts[1] // 2 + pad_y

            fb_overlay = frame.copy()
            cv2.rectangle(fb_overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
            cv2.rectangle(fb_overlay, (x1, y1), (x2, y2), self._feedback_color, 2)
            cv2.putText(fb_overlay, self._feedback_text,
                        (cx - ts[0] // 2, cy + ts[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        self._feedback_color, thickness)
            cv2.addWeighted(fb_overlay, alpha, frame, 1.0, 0, frame)

        return frame


# ──────────────────────────────────────────────────────────────
#  Eye Landmark Visualiser (draws mesh on the eye region)
# ──────────────────────────────────────────────────────────────

def _draw_eye_landmarks(frame, landmarks, w, h):
    """Draw iris and eyelid landmarks on the frame for visual feedback."""
    eye_ids = [
        LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT,
    ]
    for lm_id in eye_ids:
        if lm_id < len(landmarks):
            px, py = _lm_px(landmarks[lm_id], w, h)
            cv2.circle(frame, (px, py), 3, (80, 200, 255), -1)

    # Iris circles
    for iris_id, color in [(IRIS_LEFT_CENTER, (0, 255, 200)),
                            (IRIS_RIGHT_CENTER, (0, 255, 200))]:
        if iris_id < len(landmarks):
            px, py = _lm_px(landmarks[iris_id], w, h)
            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)


# ──────────────────────────────────────────────────────────────
#  Main Entry Point
# ──────────────────────────────────────────────────────────────

def start_eye_scroller(scroll_speed=5, enable_click=True, show_camera=True):
    """
    Start the hands-free Eye Tracker for scrolling.

    Controls:
        Look down   → scroll down
        Look up     → scroll up
        Look center → stop scrolling
        Double-blink (within 600ms) → left click  [if enable_click=True]
        q → quit
        r → re-calibrate

    Args:
        scroll_speed (int): How aggressively to scroll per trigger (default 5).
                            Higher = faster. Range 1–20 recommended.
        enable_click (bool): Enable double-blink left-click. Default True.
        show_camera (bool): Show the webcam window. Default True.
                            Set False to run silently in background.
    """
    model_path = _ensure_face_model()

    # ── MediaPipe FaceLandmarker setup ──────────────────────
    # output_face_blendshapes gives access to refined iris landmarks (468-477)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,           # Required for iris refinement
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera")

    pyautogui.PAUSE = 0

    print("=" * 55)
    print("  PiMaker Eye Scroller")
    print("=" * 55)
    print("  Look DOWN  → Scroll down")
    print("  Look UP    → Scroll up")
    print("  Look CENTER → Stop scrolling")
    if enable_click:
        print("  Double-blink (fast) → Left click")
    print("  Press 'q' to quit | 'r' to recalibrate")
    print("=" * 55)

    hud = EyeHUD()

    # ── Scroll state ─────────────────────────────────────────
    GAZE_ZONE_THRESHOLD = 0.15   # How far from neutral to trigger (wider dead zone = less accidental)
    SCROLL_INTERVAL_S   = 0.12   # Fire a scroll event every N seconds (slower = smoother)
    RATIO_SMOOTH_ALPHA  = 0.15   # EMA weight for new samples (lower = much smoother)
    DWELL_FRAMES_NEEDED = 8      # Must look in a direction for this many frames before scroll starts

    ratio_smooth    = 0.5
    last_scroll_t   = 0.0
    neutral_ratio   = 0.5
    calibrated      = False
    gaze_dir        = "NEUTRAL"
    dwell_counter   = 0          # Consecutive frames in same gaze direction
    dwell_direction = "NEUTRAL" # Which direction we're dwelling in
    scroll_active   = False      # True once dwell threshold is met
    last_click_time = 0.0        # Post-click scroll pause
    CLICK_SCROLL_PAUSE = 0.5     # Seconds to pause scroll after a click (blink shifts gaze)

    # ── Blink / click state ──────────────────────────────────
    EAR_BLINK_THRESH = 0.15      # Below this → eye closed (stricter = fewer false positives)
    DOUBLE_BLINK_GAP = 0.5       # Max seconds between two blinks for double-blink
    BLINK_MIN_FRAMES = 3         # Minimum frames eye must be closed to count as blink

    blink_count          = 0
    eyes_closed_frames   = 0
    last_blink_time      = 0.0
    pending_first_blink  = False  # Waiting for second blink

    last_timestamp_ms = -1

    with vision.FaceLandmarker.create_from_options(options) as landmarker:

        # ── Calibration ──────────────────────────────────────
        neutral_ratio, calibrated = _run_calibration(landmarker, cap)
        if not calibrated:
            hud.flash("Default neutral applied", EyeHUD.C_BLINK)
        else:
            hud.flash("Calibration complete!", EyeHUD.C_UP)

        # ── Main loop ─────────────────────────────────────────
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # MediaPipe inference
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)
            if ts <= last_timestamp_ms:
                ts = last_timestamp_ms + 1
            last_timestamp_ms = ts
            result = landmarker.detect_for_video(mp_image, ts)

            now = time.time()

            if result.face_landmarks:
                lms = result.face_landmarks[0]

                # ── Check if iris landmarks are available ────
                has_iris = len(lms) > max(IRIS_LEFT_CENTER, IRIS_RIGHT_CENTER)

                if has_iris:
                    # ── Compute raw eye ratios ───────────────
                    left_ratio  = _compute_vertical_ratio(
                        lms, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, IRIS_LEFT_CENTER, w, h)
                    right_ratio = _compute_vertical_ratio(
                        lms, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, IRIS_RIGHT_CENTER, w, h)

                    if left_ratio is not None and right_ratio is not None:
                        raw_ratio = (left_ratio + right_ratio) / 2.0
                        # Exponential Moving Average for smoothing
                        ratio_smooth = (RATIO_SMOOTH_ALPHA * raw_ratio
                                        + (1.0 - RATIO_SMOOTH_ALPHA) * ratio_smooth)

                        # ── Determine gaze direction ─────────
                        deviation = ratio_smooth - neutral_ratio
                        if deviation > GAZE_ZONE_THRESHOLD:
                            raw_gaze = "DOWN"
                        elif deviation < -GAZE_ZONE_THRESHOLD:
                            raw_gaze = "UP"
                        else:
                            raw_gaze = "NEUTRAL"

                        # ── Dwell time: require sustained gaze ─
                        if raw_gaze == dwell_direction and raw_gaze != "NEUTRAL":
                            dwell_counter += 1
                        else:
                            dwell_counter = 0
                            scroll_active = False
                        dwell_direction = raw_gaze

                        if dwell_counter >= DWELL_FRAMES_NEEDED:
                            scroll_active = True

                        if raw_gaze == "NEUTRAL":
                            scroll_active = False
                            dwell_counter = 0

                        gaze_dir = raw_gaze

                        # ── Trigger scroll (only after dwell) ─
                        in_click_pause = (now - last_click_time) < CLICK_SCROLL_PAUSE
                        if (scroll_active and gaze_dir != "NEUTRAL"
                                and (now - last_scroll_t) > SCROLL_INTERVAL_S
                                and not in_click_pause):
                            # Scale scroll amount by how far past threshold we are
                            extra = abs(deviation) - GAZE_ZONE_THRESHOLD
                            speed_mult = max(1.0, extra / GAZE_ZONE_THRESHOLD)
                            amount = int(scroll_speed * speed_mult)

                            if gaze_dir == "DOWN":
                                pyautogui.scroll(-amount)
                            else:
                                pyautogui.scroll(amount)
                            last_scroll_t = now

                    # ── Draw eye landmarks ───────────────────
                    if show_camera:
                        _draw_eye_landmarks(frame, lms, w, h)

                # ── Blink detection (EAR) ────────────────────
                if enable_click:
                    left_ear  = _compute_ear(lms, LEFT_EAR_IDS,  w, h)
                    right_ear = _compute_ear(lms, RIGHT_EAR_IDS, w, h)

                    avg_ear = None
                    if left_ear is not None and right_ear is not None:
                        avg_ear = (left_ear + right_ear) / 2.0
                    elif left_ear is not None:
                        avg_ear = left_ear
                    elif right_ear is not None:
                        avg_ear = right_ear

                    if avg_ear is not None:
                        if avg_ear < EAR_BLINK_THRESH:
                            eyes_closed_frames += 1
                        else:
                            if eyes_closed_frames >= BLINK_MIN_FRAMES:
                                # A blink just completed
                                blink_count += 1

                                if pending_first_blink:
                                    # Second blink within gap → double-blink!
                                    if (now - last_blink_time) <= DOUBLE_BLINK_GAP:
                                        try:
                                            pyautogui.click()
                                        except Exception:
                                            pass
                                        hud.flash("CLICK!", EyeHUD.C_BLINK)
                                        last_click_time = now  # Pause scrolling briefly after click
                                    pending_first_blink = False
                                else:
                                    # First blink — wait for a potential second
                                    pending_first_blink = True
                                    last_blink_time = now

                                # Cancel pending if gap exceeded
                            if pending_first_blink and (now - last_blink_time) > DOUBLE_BLINK_GAP:
                                pending_first_blink = False

                            eyes_closed_frames = 0

                # ── Face bounding box (subtle) ───────────────
                if show_camera:
                    face_lms = result.face_landmarks[0]
                    xs = [lm.x * w for lm in face_lms]
                    ys = [lm.y * h for lm in face_lms]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 1)

            else:
                # No face detected — stop scrolling
                gaze_dir = "NEUTRAL"

            # ── Render HUD ───────────────────────────────────
            if show_camera:
                frame = hud.draw(
                    frame,
                    gaze_dir     = gaze_dir,
                    eye_ratio    = ratio_smooth,
                    neutral      = neutral_ratio,
                    scroll_speed = scroll_speed,
                    enable_click = enable_click,
                    blink_count  = blink_count,
                    calibrated   = calibrated,
                )
                cv2.imshow("PiMaker Eye Scroller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Re-run calibration in-place
                hud.flash("Recalibrating...", EyeHUD.C_ACCENT)
                neutral_ratio, calibrated = _run_calibration(landmarker, cap)
                if calibrated:
                    hud.flash("Calibration complete!", EyeHUD.C_UP)
                else:
                    hud.flash("Calibration failed — default used", EyeHUD.C_BLINK)

    cap.release()
    cv2.destroyAllWindows()
    print("Eye Scroller stopped.")
