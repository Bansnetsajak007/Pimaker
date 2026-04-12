"""
Gesture-Controlled Media & Presentation Controller
====================================================
Control presentations and media playback using hand gestures.

Gestures:
  - Swipe Right  → Next Slide / Skip Track
  - Swipe Left   → Previous Slide / Previous Track
  - Open Palm    → Play / Pause (hold still with all 5 fingers)
  - Pinch        → Mute / Unmute Toggle
  - Two Hands    → Volume Control (spread apart = louder)

Usage:
  import pimakerlibrary as pimaker
  pimaker.start_gesture_controller(mode="presentation")
  pimaker.start_gesture_controller(mode="media")
"""

import os
import cv2
import time
import math
import numpy as np
from collections import deque

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise ImportError(
        "mediapipe is required for start_gesture_controller(). "
        "Install with: pip install mediapipe"
    ) from exc

try:
    import pyautogui
except ImportError as exc:
    raise ImportError(
        "pyautogui is required for start_gesture_controller(). "
        "Install with: pip install pyautogui"
    ) from exc

from .vision.fingertip_backends import _ensure_model_file

# ── Try to import pycaw for smooth Windows volume control ──────────────
_HAS_PYCAW = False
_volume_interface = None

def _init_pycaw():
    """Lazily initialize pycaw volume interface."""
    global _HAS_PYCAW, _volume_interface
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
        _HAS_PYCAW = True
    except Exception:
        _HAS_PYCAW = False
        _volume_interface = None


def _set_system_volume(level):
    """Set system volume (0.0 to 1.0). Uses pycaw on Windows, falls back to pyautogui."""
    level = max(0.0, min(1.0, level))
    if _HAS_PYCAW and _volume_interface is not None:
        _volume_interface.SetMasterVolumeLevelScalar(level, None)
    else:
        # Fallback: simulate volume keys (coarse)
        current_approx = 0.5  # We can't read volume without pycaw, so just nudge
        if level > 0.55:
            pyautogui.press("volumeup")
        elif level < 0.45:
            pyautogui.press("volumedown")


def _get_system_volume():
    """Get current system volume (0.0 to 1.0). Returns None if unavailable."""
    if _HAS_PYCAW and _volume_interface is not None:
        return _volume_interface.GetMasterVolumeLevelScalar()
    return None


def _toggle_mute():
    """Toggle system mute."""
    if _HAS_PYCAW and _volume_interface is not None:
        current_mute = _volume_interface.GetMute()
        _volume_interface.SetMute(not current_mute, None)
    else:
        pyautogui.press("volumemute")


def _is_muted():
    """Check if system is muted. Returns None if unavailable."""
    if _HAS_PYCAW and _volume_interface is not None:
        return bool(_volume_interface.GetMute())
    return None


# ── Gesture Detection Helpers ──────────────────────────────────────────

def _count_extended_fingers(hand_landmarks, handedness="Right"):
    """
    Count how many fingers are extended.
    Returns (count, list_of_booleans) for [thumb, index, middle, ring, pinky].
    """
    tips = [4, 8, 12, 16, 20]    # Tip landmarks
    pips = [3, 6, 10, 14, 18]    # PIP / IP joints (for comparison)

    extended = []

    for i, (tip_id, pip_id) in enumerate(zip(tips, pips)):
        tip = hand_landmarks[tip_id]
        pip_joint = hand_landmarks[pip_id]

        if i == 0:  # Thumb — uses x-axis comparison
            if handedness == "Right":
                extended.append(tip.x > pip_joint.x)
            else:
                extended.append(tip.x < pip_joint.x)
        else:  # Other fingers — tip above pip means extended (y is inverted)
            extended.append(tip.y < pip_joint.y)

    return sum(extended), extended


def _get_hand_center(hand_landmarks, w, h):
    """Get the center point of the palm (average of wrist + MCP joints)."""
    # Use wrist (0) and all MCP joints (5, 9, 13, 17) for a stable center
    ids = [0, 5, 9, 13, 17]
    cx = sum(hand_landmarks[i].x for i in ids) / len(ids)
    cy = sum(hand_landmarks[i].y for i in ids) / len(ids)
    return cx * w, cy * h


# ── HUD Overlay Renderer ──────────────────────────────────────────────

class HUDRenderer:
    """Renders a beautiful heads-up display overlay on the camera frame."""

    def __init__(self):
        self._action_text = ""
        self._action_icon = ""
        self._action_time = 0
        self._action_fade_duration = 1.5  # seconds

        # Color palette (BGR)
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_ACCENT = (255, 180, 50)       # Warm amber
        self.COLOR_SUCCESS = (100, 220, 100)      # Green
        self.COLOR_VOLUME_BAR = (255, 140, 50)    # Orange
        self.COLOR_VOLUME_BG = (60, 60, 60)       # Dark gray
        self.COLOR_MUTE = (80, 80, 255)           # Red-ish
        self.COLOR_TEXT = (240, 240, 240)          # White
        self.COLOR_SUBTLE = (140, 140, 140)        # Gray

    def show_action(self, text, icon=""):
        """Trigger an action notification with fade-out."""
        self._action_text = text
        self._action_icon = icon
        self._action_time = time.time()

    def render(self, frame, mode, gesture_name, volume_level, is_muted, num_hands):
        """Render the full HUD onto the frame."""
        h, w, _ = frame.shape
        overlay = frame.copy()

        # ── Top bar: Mode indicator ──
        bar_h = 45
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.COLOR_BG, -1)
        mode_label = "PRESENTATION MODE" if mode == "presentation" else "MEDIA MODE"
        mode_icon = "[ Slides ]" if mode == "presentation" else "[ Music ]"
        cv2.putText(overlay, f"{mode_icon}  {mode_label}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_ACCENT, 2)
        
        # Hands detected indicator
        hand_text = f"Hands: {num_hands}"
        hand_color = self.COLOR_SUCCESS if num_hands > 0 else self.COLOR_SUBTLE
        cv2.putText(overlay, hand_text, (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

        # ── Bottom bar: Current gesture + volume ──
        bot_y = h - 70
        cv2.rectangle(overlay, (0, bot_y), (w, h), self.COLOR_BG, -1)

        # Gesture label
        if gesture_name:
            cv2.putText(overlay, f"Gesture: {gesture_name}", (15, h - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_SUCCESS, 2)
        else:
            cv2.putText(overlay, "Gesture: None", (15, h - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_SUBTLE, 2)

        # Volume bar
        vol_bar_x = w - 280
        vol_bar_y = h - 55
        vol_bar_w = 200
        vol_bar_h = 18

        if volume_level is not None:
            # Background
            cv2.rectangle(overlay, (vol_bar_x, vol_bar_y),
                          (vol_bar_x + vol_bar_w, vol_bar_y + vol_bar_h),
                          self.COLOR_VOLUME_BG, -1)
            # Rounded corners (approximate with filled rect)
            fill_w = int(vol_bar_w * volume_level)
            bar_color = self.COLOR_MUTE if is_muted else self.COLOR_VOLUME_BAR
            if fill_w > 0:
                cv2.rectangle(overlay, (vol_bar_x, vol_bar_y),
                              (vol_bar_x + fill_w, vol_bar_y + vol_bar_h),
                              bar_color, -1)
            # Border
            cv2.rectangle(overlay, (vol_bar_x, vol_bar_y),
                          (vol_bar_x + vol_bar_w, vol_bar_y + vol_bar_h),
                          self.COLOR_TEXT, 1)

            vol_pct = int(volume_level * 100)
            mute_label = "MUTED" if is_muted else f"Vol: {vol_pct}%"
            cv2.putText(overlay, mute_label, (vol_bar_x, vol_bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        self.COLOR_MUTE if is_muted else self.COLOR_TEXT, 1)

        # ── Quit hint ──
        cv2.putText(overlay, "Press 'q' to quit", (w // 2 - 75, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_SUBTLE, 1)

        # Blend the bars
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # ── Center action notification (with fade) ──
        elapsed = time.time() - self._action_time
        if self._action_text and elapsed < self._action_fade_duration:
            alpha = 1.0 - (elapsed / self._action_fade_duration)
            alpha = max(0.0, min(1.0, alpha))

            text = f"{self._action_icon}  {self._action_text}"
            font_scale = 1.3
            thickness = 3
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

            # Background pill
            pad_x, pad_y = 30, 20
            cx = w // 2
            cy = h // 2 - 30
            x1 = cx - text_size[0] // 2 - pad_x
            y1 = cy - text_size[1] // 2 - pad_y
            x2 = cx + text_size[0] // 2 + pad_x
            y2 = cy + text_size[1] // 2 + pad_y

            action_overlay = frame.copy()
            cv2.rectangle(action_overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
            cv2.rectangle(action_overlay, (x1, y1), (x2, y2), self.COLOR_ACCENT, 2)
            tx = cx - text_size[0] // 2
            ty = cy + text_size[1] // 2
            cv2.putText(action_overlay, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.COLOR_ACCENT, thickness)
            cv2.addWeighted(action_overlay, alpha, frame, 1, 0, frame)

        return frame


# ── Main Controller ───────────────────────────────────────────────────

def start_gesture_controller(mode="presentation"):
    """
    Start the Gesture-Controlled Media & Presentation Controller.

    Args:
        mode: "presentation" or "media"
            - presentation: Swipes send Left/Right arrow keys, palm sends Space
            - media: Swipes send media Next/Prev keys, palm sends Play/Pause media key

    Gestures:
        - Swipe Right:  Next Slide / Skip Track
        - Swipe Left:   Previous Slide / Previous Track
        - Open Palm:    Play / Pause (hold still, all 5 fingers extended)
        - Pinch:        Mute / Unmute Toggle (thumb + index close)
        - Two Hands:    Volume Control (spread index fingers apart for louder)

    Press 'q' to quit.
    """
    if mode not in ("presentation", "media"):
        raise ValueError("mode must be 'presentation' or 'media'")

    # ── Initialize volume control ──
    _init_pycaw()

    # ── Setup MediaPipe Hand Landmarker ──
    package_root = os.path.dirname(__file__)
    model_path = _ensure_model_file(package_root)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera")

    pyautogui.PAUSE = 0  # No delay between pyautogui actions

    # ── State ──
    hud = HUDRenderer()

    # Swipe detection
    SWIPE_HISTORY_LEN = 10
    SWIPE_THRESHOLD_RATIO = 0.30    # 30% of frame width
    SWIPE_COOLDOWN = 0.8            # seconds between swipes

    hand_x_history = deque(maxlen=SWIPE_HISTORY_LEN)
    last_swipe_time = 0

    # Open palm (play/pause)
    PALM_HOLD_FRAMES = 12           # Must hold open palm for this many consecutive frames
    PALM_COOLDOWN = 1.5             # seconds
    palm_hold_counter = 0
    last_palm_time = 0

    # Pinch (mute)
    PINCH_THRESHOLD = 0.045         # Normalized distance (thumb tip to index tip)
    PINCH_COOLDOWN = 1.0            # seconds
    is_pinching = False
    last_pinch_time = 0

    # Volume
    VOLUME_MIN_DIST = 0.08          # Normalized - hands very close
    VOLUME_MAX_DIST = 0.55          # Normalized - hands far apart
    volume_smooth = _get_system_volume() or 0.5

    last_timestamp_ms = -1
    current_gesture_name = ""

    print("=" * 55)
    print("  Gesture Controller Started!")
    print(f"  Mode: {mode.upper()}")
    print("=" * 55)
    print("  Gestures:")
    print("    Swipe Right  → Next Slide / Skip Track")
    print("    Swipe Left   → Prev Slide / Prev Track")
    print("    Open Palm    → Play / Pause")
    print("    Pinch        → Mute / Unmute")
    print("    Two Hands    → Volume Control")
    print("  Press 'q' to quit.")
    print("=" * 55)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            now = time.time()
            current_gesture_name = ""
            num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
            volume_level = _get_system_volume()
            is_muted = _is_muted()

            if result.hand_landmarks:
                # ── Draw hand landmarks on frame ──
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    # Draw connections for visual feedback
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
                        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
                        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                        (9, 13), (13, 14), (14, 15), (15, 16),# Ring
                        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
                        (0, 17)                                 # Palm base
                    ]
                    for c1, c2 in connections:
                        p1 = (int(hand_landmarks[c1].x * w), int(hand_landmarks[c1].y * h))
                        p2 = (int(hand_landmarks[c2].x * w), int(hand_landmarks[c2].y * h))
                        cv2.line(frame, p1, p2, (100, 255, 200), 2)

                    # Draw fingertip circles
                    for tip_id in [4, 8, 12, 16, 20]:
                        px = int(hand_landmarks[tip_id].x * w)
                        py = int(hand_landmarks[tip_id].y * h)
                        cv2.circle(frame, (px, py), 8, (0, 255, 255), -1)
                        cv2.circle(frame, (px, py), 8, (0, 0, 0), 2)

                # ── Determine handedness ──
                # MediaPipe returns handedness per detected hand
                handedness_list = []
                if result.handedness:
                    for hand_h in result.handedness:
                        # hand_h is a list of Category objects
                        label = hand_h[0].category_name if hand_h else "Right"
                        # MediaPipe returns the label as seen from the camera,
                        # but since we flip the frame, we invert:
                        label = "Left" if label == "Right" else "Right"
                        handedness_list.append(label)
                else:
                    handedness_list = ["Right"] * num_hands

                primary_hand = result.hand_landmarks[0]
                primary_handedness = handedness_list[0] if handedness_list else "Right"

                # ══════════════════════════════════════════════════════
                # GESTURE 1: SWIPE DETECTION (single hand)
                # ══════════════════════════════════════════════════════
                if num_hands == 1:
                    cx, cy = _get_hand_center(primary_hand, w, h)
                    hand_x_history.append(cx)

                    if len(hand_x_history) >= SWIPE_HISTORY_LEN:
                        x_start = hand_x_history[0]
                        x_end = hand_x_history[-1]
                        delta = x_end - x_start
                        threshold = w * SWIPE_THRESHOLD_RATIO

                        if abs(delta) > threshold and (now - last_swipe_time) > SWIPE_COOLDOWN:
                            finger_count, _ = _count_extended_fingers(primary_hand, primary_handedness)
                            # Only trigger swipe when hand is relatively open (3+ fingers)
                            if finger_count >= 3:
                                if delta > 0:
                                    # Swipe RIGHT → Next
                                    if mode == "presentation":
                                        pyautogui.press("right")
                                    else:
                                        pyautogui.press("nexttrack")
                                    hud.show_action("Next", ">>")
                                    current_gesture_name = "Swipe Right"
                                else:
                                    # Swipe LEFT → Previous
                                    if mode == "presentation":
                                        pyautogui.press("left")
                                    else:
                                        pyautogui.press("prevtrack")
                                    hud.show_action("Previous", "<<")
                                    current_gesture_name = "Swipe Left"

                                last_swipe_time = now
                                hand_x_history.clear()

                    # ══════════════════════════════════════════════════
                    # GESTURE 2: OPEN PALM → PLAY / PAUSE
                    # ══════════════════════════════════════════════════
                    finger_count, _ = _count_extended_fingers(primary_hand, primary_handedness)

                    if finger_count == 5:
                        # Check hand is relatively stationary (low velocity)
                        if len(hand_x_history) >= 3:
                            recent_delta = abs(hand_x_history[-1] - hand_x_history[-3]) if len(hand_x_history) >= 3 else 999
                            if recent_delta < w * 0.05:  # Very little movement
                                palm_hold_counter += 1
                            else:
                                palm_hold_counter = max(0, palm_hold_counter - 2)
                        else:
                            palm_hold_counter += 1

                        if palm_hold_counter >= PALM_HOLD_FRAMES and (now - last_palm_time) > PALM_COOLDOWN:
                            if mode == "presentation":
                                pyautogui.press("space")
                            else:
                                pyautogui.press("playpause")
                            hud.show_action("Play / Pause", "||")
                            current_gesture_name = "Open Palm"
                            last_palm_time = now
                            palm_hold_counter = 0
                        elif palm_hold_counter > 0:
                            current_gesture_name = f"Hold Palm... ({palm_hold_counter}/{PALM_HOLD_FRAMES})"
                    else:
                        palm_hold_counter = max(0, palm_hold_counter - 1)

                    # ══════════════════════════════════════════════════
                    # GESTURE 3: PINCH → MUTE / UNMUTE
                    # ══════════════════════════════════════════════════
                    thumb_tip = primary_hand[4]
                    index_tip = primary_hand[8]
                    pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

                    if pinch_dist < PINCH_THRESHOLD:
                        if not is_pinching and (now - last_pinch_time) > PINCH_COOLDOWN:
                            _toggle_mute()
                            new_mute_state = _is_muted()
                            mute_label = "Muted" if new_mute_state else "Unmuted"
                            hud.show_action(mute_label, "M")
                            current_gesture_name = "Pinch (Mute)"
                            last_pinch_time = now
                            is_pinching = True

                        # Draw pinch indicator
                        mid_x = int((thumb_tip.x + index_tip.x) / 2 * w)
                        mid_y = int((thumb_tip.y + index_tip.y) / 2 * h)
                        cv2.circle(frame, (mid_x, mid_y), 20, (80, 80, 255), -1)
                        cv2.circle(frame, (mid_x, mid_y), 20, (255, 255, 255), 2)
                        if not current_gesture_name:
                            current_gesture_name = "Pinch"
                    else:
                        is_pinching = False

                # ══════════════════════════════════════════════════════
                # GESTURE 4: TWO HANDS → VOLUME CONTROL
                # ══════════════════════════════════════════════════════
                if num_hands == 2:
                    hand_a = result.hand_landmarks[0]
                    hand_b = result.hand_landmarks[1]

                    # Use index fingertips of both hands
                    tip_a = hand_a[8]
                    tip_b = hand_b[8]

                    dist = math.hypot(tip_a.x - tip_b.x, tip_a.y - tip_b.y)

                    # Map distance to volume level
                    raw_vol = np.interp(dist, [VOLUME_MIN_DIST, VOLUME_MAX_DIST], [0.0, 1.0])
                    raw_vol = float(np.clip(raw_vol, 0.0, 1.0))

                    # Smooth the volume change
                    volume_smooth = volume_smooth * 0.7 + raw_vol * 0.3
                    _set_system_volume(volume_smooth)

                    current_gesture_name = f"Volume: {int(volume_smooth * 100)}%"

                    # Draw a line between the two index fingertips
                    p1 = (int(tip_a.x * w), int(tip_a.y * h))
                    p2 = (int(tip_b.x * w), int(tip_b.y * h))
                    line_color = (
                        int(np.interp(volume_smooth, [0, 1], [80, 50])),
                        int(np.interp(volume_smooth, [0, 1], [80, 220])),
                        int(np.interp(volume_smooth, [0, 1], [255, 100]))
                    )
                    cv2.line(frame, p1, p2, line_color, 4)

                    # Draw volume percentage at the midpoint
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 25)
                    cv2.putText(frame, f"{int(volume_smooth * 100)}%", mid,
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

                    # Clear swipe history when using two hands (prevent accidental swipes)
                    hand_x_history.clear()
                    palm_hold_counter = 0

            else:
                # No hands → reset state
                hand_x_history.clear()
                palm_hold_counter = max(0, palm_hold_counter - 1)
                is_pinching = False

            # ── Render HUD ──
            frame = hud.render(frame, mode, current_gesture_name,
                               volume_level, is_muted, num_hands)

            cv2.imshow("PiMaker Gesture Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Gesture Controller stopped.")
