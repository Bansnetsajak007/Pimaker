"""
PiMaker Rock-Paper-Scissors — play against the computer using hand gestures!

Gestures:
  ✊ Fist         = Rock
  ✋ Open Hand    = Paper  
  ✌️ Peace Sign   = Scissors

How to play:
  1. Show your hand to the camera
  2. Hold a gesture steady for 2 seconds (countdown shown on screen)
  3. The computer picks its move — winner is announced!
  4. First to the target score wins the match

Controls:
  Q — quit
  R — reset score

Usage:
  pimaker.play_rock_paper_scissors()
  pimaker.play_rock_paper_scissors(win_score=5)
"""

import os
import cv2
import time
import math
import random
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise ImportError(
        "mediapipe is required for play_rock_paper_scissors(). "
        "Install with: pip install mediapipe"
    ) from exc

from .vision.fingertip_backends import _ensure_model_file

# ─── Gesture Detection ───────────────────────────────────────────────

def _classify_rps_gesture(landmarks):
    """
    Classify a hand landmark list into Rock, Paper, or Scissors.
    Returns: "rock", "paper", "scissors", or None if ambiguous.
    """
    if len(landmarks) < 21:
        return None

    wrist = landmarks[0]

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    # Check which fingers are extended (tip farther from wrist than PIP joint)
    index_open = dist(landmarks[8], wrist) > dist(landmarks[6], wrist)
    middle_open = dist(landmarks[12], wrist) > dist(landmarks[10], wrist)
    ring_open = dist(landmarks[16], wrist) > dist(landmarks[14], wrist)
    pinky_open = dist(landmarks[20], wrist) > dist(landmarks[18], wrist)
    thumb_open = dist(landmarks[4], landmarks[17]) > dist(landmarks[3], landmarks[17])

    open_count = sum([index_open, middle_open, ring_open, pinky_open, thumb_open])

    # ✊ ROCK — all fingers closed (fist)
    if open_count <= 1 and not index_open and not middle_open:
        return "rock"

    # ✋ PAPER — all fingers open
    if open_count >= 4:
        return "paper"

    # ✌️ SCISSORS — only index + middle open
    if index_open and middle_open and not ring_open and not pinky_open:
        return "scissors"

    return None


# ─── Game Logic ───────────────────────────────────────────────────────

MOVES = ["rock", "paper", "scissors"]
EMOJI = {"rock": "ROCK", "paper": "PAPER", "scissors": "SCISSORS"}
BEAT_MAP = {"rock": "scissors", "paper": "rock", "scissors": "paper"}

MOVE_COLORS = {
    "rock": (80, 80, 220),       # Red-ish
    "paper": (220, 180, 60),     # Blue-ish
    "scissors": (60, 200, 60),   # Green-ish
}


def _determine_winner(player, computer):
    """Returns 'player', 'computer', or 'draw'."""
    if player == computer:
        return "draw"
    if BEAT_MAP[player] == computer:
        return "player"
    return "computer"


# ─── Drawing Helpers ──────────────────────────────────────────────────

def _draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=15):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # Fill the rectangle body
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)

    # Draw corner circles
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)


def _draw_scoreboard(frame, player_score, computer_score, win_score):
    """Draw the scoreboard at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Scoreboard background
    _draw_rounded_rect(overlay, (w // 2 - 200, 8), (w // 2 + 200, 70), (30, 30, 30), -1, 12)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Player score (left, green)
    cv2.putText(frame, f"YOU: {player_score}", (w // 2 - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

    # vs
    cv2.putText(frame, "vs", (w // 2 - 15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Computer score (right, red)
    cv2.putText(frame, f"CPU: {computer_score}", (w // 2 + 50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

    # Win target
    cv2.putText(frame, f"First to {win_score}", (w // 2 - 55, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)


def _draw_countdown_bar(frame, progress, gesture_name):
    """Draw a countdown progress bar while the player holds a gesture."""
    h, w = frame.shape[:2]
    bar_w = 300
    bar_h = 25
    bar_x = (w - bar_w) // 2
    bar_y = h - 80

    color = MOVE_COLORS.get(gesture_name, (200, 200, 200))

    overlay = frame.copy()
    # Background bar
    _draw_rounded_rect(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1, 10)
    # Progress fill
    fill_w = int(bar_w * min(progress, 1.0))
    if fill_w > 5:
        _draw_rounded_rect(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1, 10)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Label
    label = f"Hold {gesture_name.upper()}..."
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, label, ((w - tw) // 2, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_result_splash(frame, player_move, computer_move, result, alpha=0.6):
    """Draw the round result as a big center splash."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent dark overlay
    cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3 + 30), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Player move (left)
    p_color = MOVE_COLORS.get(player_move, (255, 255, 255))
    cv2.putText(frame, f"YOU: {player_move.upper()}", (40, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, p_color, 3)

    # Computer move (right)
    c_color = MOVE_COLORS.get(computer_move, (255, 255, 255))
    cv2.putText(frame, f"CPU: {computer_move.upper()}", (w - 340, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, c_color, 3)

    # "VS" in center
    cv2.putText(frame, "VS", (w // 2 - 25, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Result text
    if result == "player":
        msg = "YOU WIN THIS ROUND!"
        msg_color = (100, 255, 100)
    elif result == "computer":
        msg = "COMPUTER WINS THIS ROUND!"
        msg_color = (100, 100, 255)
    else:
        msg = "IT'S A DRAW!"
        msg_color = (200, 200, 200)

    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, msg, ((w - tw) // 2, h // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, msg_color, 2)


def _draw_match_winner(frame, winner):
    """Draw the final match winner screen."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if winner == "player":
        line1 = "CONGRATULATIONS!"
        line2 = "YOU WON THE MATCH!"
        color = (100, 255, 100)
    else:
        line1 = "GAME OVER"
        line2 = "COMPUTER WINS THE MATCH!"
        color = (100, 100, 255)

    (tw1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    (tw2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(frame, line1, ((w - tw1) // 2, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, line2, ((w - tw2) // 2, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(frame, "Press R to play again  |  Q to quit", (w // 2 - 220, h // 2 + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def _draw_gesture_hint(frame):
    """Draw gesture guide at the bottom."""
    h, w = frame.shape[:2]
    hints = [
        ("ROCK = Fist", MOVE_COLORS["rock"]),
        ("PAPER = Open Hand", MOVE_COLORS["paper"]),
        ("SCISSORS = Peace", MOVE_COLORS["scissors"]),
    ]
    y = h - 20
    x_start = 20
    for i, (text, color) in enumerate(hints):
        x = x_start + i * (w // 3)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ─── Main Game Loop ──────────────────────────────────────────────────

def play_rock_paper_scissors(win_score=3, hold_time=2.0):
    """
    ✊✋✌️ Rock-Paper-Scissors — play against the computer with hand gestures!

    Show your hand gesture (rock/paper/scissors) and hold it steady.
    A countdown bar fills up — when it's full, the computer plays!
    First to `win_score` wins the match.

    Args:
        win_score (int): Points needed to win the match (default: 3).
        hold_time (float): Seconds to hold a gesture before it locks in (default: 2.0).

    Controls:
        Q — quit
        R — reset / play again
    """
    package_root = os.path.dirname(__file__)
    model_path = _ensure_model_file(package_root)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception(
            "🎥 Oops! Can't find your camera. "
            "Make sure it's plugged in and no other app is using it!"
        )

    print("\n✊✋✌️  PiMaker Rock-Paper-Scissors!")
    print(f"   First to {win_score} wins.")
    print("   Show Rock, Paper, or Scissors and hold steady!")
    print("   Press 'Q' to quit, 'R' to reset.\n")

    player_score = 0
    computer_score = 0

    # State machine
    STATE_WAITING = 0     # Waiting for player to show a gesture
    STATE_HOLDING = 1     # Player is holding a gesture, countdown active
    STATE_RESULT = 2      # Showing round result
    STATE_MATCH_OVER = 3  # Match is complete

    state = STATE_WAITING
    current_gesture = None
    hold_start = 0
    result_start = 0
    last_player_move = None
    last_computer_move = None
    last_result = None
    match_winner = None

    last_timestamp_ms = -1

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Process hand landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Detect gesture
            detected_gesture = None
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    # Draw hand outline
                    for lm in hand_landmarks:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (px, py), 3, (0, 200, 200), -1)

                    detected_gesture = _classify_rps_gesture(hand_landmarks)
                    break  # only use first hand

            # ─── State Machine ────────────────────────────────────

            if state == STATE_WAITING:
                if detected_gesture:
                    current_gesture = detected_gesture
                    hold_start = time.time()
                    state = STATE_HOLDING

                # Show instruction
                cv2.putText(frame, "Show Rock, Paper, or Scissors!", (w // 2 - 230, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            elif state == STATE_HOLDING:
                elapsed = time.time() - hold_start
                progress = elapsed / hold_time

                if detected_gesture != current_gesture:
                    # Gesture changed or lost — reset
                    if detected_gesture is not None:
                        current_gesture = detected_gesture
                        hold_start = time.time()
                    else:
                        state = STATE_WAITING
                        current_gesture = None
                elif progress >= 1.0:
                    # Gesture held long enough! Lock in and play
                    last_player_move = current_gesture
                    last_computer_move = random.choice(MOVES)
                    last_result = _determine_winner(last_player_move, last_computer_move)

                    if last_result == "player":
                        player_score += 1
                    elif last_result == "computer":
                        computer_score += 1

                    result_start = time.time()
                    state = STATE_RESULT
                    current_gesture = None
                else:
                    # Still holding — draw the countdown bar
                    _draw_countdown_bar(frame, progress, current_gesture)

                    # Show current detected gesture
                    g_color = MOVE_COLORS.get(current_gesture, (255, 255, 255))
                    cv2.putText(frame, current_gesture.upper(), (w // 2 - 60, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, g_color, 3)

            elif state == STATE_RESULT:
                _draw_result_splash(frame, last_player_move, last_computer_move, last_result)

                # Check for match winner
                if player_score >= win_score:
                    match_winner = "player"
                    state = STATE_MATCH_OVER
                elif computer_score >= win_score:
                    match_winner = "computer"
                    state = STATE_MATCH_OVER
                elif time.time() - result_start > 2.5:
                    # Show result for 2.5 seconds, then go back to waiting
                    state = STATE_WAITING

            elif state == STATE_MATCH_OVER:
                _draw_match_winner(frame, match_winner)

            # Always draw scoreboard and gesture hints
            _draw_scoreboard(frame, player_score, computer_score, win_score)
            if state != STATE_MATCH_OVER:
                _draw_gesture_hint(frame)

            cv2.imshow("PiMaker Rock-Paper-Scissors", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                # Reset match
                player_score = 0
                computer_score = 0
                state = STATE_WAITING
                current_gesture = None
                match_winner = None
                print("[PiMaker] Match reset!")

    cap.release()
    cv2.destroyAllWindows()
    print("[PiMaker] Rock-Paper-Scissors stopped.")
