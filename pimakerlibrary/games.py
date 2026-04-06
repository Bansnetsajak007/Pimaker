"""
PiMaker Games — Fruit Ninja & Balloon Pop!

Usage:
  pimaker.play_game("fruit ninja")
  pimaker.play_game("balloon pop")

Fruit Ninja:
  - Fruits launch from the bottom in arcs
  - Swipe your hand through them to slice!
  - Don't let fruits fall off screen (you lose lives)
  - Avoid the BOMBS 💣

Balloon Pop:
  - Colorful balloons float upward
  - Poke them with your finger to pop!
  - Pop as many as you can before time runs out

Controls:
  Q — quit
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
        "mediapipe is required for play_game(). "
        "Install with: pip install mediapipe"
    ) from exc

from .vision.fingertip_backends import _ensure_model_file


# ═══════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _setup_hand_tracker(package_root):
    """Create and return a HandLandmarker and VideoCapture."""
    model_path = _ensure_model_file(package_root)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def _draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=12):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)


# ═══════════════════════════════════════════════════════════════════════
#  FRUIT NINJA 🍉
# ═══════════════════════════════════════════════════════════════════════

FRUITS = [
    {"name": "Watermelon", "color": (50, 180, 50), "inner": (60, 60, 220), "radius": 40, "points": 1},
    {"name": "Orange",     "color": (0, 140, 255), "inner": (0, 180, 255),  "radius": 32, "points": 1},
    {"name": "Apple",      "color": (0, 0, 200),   "inner": (200, 255, 200),"radius": 30, "points": 1},
    {"name": "Banana",     "color": (0, 220, 255), "inner": (220, 255, 255),"radius": 28, "points": 2},
    {"name": "Grape",      "color": (180, 0, 180), "inner": (255, 100, 255),"radius": 22, "points": 2},
    {"name": "Mango",      "color": (0, 200, 255), "inner": (50, 255, 255), "radius": 34, "points": 1},
]

BOMB_COLOR = (40, 40, 40)
BOMB_INNER = (0, 0, 180)
BOMB_RADIUS = 35


class FruitObject:
    """A fruit or bomb projectile."""

    def __init__(self, frame_w, frame_h, is_bomb=False):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.is_bomb = is_bomb

        if is_bomb:
            self.name = "BOMB"
            self.color = BOMB_COLOR
            self.inner_color = BOMB_INNER
            self.radius = BOMB_RADIUS
            self.points = -1
        else:
            fruit = random.choice(FRUITS)
            self.name = fruit["name"]
            self.color = fruit["color"]
            self.inner_color = fruit["inner"]
            self.radius = fruit["radius"]
            self.points = fruit["points"]

        # Launch from random position at the bottom
        self.x = random.randint(self.radius + 50, frame_w - self.radius - 50)
        self.y = frame_h + self.radius

        # Upward velocity with some horizontal drift
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-18, -13)

        self.gravity = 0.35
        self.alive = True
        self.sliced = False
        self.slice_time = 0
        self.rotation = random.uniform(0, 360)
        self.rot_speed = random.uniform(-5, 5)

    def update(self):
        """Update position with physics."""
        if self.sliced:
            return

        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.rotation += self.rot_speed

        # Off screen (fell down past bottom)
        if self.y > self.frame_h + self.radius * 2:
            self.alive = False

    def draw(self, frame):
        """Draw the fruit or bomb."""
        ix, iy = int(self.x), int(self.y)

        if self.sliced:
            # Slice animation — two halves splitting apart
            elapsed = time.time() - self.slice_time
            if elapsed > 0.5:
                self.alive = False
                return

            alpha = elapsed / 0.5
            offset = int(30 * alpha)
            fade = max(0, 1.0 - alpha)

            # Left half
            overlay = frame.copy()
            cv2.circle(overlay, (ix - offset, iy + int(20 * alpha)), self.radius,
                       self.inner_color, -1)
            cv2.addWeighted(overlay, fade * 0.8, frame, 1 - fade * 0.8 + 0.2, 0, frame)

            # Right half
            overlay2 = frame.copy()
            cv2.circle(overlay2, (ix + offset, iy + int(25 * alpha)), self.radius,
                       self.inner_color, -1)
            cv2.addWeighted(overlay2, fade * 0.8, frame, 1 - fade * 0.8 + 0.2, 0, frame)

            # Juice particles
            for _ in range(3):
                px = ix + random.randint(-40, 40)
                py = iy + random.randint(-20, 30)
                cv2.circle(frame, (px, py), random.randint(2, 5), self.inner_color, -1)
            return

        # Draw shadow
        cv2.circle(frame, (ix + 3, iy + 3), self.radius, (30, 30, 30), -1)

        # Outer skin
        cv2.circle(frame, (ix, iy), self.radius, self.color, -1)

        # Inner highlight
        cv2.circle(frame, (ix, iy), self.radius - 6, self.inner_color, -1)

        # Shine spot
        cv2.circle(frame, (ix - self.radius // 3, iy - self.radius // 3),
                   self.radius // 4, (255, 255, 255), -1)

        if self.is_bomb:
            # Draw fuse
            cv2.line(frame, (ix, iy - self.radius),
                     (ix + 10, iy - self.radius - 15), (100, 100, 100), 3)
            # Spark
            cv2.circle(frame, (ix + 10, iy - self.radius - 15), 5, (0, 200, 255), -1)
            # X label
            cv2.putText(frame, "X", (ix - 10, iy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Label
            (tw, th), _ = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.putText(frame, self.name, (ix - tw // 2, iy + self.radius + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    def check_hit(self, hx, hy, hand_speed):
        """Check if the hand position hits this fruit."""
        if self.sliced or not self.alive:
            return False
        dist = math.hypot(hx - self.x, hy - self.y)
        # For fruit ninja: need sufficient speed to slice
        return dist < self.radius + 25 and hand_speed > 8

    def slice(self):
        """Mark this fruit as sliced."""
        self.sliced = True
        self.slice_time = time.time()


class SlashTrail:
    """Visual slash trail that follows the hand."""

    def __init__(self, max_points=15):
        self.points = []
        self.max_points = max_points

    def add(self, x, y):
        self.points.append((int(x), int(y), time.time()))
        if len(self.points) > self.max_points:
            self.points.pop(0)

    def draw(self, frame):
        """Draw a fading slash trail."""
        now = time.time()
        valid = [(x, y, t) for x, y, t in self.points if now - t < 0.3]
        self.points = valid

        for i in range(1, len(valid)):
            age = now - valid[i][2]
            alpha = max(0, 1.0 - age / 0.3)
            thickness = max(1, int(8 * alpha))
            b = int(200 * alpha)
            g = int(255 * alpha)

            cv2.line(frame,
                     (valid[i - 1][0], valid[i - 1][1]),
                     (valid[i][0], valid[i][1]),
                     (b, g, 255), thickness)


def _play_fruit_ninja():
    """Main Fruit Ninja game loop."""
    package_root = os.path.dirname(__file__)
    landmarker = _setup_hand_tracker(package_root)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("🎥 Can't find your camera!")

    print("\n🍉 PiMaker FRUIT NINJA!")
    print("   Swipe your hand to slice the fruits!")
    print("   Avoid the BOMBS 💣")
    print("   Press 'Q' to quit.\n")

    score = 0
    lives = 3
    combo = 0
    best_combo = 0
    fruits = []
    slash_trail = SlashTrail()
    game_over = False
    game_over_time = 0

    # Hand tracking state
    prev_hx, prev_hy = 0, 0
    hand_speed = 0

    # Spawn timing
    last_spawn = time.time()
    spawn_interval = 1.5
    min_spawn_interval = 0.6
    last_timestamp_ms = -1
    start_time = time.time()

    # Score popups
    popups = []  # list of (text, x, y, spawn_time, color)

    with landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # ── Hand Tracking ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hx, hy = -100, -100
            if result.hand_landmarks and not game_over:
                lm = result.hand_landmarks[0]
                # Use index fingertip (landmark 8)
                tip = lm[8]
                hx, hy = int(tip.x * w), int(tip.y * h)

                # Calculate hand speed
                if prev_hx > 0:
                    hand_speed = math.hypot(hx - prev_hx, hy - prev_hy)
                else:
                    hand_speed = 0

                prev_hx, prev_hy = hx, hy

                # Draw hand cursor
                cv2.circle(frame, (hx, hy), 18, (0, 255, 255), 3)
                cv2.circle(frame, (hx, hy), 6, (255, 255, 255), -1)

                # Add to slash trail if moving fast
                if hand_speed > 5:
                    slash_trail.add(hx, hy)

            # ── Game Logic ──
            if not game_over:
                # Spawn fruits
                now = time.time()
                elapsed = now - start_time

                # Speed up spawning over time
                current_interval = max(min_spawn_interval,
                                       spawn_interval - (elapsed / 60.0) * 0.5)

                if now - last_spawn > current_interval:
                    # Spawn 1-3 fruits
                    count = random.randint(1, min(3, 1 + int(elapsed / 30)))
                    for _ in range(count):
                        is_bomb = random.random() < 0.15  # 15% bomb chance
                        fruits.append(FruitObject(w, h, is_bomb=is_bomb))
                    last_spawn = now

                # Update fruits
                sliced_this_frame = 0
                for fruit in fruits:
                    fruit.update()

                    # Check for slicing
                    if not fruit.sliced and fruit.alive and hand_speed > 0:
                        if fruit.check_hit(hx, hy, hand_speed):
                            fruit.slice()
                            if fruit.is_bomb:
                                # Hit a bomb! Lose a life!
                                lives -= 1
                                combo = 0
                                popups.append(("BOOM! -1 ❤️", int(fruit.x), int(fruit.y),
                                               time.time(), (0, 0, 255)))
                            else:
                                sliced_this_frame += 1
                                score += fruit.points
                                combo += 1
                                if combo > best_combo:
                                    best_combo = combo
                                point_text = f"+{fruit.points}"
                                if combo >= 3:
                                    bonus = combo
                                    score += bonus
                                    point_text = f"+{fruit.points + bonus} COMBO x{combo}!"
                                popups.append((point_text, int(fruit.x), int(fruit.y),
                                               time.time(), (100, 255, 100)))

                    # Fruit missed (fell off bottom without being sliced)
                    if not fruit.alive and not fruit.sliced and not fruit.is_bomb:
                        lives -= 1
                        combo = 0

                # Remove dead fruits
                fruits = [f for f in fruits if f.alive]

                # Check game over
                if lives <= 0:
                    game_over = True
                    game_over_time = time.time()

            # ── Drawing ──

            # Draw all fruits
            for fruit in fruits:
                fruit.draw(frame)

            # Draw slash trail
            slash_trail.draw(frame)

            # Draw score popups
            now = time.time()
            active_popups = []
            for (text, px, py, t, color) in popups:
                age = now - t
                if age < 1.0:
                    alpha = 1.0 - age
                    float_y = py - int(40 * age)
                    cv2.putText(frame, text, (px - 30, float_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    active_popups.append((text, px, py, t, color))
            popups = active_popups

            # ── HUD ──
            overlay = frame.copy()

            # Score panel (top left)
            _draw_rounded_rect(overlay, (10, 8), (200, 55), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"Score: {score}", (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)

            # Lives (top right) — draw hearts
            for i in range(3):
                heart_x = w - 50 - i * 45
                heart_y = 30
                color = (0, 0, 220) if i < lives else (80, 80, 80)
                # Simple heart using circles + triangle
                cv2.circle(frame, (heart_x - 7, heart_y - 5), 10, color, -1)
                cv2.circle(frame, (heart_x + 7, heart_y - 5), 10, color, -1)
                pts = np.array([[heart_x - 17, heart_y - 2],
                                [heart_x + 17, heart_y - 2],
                                [heart_x, heart_y + 15]], np.int32)
                cv2.fillPoly(frame, [pts], color)

            # Combo counter (top center)
            if combo >= 2:
                combo_text = f"COMBO x{combo}!"
                combo_color = (0, 255, 255) if combo < 5 else (0, 165, 255) if combo < 10 else (0, 0, 255)
                (tw, _), _ = cv2.getTextSize(combo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(frame, combo_text, ((w - tw) // 2, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, combo_color, 2)

            # ── Game Over Screen ──
            if game_over:
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (0, 0), (w, h), (10, 10, 30), -1)
                cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

                # Title
                cv2.putText(frame, "GAME OVER!", (w // 2 - 140, h // 2 - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # Final score
                cv2.putText(frame, f"Final Score: {score}", (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # Best combo
                cv2.putText(frame, f"Best Combo: x{best_combo}", (w // 2 - 110, h // 2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Controls
                cv2.putText(frame, "Press R to play again | Q to quit",
                            (w // 2 - 200, h // 2 + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("PiMaker Fruit Ninja", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                # Reset
                score = 0
                lives = 3
                combo = 0
                best_combo = 0
                fruits.clear()
                popups.clear()
                game_over = False
                start_time = time.time()
                last_spawn = time.time()
                print("[PiMaker] Game restarted!")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[PiMaker] Fruit Ninja stopped. Final score: {score}")


# ═══════════════════════════════════════════════════════════════════════
#  BALLOON POP 🎈
# ═══════════════════════════════════════════════════════════════════════

BALLOON_COLORS = [
    {"body": (60, 60, 220),  "shine": (150, 150, 255), "name": "Red"},
    {"body": (220, 160, 30), "shine": (255, 210, 130), "name": "Blue"},
    {"body": (50, 200, 50),  "shine": (150, 255, 150), "name": "Green"},
    {"body": (0, 200, 255),  "shine": (130, 235, 255), "name": "Yellow"},
    {"body": (200, 50, 200), "shine": (255, 150, 255), "name": "Purple"},
    {"body": (0, 165, 255),  "shine": (100, 200, 255), "name": "Orange"},
    {"body": (255, 100, 100),"shine": (255, 180, 180), "name": "Cyan"},
]


class Balloon:
    """A floating balloon."""

    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h

        style = random.choice(BALLOON_COLORS)
        self.body_color = style["body"]
        self.shine_color = style["shine"]
        self.name = style["name"]

        self.radius_x = random.randint(28, 42)
        self.radius_y = int(self.radius_x * 1.3)

        # Start below screen at random x
        self.x = random.randint(self.radius_x + 30, frame_w - self.radius_x - 30)
        self.y = frame_h + self.radius_y + random.randint(0, 100)

        # Float speed
        self.speed = random.uniform(1.5, 3.5)
        self.wobble_amp = random.uniform(15, 30)
        self.wobble_speed = random.uniform(1.5, 3.0)
        self.wobble_offset = random.uniform(0, math.pi * 2)

        self.alive = True
        self.popped = False
        self.pop_time = 0
        self.points = 1
        self.spawn_time = time.time()

    def update(self):
        """Move upward with wobble."""
        if self.popped:
            return

        self.y -= self.speed
        elapsed = time.time() - self.spawn_time
        self.x += math.sin(elapsed * self.wobble_speed + self.wobble_offset) * 0.8

        # Off-screen top
        if self.y < -self.radius_y * 2:
            self.alive = False

    def draw(self, frame):
        """Draw the balloon."""
        ix, iy = int(self.x), int(self.y)

        if self.popped:
            elapsed = time.time() - self.pop_time
            if elapsed > 0.4:
                self.alive = False
                return

            # Pop animation — expanding ring + particles
            alpha = elapsed / 0.4
            ring_r = int(self.radius_x * (1 + alpha * 2))
            fade = max(0, 1.0 - alpha)
            thickness = max(1, int(4 * fade))

            cv2.circle(frame, (ix, iy), ring_r, self.body_color, thickness)

            # Particle burst
            for i in range(8):
                angle = (i / 8) * math.pi * 2
                dist = int(ring_r * 0.8)
                px = ix + int(math.cos(angle) * dist)
                py = iy + int(math.sin(angle) * dist)
                pr = max(1, int(4 * fade))
                cv2.circle(frame, (px, py), pr, self.shine_color, -1)

            # "POP!" text
            if elapsed < 0.3:
                cv2.putText(frame, "POP!", (ix - 25, iy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return

        # String
        string_end_y = iy + self.radius_y + 25
        cv2.line(frame, (ix, iy + self.radius_y), (ix, string_end_y), (180, 180, 180), 1)

        # Balloon body (ellipse)
        cv2.ellipse(frame, (ix, iy), (self.radius_x, self.radius_y), 0, 0, 360,
                    self.body_color, -1)

        # Shine highlight
        shine_x = ix - self.radius_x // 3
        shine_y = iy - self.radius_y // 3
        cv2.ellipse(frame, (shine_x, shine_y),
                    (self.radius_x // 4, self.radius_y // 3),
                    -30, 0, 360, self.shine_color, -1)

        # Knot at bottom
        cv2.circle(frame, (ix, iy + self.radius_y), 4, self.body_color, -1)

    def check_hit(self, hx, hy):
        """Check if finger pokes this balloon (simple distance check)."""
        if self.popped or not self.alive:
            return False
        # Elliptical distance
        dx = (hx - self.x) / self.radius_x
        dy = (hy - self.y) / self.radius_y
        return (dx * dx + dy * dy) < 1.5

    def pop(self):
        """Pop this balloon!"""
        self.popped = True
        self.pop_time = time.time()


def _play_balloon_pop():
    """Main Balloon Pop game loop."""
    package_root = os.path.dirname(__file__)
    landmarker = _setup_hand_tracker(package_root)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("🎥 Can't find your camera!")

    GAME_DURATION = 60  # seconds

    print("\n🎈 PiMaker BALLOON POP!")
    print(f"   Pop as many balloons as you can in {GAME_DURATION} seconds!")
    print("   Poke balloons with your finger to pop them!")
    print("   Press 'Q' to quit.\n")

    score = 0
    balloons = []
    game_over = False
    start_time = time.time()
    last_spawn = time.time()
    last_timestamp_ms = -1
    popups = []

    with landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # ── Hand Tracking ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hx, hy = -100, -100
            if result.hand_landmarks and not game_over:
                lm = result.hand_landmarks[0]
                tip = lm[8]  # Index fingertip
                hx, hy = int(tip.x * w), int(tip.y * h)

                # Draw finger cursor — pointer style
                cv2.circle(frame, (hx, hy), 15, (255, 255, 255), 2)
                cv2.circle(frame, (hx, hy), 5, (0, 255, 255), -1)
                cv2.line(frame, (hx, hy - 20), (hx, hy + 20), (255, 255, 255), 1)
                cv2.line(frame, (hx - 20, hy), (hx + 20, hy), (255, 255, 255), 1)

            # ── Timer ──
            elapsed = time.time() - start_time
            remaining = max(0, GAME_DURATION - elapsed)

            if remaining <= 0 and not game_over:
                game_over = True

            # ── Game Logic ──
            if not game_over:
                now = time.time()

                # Spawn balloons
                spawn_rate = max(0.4, 1.2 - (elapsed / GAME_DURATION) * 0.6)
                if now - last_spawn > spawn_rate:
                    count = random.randint(1, min(3, 1 + int(elapsed / 20)))
                    for _ in range(count):
                        balloons.append(Balloon(w, h))
                    last_spawn = now

                # Update and check hits
                for balloon in balloons:
                    balloon.update()

                    if not balloon.popped and balloon.alive:
                        if balloon.check_hit(hx, hy):
                            balloon.pop()
                            score += balloon.points
                            popups.append((f"+{balloon.points}", int(balloon.x),
                                           int(balloon.y), time.time(), (100, 255, 100)))

                balloons = [b for b in balloons if b.alive]

            # ── Drawing ──

            # Draw balloons (back to front by y-position for depth)
            for balloon in sorted(balloons, key=lambda b: b.y):
                balloon.draw(frame)

            # Score popups
            now = time.time()
            active_popups = []
            for (text, px, py, t, color) in popups:
                age = now - t
                if age < 0.8:
                    float_y = py - int(50 * age)
                    scale = max(0.5, 1.0 - age * 0.5)
                    cv2.putText(frame, text, (px - 15, float_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    active_popups.append((text, px, py, t, color))
            popups = active_popups

            # ── HUD ──
            overlay = frame.copy()

            # Score (top left)
            _draw_rounded_rect(overlay, (10, 8), (200, 55), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"Score: {score}", (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)

            # Timer (top right)
            timer_color = (100, 255, 100) if remaining > 10 else (0, 140, 255) if remaining > 5 else (0, 0, 255)
            overlay2 = frame.copy()
            _draw_rounded_rect(overlay2, (w - 170, 8), (w - 10, 55), (30, 30, 30), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"Time: {int(remaining)}s", (w - 160, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)

            # Timer bar
            bar_w = w - 40
            bar_progress = remaining / GAME_DURATION
            cv2.rectangle(frame, (20, 62), (20 + bar_w, 68), (60, 60, 60), -1)
            fill = int(bar_w * bar_progress)
            if fill > 0:
                cv2.rectangle(frame, (20, 62), (20 + fill, 68), timer_color, -1)

            # ── Game Over ──
            if game_over:
                overlay3 = frame.copy()
                cv2.rectangle(overlay3, (0, 0), (w, h), (10, 10, 30), -1)
                cv2.addWeighted(overlay3, 0.65, frame, 0.35, 0, frame)

                cv2.putText(frame, "TIME'S UP!", (w // 2 - 130, h // 2 - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)

                cv2.putText(frame, f"Balloons Popped: {score}", (w // 2 - 140, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # Rating
                if score >= 50:
                    rating = "LEGENDARY!"
                    r_color = (0, 215, 255)
                elif score >= 30:
                    rating = "AMAZING!"
                    r_color = (0, 255, 0)
                elif score >= 15:
                    rating = "GREAT JOB!"
                    r_color = (255, 255, 0)
                else:
                    rating = "KEEP TRYING!"
                    r_color = (200, 200, 200)

                (tw, _), _ = cv2.getTextSize(rating, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.putText(frame, rating, ((w - tw) // 2, h // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, r_color, 2)

                cv2.putText(frame, "Press R to play again | Q to quit",
                            (w // 2 - 200, h // 2 + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("PiMaker Balloon Pop", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                score = 0
                balloons.clear()
                popups.clear()
                game_over = False
                start_time = time.time()
                last_spawn = time.time()
                print("[PiMaker] Game restarted!")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[PiMaker] Balloon Pop stopped. Final score: {score}")


# ═══════════════════════════════════════════════════════════════════════
#  GAME DISPATCHER
# ═══════════════════════════════════════════════════════════════════════

GAME_REGISTRY = {
    "fruit ninja": _play_fruit_ninja,
    "fruitninja": _play_fruit_ninja,
    "ninja": _play_fruit_ninja,
    "balloon pop": _play_balloon_pop,
    "balloonpop": _play_balloon_pop,
    "balloon": _play_balloon_pop,
    "rock paper scissors": None,  # placeholder — actual impl in rock_paper_scissors.py
    "rps": None,
}


def play_game(game_name):
    """
    🎮 Play a PiMaker game!

    Available games:
      - "fruit ninja"   — Swipe to slice fruits, avoid bombs!
      - "balloon pop"   — Pop floating balloons before time runs out!
      - "rock paper scissors" — Gesture-based RPS vs the computer!

    Args:
        game_name (str): Name of the game to play (case-insensitive).

    Example:
        pimaker.play_game("fruit ninja")
        pimaker.play_game("balloon pop")
    """
    key = game_name.strip().lower()

    if key in ("rock paper scissors", "rps", "rockpaperscissors"):
        from .rock_paper_scissors import play_rock_paper_scissors
        play_rock_paper_scissors()
        return

    game_func = GAME_REGISTRY.get(key)

    if game_func is None:
        available = ["fruit ninja", "balloon pop", "rock paper scissors"]
        raise ValueError(
            f"🎮 Unknown game '{game_name}'!\n"
            f"   Available games: {', '.join(available)}\n"
            f"   Example: pimaker.play_game(\"fruit ninja\")"
        )

    game_func()
