import os
import cv2
import time
import math
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise ImportError(
        "mediapipe is required for pool. "
        "Install with: pip install mediapipe"
    ) from exc

from .vision.fingertip_backends import _ensure_model_file
from .pool_physics import check_ball_collision


class Ball:
    def __init__(self, x, y, number, type, is_cue=False, is_8ball=False):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.radius = 16
        self.number = number # 1-15, 0 for cue
        self.type = type # "solid", "stripe", "cue", "8ball"
        self.is_cue = is_cue
        self.is_8ball = is_8ball
        self.active = True
        self.pocketed = False
        
        # Colors
        self.base_color = self._get_color(number)
        
    def _get_color(self, num):
        if num == 0: return (240, 240, 240) # Cue
        if num == 8: return (30, 30, 30) # 8ball
        
        colors = [
            (50, 180, 220),  # 1/9: Yellow
            (200, 50, 50),   # 2/10: Blue 
            (50, 50, 200),   # 3/11: Red
            (150, 50, 150),  # 4/12: Purple
            (50, 150, 255),  # 5/13: Orange
            (50, 200, 50),   # 6/14: Green
            (50, 50, 150),   # 7/15: Maroon
        ]
        
        if num > 8:
            num -= 8
            
        return colors[(num - 1) % 7]
        
    def draw(self, frame):
        if not self.active or self.pocketed:
            return
            
        ix, iy = int(self.x), int(self.y)
        
        # Shadow
        cv2.circle(frame, (ix+2, iy+2), self.radius, (30, 30, 30), -1)
        
        if self.is_cue:
            cv2.circle(frame, (ix, iy), self.radius, self.base_color, -1)
            # Dot on cue ball
            cv2.circle(frame, (ix, iy), 3, (150, 150, 150), -1)
        elif self.is_8ball:
            cv2.circle(frame, (ix, iy), self.radius, self.base_color, -1)
            cv2.circle(frame, (ix, iy), int(self.radius*0.6), (220, 220, 220), -1)
            cv2.putText(frame, "8", (ix-5, iy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
        elif self.type == "solid":
            cv2.circle(frame, (ix, iy), self.radius, self.base_color, -1)
            cv2.circle(frame, (ix, iy), int(self.radius*0.5), (220, 220, 220), -1)
            text_x = ix - 4 if self.number < 10 else ix - 7
            cv2.putText(frame, str(self.number), (text_x, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        elif self.type == "stripe":
            # White base
            cv2.circle(frame, (ix, iy), self.radius, (240, 240, 240), -1)
            # Colored stripe (drawn as a thick line)
            cv2.line(frame, (ix - self.radius, iy), (ix + self.radius, iy), self.base_color, self.radius)
            # Circle with number
            cv2.circle(frame, (ix, iy), int(self.radius*0.5), (240, 240, 240), -1)
            text_x = ix - 4 if self.number < 10 else ix - 7
            cv2.putText(frame, str(self.number), (text_x, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            
        # Highlight (glassy reflection)
        cv2.circle(frame, (ix-self.radius//3, iy-self.radius//3), self.radius//4, (255, 255, 255), -1)
        
    def update(self):
        if not self.active or self.pocketed:
            return
            
        self.x += self.vx
        self.y += self.vy
        
        # Friction (exponential drag looks most realistic for pool balls)
        self.vx *= 0.985
        self.vy *= 0.985
        
        # Stop completely when very slow to avoid endless micro-crawling
        if math.hypot(self.vx, self.vy) < 0.15:
            self.vx = 0
            self.vy = 0


class PoolTable:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.cushion_w = 20
        self.pocket_radius = 26
        
        # Inside play area bounds
        self.left = self.x + self.cushion_w
        self.right = self.x + self.w - self.cushion_w
        self.top = self.y + self.cushion_w
        self.bottom = self.y + self.h - self.cushion_w
        
        self.pockets = [
            (self.left, self.top),
            (self.left + (self.right - self.left)//2, self.top - 5),
            (self.right, self.top),
            (self.left, self.bottom),
            (self.left + (self.right - self.left)//2, self.bottom + 5),
            (self.right, self.bottom)
        ]
        
    def draw(self, frame):
        # Outer rail
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (30, 40, 60), -1)
        
        # Play surface
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.left, self.top), (self.right, self.bottom), (40, 160, 60), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Cushion edges
        cv2.rectangle(frame, (self.left, self.top), (self.right, self.bottom), (20, 120, 40), 4)

        # Draw Pockets
        for px, py in self.pockets:
            cv2.circle(frame, (int(px), int(py)), self.pocket_radius, (20, 20, 20), -1)
            
    def check_wall_collisions(self, ball):
        if ball.pocketed:
            return
            
        r = ball.radius
        e = 0.85 # Wall bounciness
        
        # Avoid checking collision if ball is basically in a pocket to allow it to fall in
        in_pocket_zone = False
        for px, py in self.pockets:
            dist = math.hypot(ball.x - px, ball.y - py)
            if dist < self.pocket_radius * 1.5:
                in_pocket_zone = True
                break
                
        if not in_pocket_zone:
            if ball.x - r < self.left:
                ball.x = self.left + r
                ball.vx = abs(ball.vx) * e
            elif ball.x + r > self.right:
                ball.x = self.right - r
                ball.vx = -abs(ball.vx) * e
                
            if ball.y - r < self.top:
                ball.y = self.top + r
                ball.vy = abs(ball.vy) * e
            elif ball.y + r > self.bottom:
                ball.y = self.bottom - r
                ball.vy = -abs(ball.vy) * e
                
    def check_pocket(self, ball):
        if ball.pocketed:
            return False
            
        for px, py in self.pockets:
            dist = math.hypot(ball.x - px, ball.y - py)
            # Ball falls in when center is near pocket rim
            if dist < self.pocket_radius:
                ball.pocketed = True
                ball.active = False
                ball.vx = 0
                ball.vy = 0
                return True
                
        return False


def _setup_hand_tracker(package_root):
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


def _play_8ball_pool():
    package_root = os.path.dirname(__file__)
    landmarker = _setup_hand_tracker(package_root)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("🎥 Can't find your camera!")

    print("\n🎱 PiMaker 8-Ball Pool!")
    print("   Take turns playing on the virtual table.")
    print("   Aim: Move index finger")
    print("   Shoot: Pinch thumb + index to charge power, release to shoot")
    print("   Press 'Q' to quit.\n")
    
    # Needs to be called with a dummy frame to get dimensions
    ret, frame = cap.read()
    h, w, _ = frame.shape
    
    table_x = 40
    table_y = 60
    table_w = w - 80
    table_h = h - 120
    
    table = PoolTable(table_x, table_y, table_w, table_h)
    
    # State Machine
    STATE_AIMING = 0
    STATE_CHARGING = 1
    STATE_MOVING = 2
    STATE_P_TURN_RESULT = 3
    STATE_PLACING_CUE = 4
    STATE_GAME_OVER = 5
    
    state = STATE_AIMING
    
    # Player state
    current_player = 1
    player1_type = None # "solid" or "stripe"
    player2_type = None
    
    balls = []
    
    def reset_rack():
        nonlocal balls
        balls.clear()
        
        # Head string (cue ball line)
        head_x = table.left + (table.right - table.left) * 0.25
        cy = table.y + table.h / 2
        
        # Foot spot (triangle tip)
        foot_x = table.left + (table.right - table.left) * 0.75
        
        r = 16
        spacing = r * 2.05 # slightly more than 2r to prevent physics explosion
        rows = 5
        
        # Standard rack order
        rack_setup = [
            ("solid", 1),
            ("stripe", 9), ("solid", 2),
            ("stripe", 10), ("8ball", 8), ("solid", 3),
            ("solid", 4), ("stripe", 11), ("solid", 5), ("stripe", 12),
            ("stripe", 13), ("solid", 6), ("stripe", 14), ("solid", 7), ("stripe", 15)
        ]
        
        idx = 0
        for row in range(rows):
            col_x = foot_x + row * (spacing * math.sqrt(3)/2)
            start_y = cy - (row * spacing / 2)
            for i in range(row + 1):
                col_y = start_y + i * spacing
                if idx < len(rack_setup):
                    btype, num = rack_setup[idx]
                    balls.append(Ball(col_x, col_y, num, btype, is_8ball=(num==8)))
                    idx += 1
                    
        # Add cue ball
        balls.append(Ball(head_x, cy, 0, "cue", is_cue=True))
        
    reset_rack()
    
    cue_ball = next(b for b in balls if b.is_cue)
    
    charge_level = 0
    max_charge = 100
    last_timestamp_ms = -1
    
    msg_overlay = ""
    msg_timer = 0
    
    def show_msg(msg, duration=2.0):
        nonlocal msg_overlay, msg_timer
        msg_overlay = msg
        msg_timer = time.time() + duration

    was_pinching = False
    
    # Turn tracking
    turn_made_contact = False
    turn_pocketed_valid = False
    turn_pocketed_balls = []
    
    # Hand tracking smoothing
    smooth_aim_x, smooth_aim_y = None, None
    
    with landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hx, hy = -100, -100
            tx, ty = -100, -100
            is_pinching = False
            
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                tip = lm[8]  # index
                thumb = lm[4] # thumb
                hx, hy = int(tip.x * w), int(tip.y * h)
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                
                dist = math.hypot(tx - hx, ty - hy)
                
                # Pinch hysteresis: if already pinching, require fingers to be wide apart to release (shoot).
                # If not pinching, require fingers to be close to start pinching (aiming).
                if was_pinching:
                    is_pinching = dist < 70
                else:
                    is_pinching = dist < 35

                # Cursor
                cv2.circle(frame, (hx, hy), 12, (0, 255, 255), 2)
                cv2.circle(frame, (hx, hy), 4, (255, 255, 255), -1)
                
                # Smooth aiming
                if smooth_aim_x is None:
                    smooth_aim_x, smooth_aim_y = float(hx), float(hy)
                else:
                    smooth_aim_x = smooth_aim_x * 0.7 + hx * 0.3
                    smooth_aim_y = smooth_aim_y * 0.7 + hy * 0.3
            else:
                smooth_aim_x, smooth_aim_y = None, None

            # --- Physics / Logic Update ---
            table.draw(frame)
            
            all_stopped = True
            for b in balls:
                if b.active:
                    b.update()
                    table.check_wall_collisions(b)
                    
                    if math.hypot(b.vx, b.vy) > 0.05:
                        all_stopped = False
                        
                    # Check pockets
                    if table.check_pocket(b):
                        turn_pocketed_balls.append(b)
            
            # Collisions (n^2 but n is max 16 so it's fine)
            for i in range(len(balls)):
                if not balls[i].active: continue
                for j in range(i+1, len(balls)):
                    if not balls[j].active: continue
                    check_ball_collision(balls[i], balls[j])
                        
            # Draw balls
            for b in sorted(balls, key=lambda x: not x.active): # Draw active on top
                b.draw(frame)
                
            # State transitions
            if state == STATE_AIMING:
                if cue_ball.active and smooth_aim_x is not None:
                    # Draw aim line from cue to finger
                    dx = smooth_aim_x - cue_ball.x
                    dy = smooth_aim_y - cue_ball.y
                    dist = math.hypot(dx, dy)
                    
                    if dist > 0:
                        nx = dx / dist
                        ny = dy / dist
                        
                        end_x = int(cue_ball.x + nx * 800)
                        end_y = int(cue_ball.y + ny * 800)
                        
                        # Dotted line
                        cv2.line(frame, (int(cue_ball.x), int(cue_ball.y)), (end_x, end_y), (255, 255, 255), 1)
                        
                if is_pinching and not was_pinching:
                    state = STATE_CHARGING
                    charge_level = 0
                    
            elif state == STATE_CHARGING:
                # Draw aim line
                if smooth_aim_x is not None:
                    dx = smooth_aim_x - cue_ball.x
                    dy = smooth_aim_y - cue_ball.y
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        nx = dx / dist
                        ny = dy / dist
                        end_x = int(cue_ball.x + nx * 800)
                        end_y = int(cue_ball.y + ny * 800)
                        cv2.line(frame, (int(cue_ball.x), int(cue_ball.y)), (end_x, end_y), (100, 255, 255), 2)
                    
                charge_level += 2
                if charge_level > max_charge:
                    charge_level = max_charge
                    
                # Charge bar at finger
                cv2.rectangle(frame, (hx + 20, hy - 50), (hx + 30, hy), (50, 50, 50), -1)
                fill_h = int(50 * (charge_level / max_charge))
                cv2.rectangle(frame, (hx + 20, hy - fill_h), (hx + 30, hy), (0, 255, 0), -1)
                
                if not is_pinching: # Released!
                    if charge_level >= 5:
                        # Shoot!
                        power = charge_level / max_charge * 30 # Max velocity
                        cue_ball.vx = nx * power
                        cue_ball.vy = ny * power
                        
                        state = STATE_MOVING
                        turn_pocketed_balls = []
                    else:
                        # Cancel accidental micro-charge
                        state = STATE_AIMING
                        
            elif state == STATE_MOVING:
                if all_stopped:
                    # If ball moved but nothing pocketed, check if scratch was just due to velocity
                    # But we'll mostly defer to results
                    state = STATE_P_TURN_RESULT
                    
            elif state == STATE_P_TURN_RESULT:
                scratch = next((b for b in turn_pocketed_balls if b.is_cue), None) is not None
                eight_ball = next((b for b in turn_pocketed_balls if b.is_8ball), None) is not None
                
                current_type = player1_type if current_player == 1 else player2_type
                
                if eight_ball:
                    # Win/loss check
                    remaining_own = sum(1 for b in balls if b.type == current_type and b.active)
                    if remaining_own == 0 and not scratch:
                        show_msg(f"PLAYER {current_player} WINS!")
                    else:
                        other = 2 if current_player == 1 else 1
                        show_msg(f"PLAYER {other} WINS! (8-ball foul)")
                    state = STATE_GAME_OVER
                else:
                    valid_pocket = False
                    foul = scratch
                    
                    for pb in turn_pocketed_balls:
                        if pb.is_cue: continue
                        if current_type is None:
                            # Assign types
                            valid_pocket = True
                            if current_player == 1:
                                player1_type = pb.type
                                player2_type = "solid" if pb.type == "stripe" else "stripe"
                            else:
                                player2_type = pb.type
                                player1_type = "solid" if pb.type == "stripe" else "stripe"
                            show_msg(f"P{current_player} is {pb.type}s")
                        elif pb.type == current_type:
                            valid_pocket = True
                        elif pb.type != current_type:
                            # Pocketed opponent ball, no turn continuation
                            pass

                    if scratch:
                        cue_ball.active = True
                        cue_ball.pocketed = False
                        cue_ball.x = hx
                        cue_ball.y = hy
                        cue_ball.vx = 0
                        cue_ball.vy = 0
                        current_player = 2 if current_player == 1 else 1
                        show_msg("SCRATCH! Opponent plays.")
                        state = STATE_PLACING_CUE
                    elif not valid_pocket:
                        current_player = 2 if current_player == 1 else 1
                        show_msg(f"PLAYER {current_player}'S TURN")
                        state = STATE_AIMING
                    else:
                        show_msg("KEEP SHOOTING!")
                        state = STATE_AIMING

            elif state == STATE_PLACING_CUE:
                if smooth_aim_x is not None:
                    # Move cue ball to finger
                    clamped_x = max(table.left + cue_ball.radius, min(table.right - cue_ball.radius, smooth_aim_x))
                    clamped_y = max(table.top + cue_ball.radius, min(table.bottom - cue_ball.radius, smooth_aim_y))
                    cue_ball.x = clamped_x
                    cue_ball.y = clamped_y
                
                # Check intersection with other balls
                valid_spot = True
                for b in balls:
                    if b != cue_ball and b.active:
                        if math.hypot(b.x - cue_ball.x, b.y - cue_ball.y) < cue_ball.radius * 2:
                            valid_spot = False
                            break
                            
                color = (0, 255, 0) if valid_spot else (0, 0, 255)
                cv2.circle(frame, (int(cue_ball.x), int(cue_ball.y)), cue_ball.radius + 2, color, 2)
                
                if is_pinching and not was_pinching and valid_spot:
                    state = STATE_AIMING
                    
            elif state == STATE_GAME_OVER:
                cv2.putText(frame, "GAME OVER", (w//2 - 100, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, "Press R to Restart", (w//2 - 120, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            was_pinching = is_pinching

            # HUD
            if time.time() < msg_timer:
                (tw, th), _ = cv2.getTextSize(msg_overlay, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.putText(frame, msg_overlay, (w//2 - tw//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
            # Scoreboard
            p1_color = (255, 255, 255) if current_player == 1 else (150, 150, 150)
            p2_color = (255, 255, 255) if current_player == 2 else (150, 150, 150)
            
            p1_lbl = f"Player 1 ({player1_type or '?'})"
            p2_lbl = f"Player 2 ({player2_type or '?'})"
            
            cv2.putText(frame, p1_lbl, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p1_color, 2)
            cv2.putText(frame, p2_lbl, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p2_color, 2)
            
            # Indicator
            if current_player == 1:
                cv2.circle(frame, (10, 25), 5, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (w - 210, 25), 5, (0, 255, 0), -1)

            cv2.imshow("PiMaker 8-Ball Pool", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                reset_rack()
                cue_ball = next(b for b in balls if b.is_cue)
                current_player = 1
                player1_type = None
                player2_type = None
                state = STATE_AIMING
                show_msg("GAME RESET")

    cap.release()
    cv2.destroyAllWindows()
