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
    raise ImportError("mediapipe is required for start_air_mouse(). Install with: pip install mediapipe") from exc

try:
    import pyautogui
except ImportError as exc:
    raise ImportError("pyautogui is required for start_air_mouse(). Install with: pip install pyautogui") from exc

from .vision.fingertip_backends import _ensure_model_file

def start_air_mouse():
    """
    Starts the Air Mouse and Presentation Controller.
    Uses hand tracking to control the cursor.
    - Move pointer: Move your Index Finger
    - Left Click & Select/Drag: Pinch Index Finger & Thumb together
    - Right Click: Pinch Middle Finger & Thumb together
    """
    package_root = os.path.dirname(__file__)
    model_path = _ensure_model_file(package_root)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()
    
    smooth_factor = 4 # slightly higher for smoother drags
    ploc_x, ploc_y = 0, 0
    cloc_x, cloc_y = 0, 0
    
    click_threshold = 35
    is_left_clicking = False
    is_right_clicking = False
    
    print("Air Mouse started. Press 'q' to stop.")
    print("- Move: Index Finger")
    print("- Left Click/Drag: Pinch Index & Thumb")
    print("- Right Click: Pinch Middle & Thumb")
    
    # Reduce pyautogui delay for smoother tracking
    pyautogui.PAUSE = 0
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Define a bounding box to make it easier to reach screen edges
            margin_x, margin_y = int(w * 0.15), int(h * 0.2)
            cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y), (255, 0, 0), 2)
            
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    # Index tip (8), Thumb tip (4), Middle tip (12)
                    index_tip = hand_landmarks[8]
                    thumb_tip = hand_landmarks[4]
                    middle_tip = hand_landmarks[12]
                    
                    # Convert to pixel coordinates
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
                    
                    # Draw circles on tips
                    cv2.circle(frame, (ix, iy), 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (tx, ty), 10, (0, 255, 0), cv2.FILLED)
                    cv2.circle(frame, (mx, my), 10, (0, 255, 255), cv2.FILLED)
                    
                    # Map Index finger to screen dimensions using the tight bounding box
                    mapped_x = np.interp(ix, (margin_x, w - margin_x), (0, screen_w))
                    mapped_y = np.interp(iy, (margin_y, h - margin_y), (0, screen_h))
                    
                    # Smooth the cursor movement
                    cloc_x = ploc_x + (mapped_x - ploc_x) / smooth_factor
                    cloc_y = ploc_y + (mapped_y - ploc_y) / smooth_factor
                    
                    # Constrain to screen bounds
                    cursor_x = max(0, min(screen_w - 1, cloc_x))
                    cursor_y = max(0, min(screen_h - 1, cloc_y))
                    
                    try:
                        pyautogui.moveTo(cursor_x, cursor_y)
                        ploc_x, ploc_y = cloc_x, cloc_y
                    except Exception:
                        # Ignore pyautogui fail-safe exception
                        pass
                    
                    # Calculate distances
                    dist_index_thumb = math.hypot(tx - ix, ty - iy)
                    dist_middle_thumb = math.hypot(tx - mx, ty - my)
                    
                    # --- LEFT CLICK / DRAG (Index + Thumb) ---
                    if dist_index_thumb < click_threshold:
                        cv2.circle(frame, ((ix + tx) // 2, (iy + ty) // 2), 15, (0, 0, 255), cv2.FILLED)
                        if not is_left_clicking:
                            try:
                                pyautogui.mouseDown(button='left')
                            except Exception:
                                pass
                            is_left_clicking = True
                    else:
                        if is_left_clicking:
                            try:
                                pyautogui.mouseUp(button='left')
                            except Exception:
                                pass
                            is_left_clicking = False
                            
                    # --- RIGHT CLICK (Middle + Thumb) ---
                    if dist_middle_thumb < click_threshold:
                        cv2.circle(frame, ((mx + tx) // 2, (my + ty) // 2), 15, (255, 0, 0), cv2.FILLED)
                        if not is_right_clicking:
                            try:
                                pyautogui.click(button='right')
                            except Exception:
                                pass
                            is_right_clicking = True
                    else:
                        is_right_clicking = False
                        
            else:
                # If no hands detected but we were holding left click, release it so it doesn't get stuck!
                if is_left_clicking:
                    try:
                        pyautogui.mouseUp(button='left')
                    except Exception:
                        pass
                    is_left_clicking = False
                    
            cv2.imshow("Air Mouse Tracker", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    # Failsafe release all on exit
    if is_left_clicking:
        pyautogui.mouseUp(button='left')
    cap.release()
    cv2.destroyAllWindows()
