import cv2
import numpy as np
import os
import time

from .fingertip_backends import _ensure_model_file

def detect_gesture(lm_list):
    """
    Returns string: 'peace', 'fist', 'thumbs_up', or None
    """
    if len(lm_list) < 21: 
        return None

    # Finger states (True if UP)
    # Thumbs up: Tip is significantly higher than knuckles
    thumb_up = lm_list[4][1] < lm_list[5][1] and lm_list[4][1] < lm_list[9][1] 
    index_up = lm_list[8][1] < lm_list[6][1]
    middle_up = lm_list[12][1] < lm_list[10][1]
    ring_up = lm_list[16][1] < lm_list[14][1]
    pinky_up = lm_list[20][1] < lm_list[18][1]

    fingers = [index_up, middle_up, ring_up, pinky_up]

    if fingers == [False, False, False, False] and not thumb_up:
        return 'fist'
    elif fingers == [False, False, False, False] and thumb_up:
        return 'thumbs_up'
    elif fingers == [True, True, True, True]:
        return 'open_palm'
    
    return None

def open_canvas(default_color=(0, 255, 0), thickness=5):
    """
    Opens the webcam and allows drawing on an air canvas using the index finger.
    Features a color palette at the top of the screen to change paint color or clear canvas.
    Hover mode is activated by raising both the index and middle fingers.
    Press 'c' to clear the canvas manually.
    Press 'q' to quit.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for open_canvas(). Install with: pip install mediapipe"
        ) from exc

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("Could not open camera")

    print("\nAir Canvas Started.")
    print(" - Index finger UP to draw")
    print(" - Index + Middle fingers UP to hover")
    print(" - Hover over top rectangles to change colors or clear")
    print(" - Press 'q' to quit\n")
    print("GESTURE CONTROLS:")
    print(" - Hold Open Palm (🖐️): Save screenshot of your video feed")
    print(" - Hold Thumbs Up (👍): Save the transparent canvas drawing")
    print(" - Hold Closed Fist (✊): Exit application cleanly\n")

    canvas = None
    px, py = 0, 0
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    current_color = default_color

    package_root = os.path.dirname(os.path.dirname(__file__))
    model_path = _ensure_model_file(package_root)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    last_gesture_time = 0
    gesture_hold_start = 0
    current_held_gesture = None
    exit_requested = False

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while not exit_requested:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Retrying...")
                continue

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            if canvas is None:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Build the visual frame first before doing gesture screenshotting
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, inv_mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
            inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
            frame_bg = cv2.bitwise_and(frame, inv_mask)
            frame_with_canvas = cv2.bitwise_or(frame_bg, canvas)

            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    lm_list = []
                    for id, lm in enumerate(landmarks):
                        hx, hy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((hx, hy))

                    if len(lm_list) == 21:
                        gesture = detect_gesture(lm_list)
                        if gesture:
                            if gesture == current_held_gesture:
                                hold_duration = time.time() - gesture_hold_start
                                # Visual indicator of hold
                                if gesture == 'open_palm':
                                    cv2.putText(frame_with_canvas, "Saving Screenshot...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                elif gesture == 'thumbs_up':
                                    cv2.putText(frame_with_canvas, "Saving Canvas...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                elif gesture == 'fist':
                                    cv2.putText(frame_with_canvas, "Quitting...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                if hold_duration > 1.5:
                                    if gesture == 'open_palm':
                                        if time.time() - last_gesture_time > 3:
                                            cv2.imwrite("screenshot.png", frame_with_canvas)
                                            print("Screenshot saved to screenshot.png!")
                                            last_gesture_time = time.time()
                                    elif gesture == 'thumbs_up':
                                        if time.time() - last_gesture_time > 3:
                                            cv2.imwrite("canvas_drawing.png", canvas)
                                            print("Drawing saved to canvas_drawing.png!")
                                            last_gesture_time = time.time()
                                    elif gesture == 'fist':
                                        print("Fist detected, quitting!")
                                        exit_requested = True
                            else:
                                current_held_gesture = gesture
                                gesture_hold_start = time.time()
                        else:
                            current_held_gesture = None
                            gesture_hold_start = time.time() # Reset clock

                        # Continue with standard drawing logic
                        x1, y1 = lm_list[8]
                        
                        if px == 0 and py == 0:
                            sm_x, sm_y = x1, y1
                        else:
                            sm_x = int(0.6 * x1 + 0.4 * px)
                            sm_y = int(0.6 * y1 + 0.4 * py)

                        index_up = lm_list[8][1] < lm_list[6][1]
                        middle_up = lm_list[12][1] < lm_list[10][1]
                        
                        if index_up:
                            if sm_y < 65:
                                if 40 <= sm_x <= 140:
                                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                                elif 160 <= sm_x <= 260:
                                    current_color = colors[0]
                                elif 280 <= sm_x <= 380:
                                    current_color = colors[1]
                                elif 400 <= sm_x <= 500:
                                    current_color = colors[2]
                                elif 520 <= sm_x <= 620:
                                    current_color = colors[3]

                        if index_up and middle_up:
                            px, py = 0, 0
                            cv2.circle(frame_with_canvas, (sm_x, sm_y), thickness + 5, current_color, cv2.FILLED)
                        elif index_up and not middle_up:
                            cv2.circle(frame_with_canvas, (sm_x, sm_y), thickness, current_color, cv2.FILLED)
                            
                            if px == 0 and py == 0:
                                px, py = sm_x, sm_y
                                
                            if sm_y > 65:
                                cv2.line(canvas, (px, py), (sm_x, sm_y), current_color, thickness)
                                # Update frame_with_canvas interactively so we see it during this frame too
                                cv2.line(frame_with_canvas, (px, py), (sm_x, sm_y), current_color, thickness)
                            px, py = sm_x, sm_y
                        else:
                            px, py = 0, 0

            # Draw Palette UI
            # Drawn onto frame_with_canvas directly so they overlay properly
            cv2.rectangle(frame_with_canvas, (40, 10), (140, 60), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame_with_canvas, "CLEAR", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame_with_canvas, (160, 10), (260, 60), colors[0], cv2.FILLED)
            cv2.rectangle(frame_with_canvas, (280, 10), (380, 60), colors[1], cv2.FILLED)
            cv2.rectangle(frame_with_canvas, (400, 10), (500, 60), colors[2], cv2.FILLED)
            cv2.rectangle(frame_with_canvas, (520, 10), (620, 60), colors[3], cv2.FILLED)

            cv2.imshow("PiMaker Air Canvas", frame_with_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                print("Canvas cleared.")

    cap.release()
    cv2.destroyAllWindows()
