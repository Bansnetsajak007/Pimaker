import os
import cv2
import time
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    raise ImportError("mediapipe is required. Install with: pip install mediapipe") from exc

try:
    import sounddevice as sd
except ImportError as exc:
    raise ImportError("sounddevice is required. Install with: pip install sounddevice") from exc

from .vision.fingertip_backends import _ensure_model_file

def play_note_sharp(frequency, duration_ms=250):
    """
    Plays a sharp, 8-bit style retro square wave asynchronously.
    """
    sample_rate = 44100
    duration_s = duration_ms / 1000.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), False)
    
    # Generate a sharp square wave (retro video game synth style)
    tone = np.sign(np.sin(frequency * t * 2 * np.pi)) * 0.5  # 0.5 volume so it doesn't blast ears!
    
    # Apply fade in/out envelope to prevent clicking/popping speaker noises
    fade_samples = int(sample_rate * 0.05)
    if len(tone) > fade_samples * 2:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
        
    sd.play(tone, samplerate=sample_rate)

# Drum Pads / Floating Keys
KEYS = [
    {"name": "Do", "freq": 523.25, "color": (50, 50, 200), "active_color": (150, 150, 255)}, # Red (C5)
    {"name": "Re", "freq": 587.33, "color": (50, 200, 50), "active_color": (150, 255, 150)}, # Green (D5)
    {"name": "Mi", "freq": 659.25, "color": (200, 100, 50), "active_color": (255, 150, 150)}, # Blue (E5)
    {"name": "Fa", "freq": 698.46, "color": (50, 200, 200), "active_color": (150, 255, 255)} # Yellow (F5)
]

def start_virtual_piano():
    package_root = os.path.dirname(__file__)
    model_path = _ensure_model_file(package_root)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    print("Virtual Drum/Piano Started. Press 'q' to stop.")

    last_pressed_idx = -1

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw floating drum pads in the center of the screen (not the bottom, to avoid torso overlap)
            num_keys = len(KEYS)
            pad_width = int(w * 0.18)
            pad_height = int(h * 0.3)
            gap = int(w * 0.05)
            
            total_pads_width = (pad_width * num_keys) + (gap * (num_keys - 1))
            start_x = (w - total_pads_width) // 2
            start_y = (h - pad_height) // 2 # Centered vertically
            
            key_rects = []
            
            # Using an overlay for transparency
            overlay = frame.copy()
            
            for i in range(num_keys):
                x1 = start_x + (i * (pad_width + gap))
                y1 = start_y
                x2 = x1 + pad_width
                y2 = y1 + pad_height
                key_rects.append((x1, y1, x2, y2))
                
                # Draw resting state of the key
                cv2.rectangle(overlay, (x1, y1), (x2, y2), KEYS[i]["color"], -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                
                # Center the text
                text_size = cv2.getTextSize(KEYS[i]["name"], cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                tx = x1 + (pad_width - text_size[0]) // 2
                ty = y1 + (pad_height + text_size[1]) // 2
                cv2.putText(frame, KEYS[i]["name"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Alpha blend the boxes so you can still see yourself behind them
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            current_pressed_idx = -1

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    # Index fingertip is element 8
                    index_tip = hand_landmarks[8]
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Highlight the tip of the kid's finger with a cool cursor!
                    cv2.circle(frame, (ix, iy), 20, (0, 255, 255), -1)
                    cv2.circle(frame, (ix, iy), 20, (0, 0, 0), 2)
                    
                    # Hitbox collision logic
                    for i, (x1, y1, x2, y2) in enumerate(key_rects):
                        if x1 <= ix <= x2 and y1 <= iy <= y2:
                            current_pressed_idx = i
                            # Make the key light up brightly instantly (solid, no transparency)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), KEYS[i]["active_color"], -1)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)
                            
                            text_size = cv2.getTextSize(KEYS[i]["name"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
                            tx = x1 + (pad_width - text_size[0]) // 2
                            ty = y1 + (pad_height + text_size[1]) // 2
                            cv2.putText(frame, KEYS[i]["name"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 4)
                            break
            
            # Audio trigger logic
            if current_pressed_idx != -1 and current_pressed_idx != last_pressed_idx:
                play_note_sharp(KEYS[current_pressed_idx]["freq"], 300)
                
            last_pressed_idx = current_pressed_idx
            
            cv2.imshow("PiMaker Virtual Piano", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
