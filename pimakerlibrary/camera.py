import cv2
import os
import time
import urllib.request

def open_camera():
    cap = cv2.VideoCapture(0)  # 0 = default camera

    if not cap.isOpened():
        raise Exception("Could not open camera")

    print("Press 'q' to exit camera")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("PiMaker Camera", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_fingertip():
    """Detect and visualize the index fingertip using MediaPipe."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for detect_fingertip(). Install with: pip install mediapipe"
        ) from exc

    def _draw_tip_box(frame, x, y, color=(0, 255, 255), box_size=28):
        half = box_size // 2
        h, w, _ = frame.shape
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w - 1, x + half)
        y2 = min(h - 1, y + half)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    fingertip_ids = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }

    def _run_legacy_solutions():
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Could not open camera")

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            print("Press 'q' to exit fingertip detector")

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        for name, tip_id in fingertip_ids.items():
                            tip = hand_landmarks.landmark[tip_id]
                            x, y = int(tip.x * w), int(tip.y * h)

                            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
                            _draw_tip_box(frame, x, y)
                            cv2.putText(
                                frame,
                                name,
                                (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (50, 255, 50),
                                1,
                            )

                cv2.imshow("PiMaker Finger Tip Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def _run_tasks_api():
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        if not os.path.exists(model_path):
            os.makedirs(model_dir, exist_ok=True)
            url = (
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )
            print("Downloading MediaPipe hand model...")
            urllib.request.urlretrieve(url, model_path)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        with vision.HandLandmarker.create_from_options(options) as landmarker:
            print("Press 'q' to exit fingertip detector")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    h, w, _ = frame.shape
                    for landmarks in result.hand_landmarks:
                        for a, b in hand_connections:
                            ax, ay = int(landmarks[a].x * w), int(landmarks[a].y * h)
                            bx, by = int(landmarks[b].x * w), int(landmarks[b].y * h)
                            cv2.line(frame, (ax, ay), (bx, by), (0, 200, 0), 2)

                        for point in landmarks:
                            px, py = int(point.x * w), int(point.y * h)
                            cv2.circle(frame, (px, py), 3, (255, 100, 0), -1)

                        for name, tip_id in fingertip_ids.items():
                            tip = landmarks[tip_id]
                            x, y = int(tip.x * w), int(tip.y * h)
                            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
                            _draw_tip_box(frame, x, y)
                            cv2.putText(
                                frame,
                                name,
                                (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (50, 255, 50),
                                1,
                            )

                cv2.imshow("PiMaker Finger Tip Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        _run_legacy_solutions()
    else:
        _run_tasks_api()