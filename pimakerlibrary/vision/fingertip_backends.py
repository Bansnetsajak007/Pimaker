import os
import time
import urllib.request


def _ensure_model_file(package_root):
    model_dir = os.path.join(package_root, "models")
    model_path = os.path.join(model_dir, "hand_landmarker.task")
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        print("Downloading MediaPipe hand model...")
        urllib.request.urlretrieve(url, model_path)
    return model_path


def run_legacy_webcam(cv2, mp, annotate_legacy_frame, selected_tip_ids, show_gesture=False):
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
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            annotated = annotate_legacy_frame(
                frame,
                results,
                mp_hands,
                mp_draw,
                cv2,
                selected_tip_ids,
                show_gesture=show_gesture
            )

            cv2.imshow("PiMaker Finger Tip Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_tasks_webcam(cv2, mp, annotate_tasks_frame, selected_tip_ids, show_gesture=False):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    package_root = os.path.dirname(os.path.dirname(__file__))
    model_path = _ensure_model_file(package_root)

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

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        print("Press 'q' to exit fingertip detector")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            annotated = annotate_tasks_frame(frame, result.hand_landmarks, cv2, selected_tip_ids, show_gesture=show_gesture)

            cv2.imshow("PiMaker Finger Tip Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
