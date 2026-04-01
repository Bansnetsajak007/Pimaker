import cv2

from .vision.fingertip_backends import run_legacy_webcam, run_tasks_webcam
from .vision.fingertip_draw import FINGERTIP_IDS, annotate_legacy_frame, annotate_tasks_frame

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


def detect_fingertip(finger=None):
    """Detect fingertips from webcam only.

    Args:
        finger: Optional finger selector from 1 to 5.
            1=thumb, 2=index, 3=middle, 4=ring, 5=pinky.
            If omitted, all fingertips are detected.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for detect_fingertip(). Install with: pip install mediapipe"
        ) from exc

    selected_tip_ids = FINGERTIP_IDS
    if finger is not None:
        if not isinstance(finger, int) or finger < 1 or finger > 5:
            raise ValueError("finger must be an integer from 1 to 5")

        ordered_names = ["thumb", "index", "middle", "ring", "pinky"]
        selected_name = ordered_names[finger - 1]
        selected_tip_ids = {selected_name: FINGERTIP_IDS[selected_name]}

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        run_legacy_webcam(cv2, mp, annotate_legacy_frame, selected_tip_ids)
    else:
        run_tasks_webcam(cv2, mp, annotate_tasks_frame, selected_tip_ids)