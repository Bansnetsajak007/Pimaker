import cv2
import numpy as np
from ultralytics import YOLO

# --- Color palette for 80 COCO classes ---
# Generate distinct, vibrant colors using HSV spacing
np.random.seed(42)
COLORS = [
    tuple(int(c) for c in color)
    for color in np.random.randint(60, 255, size=(80, 3)).tolist()
]


def detect_objects(confidence_threshold=0.5, model_size="n"):
    """
    Real-time object detection using YOLOv8 (Ultralytics).

    Detects 80 COCO object classes including people, vehicles, animals,
    furniture, electronics, food, sports equipment, and more.
    Draws color-coded bounding boxes with class labels and confidence scores.

    Args:
        confidence_threshold (float): Minimum confidence (0.0-1.0) to display
                                      a detection. Defaults to 0.5 (50%).
        model_size (str): YOLOv8 model variant. Options:
                          "n" (nano, fastest), "s" (small), "m" (medium),
                          "l" (large), "x" (extra-large, most accurate).
                          Defaults to "n" for real-time performance.

    Controls:
        Q / ESC  -- quit
        +/-      -- raise / lower confidence threshold on the fly
    """
    valid_sizes = ("n", "s", "m", "l", "x")
    if model_size not in valid_sizes:
        raise ValueError(
            f"[PiMaker] Invalid model_size '{model_size}'. "
            f"Choose from {valid_sizes}"
        )

    model_name = f"yolov8{model_size}.pt"
    print(f"[PiMaker] Loading YOLOv8-{model_size} model ({model_name})...")
    print("[PiMaker] (First run will download the model automatically)")
    model = YOLO(model_name)
    print("[PiMaker] Model loaded. Starting object detection...")
    print("Controls: [Q/ESC] quit  |  [+] raise threshold  |  [-] lower threshold")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("[PiMaker] Could not open camera.")

    threshold = confidence_threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[PiMaker] Failed to grab frame. Retrying...")
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Run YOLOv8 inference (verbose=False suppresses per-frame logs)
        results = model(frame, conf=threshold, verbose=False)

        detected_count = 0

        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Extract box coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                color = COLORS[cls_id % len(COLORS)]

                detected_count += 1

                # Clamp to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw filled label background
                text = f"{label}: {conf:.0%}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_h - baseline - 4),
                    (x1 + text_w, y1),
                    color, -1
                )
                # Draw label text (white for visibility)
                cv2.putText(
                    frame, text,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
                )

        # --- HUD overlay ---
        overlay_text = [
            f"YOLOv8-{model_size} | Objects: {detected_count}",
            f"Confidence threshold: {threshold:.0%}",
            "Q/ESC: quit   +/-: adjust threshold",
        ]
        for idx, line in enumerate(overlay_text):
            y_pos = 30 + idx * 28
            cv2.putText(frame, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(frame, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        cv2.imshow("PiMaker - Object Detection (YOLOv8)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # Q or ESC
            break
        elif key == ord('+') or key == ord('='):
            threshold = min(0.95, round(threshold + 0.05, 2))
            print(f"[PiMaker] Threshold raised to {threshold:.0%}")
        elif key == ord('-') or key == ord('_'):
            threshold = max(0.05, round(threshold - 0.05, 2))
            print(f"[PiMaker] Threshold lowered to {threshold:.0%}")

    cap.release()
    cv2.destroyAllWindows()
    print("[PiMaker] Object detection stopped.")
