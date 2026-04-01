FINGERTIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_tip_box(frame, x, y, cv2, color=(0, 255, 255), box_size=28):
    half = box_size // 2
    h, w, _ = frame.shape
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w - 1, x + half)
    y2 = min(h - 1, y + half)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def annotate_legacy_frame(frame, results, mp_hands, mp_draw, cv2, selected_tip_ids=None):
    if not results.multi_hand_landmarks:
        return frame

    tip_ids = selected_tip_ids or FINGERTIP_IDS

    h, w, _ = frame.shape
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for name, tip_id in tip_ids.items():
            tip = hand_landmarks.landmark[tip_id]
            x, y = int(tip.x * w), int(tip.y * h)
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            draw_tip_box(frame, x, y, cv2)
            cv2.putText(
                frame,
                name,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 255, 50),
                1,
            )
    return frame


def annotate_tasks_frame(frame, landmarks_per_hand, cv2, selected_tip_ids=None):
    if not landmarks_per_hand:
        return frame

    tip_ids = selected_tip_ids or FINGERTIP_IDS

    h, w, _ = frame.shape
    for landmarks in landmarks_per_hand:
        for a, b in HAND_CONNECTIONS:
            ax, ay = int(landmarks[a].x * w), int(landmarks[a].y * h)
            bx, by = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 200, 0), 2)

        for point in landmarks:
            px, py = int(point.x * w), int(point.y * h)
            cv2.circle(frame, (px, py), 3, (255, 100, 0), -1)

        for name, tip_id in tip_ids.items():
            tip = landmarks[tip_id]
            x, y = int(tip.x * w), int(tip.y * h)
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            draw_tip_box(frame, x, y, cv2)
            cv2.putText(
                frame,
                name,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 255, 50),
                1,
            )
    return frame
