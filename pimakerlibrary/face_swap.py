"""
PiMaker Face Swap — swap two faces in real-time using your webcam!

How it works:
  1. MediaPipe detects 468 face landmarks on each face
  2. A convex hull outlines each face region
  3. Delaunay triangulation splits each face into tiny triangles
  4. Each triangle from Face A is warped onto Face B's position (and vice-versa)
  5. OpenCV's seamlessClone blends the result so skin tones match

Usage:
  pimaker.start_face_swap()

Controls:
  Q  — quit
"""

import os
import cv2
import time
import math
import numpy as np
import urllib.request


def _ensure_face_model():
    """Download the MediaPipe Face Landmarker model if needed."""
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print("[PiMaker] Downloading Face Landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            model_path,
        )
    return model_path


def _get_landmarks_as_points(face_landmarks, frame_w, frame_h):
    """Convert MediaPipe normalized landmarks → list of (x, y) pixel tuples."""
    points = []
    for lm in face_landmarks:
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        points.append((x, y))
    return points


def _get_face_hull_indices(points):
    """Return the convex-hull indices for a set of landmark points."""
    pts_array = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts_array, returnPoints=False)
    return hull.flatten().tolist()


def _rect_contains(rect, point):
    """Check if a point is inside a rectangle (x, y, w, h)."""
    x, y, w, h = rect
    return x <= point[0] < x + w and y <= point[1] < y + h


def _get_delaunay_triangles(rect, points, hull_indices):
    """
    Compute Delaunay triangulation over the convex hull points.
    Returns list of (i, j, k) — indices into `points`.
    """
    subdiv = cv2.Subdiv2D(rect)
    hull_points = [points[i] for i in hull_indices]

    # Map from (x,y) → original index in `points`
    point_to_idx = {}
    for idx in hull_indices:
        pt = points[idx]
        point_to_idx[(pt[0], pt[1])] = idx

    for pt in hull_points:
        # Clamp to rect to avoid Subdiv2D assertion errors
        clamped = (
            max(rect[0], min(rect[0] + rect[2] - 1, pt[0])),
            max(rect[1], min(rect[1] + rect[3] - 1, pt[1])),
        )
        subdiv.insert(clamped)
        # Also map the clamped version
        if clamped != pt:
            point_to_idx[clamped] = point_to_idx.get(pt, -1)

    triangle_list = subdiv.getTriangleList()
    triangles = []

    for t in triangle_list:
        p1 = (int(t[0]), int(t[1]))
        p2 = (int(t[2]), int(t[3]))
        p3 = (int(t[4]), int(t[5]))

        if not (_rect_contains(rect, p1) and _rect_contains(rect, p2) and _rect_contains(rect, p3)):
            continue

        # Find indices
        i1 = point_to_idx.get(p1)
        i2 = point_to_idx.get(p2)
        i3 = point_to_idx.get(p3)

        if i1 is not None and i2 is not None and i3 is not None:
            triangles.append((i1, i2, i3))

    return triangles


def _warp_triangle(src_img, dst_img, src_tri, dst_tri):
    """Warp a single triangle from src_img onto dst_img."""
    # Bounding rects
    sr = cv2.boundingRect(np.float32([src_tri]))
    dr = cv2.boundingRect(np.float32([dst_tri]))

    # Crop triangles
    src_cropped = []
    dst_cropped = []
    for i in range(3):
        src_cropped.append((src_tri[i][0] - sr[0], src_tri[i][1] - sr[1]))
        dst_cropped.append((dst_tri[i][0] - dr[0], dst_tri[i][1] - dr[1]))

    # Crop the source image region
    src_crop = src_img[sr[1]: sr[1] + sr[3], sr[0]: sr[0] + sr[2]]
    if src_crop.size == 0 or dr[2] == 0 or dr[3] == 0:
        return

    # Affine transform
    warp_mat = cv2.getAffineTransform(
        np.float32(src_cropped), np.float32(dst_cropped)
    )
    warped = cv2.warpAffine(
        src_crop,
        warp_mat,
        (dr[2], dr[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Create mask for the destination triangle
    mask = np.zeros((dr[3], dr[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_cropped), (255, 255, 255))

    # Blend into destination
    y1, y2 = dr[1], dr[1] + dr[3]
    x1, x2 = dr[0], dr[0] + dr[2]

    # Bounds check
    if y2 > dst_img.shape[0] or x2 > dst_img.shape[1] or y1 < 0 or x1 < 0:
        return

    dst_region = dst_img[y1:y2, x1:x2]
    if dst_region.shape != mask.shape:
        return

    mask_bool = mask > 0
    dst_region[mask_bool] = warped[mask_bool]


def _swap_single_face(src_img, dst_img, output_img, src_points, dst_points, hull_indices, triangles):
    """Warp the face region from src onto the dst position in output_img."""
    warped_face = np.zeros_like(dst_img)

    for (i, j, k) in triangles:
        if i >= len(src_points) or j >= len(src_points) or k >= len(src_points):
            continue
        if i >= len(dst_points) or j >= len(dst_points) or k >= len(dst_points):
            continue

        src_tri = [src_points[i], src_points[j], src_points[k]]
        dst_tri = [dst_points[i], dst_points[j], dst_points[k]]

        try:
            _warp_triangle(src_img, warped_face, src_tri, dst_tri)
        except Exception:
            continue

    # Create a mask from the destination convex hull
    dst_hull_pts = np.array([dst_points[i] for i in hull_indices], dtype=np.int32)
    mask = np.zeros(dst_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_hull_pts, 255)

    # Erode mask slightly to avoid harsh edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=1)

    # Find center for seamless clone
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return output_img

    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

    # Clamp center to valid range
    h, w = output_img.shape[:2]
    center = (max(1, min(w - 2, center[0])), max(1, min(h - 2, center[1])))

    try:
        output = cv2.seamlessClone(warped_face, output_img, mask, center, cv2.NORMAL_CLONE)
        return output
    except Exception:
        # Fallback: simple alpha blend if seamlessClone fails
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        output = (warped_face * mask_3ch + output_img * (1 - mask_3ch)).astype(np.uint8)
        return output


def start_face_swap():
    """
    🔄 Face Swap — Swap faces between two people in real-time!

    Stand in front of the camera with a friend. PiMaker will detect both
    faces and swap them live. The more centered and well-lit the faces are,
    the better the result.

    Controls:
        Q — quit

    Requires:
        - mediapipe
        - Two faces visible in the camera
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for start_face_swap(). "
            "Install with: pip install mediapipe"
        ) from exc

    model_path = _ensure_face_model()

    # Setup FaceLandmarker for 2 faces
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception(
            "🎥 Oops! Can't find your camera. "
            "Make sure it's plugged in and no other app is using it!"
        )

    print("\n🔄 PiMaker Face Swap Started!")
    print("   Get two faces in frame to see the magic.")
    print("   Press 'Q' to quit.\n")

    last_timestamp_ms = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[PiMaker] Failed to grab frame. Retrying...")
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        try:
            results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            cv2.imshow("PiMaker Face Swap", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        if results and results.face_landmarks and len(results.face_landmarks) >= 2:
            # Got two faces! Let's swap 'em
            face1_lm = results.face_landmarks[0]
            face2_lm = results.face_landmarks[1]

            pts1 = _get_landmarks_as_points(face1_lm, w, h)
            pts2 = _get_landmarks_as_points(face2_lm, w, h)

            # Compute convex hull on face 1 (indices are the same for both faces)
            hull_indices = _get_face_hull_indices(pts1)

            # Delaunay on face 1
            rect = (0, 0, w, h)
            triangles = _get_delaunay_triangles(rect, pts1, hull_indices)

            if triangles:
                output = frame.copy()

                # Swap: warp face1 → face2 position
                output = _swap_single_face(
                    frame, frame, output, pts1, pts2, hull_indices, triangles
                )

                # Swap: warp face2 → face1 position
                output = _swap_single_face(
                    frame, frame, output, pts2, pts1, hull_indices, triangles
                )

                # Draw a fun indicator
                cv2.putText(
                    output,
                    "FACE SWAP ACTIVE",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("PiMaker Face Swap", output)
            else:
                # Triangulation failed, show raw frame
                cv2.putText(
                    frame,
                    "Adjusting...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("PiMaker Face Swap", frame)

        else:
            # Not enough faces — show helpful message
            num_faces = len(results.face_landmarks) if results and results.face_landmarks else 0
            if num_faces == 1:
                msg = "Need 1 more face! Grab a friend!"
            else:
                msg = "Show 2 faces to start swapping!"

            # Draw a stylish info box
            box_w, box_h = 420, 50
            box_x = (w - box_w) // 2
            box_y = h - 70
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(
                frame,
                msg,
                (box_x + 15, box_y + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
            cv2.imshow("PiMaker Face Swap", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[PiMaker] Face Swap stopped.")
