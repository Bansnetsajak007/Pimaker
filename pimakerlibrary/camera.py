import cv2
import numpy as np

from .vision.fingertip_backends import run_legacy_webcam, run_tasks_webcam
from .vision.fingertip_draw import FINGERTIP_IDS, annotate_legacy_frame, annotate_tasks_frame

import urllib.request
import os
import math

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    Overlays a transparent PNG image onto a background image.
    If overlay_size is given, resizes the image_to_overlay first.
    """
    try:
        bg_h, bg_w, _ = background_img.shape
        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t, overlay_size)

        h, w, c = img_to_overlay_t.shape
        if c < 4:  # Needs an alpha channel
            return background_img

        x, y = int(x), int(y)
        # Bounding box of overlay
        y0, y1 = max(0, y), min(bg_h, y + h)
        x0, x1 = max(0, x), min(bg_w, x + w)
        # Bounding box of overlay image pixels
        y0_o, y1_o = max(0, -y), min(h, bg_h - y)
        x0_o, x1_o = max(0, -x), min(w, bg_w - x)

        if y1 <= y0 or x1 <= x0 or y1_o <= y0_o or x1_o <= x0_o:
            return background_img

        bg_crop = background_img[y0:y1, x0:x1]
        overlay_crop = img_to_overlay_t[y0_o:y1_o, x0_o:x1_o]

        alpha_channel = overlay_crop[:, :, 3] / 255.0
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

        bg_crop[:] = alpha_mask * overlay_crop[:, :, :3] + (1 - alpha_mask) * bg_crop
    except Exception as e:
        import traceback
        traceback.print_exc()
        pass
    return background_img

def open_camera(detect_emotion=False, cyborg_face=False, skeleton_mirror=False, app_filters=None, laser_eyes=False):
    
    cap = cv2.VideoCapture(0)  # 0 = default camera

    if not cap.isOpened():
        raise Exception("Could not open camera")

    print("Press 'q' to exit camera")

    face_landmarker = None
    pose_landmarker = None
    mp_drawing = None
    mp_drawing_styles = None
    mp_face_connections = None
    mp_pose_connections = None

    filter_img = None
    if app_filters:
        os.makedirs("filters", exist_ok=True)
        filter_path = f"filters/{app_filters}.png"
        
        if not os.path.exists(filter_path):
            print(f"Could not find {filter_path}. Please save a transparent PNG named {app_filters}.png in the 'filters' folder!")
            app_filters = None
        else:
            filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)
            if filter_img is None or filter_img.shape[2] < 4:
                print(f"Could not load {filter_path} - make sure it's a valid PNG with transparency!")
                app_filters = None

    if cyborg_face or skeleton_mirror or app_filters or laser_eyes:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            mp_drawing = mp.tasks.vision.drawing_utils
            mp_drawing_styles = mp.tasks.vision.drawing_styles
        except ImportError as exc:
            raise ImportError("mediapipe is required for cyborg_face, skeleton_mirror, app_filters, or laser_eyes.") from exc

    if cyborg_face or app_filters or laser_eyes:
        # Download Face Landmarker model if it doesn't exist
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading MediaPipe Face Landmarker model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                model_path
            )

        # Setup FaceLandmarker options
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
        mp_face_connections = mp.tasks.vision.FaceLandmarksConnections

    if skeleton_mirror:
        pose_model_path = "pose_landmarker_lite.task"
        if not os.path.exists(pose_model_path):
            print("Downloading MediaPipe Pose Landmarker model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                pose_model_path
            )

        pose_base_options = mp_python.BaseOptions(model_asset_path=pose_model_path)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        mp_pose_connections = mp.tasks.vision.PoseLandmarksConnections

    if detect_emotion:
        # Download models if they don't exist
        onnx_path = "emotion-ferplus-8.onnx"
        haar_path = "haarcascade_frontalface_default.xml"
        
        if not os.path.exists(onnx_path):
            print("Downloading ONNX emotion model...")
            urllib.request.urlretrieve("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx", onnx_path)
            
        if not os.path.exists(haar_path):
            print("Downloading Haar Cascade face detector...")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", haar_path)

        # Load models
        face_cascade = cv2.CascadeClassifier(haar_path)
        emotion_net = cv2.dnn.readNetFromONNX(onnx_path)
        emotion_labels = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

    last_timestamp_ms = -1

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame in camera. Retrying...")
            continue

        if (cyborg_face or app_filters or laser_eyes) and face_landmarker or (skeleton_mirror and pose_landmarker):
            import time
            try:
                # MediaPipe works with RGB frames
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                
                # CYBORG FACE & APP FILTERS & LASER EYES
                if (cyborg_face or app_filters or laser_eyes) and face_landmarker:
                    results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                    if results and results.face_landmarks:
                        for face_landmarks in results.face_landmarks:
                            frame_h, frame_w, _ = frame.shape
                            
                            # cyborg_face logic
                            if cyborg_face:
                                # Draw the face mesh tessellation (wireframe)
                                mp_drawing.draw_landmarks(
                                    image=frame,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_connections.FACE_LANDMARKS_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                                )
                                # Optionally draw face contours (eyes, lips)
                                mp_drawing.draw_landmarks(
                                    image=frame,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_connections.FACE_LANDMARKS_CONTOURS,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                                )
                                
                            # App filter logic (draw glasses)
                            if app_filters and filter_img is not None:
                                # We only want the first face landmarks to find eye coordinates
                                left_eye = face_landmarks.landmark[33] if hasattr(face_landmarks, "landmark") else face_landmarks[33]
                                right_eye = face_landmarks.landmark[263] if hasattr(face_landmarks, "landmark") else face_landmarks[263]
                                nose_bridge = face_landmarks.landmark[8] if hasattr(face_landmarks, "landmark") else face_landmarks[8]
                                upper_lip = face_landmarks.landmark[164] if hasattr(face_landmarks, "landmark") else face_landmarks[164]
                                nose_tip = face_landmarks.landmark[1] if hasattr(face_landmarks, "landmark") else face_landmarks[1]
                                
                                # Convert normalized coordinates to pixel coords
                                lx, ly = int(left_eye.x * frame_w), int(left_eye.y * frame_h)
                                rx, ry = int(right_eye.x * frame_w), int(right_eye.y * frame_h)
                                nbx, nby = int(nose_bridge.x * frame_w), int(nose_bridge.y * frame_h)
                                ulx, uly = int(upper_lip.x * frame_w), int(upper_lip.y * frame_h)
                                ntx, nty = int(nose_tip.x * frame_w), int(nose_tip.y * frame_h)
                                
                                # Scale the glasses dynamically based on face size
                                dx, dy = rx - lx, ry - ly
                                eye_distance = math.hypot(dx, dy)
                                
                                # Customize width & placement depending on what filter they requested
                                if app_filters == "mustache":
                                    filter_w = int(eye_distance * 1.5)
                                    anchor_x, anchor_y = ulx, uly
                                    y_offset_ratio = 1.0 # Centered on upper lip
                                elif app_filters == "sunglasses":
                                    filter_w = int(eye_distance * 2.2)
                                    anchor_x, anchor_y = nbx, nby
                                    y_offset_ratio = 1.7 # Lift it onto eyes
                                else:
                                    # Base generic filters off the center of the nose
                                    filter_w = int(eye_distance * 2.2)
                                    anchor_x, anchor_y = ntx, nty
                                    y_offset_ratio = 2.0 
                                
                                filter_ratio = filter_w / filter_img.shape[1]
                                filter_h = int(filter_img.shape[0] * filter_ratio)
                                
                                # Find angle
                                angle = math.degrees(math.atan2(dy, dx))
                                
                                # Center the glasses above the nose bridge
                                overlay_x = anchor_x - int(filter_w / 2)
                                overlay_y = anchor_y - int(filter_h / y_offset_ratio) 
                                
                                # Rotate the glasses image (very simply, using cv2 rotation)
                                center = (filter_img.shape[1]//2, filter_img.shape[0]//2)
                                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                                rotated_filter = cv2.warpAffine(filter_img, M, (filter_img.shape[1], filter_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                                
                                # Apply transparent overlay
                                overlay_transparent(frame, rotated_filter, overlay_x, overlay_y, overlay_size=(filter_w, filter_h))

                            # Laser eyes logic
                            if laser_eyes:
                                import math
                                # Use irises if refined landmarks are available (usually 468, 473 for left/right iris)
                                # Fallback to inner eye corners if using standard 468 mesh
                                try:
                                    left_eye = face_landmarks.landmark[468] if hasattr(face_landmarks, "landmark") else face_landmarks[468]
                                    right_eye = face_landmarks.landmark[473] if hasattr(face_landmarks, "landmark") else face_landmarks[473]
                                except IndexError:
                                    left_eye = face_landmarks.landmark[33] if hasattr(face_landmarks, "landmark") else face_landmarks[33]
                                    right_eye = face_landmarks.landmark[263] if hasattr(face_landmarks, "landmark") else face_landmarks[263]

                                lx, ly = int(left_eye.x * frame_w), int(left_eye.y * frame_h)
                                rx, ry = int(right_eye.x * frame_w), int(right_eye.y * frame_h)
                                
                                # Find angle between the eyes to shoot perpendicular lasers
                                dx, dy = rx - lx, ry - ly
                                face_angle = math.atan2(dy, dx)
                                laser_angle = face_angle - (math.pi / 2) # Perpendicular to the eye line
                                
                                # End points way off screen
                                end_lx = int(lx + 2000 * math.cos(laser_angle))
                                end_ly = int(ly + 2000 * math.sin(laser_angle))
                                end_rx = int(rx + 2000 * math.cos(laser_angle))
                                end_ry = int(ry + 2000 * math.sin(laser_angle))
                                
                                # Draw glowing lasers using cv2.addWeighted for transparency
                                overlay = frame.copy()
                                # Red glow
                                cv2.line(overlay, (lx, ly), (end_lx, end_ly), (0, 0, 255), 18)
                                cv2.line(overlay, (rx, ry), (end_rx, end_ry), (0, 0, 255), 18)
                                # White hot core
                                cv2.line(overlay, (lx, ly), (end_lx, end_ly), (255, 255, 255), 6)
                                cv2.line(overlay, (rx, ry), (end_rx, end_ry), (255, 255, 255), 6)
                                
                                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                # SKELETON MIRROR
                if skeleton_mirror and pose_landmarker:
                    pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                    if pose_results and pose_results.pose_landmarks:
                        for pose_landmarks in pose_results.pose_landmarks:
                            # Draw the full body pose tracking skeleton
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=pose_landmarks,
                                connections=mp_pose_connections.POSE_LANDMARKS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                            )
            except Exception as e:
                import traceback
                traceback.print_exc()
                pass  # Ignore frames where detection fails or glitches

        if detect_emotion:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (64, 64))
                    
                    # Prepare blob for ONNX model (shape: 1, 1, 64, 64)
                    blob = cv2.dnn.blobFromImage(face_resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
                    emotion_net.setInput(blob)
                    preds = emotion_net.forward()
                    
                    # Get highest scoring emotion
                    emotion_idx = preds[0].argmax()
                    dominant_emotion = emotion_labels[emotion_idx]
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                pass  # Ignore frames where detection fails or glitches

        cv2.imshow("PiMaker Camera", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_fingertip(finger=None, show_gesture=False):
    """Detect fingertips from webcam only.

    Args:
        finger: Optional finger selector from 1 to 5.
            1=thumb, 2=index, 3=middle, 4=ring, 5=pinky.
            If omitted, all fingertips are detected.
        show_gesture: If True, identifies and displays simple gestures (Thumbs up, Peace sign, etc.)
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
        run_legacy_webcam(cv2, mp, annotate_legacy_frame, selected_tip_ids, show_gesture=show_gesture)
    else:
        run_tasks_webcam(cv2, mp, annotate_tasks_frame, selected_tip_ids, show_gesture=show_gesture)