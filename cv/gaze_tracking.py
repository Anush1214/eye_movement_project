import cv2
import time
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Landmark indices
# -------------------------------
LEFT_EYE  = [33, 159, 145, 133]
RIGHT_EYE = [362, 386, 374, 263]

LEFT_EYE_CORNERS  = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_IRIS  = [468, 469, 470, 471]
RIGHT_IRIS = [472, 473, 474, 475]

# -------------------------------
# Parameters 
# -------------------------------
# Tweaked these thresholds for stability
DEADZONE_X = 0.02
DEADZONE_Y = 0.02

H_THRESH = 0.04
UP_THRESH = 0.04
DOWN_THRESH = 0.04

EAR_DOWN_RATIO = 0.85
DOWN_Y_THRESH = 0.02

SMOOTHING = 0.85  # Stronger temporal smoothing 

# -------------------------------
# Utility functions
# -------------------------------
def compute_ear(face_landmarks, eye_indices, w, h):
    left = np.array([face_landmarks[eye_indices[0]].x * w, face_landmarks[eye_indices[0]].y * h])
    top = np.array([face_landmarks[eye_indices[1]].x * w, face_landmarks[eye_indices[1]].y * h])
    bottom = np.array([face_landmarks[eye_indices[2]].x * w, face_landmarks[eye_indices[2]].y * h])
    right = np.array([face_landmarks[eye_indices[3]].x * w, face_landmarks[eye_indices[3]].y * h])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    return vertical / horizontal if horizontal != 0 else 0.0

def iris_center(iris_points):
    return np.mean(np.array(iris_points), axis=0)

# -------------------------------
# Main
# -------------------------------
def main():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    smoothed_dx = 0.0
    smoothed_dy = 0.0
    smoothed_ear = 0.0
    down_frames = 0

    # Calibration variables
    calib_duration = 2.0
    is_calibrated = False
    calib_dx_list = []
    calib_dy_list = []
    calib_ear_list = []

    base_dx = 0.0
    base_dy = 0.0
    base_ear = 0.25 # Fallback reasonable average EAR

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        gaze_text = "NO FACE"
        elapsed = time.time() - start_time

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            left_ear = compute_ear(face_landmarks, LEFT_EYE, w, h)
            right_ear = compute_ear(face_landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # LEFT EYE corners
            eye_left_l_pt = face_landmarks[LEFT_EYE_CORNERS[0]]
            eye_left_r_pt = face_landmarks[LEFT_EYE_CORNERS[1]]
            l_lex, l_ley = eye_left_l_pt.x * w, eye_left_l_pt.y * h
            l_rex, l_rey = eye_left_r_pt.x * w, eye_left_r_pt.y * h

            # RIGHT EYE corners
            eye_right_l_pt = face_landmarks[RIGHT_EYE_CORNERS[0]]
            eye_right_r_pt = face_landmarks[RIGHT_EYE_CORNERS[1]]
            r_lex, r_ley = eye_right_l_pt.x * w, eye_right_l_pt.y * h
            r_rex, r_rey = eye_right_r_pt.x * w, eye_right_r_pt.y * h

            left_iris_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LEFT_IRIS]
            right_iris_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in RIGHT_IRIS]

            left_iris_c = iris_center(left_iris_pts)
            right_iris_c = iris_center(right_iris_pts)

            # ----- FIX: Normalize using eye corners ONLY (rigid structure) -----
            # Calculate center of the eye (midpoint between corners)
            left_cx = (l_lex + l_rex) / 2.0
            left_cy = (l_ley + l_rey) / 2.0
            left_w = abs(l_rex - l_lex) + 1e-6

            right_cx = (r_lex + r_rex) / 2.0
            right_cy = (r_ley + r_rey) / 2.0
            right_w = abs(r_rex - r_lex) + 1e-6

            gaze_x_left = (left_iris_c[0] - left_cx) / left_w
            gaze_y_left = (left_iris_c[1] - left_cy) / left_w

            gaze_x_right = (right_iris_c[0] - right_cx) / right_w
            gaze_y_right = (right_iris_c[1] - right_cy) / right_w

            # Average both eyes
            raw_dx = (gaze_x_left + gaze_x_right) / 2.0
            raw_dy = (gaze_y_left + gaze_y_right) / 2.0

            if not is_calibrated:
                if elapsed < calib_duration:
                    calib_dx_list.append(raw_dx)
                    calib_dy_list.append(raw_dy)
                    calib_ear_list.append(avg_ear)
                    gaze_text = "CALIBRATING... LOOK CENTER"
                else:
                    if len(calib_dx_list) > 0:
                        base_dx = np.mean(calib_dx_list)
                        base_dy = np.mean(calib_dy_list)
                        base_ear = np.mean(calib_ear_list)
                    is_calibrated = True
                    print(f"Calibrated: base_dx={base_dx:.3f}, base_dy={base_dy:.3f}, base_ear={base_ear:.3f}")
            else:
                dx_adj = raw_dx - base_dx
                dy_adj = raw_dy - base_dy

                # Temporal smoothing (moving average)
                smoothed_dx = SMOOTHING * smoothed_dx + (1 - SMOOTHING) * dx_adj
                smoothed_dy = SMOOTHING * smoothed_dy + (1 - SMOOTHING) * dy_adj
                smoothed_ear = SMOOTHING * smoothed_ear + (1 - SMOOTHING) * avg_ear

                # DOWN Time-based confirmation
                if smoothed_ear < base_ear * EAR_DOWN_RATIO and smoothed_dy > DOWN_Y_THRESH:
                    down_frames += 1
                else:
                    down_frames = max(0, down_frames - 1)

                # Direction logic
                if down_frames > 5:
                    gaze_text = "DOWN"
                elif abs(smoothed_dx) < DEADZONE_X and abs(smoothed_dy) < DEADZONE_Y:
                    gaze_text = "CENTER"
                else:
                    # Determine primary axis (add 1.2x weight to horizontal to favor looking left/right over slight up/down noise)
                    if abs(smoothed_dx) > abs(smoothed_dy) * 1.2:
                        if smoothed_dx < -H_THRESH:
                            gaze_text = "LEFT"
                        elif smoothed_dx > H_THRESH:
                            gaze_text = "RIGHT"
                        else:
                            gaze_text = "CENTER"
                    else:
                        if smoothed_dy < -UP_THRESH:
                            gaze_text = "UP"
                        elif smoothed_dy > DOWN_THRESH: # fallback structural DOWN
                            gaze_text = "DOWN"
                        else:
                            gaze_text = "CENTER"

            # Draw eye indicators
            for (x, y) in left_iris_pts + right_iris_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.circle(frame, tuple(left_iris_c.astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_iris_c.astype(int)), 3, (0, 0, 255), -1)

        color = (0, 0, 255) if gaze_text == "CALIBRATING... LOOK CENTER" else (255, 0, 0)
        cv2.putText(frame, gaze_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        cv2.imshow("Stable Gaze Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()

if __name__ == "__main__":
    main()
