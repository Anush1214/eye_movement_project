import cv2
import time
import numpy as np
import csv
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Landmark indices
# -------------------------------
LEFT_EYE  = [33, 159, 145, 133]
RIGHT_EYE = [362, 386, 374, 263]

LEFT_EYE_CORNERS = [33, 133]
LEFT_EYE_TOP_BOTTOM = [159, 145]

LEFT_IRIS = [468, 469, 470, 471]

# -------------------------------
# Parameters
# -------------------------------
H_THRESH = 0.18
UP_THRESH = 0.20
DOWN_THRESH = 0.10
SMOOTHING = 0.8

# -------------------------------
# Utility functions
# -------------------------------
def compute_ear(eye_points):
    left = np.array(eye_points[0])
    top = np.array(eye_points[1])
    bottom = np.array(eye_points[2])
    right = np.array(eye_points[3])

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

    data = []

    smoothed_dx = 0.0
    smoothed_dy = 0.0

    print("Press L / R / U / D / C to label | Q to quit")

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

        key = cv2.waitKey(1) & 0xFF

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            # ----- EAR -----
            left_eye_pts = [(int(face_landmarks[i].x * w),
                             int(face_landmarks[i].y * h)) for i in LEFT_EYE]

            right_eye_pts = [(int(face_landmarks[i].x * w),
                              int(face_landmarks[i].y * h)) for i in RIGHT_EYE]

            left_ear = compute_ear(left_eye_pts)
            right_ear = compute_ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            # ----- Iris & gaze ratios -----
            eye_left = face_landmarks[LEFT_EYE_CORNERS[0]]
            eye_right = face_landmarks[LEFT_EYE_CORNERS[1]]

            lex = int(eye_left.x * w)
            rex = int(eye_right.x * w)

            iris_pts = [(int(face_landmarks[i].x * w),
                         int(face_landmarks[i].y * h)) for i in LEFT_IRIS]

            iris_c = iris_center(iris_pts)

            gaze_x = (iris_c[0] - lex) / (rex - lex + 1e-6)

            eye_top = face_landmarks[LEFT_EYE_TOP_BOTTOM[0]]
            eye_bottom = face_landmarks[LEFT_EYE_TOP_BOTTOM[1]]

            top_y = int(eye_top.y * h)
            bottom_y = int(eye_bottom.y * h)

            gaze_y = (iris_c[1] - top_y) / (bottom_y - top_y + 1e-6)

            dx = gaze_x - 0.5
            dy = gaze_y - 0.5

            # ----- Temporal smoothing -----
            smoothed_dx = SMOOTHING * smoothed_dx + (1 - SMOOTHING) * dx
            smoothed_dy = SMOOTHING * smoothed_dy + (1 - SMOOTHING) * dy

            # ----- Dominant direction (STABLE) -----
            detected = "CENTER"
            if abs(smoothed_dx) > abs(smoothed_dy):
                if smoothed_dx < -H_THRESH:
                    detected = "LEFT"
                elif smoothed_dx > H_THRESH:
                    detected = "RIGHT"
            else:
                if smoothed_dy < -UP_THRESH:
                    detected = "UP"
                elif smoothed_dy > DOWN_THRESH:
                    detected = "DOWN"

            cv2.putText(frame, f"Detected: {detected}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ----- Manual labeling -----
            label = None
            if key == ord('l'):
                label = 0
            elif key == ord('r'):
                label = 1
            elif key == ord('u'):
                label = 2
            elif key == ord('d'):
                label = 3
            elif key == ord('c'):
                label = 4
            elif key == ord('q'):
                break

            if label is not None:
                data.append([
                    left_ear,
                    right_ear,
                    avg_ear,
                    smoothed_dx,
                    smoothed_dy,
                    label
                ])
                print("Saved sample:", label)

        cv2.imshow("Dataset Collection (Stable Gaze)", frame)

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()

    # ----- Save CSV -----
    with open("eye_movement_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EAR_L",
            "EAR_R",
            "EAR_AVG",
            "DX",
            "DY",
            "LABEL"
        ])
        writer.writerows(data)

    print("Dataset saved as eye_movement_dataset.csv")

if __name__ == "__main__":
    main()
