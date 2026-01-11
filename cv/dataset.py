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

LEFT_EYE_CORNERS  = [33, 133]
LEFT_IRIS  = [468, 469, 470, 471]

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

    print("Press L / R / C / B to label data, Q to quit")

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

            left_eye_pts = [(int(face_landmarks[i].x * w),
                             int(face_landmarks[i].y * h)) for i in LEFT_EYE]

            right_eye_pts = [(int(face_landmarks[i].x * w),
                              int(face_landmarks[i].y * h)) for i in RIGHT_EYE]

            left_ear = compute_ear(left_eye_pts)
            right_ear = compute_ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            # Gaze X
            eye_left = face_landmarks[LEFT_EYE_CORNERS[0]]
            eye_right = face_landmarks[LEFT_EYE_CORNERS[1]]

            lex = int(eye_left.x * w)
            rex = int(eye_right.x * w)

            iris_pts = [(int(face_landmarks[i].x * w),
                         int(face_landmarks[i].y * h)) for i in LEFT_IRIS]

            iris_c = iris_center(iris_pts)
            gaze_x = (iris_c[0] - lex) / (rex - lex + 1e-6)

            # Labeling
            label = None
            if key == ord('l'):
                label = 0
            elif key == ord('r'):
                label = 1
            elif key == ord('c'):
                label = 2
            elif key == ord('b'):
                label = 3
            elif key == ord('q'):
                break

            if label is not None:
                data.append([left_ear, right_ear, avg_ear, gaze_x, label])
                print("Saved sample:", label)

            cv2.putText(frame, "Press L/R/C/B to label", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Day 5: Dataset Collection", frame)

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()

    # Save CSV
    with open("eye_movement_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EAR_L", "EAR_R", "EAR_AVG", "GAZE_X", "LABEL"])
        writer.writerows(data)

    print("Dataset saved as eye_movement_dataset.csv")

if __name__ == "__main__":
    main()
