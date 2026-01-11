import cv2
import time
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Eye landmark indices
# -------------------------------
LEFT_EYE  = [33, 159, 145, 133]
RIGHT_EYE = [362, 386, 374, 263]

# -------------------------------
# EAR computation
# -------------------------------
def compute_ear(eye_points):
    # eye_points: list of (x, y)
    left = np.array(eye_points[0])
    top = np.array(eye_points[1])
    bottom = np.array(eye_points[2])
    right = np.array(eye_points[3])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal == 0:
        return 0.0

    ear = vertical / horizontal
    return ear

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

    start_time = time.time()

    blink_counter = 0
    blink_start_time = None

    EAR_THRESHOLD = 0.18       # empirical
    LONG_BLINK_TIME = 0.7      # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            h, w, _ = frame.shape

            left_eye_pts = []
            right_eye_pts = []

            for idx in LEFT_EYE:
                lm = face_landmarks[idx]
                left_eye_pts.append((int(lm.x * w), int(lm.y * h)))

            for idx in RIGHT_EYE:
                lm = face_landmarks[idx]
                right_eye_pts.append((int(lm.x * w), int(lm.y * h)))

            left_ear = compute_ear(left_eye_pts)
            right_ear = compute_ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            # -------------------------------
            # Blink logic
            # -------------------------------
            if avg_ear < EAR_THRESHOLD:
                if blink_start_time is None:
                    blink_start_time = time.time()
            else:
                if blink_start_time is not None:
                    blink_duration = time.time() - blink_start_time
                    blink_start_time = None

                    if blink_duration >= LONG_BLINK_TIME:
                        blink_counter += 1
                        print("LONG BLINK detected")
                    else:
                        blink_counter += 1
                        print("BLINK detected")

            # Draw eyes
            for (x, y) in left_eye_pts + right_eye_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Display EAR
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Blinks: {blink_counter}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        cv2.imshow("Day 3: Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()

if __name__ == "__main__":
    main()
