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
LEFT_EYE_CORNERS  = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_TOP_BOTTOM = [386, 374]

LEFT_IRIS  = [468, 469, 470, 471]
RIGHT_IRIS = [472, 473, 474, 475]

# -------------------------------
# Parameters (IMPORTANT)
# -------------------------------
H_THRESH = 0.18
UP_THRESH = 0.20
DOWN_THRESH = 0.10
SMOOTHING = 0.8   # higher = smoother

# -------------------------------
# Utility functions
# -------------------------------
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

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            eye_left = face_landmarks[LEFT_EYE_CORNERS[0]]
            eye_right = face_landmarks[LEFT_EYE_CORNERS[1]]

            lex = int(eye_left.x * w)
            rex = int(eye_right.x * w)

            eye_top = face_landmarks[LEFT_EYE_TOP_BOTTOM[0]]
            eye_bottom = face_landmarks[LEFT_EYE_TOP_BOTTOM[1]]

            top_y = int(eye_top.y * h)
            bottom_y = int(eye_bottom.y * h)

            left_iris_pts = [(int(face_landmarks[i].x * w),
                              int(face_landmarks[i].y * h)) for i in LEFT_IRIS]

            right_iris_pts = [(int(face_landmarks[i].x * w),
                               int(face_landmarks[i].y * h)) for i in RIGHT_IRIS]

            left_iris_center = iris_center(left_iris_pts)

            gaze_x = (left_iris_center[0] - lex) / (rex - lex + 1e-6)
            gaze_y = (left_iris_center[1] - top_y) / (bottom_y - top_y + 1e-6)

            dx = gaze_x - 0.5
            dy = gaze_y - 0.5

            # ----- Temporal smoothing -----
            smoothed_dx = SMOOTHING * smoothed_dx + (1 - SMOOTHING) * dx
            smoothed_dy = SMOOTHING * smoothed_dy + (1 - SMOOTHING) * dy

            # ----- Direction logic -----
            if abs(smoothed_dx) > abs(smoothed_dy):
                if smoothed_dx < -H_THRESH:
                    gaze_text = "LEFT"
                elif smoothed_dx > H_THRESH:
                    gaze_text = "RIGHT"
                else:
                    gaze_text = "CENTER"
            else:
                if smoothed_dy < -UP_THRESH:
                    gaze_text = "UP"
                elif smoothed_dy > DOWN_THRESH:
                    gaze_text = "DOWN"
                else:
                    gaze_text = "CENTER"

            # ----- Draw eye indicators -----
            for (x, y) in left_iris_pts + right_iris_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.circle(
                frame,
                tuple(left_iris_center.astype(int)),
                3, (0, 0, 255), -1
            )

        cv2.putText(frame, gaze_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        cv2.imshow("Stable Gaze Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()

if __name__ == "__main__":
    main()
