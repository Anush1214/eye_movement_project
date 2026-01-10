import cv2
import time
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def main():
    # -------- MediaPipe FaceLandmarker setup --------
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # -------- OpenCV camera --------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Remove legacy solution references
    # mp_drawing = mp.solutions.drawing_utils
    # mp_face_mesh = mp.solutions.face_mesh

    start_time = time.time()
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Timestamp in milliseconds (required for VIDEO mode)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = face_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # -------- Draw landmarks --------
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                for lm in face_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("Face Mesh (MediaPipe Tasks)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()


if __name__ == "__main__":
    main()
