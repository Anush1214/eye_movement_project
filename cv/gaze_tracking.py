import cv2
import time
import numpy as np
import os
import joblib
import pyttsx3

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
DEADZONE_Y = 0.03  # Stronger vertical dead-zone to protect CENTER

H_THRESH = 0.04
UP_THRESH = 0.06   # Harder to trigger UP
DOWN_THRESH = 0.08 # Hardest raw threshold for DOWN

EAR_DOWN_RATIO = 0.90  # Eyelid needs to close by ~10% (was 15%) to be sensitive to DOWN
DOWN_Y_THRESH = 0.01   # Minimal downward iris required when eyelid drops

SMOOTHING = 0.85
DOWN_FRAMES_REQ = 7    # DOWN requires 7 frames of sustained time-integration  # Stronger temporal smoothing 

STABLE_FRAMES_REQ = 5  # Frames required to reliably trigger cursor movement
BLINK_THRESH_RATIO = 0.85 # Frame EAR must fall below 85% of baseline
LONG_BLINK_TIME = 0.8  # Seconds for a long blink

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

def draw_ui(frame, selection, gaze_text, blink_status, selected_option):
    h, w, _ = frame.shape
    grid_texts = [["YES", "NO"], ["HELP", "WATER"]]
    cell_w = w // 2
    cell_h = 100
    
    # Draw Grid at bottom
    for r in range(2):
        for c in range(2):
            x1, y1 = c * cell_w, h - 200 + r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            
            color = (200, 200, 200)
            if selection == [r, c]:
                color = (0, 255, 0) # Highlight selected
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                text_color = (0, 0, 0)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_color = color
                
            # Center text in cell
            text = grid_texts[r][c]
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cx = x1 + (cell_w - text_size[0]) // 2
            cy = y1 + (cell_h + text_size[1]) // 2
            cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Draw Status Overlay at top
    cv2.putText(frame, f"Gaze: {gaze_text}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink State: {blink_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if selected_option:
        cv2.putText(frame, f"Last Selected: {selected_option}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# -------------------------------
# Main
# -------------------------------
def main():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "gaze_blink_rf_model.pkl")
    SCALER_PATH = os.path.join(os.path.dirname(__file__), "feature_scaler.pkl")

    rf_model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # Initialize TTS engine once
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    best_voice_id = None
    for voice in voices:
        name = voice.name.lower()
        if 'female' in name or 'zira' in name or 'samantha' in name:
            best_voice_id = voice.id
            break
    if best_voice_id:
        engine.setProperty('voice', best_voice_id)

    last_spoken = None
    last_spoken_time = 0.0
    cooldown_seconds = 1.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    smoothed_dx = 0.0
    smoothed_dy = 0.0
    smoothed_ear = 0.0
    down_frames = 0

    # UI & Assistive State Variables
    ui_selection = [0, 0] # [row, col]
    selected_option = ""
    grid_texts = [["YES", "NO"], ["HELP", "WATER"]]
    
    stable_gaze_frames = 0
    last_gaze_pred = "CENTER"
    confirmed_gaze = "CENTER"
    
    blink_start_time = 0
    is_blinking = False
    blink_status = "None"
    long_blink_triggered = False
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

                # 1. BLINK DETECTION
                current_blink_active = avg_ear < base_ear * BLINK_THRESH_RATIO
                if current_blink_active:
                    if not is_blinking: # Just started blinking
                        is_blinking = True
                        blink_start_time = time.time()
                        long_blink_triggered = False
                        blink_status = "Blinking..."
                    else: # Currently blinking
                        if time.time() - blink_start_time > LONG_BLINK_TIME:
                            if not long_blink_triggered:
                                blink_status = "Long Blink"
                                long_blink_triggered = True
                else: # Eyes are open
                    if is_blinking:
                        if not long_blink_triggered:
                            # Short blink finished! Trigger selection.
                            selected_option = grid_texts[ui_selection[0]][ui_selection[1]]
                            print(f">>> SELECTED: {selected_option}")
                            
                            # Speech execution
                            current_time = time.time()
                            if selected_option and selected_option != last_spoken:
                                if (current_time - last_spoken_time) > cooldown_seconds:
                                    engine.say(selected_option)
                                    engine.runAndWait()
                                    last_spoken = selected_option
                                    last_spoken_time = current_time
                        
                        is_blinking = False
                        blink_status = "None"

                # 2. MACHINE LEARNING PREDICTION
                features = np.array([[left_ear, right_ear, avg_ear, smoothed_dx, smoothed_dy]])
                scaled_features = scaler.transform(features)
                pred = rf_model.predict(scaled_features)[0]

                label_map = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN", 4: "CENTER"}
                raw_gaze = label_map.get(pred, "CENTER")

                # 3. STABLE GAZE OUTPUT & CONTROL LOGIC
                if raw_gaze == last_gaze_pred:
                    stable_gaze_frames += 1
                else:
                    stable_gaze_frames = 0
                
                last_gaze_pred = raw_gaze

                if stable_gaze_frames == STABLE_FRAMES_REQ:
                    confirmed_gaze = raw_gaze
                    # Move grid selection on confirmation
                    if confirmed_gaze == "LEFT":
                        ui_selection[1] = max(0, ui_selection[1] - 1)
                    elif confirmed_gaze == "RIGHT":
                        ui_selection[1] = min(1, ui_selection[1] + 1)
                    elif confirmed_gaze == "UP":
                        ui_selection[0] = max(0, ui_selection[0] - 1)
                    elif confirmed_gaze == "DOWN":
                        ui_selection[0] = min(1, ui_selection[0] + 1)
                
                gaze_text = confirmed_gaze

            # Draw eye indicators
            for (x, y) in left_iris_pts + right_iris_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.circle(frame, tuple(left_iris_c.astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_iris_c.astype(int)), 3, (0, 0, 255), -1)

        if not is_calibrated:
            cv2.putText(frame, "CALIBRATING... LOOK CENTER", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            draw_ui(frame, ui_selection, gaze_text, blink_status, selected_option)

        cv2.imshow("Stable Gaze Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()

if __name__ == "__main__":
    main()
