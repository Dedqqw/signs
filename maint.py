import os
import sys

# Mute all background logs and C++ output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load trained model
model = joblib.load("asl_model.p")

# Initialize MediaPipe Tasks (New API)
# Ensure you downloaded 'hand_landmarker.task' to the same folder!
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"Error: Please download '{model_path}' and place it in the project folder.")
    sys.exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for manual processing per frame
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)  # Change to 0 if using default webcam

# Sentence building tracking variables
sentence = []
current_word = ""

last_prediction = ""
letter_start_time = None
HOLD_THRESHOLD = 2.5  # Time in seconds to hold sign
cooldown = False

print("\n" + "=" * 55)
print(" LIVE SIGN-TO-SENTENCE TRANSLATOR (NEW API)")
print("=" * 55)
print(" Hold sign for 2.5s to add letter.")
print(" [SPACE]     : Add space to sentence")
print(" [BACKSPACE] : Delete last character")
print(" [C]         : Clear all text")
print(" [Q]         : Quit")
print("=" * 55 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Wrap frame into MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    detected_letter = None

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw skeleton manually since mp.solutions.drawing_utils is deprecated
            x_coords = [int(lm.x * w) for lm in hand_landmarks]
            y_coords = [int(lm.y * h) for lm in hand_landmarks]
            
            # Simple drawing of connections and dots
            for x, y in zip(x_coords, y_coords):
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Normalize landmark coordinates (relative to wrist)
            wrist_x, wrist_y, wrist_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z

            normalized_features = []
            for lm in hand_landmarks:
                normalized_features.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

            detected_letter = str(model.predict([normalized_features])[0])

            # Position overlay text around hand bounding box
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            cv2.rectangle(frame, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), (0, 255, 0), 2)
            cv2.putText(frame, f"Sign: {detected_letter}", (min_x, min_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Hold sign logic
    current_time = time.time()
    if detected_letter:
        if detected_letter == last_prediction:
            if letter_start_time is None:
                letter_start_time = current_time
            
            elapsed = current_time - letter_start_time
            progress = min(1.0, elapsed / HOLD_THRESHOLD)
            
            # Visual loading bar under top banner
            bar_width = int(w * progress)
            cv2.rectangle(frame, (0, 100), (bar_width, 105), (0, 255, 0), -1)

            if elapsed >= HOLD_THRESHOLD and not cooldown:
                current_word += detected_letter
                cooldown = True
        else:
            last_prediction = detected_letter
            letter_start_time = current_time
            cooldown = False
    else:
        last_prediction = None
        letter_start_time = None
        cooldown = False

    # Top Banner UI overlay
    cv2.rectangle(frame, (0, 0), (w, 100), (30, 30, 30), -1)
    full_text = " ".join(sentence) + (" " + current_word if current_word else "")
    
    cv2.putText(frame, f"Word: {current_word}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Sentence: {full_text}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Sign-to-Sentence Translator", frame)

    # Keyboard listener
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # SPACEBAR
        if current_word:
            sentence.append(current_word)
            current_word = ""
    elif key == 8:   # BACKSPACE
        if current_word:
            current_word = current_word[:-1]
        elif sentence:
            current_word = sentence.pop()
    elif key == ord('c') or key == ord('C'):
        sentence = []
        current_word = ""

cap.release()
cv2.destroyAllWindows()
detector.close()
