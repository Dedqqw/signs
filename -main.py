import cv2
import numpy as np
import joblib
import mediapipe as mp
import os

# Hide TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load trained Random Forest model
model = joblib.load("asl_model.p")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Select active camera index (use 1 if external webcam, 0 if internal)
cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    print("=" * 60)
    print("LIVE SIGN LANGUAGE TRANSLATOR ACTIVE")
    print("Press 'Q' to quit.")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton joints
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract relative (x, y, z) features
                raw_coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                wrist_x, wrist_y, wrist_z = raw_coords[0]

                normalized_features = []
                for x, y, z in raw_coords:
                    normalized_features.extend([x - wrist_x, y - wrist_y, z - wrist_z])

                # Predict gesture class
                prediction = model.predict([normalized_features])[0]

                # Find hand bounding box coordinates to position projected text
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                # Draw bounding box and project detected letter above hand
                cv2.rectangle(frame, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), (0, 255, 0), 2)
                cv2.rectangle(frame, (min_x - 10, min_y - 50), (max_x + 10, min_y - 10), (0, 255, 0), -1)
                cv2.putText(frame, f"Letter: {prediction}", (min_x, min_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Sign Language Translator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()