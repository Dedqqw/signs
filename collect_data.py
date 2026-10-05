import cv2
import csv
import os
import time
import mediapipe as mp

# --- CONFIGURATION ---
CSV_FILE = "hand_landmarks_dataset.csv"
LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # ['A', 'B', 'C', ..., 'Z']
SAMPLES_PER_CLASS = 100

# --- INITIALIZE MEDIAPIPE SOLUTIONS ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hide TensorFlow / C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = ['label']
        for i in range(21):
            headers.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        writer.writerow(headers)

cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    
    print("=" * 60)
    print("HAND LANDMARK DATASET COLLECTOR")
    print("Controls:")
    print(" - Press 'S' to start recording samples for current letter.")
    print(" - Press 'N' to skip to next letter.")
    print(" - Press 'Q' to quit anytime.")
    print("=" * 60)

    for label in LABELS:
        recording = False
        sample_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to access camera feed.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            cv2.putText(frame, f"Target Letter: '{label}'", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Recorded: {sample_count}/{SAMPLES_PER_CLASS}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if not recording:
                cv2.putText(frame, "Press 'S' to Start Recording | 'N' to Skip", (20, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    if recording and sample_count < SAMPLES_PER_CLASS:
                        raw_coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        wrist_x, wrist_y, wrist_z = raw_coords[0]
                        
                        normalized_features = []
                        for x, y, z in raw_coords:
                            normalized_features.extend([x - wrist_x, y - wrist_y, z - wrist_z])

                        with open(CSV_FILE, mode='a', newline='') as f:
                            csv.writer(f).writerow([label] + normalized_features)

                        sample_count += 1
                        time.sleep(0.05)

                        if sample_count >= SAMPLES_PER_CLASS:
                            recording = False
                            print(f"--> Completed gathering samples for '{label}'!")

            if recording:
                cv2.circle(frame, (w - 40, 40), 15, (0, 0, 255), -1)

            cv2.imshow("Hand Landmark Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                recording = True
                print(f"Recording started for letter: '{label}'...")
            elif key == ord('n') or key == ord('N'):
                print(f"Skipped letter: '{label}'.")
                break
            elif key == ord('q') or key == ord('Q'):
                print("Exiting data collection...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            if sample_count >= SAMPLES_PER_CLASS:
                time.sleep(0.5)
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDataset collection finished successfully! Saved to: '{CSV_FILE}'")