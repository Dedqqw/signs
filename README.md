# Sign Language to Sentence Translator

An AI-powered live Sign Language translator that captures hand gestures via webcam, normalizes landmarks using **MediaPipe**, and translates them into letters and full sentences in real-time using a trained **Random Forest** model.

---

## Prerequisites & Installation

This project supports **Python 3.10+** (fully tested on modern Python versions up to **3.14**). Choose the installation guide based on your Operating System below.

### Windows Setup

1. Open **Command Prompt (cmd)** or **PowerShell** as Administrator.
2. Clone the repository and navigate to the folder.
3. Install the required frameworks directly via `pip`:
   ```cmd
   pip install opencv-python mediapipe scikit-learn joblib pandas numpy
   ```

---

### inux Setup (Ubuntu / Debian / Kali)

On Linux systems, you must install core multimedia and OpenGL packages before handling Python dependencies to prevent `ImportError: libGL.so` crashes.

1. Open your terminal and install system dependencies:
   ```bash
   sudo apt update && sudo apt install -y libgl1 libglib2.0-0t64 python3-venv
   ```
2. Install Python packages:
   * **Option A: Inside a Virtual Environment (Recommended)**
     ```bash
     python3 -m venv myenv
     source myenv/bin/activate
     pip install opencv-python mediapipe scikit-learn joblib pandas numpy
     ```
   * **Option B: Globally (Kali Linux / Obeying system protection flags)**
     ```bash
     python3 -m pip install mediapipe scikit-learn joblib pandas numpy opencv-python --break-system-packages
     ```

---

## Required Assets Download

Because the modern MediaPipe API uses decoupled architecture, you **must download** the core pipeline assets manually before launching the translator.

Run the following command in your project directory to download the official Google AI bundle:

```bash
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### Model Directory Structure
Make sure your root folder contains these precise files before running the project:
* `hand_landmarker.task` ──► High-level pipeline bundle (downloaded via wget above)
* `hand_detector.tflite` ──► Core Palm Detection model (MediaPipe internal)
* `hand_landmarks_detector.tflite` ──► 21-Skeletal Landmark extractor
* `asl_model.p` ──► Your trained Sign Language classification model

---

##  How to Run

1. **Collect your own dataset** (Optional):
   ```bash
   python collect_data.py
   ```
2. **Train the Random Forest model**:
   ```bash
   python train_model.py
   ```
3. **Run the Live Sentence Translator**:
   ```bash
   python main.py
   ```

### Live Controls
* `Hold sign for 2.5 seconds` ──► Lock and append current letter to the active word.
* `[SPACEBAR]` ──► Add a space and complete the word.
* `[BACKSPACE]` ──► Delete the last character / word.
* `[C]` ──► Clear the entire sentence overlay.
* `[Q]` ──► Safely release webcam and Quit application.
