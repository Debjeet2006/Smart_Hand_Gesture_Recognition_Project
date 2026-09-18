# Smart Hand Gesture Recognition

## 1. Project Title
**Smart Hand Gesture Recognition**

A real-time Computer Vision project for recognizing common hand gestures from a webcam feed.

## 2. Overview
This project demonstrates a complete image/video-processing pipeline using Python, OpenCV and MediaPipe. A webcam frame is processed through multiple Computer Vision stages before the hand gesture is classified.

The current implementation recognizes five gestures:

- Fist
- Open Palm
- Thumbs Up
- Victory
- OK

The application also displays the intermediate Computer Vision stages in a dashboard so that the processing performed on each frame can be observed.

## 3. Objectives
1. Capture hand images from a live webcam.
2. Apply fundamental image preprocessing techniques.
3. Detect edges using Canny edge detection.
4. Segment the hand region.
5. Detect hand landmarks.
6. Extract meaningful hand/finger features.
7. Classify the detected gesture.
8. Demonstrate the complete process through a real-time visual dashboard.

## 4. Main Functional Modules
### Module 1 — Image Preprocessing
Converts the camera frame to grayscale and applies Gaussian filtering to reduce noise.

### Module 2 — Edge Detection
Uses the Canny method to identify significant intensity boundaries in the image.

### Module 3 — Hand Segmentation
Uses a color-based segmentation process and morphological operations to obtain a hand-region representation.

### Module 4 — Hand Landmark Detection
Uses MediaPipe Hands to locate the hand's 21 landmarks.

### Module 5 — Feature Extraction and Classification
Extracts finger/hand geometric information from the landmarks and uses the current rule-based classifier to recognize one of the supported gestures.

### Module 6 — Visualization and Statistics
Displays the eight processing stages, the predicted gesture, confidence information, FPS, gesture counts and gesture history.

## 5. Computer Vision Pipeline

```text
Webcam Frame
     |
     v
Original Image
     |
     v
Grayscale Conversion
     |
     v
Gaussian Filtering
     |
     v
Canny Edge Detection
     |
     v
Hand Segmentation
     |
     v
MediaPipe Hand Landmarks
     |
     v
Feature Extraction
     |
     v
Gesture Classification
     |
     v
Visual Dashboard / Result
```

## 6. Current Eight Stages
1. Original Image
2. Grayscale
3. Gaussian Filtering
4. Canny Edge Detection
5. Hand Segmentation
6. Hand Landmarks
7. Feature Extraction
8. Gesture Classification

## 7. Technologies Used
- Python 3.12
- OpenCV
- MediaPipe
- NumPy

## 8. Project Structure

```text
Smart_Hand_Gesture_Recognition_Project/
│
├── docs/
│   └── design_diagrams.md
│
├── modules/
│   ├── __init__.py
│   ├── cv_analysis.py
│   ├── feature_extraction.py
│   ├── gesture_classifier.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── screenshots/
│   ├── Screenshot 2026-09-18 222738.png
│   ├── Screenshot 2026-09-18 222751.png
│   ├── Screenshot 2026-09-18 222809.png
│   ├── Screenshot 2026-09-18 222825.png
│   ├── Screenshot 2026-09-18 222839.png
│   └── Screenshot 2026-09-18 223003.png
│
├── tests/
│   ├── __init__.py
│   └── test_classifier.py
│
├── .gitignore
├── app.py
├── docs_project_structure.txt
├── README.md
├── requirements.txt
└── statement.md
```

> The existing application can be run without introducing a new ML model. The KNN-related files shown above are retained only if they are already part of the repository; the current working application uses the existing rule-based gesture classifier.

## 9. Installation

Create/activate a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Recommended current versions:

```text
opencv-python==4.11.0.86
mediapipe==0.10.21
numpy==1.26.4
```

## 10. Running the Project

From the project root:

```bash
python app.py
```

Allow access to the computer webcam when requested.

## 11. Controls

- **Q** — Quit the application
- **R** — Reset gesture statistics
- **H** — Display gesture history

## 12. Input and Output

**Input:** Live video frames from the webcam.

**Intermediate outputs:** Grayscale image, filtered image, edge image, segmentation result, landmark visualization and extracted feature information.

**Final output:** Recognized gesture with the dashboard and statistics.

## 13. Testing Approach

The current application is validated using live webcam testing.

Test cases include:

| Test | Input | Expected Result |
|---|---|---|
| T1 | No hand in camera | `No Hand` |
| T2 | Closed fist | `Fist` |
| T3 | All fingers open | `Open Palm` |
| T4 | Thumb raised | `Thumbs Up` |
| T5 | Index and middle fingers raised | `Victory` |
| T6 | Thumb and index forming OK shape | `OK` |
| T7 | Webcam unavailable | Error message / safe termination |
| T8 | Reset command | Gesture statistics reset |

## 14. Error Handling
The application checks whether the webcam can be opened and whether a frame can be read. If no hand is detected, the application does not attempt to classify a gesture and displays the appropriate no-hand state.

## 15. Non-Functional Requirements
- **Performance:** Process webcam frames continuously for real-time interaction.
- **Usability:** Provide a visual dashboard and simple keyboard controls.
- **Reliability:** Handle missing camera frames and missing hand detections safely.
- **Maintainability:** Separate Computer Vision processing, feature extraction, classification and visualization into modules.
- **Resource Efficiency:** Use landmark-based hand representation instead of storing or processing large image datasets during runtime.
- **Error Handling:** Detect camera-access and frame-capture failures.

## 16. Design Documents
See [`docs/design_diagrams.md`](docs/design_diagrams.md) for the architecture, workflow, use-case, sequence and component diagrams.

## 17. Scope and Limitations
The project is designed as an academic Computer Vision demonstration. It recognizes a limited set of five predefined gestures and is intended for controlled webcam-based use.

The current classifier is rule-based; no new machine-learning training component is required for the present submission.

## 18. Future Enhancements
Possible future work includes improving robustness to different lighting/background conditions, adding more gestures, and evaluating a trained classifier. These are outside the current implementation scope.

## 19. Academic Alignment
The project applies Computer Vision concepts including image preprocessing, Gaussian filtering, Canny edge detection, segmentation, feature extraction, landmark-based representation, classification and real-time video processing.

## 20. 📸 Screenshots

### 1. Smart Hand Gesture Recognition Dashboard

![Smart Hand Gesture Recognition Dashboard](screenshots/Screenshot%202026-09-18%20222738.png)

---

### 2. Gesture Recognition Result

![Gesture Recognition Result](screenshots/Screenshot%202026-09-18%20222751.png)

---

### 3. Computer Vision Processing Pipeline

![Computer Vision Processing Pipeline](screenshots/Screenshot%202026-09-18%20222809.png)

---

### 4. Hand Landmark Detection

![Hand Landmark Detection](screenshots/Screenshot%202026-09-18%20222825.png)

---

### 5. Feature Extraction and Classification

![Feature Extraction and Classification](screenshots/Screenshot%202026-09-18%20222839.png)

---

### 6. Gesture Statistics

![Gesture Statistics](screenshots/Screenshot%202026-09-18%20223003.png)