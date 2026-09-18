# Smart Hand Gesture Recognition System

## 1. Project Overview

Smart Hand Gesture Recognition System is a Computer Vision project that detects a hand from a webcam, extracts hand landmark features, and recognizes common gestures in real time.

### Supported gestures
1. Open Palm
2. Fist
3. Thumbs Up
4. Victory
5. OK

## 2. Problem Statement

Traditional computer interfaces depend heavily on keyboards, mice and touch input. This project explores a vision-based interaction method in which hand gestures are recognized from camera frames and converted into understandable commands.

## 3. Objectives

- Capture live video from a webcam.
- Detect a human hand using Computer Vision.
- Extract geometric hand landmark features.
- Classify common hand gestures.
- Display the predicted gesture and confidence in real time.
- Demonstrate image/video processing and feature extraction concepts from Computer Vision.

## 4. Main Functional Modules

### Module 1: Video Capture & Preprocessing
Captures webcam frames and performs basic frame preprocessing.

### Module 2: Hand Landmark & Feature Extraction
Detects hand landmarks and calculates geometric features such as finger states and distances.

### Module 3: Gesture Classification
Uses extracted features and rule-based classification to recognize the selected gestures.

### Module 4: Visualization
Displays hand landmarks, predicted gesture and confidence on the video frame.

### Module 5: Testing
Contains validation tests for the feature/classification pipeline.

## 5. System Workflow

Webcam
-> Frame Capture
-> Preprocessing
-> Hand Detection
-> Landmark Extraction
-> Feature Extraction
-> Gesture Classification
-> Result Visualization

## 6. Technologies

- Python
- OpenCV
- MediaPipe
- NumPy

## 7. Computer Vision Concepts Demonstrated

- Image/video acquisition
- Image preprocessing
- Coordinate/landmark representation
- Feature extraction
- Classification
- Real-time video processing
- Visualization

These concepts are aligned with the CSE3010 Computer Vision syllabus, particularly feature extraction, classification and video/motion analysis.

## 8. Installation

Use Python 3.10 or another version supported by the installed MediaPipe release.

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 9. Run

```bash
python app.py
```

Allow camera access when prompted.

Press `Q` to close the application.

## 10. Testing

Run:

```bash
python -m unittest discover tests
```

## 11. Limitations

- Recognition works best when one hand is clearly visible.
- Lighting and camera angle can affect landmark detection.
- The current classifier recognizes a limited set of predefined gestures.
- Confidence values are heuristic scores from the rule-based classifier, not calibrated probabilities.

## 12. Future Enhancements

- Add more gestures.
- Add voice feedback.
- Add gesture-controlled media/player commands.
- Add a graphical web dashboard.
- Train an ML classifier using landmark features.
- Add two-hand gesture recognition.
- Add gesture history and analytics.
