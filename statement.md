# Project Statement

## Project Title
**Smart Hand Gesture Recognition**

## 1. Problem Statement
Human-computer interaction often requires a physical input device such as a keyboard, mouse or touch interface. A hand gesture provides a natural visual form of interaction, but a computer must first process the camera image and identify the hand configuration.

The problem addressed by this project is to design and implement a real-time Computer Vision system that receives hand images from a webcam, processes the video frames through standard image-processing stages, detects hand landmarks, extracts hand features and recognizes a predefined set of hand gestures.

## 2. Scope of the Project
The current project is limited to webcam-based recognition of five predefined gestures:

1. Fist
2. Open Palm
3. Thumbs Up
4. Victory
5. OK

The project focuses on demonstrating the Computer Vision processing pipeline rather than building a general-purpose gesture-recognition system.

## 3. Target Users
- Students learning Computer Vision
- Faculty evaluating a Computer Vision implementation
- Users demonstrating basic gesture-based interaction
- Developers studying webcam-based hand landmark processing

## 4. High-Level Features
- Real-time webcam input
- Grayscale preprocessing
- Gaussian filtering
- Canny edge detection
- Hand segmentation
- Hand landmark detection
- Feature extraction
- Gesture classification
- Visual processing dashboard
- Gesture statistics and history
- FPS display
- Basic camera/frame error handling

## 5. Input
Live video frames captured from the system webcam.

## 6. Output
The system displays the processed image stages and the detected gesture in real time.

## 7. Functional Modules
### F1 — Image Processing
Processes each camera frame using grayscale conversion, Gaussian filtering and edge detection.

### F2 — Segmentation
Produces a hand-region representation using image segmentation and morphological processing.

### F3 — Landmark and Feature Processing
Detects hand landmarks and extracts geometric/finger information.

### F4 — Gesture Classification
Uses the existing rule-based logic to map extracted hand features to the five supported gestures.

### F5 — Visualization and Statistics
Displays the processing pipeline, classification result, FPS, counts and history.

## 8. Non-Functional Requirements
### NFR1 — Performance
The application should process frames continuously with practical real-time responsiveness.

### NFR2 — Usability
The dashboard should make the processing stages understandable and controls should remain simple.

### NFR3 — Reliability
The system should handle unavailable cameras, failed frame capture and absence of a detected hand.

### NFR4 — Maintainability
The implementation should keep major processing responsibilities separated into modules.

### NFR5 — Resource Efficiency
The runtime system should avoid unnecessary persistent data storage and use a compact landmark representation.

### NFR6 — Error Handling
Camera and frame-processing failures should produce controlled behavior instead of an uncontrolled crash.

## 9. High-Level Workflow

```text
Start
  |
  v
Open Webcam
  |
  v
Capture Frame
  |
  v
Preprocess Frame
  |
  v
Gaussian Filter
  |
  v
Canny Edge Detection
  |
  v
Hand Segmentation
  |
  v
Detect Hand Landmarks
  |
  v
Extract Features
  |
  v
Classify Gesture
  |
  v
Display Result and Statistics
  |
  +----> Next Frame
```

## 10. Out of Scope
The current project does not introduce:
- a new machine-learning training pipeline,
- a database,
- user authentication,
- cloud deployment,
- additional gesture classes beyond the existing five.

These are intentionally outside the current project scope.
