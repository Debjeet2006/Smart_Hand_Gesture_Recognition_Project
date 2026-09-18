# Design Diagrams

The diagrams below document the **existing** Smart Hand Gesture Recognition implementation. They do not introduce additional application features.

## 1. System Architecture Diagram

```mermaid
flowchart LR
    A[Webcam] --> B[OpenCV Frame Capture]
    B --> C[Image Preprocessing]
    C --> C1[Grayscale]
    C1 --> C2[Gaussian Filtering]
    C2 --> D[Canny Edge Detection]
    B --> E[Hand Segmentation]
    B --> F[MediaPipe Hand Detection]
    F --> G[21 Hand Landmarks]
    G --> H[Feature Extraction]
    H --> I[Rule-Based Gesture Classifier]
    I --> J[Gesture Result]
    C2 --> K[Visualization Dashboard]
    D --> K
    E --> K
    G --> K
    H --> K
    J --> K
    K --> L[Statistics / History / FPS]
```

## 2. Process Flow Diagram

```mermaid
flowchart TD
    A([Start]) --> B[Initialize Webcam]
    B --> C{Camera Available?}
    C -- No --> D[Display Error]
    D --> Z([End])
    C -- Yes --> E[Capture Frame]
    E --> F[Flip Frame]
    F --> G[Grayscale]
    G --> H[Gaussian Filtering]
    H --> I[Canny Edge Detection]
    F --> J[Hand Segmentation]
    F --> K[MediaPipe Processing]
    K --> L{Hand Detected?}
    L -- No --> M[Display No Hand]
    L -- Yes --> N[Draw 21 Landmarks]
    N --> O[Extract Features]
    O --> P[Classify Gesture]
    P --> Q[Update Statistics and History]
    M --> R[Display Dashboard]
    Q --> R
    R --> S{Quit?}
    S -- No --> E
    S -- Yes --> Z([End])
```

## 3. Use Case Diagram

```mermaid
flowchart LR
    U[User] --> A[Start Application]
    U --> B[Show Hand Gesture]
    U --> C[Observe CV Pipeline]
    U --> D[View Gesture Result]
    U --> E[Reset Statistics]
    U --> F[View Gesture History]
    U --> G[Quit Application]

    B --> H[Webcam]
    H --> I[Process Video Frame]
    I --> J[Detect Landmarks]
    J --> K[Extract Features]
    K --> L[Classify Gesture]
    L --> D
```

## 4. Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant App as Application
    participant CV as OpenCV
    participant MP as MediaPipe
    participant Classifier as Gesture Classifier
    participant UI as Dashboard

    User->>App: Start application
    App->>CV: Open webcam
    loop Each video frame
        CV-->>App: Camera frame
        App->>CV: Grayscale/filter/edges/segmentation
        App->>MP: Process RGB frame
        MP-->>App: Hand landmarks
        App->>Classifier: Extract features and classify
        Classifier-->>App: Gesture and confidence
        App->>UI: Update pipeline and result
        UI-->>User: Display dashboard
    end
    User->>App: Quit
    App->>CV: Release webcam
```

## 5. Component/Class-Level Diagram

```mermaid
classDiagram
    class App {
        +main()
        +capture webcam frames
        +create dashboard
        +track statistics
        +track history
    }

    class Preprocessing {
        +grayscale_image()
        +gaussian_filter()
        +canny_edges()
        +segment_hand()
    }

    class CVAnalysis {
        +run analysis stages
    }

    class FeatureExtraction {
        +distance()
        +is_finger_extended()
        +extract_features()
    }

    class GestureClassifier {
        +classify_gesture()
    }

    class Visualization {
        +draw_landmarks()
        +create_feature_panel()
        +create_classification_panel()
        +create_statistics_panel()
        +create_dashboard()
    }

    App --> Preprocessing
    App --> CVAnalysis
    App --> FeatureExtraction
    App --> GestureClassifier
    App --> Visualization
```

## 6. Storage Design

No persistent database is used by the current application. Gesture statistics and history are maintained during the running session. Therefore, an ER diagram is **not applicable** to the present implementation.

## 7. Design Rationale

- OpenCV is used for image and video processing.
- Gaussian filtering is placed before Canny edge detection to reduce image noise.
- MediaPipe is used to obtain a structured representation of the hand through 21 landmarks.
- Feature extraction converts landmark information into geometric/finger information used by the current classifier.
- A rule-based classifier is retained because it is the existing working implementation and no additional ML component is being introduced.
- The dashboard exposes intermediate stages so the Computer Vision workflow can be demonstrated and evaluated.
