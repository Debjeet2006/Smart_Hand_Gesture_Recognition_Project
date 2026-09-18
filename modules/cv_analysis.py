import cv2
import numpy as np
import mediapipe as mp
import time

from modules.feature_extraction import extract_features
from modules.gesture_classifier import classify_gesture


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ============================================================
# BASIC COMPUTER VISION FUNCTIONS
# ============================================================

def grayscale_image(frame):
    """Convert BGR image into grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def gaussian_filter(gray):
    """Apply Gaussian smoothing."""
    return cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


def canny_edges(blurred):
    """Detect edges using Canny."""
    return cv2.Canny(
        blurred,
        50,
        150
    )


# ============================================================
# HAND SEGMENTATION
# ============================================================

def segment_hand(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # Basic skin-color HSV range
    lower_skin = np.array(
        [0, 25, 50],
        dtype=np.uint8
    )

    upper_skin = np.array(
        [25, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower_skin,
        upper_skin
    )

    # Morphological operations
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    segmented = cv2.bitwise_and(
        frame,
        frame,
        mask=mask
    )

    return mask, segmented


# ============================================================
# HAND LANDMARK DETECTION
# ============================================================

def detect_landmarks(frame, results):

    output = frame.copy()

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                output,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    return output


# ============================================================
# LABEL IMAGE
# ============================================================

def add_label(image, text):

    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 38),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        output,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return output


# ============================================================
# RESIZE
# ============================================================

def resize_panel(
    image,
    width=320,
    height=230
):

    return cv2.resize(
        image,
        (width, height)
    )


# ============================================================
# FEATURE VISUALIZATION
# ============================================================

def create_feature_panel(
    original,
    features
):

    panel = original.copy()

    height, width = panel.shape[:2]

    # Draw the hand landmarks if available
    if features is not None:

        landmarks = features["landmarks"]

        # Draw landmark points
        for i, landmark in enumerate(landmarks):

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                panel,
                (x, y),
                4,
                (0, 255, 255),
                -1
            )

            # Landmark number
            cv2.putText(
                panel,
                str(i),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )

    panel = add_label(
        panel,
        "7. Feature Extraction"
    )

    return panel


# ============================================================
# CLASSIFICATION INFORMATION PANEL
# ============================================================

def create_classification_panel(
    gesture,
    confidence,
    features
):

    width = 320
    height = 230

    panel = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # Title
    cv2.putText(
        panel,
        "8. Gesture Classification",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # Gesture
    cv2.putText(
        panel,
        f"Gesture: {gesture}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    # Confidence
    cv2.putText(
        panel,
        f"Confidence: {confidence * 100:.1f}%",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    if features is not None:

        fingers = features["fingers"]

        extended = sum(
            fingers.values()
        )

        cv2.putText(
            panel,
            f"Extended fingers: {extended}",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            panel,
            f"Index:  {'YES' if fingers['index'] else 'NO'}",
            (10, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        cv2.putText(
            panel,
            f"Middle: {'YES' if fingers['middle'] else 'NO'}",
            (160, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        cv2.putText(
            panel,
            f"Ring:   {'YES' if fingers['ring'] else 'NO'}",
            (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        cv2.putText(
            panel,
            f"Pinky:  {'YES' if fingers['pinky'] else 'NO'}",
            (160, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

    return panel


# ============================================================
# CREATE COMPLETE DASHBOARD
# ============================================================

def create_dashboard(
    original,
    gray,
    blurred,
    edges,
    segmented,
    landmarks,
    feature_panel,
    classification_panel
):

    panel1 = add_label(
        resize_panel(original),
        "1. Original Image"
    )

    panel2 = add_label(
        resize_panel(gray),
        "2. Grayscale"
    )

    panel3 = add_label(
        resize_panel(blurred),
        "3. Gaussian Filtering"
    )

    panel4 = add_label(
        resize_panel(edges),
        "4. Canny Edge Detection"
    )

    panel5 = add_label(
        resize_panel(segmented),
        "5. Hand Segmentation"
    )

    panel6 = add_label(
        resize_panel(landmarks),
        "6. Hand Landmarks"
    )

    panel7 = resize_panel(
        feature_panel
    )

    panel8 = classification_panel

    row1 = np.hstack(
        (
            panel1,
            panel2,
            panel3,
            panel4
        )
    )

    row2 = np.hstack(
        (
            panel5,
            panel6,
            panel7,
            panel8
        )
    )

    dashboard = np.vstack(
        (
            row1,
            row2
        )
    )

    return dashboard


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "ERROR: Could not access webcam."
        )

        return

    print("=" * 70)
    print("CSE3010 COMPUTER VISION - COMPLETE ANALYSIS")
    print("=" * 70)

    print()
    print("Pipeline:")
    print()
    print("1. Original Image")
    print("2. Grayscale")
    print("3. Gaussian Filtering")
    print("4. Canny Edge Detection")
    print("5. Hand Segmentation")
    print("6. Hand Landmark Detection")
    print("7. Feature Extraction")
    print("8. Gesture Classification")
    print()
    print("Press Q to quit.")
    print("=" * 70)

    previous_time = time.time()

    with mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=1,

        min_detection_confidence=0.6,

        min_tracking_confidence=0.6

    ) as hands:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "ERROR: Could not read webcam frame."
                )

                break

            # Mirror camera
            frame = cv2.flip(
                frame,
                1
            )

            # ====================================================
            # STAGE 1
            # ====================================================

            original = frame.copy()

            # ====================================================
            # STAGE 2
            # ====================================================

            gray = grayscale_image(
                frame
            )

            # ====================================================
            # STAGE 3
            # ====================================================

            blurred = gaussian_filter(
                gray
            )

            # ====================================================
            # STAGE 4
            # ====================================================

            edges = canny_edges(
                blurred
            )

            # ====================================================
            # STAGE 5
            # ====================================================

            mask, segmented = segment_hand(
                frame
            )

            # ====================================================
            # MEDIAPIPE
            # ====================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(
                rgb
            )

            # Default values
            gesture = "No Hand"
            confidence = 0.0
            features = None

            # ====================================================
            # STAGE 6 + 7 + 8
            # ====================================================

            if results.multi_hand_landmarks:

                hand_landmarks = (
                    results.multi_hand_landmarks[0]
                )

                # ------------------------------
                # Stage 6: Landmarks
                # ------------------------------

                landmarks = detect_landmarks(
                    frame,
                    results
                )

                # ------------------------------
                # Stage 7: Feature Extraction
                # ------------------------------

                features = extract_features(
                    hand_landmarks
                )

                # ------------------------------
                # Stage 8: Classification
                # ------------------------------

                gesture, confidence = classify_gesture(
                    features
                )

            else:

                landmarks = frame.copy()

            # ====================================================
            # FEATURE PANEL
            # ====================================================

            feature_panel = create_feature_panel(
                frame,
                features
            )

            # ====================================================
            # CLASSIFICATION PANEL
            # ====================================================

            classification_panel = (
                create_classification_panel(
                    gesture,
                    confidence,
                    features
                )
            )

            # ====================================================
            # DASHBOARD
            # ====================================================

            dashboard = create_dashboard(

                original,

                gray,

                blurred,

                edges,

                segmented,

                landmarks,

                feature_panel,

                classification_panel

            )

            # ====================================================
            # FPS
            # ====================================================

            current_time = time.time()

            fps = 1 / (
                current_time - previous_time
            )

            previous_time = current_time

            cv2.putText(
                dashboard,
                f"FPS: {fps:.1f}",
                (1180, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            # ====================================================
            # DISPLAY
            # ====================================================

            cv2.imshow(
                "CSE3010 - Complete Computer Vision Analysis",
                dashboard
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()