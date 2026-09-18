import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter

from modules.feature_extraction import extract_features
from modules.gesture_classifier import classify_gesture


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ============================================================
# COMPUTER VISION FUNCTIONS
# ============================================================

def grayscale_image(frame):
    """Stage 2: Convert image to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def gaussian_filter(gray):
    """Stage 3: Reduce image noise using Gaussian filtering."""
    return cv2.GaussianBlur(gray, (5, 5), 0)


def canny_edges(blurred):
    """Stage 4: Detect edges using Canny edge detection."""
    return cv2.Canny(blurred, 50, 150)


def segment_hand(frame):
    """
    Stage 5: Basic HSV-based skin segmentation.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    # Remove small noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
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
# STAGE 6: HAND LANDMARKS
# ============================================================

def draw_landmarks(frame, results):

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
# IMAGE PANEL FUNCTIONS
# ============================================================

def resize_panel(
    image,
    width=320,
    height=220
):

    return cv2.resize(
        image,
        (width, height)
    )


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
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return output


# ============================================================
# STAGE 7: FEATURE EXTRACTION PANEL
# ============================================================

def create_feature_panel(
    frame,
    features
):

    panel = frame.copy()

    height, width = panel.shape[:2]

    if features is not None:

        landmarks = features["landmarks"]

        # Draw feature points
        for i, landmark in enumerate(landmarks):

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                panel,
                (x, y),
                4,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                panel,
                str(i),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )

    panel = resize_panel(panel)

    panel = add_label(
        panel,
        "7. Feature Extraction"
    )

    return panel


# ============================================================
# STAGE 8: CLASSIFICATION PANEL
# ============================================================

def create_classification_panel(
    gesture,
    confidence,
    features
):

    width = 320
    height = 220

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
        0.52,
        (255, 255, 255),
        2
    )

    # Gesture
    cv2.putText(
        panel,
        f"Gesture: {gesture}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )

    # Confidence
    cv2.putText(
        panel,
        f"Confidence: {confidence * 100:.1f}%",
        (10, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
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
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1
        )

        cv2.putText(
            panel,
            f"Index: {'YES' if fingers['index'] else 'NO'}",
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1
        )

        cv2.putText(
            panel,
            f"Middle: {'YES' if fingers['middle'] else 'NO'}",
            (160, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1
        )

        cv2.putText(
            panel,
            f"Ring: {'YES' if fingers['ring'] else 'NO'}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1
        )

        cv2.putText(
            panel,
            f"Pinky: {'YES' if fingers['pinky'] else 'NO'}",
            (160, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1
        )

    return panel


# ============================================================
# STATISTICS PANEL
# ============================================================

def create_statistics_panel(
    counter,
    history
):

    width = 320
    height = 220

    panel = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    cv2.putText(
        panel,
        "GESTURE STATISTICS",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    gestures = [
        "Open Palm",
        "Fist",
        "Thumbs Up",
        "Victory",
        "OK"
    ]

    y = 60

    for gesture in gestures:

        count = counter.get(
            gesture,
            0
        )

        cv2.putText(
            panel,
            f"{gesture}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        y += 28

    cv2.putText(
        panel,
        f"Total events: {sum(counter.values())}",
        (10, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 255, 255),
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
    classification_panel,
    statistics_panel
):

    # First row
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

    # Second row
    panel5 = add_label(
        resize_panel(segmented),
        "5. Hand Segmentation"
    )

    panel6 = add_label(
        resize_panel(landmarks),
        "6. Hand Landmarks"
    )

    panel7 = feature_panel

    panel8 = classification_panel

    # Third row
    panel9 = statistics_panel

    # Three intentionally blank panels
    # No placeholder text is displayed.
    blank1 = np.zeros(
        (220, 320, 3),
        dtype=np.uint8
    )

    blank2 = np.zeros(
        (220, 320, 3),
        dtype=np.uint8
    )

    blank3 = np.zeros(
        (220, 320, 3),
        dtype=np.uint8
    )

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

    row3 = np.hstack(
        (
            panel9,
            blank1,
            blank2,
            blank3
        )
    )

    dashboard = np.vstack(
        (
            row1,
            row2,
            row3
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

    # ========================================================
    # STATISTICS
    # ========================================================

    gesture_counter = Counter()

    gesture_history = []

    previous_gesture = None

    # ========================================================
    # FPS
    # ========================================================

    previous_time = time.time()

    print("=" * 70)
    print("SMART HAND GESTURE RECOGNITION")
    print("CSE3010 COMPLETE COMPUTER VISION APPLICATION")
    print("=" * 70)

    print()
    print("8-STAGE COMPUTER VISION PIPELINE")
    print()
    print("1. Original Image")
    print("2. Grayscale")
    print("3. Gaussian Filtering")
    print("4. Canny Edge Detection")
    print("5. Hand Segmentation")
    print("6. Hand Landmarks")
    print("7. Feature Extraction")
    print("8. Gesture Classification")
    print()
    print("CONTROLS")
    print("Q = Quit")
    print("R = Reset Statistics")
    print("H = Show Gesture History")
    print("=" * 70)

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    with mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=1,

        min_detection_confidence=0.6,

        min_tracking_confidence=0.6

    ) as hands:

        while True:

            # =================================================
            # CAMERA
            # =================================================

            success, frame = camera.read()

            if not success:

                print(
                    "ERROR: Could not read webcam."
                )

                break

            # Mirror camera
            frame = cv2.flip(
                frame,
                1
            )

            # =================================================
            # STAGE 1: ORIGINAL
            # =================================================

            original = frame.copy()

            # =================================================
            # STAGE 2: GRAYSCALE
            # =================================================

            gray = grayscale_image(
                frame
            )

            # =================================================
            # STAGE 3: GAUSSIAN FILTER
            # =================================================

            blurred = gaussian_filter(
                gray
            )

            # =================================================
            # STAGE 4: CANNY
            # =================================================

            edges = canny_edges(
                blurred
            )

            # =================================================
            # STAGE 5: SEGMENTATION
            # =================================================

            mask, segmented = segment_hand(
                frame
            )

            # =================================================
            # MEDIAPIPE PROCESSING
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(
                rgb
            )

            gesture = "No Hand"

            confidence = 0.0

            features = None

            # =================================================
            # STAGE 6: LANDMARKS
            # STAGE 7: FEATURES
            # STAGE 8: CLASSIFICATION
            # =================================================

            if results.multi_hand_landmarks:

                hand_landmarks = (
                    results.multi_hand_landmarks[0]
                )

                # Stage 6
                landmarks = draw_landmarks(
                    frame,
                    results
                )

                # Stage 7
                features = extract_features(
                    hand_landmarks
                )

                # Stage 8
                gesture, confidence = (
                    classify_gesture(
                        features
                    )
                )

                # ---------------------------------------------
                # Gesture Event Tracking
                # ---------------------------------------------

                if (
                    gesture != previous_gesture
                    and gesture != "Unknown"
                    and gesture != "No Hand"
                ):

                    gesture_counter[
                        gesture
                    ] += 1

                    gesture_history.append(
                        {
                            "gesture": gesture,
                            "confidence": confidence,
                            "time": time.strftime(
                                "%H:%M:%S"
                            )
                        }
                    )

                    previous_gesture = gesture

            else:

                landmarks = frame.copy()

                previous_gesture = None

            # =================================================
            # FEATURE PANEL
            # =================================================

            feature_panel = create_feature_panel(
                frame,
                features
            )

            # =================================================
            # CLASSIFICATION PANEL
            # =================================================

            classification_panel = (
                create_classification_panel(
                    gesture,
                    confidence,
                    features
                )
            )

            # =================================================
            # STATISTICS PANEL
            # =================================================

            statistics_panel = (
                create_statistics_panel(
                    gesture_counter,
                    gesture_history
                )
            )

            # =================================================
            # COMPLETE DASHBOARD
            # =================================================

            dashboard = create_dashboard(

                original,

                gray,

                blurred,

                edges,

                segmented,

                landmarks,

                feature_panel,

                classification_panel,

                statistics_panel

            )

            # =================================================
            # FPS
            # =================================================

            current_time = time.time()

            elapsed = (
                current_time
                - previous_time
            )

            if elapsed > 0:

                fps = 1 / elapsed

            else:

                fps = 0

            previous_time = current_time

            # =================================================
            # HEADER
            # =================================================

            cv2.rectangle(
                dashboard,
                (0, 0),
                (1280, 45),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                dashboard,
                "SMART HAND GESTURE RECOGNITION",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            cv2.putText(
                dashboard,
                f"FPS: {fps:.1f}",
                (1160, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                "Smart Hand Gesture Recognition - CSE3010",
                dashboard
            )

            # =================================================
            # KEYBOARD
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            # Q = Quit
            if key == ord("q"):

                break

            # R = Reset
            elif key == ord("r"):

                gesture_counter.clear()

                gesture_history.clear()

                previous_gesture = None

                print(
                    "\nStatistics reset."
                )

            # H = History
            elif key == ord("h"):

                print()
                print("=" * 55)
                print("GESTURE HISTORY")
                print("=" * 55)

                if not gesture_history:

                    print(
                        "No gestures recorded."
                    )

                else:

                    for item in gesture_history:

                        print(
                            f"{item['time']} | "
                            f"{item['gesture']} | "
                            f"{item['confidence'] * 100:.1f}%"
                        )

                print("=" * 55)

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    print()
    print("=" * 55)
    print("FINAL GESTURE STATISTICS")
    print("=" * 55)

    if gesture_counter:

        for gesture, count in (
            gesture_counter.items()
        ):

            print(
                f"{gesture}: {count}"
            )

    else:

        print(
            "No gestures detected."
        )

    print("=" * 55)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()