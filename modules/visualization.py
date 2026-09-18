import cv2
import time


def draw_header(frame, gesture, confidence, fps, total_gestures):
    height, width = frame.shape[:2]

    # Top information panel
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 125),
        (25, 25, 25),
        -1
    )

    # Project title
    cv2.putText(
        frame,
        "SMART HAND GESTURE RECOGNITION",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # Gesture
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    # Confidence
    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (20, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (width - 150, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # Total detections
    cv2.putText(
        frame,
        f"Detections: {total_gestures}",
        (width - 200, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


def draw_footer(frame):
    height, width = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, height - 45),
        (width, height),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        "Q: Quit    R: Reset Statistics    H: Show History",
        (20, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


def draw_hand_status(frame, detected):
    height, width = frame.shape[:2]

    if detected:
        text = "HAND DETECTED"
        color = (0, 255, 0)
    else:
        text = "NO HAND DETECTED"
        color = (0, 0, 255)

    cv2.putText(
        frame,
        text,
        (width - 230, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


def draw_result(
    frame,
    gesture,
    confidence,
    fps=0,
    total_gestures=0,
    hand_detected=False
):
    draw_header(
        frame,
        gesture,
        confidence,
        fps,
        total_gestures
    )

    draw_hand_status(
        frame,
        hand_detected
    )

    draw_footer(frame)