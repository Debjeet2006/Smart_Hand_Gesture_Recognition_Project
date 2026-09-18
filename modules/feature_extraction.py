import math


def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def is_finger_extended(landmarks, tip, pip):
    """
    For index, middle, ring and pinky:
    fingertip should be above the PIP joint
    for the normal upright hand position.
    """
    return landmarks[tip].y < landmarks[pip].y


def extract_features(hand_landmarks):
    lm = hand_landmarks.landmark

    # Four main fingers
    index = is_finger_extended(lm, 8, 6)
    middle = is_finger_extended(lm, 12, 10)
    ring = is_finger_extended(lm, 16, 14)
    pinky = is_finger_extended(lm, 20, 18)

    # Thumb measurements
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]
    wrist = lm[0]

    thumb_wrist_distance = distance(thumb_tip, wrist)
    thumb_mcp_distance = distance(thumb_tip, thumb_mcp)

    # For our upright gestures, a thumbs-up has its tip
    # clearly above the wrist.
    thumb_up = (
        thumb_tip.y < wrist.y - 0.08
        and thumb_wrist_distance > 0.16
    )

    return {
        "landmarks": lm,

        "fingers": {
            "index": index,
            "middle": middle,
            "ring": ring,
            "pinky": pinky,
        },

        "thumb_up": thumb_up,

        "thumb_index_distance": distance(lm[4], lm[8]),
        "wrist_index_distance": distance(lm[0], lm[8]),
        "thumb_wrist_distance": thumb_wrist_distance,
        "thumb_mcp_distance": thumb_mcp_distance,
    }