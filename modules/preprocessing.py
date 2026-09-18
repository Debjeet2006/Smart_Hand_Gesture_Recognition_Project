import cv2

def preprocess_frame(frame):
    """Basic CV preprocessing used before hand landmark detection."""
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb
