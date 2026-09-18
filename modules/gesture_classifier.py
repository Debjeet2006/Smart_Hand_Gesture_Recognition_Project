def classify_gesture(features):

    fingers = features["fingers"]

    index = fingers["index"]
    middle = fingers["middle"]
    ring = fingers["ring"]
    pinky = fingers["pinky"]

    thumb_up = features["thumb_up"]
    thumb_index_distance = features["thumb_index_distance"]

    # -------------------------------------------------
    # 1. FIST
    # -------------------------------------------------
    # If all four main fingers are folded and the thumb
    # is NOT clearly pointing upward.
    if not index and not middle and not ring and not pinky:

        if thumb_up:
            return "Thumbs Up", 0.92

        return "Fist", 0.95

    # -------------------------------------------------
    # 2. OPEN PALM
    # -------------------------------------------------
    if index and middle and ring and pinky:
        return "Open Palm", 0.95

    # -------------------------------------------------
    # 3. VICTORY
    # -------------------------------------------------
    if index and middle and not ring and not pinky:
        return "Victory", 0.92

    # -------------------------------------------------
    # 4. OK
    # -------------------------------------------------
    # Thumb and index close together while the other
    # fingers are extended.
    if (
        thumb_index_distance < 0.08
        and middle
        and ring
        and pinky
    ):
        return "OK", 0.90

    # -------------------------------------------------
    # 5. THUMBS UP
    # -------------------------------------------------
    if (
        thumb_up
        and not index
        and not middle
        and not ring
        and not pinky
    ):
        return "Thumbs Up", 0.90

    # -------------------------------------------------
    # FALLBACK
    # -------------------------------------------------
    return "Unknown", 0.40